# spb_pixel_art — 像素风鸟图生成 CLI 设计文档

- 日期：2026-08-29
- 状态：已实现（v1）
- 相关模块：`spb_pixel_art.py`、`tools/comfy_client.py`、`workflows_template/pixel_art_api.json`

## 1. 背景与动机 / Motivation

想把部分鸟种生成像素风图片（YOLO 截图 + ComfyUI 工作流）。SuperPicky 的
YOLO 框已逐框存于 report.db `bird_detections` 表与 sidecar JSON（原图像素
`[x, y, w, h]`），但调用方 BirdIndex 网站自己持有鸟图文件和 bbox，因此
本工具设计为**输入驱动**：不读项目数据库，输入完全来自 CLI 参数或任务
清单 JSON，由 BirdIndex 以子进程方式调用（与 `spb_rename_species.py`
相同的委托模式）。

## 2. 接口 / Interface

### 2.1 调用形态

```bash
# 单张
python spb_pixel_art.py --image <路径> [--bbox X Y W H] [--label 白鹭] \
    [--workflow api.json] [--out DIR] [--json]

# 批量（BirdIndex 主用）
python spb_pixel_art.py --batch tasks.json [--workflow api.json] [--out DIR] [--json]
```

### 2.2 tasks.json 清单格式

```json
{"out_dir": "可选，输出根目录",
 "items": [{"id": "调用方追踪ID", "image": "图片路径",
            "bbox": [x, y, w, h], "label": "白鹭", "name": "可选输出主干"}]}
```

- `bbox` 与 sidecar `detections[].bbox` 完全一致：原图像素 `[x, y, w, h]`
  （EXIF 转正后的显示坐标系）；省略 = 整图作源图（已是干净鸟图时用）。
- v1 仅支持 PIL 可解码格式（JPEG/PNG 等），不支持 RAW。

### 2.3 输出协议（子进程集成契约）

`--json` 时日志走 stderr，stdout 仅一行结果 JSON（`ensure_ascii=False`）：

```json
{"status": "ok|partial|error", "out_dir": "...",
 "results": [{"id", "image", "status": "ok|crop_only|error",
              "crop_path", "pixel_path", "pixel_paths",
              "seed", "prompt_id", "error"}],
 "summary": {"total", "ok", "crop_only", "failed"}}
```

退出码：`0` 全部成功；`1` 存在单项失败；`2` 致命（参数/文件/ComfyUI
不可达）。致命错误在 `--json` 模式下输出 `{"status":"error","error":...}`。

## 3. 处理流水线 / Pipeline

每任务项（串行执行，单项失败不中断批量）：

1. **读图**：`core.crop_advisor._load_image_exif_aware`（PIL 路径——
   中文/NAS 非 ASCII 路径安全 + EXIF 方向自动转正；与
   `core.crop_export.export_crop` 的「同 loader + box 直裁」既有组合
   保持同一坐标系语义）。
2. **方形裁剪**：bbox `(x,y,w,h)→(x1,y1,x2,y2)` 后走
   `tools.image_crop.smart_square_crop`（`--padding` 默认 0.25 留环境
   上下文）；无 bbox 整图直用；边长 < `--min-crop-px`（默认 320）判失败。
3. **存源裁剪图**：PIL 写 JPEG q95 至 `<out>/<label>/<name>_src.jpg`
   （label 做 Windows 非法字符/保留名清理；同 run 内重名自动 `_2/_3`）。
4. **ComfyUI 像素化**（有 `--workflow` 时）：上传（UUID 名防串图）→
   注入 LoadImage 节点（`--image-node` 指定或自动探测唯一节点）→
   写 seed（`--seed` 固定 / 默认每张随机，实际值回传结果 JSON）→
   `/prompt` 排队 → 轮询 `/history` → `/view` 收全部输出图，落
   `<name>_pixel<ext>`（多张 `_pixel_2..n`）。无 `--workflow` 时止步于
   裁剪导出（status=crop_only）。

## 4. ComfyUI 客户端 / tools/comfy_client.py

- `ComfyClient(host)`：`is_alive/upload_image/queue_prompt/wait_result/
  fetch_output`，httpx 同步实现，`with` 语义确定性关闭连接。
- 端点与官方示例脚本一致：`GET /system_stats`、`POST /upload/image`
  （multipart，overwrite）、`POST /prompt`（`{"prompt": wf,
  "client_id": uuid}`，`node_errors` 非空即报错）、`GET /history/{id}`
  （执行期间返回 `{}`，条目出现即结束，`status_str=="error"` 抛错）、
  `GET /view` 下载。
- **不启动/不管理 ComfyUI 进程**（调用方自启）；逐张串行（ComfyUI 自身
  即队列）；单张超时 `--timeout` 默认 300s。

## 5. 安全与非破坏性 / Safety

- 默认拒绝覆盖已存在输出，`--overwrite` 才覆盖；只新增文件。
- 不读不写 report.db / sidecar；不修改输入图片；无全局变量。
- Windows 控制台 `_force_utf8_stdio()`；文件一律 UTF-8。

## 6. 工作流模板 / workflows_template/pixel_art_api.json

纯内置节点（LoadImage → ImageScale 512 → VAEEncode → KSampler
img2img → VAEDecode → 最近邻缩 64px → 最近邻放回 512 →
SaveImage；SD1.5 系按 512 采样避免大分辨率重复伪影，换 SDXL 系调回
1024）。随 ComfyUI 安装已配好 `dreamshaper_8_pruned.safetensors`
（SD1.5 系 fp16）；调参说明见
`workflows_template/pixel_art_README.md`。工具会自动随机化模板里的
seed。注意 API 格式顶层只允许节点对象，说明文字一律放 README。

### 6.1 白底可爱风调参记录（2026-08-29）

目标：像素风 + 纯白底 + 可爱 Q 版 + 保留鸟种主要特征。同一翠鸟样张、
同 seed 的实测结论：

| 方案 | 结果 |
|---|---|
| 原图 i2i，denoise 0.55–0.75 | 鸟形连贯但背景杂色完全压不掉 |
| 原图 i2i，denoise 0.85–0.9 | 构图干净但背景粉/紫噪点，特征丢失 |
| 源图先 rembg 白底化再 0.8/0.85 | 构图干净、鸟形连贯 ✓ |
| 提示词强调 solid white background | 无裁决力，各档都有残留 |
| 生成后 rembg 抠图贴白 | 纯白底唯一确定性手段 ✓ |

最终方案（已固化为模板默认 + `--white-bg` 语义）：

1. 模板提示词换成可爱 Q 版系（cute pixel art bird/chibi/solid white
   background 正向加权，complex background/anti-aliasing 负向），
   denoise 0.85、CFG 8、dpmpp_2m + karras、steps 28。
2. `--white-bg` = 生成前源图 rembg 白底化 + 生成后 rembg 抠鸟贴白并按
   `--pixel-grid` 对齐 alpha（64px 格二值化，边缘吸附整数格）。
3. `--prompt`（批量 item 的 `prompt` 字段）注入鸟种英文特征词（如
   `"(common kingfisher:1.2), teal blue and orange plumage, long sharp
   beak"`）——高重绘下「抓住主要特征」依赖提示词而非源图。

rembg 为可选依赖（`pip install "rembg[cpu]"`），缺失时两级处理各自
告警降级、管线不中断。

### 6.2 全鸟种批量投产发现（2026-08-29，BirdIndex 462 种实测）

单样张调参通过后，BirdIndex 全鸟种（462 项）投产暴露两个缺陷与一处
提示词弱点，已在 `spb_pixel_art.py` 修复：

1. **mask 碎片**（`_keep_subject_alpha` 新增）：源图背景里有其他鸟/
   浪花时，rembg 软 mask 保留背景碎片（原 `alpha > 8` 阈值过松），
   白底化后成杂色斑点并被 img2img 二次固化。修复：alpha 按 127 二值化
   后只保留主体连通域（最大域 + 面积 ≥8% 的伴生域），两级白底处理
   共用。
2. **rembg session 重复初始化**（`_rembg_session` 新增）：rembg 2.0.8x
   的 `remove()` 不带 session 时每次调用新建 onnxruntime session，CPU
   图优化单次 ~40s（单次推理实际 ~1s），批量耗时完全失控。修复：按
   模型名缓存 session 进程内复用，单张端到端从 ~85s 降到 ~11s。
3. **场景外溢**：干净源图下模型仍偶发画岩石/地面（如群居海鸟场景，
   单 seed 复现）。投产清单改用加强约束的工作流变体（白底 1.5 加权、
   `single bird only`、负向补 rocks/ground/water/multiple birds 等），
   BirdIndex 侧存于其 `data/pixel_regen/pixel_art_atlas.json`，最难
   样张 3 seed 全部干净；后续可考虑合并回模板默认。

另：批量尾段 39 项因裁剪边长 <320px 失败（远距小鸟），用
`--min-crop-px 96` 补跑全数成功（工作流 512 上采样 + 64 像素格量化
对小源的糊化不敏感）。

## 7. 边界与后续 / Out of scope & future work

- v1 不支持 RAW 输入（rawpy 的 EXIF 方向语义与 bbox 坐标系需单独验证）。
- 不进 GUI / 不入 PyInstaller 打包（spb_* 工具源码运行惯例）。
- 后续可选：按 report.db 选图的批处理入口、生成结果对 BirdIndex 的
  回写约定、gif/动画输出支持。
