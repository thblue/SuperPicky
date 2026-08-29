# pixel_art_api.json 使用说明 / Workflow README

spb_pixel_art 的像素风工作流模板（ComfyUI **API 格式**，可直接被
`--workflow workflows_template/pixel_art_api.json` 加载；也可在 ComfyUI
界面载入后微调，再用 *Save (API Format)* 重新导出覆盖本文件）。

API-format pixel-art starter workflow for spb_pixel_art; load it in the
ComfyUI UI, tune, then re-export via "Save (API Format)".

## 链路 / Graph

```
LoadImage(1) → ImageScale 512(lanczos)(2) → VAEEncode(3)
  → KSampler img2img denoise=0.85(7) → VAEDecode(8)
  → ImageScale 64 nearest-exact(9)   ← 像素化：缩到 64px 形成像素格
  → ImageScale 512 nearest-exact(11) ← 8 倍整数放大，格子边缘锐利
  → SaveImage(12)
```

Checkpoint 加载器是节点 `10`，当前配置
`dreamshaper_8_pruned.safetensors`（SD1.5 系、fp16；采样分辨率按
SD1.5 习惯设为 512，换 SDXL 系请把节点 `2`/`11` 调回 1024）。

## 白底可爱像素风配方 / Cute white-background recipe

模板默认提示词与参数即为「Q 版可爱 + 纯白底」风格调校（实测记录见
设计文档），CLI 侧配套用法：

```bash
python spb_pixel_art.py --image 鸟图.jpg --bbox X Y W H --label 翠鸟 \
    --workflow workflows_template/pixel_art_api.json \
    --white-bg --pixel-grid 64 \
    --prompt "(common kingfisher:1.2), teal blue and orange plumage, long sharp beak" \
    --out 输出目录 --json
```

- `--white-bg`：**必开**。两段 rembg 处理——生成前把源图裁剪白底化
  （消除照片背景杂色源头，Q 版构图才干净），生成后抠鸟贴纯白并按
  `--pixel-grid` 对齐 alpha（提示词永远无法保证纯白底，后处理才能）。
- `--prompt`：传鸟种英文名+特征词（颜色/嘴形），高重绘下保住「主要
  特征」全靠它；批量模式写在每个 item 的 `prompt` 字段。中文鸟种名
  需自行译成英文，SD 系模型对英文提示词响应最好。
- `--pixel-grid 64` 须与节点 `9` 的宽高一致。

## 常用调参 / Tuning

| 想要的效果 | 改哪里 |
|---|---|
| 更像照片 / 更 Q 版放飞 | 节点 `7` `denoise`（0.7 保形 / 0.85 默认 / 0.9 通用化） |
| 像素格粗 / 细 | 节点 `9` 宽高（64=粗格 / 128=细格），同步改 `--pixel-grid` |
| 出图尺寸 | 节点 `11` 宽高（保持是节点 9 的整数倍，如 64×8=512） |
| 提示词风格 | 节点 `5` 正向 / 节点 `6` 负向 |
| 采样器/步数 | 节点 `7` `dpmpp_2m`+`karras`、`steps 28`、`cfg 8` |

注意：节点 `9` 与 `11` 宽高保持相等比例可避免拉伸；`seed` 无需手改，
spb_pixel_art 每次运行会随机覆盖（`--seed` 可固定复现）。

## 自定义工作流的接入要求 / Custom workflows

- 必须恰好含一个 `LoadImage` 节点（多个时用 `--image-node` 指定）。
- 至少一个 `SaveImage` 节点产出图片（工具按 `/history` outputs 收图）。
- 顶层只允许节点对象（ComfyUI API 格式约束），说明文字请放本 README。
