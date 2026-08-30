# SuperPicky 架构总览 / Architecture Overview

> 面向新维护者的一页式全景：数据怎么流、进程怎么起、数据写在哪里、哪些模块危险。
> 接口冻结清单见 [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md)；
> 日常操作见 [howto/](howto/)；本目录结构见 [README.md](README.md)。
>
> A one-page map for new maintainers: data flow, process topology, where data is
> written, which modules are dangerous. Frozen contracts: INTERFACE_CONTRACTS.md.

## 1. 双项目全景 / Two-Project Panorama

```
                    ┌──────────────────────────────────────────────┐
                    │              SuperPicky（本仓库）              │
                    │                                              │
  照片目录 ────────▶ │  photo_processor（编排）                      │
  （NAS/本机）       │   ├─ ai_model        YOLO 检测/分割           │
                    │   ├─ birdid/         识鸟（YOLO+参考库）       │
                    │   ├─ rating_engine   评分/配额/定星            │
                    │   ├─ exiftool_manager 唯一 EXIF/XMP 写闸门     │
                    │   └─ report.db       结果库（schema v13）      │
                    │          │                                   │
                    │          └─ sidecar_export ──▶ meta/*.json   │
                    └───────┬──────────────────────┬───────────────┘
                            │                      │
              Lightroom 插件 │                      │ sidecar JSON（唯一数据源）
              HTTP 127.0.0.1:5156                  ▼
                            │            ┌──────────────────────┐
                            │            │   BirdIndex（姊妹仓）  │
                            │            │  FastAPI 127.0.0.1:8300│
                            │            │  birdindex.db 索引     │
                            │            │  照片墙/图鉴/旅程/统计  │
                            │            └──────────┬───────────┘
                            │                       │ 写回仅两条（子进程委托）：
                            ▼                       │  改种 → spb_rename_species.py
                 SuperBirdIDPlugin.lrplugin          │  像素图 → spb_pixel_art.py (+ComfyUI:8188)
```

**单一写者原则**：只有 SuperPicky 写照片数据（照片文件、EXIF/XMP、report.db、
sidecar JSON）。BirdIndex 对照片库严格只读，写回全部经子进程委托回本仓库。
契约细节见 INTERFACE_CONTRACTS.md。

## 2. 进程拓扑 / Process Topology

| 进程 | 端口 | 启动者 | 生命周期 |
|---|---|---|---|
| SuperPicky 主 GUI | — | `main.py` | 用户关闭即退 |
| birdid_server（Flask） | 5156（`SUPERPICKY_SERVER_HOST/PORT` 可覆盖） | GUI 内线程（打包版）/ `server_manager.py start` 守护子进程（CLI/LR） | 所有权登记 + atexit/信号兜底，见 `server_manager.py` |
| SPBBrowse（轻量浏览器） | — | `spb_browse.py`（PyInstaller 独立包） | 独立窗口 |
| BirdIndex Web | 8300 | `run.py` / `start_server.bat` | 常驻；`stop_server.bat` 按端口杀 |
| ComfyUI | 8188 | BirdIndex sprite 管线按需拉起 | 生成完收工；日志 `BirdIndex/data/comfyui.log` |
| Lightroom 插件 | —（纯 Lua HTTP 客户端） | Lightroom | 只调 5156，不启动服务 |

## 3. 目录分层 / Layout Layers

```
SuperPicky/
├── main.py / superpicky_cli.py / birdid_cli.py / birdid_server.py /
│   server_manager.py / spb_browse.py      ← 操作入口层（等价 /usr/bin，保持在根）
├── spb_*.py（9 个）                        ← 运维入口层：照片库批处理/审核/修复
│                                             （spb_rename_species / spb_pixel_art 路径冻结，
│                                              BirdIndex 固定路径调用）
├── config.py / advanced_config.py / constants.py  ← 配置层（保持在根：全仓 import + PyInstaller 依赖）
├── core/        处理流水线（photo_processor 编排、burst/rating/mover、sidecar_export、enhance 子包）
├── ui/          PySide6 界面（main_window 主窗口、settings_center 设置中心、results_browser…）
├── tools/       运行时支撑包（exiftool_manager 唯一写闸门、report_db、i18n、system_logger、
│                cli_processor、comfy_client…）——注意：不是脚本目录
├── birdid/      识鸟引擎（bird_identifier、osea_classifier、geo_filter）+ data/ 参考库
├── ioc/         birdname.db（IOC 14.2 名录/拼音）
├── models/      根级权重（cfanet/keypoint/yolo-seg…；下载由 scripts/download_models.py）
├── scripts/     发布/CI 辅助（不碰用户数据）：ci_release、download_models、sync_exiftool…
├── scripts_dev/ 运维脚本（直接读写真实数据，风险分级见其 README.md）
├── locales/     zh_CN / en_US 文案
├── SuperBirdIDPlugin.lrplugin   Lightroom 插件（Lua，HTTP 调 5156）
├── tests/       正式测试包（pytest；契约测试守护冻结接口）
├── dev-docs/    内部开发文档（本目录）
├── docs/        GitHub Pages 网站（superpicky.app，别与内部文档混放）
├── workflows_template/  ComfyUI 像素图工作流模板
└── inno/ packaging/ *.spec  安装器与打包清单
```

### BirdIndex 侧 / BirdIndex side

```
BirdIndex/
├── run.py                入口（FastAPI + uvicorn）
├── indexer/              scan（增量扫描）/ contract（sidecar 解析）/ db（schema v5）
│                         / queries / thumb（四级取图链）/ sprite（像素图编排）
│                         / fixer（子进程委托改种）/ speciesmeta（跨仓只读参考库）
├── web/                  pages.py（页面）+ api.py（/api/*）
├── data/                 birdindex.db(84MB WAL) + thumbs/ + fixes.jsonl（全部本地生成）
├── overrides/            人工覆写（代表照/简介校正/维基缓存）
└── config.json           78 个 NAS photo_roots + superpicky_dir + comfyui 路径（本地私有）
```

## 4. 数据写入点清单 / Where Data Gets Written

**重构时对照本表评估任何改动的数据影响。**

| 位置 | 写入者 | 内容 |
|---|---|---|
| `<照片库>/.superpicky/report.db` | `tools/report_db.py`（v13，迁移链 2→13） | photos/meta/corrections/bird_detections/export_stamps |
| `<照片库>/.superpicky/meta/*.json` | `core/sidecar_export.py` | 对外 sidecar（BirdIndex 数据源；V5.3 锚点/V5.5 无鸟不导出） |
| `<照片库>/.superpicky/cache/`、`review/` | photo_processor、spb_review | 生成预览/调试图/审阅图（可删，BirdIndex 取图链隐式依赖 temp_preview 约定） |
| 照片文件 EXIF/IPTC/XMP | `tools/exiftool_manager.py`（唯一闸门，`-overwrite_original_in_place`） | 评分/精选/物种标题说明；写模式 embedded/sidecar/none 由 advanced_config 控制 |
| 照片文件位置 | ⚠️ `core/rating_mover.py`、`core/file_manager.py`、`core/photo_processor.py`（整理移动）、`spb_flatten.py`、`spb_rename_species.py`（永不移动照片，只清缓存） | 评星/整理/扁平化移动（均有 manifest 或撤销清单） |
| 照片文件删除 | ⚠️ `spb_dedupe_jpg.py`（RAW+JPG 冗余 JPG）、`spb_rename_species.py --wipe`（仅软删 DB 检测框） | 均有确认/dry-run |
| `~/AppData/Local/SuperPicky/`（win） | config/advanced_config/server_manager/telemetry | advanced_config.json、birdid_server.pid、日志 |
| 当前活跃照片目录 `superpicky.log` | `tools/system_logger.py` | 运行日志 |
| `birdid/data/*.sqlite` | `scripts_dev/build_*` | 鸟种参考库（随发行包分发） |
| `BirdIndex/data/`、`overrides/` | BirdIndex | 本机索引/缓存/覆写，永不写 NAS |

## 5. 识别与评分流水线 / Pipeline at a Glance

扫描（`recursive_scanner` + `source_probe*`）→ RAW 转预览（rawpy）→ 有鸟检测
（`ai_model` YOLO）→ 无鸟补救扫描 → 识别（`birdid/bird_identifier` + `geo_filter`
地理候选过滤）→ 画质（锐度/美学 `iqa_scorer`/`topiq_model`）→ 定星（`rating_engine`
+ `rating_quota` V2 配额 + `burst_ranking` 连拍去重）→ EXIF 写回（exiftool_manager）
→ 整理移动（rating_mover，manifest 可逆）→ report.db 落库 → sidecar 导出 →
BirdIndex 增量扫描呈现。

视频链路：`video_analyzer`/`video_batch_engine`/`video_segment`（帧采样后复用同一识鸟/评分栈，分类器经 `core/birdid_adapter` 进程内直调，不走 HTTP）。

## 6. 已知结构债（二期提案）/ Known Debts (Phase-2 Candidates)

以下为**有意推迟**的重构对象（本次 refactor/cleanup 不动，避免高风险 GUI 回归）：
1. **god modules**：`ui/main_window.py`（4432 行）、`core/photo_processor.py`（4059 行）、
   `ui/settings_center.py`（2662 行）——拆分前需先补 UI 回归测试。
2. **core↔tools 循环依赖**：双向 import 靠函数内延迟 import 缓解；边界规则见
   `tools/__init__.py` 文档。
3. **稀有度语义双实现**：中国分/全球分 COALESCE 在 SuperPicky 运行时与 BirdIndex
   speciesmeta.py 各一份；语义一致性由参考库单一事实源 + 双侧测试守护。
4. **配置三层重叠**：constants / config / advanced_config 各持一份"目录名/阈值"语义，
   设置中心 SSOT 约定见 CLAUDE.md。
