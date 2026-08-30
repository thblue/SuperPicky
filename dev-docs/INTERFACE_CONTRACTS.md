# 对外接口契约（重构冻结清单）/ External Interface Contracts (Refactor Freeze List)

> **重构红线 / Red line**：本清单所列契约在 `refactor/cleanup` 期间**一个字节不动**。
> 每项契约有对应契约测试守护（`tests/` 下 `test_sidecar_contract_snapshot.py`、
> `test_report_db_schema_v13.py`、`test_spb_cli_surface.py`）。
> 需要变更时必须：先改本文档 → 与下游消费方（BirdIndex / Lightroom 插件）同步 → 双方测试同批更新。

## 0. 单一写者原则 / Single-Writer Principle

**只有 SuperPicky 写照片数据（照片文件、EXIF/XMP、report.db、sidecar JSON）。**
BirdIndex 对照片库严格只读；其唯一的写回路径是把改种/清除请求**子进程委托**回
SuperPicky 的 `spb_rename_species.py`。任何重构不得引入第二个写入者。

## 1. sidecar JSON 契约（BirdIndex 唯一数据源）

- 生产者：`core/sidecar_export.py`（report.db → `<照片库>/.superpicky/meta/<前缀>.json`）
- 消费者：BirdIndex `indexer/contract.py`（解析）+ `indexer/scan.py`（增量扫描）
- 关键约定：
  - `schema_version = 1`
  - `photo.filename`：带扩展名的文件名
  - `photo.library_path`：照片相对库根的**当前真实位置**（下游以此定位照片文件；
    V5.3 起契约锚点，存量缺失会自愈迁移重写）
  - `photo.relative_path`：首次处理位置（`original_path` 原样，存档语义）
  - `photo.preview_path`：可直接显示的 JPEG（RAW 预览缓存或伴随 JPG）
  - V5.5 起：无鸟照片**不导出** sidecar，且幂等删除历史遗留 JSON
  - 人工 `edits` 字段原样保留；内容哈希导出戳增量重写；tmp + `os.replace` 原子写
- 隐式依赖：`<库>/.superpicky/cache/temp_preview` 路径约定被 BirdIndex
  `indexer/thumb.py` / `indexer/sprite.py` 的取图链引用。

## 2. report.db schema（照片库内 `.superpicky/report.db`）

- 管理者：`tools/report_db.py`，`SCHEMA_VERSION = "13"`（meta 表存版本）
- 迁移链：v2 → v13 逐版 ALTER（`_upgrade_schema_if_needed`），**不得跳版、不得改历史迁移步骤**
- 表：`photos`（约 60 列，列定义见 report_db.py 顶部有序列表）、`meta`、`corrections`、
  `bird_detections`、`export_stamps`
- 消费者：SuperPicky 全链路；BirdIndex `indexer/fixer.py`（探测 `<dir>/.superpicky/report.db`
  判定处理根）；`scripts_dev/backfill_china_fields.py`（v12→v13 迁移 + 回填）；
  `tools/merged_report_db.py`（ATTACH 联查，依赖各库 schema 一致）

## 3. birdid_server HTTP API（Lightroom 插件依赖）

- 进程：`birdid_server.py`（Flask + flask-cors），默认 `127.0.0.1:5156`，
  可用环境变量 `SUPERPICKY_SERVER_HOST/PORT` 覆盖；生命周期由 `server_manager.py` 管理
- 路由（**无 /api 前缀**）：
  - `GET  /health`
  - `POST /recognize`
  - `POST /exif/write-title`
  - `POST /exif/write-caption`
- 消费者：`SuperBirdIDPlugin.lrplugin`（Lua，HTTP 调用，URL 可在导出设置里改）

## 4. `spb_rename_species.py` CLI（BirdIndex 站内改种的写后端）

- 调用方：BirdIndex `indexer/fixer.py`（子进程，单根超时 900s，解析 stdout 末行 JSON）
- 文件位置固定在仓库根；参数面：
  - 子命令 `photo <root> <filename> [--to-cn|--to-en|--to-sci | --wipe] [--apply] [--json]`
  - 子命令 `species <root> [--old-cn|--old-en|--old-sci] [--to-cn|--to-en|--to-sci | --wipe] [--apply] [--json]`
- 行为红线：默认 dry-run（`--apply` 才落盘）；照片文件永不移动/删除；软删可恢复；
  写 report.db + 同步 sidecar + 触发召回重算

## 5. `spb_pixel_art.py` CLI（BirdIndex 像素精灵的生成后端）

- 调用方：BirdIndex `indexer/sprite.py`（子进程，用 SuperPicky venv 的 Python 执行）
- 文件位置固定在仓库根；参数面（BirdIndex 实际使用）：
  `--batch <TASKS_JSON> --workflow <API_JSON> --host 127.0.0.1:8188 --white-bg
  --pixel-grid 128 --min-crop-px 96 --out <DIR> --json`
  （另有单图模式 `--image/--bbox/--label/--name` 及 `--image-node/--padding/--timeout/
  --seed/--overwrite/--prompt`，改动需评估 BirdIndex manifest 构造兼容性）

## 6. 鸟种参考库（跨仓只读）

- `birdid/data/bird_reference.sqlite`：表 `BirdCountInfo`、`bird_ioc`、`avilist_map`、
  `china_protection`、`gbif_rarity_by_country`（countrycode='CN'）、`gbif_rarity_100`
- `ioc/birdname.db`：IOC 14.2 名录（`version_id=10`），中文目科/拼音
- 消费者：BirdIndex `indexer/speciesmeta.py`（`mode=ro` 整表替换 species_meta）与
  `web/api.py` 的 `GET /api/taxonomy`（运行时只读 LIKE 查询）
- 稀有度语义（中国分优先、全球分兜底的 COALESCE）在 SuperPicky 运行时与 BirdIndex SQL
  各有一份实现，**语义必须保持一致**；改参考库结构属重大变更，须双仓同步。

## 7. 用户 CLI（操作习惯契约）

- `superpicky_cli.py`：process / reset / restar(t) / info / burst / identify / batch /
  batch-reset（详见 `dev-docs/reference/cli-reference.md`）
- `birdid_cli.py`：独立识鸟（eBird 区域过滤、`--write-exif`）
- `server_manager.py`：start / stop / restart / status
- 重构可加新命令/新选项，但不得改变既有命令的拼写、语义与输出格式。

## 8. BirdIndex 侧的依赖约定

- `config.json` 的 `superpicky_dir` 指向 SuperPicky 工作副本；Python 解释器解析链：
  `superpicky_python`（config 覆盖）→ `<superpicky_dir>/.venv/Scripts/python.exe` → `sys.executable`
- BirdIndex 从不写照片库；`POST /api/fix` 只是子进程转发到契约 §4 的 CLI。
