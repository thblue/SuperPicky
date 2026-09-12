# scripts_dev — 运维与数据构建脚本 / Ops & Data-Build Scripts

本目录是**正式入库**的运维脚本层：面向开发者的一次性数据构建、历史库回填与算法校准验证。
与 `scripts/`（面向发布/CI 的构建辅助）的边界：**scripts_dev 会直接读写真实数据**（参考库、存量照片库的 report.db），`scripts/` 不碰用户数据。

This directory holds **tracked** ops scripts: one-off data builds, historical-DB backfills,
and algorithm calibration/validation. Boundary vs `scripts/`: scripts_dev may touch real data
(reference DBs, existing photo libraries' report.db); scripts/ never touches user data.

## 风险分级 / Risk Tiers

### 🔴 直接写存量照片库（NAS 真实数据）/ Writes existing photo libraries (real NAS data)

| 脚本 | 作用 | 风险与防护 |
|---|---|---|
| `backfill_china_fields.py` | 为历史 `.superpicky/report.db` 回填中国稀有度 + 国家保护等级两列 | 内置 v12→v13 迁移；先在 `_backfill_sandbox/` 副本验证，再上真库；配套测试 `test_backfill_china_fields.py` |
| `run_backfill_all.py` | 批量驱动上一脚本：目录清单读自 `G:/code/BirdIndex/config.json` 的 `photo_roots`（77+ NAS 库） | 全量回填已执行完毕（2026-08）；重跑前务必确认 `--dry-run` 行为与目标清单 |
| `rerate_v2_conf_gate.py` | V2 配额重定星：不重跑检测/识鸟，从 report.db 现成指标 + 日志鸟种标签重建定星输入（替代 V2 模式下不可用的 CLI `restar`，见 cli-reference.md） | 默认 dry-run；两道自校验（当前参数复现存库评级 + 鸟种分组校验和）不通过即拒绝写库；`--execute` 前自动备份 report.db；只写 rating/caption，零接触照片文件。**已转正为 CLI `rerate-v2` 子命令，本脚本仅剩薄壳** |
| `backfill_species.py` | 未定种照片补种：按指定采纳门槛（如 40%）对有鸟未定种照片重识别，镜像写入 photos / bird_detections / sidecar 三处 | 默认 dry-run；`--execute` 前自动备份 report.db；零接触照片文件；低置信段（40-50%）候选错误率偏高，采纳后需人工复核 |

### 🟡 写 birdid 参考库 / Writes birdid reference DBs (`birdid/data/*.sqlite`)

| 脚本 | 作用 |
|---|---|
| `build_china_rarity.py` | 构建 `bird_reference.sqlite` 的 `gbif_rarity_by_country`（CN）中国国别稀有度表 |
| `build_china_protection.py` | 构建国家保护等级表 `china_protection` | 
| `build_geo_distribution.py` | 调 GBIF Occurrence API 生成 1° 网格地理分布库 `geo_distribution.db` |
| `build_iratebirds_table.py` | 从 iRateBird（figshare, CC-BY 4.0）构建鸟种美学指数表 |

> 参考库随发行包分发；重建会改变识别/稀有度行为，需同步核对 `docs`（现 `dev-docs/reference/`）下的
> GBIF_RARITY_INDEX / CHINA_RARITY_AND_PROTECTION 说明并重跑 BirdIndex 的 `python -m indexer meta|rarity`。

### 🟢 只读/校准验证 / Read-only or calibration

| 脚本 | 作用 |
|---|---|
| `calibrate_geo_threshold.py` | 从 GBIF 采样标定 `geo_filter` 的 L1 候选集阈值（产出报告，不改库） |
| `validate_geo_filter.py` | 用 433 张法罗群岛/冰岛真实素材对地理过滤做回归验证（对照 spec §8） |
| `validate_rescue_scan.py` | 用 39 张确认有鸟的漏检 ARW 验证无鸟补救扫描的救回率（跑真实模型推理） |
| `reexport_sidecars.py` | 存量库 sidecar 升级重导：sqlite `mode=ro` 只读打开 report.db，复用生产导出逻辑只写 `meta/*.json`，零接触照片（原 tools/ 迁入） |
| `diag_missed_detection.py` | 漏检诊断：对单张 RAW 复现「主检 640 / 补救 1024 / 单张全分辨率」三条 YOLO 链路并输出原始分数（只读） |
| `scan_no_bird_rescue.py` | 无鸟照片补救扫描报告：用当前守门逻辑重判目录内全部 has_bird=0 照片并列出可救回清单；不写 report.db，仅重新生成无鸟预览到应用自管缓存 |

## 本地目录（不入库）/ Local-only dirs (untracked)

- `_backfill_sandbox/` — 回填演练用的历史库副本；**永远先在这里验证再碰真库**
- `data_sources/` — iRateBird figshare 原始 CSV（手动下载，CC-BY 4.0）

## 测试 / Tests

`test_backfill_china_fields.py`、`test_build_china_tables.py`、`test_build_iratebirds_table.py`
（仓库 `tests/` 下）直接 import 本目录脚本；改名/移动前先同步这三处。
