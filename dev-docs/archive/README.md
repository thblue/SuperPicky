# archive — 历史归档 / Historical Archive

本目录内容**不再维护**，仅供追溯。搬运自原 docs/、workflows/ 与根目录（2026-08 重构）。

Contents here are **unmaintained**, kept for reference only. Relocated from docs/,
workflows/ and the repo root during the 2026-08 refactor.

| 条目 | 原位置 | 归档原因 |
|---|---|---|
| `ChangeLog-4.5.0-details.md` / `ChangeLog-4.5.0-rc-history.md` | docs/ | 已并入根 `ChangeLog.md` |
| `API使用说明.md`（2026-01） | docs/ | birdid_server API 已演进；现行契约见 `dev-docs/INTERFACE_CONTRACTS.md` §3 |
| `BIRDID_OPTIMIZATION_GUIDE.md` | docs/ | CoreML 时代文档，4.2 起转 ONNX |
| `LIGHTROOM_PLUGIN_FEATURES.md`（v3.9.3） | docs/ | 插件功能已并入主文档与设置中心 |
| `dependency-version-audit-2026-04-11.md` | docs/ | 一次性核对报告 |
| `RELEASE_NOTES.md`（止于 4.2.0） | 仓库根 | 被 `ChangeLog.md` 取代 |
| `design/`（browser-rating-move） | docs/design/ | 早期设计，后被 specs/plans 体系取代 |
| `Focus-Points-Analysis.md` | workflows/ | 一次性分析 |
| `project_structure.md`（2026-01） | workflows/dev_docs/ | 目录结构早已变化；现行版见 `dev-docs/ARCHITECTURE.md` |
| `scripts/` | scripts_dev/ 等处 | 一次性修复/诊断脚本（aesthetic_topiq_diagnostic、fix_singlebird_bbox、probe_multibird_ui、reset_photo_dir）与坏测试 test_submission_intake_contract（import 从未入库的 tools.train） |
| `website/` | docs/ | 无人引用的旧版教程页（tutorial-3.9.4/4.2.1.html）及其专属截图（img/V3_9_3、V_3_9_4、V_4_0_6、Howto、tutorial） |

> 仓库根还有一个 gitignore 的 `archive/`（跑批临时产物：_tm_hasbird.json、
> 有鸟目录审核清单.html、keep_list.txt）——那是**本地产物**归档，不随 git 走；
> 本目录是**文档**归档，随 git 走。
