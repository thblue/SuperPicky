# 重构基线记录 / Refactor Baseline

记录 `refactor/cleanup` 分支起点（2026-08-30）的测试基线，重构全程以此为对照，
**任何阶段结束时若失败集合超出本清单即视为回归**。

This file records the test baseline at the start of the `refactor/cleanup` branch (2026-08-30).
Any new failure beyond this list at any stage is a regression.

## SuperPicky（.venv, Windows, QT_QPA_PLATFORM=offscreen）

命令：`pytest . -q --tb=no --ignore=test_submission_intake_contract.py`

- 结果：**416 passed, 4 failed, 13 skipped**（67.9s）
- 基线前已存在的失败（pre-existing failures，重构期间不要求修复，但不得新增）：
  1. `test_aesthetic_xmp_roundtrip.py::TestAestheticXmpRoundtrip::test_batch_write_roundtrip`
  2. `test_iqa_scorer_resize.py::test_calculate_aesthetic_matches_reference_and_is_faster`
  3. `test_multibird_pipeline.py::TestV54MultiMainAndSoftDelete::test_batch_soft_delete_species`
  4. `test_rating_quota.py::TestQuotaAssignment::test_burst_cap`
- 收集错误（collection error，排除在基线外）：
  - `test_submission_intake_contract.py` — import `tools.train.intake`，而 `tools/train` 按设计**从未入库**
    （.gitignore 注明"内部私有，不开源、不入库、不打包"）；该测试须归档或加本地 skip。
- skipped 13 项：多为真权重/真模型冒烟（`skipif` 资源不在位即跳），属预期。

## BirdIndex（.venv, Windows）

命令：`pytest tests -q`

- 结果：**111 passed**（6.2s），零失败零跳过。

## 接口冻结红线 / Frozen-interface red lines

见 `dev-docs/INTERFACE_CONTRACTS.md`。重构期间下列契约一个字节不动：
sidecar JSON 契约、report.db schema v13 及迁移链、birdid_server HTTP API、
`spb_rename_species.py` / `spb_pixel_art.py` 的路径与 CLI 参数面、
bird_reference.sqlite / birdname.db schema、superpicky_cli / birdid_cli 命令行接口。

## 重构结果 / Refactor Outcome（2026-08-30 收官）

- 每阶段结束全量回归：`pytest tests` = **428 passed / 4 failed（=基线原失败）/ 13 skipped**，与分支起点完全一致，零新增回归。
- 契约测试 12 项全绿（report.db schema v13 / sidecar 快照 / CLI 参数面）。
- BirdIndex 侧：111 项测试全绿；`web/api.py` 死代码清除。
- **端到端 sandbox 冒烟通过**（`scripts_dev/e2e_refactor_smoke.py`，只在
  `scripts_dev/_backfill_sandbox/` 演练，不碰真实库）：
  SuperPicky process → report.db v13 + sidecar → BirdIndex scan → Web 页面/API/缩略图
  → `POST /api/fix` → `spb_rename_species.py --apply` 子进程 → report.db/sidecar 同步
  → BirdIndex 定向重扫一致。
- ⚠️ **待办提醒**：模块布局有变（ai_model/iqa_scorer/topiq_model/post_adjustment_engine
  并入 core/），spec 无需改动（靠顶层 import 链静态分析），但**下次打包发版前必须跑一次
  打包启动冒烟**（AGENTS.md 最低验证标准）。

