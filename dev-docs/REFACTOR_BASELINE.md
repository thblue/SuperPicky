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
