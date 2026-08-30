# dev-docs — 内部开发文档 / Internal Developer Docs

> 对外网站在 `docs/`（GitHub Pages，superpicky.app）；本目录是**内部**文档层，
> 不随网站发布。重构于 2026-08（原 docs/ 内部文档全部迁入此处）。
>
> The public site lives in `docs/`. This tree is internal-only documentation.

## 入口导航 / Entry Points

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 双项目全景：数据流/进程拓扑/目录分层/数据写入点/结构债 | 新人报到、动架构之前 |
| [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md) | 对外接口冻结清单（sidecar/report.db/HTTP/CLI 面）+ 单一写者原则 | **改任何对外接口之前，必读** |
| [REFACTOR_BASELINE.md](REFACTOR_BASELINE.md) | refactor/cleanup 分支测试基线与失败清单 | 重构期间对照回归 |
| [howto/](howto/) | 操作手册：新照片处理 runbook、spb 工具速查、构建发布 | 日常处理照片、发版 |
| [reference/](reference/) | 机制与算法参考：CLI 参考、定星去重规则、GBIF 稀有度、中国保护等级、识别管线… | 需要行为/算法细节时 |
| [specs/](specs/) | 功能设计规格（一对specs/plans对应一个功能），带状态索引 | 查某个功能"为什么这么设计" |
| [plans/](plans/) | 实施计划（与 specs 成对） | 同上 |
| [archive/](archive/) | 历史文档与一次性脚本（不再维护，仅供追溯） | 考古 |

## 文档规则 / Conventions

- 新功能开发：在 `specs/` 放 `<日期>-<功能>-design.md`，`plans/` 放同名 plan；
  落地后更新对应 INDEX.md 状态为「已落地」。
- 会影响对外契约的改动：先改 `INTERFACE_CONTRACTS.md`，再改代码，同批更新
  `tests/` 下契约测试。
- 涉及照片库写入的脚本/工具：在 `scripts_dev/README.md` 或 howto 中标注风险等级。
