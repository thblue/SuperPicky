# 选片工具链收敛（restar 废弃 / rerate 转正 / 编辑器边界）设计

- 日期 / Date: 2026-09-06
- 状态 / Status: 草案，待评审（2026-09-06 天坛/海淀实跑事故与修复后沉淀）
- 关联 / Related: `dev-docs/howto/PROCESS_QUICKSTART.md`（已知坑）、`dev-docs/reference/cli-reference.md`（restar 警示）、`scripts_dev/rerate_v2_conf_gate.py`

## 1. 背景与问题 / Background

2026-08-31（天坛）与 2026-09-06（海淀）两个批次的实跑暴露了选片工具链的
三处结构性问题，临时修复已落地，但工具分工的正式设计一直缺位：

1. **restar 事故（2026-08-31）**：`superpicky_cli.py restar` 在 V2 配额模式下
   把全部 64 张有鸟照片砸成 0★（0★ 下限读配置 `min_sharpness/min_nima`
   而非 `-s/-n` 参数）、并无视 `folder_layout=flat` 把 44 张照片移进
   `0星_放弃/`。事后 DB 回滚、文件复位，但发现移动过程经
   `find_image_file` 把 44 个文件的扩展名存储大小写改成了小写
   （`.CR3`→`.cr3`，见 §4.3）。
2. **补救扫描守门缺陷（2026-09-06 已修复）**：识鸟守门从 1024 预处理图上
   抠候选框，占画面 0.1% 的目标只剩几十像素，分类器必然低分——
   「YOLO 检出但救不回」。已修复为优先全分辨率原图裁剪
   （`core/ai_model.py` `_load_fullres_for_confirm`），并暴露守门门槛
   `rescue_birdid_gate=10%` 在高分辨率裁剪下偏松（枯花序被 10% 压线误救）。
3. **工具分工含糊**：用户面对「重定星」需求时有 restar / rerate 脚本 /
   浏览器多鸟编辑 / 右键修改鸟种四个入口，职责边界靠口口相传，
   本次用代码事实厘清（§3），需要固化为设计。

## 2. 目标 / Goals

- 明确四类「改评分 / 改鸟种」需求的**唯一正确入口**，写进 howto。
- **废弃 restar 在 V2 模式下的可用性**（防呆，而不仅是文档警告）。
- **rerate 能力转正**：从 scripts_dev 一次性脚本升格为受支持的 CLI 入口。
- 修复 `find_image_file` 的大小写缺陷，消除未来同类事故的土壤。
- （远期）多鸟编辑器补齐「新增框」与纠错样本记录，收敛补录场景。

## 3. 非目标 / Non-Goals (YAGNI)

- 不改变 V2 定星算法本身（硬门槛/配额/Q 分/封顶规则不动）。
- 不改动浏览器右键「修改鸟种」与多鸟编辑器的现有行为（除 §4.5 远期项）。
- 不引入新的持久化层；三处存储（DB / sidecar / EXIF）的写入时机只在
  §4.2 统一，不做同步后台任务。
- 不处理天坛 44 个小写扩展名文件以外的新增迁移。

## 4. 现状事实盘点 / Facts（代码核实，2026-09-06）

### 4.1 三个工具的能力矩阵

| 能力 | 多鸟编辑器 | 右键修改鸟种 | restar | rerate 脚本 |
|---|---|---|---|---|
| 层次 | 单张逐框精修 | 单张快速定种/补录 | 整批重算（V1 逻辑） | 整批重算（V2 逻辑） |
| 无检测框照片 | ✗（依赖已有框，无新增框能力） | ✓（补录设计场景） | 不适用 | 不适用 |
| corrections 纠错样本 | ✗ 不记录 | ✓ 记录并可供「提交纠错」 | ✗ | ✗ |
| 整批操作 | ✗ | 整批改种/删种（鸟种下拉右键） | ✓ | ✓ |
| 写入目标 | sidecar JSON（镜像 bird_detections） | DB 鸟种字段直写 | DB + 移动文件 | DB rating/caption |
| 移动照片文件 | ✗ | 非 flat 布局时移动 | ✓（无视 flat，缺陷） | ✗ |

### 4.2 评星的三处存储与写入语义

- `report.db photos.rating/caption` —— **权威存储**，浏览器列表/筛选/统计读它；
- sidecar JSON `processing.rating` —— 浏览器人工编辑层（多鸟编辑器写），
  经 `spb_sync_edits` 回放进 DB；
- EXIF 星级 —— 按 `metadata_write_mode`（当前 none 不写）。

**任何「重定星」必须同步前两处**（EXIF 按模式），只改其一会造成
浏览器与权威库脱节。rerate 脚本当前只写 DB rating（caption 未同步）
——转正时需补 sidecar 同步（§4.6 开放问题）。

### 4.3 `find_image_file` 大小写缺陷（44 个小写扩展名的根因）

`core/post_adjustment_engine.py:109-120`：按「前缀 + 候选扩展名列表」拼路径
找文件，候选列表**小写优先**；Windows/SMB `os.path.exists()` 大小写不敏感，
命中的是真实大写文件、返回的却是小写构造路径。restar 据此 `shutil.move`
落盘成小写（44 张 = 当时评级变更集合），恢复现场时原名带回。
DB `current_path` 仍全为大写，与磁盘不一致（Windows 下无功能影响，
跨平台/大小写敏感消费方有隐患）。

## 5. 设计方案 / Design

### 5.1 D1 — restar 防呆废弃（本周期）

- CLI `restar` 入口检测 `rating_algorithm == "v2"`：直接拒绝执行，打印
  指引「V2 批次重定星请用 `rerate-v2` 子命令（见 §5.2）」，退出码非 0。
- 无论 V1/V2：organize 移动逻辑强制尊重 `folder_layout=flat`
  （flat = 永不移动，`--organize` 显式传参才允许，且提示数据安全红线）。
- V1 用户的绝对阈值重评星行为保持原样（仅去移动缺陷）。
- `cli-reference.md` / QUICKSTART 的警示段落随之收敛为一句「V2 已禁用，
  用 rerate-v2」。

### 5.2 D2 — rerate 转正为 CLI 子命令（本周期）

- `superpicky_cli.py rerate-v2 <目录>`：逻辑即
  `scripts_dev/rerate_v2_conf_gate.py`，参数保持 `--min-conf/--quota3/
  --quota2/--execute`，默认 dry-run + 双自校验（复现存库评级、鸟种分组
  校验和）+ 写前自动备份。
- **转正时补齐写入面**：变更照片同步重写 sidecar `processing.rating`
  （编辑层一致），caption 首行已同步（现实现已含）。
- V2 定星语义的唯一批量入口；脚本版保留一个薄壳转发到 CLI，避免双实现。
- howto QUICKSTART「配套命令」表更新为唯一推荐入口。

### 5.3 D3 — find_image_file 大小写修复 + 一次性归位（本周期）

- `find_image_file` 命中后用 `os.listdir` 取**磁盘真实存储名**拼返回路径
  （保持现遍历顺序语义，仅替换名字来源）；递归分支同理。
- 一次性运维：把天坛 44 个 `.cr3` 改回 `.CR3`（Windows 下 `os.rename`
  仅大小写变更可直接执行；操作前以 DB `current_path` 为准核对清单）。
  归属 scripts_dev 一次性脚本，跑完归档。

### 5.4 D4 — 守门门槛再校准（✅ 已定档 25%，2026-09-12）

- 修复后守门置信度分布整体上移，原 `rescue_birdid_gate=10` 偏松
  （实误救：天坛 027A4902 枯花序「紫金鹃 10%」压线；修复前同模式误救
  已存在于沙河批：红鹑鸠 13%、棕尾鹟䴕 14%）。
- **定档 25**：实救正例 027A5094（83%）不受影响，压线误救被拦。
  代码默认值与测试已同步；本机 GUI 配置已显式改为 25（存量配置
  不自动迁移）。

### 5.5 D5 — 多鸟编辑器补齐「补录」能力（远期，单独 spec）

- 新增框：在无框/少框照片上手动画框后走逐鸟分类（需复用 crop_advisor
  与 birdid 通道，工作量中等）。
- corrections 记录：编辑器内改种/删框也写 corrections 表，对齐右键
  修改鸟种的数据流。
- 完成前，右键「修改鸟种」**保留**，作为无框补录与纠错样本的唯一入口。

## 6. 开放问题 / Open Questions

1. rerate-v2 转正后，caption 首行重写是否也要覆盖「星级未变但 caption
   原因措辞过时」的照片？（现实现只动变更照片。）
2. ~~D4 门槛定档数值（25 vs 30）~~ 已定档 25（2026-09-12）。
3. 天坛 44 个小写文件归位脚本是否顺带扫描全库（77+ NAS 目录）同类
   不一致？建议先只做天坛，全库扫描另立任务。

## 7. 里程碑 / Milestones

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 守门全分辨率修复 + 测试 + 文档警示（§1.2） | ✅ 已落地（7fd562b） |
| M2 | rerate 脚本 + 扫描诊断脚本 + 文档 | ✅ 已落地（7fd562b） |
| M3 | D1 restar 防呆 + D2 rerate-v2 子命令转正 | ✅ 已落地（本提交） |
| M4 | D3 大小写代码修复 | ✅ 已落地（本提交）；44 文件归位：用户手动处理未生效（Windows 不支持仅大小写改名），待执行归位命令 |
| M5 | D4 门槛定档 | ✅ 已落地（25%，2026-09-12） |
| M6 | D5 编辑器补录能力 | 远期，另立 spec |

## 8. 实施记录与偏差 / Implementation Notes（2026-09-06）

> 2026-09-12 追加（模型自查 review 修复）：
> - **P1**：rerate-v2 日志参数解析的正则 `(\d+)%\s*<\s*(\d+)%` 同时匹配评级
>   守门行（紧凑，MM = -c）与识鸟低置信行（带空格，MM = birdid 阈值），
>   结果取决于行序。修正为仅匹配紧凑形态；并让 process 收尾把生效参数
>   （min_conf/quota3/quota2）写入 report.db meta 表，rerate 按
>   meta > 日志 > 配置 解析（同时解决 quota2 不入日志的历史盲区）。
> - **P2-2**：backfill_species 采纳时漏写 photos 级稀有度四列
>   （iucn/gbif_rarity_100/aesthetic_index/china_protection）——已补；
>   存量 25 张经 `--repair-rarity` 模式修复（detections 行复制 gbif/china，
>   class_id 查参考库补 iucn/aesthetic，不动鸟种与置信）。

- **D1 简化**：flat 布局下 organize 一律不移动（未设 `--organize` 覆盖开关
  ——移动与 flat 语义本就互斥，多一个开关只增加误用面）。
- **D2 实现形态**：核心逻辑入 `core/rerate_v2.py`（CLI 与 scripts_dev 薄壳
  共用单一实现，替代原「薄壳转发到 CLI 子进程」设想）；新增
  **人工改星覆盖层豁免**——自校验发现少量与管线复算不符的照片（≤
  max(3, 有鸟数 10%)）判定为浏览器人工改星，重算时保留人工评级、跳过
  caption/sidecar 重写；超容忍度仍拒绝写库。该机制源于实跑：海淀两批
  后用户已在浏览器人工改星，纯管线复现校验会误拦。
- **D2 自校验参数来源**：置信门槛解析自日志守门拒绝行「NN%<MM%」，
  3★ 配额解析自 V2 定星汇总行；quota2 未入日志，回退当前配置并告警，
  可用 `--current-*` 三参数显式覆盖。
- **D3 归位**：用户手动改名未生效（Windows 资源管理器不支持仅大小写
  改名），归位命令见 howto 或用
  `python -c` 两段式 rename（先改临时名再改回目标大小写）。
