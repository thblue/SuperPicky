---
name: batch-identify
description: 对指定照片文件夹执行 SuperPicky 标准批量处理（检测→评分→定星→识鸟→视频封面入库→产出浏览库），含 DPP 伽马编辑检测与重跑（刷亮缩略图→重算指标→补种→重定星）。只要用户提到「批量识别」「批量处理」「跑批」「处理一下这批/这个文件夹」且涉及鸟片目录（如 \\NAS-server\PHOTO\观鸟\ 下的新拍文件夹），或者说「跟之前一样处理」，就用本 skill。固化了实测过的命令参数、跑前检查、跑后核对与汇报格式。
---

# SuperPicky 批量识别工作流

把一个新拍的鸟片目录跑完「检测 → 评分 → 定星 → 识鸟 → 视频封面入库 → 产出浏览库」，并核对结果后向用户汇报。
完整背景文档：`dev-docs/howto/PROCESS_QUICKSTART.md`（本 skill 是它的可执行浓缩版）。

## 1. 跑前检查

1. **定位目录**：用户可能给完整 UNC 路径（`\\NAS-server\PHOTO\观鸟\2026 观鸟\2026.8 天坛`）
   或只给文件夹名——按名字在 `\\NAS-server\PHOTO\观鸟\` 及其子目录下定位，找不到再问。
2. **确认未处理过**：目录下已存在 `.superpicky/report.db` 说明跑过。首次处理直接继续；
   重跑必须先：① 备份 `report.db` 为 `.superpicky/report.db.bak_<原因>_<时间戳>`；
   ② 确认没有 SPBBrowse/主程序进程开着该目录（编辑中的浏览库不能重跑覆盖）。
3. **统计文件数**：数一下 RAW（CR3/NEF/ARW…）和视频（MP4/MOV/M4V）数量，后面核对要用。
4. **确认 venv**：一律用 `G:\code\SuperPicky\.venv\Scripts\python.exe`，命令加 `-X utf8`。

## 2. 执行（参数已固化，不要自行增减）

```bash
cd G:/code/SuperPicky
.venv/Scripts/python.exe -X utf8 superpicky_cli.py process -i "<目录>" --birdid-country CN -c 40 --birdid-threshold 40
```

- **`-i`（= `--auto-identify`）必须显式加**：识鸟总开关只认这个 CLI 参数、不回落配置。
  漏掉不报错——评分照跑，但鸟种沿用库里旧数据，等于白跑。日志里出现
  `Multi-bird` / `Bird ID` / `Low confidence` 行才是真跑了识鸟。
- **`--birdid-country CN` 是国内目录默认**；海外拍摄按实际国家传（如 `AU`、`SG`）。
  不传时链路是「GPS 反查国 → 兜底 CN」，国内显式传 CN 最稳。
- **`-c 40`（AI 置信度门槛）是本工作流的定档参数**（2026-09-06 用户定档 40，
  50 偏严）：显式传以覆盖配置（当前配置 min_confidence=0.7）。它同时控制
  有鸟判定、V2 排序池入池线和补救扫描的直接救回线，影响星级密度。
- **`--birdid-threshold 40`（鸟种采纳/存储门槛）同为本工作流定档参数**
  （2026-09-12 定档 40；GUI 配置 birdid_confidence 与代码默认值已同步为 40，
  显式传参作双保险）：低于 40% 的候选不定种入库。注意种名错误率随门槛
  降低而上升，跑后对「可疑鸟种」（跨分布物种、40-50% 低置信段）的
  人工复核更重要。
- **视频阶段默认开启**（不需要额外参数）：目录里的 MP4/MOV/M4V 会在照片阶段后
  自动抽「YOLO 置信度最高的有鸟帧」生成封面 `<视频名>_vcover.jpg`，作为普通
  photos 记录入库——有鸟默认 **2 星**（不进 40%/20% 配额定星）、无鸟 -1 不出
  sidecar；封面走多鸟识别，视频作为封面伴生文件一起归入 `{鸟种}/2星_良好/`。
  浏览库里封面即缩略图，改星/改种视频随行，删除连带视频，详情面板可「打开视频」。
  确实不要处理视频时加 `--no-videos`。
- **其余参数一律不传**：锐度/美学/配额/布局/写入模式全部跟随
  `advanced_config.json`（GUI 高级设置改了就跟着变），这正是「跟之前一样」的含义。
  特别注意：V2 配额定星**不消费** `-s/-n` 阈值；2★/3★ 名额密度由配置里的
  `custom_quota3`/`custom_quota2` 控制。
- **长任务放后台**（Bash `run_in_background`），预计 ≈1.3 秒/张（RTX 3090 + NAS 实测），
  每隔一两分钟看一眼输出确认在推进即可。

## 3. 跑后核对（必做，缺一不可）

1. **行数对账**：`report.db` photos 行数 == RAW 文件数 + 视频数（视频封面也是 photos 行）。
   不等 → 在输出里搜「异常被跳过」（NAS 偶发 WinError 5 锁文件）或「视频阶段」失败行，
   被跳过的照片/视频保留旧结果，需补跑。
2. **日志统计**：从输出尾部提取「处理完成统计」（星级分布/飞鸟标记/总耗时）和
   「视频阶段完成」行（封面数/有鸟数/已定种数/无鸟数/已归类数）。
3. **sidecar 对账**：`.superpicky/meta/` 的 JSON 数 == has_bird 数（无鸟照片和
   无鸟视频封面按瘦身策略不导出；有鸟视频封面正常导出）。
4. **DPP 伽马编辑核对**：V5.9.4（W1）起跑批内已「人工优先」消化 DPP 编辑——
   RAW 转换前批量预读 CR3 的 CanonVRD recipe，本次新抽取的预览先套人工 LUT、
   跳过自动提亮不叠加（日志见「DPP 人工伽马编辑 N 张」与逐张 `DPPGAMMA` 行，
   开关 `advanced_config.preview_dpp_gamma` 默认开）。因此：
   - 跑批日志已有 DPPGAMMA 行、且跑批后没再进过 DPP → 本步跳过；
   - 跑批**之后**又做了 DPP 伽马编辑、或处理 V5.9.4 之前的老批次 → 走第 4 节
     （其 `--dry` 只读不动缓存，可放心先用）。
5. **未定种回捞**：`scripts_dev/backfill_species.py "<目录>" --threshold 40`
   （dry）→ 可采纳 >0 就 `--execute`。V5.6 起识鸟门控有替代通道（喙可见或
   YOLO≥0.6 照常识鸟）+ 门控拒绝仍落框行，「零机会」照片已从源头消灭；
   此步兜住残留的：分类置信 <40% 的换跑批时点重试、无检测行旧照片重检。
   采纳清单按 40-50% 段标注复核。跑过伽马重跑（第 4 节③）的不用重复。
6. **EXIF 缺口核对**：`scripts_dev/backfill_rescued_exif.py "<目录>"`（dry）。
   应报「无待回填行」；若列出照片（有鸟但日期为空——无鸟救回/改种链路
   缺口的指纹特征），加 `--execute` 回填（写前自动备份），再跑
   `scripts_dev/reexport_sidecars.py "<目录>"` 重导。视频封面行自本次
   修复起随封面写入拍摄日期（本地墙钟），老批次缺日期的封面也被这一步兜住。

## 4. DPP 伽马编辑重跑（跑批后新做的编辑 / V5.9.4 之前的老批次，顺序不可乱）

```bash
cd G:/code/SuperPicky
D="<目录>"
.venv/Scripts/python.exe -X utf8 scripts_dev/refresh_gamma_previews.py  "$D"            # ① 刷亮缩略图
.venv/Scripts/python.exe -X utf8 scripts_dev/recalc_gamma_scores.py     "$D" --execute  # ② 重算指标（先 dry 看清单）
.venv/Scripts/python.exe -X utf8 scripts_dev/backfill_species.py        "$D" --threshold 40 --execute  # ③ 补种
.venv/Scripts/python.exe -X utf8 superpicky_cli.py rerate-v2 "$D" --min-conf 0.4 --metrics-rebuilt --execute  # ④ 重定星
```

- V5.9.4（W1）起，跑批**前**已存在的 DPP 编辑由跑批内自动消化（人工 LUT 优先，
  跳过自动提亮）；本节只服务两种情况——跑批**后**新做的编辑回补、V5.9.4 之前
  处理的老批次。对已消化的批次实跑本节 ① 幂等无害（重抽原始预览再套同一 LUT，
  产出一致），但 ②③④ 不必空转（④ 的 `--metrics-rebuilt` 无人工改星豁免，
  指标没变就不要跑）。
- **① 刷缩略图**：解析每张 CanonVRD 伽马中点 → 幂律 LUT（`g=2^中点值`）应用到
  `.superpicky/cache/temp_preview/`。幂等（每次从 CR3 重抽原始预览再套 LUT，
  重复跑不叠加）；`--dry` 只读不动缓存；回退 = 删缓存 jpg 自动重生原始版。
  只动缓存，零接触照片/DB。
- **② 重算指标**：用提亮预览按主管线口径（主鸟框+15% padding → 关键点锐度/眼/喙
  → TOPIQ 鸟裁剪区 → ISO 归一化 + 原 caption 对焦权重 + 飞版加成）更新
  head_sharp/eyes/beak/nima_score/adj_* 列。无库内框的照片（当时锐度 0 被识鸟
  门控挡下、没写检测行）自动 YOLO 重检兜底。真糊片（拍摄即脱焦）提亮也救不了，
  维持 0★ 是正确结果，不是流程失败。
- **③ 补种**：未定种照片用亮图重识别（40% 采纳线，人工改种/删鸟自动保护）。
  40-50% 低置信段错误率偏高，采纳清单要进汇报供人工复核。
- **④ 重定星必须带 `--metrics-rebuilt`**：指标已变，rerate-v2 的复现校验
  （分组校验/自校验）前提「指标未变」不成立，不带会被正确拦截。该模式全部
  按新指标重算（无人工改星豁免层），写前自动备份。
- 每步写库前自动备份 report.db；全部完成后再向用户汇报第 4 节结果。

## 5. 汇报格式

向用户汇报：星级分布表（3★/2★/1★/0★/无鸟，2★ 含视频封面数单独注明）、
识出的鸟种及张数（标注罕见度 ○常见/◔能见/◑少见/●传奇）、视频处理情况
（封面数/有鸟数/定种数/无鸟数）、总耗时与速度、对账结论（无丢失）、
提醒复核两个点（低置信未定种的数量；可疑鸟种，如跨分布物种可能是误判）。
若走了第 4 节伽马重跑，追加：编辑张数、缩略图刷新、指标重算（None→有值
救回几张）、星级升降清单（逐张 新旧对比）、补种采纳清单（40-50% 段标注复核）。

## 6. 复核入口（给用户，不主动执行）

```bash
cd G:/code/SuperPicky
.venv/Scripts/python.exe -X utf8 spb_browse.py "<目录>"
```

浏览器里：双击缩略图直达多鸟编辑，左侧鸟种下拉框右键可整批改种/删种。
视频封面与照片操作完全一致（改星/改种/多鸟编辑/删除），详情面板「打开视频」
按钮可直接调系统播放器播放伴生视频。

## 7. 事后改星 / 补种（常见后续需求）

- **不要用 `superpicky_cli.py restar`**：V2 配额模式下已被代码级禁用
  （执行即拒绝；2026-08-31 事故，详见 dev-docs/reference/cli-reference.md）。
- **重定星用** `superpicky_cli.py rerate-v2 <目录> [--min-conf 0.4 --execute]`：
  不重跑检测/识鸟，从 report.db 重建 V2 定星输入；默认 dry-run，双自校验
  （复现存库评级、人工改星豁免保留 + 鸟种分组校验和）通过后写库，写前
  自动备份并同步 sidecar。改 2★ 名额用 `--quota2`。视频封面行
  （`*_vcover`）自动跳过——封面固定 2 星，改单个视频星级在浏览库手改。
  指标重算过的批次（如第 4 节伽马流程②）必须加 `--metrics-rebuilt`。
- **未定种补种用** `scripts_dev/backfill_species.py <目录> --threshold 40
  --execute`：仅对有鸟未定种的照片按新采纳门槛重识别，镜像写入
  photos / bird_detections / sidecar 三处，dry-run 默认、写前备份。
  低置信候选（40-50% 段）错误率偏高，采纳后建议人工过目。
