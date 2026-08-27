# 新目录处理速查 / New Directory Quickstart

> 目标：拿到一个新拍的鸟片目录，**一条命令跑完「检测 → 评分 → 定星 → 识鸟 → 产出浏览库」**，不用再翻代码。
> 最后核对：2026-08-27，基于 `superpicky_cli.py`（CLI v4.2.0）。

## 一条命令

```bash
cd G:/code/SuperPicky
.venv/Scripts/python.exe -X utf8 superpicky_cli.py process -i "//NAS-server/PHOTO/观鸟/2026 观鸟/XXXX"
```

> **`-i` 必须显式加**（`--auto-identify`）：CLI 的识鸟总开关**只认这个参数，不回落配置文件**
> （`tools/cli_settings.py` 的 `auto_identify=bool(_arg(...) or False)`）。主鸟识别、
> 多鸟逐鸟分类、鸟种分目录、BirdID 地理过滤都挂在这个开关后面
> （`core/photo_processor.py` 的识鸟执行器与提交门控）。漏掉时不报错——
> 评分照常跑，但鸟种**沿用库里旧数据**，极易误以为识别过了。
> 注意：**物种召回**（批末 notable 标记，`core/species_recall.py`）与**补救扫描里的
> 识鸟确认**（`ai_model.py` 的 `_birdid_confirm`）不受此开关门控——漏 `-i` 它们照跑，
> 但召回消费的是库内旧分类数据。

跑完用浏览器人工复核/改种。两种方式：

```bash
# 方式一：CLI 直开（目录本身或其子目录含处理结果即可；传父目录自动合并多个子目录）
.venv/Scripts/python.exe -X utf8 spb_browse.py "<同目录>"

# 方式二：SPBBrowse.exe（双击，无参数时弹「打开照片目录」启动器）
#   最近目录 + 浏览按钮，与主程序共用 advanced_config.json 的最近目录历史
dist_SPBBrowse\SPBBrowse\SPBBrowse.exe
```

浏览器「文件」菜单里有 **打开目录（Ctrl+O）**、**已处理目录** 和 **最近目录** 子菜单，浏览中随时切换目录，不必退出重开。缩略图键位：**双击 = 直达多鸟编辑**（无检测框的照片回退全屏）、**Enter = 全屏浏览**（默认全图，F 切裁切诊断）、数字键 0-3 打星。左侧「鸟种」下拉框**右键**当前选中的鸟种可**整批删除**（含主鸟框，照片回无鸟种，星级不变）或**整批改为其他鸟种**（弹搜索框选新种，全目录含主鸟一并改写）——AI 整批识别错时不用一张张改。

> **已处理目录清单**：跑过 `process` 的目录会自动记入 `advanced_config.json` 的
> `processed_directories`（`PhotoProcessor.process` 收口，CLI/GUI 都记，最近处理的排最上）。
> SPBBrowse.exe 启动器直接列出整个清单，双击即开，不用翻 NAS。
> **存量目录一次性导入**：启动器里点「扫描导入…」，选 NAS 观鸟根目录
> （如 `\\NAS-server\PHOTO\观鸟`），自动发现所有含 `report.db` 的子目录灌入清单。
> exe 打包配置在 `spb_browse_win.spec`，构建：`build_spb_browse.bat`（排除 torch/模型/exiftool 二进制，体积远小于主程序；打星遵守 `metadata_write_mode`——`none` 时只进 report.db 不写照片 XMP，鸟种搜索所需的 ioc 库保留）。

实测参考（RTX 3090，NAS 目录）：2881 张约 64 分钟（≈1.3 秒/张），CPU/GPU 满载阶段在前 50 分钟。

## 产物去向（不动原片，除非布局配置要求移动）

| 位置 | 内容 |
|---|---|
| `<目录>/.superpicky/report.db` | 全部结果：评分、鸟种、检测框、召回标记（sqlite） |
| `<目录>/.superpicky/meta/<前缀>.json` | 每张照片的 sidecar JSON（BirdIndex 网站的数据源） |
| `<目录>/.superpicky/cache/temp_preview/` | RAW 的临时预览 JPG（供浏览/编辑用，配置了保留） |
| `<目录>/superpicky.log` | 运行日志（追加式，注意区分新旧段落） |

当前配置 `metadata_write_mode=none`：**不写 EXIF/XMP、不动文件位置**（flat 布局），结果全部落在 report.db + sidecar。星级和鸟种只在浏览器/BirdIndex 里看。

## 默认参数从哪来（重要机制）

命令行**不带参数时，默认值 = GUI 高级设置**，同一个文件：

```
C:\Users\<用户>\AppData\Local\SuperPicky\advanced_config.json
```

优先级：**CLI 显式参数 > advanced_config.json > 内置预设**。改 GUI 里的设置会改这个文件，下次 CLI 跑就跟着变；反之 CLI 临时参数只影响当次，不回写。

### 本机当前生效值（2026-08-27 快照，改动请以配置文件为准）

| 项 | 当前值 | 配置键 |
|---|---|---|
| 锐度阈值 | 380 | `custom_sharpness`（skill_level=custom） |
| 美学阈值 TOPIQ | 4.8 | `custom_aesthetics` |
| AI 置信度 | 50% | `min_confidence: 0.5` |
| 飞鸟检测 | 开 | `flight_check` |
| 曝光检测 | 关 | `exposure_check` |
| 连拍检测 | 关 | `burst_check` |
| 定星算法 | V2 配额制（3★ 20%、2★ 40%） | `rating_algorithm` / `custom_quota3/2` |
| 多鸟分类 | 开（面积≥0.1%、采纳≥35%） | `multibird_enabled` 等 |
| **识鸟总开关（CLI）** | **仅 `-i` 参数，无配置回落** | `auto_identify`（settings 级） |
| 补救扫描（小图漏检重扫） | 开 | `rescue_scan_enabled` |
| 目录布局 | flat（不移动文件） | `folder_layout` |
| 元数据写入 | none（不写 EXIF） | `metadata_write_mode` |

### 常用临时覆盖（只影响当次运行）

```bash
# 换阈值 / 开关检测
... process "<目录>" -s 500 -n 5.2          # 锐度、美学
... process "<目录>" --burst --exposure     # 打开连拍、曝光检测
... process "<目录>" --no-flight            # 关飞鸟检测

# 布局 / 写入模式
... process "<目录>" --metadata-mode sidecar   # 临时写 XMP 侧车
... process "<目录>" --folder-layout rating-first

# 识鸟地理过滤（默认开；海外拍摄时指定国家更准）
# 注意：不传 --birdid-country 时真实链路是「GPS反查国 → 兜底硬编码CN」，
# 不读 advanced_config 里的 birdid_country_code；国内目录建议显式传 CN 最稳
... process "<目录>" --birdid-country AU
```

完整参数表：`superpicky_cli.py process --help`；各参数语义详见 [cli-reference.md](cli-reference.md)（默认值同为「跟随配置」机制）。

## 配套命令

| 命令 | 用途 |
|---|---|
| `spb_browse.py <目录>` | 结果浏览器（缩略图 + 详情 + 多鸟编辑右键；无参数弹启动器） |
| `build_spb_browse.bat` | 打包 SPBBrowse.exe（独立结果浏览器，`dist_SPBBrowse\SPBBrowse\`） |
| `superpicky_cli.py restar <目录> -s 500 -n 5.5` | 只重新评星（不重跑检测） |
| `superpicky_cli.py reset <目录> -y` | 重置目录（移回文件、清评分，**破坏性，先想清楚**） |
| `superpicky_cli.py info <目录>` | 查看目录处理状态 |
| `spb_flatten.py <目录> --execute` | 把历史「鸟种/星级」目录结构摊平回原位（先 dry-run） |
| `spb_sync_edits.py <目录>` | 浏览器 sidecar 里的人工编辑（改主鸟/删框/改种）回放进 report.db，幂等 |

## 已知坑

- **忘加 `-i` = 静默跳过识鸟**：不报错、评分照跑，鸟种沿用库里旧数据（photos 主鸟种 / bird_detections 旧分类行原样保留，不会丢，但也不更新）。判断方法：日志里搜 `Multi-bird` / `Low confidence`，有才是真跑了识鸟。
- **NAS 目录偶发 WinError 5**：断点文件 `resume_state.json` 在 SMB 上每张重命名一次，NAS 索引/杀毒短暂锁文件会导致个别照片被跳过（8/24 跑丢过 42 张，概率 ≈1.5%）。跑完看日志末尾「N 张照片处理异常被跳过」汇总；被跳过的保留上次结果。2026-08-27 起 `tools/resume_state.py` 写入端已加退避重试+静默降级（重试耗尽只告警一次，不再把当张照片连坐成失败），「跑完核对 report.db 行数 vs 文件数」的习惯仍保留。
- **重跑前备份 report.db**：重跑会按新结果覆盖库内计算字段。惯例是在 `.superpicky/` 里留 `report.db.bak_<原因>_<时间戳>`。
- 跑之前确认没有别的进程在写同一目录（浏览器开着编辑时不要重跑）。
