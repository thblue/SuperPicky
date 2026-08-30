# 中国口径稀有度 + 国家保护等级 / China-scoped Rarity & National Protection Level

> 一句话：**在中国拍的照片显示「中国境内有多难见」的分数**（而不是全球分数），
> 并给国家重点保护鸟类单独贴上「国家一级 / 国家二级」标签。两者相互独立，
> 保护等级不参与打分。
>
> In one sentence: photos taken in China now show a China-scoped rarity
> score instead of the global one, plus a separate Class I / Class II
> national-protection label. The label never affects the score.

深度细节（数据构建、校验、审计）见 [GBIF_RARITY_INDEX.md](GBIF_RARITY_INDEX.md) 第 3.6 / 3.7 节。

---

## 1. 效果对照 / Before & after

| 鸟种 | 全球分 | 中国分 | 保护等级 |
|---|---|---|---|
| 金雕 | 4.17（看似烂大街） | **34.43**（少见） | 国家一级 |
| 大天鹅 | 2.33 | 34.82 | 国家二级 |
| 白头鹎 | 0.48 | 0.48 | — |
| 麻雀 | 0.0 | 0.0（中国 98,006 条记录） | — |

五级分档沿用全球阈值：`[8, 25, 50, 75]` → 常见 / 能见 / 少见 / 罕见 / 传奇。

## 2. 两个新东西分别是什么 / The two additions

### 2.1 中国口径稀有度（`gbif_rarity_by_country` 表，countrycode='CN'）

- 数据源：GBIF Occurrence API `country=CN`，只统计 CC0 + CC-BY-4.0 记录，
  与全球分口径完全一致（快照 2026-08-29，1,412 个物种有 CN 分）。
- 算法：中国境内记录数取对数归一化到 0-100，分数越高越难见。
- **不设 IUCN 下限**（全球分有 CR/EN/VU 下限，中国分按你的决定不合并受胁状态）。
- 中国境内没有 GBIF 记录的物种（如澳洲特有鸟）不写行 → 自动回退全球分，
  与运行时回退语义一致。

### 2.2 国家保护等级（`china_protection` 表）

- 数据源：《国家重点保护野生动物名录》（2021 年第 3 号公告）鸟纲部分，
  经 zh.wikipedia 转载（CC BY-SA 4.0），构建时做了学名勘误与旗舰种断言校验。
- 规模：一级 89 种、二级 301 种，共 390 种。
- 展示：红色 `国家一级`、橙色 `国家二级`，与拍摄国家无关——
  在日本拍到黄胸鹀照样标一级（物种级属性）。

## 3. 照片按哪个国家算 / Country resolution

```
GPS 反解国家  →  用户手选国家（birdid_country_code 设置）  →  放弃，用全球分
```

- **没有 GPS 也绝不默认中国**：稀有度会写库持久化，猜错国家 = 伦敦/新加坡
  照片被永久误标成中国口径。无证据就不猜。
- 地理过滤（候选鸟种列表）无 GPS 时仍默认 CN——那是自纠错的启发式，
  选错了后面人工改种即可，与持久化的稀有度是两回事。

## 4. 在哪能看到 / Where to see it

| 位置 | 内容 |
|---|---|
| 详情面板 | 鸟种信息块新增「国家保护」行（罕见度行下方） |
| 裁剪工作室 / 多鸟编辑 | 名字旁保护等级小徽标 |
| XMP（EXIF 写入） | 稀有度 → `XMP:Event`；保护 → `XMP-iptcCore:SubjectCode`（值：国家一级保护动物 / 国家二级保护动物） |
| sidecar JSON | `gbif_rarity_100`、`china_protection_level`（顶层 + 逐检测） |
| report.db | `photos` / `bird_detections` 表各加 `china_protection_level INTEGER` 列（schema v13） |

## 5. 历史照片已经回填 / Historical data already backfilled

2026-08-30 已对 NAS 全部 **77 个**照片目录的 `.superpicky/report.db` 回填完成
（BirdIndex 清单 76 + keep_list 补充 1），无需重新跑识别：

- schema v12 → v13 自动迁移，与 App 内置迁移语义一致；
- 保护等级：photos +2,540 / detections +11,521；
- 中国口径稀有度：photos +11,327 / detections +36,060（识别为中国照片 24,572 张）；
- 海外目录（香港 / 澳门混拍 / 日本 / 新加坡 / 伦敦 / 东京）**只回填保护等级，
  稀有度一行未动**；
- 每个库改写前自动备份为 `report.db.bak-<时间戳>`，回滚 = 用备份覆盖回来；
- 脚本可重复执行（幂等），已回填过的库再跑增量约为 0。

注意：**sidecar JSON 和嵌入 XMP 是识别时导出的产物**，要在 App 里重新导出
才会带上新值；report.db 本身已就绪。

## 6. 已知边界 / Known limitations

- 维基名录存在个别学名笔误，构建脚本已勘误 3 处（如红隼）并输出审计 CSV
  （`scripts_dev/data_sources/china_protection_audit.csv`，90 行人工过目，1 项未解决）；
- 人工改过鸟种名的历史照片，保护等级按新中文名能连上就连，连不上保持空；
- 数据重建：`scripts_dev/build_china_rarity.py`（GBIF API 计数）与
  `scripts_dev/build_china_protection.py`（维基解析 + 校验）。
