# spb_* 运维工具速查 / Ops Tools Quick Reference

> 9 个 `spb_*` 入口 + 相关 CLI 的「什么时候用 / 风险 / 防护」速查表。
> 原理细节见各文件 docstring；架构背景见 [../ARCHITECTURE.md](../ARCHITECTURE.md)。
>
> Quick reference for the 9 spb_* ops entry points: when to use, risk, and guardrails.

## 处理新照片库（主链路） / Main pipeline

```bash
.venv/Scripts/python superpicky_cli.py process <照片目录>   # 检测→评分→定星→识鸟→整理
.venv/Scripts/python superpicky_cli.py info <照片目录>      # 查看结果摘要
.venv/Scripts/python spb_browse.py <照片目录>               # 打开结果浏览器
# 完整 runbook 见 PROCESS_QUICKSTART.md；命令细节见 reference/cli-reference.md
```

## 工具矩阵 / Tool Matrix

| 工具 | 用途 | 动照片文件？ | 动 DB/sidecar？ | 防护机制 |
|---|---|---|---|---|
| `spb_review.py` | 人工审核：原图叠加 bbox/物种/置信度出审阅图 | ❌ 只读（读 sidecar，降级读 DB） | ❌ | 输出到 `.superpicky/review/` |
| `spb_sync_edits.py` | 把 sidecar JSON 的人工编辑回放进 report.db | ❌ | ✅ 写 DB | 按编辑留痕回放 |
| `spb_fix_mainspecies.py` | 修历史 bug：sidecar main_species 与检测框不一致 | ❌ | ✅ 只改 JSON | 只改不一致条目 |
| `spb_wipe_ident.py` | 整目录清除误检识别结果 | ❌ | ✅ 检测框软删 + photos 归一无鸟态 | 软删可恢复（deleted=1） |
| `spb_rename_species.py` | 站内改种写后端（BirdIndex /api/fix 调用） | ❌ 永不移动照片（只清缓存预览） | ✅ DB + sidecar 同步 + 召回重算 | 默认 dry-run，`--apply` 才落盘；**路径与 CLI 面冻结** |
| `spb_flatten.py` | 把按鸟种/星级整理的库还原为扁平结构 | ⚠️ **移动照片**回原位 | ✅ 回写 DB + 重导 sidecar | dry-run 默认；写撤销清单 `flatten_undo_*.json` |
| `spb_dedupe_jpg.py` | 删 RAW+JPG 双拍摄的冗余同名 JPG | ⚠️ **删除 JPG** | ❌ | 交互确认 + RAW 在位复查 |
| `spb_pixel_art.py` | 像素风鸟图生成（BirdIndex sprite 调用） | ❌ 只读源图 | ❌ | 输出全走 `--out` 指定目录；**路径与 CLI 面冻结** |
| `spb_browse.py` | 独立结果浏览器（SPBBrowse.exe 源码） | ❌ | ❌ | — |

## 黄金法则 / Golden Rules

1. **先 dry-run**：所有带 `--apply` 的工具，先不带参数看一遍将要发生什么。
2. **先沙箱**：没把握时把整个库复制到 `scripts_dev/_backfill_sandbox/` 演练一遍。
3. **NAS 断连保护**：批处理前确认网络盘稳定；BirdIndex 扫描器自带断网不清库保护，
   但 SuperPicky 侧的移动/删除没有——操作中途断连立即停下检查。
4. **动过的库跑一遍 `spb_review.py`** 出审阅图抽查，再去 BirdIndex 上看统计是否合理。

## 审核工作流（历史惯例） / Audit workflow

根目录跑批产生的审核清单曾用 `keep_list.txt`（NAS 目录清单）配合 `spb_review.py`
逐库人工审核；相关清单与产物现归档于仓库根 `archive/`（不入库）。
