# -*- coding: utf-8 -*-
"""
批量回填所有历史照片目录的 report.db（中国稀有度 + 国家保护等级）。

目录清单来源 / Directory sources:
  1. G:/code/BirdIndex/config.json 的 photo_roots（权威清单，自带 province）
  2. keep_list.txt（timeMachine/60d 旧目录补充，大部分与上面重合）

国家分类 / Country classification:
  海外/混拍目录（香港、澳门、台湾、日本、新加坡、英国、伦敦、东京）不传
  --country —— 只靠 GPS 反解，无 GPS 照片保持全球分；其余中国目录传
  --country CN 兜底无 GPS 照片（GPS 反解永远优先）。
  Overseas/mixed directories get no --country fallback (GPS only); China
  directories pass --country CN for GPS-less photos (GPS always wins).

安全 / Safety:
  默认 dry-run（零写入，schema 迁移也回滚）；--apply 才真正落盘，且每个库
  先做一致性备份 report.db.bak-<时间戳>。单库失败不影响其余目录。
  Dry-run by default (zero writes incl. the schema migration); --apply
  commits with a per-DB snapshot backup. One failure never stops the rest.

用法 / Usage:
    .venv/Scripts/python scripts_dev/run_backfill_all.py            # 全量 dry-run
    .venv/Scripts/python scripts_dev/run_backfill_all.py --apply    # 正式跑
    .venv/Scripts/python scripts_dev/run_backfill_all.py --limit 3  # 试前 3 个
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from scripts_dev.backfill_china_fields import backfill  # noqa: E402

BIRDINDEX_CONFIG = "G:/code/BirdIndex/config.json"
KEEP_LIST = os.path.join(PROJ, "keep_list.txt")

# 海外/混拍判定：province 命中，或目录名含境外关键词（珠海澳门→混拍）
# Overseas/mixed detection: province match or keyword in the name.
OVERSEAS_PROVINCES = {"香港", "澳门", "台湾", "日本", "英国", "新加坡"}
OVERSEAS_KEYWORDS = ("香港", "澳门", "台湾", "日本", "新加坡", "伦敦", "东京", "英国")


def _norm(path: str) -> str:
    """路径归一化（小写 + 正斜杠）用于去重。"""
    return path.replace("\\", "/").lower().rstrip("/")


def is_overseas(name: str, province: str, path: str) -> bool:
    """判定目录是否海外/混拍（不传 --country 兜底）。"""
    if province in OVERSEAS_PROVINCES:
        return True
    text = f"{name} {path}"
    return any(k in text for k in OVERSEAS_KEYWORDS)


def collect_dirs() -> List[Tuple[str, str, bool]]:
    """
    汇总去重后的 (目录路径, 显示名, 是否海外) 清单。

    Returns:
        list[tuple[str, str, bool]]: BirdIndex roots 在前，keep_list 补充
        在后；按归一化路径去重。
    """
    out: List[Tuple[str, str, bool]] = []
    seen = set()
    with open(BIRDINDEX_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for r in cfg.get("photo_roots", []):
        key = _norm(r["path"])
        if key in seen:
            continue
        seen.add(key)
        out.append((r["path"], r.get("name", os.path.basename(r["path"])),
                    is_overseas(r.get("name", ""), r.get("province", ""),
                                r["path"])))
    if os.path.exists(KEEP_LIST):
        with open(KEEP_LIST, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                key = _norm(p)
                if key in seen:
                    continue
                seen.add(key)
                name = os.path.basename(p.replace("\\", "/"))
                out.append((p, name, is_overseas(name, "", p)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch backfill CN rarity + protection into all report.db")
    p.add_argument("--apply", action="store_true",
                   help="正式写入（默认 dry-run 零写入）/ write for real")
    p.add_argument("--limit", type=int, default=None,
                   help="只处理前 N 个目录 / process only the first N dirs")
    a = p.parse_args()

    dirs = collect_dirs()
    if a.limit:
        dirs = dirs[:a.limit]
    print(f"[batch] 目标目录 {len(dirs)} 个，模式 / mode: "
          f"{'APPLY（写入）' if a.apply else 'DRY-RUN（零写入）'}\n")

    header = (f"{'目录 / dir':<34} {'cc':<4} {'prot+p':>7} {'prot+d':>7} "
              f"{'cn':>5} {'rar+p':>6} {'rar+d':>6}")
    print(header)
    print("-" * len(header))
    totals: Dict[str, int] = {}
    n_skip = n_err = 0
    for path, name, overseas in dirs:
        db = os.path.join(path, ".superpicky", "report.db")
        if not os.path.exists(db):
            n_skip += 1
            print(f"{name:<34} {'--':<4} {'SKIP（无 report.db）':>20}")
            continue
        country: Optional[str] = None if overseas else "CN"
        try:
            # 静默 backfill 自身的逐项输出，报表只留一行汇总
            with contextlib.redirect_stdout(io.StringIO()):
                stats = backfill(db, dry_run=not a.apply, do_rarity=True,
                                 default_country=country)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v
            print(f"{name:<34} {country or 'GPS':<4} "
                  f"{stats['prot_photos']:>7} {stats['prot_detections']:>7} "
                  f"{stats['cn_photos']:>5} {stats['rarity_photos']:>6} "
                  f"{stats['rarity_detections']:>6}")
        except Exception as e:
            n_err += 1
            print(f"{name:<34} {country or 'GPS':<4} "
                  f"ERROR: {type(e).__name__}: {e}")

    print("-" * len(header))
    print(f"[batch] 完成 / done: 目录 {len(dirs)}，跳过 {n_skip}，失败 {n_err}")
    print(f"[batch] 合计 / totals: 迁移 {totals.get('migrated', 0)} 库，"
          f"保护等级 photos +{totals.get('prot_photos', 0)} / "
          f"detections +{totals.get('prot_detections', 0)}，"
          f"CN 照片 {totals.get('cn_photos', 0)} 张，"
          f"稀有度 photos +{totals.get('rarity_photos', 0)} / "
          f"detections +{totals.get('rarity_detections', 0)}")
    if not a.apply:
        print("[batch] 以上为 dry-run 统计，未写入任何更改；加 --apply 正式执行")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
