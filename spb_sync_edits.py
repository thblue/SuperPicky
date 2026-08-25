#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_sync_edits — 把 sidecar JSON 里的人工编辑一次性回放进 report.db

多鸟编辑器的人工编辑持久层是 sidecar JSON（`.superpicky/meta/*.json`）。
V5.4 之前编辑器只写 JSON 不写库；重跑批处理也会用 AI 结果重建
bird_detections。本工具扫描目录内全部 sidecar，把人工结果同步回
report.db，让结果浏览器的筛选/召回/标题立即生效，等价于把每张照片
在编辑器里重新点一次保存：

- main_species（主鸟多选）→ is_selected + photos 表第一主鸟；
- deleted 检测框 → 软删除（deleted=1，只隐藏不物理删行）；
- edited 检测框的人工物种 → species 三名 + edited=1。

回放完成后重算物种召回并增量重导出 sidecar（幂等，可重复执行）。

用法:
    python spb_sync_edits.py <照片目录>

Replay manual edits (multi-main selection, box deletions, species
renames) from sidecar JSONs back into report.db, then re-run the
species recall and an incremental sidecar re-export. Idempotent.
"""

from __future__ import annotations

import json
import os
import sys


def _replay_one(report_db, prefix: str, data: dict) -> dict:
    """
    回放一张照片的人工编辑（假定调用方已确认 JSON 含人工痕迹）。

    参数:
    report_db: 已打开的 ReportDB
    prefix (str): 照片前缀（无扩展名）
    data (dict): sidecar JSON 全文

    返回:
    dict: {'main': bool, 'deleted': int, 'renamed': int}

    Replay one photo's manual edits into the DB.
    """
    stats = {"main": False, "deleted": 0, "renamed": 0}

    # 1) 主鸟多选（bird_index 列表）+ photos 表第一主鸟
    main_species = data.get("main_species") or []
    indexes = [m.get("bird_index") for m in main_species
               if isinstance(m, dict) and m.get("bird_index") is not None]
    if indexes:
        report_db.update_detection_selection(prefix, indexes)
        first = main_species[0]
        if first.get("cn") or first.get("en"):
            report_db.update_photo(prefix, {
                "bird_species_cn": first.get("cn") or None,
                "bird_species_en": first.get("en") or None,
            })
        stats["main"] = True

    # 2) 逐框：软删除 + 人工改种
    deleted_idx, renamed = [], 0
    for det in data.get("detections") or []:
        if not isinstance(det, dict) or det.get("index") is None:
            continue
        if det.get("deleted"):
            deleted_idx.append(det["index"])
        elif det.get("edited"):
            species = det.get("species") or {}
            report_db.update_detection_species(
                prefix, det["index"],
                species.get("cn"), species.get("en"),
                species.get("scientific") or None)
            renamed += 1
    if deleted_idx:
        report_db.soft_delete_detections(prefix, deleted_idx)
        stats["deleted"] = len(deleted_idx)
    stats["renamed"] = renamed
    return stats


def sync_manual_edits(report_db, directory: str,
                      log=print) -> dict:
    """
    扫描目录全部 sidecar JSON，回放人工编辑进 report.db，随后重算
    物种召回 + 增量重导出 sidecar（幂等）。

    判定「含人工痕迹」：JSON 有 edits 数组或 main_species 字段。

    参数:
    report_db: 已打开的 ReportDB
    directory (str): 照片目录（sidecar 在 .superpicky/meta/）
    log: 日志函数

    返回:
    dict: {'photos': 回放照片数, 'main': 主鸟变更照片数,
           'deleted': 软删框数, 'renamed': 改种框数,
           'recall_photos': 重算后召回照片数, 'exported': 重导出 JSON 数}

    Replay every manually-edited sidecar into the DB, then rebuild
    recall marks and re-export changed sidecars.
    """
    from core.sidecar_export import export_directory_sidecars
    from core.species_recall import run_species_recall

    meta_dir = os.path.join(directory, ".superpicky", "meta")
    totals = {"photos": 0, "main": 0, "deleted": 0, "renamed": 0}
    replayed: list = []
    for name in sorted(os.listdir(meta_dir)) if os.path.isdir(meta_dir) else []:
        if not name.endswith(".json"):
            continue
        prefix = name[:-5]
        path = os.path.join(meta_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if not (data.get("edits") or data.get("main_species")):
            continue
        stats = _replay_one(report_db, prefix, data)
        totals["photos"] += 1
        totals["main"] += 1 if stats["main"] else 0
        totals["deleted"] += stats["deleted"]
        totals["renamed"] += stats["renamed"]
        replayed.append((prefix, stats))

    for prefix, stats in replayed[:10]:
        log(f"  ↩️ {prefix}: 主鸟={stats['main']} "
            f"删框={stats['deleted']} 改种={stats['renamed']}")
    if len(replayed) > 10:
        log(f"  …（共 {len(replayed)} 张）")

    log(f"↩️ 人工编辑回放: {totals['photos']} 张照片 "
        f"（主鸟 {totals['main']} / 删框 {totals['deleted']} / "
        f"改种 {totals['renamed']}）")

    # 重算召回 + 增量重导出（与编辑器保存后的链路一致）
    from advanced_config import get_advanced_config
    threshold = get_advanced_config().recall_species_threshold
    recall = run_species_recall(report_db, species_threshold=threshold,
                                log=log)
    written = export_directory_sidecars(report_db, directory, log=log)
    log(f"📦 sidecar 增量重导出: {written} 个")
    totals["recall_photos"] = recall.get("flagged_photos", 0)
    totals["exported"] = written
    return totals


def main(argv=None) -> int:
    """命令行入口 / CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法 / usage: python spb_sync_edits.py <照片目录>")
        return 1
    directory = argv[0]
    if not os.path.isdir(os.path.join(directory, ".superpicky")):
        print(f"未找到处理结果 (.superpicky)，请先跑 process / "
              f"not a processed directory: {directory}")
        return 1

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from tools.report_db import ReportDB
    db = ReportDB(directory)
    try:
        sync_manual_edits(db, directory)
    finally:
        db.close()
    print("✅ 完成 / done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
