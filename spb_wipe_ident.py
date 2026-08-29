#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_wipe_ident — 整目录清除鸟识别结果（人工审核判定「全是误检」时用）

场景：timeMachine 这类老归档里，大量目录的「鸟」是 AI 误检（游客照、
雕像、杂物被当成鸟）。人工在结果浏览器里逐目录审核后，对确认整目录
无真鸟的，本工具把识别结果一键清空，目录归一为「无鸟目录」：

- bird_detections 全部软删（deleted=1，行保留可追溯，原照片不动）；
- photos 有鸟行归一无鸟态：has_bird=0、rating=-1（无鸟）、主鸟种/
  鸟id置信度/召回标记/精选旗标/飞鸟标记/稀有度字段全部清空；
- sidecar 增量重导出：无鸟照片的历史 JSON 幂等删除（BirdIndex 数据
  源只留有鸟照片）；
- 被清照片的临时预览（.superpicky/cache/temp_preview/*.jpg）一并清理。

默认 dry-run 只统计不落盘，加 --apply 才真正执行。软删的检测行理
论上可手工恢复（deleted=0），但产品层面请当作不可逆操作对待。

用法:
    python spb_wipe_ident.py <目录> [目录2 ...] [--apply]
    python spb_wipe_ident.py -f 清单.txt [--apply]     # 每行一个目录

Wipe every bird identification in whole directories (for dirs whose
"birds" were all false positives). Dry-run by default; --apply executes.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# 允许直接 `python spb_wipe_ident.py` 运行（仓库根即 CWD）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.report_db import ReportDB


def _snapshot(db: ReportDB, directory: str) -> dict:
    """
    取目录清理前的状态快照（dry-run 与执行前报告共用）。

    参数:
    db (ReportDB): 已打开的库
    directory (str): 照片目录

    返回:
    dict: 照片/有鸟/鸟种数、未删检测数、meta JSON 数、预览文件数

    Pre-wipe snapshot for reporting (photos, birds, species, live
    detections, sidecar JSONs, preview files).
    """
    total, has_bird = db._conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(has_bird), 0) FROM photos").fetchone()
    species = db._conn.execute(
        "SELECT COUNT(DISTINCT bird_species_cn) FROM photos "
        "WHERE has_bird = 1 AND bird_species_cn IS NOT NULL "
        "AND bird_species_cn != ''").fetchone()[0]
    dets = db._conn.execute(
        "SELECT COUNT(*) FROM bird_detections WHERE deleted = 0").fetchone()[0]
    meta_dir = os.path.join(directory, '.superpicky', 'meta')
    jsons = (len([f for f in os.listdir(meta_dir) if f.endswith('.json')])
             if os.path.isdir(meta_dir) else 0)
    return {"total": total, "has_bird": has_bird, "species": species,
            "detections": dets, "jsons": jsons}


def _wipe_directory(directory: str, apply: bool) -> bool:
    """
    清理单个目录（dry-run 或 --apply）。

    参数:
    directory (str): 照片目录（含 .superpicky/report.db）
    apply (bool): True 执行落盘，False 仅统计

    返回:
    bool: 成功（或 dry-run 完成）返回 True
    """
    db_path = os.path.join(directory, '.superpicky', 'report.db')
    if not os.path.isfile(db_path):
        print(f"  ⚠️ 跳过（无 report.db）: {directory}")
        return False
    db = ReportDB(directory)
    try:
        before = _snapshot(db, directory)
        print(f"  清理前: 照片 {before['total']} | 有鸟 {before['has_bird']} | "
              f"鸟种 {before['species']} | 未删检测 {before['detections']} | "
              f"meta JSON {before['jsons']}")
        if not apply:
            print("  [dry-run] 以上识别将被清空（加 --apply 执行）")
            return True

        result = db.wipe_all_identifications()

        # 1) 删除被清照片的临时预览。temp_jpeg_path 对纯 JPG / RAW+JPG
        #    配对照片指向原片或伴随 JPG（core/photo_processor.py:2097-
        #    2104），绝对不能删——只删 .superpicky/cache/ 下的生成预览，
        #    其余仅清空 DB 字段。这是防误删原片的硬闸门。
        #    Delete only generated previews under .superpicky/cache/; for
        #    native-JPG photos temp_jpeg_path IS the original file.
        cache_root = os.path.normpath(
            os.path.join('.superpicky', 'cache'))
        removed_previews = 0
        for rel in result["preview_paths"]:
            norm = os.path.normpath(rel)
            if not norm.startswith(cache_root + os.sep):
                continue
            abs_path = os.path.join(directory, norm)
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    removed_previews += 1
            except OSError as e:
                print(f"  ⚠️ 预览删除失败 {rel}: {e}")

        # 2) sidecar 增量重导出（无鸟 JSON 幂等删除）
        from core.sidecar_export import export_directory_sidecars
        export_directory_sidecars(db, directory, log=lambda *_: None)

        after = _snapshot(db, directory)
        print(f"  ✅ 已清除: 检测 {result['detections']} 行 | 照片 "
              f"{result['photos']} 张归一无鸟 | 预览删 {removed_previews} 个")
        print(f"  清理后: 有鸟 {after['has_bird']} | 未删检测 "
              f"{after['detections']} | meta JSON {after['jsons']}")
        return True
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="整目录清除鸟识别结果（默认 dry-run，--apply 执行）")
    parser.add_argument('directories', nargs='*', help='照片目录（可多个）')
    parser.add_argument('-f', '--file', help='目录清单文件（每行一个目录）')
    parser.add_argument('--apply', action='store_true',
                        help='真正执行（默认只统计）')
    args = parser.parse_args()

    dirs: List[str] = list(args.directories)
    if args.file:
        with open(args.file, 'r', encoding='utf-8-sig') as f:
            dirs.extend(line.strip() for line in f if line.strip())
    if not dirs:
        parser.print_help()
        return 1

    mode = "执行" if args.apply else "dry-run（只统计，不改任何文件）"
    print(f"=== spb_wipe_ident: {len(dirs)} 个目录, 模式: {mode} ===")
    ok = 0
    for d in dirs:
        print(f"[{ok + 1}/{len(dirs)}] {d}")
        if _wipe_directory(d, args.apply):
            ok += 1
    print(f"=== 完成: {ok}/{len(dirs)} 个目录"
          + ("（已落盘）" if args.apply else "（未落盘，加 --apply 执行）")
          + " ===")
    return 0 if ok == len(dirs) else 1


if __name__ == '__main__':
    sys.exit(main())
