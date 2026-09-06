#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无鸟照片补救扫描（报告模式）：对目录内全部 has_bird=0 的照片，用修复后的
补救扫描（含全分辨率守门裁剪）重新判定，列出会被救回的照片。不写 report.db。

前置：先跑 process 生成 report.db；无鸟照片预览已被清理，本脚本会用
raw_to_jpeg 重新生成（写入 .superpicky/cache/temp_preview/，属应用自管缓存）。

用法:
    python scripts_dev/scan_no_bird_rescue.py <照片目录> [--accept-conf 0.5]
"""

import argparse
import os
import sqlite3
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.find_bird_util import raw_to_jpeg  # noqa: E402


def find_raw(directory: str, prefix: str) -> Optional[str]:
    """
    按前缀在目录里找 RAW 原文件。

    参数:
        directory (str): 照片目录
        prefix (str): 文件名前缀（无扩展名）

    返回:
        Optional[str]: RAW 路径；找不到返回 None
    """
    for ext in (".CR3", ".cr3", ".NEF", ".nef", ".ARW", ".arw",
                ".RAF", ".raf", ".ORF", ".orf", ".DNG", ".dng"):
        p = os.path.join(directory, prefix + ext)
        if os.path.exists(p):
            return p
    return None


def main() -> int:
    """
    扫描无鸟照片并报告可救回清单。

    返回:
        int: 0 成功
    """
    ap = argparse.ArgumentParser(description="无鸟照片补救扫描（报告模式）")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--accept-conf", type=float, default=0.4,
                    help="UI 置信度阈值（默认 0.4，与批量识别工作流定档一致）")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    db_path = os.path.join(root, ".superpicky", "report.db")
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT filename FROM photos WHERE has_bird=0 ORDER BY filename"
    ).fetchall()
    conn.close()
    print(f"📁 {root}  无鸟 {len(rows)} 张")

    from core.ai_model import preprocess_image, load_yolo_model, _rescue_scan
    model = load_yolo_model()

    rescued: List[tuple] = []
    for i, (prefix,) in enumerate(rows, 1):
        raw = find_raw(root, prefix)
        if raw is None:
            print(f"  [{i}/{len(rows)}] {prefix}: 找不到 RAW，跳过")
            continue
        preview = raw_to_jpeg(raw)
        if not preview or not os.path.exists(preview):
            print(f"  [{i}/{len(rows)}] {prefix}: 预览生成失败，跳过")
            continue
        img = preprocess_image(preview)
        r = _rescue_scan(model, img, args.accept_conf, 10, None, None,
                         image_path=preview)
        if r is not None:
            species = r.get("species") or "?"
            rescued.append((prefix, r["conf"], species, r.get("species_conf", 0.0)))
            print(f"  [{i}/{len(rows)}] {prefix}: 🐦 救回 {r['conf']:.2f} "
                  f"→ {species} {r.get('species_conf', 0.0):.0f}%")
        else:
            print(f"  [{i}/{len(rows)}] {prefix}: 仍无鸟")

    print(f"\n📊 可救回 {len(rescued)}/{len(rows)} 张:")
    for prefix, conf, species, sc in rescued:
        print(f"   {prefix}: YOLO {conf:.2f} → {species} {sc:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
