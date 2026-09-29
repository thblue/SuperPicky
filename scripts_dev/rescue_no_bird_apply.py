#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无鸟照片救回写库（scan_no_bird_rescue 的执行伴侣）。

对 scan 报告确认的救回清单，重跑 1024 整图补救 + 常规瓦片检测
（V5.8，门槛与批量一致）拿框与识别结果并写库：
- photos: has_bird=1、confidence=主框置信度、rating=0（保守，浏览库人工调）；
- EXIF 元数据回填：门控拦截的照片从未提取 EXIF，救回时从原片补读
  拍摄时间/机身/ISO 等标准字段（仅填 NULL 列，见
  backfill_rescued_exif.py），否则浏览库 sidebar 与 sidecar 无日期；
- bird_detections: 全部过门槛的框逐行入库（置信度最高者 is_selected=1，
  其余为次要框 is_selected=0）+ 守门识别结果照实入库；
- 低于采纳线（--threshold）的种不写 photos 主鸟种列，与管线口径一致；
- 人工清理保护：浏览库里删过鸟的照片（detections.deleted=1 或
  has_bird=0 且 rating=0）不再自动救回；
- sidecar 交 reexport_sidecars 补。

适用约束：V1 老批次库（不能全量 process 重跑覆盖人工编辑）、救回量小。
写前自动备份 report.db。单一写者原则：零接触照片文件。

用法:
    python scripts_dev/rescue_no_bird_apply.py <照片目录> --files IMG_9392,IMG_9393 [--threshold 40] [--execute]
"""

import argparse
import os
import shutil
import sqlite3
import sys
import time
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.find_bird_util import raw_to_jpeg  # noqa: E402
from scripts_dev.backfill_species import find_raw, find_plain_image, read_bgr  # noqa: E402
from scripts_dev.backfill_rescued_exif import (  # noqa: E402
    backfill_photos_exif, resolve_exiftool)


def load_fullres(prefix: str, root: str) -> Tuple[Optional[str],
                                                   Optional[np.ndarray],
                                                   Optional[str]]:
    """
    取全分辨率预览（缓存优先，缺失时从 RAW/JPEG 本体重生成）。

    返回:
        (预览路径, BGR 图, 1024 处理图尺寸描述用占位) 之三，失败 (None, None, None)
    """
    cache = os.path.join(root, '.superpicky', 'cache',
                         'temp_preview', prefix + '.jpg')
    if os.path.exists(cache):
        img = read_bgr(cache)
        if img is not None:
            return cache, img, cache
    raw = find_raw(root, prefix)
    if raw is not None:
        preview = raw_to_jpeg(raw)
        if preview and os.path.exists(preview):
            img = read_bgr(preview)
            if img is not None:
                return preview, img, preview
    plain = find_plain_image(root, prefix)
    if plain is not None:
        img = read_bgr(plain)
        if img is not None:
            return plain, img, plain
    return None, None, None


def main() -> int:
    """
    主流程：逐张重跑补救扫描 → 备份 → 写 photos/detections。

    返回:
        int: 0 成功；1 输入错误
    """
    ap = argparse.ArgumentParser(description="无鸟照片救回写库")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--files", required=True,
                    help="救回文件前缀（逗号分隔，来自 scan 报告）")
    ap.add_argument("--threshold", type=float, default=40.0,
                    help="主鸟种采纳线（百分比，默认 40；低于不写 photos 种列）")
    ap.add_argument("--accept-conf", type=float, default=0.4,
                    help="补救扫描直接救回线（与跑批 -c 一致）")
    ap.add_argument("--execute", action="store_true",
                    help="写库（默认 dry-run；写前自动备份）")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    db_path = os.path.join(root, ".superpicky", "report.db")
    prefixes = [p.strip() for p in args.files.split(",") if p.strip()]
    if not prefixes:
        print("未指定救回清单")
        return 1

    from core.ai_model import (preprocess_image, load_yolo_model,
                               _rescue_scan, _tile_detect_pass)
    model = load_yolo_model()
    # V5.8: 守门门槛对齐批量口径（advanced_config，当前定档 25）
    from advanced_config import get_advanced_config
    gate = get_advanced_config().rescue_birdid_gate

    rescued = []
    for i, prefix in enumerate(prefixes, 1):
        preview, img, _ = load_fullres(prefix, root)
        if img is None:
            print(f"  [{i}/{len(prefixes)}] {prefix}: 预览不可得，跳过")
            continue
        proc = preprocess_image(preview)
        # V5.8: 1024 整图补救 + 常规瓦片检测都跑，全部过门槛的框一起
        # 收集入库（瓦片框已对整图补救框做包含去重，不会重复写同一只鸟）。
        # Both the 1024 rescan and the universal tile pass contribute;
        # every gate-passing box is written for the manual review pass.
        r = _rescue_scan(model, proc, args.accept_conf, gate, None, None,
                         image_path=preview)
        existing = [r["xyxy"]] if r is not None else None
        tiles = _tile_detect_pass(model, preview, proc, existing, gate,
                                  None, None)
        if r is None and not tiles:
            print(f"  [{i}/{len(prefixes)}] {prefix}: 本轮扫描未救回（漂移），跳过")
            continue
        ph, pw = proc.shape[:2]
        h, w = img.shape[:2]
        sx, sy = w / pw, h / ph
        entries = []
        if r is not None:
            entries.append({"conf": float(r["conf"]),
                            "species": r.get("species") or "",
                            "species_conf": float(r.get("species_conf") or 0.0),
                            "xyxy": r["xyxy"]})
        for t in tiles:
            entries.append({"conf": float(t["conf"]),
                            "species": t["species"] or "",
                            "species_conf": float(t["species_conf"]),
                            "xyxy": t["xyxy"]})
        entries.sort(key=lambda e: -e["conf"])
        boxes = []
        for e in entries:
            x1, y1, x2, y2 = e["xyxy"]
            boxes.append({"conf": e["conf"], "species": e["species"],
                          "species_conf": e["species_conf"],
                          "box": (float(x1 * sx), float(y1 * sy),
                                  float((x2 - x1) * sx),
                                  float((y2 - y1) * sy))})
        best = boxes[0]
        rescued.append({"prefix": prefix, "conf": best["conf"],
                        "box": best["box"], "species": best["species"],
                        "species_conf": best["species_conf"],
                        "extra_boxes": boxes[1:]})
        name = f"{best['species']} {best['species_conf']:.0f}%" \
            if best["species"] else "（分类无结果）"
        extra = f"（另 {len(boxes) - 1} 框）" if len(boxes) > 1 else ""
        print(f"  [{i}/{len(prefixes)}] {prefix}: conf={best['conf']:.2f} "
              f"{name} 框 {best['box'][2]:.0f}x{best['box'][3]:.0f}px{extra}")

    if not rescued:
        print("无救回结果")
        return 0

    # V5.8: 人工清理保护——用户在浏览库删过鸟的照片不再自动救回：
    # 检测行有 deleted=1、或 photos 已是 has_bird=0 且 rating=0
    # （批量无鸟是 -1，0 只能是人工处理过的痕迹）。
    # V5.8: never re-rescue a photo the user manually stripped of birds.
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        stripped = []
        for r in rescued:
            n_del = cur.execute(
                "SELECT COUNT(*) FROM bird_detections "
                "WHERE filename=? AND deleted=1", (r["prefix"],)).fetchone()[0]
            ph = cur.execute(
                "SELECT has_bird, rating FROM photos WHERE filename=?",
                (r["prefix"],)).fetchone()
            if n_del or (ph and ph[0] == 0 and ph[1] == 0):
                stripped.append(r["prefix"])
        if stripped:
            rescued = [r for r in rescued if r["prefix"] not in stripped]
            print(f"🛡️ 人工清理保护，跳过 {len(stripped)} 张: "
                  f"{','.join(stripped)}")
    finally:
        conn.close()

    if not rescued:
        print("全部命中人工清理保护，无写入")
        return 0
    if not args.execute:
        print(f"（dry-run，{len(rescued)} 张可救回。加 --execute 写库，"
              f"写入时将同时回填缺失的 EXIF 元数据）")
        return 0

    bak = db_path + f".bak_无鸟救回_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(db_path, bak)
    print(f"📦 已备份: {os.path.basename(bak)}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for r in rescued:
            now = time.strftime('%Y-%m-%dT%H:%M:%SZ')
            all_boxes = [{"conf": r["conf"], "box": r["box"],
                          "species": r["species"],
                          "species_conf": r["species_conf"]}]
            all_boxes += r["extra_boxes"]
            # 主框 = 置信度最高（is_selected=1），其余框 is_selected=0；
            # 种名全部照实入行，photos 主鸟种列只从主框按采纳线写
            for k, b in enumerate(all_boxes):
                cur.execute(
                    "INSERT INTO bird_detections (filename, bird_index, "
                    "is_selected, bbox_x, bbox_y, bbox_w, bbox_h, "
                    "area_ratio, yolo_conf, species_cn, "
                    "species_confidence, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                    (r["prefix"], k, 1 if k == 0 else 0,
                     b["box"][0], b["box"][1], b["box"][2], b["box"][3],
                     0.0, b["conf"], b["species"], b["species_conf"], now))
            adopt = (r["species_conf"] >= args.threshold)
            cur.execute(
                "UPDATE photos SET has_bird=1, confidence=?, rating=0, "
                + ("bird_species_cn=?, birdid_confidence=?, " if adopt else "")
                + "updated_at=CURRENT_TIMESTAMP WHERE filename=?",
                ((r["conf"], r["species"], r["species_conf"], r["prefix"])
                 if adopt else (r["conf"], r["prefix"])))
            mark = "✅ 采纳种" if adopt else "种照实入行、photos 未采纳"
            extra = f"，另 {len(all_boxes) - 1} 个次要框" \
                if len(all_boxes) > 1 else ""
            print(f"✅ {r['prefix']}: has_bird=1 rating=0（{mark}）{extra}")
        # V5.8.1: 门控拦截照片从未提取 EXIF——救回写库时顺带回填
        # 拍摄时间/机身/ISO 等标准字段（仅填 NULL 列，不覆盖已有值），
        # 否则浏览库 sidebar 与 sidecar 的日期等字段为空。
        # V5.8.1: Gated-out photos never had EXIF extracted; backfill
        # the standard fields from originals (NULL columns only) so the
        # browse sidebar/sidecar show dates.
        exiftool_path = resolve_exiftool()
        if exiftool_path is None:
            print("⚠️ 未找到 ExifTool，跳过 EXIF 回填")
        else:
            n_exif = backfill_photos_exif(
                conn, exiftool_path, root, [r["prefix"] for r in rescued])
            print(f"✅ EXIF 回填 {n_exif} 张")
        conn.commit()
    finally:
        conn.close()
    print(f"→ 完成 {len(rescued)} 张；sidecar 用 reexport_sidecars.py 补")
    return 0


if __name__ == "__main__":
    sys.exit(main())
