#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPP 伽马提亮照片的指标重算：对 refresh_gamma_previews.py 刷亮过预览的
照片，用提亮后的预览重跑关键点锐度/眼睛可见度/TOPIQ，并按管线口径更新
report.db 指标列。配合 rerate-v2 完成整批重新配额定星。

背景（2026-09-21 玉渊潭）：21 张飞版暗片的关键点在原始暗预览上找不到
（head_sharp=None 或接近 0），星级被压死。DPP 伽马提亮后预览可读，
重算指标让 V2 有机会正常入池。

口径对齐 core/photo_processor.py 主管线：
- 鸟裁剪：全分辨率预览按 DB 主鸟框 + 15% padding（:2370-2374）；
- head_sharp 存关键点原始锐度（不乘 ISO，rerate-v2 侧再乘 iso_factor）；
- nima_score / adj_topiq = TOPIQ × 对焦美学权重 × (飞版×1.1)；
- adj_sharpness = 原始锐度 × iso_factor × 对焦锐度权重 × (飞版×1.2)；
- 对焦权重从该照片原 caption「[修正] 对焦锐度权重: X | 对焦美学权重: Y」
  解析（与原跑批逐张一致），解析失败回退 (1.0, 1.0)。

不动的列：confidence / is_flying / focus_status / burst_id / iso /
鸟种 / 星级（星级由 rerate-v2 全批重排）。写前自动备份 report.db。
单一写者原则：零接触照片文件，只更新 DB 指标列。

用法:
    python scripts_dev/recalc_gamma_scores.py <照片目录> [--execute]
"""

import argparse
import os
import re
import shutil
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts_dev.refresh_gamma_previews import read_gamma_params  # noqa: E402

FOCUS_W_RE = re.compile(
    r"对焦锐度权重:\s*([\d.]+)\s*\|\s*对焦美学权重:\s*([\d.]+)")
PAD_RATIO = 0.15  # 与管线一致的鸟裁剪 padding / crop padding as in pipeline


def read_bgr(path: str) -> Optional[np.ndarray]:
    """中文/UNC 安全读图。/ UNC-safe read."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def crop_with_padding(img: np.ndarray, bbox: Tuple[float, float, float, float]):
    """
    按主鸟框 + 15% padding 裁剪（与管线 :2370-2374 同法）。

    参数:
        img (np.ndarray): 全分辨率 BGR 预览
        bbox (Tuple[float, float, float, float]): DB 主鸟框 (x, y, w, h)

    返回:
        Tuple[int, int, np.ndarray]: (x偏移, y偏移, 裁剪图)；无效返回 (0, 0, None)
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w, int(x + bw))
    y2 = min(h, int(y + bh))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, None
    pad = int(max(x2 - x1, y2 - y1) * PAD_RATIO)
    ox = max(0, x1 - pad)
    oy = max(0, y1 - pad)
    crop = img[oy:min(h, y2 + pad), ox:min(w, x2 + pad)]
    if crop.size == 0:
        return 0, 0, None
    return ox, oy, crop.copy()


def main() -> int:
    """
    主流程：重算伽马提亮照片的关键点/TOPIQ 指标并更新 DB。

    返回:
        int: 0 成功
    """
    ap = argparse.ArgumentParser(description="DPP 伽马照片指标重算")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--execute", action="store_true",
                    help="写库（默认 dry-run；写前自动备份 report.db）")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    db_path = os.path.join(root, ".superpicky", "report.db")
    exiftool = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'exiftools_win', 'exiftool.exe')
    cr3s = sorted(
        os.path.join(root, f) for f in os.listdir(root)
        if f.lower().endswith('.cr3'))
    params = read_gamma_params(cr3s, exiftool)
    if not params:
        print("未发现 DPP 伽马编辑照片")
        return 1
    targets = sorted(params)
    print(f"📁 {root}\n   待重算 {len(targets)} 张")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    plans: List[Dict] = []
    for prefix in targets:
        row = conn.execute(
            "SELECT p.filename, p.iso, p.is_flying, p.caption, p.head_sharp, "
            "p.nima_score, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h "
            "FROM photos p LEFT JOIN bird_detections d "
            "  ON d.filename = p.filename AND d.is_selected = 1 "
            "  AND COALESCE(d.deleted, 0) = 0 "
            "WHERE p.filename = ?", (prefix,)).fetchone()
        if row is None:
            print(f"  {prefix}: photos 行不存在，跳过")
            continue
        if row["bbox_x"] is None:
            # 无检测行照片（原跑批锐度 0 被识鸟门控挡下、未写
            # bird_detections）用提亮预览 YOLO 重检取最佳鸟框。
            # No stored box (sharpness-0 photos never reached the BirdID
            # stage): redetect the best bird box on the brightened preview.
            from scripts_dev.backfill_species import detect_fallback_box
            preview = os.path.join(root, '.superpicky', 'cache',
                                   'temp_preview', prefix + '.jpg')
            img = read_bgr(preview)
            if img is None:
                print(f"  {prefix}: 预览不可读且无主鸟框，跳过")
                continue
            box, det_conf = detect_fallback_box(img, 0.25)
            if box is None:
                print(f"  {prefix}: 提亮预览重检无鸟框，跳过")
                continue
            bbox = (float(box[0]), float(box[1]),
                    float(box[2] - box[0]), float(box[3] - box[1]))
            print(f"  {prefix}: 无库内框 → 重检 {det_conf:.0%}")
        else:
            bbox = (row["bbox_x"], row["bbox_y"],
                    row["bbox_w"], row["bbox_h"])
        fw = (1.0, 1.0)
        if row["caption"]:
            m = FOCUS_W_RE.search(row["caption"])
            if m:
                fw = (float(m.group(1)), float(m.group(2)))
        plans.append({
            "prefix": prefix, "iso": row["iso"],
            "is_flying": bool(row["is_flying"]),
            "fw": fw, "old_sharp": row["head_sharp"],
            "old_topiq": row["nima_score"],
            "bbox": bbox,
        })
    conn.close()
    print(f"   有主鸟框可重算 {len(plans)} 张")

    from core.keypoint_detector import get_keypoint_detector
    from core.iqa_scorer import get_iqa_scorer
    from config import get_best_device
    from core.rerate_v2 import iso_factor

    device = get_best_device().type
    kp = get_keypoint_detector()
    kp.load_model()
    scorer = get_iqa_scorer(device=device)

    results: List[Dict] = []
    for i, p in enumerate(plans, 1):
        prefix = p["prefix"]
        preview = os.path.join(root, '.superpicky', 'cache',
                               'temp_preview', prefix + '.jpg')
        img = read_bgr(preview)
        if img is None:
            print(f"  [{i}/{len(plans)}] {prefix}: 提亮预览不可读，跳过")
            continue
        ox, oy, crop = crop_with_padding(img, p["bbox"])
        if crop is None:
            print(f"  [{i}/{len(plans)}] {prefix}: 框无效，跳过")
            continue
        kp_result = kp.detect(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                              box=p["bbox"], seg_mask=None)
        head_sharp = float(kp_result.head_sharpness) if kp_result else 0.0
        left_eye = float(kp_result.left_eye_vis) if kp_result else 0.0
        right_eye = float(kp_result.right_eye_vis) if kp_result else 0.0
        beak = float(kp_result.beak_vis) if kp_result else 0.0
        topiq = float(scorer.calculate_from_array(crop))
        # 管线口径：adj 值带 ISO 归一化 + 对焦权重 + 飞版加成
        f_sharp, f_aesth = p["fw"]
        norm_sharp = head_sharp * iso_factor(p["iso"])
        adj_sharp = norm_sharp * f_sharp if norm_sharp else 0.0
        adj_topiq = topiq * f_aesth
        if p["is_flying"]:
            if head_sharp:
                adj_sharp *= 1.2
            adj_topiq *= 1.1
        results.append({
            "prefix": prefix, "head_sharp": head_sharp,
            "left_eye": left_eye, "right_eye": right_eye, "beak": beak,
            "nima_score": adj_topiq, "adj_sharpness": adj_sharp,
            "adj_topiq": adj_topiq, "norm_sharp": norm_sharp,
            "old_sharp": p["old_sharp"], "old_topiq": p["old_topiq"],
        })
        old_s = f"{p['old_sharp']:.0f}" if p["old_sharp"] is not None else "None"
        print(f"  [{i}/{len(plans)}] {prefix}: 锐度 {old_s} → "
              f"{head_sharp:.0f}(归一化 {norm_sharp:.0f})，"
              f"TOPIQ {p['old_topiq'] or 0:.2f} → {adj_topiq:.2f}，"
              f"眼/喙可见 {max(left_eye, right_eye):.2f}/{beak:.2f}")

    if not results:
        print("无可更新结果")
        return 1
    if not args.execute:
        print("（dry-run，未写库。确认后加 --execute 执行）")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path + f".bak_伽马重算_{ts}"
    shutil.copy(db_path, bak)
    print(f"📦 已备份: {os.path.basename(bak)}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for r in results:
            cur.execute(
                "UPDATE photos SET head_sharp=?, left_eye=?, right_eye=?,"
                " beak=?, nima_score=?, adj_sharpness=?, adj_topiq=?,"
                " updated_at=CURRENT_TIMESTAMP WHERE filename=?",
                (r["head_sharp"], r["left_eye"], r["right_eye"], r["beak"],
                 r["nima_score"], r["adj_sharpness"], r["adj_topiq"],
                 r["prefix"]))
        conn.commit()
        print(f"✅ 已更新 {len(results)} 张指标列")
    finally:
        conn.close()
    print("→ 下一步：superpicky_cli.py rerate-v2 <目录> --min-conf 0.4 "
          "--current-conf 40 [--execute]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
