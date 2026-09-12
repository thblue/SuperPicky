#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未定种照片补种：按指定置信度门槛（默认 40%）重新识别并采纳物种。

面向「已跑过批处理、有鸟但 species 为空」的照片（分类器 top1 当时低于
birdid 存储门槛）。用与管线逐鸟分类完全相同的路径重识别：
smart_square_crop(0.15 padding) → identify_bird(preloaded_crop, 地理过滤)，
采纳（conf ≥ 门槛）后镜像管线的三处存储：

- report.db photos: bird_species_cn/en + birdid_confidence
- report.db bird_detections（主鸟行）: species三名 + species_confidence
  + class_id + gbif_rarity_100 + china_protection_level（top-1 照实入库）
- sidecar meta/<前缀>.json: processing.species_main + detections[].species

低于门槛的照片不动。--execute 前自动备份 report.db；默认 dry-run。
单一写者原则：本脚本即 SuperPicky 运维上下文，零接触照片文件。

用法:
    python scripts_dev/backfill_species.py <照片目录> [--threshold 40] [--execute]
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.find_bird_util import raw_to_jpeg  # noqa: E402
from tools.image_crop import smart_square_crop  # noqa: E402

RAW_EXTS = (".CR3", ".cr3", ".NEF", ".nef", ".ARW", ".arw",
            ".RAF", ".raf", ".ORF", ".orf", ".DNG", ".dng")
_PADDING = 0.15  # 与 core/multi_bird._BIRDID_PADDING_RATIO 一致


def find_raw(directory: str, prefix: str) -> Optional[str]:
    """按前缀探测 RAW 文件路径。/ Probe the RAW file path by prefix."""
    for ext in RAW_EXTS:
        p = os.path.join(directory, prefix + ext)
        if os.path.exists(p):
            return p
    return None


def read_bgr(path: str) -> Optional[np.ndarray]:
    """中文/UNC 安全读图（与 ai_model.read_image_bgr 同法）。/ UNC-safe read."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def repair_photos_rarity(root: str, db_path: str, execute: bool,
                         log=print) -> int:
    """
    修复模式：为「已定种但 photos 行缺稀有度列」的照片补齐稀有度四列。

    背景（2026-09-12 P2-2 修复）：正常采纳路径会连 iucn/gbif/aesthetic/
    china_protection 四列一起写 photos 行（photo_processor），早期版本的
    本脚本只写了鸟种+置信（detections 行倒是有 gbif/china）。本模式不重跑
    推理、不动鸟种与置信：gbif/china 从主鸟 detections 行原样复制，
    iucn/aesthetic 按 detections.class_id 查参考库（鸟种级常量，无漂移）。

    参数:
        root (str): 照片目录（未用，保持签名一致）
        db_path (str): report.db 路径
        execute (bool): 写库（默认 dry-run）
        log: 打印回调

    返回:
        int: 0 成功；1 无待修复

    Repair mode: fill the four photos-level rarity columns for species-
    adopted photos missing them, from the detections row + reference DB.
    Never touches species/confidence.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT p.filename, d.class_id, d.gbif_rarity_100, d.china_protection_level
        FROM photos p
        JOIN bird_detections d ON d.filename = p.filename AND d.is_selected = 1
        WHERE p.has_bird = 1
          AND p.bird_species_cn IS NOT NULL AND p.bird_species_cn != ''
          AND p.gbif_rarity_100 IS NULL AND p.iucn_category IS NULL
        ORDER BY p.filename""").fetchall()
    conn.close()
    if not rows:
        log("无待修复照片（稀有度列齐全或无主鸟检测行）")
        return 0
    log(f"待修复稀有度列: {len(rows)} 张")

    from birdid.bird_database_manager import BirdDatabaseManager
    mgr = BirdDatabaseManager()

    plans = []
    for r in rows:
        if r["class_id"] is None:
            log(f"  ⚠️ {r['filename']}: detections 行无 class_id，跳过")
            continue
        iucn = mgr.get_iucn_by_class_id(int(r["class_id"]))
        aesthetic = mgr.get_aesthetic_by_class_id(int(r["class_id"]))
        plans.append((r["filename"], iucn, r["gbif_rarity_100"],
                      aesthetic, r["china_protection_level"]))
        log(f"  {r['filename']}: iucn={iucn} gbif={r['gbif_rarity_100']} "
            f"aesthetic={aesthetic if aesthetic is not None else '-'} "
            f"china={r['china_protection_level']}")

    if not execute:
        log("（dry-run，未写库。确认后加 --execute 执行）")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path + f".bak_修复稀有度_{ts}"
    shutil.copy(db_path, bak)
    log(f"📦 已备份: {os.path.basename(bak)}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for fn, iucn, gbif, aesthetic, china in plans:
            cur.execute(
                "UPDATE photos SET iucn_category=?, gbif_rarity_100=?,"
                " aesthetic_index=?, china_protection_level=?,"
                " updated_at=CURRENT_TIMESTAMP WHERE filename=?",
                (iucn, gbif, aesthetic, china, fn))
        conn.commit()
        log(f"✅ 已修复 {len(plans)} 张的稀有度四列")
    finally:
        conn.close()
    return 0


def main() -> int:
    """
    补种主流程：查未定种 → 逐张重识别 → dry-run 报告 / --execute 采纳写库。

    返回:
        int: 0 成功
    """
    ap = argparse.ArgumentParser(description="未定种照片补种（低置信度门槛）")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--threshold", type=float, default=40.0,
                    help="采纳门槛（百分比，默认 40）")
    ap.add_argument("--country", default="CN", help="地理过滤国家码（默认 CN）")
    ap.add_argument("--repair-rarity", action="store_true",
                    help="修复模式：只补已定种照片缺失的稀有度四列"
                         "（不重跑推理、不动鸟种与置信）")
    ap.add_argument("--execute", action="store_true",
                    help="写库（默认 dry-run；写前自动备份 report.db）")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    db_path = os.path.join(root, ".superpicky", "report.db")

    if args.repair_rarity:
        return repair_photos_rarity(root, db_path, args.execute, log=print)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT p.filename, p.rating, p.confidence,
               d.id AS det_id, d.bird_index,
               d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
        FROM photos p
        JOIN bird_detections d ON d.filename = p.filename AND d.is_selected = 1
        WHERE p.has_bird = 1
          AND (p.bird_species_cn IS NULL OR p.bird_species_cn = '')
        ORDER BY p.confidence DESC""").fetchall()
    no_row = conn.execute("""
        SELECT COUNT(*) FROM photos p
        WHERE p.has_bird = 1
          AND (p.bird_species_cn IS NULL OR p.bird_species_cn = '')
          AND NOT EXISTS (SELECT 1 FROM bird_detections d
                          WHERE d.filename = p.filename AND d.is_selected = 1)
        """).fetchone()[0]
    conn.close()
    if no_row:
        print(f"⚠️ 另有 {no_row} 张未定种照片无主鸟检测行（无法定点重识别），跳过")
    print(f"📁 {root}  待补种 {len(rows)} 张（含 0★），采纳门槛 {args.threshold}%")

    from birdid.bird_identifier import identify_bird

    adopted: List[Dict] = []
    still_low: List[Dict] = []
    for i, r in enumerate(rows, 1):
        prefix = r["filename"]
        raw = find_raw(root, prefix)
        if raw is None:
            print(f"  [{i}/{len(rows)}] {prefix}: 找不到 RAW，跳过")
            continue
        preview = raw_to_jpeg(raw)
        img = read_bgr(preview) if preview else None
        if img is None:
            print(f"  [{i}/{len(rows)}] {prefix}: 预览不可读，跳过")
            continue
        x1 = int(r["bbox_x"])
        y1 = int(r["bbox_y"])
        x2 = x1 + int(r["bbox_w"])
        y2 = y1 + int(r["bbox_h"])
        try:
            crop = smart_square_crop(img, (x1, y1, x2, y2),
                                     padding_ratio=_PADDING)
            from PIL import Image as _PILImage
            pil_crop = _PILImage.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            result = identify_bird(raw, False, True, True, args.country,
                                   None, 1, None, pil_crop)
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] {prefix}: 识别异常 {exc}")
            continue
        if not (result and result.get("success") and result.get("results")):
            still_low.append({"prefix": prefix, "name": "（无结果）", "conf": 0.0,
                              "row": dict(r)})
            continue
        top = result["results"][0]
        # identify_bird 的 results[].confidence 已是 0-100 百分数，勿再 ×100
        conf = float(top.get("confidence") or 0.0)
        name = top.get("cn_name") or top.get("en_name") or "?"
        entry = {"prefix": prefix, "name": name, "conf": conf,
                 "row": dict(r), "top": top,
                 "en": top.get("en_name"), "sci": top.get("scientific_name")}
        if conf >= args.threshold:
            adopted.append(entry)
            print(f"  [{i}/{len(rows)}] {prefix}: ✅ {name} {conf:.0f}%"
                  f"（{r['rating']}★，原检测conf {r['confidence']:.0%}）")
        else:
            still_low.append(entry)
            print(f"  [{i}/{len(rows)}] {prefix}: 仍低于门槛 {name} {conf:.0f}%")

    print(f"\n📊 可采纳 {len(adopted)}/{len(rows)}；"
          f"仍低于 {args.threshold}% {len(still_low)} 张")
    if not adopted:
        return 0

    if not args.execute:
        print("（dry-run，未写库。确认后加 --execute 执行）")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path + f".bak_补种{args.threshold:.0f}_{ts}"
    shutil.copy(db_path, bak)
    print(f"📦 已备份: {os.path.basename(bak)}")

    meta_dir = os.path.join(root, ".superpicky", "meta")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for e in adopted:
            top, prefix = e["top"], e["prefix"]
            cur.execute(
                "UPDATE photos SET bird_species_cn=?, bird_species_en=?,"
                " birdid_confidence=?, iucn_category=?, gbif_rarity_100=?,"
                " aesthetic_index=?, china_protection_level=?,"
                " updated_at=CURRENT_TIMESTAMP WHERE filename=?",
                (top.get("cn_name"), top.get("en_name"),
                 e["conf"], top.get("iucn_category"),
                 top.get("gbif_rarity_100"), top.get("aesthetic_index"),
                 top.get("china_protection_level"), prefix))
            cur.execute(
                "UPDATE bird_detections SET species_cn=?, species_en=?,"
                " scientific_name=?, species_confidence=?, class_id=?,"
                " gbif_rarity_100=?, china_protection_level=?,"
                " updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (top.get("cn_name"), top.get("en_name"),
                 top.get("scientific_name"), e["conf"],
                 top.get("class_id"), top.get("gbif_rarity_100"),
                 top.get("china_protection_level"), e["row"]["det_id"]))
        conn.commit()
        print(f"✅ 已写 DB：{len(adopted)} 张 photos + 主鸟 detections 行")
    finally:
        conn.close()

    # sidecar 镜像：processing.species_main + detections[bird_index].species
    synced = 0
    for e in adopted:
        path = os.path.join(meta_dir, e["prefix"] + ".json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sp = {"cn": e["top"].get("cn_name"), "en": e["top"].get("en_name"),
                  "confidence": round(e["conf"], 4)}
            data.setdefault("processing", {})["species_main"] = sp
            dets = data.get("detections") or []
            idx = e["row"]["bird_index"]
            if isinstance(idx, int) and 0 <= idx < len(dets):
                dets[idx]["species"] = sp
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            synced += 1
        except Exception as exc:
            print(f"  ⚠️ sidecar 同步失败 {e['prefix']}: {exc}")
    print(f"✅ sidecar 同步 {synced}/{len(adopted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
