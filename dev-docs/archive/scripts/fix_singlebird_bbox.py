#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复单鸟照片 bbox/polygon 坐标空间错位的存量数据（一次性维护脚本）。

背景 / Background:
core/photo_processor.py 的 birdid worker 在照片只有主鸟（无待分类次要鸟）
时不整图解码，orig_dims 曾回退为处理图（长边 1024）尺寸，导致
core/multi_bird._bbox_to_orig 的缩放比恒为 1——bbox/mask_polygon 以
1024 处理图坐标入库并导出到 sidecar，多鸟编辑器按原图尺寸绘制时，
框缩成左上区域一个错位的小点。多鸟照片（存在次要鸟）不受影响。

While processing single-bird photos the birdid worker used to fall back to
the processed-frame (1024-long-edge) dims as "original" dims, so the
proc→orig scaling factor was 1.0 and bboxes/polygons were stored in
1024-space. Multi-bird photos were unaffected (the original image is
loaded for secondary-bird crops).

修复方式（确定性回放，不重跑模型）/ Deterministic replay, no re-inference:
1. 逐行用 area_ratio 反推坐标所在空间：implied = bbox_w*bbox_h/area_ratio。
   等于处理图面积（由原图尺寸按 ai_model.preprocess_image 的
   int() 截断规则复现）→ 该行是错位的处理图坐标；
2. 原图尺寸优先取 .superpicky/cache/temp_preview/<prefix>.jpg 文件头
   （即检测时实际使用的同一幅内嵌预览，坐标空间天然一致）；缓存缺失时
   回退到目录内 <prefix>.* 原文件：JPEG 直读头部，RAW 经 exiftool
   提取内嵌 JPEG 后读头部；
3. 按原图/处理图比例回放 bbox（复现 _bbox_to_orig 的 round+clip+最小
   1px）与 polygon（复现 _polygon_to_orig 的逐点 round），更新
   report.db，并把 .superpicky/meta/<prefix>.json 的 detections[].bbox
   同步缩放（保持导出格式：UTF-8 / indent=2 / ensure_ascii=False /
   tmp+replace 原子写）。

幂等性 / Idempotency:
修复后 implied == 原图面积，重复运行不再命中任何行。

用法 / Usage:
    python scripts_dev/fix_singlebird_bbox.py <photo_dir>            # 干跑
    python scripts_dev/fix_singlebird_bbox.py <photo_dir> --apply    # 实际写库
    python scripts_dev/fix_singlebird_bbox.py <photo_dir> --apply --no-backup

运行前请关闭 SuperPicky 主程序，避免 SQLite 并发写冲突。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# 处理图长边目标尺寸，须与 config.py TARGET_IMAGE_SIZE 一致
# Processed-frame long edge; must match config.TARGET_IMAGE_SIZE.
TARGET_IMAGE_SIZE = 1024

# implied 面积与处理图面积的相对容差（浮点往返误差远小于此值，
# 而与原图面积（约大 46 倍）差距巨大，可安全二分）
# Relative tolerance when matching implied area against processed-frame area.
AREA_TOLERANCE = 0.005

# RAW 扩展名（与 tools/find_bird_util 的 RAW 集合保持一致的小写子集）
# RAW extensions for the exiftool fallback.
RAW_EXTS = {".cr3", ".cr2", ".crw", ".nef", ".nrw", ".arw", ".raf",
            ".dng", ".orf", ".rw2", ".heic", ".heif", ".avif"}

# 目录内按前缀找原文件时认可的图像扩展名（排除 .xmp 等附属文件）
# Image extensions accepted when matching by prefix in the photo tree.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"} | RAW_EXTS


def _pil_header_size(blob_or_path) -> Optional[Tuple[int, int]]:
    """
    用 PIL 惰性打开读 (w, h)，接受路径或字节流，失败返回 None。

    Read (w, h) via PIL lazy open from a path or bytes; None on failure.
    """
    try:
        from PIL import Image
        if isinstance(blob_or_path, (bytes, bytearray)):
            im = Image.open(io.BytesIO(bytes(blob_or_path)))
        else:
            im = Image.open(blob_or_path)
        return im.size
    except Exception:
        return None


def processed_dims(orig_w: int, orig_h: int) -> Tuple[int, int]:
    """
    复现 ai_model.preprocess_image 的缩放取整：scale = 1024/max(w,h)，
    新尺寸用 int() 截断。处理图面积用于判定行坐标所在空间。

    Replicate preprocess_image sizing (int truncation) so the implied-area
    discriminator uses the exact processed dims the pipeline produced.
    """
    scale = TARGET_IMAGE_SIZE / float(max(orig_w, orig_h))
    return int(orig_w * scale), int(orig_h * scale)


def resolve_orig_dims(photo_dir: str, prefix: str,
                      cache_dir: str,
                      exiftool: str) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    解析某照片检测时所用「原图」（内嵌预览 JPEG）的精确尺寸。

    优先级 / Priority:
    1. .superpicky/cache/temp_preview/<prefix>.jpg 文件头（检测输入本身）；
    2. 目录内 <prefix>.* 原文件：JPEG 直接读头部；
       RAW 用 exiftool 依次提取 JpgFromRaw / PreviewImage 后读头部。

    返回:
    (dims, source)：dims=((w,h)) 或 None；source 为尺寸来源描述（日志用）。

    Resolve the exact dims of the preview JPEG the detector consumed.
    Returns ((w, h), source) or (None, reason).
    """
    cached = os.path.join(cache_dir, prefix + ".jpg")
    if os.path.exists(cached):
        dims = _pil_header_size(cached)
        if dims:
            return dims, "cache"

    # 缓存缺失：在照片树里收集所有同名文件（跳过 .superpicky 与非图像
    # 附属文件如 .xmp），按 JPEG → RAW → 其他图像 的优先级解析头部
    # Cache miss: collect same-prefix files in the tree (ignoring .xmp
    # sidecars) and resolve by priority JPEG → RAW → other image.
    matches: List[Tuple[str, str]] = []      # (path, ext)
    for root, dirs, files in os.walk(photo_dir):
        dirs[:] = [d for d in dirs if d != ".superpicky"]
        for f in files:
            name, ext = os.path.splitext(f)
            if name == prefix and ext.lower() in IMAGE_EXTS:
                matches.append((os.path.join(root, f), ext.lower()))
    matches.sort(key=lambda m: (m[1] not in (".jpg", ".jpeg"),
                                m[1] not in RAW_EXTS))

    for path, ext in matches:
        if ext not in RAW_EXTS:
            dims = _pil_header_size(path)
            if dims:
                return dims, f"file:{os.path.basename(path)}"
        else:
            for tag in ("-JpgFromRaw", "-PreviewImage"):
                try:
                    r = subprocess.run(
                        [exiftool, "-b", tag, "--", path],
                        capture_output=True, timeout=30)
                    if r.stdout and len(r.stdout) > 64:
                        dims = _pil_header_size(r.stdout)
                        if dims:
                            return dims, f"exiftool:{tag.lstrip('-')}:{os.path.basename(path)}"
                except Exception:
                    continue
    return None, "not-found"


def scale_bbox_row(bbox: Tuple[float, float, float, float],
                   sx: float, sy: float,
                   orig_w: int, orig_h: int) -> Tuple[float, float, float, float]:
    """
    复现 core/multi_bird._bbox_to_orig：xyxy 逐边 round+clip，保证 ≥1px，
    返回 (x, y, w, h)。round 采用 Python 银行家舍入，与原实现一致。

    Replay _bbox_to_orig (round + clip + min 1px) on an (x,y,w,h) row.
    """
    x1, y1, w, h = bbox
    ox1 = max(0, int(round(x1 * sx)))
    oy1 = max(0, int(round(y1 * sy)))
    ox2 = min(orig_w, int(round((x1 + w) * sx)))
    oy2 = min(orig_h, int(round((y1 + h) * sy)))
    if ox2 <= ox1:
        ox2 = min(orig_w, ox1 + 1)
    if oy2 <= oy1:
        oy2 = min(orig_h, oy1 + 1)
    return float(ox1), float(oy1), float(ox2 - ox1), float(oy2 - oy1)


def scale_polygon_json(poly_json: Optional[str],
                       sx: float, sy: float) -> Optional[str]:
    """
    复现 core/multi_bird._polygon_to_orig：逐点 round 后重新序列化。

    Replay _polygon_to_orig: per-point round and re-serialise.
    """
    if not poly_json:
        return poly_json
    try:
        pts = json.loads(poly_json)
    except (ValueError, TypeError):
        return poly_json
    if not isinstance(pts, list):
        return poly_json
    scaled = [[int(round(p[0] * sx)), int(round(p[1] * sy))] for p in pts]
    return json.dumps(scaled, ensure_ascii=False)


def atomic_write_json(path: str, payload: dict) -> None:
    """
    与 core/sidecar_export._atomic_write_json 相同的写出约定：
    UTF-8 无 BOM、ensure_ascii=False、indent=2、tmp+os.replace 原子替换。

    Same write conventions as the sidecar exporter (atomic tmp+replace).
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="修复单鸟照片 bbox/polygon 的处理图坐标错位（默认干跑）")
    parser.add_argument("photo_dir", help="照片根目录（含 .superpicky/report.db）")
    parser.add_argument("--apply", action="store_true",
                        help="实际写库与改写 sidecar JSON（默认只打印计划）")
    parser.add_argument("--no-backup", action="store_true",
                        help="跳过 report.db 备份（默认自动备份）")
    args = parser.parse_args()

    photo_dir = os.path.abspath(args.photo_dir)
    sp_dir = os.path.join(photo_dir, ".superpicky")
    db_path = os.path.join(sp_dir, "report.db")
    cache_dir = os.path.join(sp_dir, "cache", "temp_preview")
    meta_dir = os.path.join(sp_dir, "meta")
    exiftool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "exiftools_win", "exiftool.exe")
    if not os.path.exists(exiftool):
        exiftool = "exiftool"

    if not os.path.exists(db_path):
        print(f"❌ 未找到数据库: {db_path}")
        return 1
    if not args.apply:
        print("=== 干跑模式（加 --apply 实际写入） ===")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, filename, bird_index, bbox_x, bbox_y, bbox_w, bbox_h,"
        " mask_polygon, area_ratio FROM bird_detections").fetchall()

    # 按照片分组：同一张照片所有行共享同一次检测运行的坐标空间；
    # 行按 bird_index 排序，保证与 sidecar detections[] 的顺序一一对应
    # Group by photo; sort rows by bird_index to match sidecar order.
    by_photo: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        by_photo.setdefault(r["filename"], []).append(r)
    for prefix in by_photo:
        by_photo[prefix].sort(key=lambda r: r["bird_index"])

    fixed_rows = 0
    fixed_photos = 0
    skipped: List[str] = []
    db_updates: List[Tuple] = []          # (x, y, w, h, polygon, id)
    json_fixes: List[Tuple[str, int, Tuple[float, float, float, float]]] = []

    for prefix, prows in sorted(by_photo.items()):
        # 任一行 implied==处理图面积 → 该照片所有行都是处理图坐标
        # Any row implying processed-space marks the whole photo.
        probe = next((r for r in prows
                      if r["area_ratio"] and r["bbox_w"] and r["bbox_h"]), None)
        if probe is None:
            continue
        implied = probe["bbox_w"] * probe["bbox_h"] / probe["area_ratio"]

        dims, src = resolve_orig_dims(photo_dir, prefix, cache_dir, exiftool)
        if dims is None:
            skipped.append(f"{prefix} ({src})")
            continue
        ow, oh = dims
        pw, ph = processed_dims(ow, oh)
        if abs(implied - pw * ph) > AREA_TOLERANCE * pw * ph:
            continue  # 原图坐标，无需修复 / Already in original space.

        sx = ow / float(pw)
        sy = oh / float(ph)
        fixed_photos += 1
        for r in prows:
            nb = scale_bbox_row(
                (r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]),
                sx, sy, ow, oh)
            np_ = scale_polygon_json(r["mask_polygon"], sx, sy)
            db_updates.append((nb[0], nb[1], nb[2], nb[3], np_, r["id"]))
            json_fixes.append((prefix, r["bird_index"], nb))
            fixed_rows += 1

    print(f"命中待修复: {fixed_photos} 张照片 / {fixed_rows} 行"
          f"（总检测照片 {len(by_photo)}）")
    if skipped:
        print(f"⚠️ 无法解析原图尺寸而跳过 {len(skipped)} 张: "
              + ", ".join(skipped))

    for prefix, _idx, nb in json_fixes[:5]:
        print(f"  例 {prefix}: bbox -> ({nb[0]:.0f}, {nb[1]:.0f}, "
              f"{nb[2]:.0f}, {nb[3]:.0f})")

    if not args.apply:
        conn.close()
        return 0

    # 备份 → 事务更新 DB → 同步 sidecar JSON
    # WAL 模式下未 checkpoint 的内容在 -wal 文件里，备份前先落盘。
    # Backup → transactional DB update → sidecar sync. For WAL databases,
    # checkpoint first so the backup file copy is complete on its own.
    if not args.no_backup:
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower() == "wal":
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = f"{db_path}.bak_singlebird_bbox_{stamp}"
        shutil.copy2(db_path, bak)
        print(f"已备份数据库 -> {os.path.basename(bak)}")

    with conn:
        conn.executemany(
            "UPDATE bird_detections SET bbox_x=?, bbox_y=?, bbox_w=?,"
            " bbox_h=?, mask_polygon=? WHERE id=?", db_updates)
    print(f"✅ report.db 已更新 {len(db_updates)} 行")

    # 按照片聚合修复结果，以 detection 的 index 字段显式对齐 sidecar
    # 而非依赖数组顺序，避免顺序漂移时交叉写错行。
    # Rewrite each photo's sidecar bboxes, keyed by the detection "index"
    # field instead of array order, to be robust to ordering drift.
    per_photo: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}
    for prefix, bird_index, nb in json_fixes:
        per_photo.setdefault(prefix, {})[bird_index] = nb

    json_done = 0
    for prefix, box_map in per_photo.items():
        path = os.path.join(meta_dir, prefix + ".json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            dets = payload.get("detections")
            if not isinstance(dets, list):
                print(f"  ⚠️ {prefix}: sidecar 无 detections，跳过 JSON（DB 已修）")
                continue
            hit = 0
            for det in dets:
                nb = box_map.get(det.get("index"))
                if nb is not None:
                    det["bbox"] = [nb[0], nb[1], nb[2], nb[3]]
                    hit += 1
            if hit == 0:
                print(f"  ⚠️ {prefix}: sidecar index 未命中，跳过 JSON（DB 已修）")
                continue
            atomic_write_json(path, payload)
            json_done += 1
        except (OSError, ValueError) as e:
            print(f"  ⚠️ {prefix}: JSON 更新失败 {e}")
    print(f"✅ sidecar JSON 已同步 {json_done} 张")

    # 收尾自检：重查数据库并复跑同一判别逻辑，真实缩放错位应为 0
    # （原图长边 ≤1024 的照片处理图==原图，scale 恒为 1，属正常）
    # Post-check: re-query the DB and re-run the discriminator; no
    # genuinely mis-scaled photo should remain.
    remain = 0
    fresh = conn.execute(
        "SELECT filename, bird_index, bbox_w, bbox_h, area_ratio"
        " FROM bird_detections ORDER BY filename, bird_index").fetchall()
    fresh_groups: Dict[str, List[sqlite3.Row]] = {}
    for r in fresh:
        fresh_groups.setdefault(r["filename"], []).append(r)
    for prefix, prows in fresh_groups.items():
        probe = next((r for r in prows
                      if r["area_ratio"] and r["bbox_w"] and r["bbox_h"]), None)
        if probe is None:
            continue
        implied = probe["bbox_w"] * probe["bbox_h"] / probe["area_ratio"]
        dims, _src = resolve_orig_dims(photo_dir, prefix, cache_dir, exiftool)
        if dims is None:
            continue
        pw, ph = processed_dims(*dims)
        if (pw, ph) == dims:
            continue
        if abs(implied - pw * ph) <= AREA_TOLERANCE * pw * ph:
            remain += 1
    print(f"自检: 仍疑似处理图坐标的照片 = {remain}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
