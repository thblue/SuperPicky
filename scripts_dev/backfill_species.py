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

2026-09-12 年度捞回扩展（五个 2026 库统一补跑）：
- 排除人工处理过的照片：bird_detections 有行但主鸟行（is_selected=1）
  均为 deleted=1 → 用户已在编辑器里人工去掉鸟，一律跳过；主鸟行
  edited=1（人工改种，个别照片 photos 行未同步）同样跳过不覆盖；
  仅删除了次要框的照片不受影响（主鸟行仍存活，继续参与）。
- 无任何检测行的旧照片（老版本批处理未写 bird_detections）用当前 YOLO
  低门槛（conf ≥ --redetect-conf，默认 0.25）重检取最佳鸟框再识别；
  检测置信度照实写入报告供人工权衡。
- 视频封面（filename 以 _vcover 结尾）不属于照片流，跳过（归视频工作流）。
- 提速：photos.temp_jpeg_path 全分辨率预览缓存直接复用（实测与 bbox 同为
  原图像素空间，省去每张 RAW 重解码）；仅缓存缺失时才 raw_to_jpeg。
- 纯 JPEG 照片（IMG_* 等，无 RAW）：按前缀直读 JPG/JPEG/PNG 本体。

低于门槛的照片不动。--execute 前自动备份 report.db；默认 dry-run。
单一写者原则：本脚本即 SuperPicky 运维上下文，零接触照片文件。

用法:
    python scripts_dev/backfill_species.py <照片目录> [--threshold 40]
        [--country CN] [--redetect-conf 0.25] [--report <md路径>] [--execute]
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from constants import VIDEO_COVER_SUFFIX  # noqa: E402
from tools.find_bird_util import raw_to_jpeg  # noqa: E402
from tools.image_crop import smart_square_crop  # noqa: E402

RAW_EXTS = (".CR3", ".cr3", ".NEF", ".nef", ".ARW", ".arw",
            ".RAF", ".raf", ".ORF", ".orf", ".DNG", ".dng")
PLAIN_EXTS = (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png")
_PADDING = 0.15  # 与 core/multi_bird._BIRDID_PADDING_RATIO 一致


def find_raw(directory: str, prefix: str) -> Optional[str]:
    """按前缀探测 RAW 文件路径。/ Probe the RAW file path by prefix."""
    for ext in RAW_EXTS:
        p = os.path.join(directory, prefix + ext)
        if os.path.exists(p):
            return p
    return None


def find_plain_image(directory: str, prefix: str) -> Optional[str]:
    """按前缀探测纯图像文件（无 RAW 的 JPEG/PNG 照片）。

    Probe a plain image file (JPEG/PNG photos that have no RAW sibling).
    """
    for ext in PLAIN_EXTS:
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


def load_source(root: str, prefix: str,
                temp_jpeg_path: Optional[str]) -> Tuple[Optional[str],
                                                        Optional[np.ndarray]]:
    """
    加载识别用的全分辨率 BGR 图与 EXIF 源路径。

    优先级：temp_jpeg_path 预览缓存（全分辨率、与 bbox 同像素空间）→
    RAW 经 raw_to_jpeg 现生成 → 同前缀纯图像直读。EXIF 源路径优先 RAW
    （GPS/拍摄时间齐全），其次图像本体。

    参数:
        root (str): 照片目录
        prefix (str): 文件名前缀
        temp_jpeg_path (Optional[str]): photos 表记录的预览缓存相对路径

    返回:
        Tuple[Optional[str], Optional[np.ndarray]]: (exif 源路径, BGR 图)，
        均不可得时返回 (None, None)
    """
    raw = find_raw(root, prefix)
    if temp_jpeg_path:
        p = os.path.normpath(os.path.join(root, temp_jpeg_path))
        if os.path.exists(p):
            img = read_bgr(p)
            if img is not None:
                return (raw or p), img
    if raw is not None:
        preview = raw_to_jpeg(raw)
        if preview and os.path.exists(preview):
            img = read_bgr(preview)
            if img is not None:
                return raw, img
    plain = find_plain_image(root, prefix)
    if plain is not None:
        img = read_bgr(plain)
        if img is not None:
            return plain, img
    return None, None


_YOLO_MODEL = None  # 进程内单例：兜底检测避免每张重新加载模型 / process-wide singleton


def detect_fallback_box(bgr: np.ndarray,
                        redetect_conf: float) -> Tuple[Optional[Tuple[int,
                                                                      int,
                                                                      int,
                                                                      int]],
                                                       float]:
    """
    无检测行照片的兜底：当前 YOLO 低门槛单遍检测，取最佳鸟框。

    与主管线同法：长边 1024 预处理 → model(imgsz=1024)，只取 bird 类，
    坐标按全分辨率比例还原。检测置信度返给调用方写入报告。

    参数:
        bgr (np.ndarray): 全分辨率 BGR 图
        redetect_conf (float): 检测置信度地板（0-1）

    返回:
        Tuple[Optional[Tuple[int, int, int, int]], float]:
        ((x1, y1, x2, y2), conf)；无鸟框时 (None, 0.0)
    """
    global _YOLO_MODEL
    from config import config, get_best_device
    from core.ai_model import load_yolo_model, preprocess_image

    if _YOLO_MODEL is None:
        _YOLO_MODEL = load_yolo_model()
    model = _YOLO_MODEL
    device = get_best_device()
    proc = preprocess_image("", target_size=config.ai.RESCUE_IMGSZ,
                            source_image=bgr)
    results = model(proc, imgsz=config.ai.RESCUE_IMGSZ,
                    conf=redetect_conf, device=device.type, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    bird_ix = np.flatnonzero(clss == config.ai.BIRD_CLASS_ID)
    if len(bird_ix) == 0:
        return None, 0.0
    best = bird_ix[int(np.argmax(confs[bird_ix]))]
    h, w = bgr.shape[:2]
    ph, pw = proc.shape[:2]
    sx, sy = w / pw, h / ph
    x1, y1, x2, y2 = xyxy[best]
    box = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
    return box, float(confs[best])


def classify(bgr: np.ndarray, box: Tuple[int, int, int, int],
             exif_path: str, country: str,
             dark_retry_conf: float = None,
             dark_bgr: np.ndarray = None,
             dark_box: Tuple[int, int, int, int] = None) -> Optional[Dict]:
    """
    与管线逐鸟分类完全相同的路径识别单框：方形的智能裁剪 → identify_bird。

    V5.9.2: 提供暗版渲染（dark_bgr + dark_box）时走双渲染对比——亮版首判
    低于采纳线时用原始暗版框图重判一次，净胜择优（暗图直判在部分样本上
    本就有 65-72% 正确率，见 bird_identifier.retry_crop）。

    参数:
        bgr (np.ndarray): 全分辨率 BGR 图（亮版/管线预览）
        box (Tuple[int, int, int, int]): 原图像素空间 (x1, y1, x2, y2)
        exif_path (str): 供 GPS/EXIF 读取的源文件路径
        country (str): 地理过滤国家码
        dark_retry_conf (float): 暗框重识别置信度线（0-100，V5.9 起
            与主管线一致；None=关闭）
        dark_bgr (Optional[np.ndarray]): 原始暗渲染全图（V5.9.2 双渲染）
        dark_box (Optional[Tuple]): 暗版坐标系的同位框

    返回:
        Optional[Dict]: identify_bird 原始结果；异常/无结果返回 None
    """
    from birdid.bird_identifier import identify_bird

    try:
        crop = smart_square_crop(bgr, box, padding_ratio=_PADDING)
        from PIL import Image as _PILImage
        pil_crop = _PILImage.fromarray(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        retry_pil = None
        if dark_bgr is not None and dark_box is not None:
            dark_crop = smart_square_crop(dark_bgr, dark_box,
                                          padding_ratio=_PADDING)
            retry_pil = _PILImage.fromarray(
                cv2.cvtColor(dark_crop, cv2.COLOR_BGR2RGB))
        return identify_bird(exif_path, False, True, True, country,
                             None, 1, None, pil_crop,
                             dark_retry_conf=dark_retry_conf,
                             retry_crop=retry_pil)
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


def collect_candidates(db_path: str) -> Dict:
    """
    查询未定种照片并按可处理性分桶。

    分桶规则（2026-09-12 年度捞回）：
    - with_box: bird_detections 有未删除且非人工改种（deleted=0, edited=0）
      的主鸟行 → 存储框直接识别；
    - redetect: 完全无检测行的旧照片 → YOLO 兜底重检；
    - skipped_manual: 有检测行但无存活主鸟行 → 用户已人工去掉鸟，排除；
    - skipped_edited: 主鸟行 edited=1（用户人工改过种，部分照片 photos 行
      未同步，如海淀 027A5117）→ 一律不覆盖人工结果，排除并上报；
    视频封面（_vcover 后缀）从所有桶剔除。

    参数:
        db_path (str): report.db 路径

    返回:
        Dict: {'with_box': [row...], 'redetect': [row...],
               'skipped_manual': [filename...],
               'skipped_edited': [filename...]}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    base = ("FROM photos p WHERE p.has_bird = 1 "
            "AND (p.bird_species_cn IS NULL OR p.bird_species_cn = '') ")
    with_box = conn.execute(f"""
        SELECT p.filename, p.rating, p.confidence, p.temp_jpeg_path,
               d.id AS det_id, d.bird_index,
               d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
        FROM photos p
        JOIN bird_detections d
          ON d.filename = p.filename AND d.is_selected = 1
         AND COALESCE(d.deleted, 0) = 0 AND COALESCE(d.edited, 0) = 0
        WHERE p.has_bird = 1
          AND (p.bird_species_cn IS NULL OR p.bird_species_cn = '')
        ORDER BY p.confidence DESC""").fetchall()
    no_row = conn.execute(f"""
        SELECT p.filename, p.rating, p.confidence, p.temp_jpeg_path
        {base}
          AND NOT EXISTS (SELECT 1 FROM bird_detections d
                          WHERE d.filename = p.filename)
        ORDER BY p.confidence DESC""").fetchall()
    # 有检测行但主鸟行均被删除（或无存活主鸟行）→ 人工去掉鸟，排除
    # 注意 deleted/edited 未编辑时为 NULL 而非 0，必须 COALESCE（2026-09-12
    # 教训：d.deleted=0 会把 NULL 行全部筛掉，导致有框桶整体漏跑）
    manual = conn.execute(f"""
        SELECT DISTINCT p.filename
        {base}
          AND EXISTS (SELECT 1 FROM bird_detections d
                      WHERE d.filename = p.filename)
          AND NOT EXISTS (SELECT 1 FROM bird_detections d
                          WHERE d.filename = p.filename
                            AND d.is_selected = 1
                            AND COALESCE(d.deleted, 0) = 0)
        ORDER BY p.filename""").fetchall()
    # 主鸟行被人工改种（edited=1）但 photos 行未同步 → 排除，不覆盖人工结果
    edited = conn.execute(f"""
        SELECT DISTINCT p.filename
        {base}
          AND EXISTS (SELECT 1 FROM bird_detections d
                      WHERE d.filename = p.filename
                        AND d.is_selected = 1
                        AND COALESCE(d.deleted, 0) = 0
                        AND d.edited = 1)
        ORDER BY p.filename""").fetchall()
    conn.close()
    return {
        "with_box": [dict(r) for r in with_box
                     if not r["filename"].endswith(VIDEO_COVER_SUFFIX)],
        "redetect": [dict(r) for r in no_row
                     if not r["filename"].endswith(VIDEO_COVER_SUFFIX)],
        "skipped_manual": [r["filename"] for r in manual
                           if not r["filename"].endswith(VIDEO_COVER_SUFFIX)],
        "skipped_edited": [r["filename"] for r in edited
                           if not r["filename"].endswith(VIDEO_COVER_SUFFIX)],
    }


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
    ap.add_argument("--redetect-conf", type=float, default=0.25,
                    help="无检测行照片的 YOLO 兜底检测地板（默认 0.25）")
    ap.add_argument("--report", default=None,
                    help="追加写入采纳清单的 markdown 路径（可选）")
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

    cand = collect_candidates(db_path)
    print(f"📁 {root}\n   有框待补种 {len(cand['with_box'])} 张，"
          f"无检测行待重检 {len(cand['redetect'])} 张"
          f"（采纳门槛 {args.threshold}%）；"
          f"人工去掉鸟跳过 {len(cand['skipped_manual'])} 张，"
          f"人工已改种跳过 {len(cand['skipped_edited'])} 张")
    if cand["skipped_manual"]:
        print(f"   ↳ 人工排除: {', '.join(cand['skipped_manual'][:10])}"
              f"{' ...' if len(cand['skipped_manual']) > 10 else ''}")
    if cand["skipped_edited"]:
        print(f"   ↳ 人工改种不覆盖: {', '.join(cand['skipped_edited'][:10])}"
              f"{' ...' if len(cand['skipped_edited']) > 10 else ''}")

    rows = cand["with_box"]
    adopted: List[Dict] = []
    still_low: List[Dict] = []
    failed = 0

    def _run_one(entry_prefix: str, row: Dict,
                 box: Tuple[int, int, int, int], det_conf: float,
                 source: str, idx: int, total: int) -> None:
        """单张识别 + 计数归桶（采纳/仍低/失败），打印进度行。"""
        nonlocal failed
        exif_path, img = load_source(root, entry_prefix,
                                     row.get("temp_jpeg_path"))
        if img is None:
            failed += 1
            print(f"  [{idx}/{total}] {entry_prefix}: 预览/RAW/本体均不可读，跳过")
            return
        # V5.9.2: 暗版双渲染——存在 <前缀>_dark.jpg（原始暗渲染）时裁同位
        # 框供重试对比；亮版首判低于采纳线才触发，无暗版自动回退伽马。
        # V5.9.2: dual-rendition — crop the co-registered box from
        # <prefix>_dark.jpg (original dark rendition) when present; the
        # retry fires only below the adoption line and falls back to
        # gamma without a dark rendition.
        dark_bgr, dark_box = None, None
        dark_sidecar = os.path.join(root, ".superpicky", "cache",
                                    "temp_preview", entry_prefix + "_dark.jpg")
        if os.path.exists(dark_sidecar):
            dark_bgr = read_bgr(dark_sidecar)
            if dark_bgr is not None:
                ih, iw = img.shape[:2]
                dh, dw = dark_bgr.shape[:2]
                x1, y1, x2, y2 = box
                sx, sy = dw / float(iw), dh / float(ih)
                dark_box = (max(0, int(x1 * sx)), max(0, int(y1 * sy)),
                            min(dw, int(x2 * sx)), min(dh, int(y2 * sy)))
        # V5.9: 暗框提亮重识别与主管线同线（--threshold 即采纳线）
        # V5.9: brightened retry shares the pipeline adoption line.
        result = classify(img, box, exif_path, args.country,
                          dark_retry_conf=float(args.threshold),
                          dark_bgr=dark_bgr, dark_box=dark_box)
        if not (result and result.get("success") and result.get("results")):
            still_low.append({"prefix": entry_prefix, "name": "（无结果）",
                              "conf": 0.0, "row": row, "source": source,
                              "det_conf": det_conf})
            print(f"  [{idx}/{total}] {entry_prefix}: 分类器无结果")
            return
        top = result["results"][0]
        # identify_bird 的 results[].confidence 已是 0-100 百分数，勿再 ×100
        conf = float(top.get("confidence") or 0.0)
        name = top.get("cn_name") or top.get("en_name") or "?"
        e = {"prefix": entry_prefix, "name": name, "conf": conf,
             "row": row, "top": top, "source": source, "det_conf": det_conf,
             "en": top.get("en_name"), "sci": top.get("scientific_name"),
             "rating": row.get("rating")}
        if conf >= args.threshold:
            adopted.append(e)
            print(f"  [{idx}/{total}] {entry_prefix}: ✅ {name} {conf:.0f}%"
                  f"（{row.get('rating')}★，{source}，检测conf {det_conf:.0%}）")
        else:
            still_low.append(e)
            print(f"  [{idx}/{total}] {entry_prefix}: 仍低于门槛 "
                  f"{name} {conf:.0f}%")

    total = len(rows) + len(cand["redetect"])
    for i, r in enumerate(rows, 1):
        prefix = r["filename"]
        box = (int(r["bbox_x"]), int(r["bbox_y"]),
               int(r["bbox_x"] + r["bbox_w"]), int(r["bbox_y"] + r["bbox_h"]))
        _run_one(prefix, r, box, float(r["confidence"] or 0.0),
                 "库内框", i, total)

    for j, r in enumerate(cand["redetect"], len(rows) + 1):
        prefix = r["filename"]
        exif_path, img = load_source(root, prefix, r.get("temp_jpeg_path"))
        if img is None:
            failed += 1
            print(f"  [{j}/{total}] {prefix}: 预览/RAW/本体均不可读，跳过")
            continue
        box, det_conf = detect_fallback_box(img, args.redetect_conf)
        if box is None:
            still_low.append({"prefix": prefix, "name": "（重检无鸟框）",
                              "conf": 0.0, "row": r, "source": "重检",
                              "det_conf": 0.0})
            print(f"  [{j}/{total}] {prefix}: 当前模型重检无鸟框")
            continue
        _run_one(prefix, r, box, det_conf, f"重检{det_conf:.0%}", j, total)

    print(f"\n📊 可采纳 {len(adopted)}/{total}；"
          f"仍低于 {args.threshold}% {len(still_low)} 张；"
          f"素材缺失跳过 {failed} 张")

    if args.report and adopted:
        try:
            head = not os.path.exists(args.report)
            with open(args.report, "a", encoding="utf-8") as f:
                if head:
                    f.write("# 未定种补种采纳清单（门槛 "
                            f"{args.threshold:.0f}%）\n\n"
                            "| 目录 | 文件 | 星级 | 鸟种 | 置信% | "
                            "来源 | 备注 |\n|---|---|---|---|---|---|---|\n")
                for e in adopted:
                    f.write(f"| {os.path.basename(root)} | {e['prefix']} "
                            f"| {e.get('rating')} | {e['name']} "
                            f"| {e['conf']:.0f} | {e['source']} "
                            f"| {'40-50 需复核' if e['conf'] < 50 else ''} |\n")
        except Exception as exc:
            print(f"  ⚠️ 报告写入失败: {exc}")

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
            det_id = e["row"].get("det_id")
            if det_id is not None:
                cur.execute(
                    "UPDATE bird_detections SET species_cn=?, species_en=?,"
                    " scientific_name=?, species_confidence=?, class_id=?,"
                    " gbif_rarity_100=?, china_protection_level=?,"
                    " updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (top.get("cn_name"), top.get("en_name"),
                     top.get("scientific_name"), e["conf"],
                     top.get("class_id"), top.get("gbif_rarity_100"),
                     top.get("china_protection_level"), det_id))
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
            idx = e["row"].get("bird_index")
            dets = data.get("detections") or []
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
