#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidecar JSON 导出模块（per-photo rich data sidecar）

批处理结束时把 report.db 的 photos 行 + bird_detections 行导出为每照片
一份 JSON，存放于 `<照片目录>/.superpicky/meta/<前缀>.json`。这是给外部
消费（用户的照片管理网站、人工审核工具）的对外数据契约层：

- RAW/JPG 原文件零接触（非破坏工作流的数据出口）；
- 增量导出：JSON 内记录 `_export_stamp`（photo 与其全部 detection 的
  updated_at 最大值），未变化的照片跳过重写；
- 人工编辑（edits 数组，二期编辑工具写入）在重导出时原样保留；
- 原子写：先写 .tmp 再 os.replace（SMB 网络盘安全）；
- UTF-8 无 BOM，中文物种名直接可读。

Exports per-photo JSON sidecars from report.db at batch end. The JSON
is the external contract for downstream consumers (management website,
review tools). Incremental, edit-preserving, atomic, UTF-8.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

SCHEMA_VERSION = 1
META_SUBDIR = os.path.join(".superpicky", "meta")


def _sidecar_path(directory: str, prefix: str) -> str:
    """返回某照片的 sidecar JSON 路径。/ Sidecar path for one photo prefix."""
    return os.path.join(directory, META_SUBDIR, f"{prefix}.json")


def _export_stamp(photo_row: dict, detection_rows: List[dict]) -> str:
    """
    计算导出戳：photo 行与全部 detection 行 updated_at 的最大值。

    用于增量判断——任何一个字段更新都会使戳变化，触发重写。

    Compute the export stamp (max updated_at of photo + detections)
    for incremental re-export decisions.
    """
    stamps = [photo_row.get("updated_at") or ""]
    for det in detection_rows:
        stamps.append(det.get("updated_at") or "")
    return max(stamps)


def _build_photo_section(photo_row: dict) -> dict:
    """
    从 photos 行提取照片基础信息（EXIF 摘要 + GPS + 文件信息）。

    文件大小/mtime 在文件存在时实时读取；路径记录相对照片目录的
    相对路径（跨机器可移植），取不到时回退原始值。

    Build the "photo" section from a photos row: EXIF summary, GPS and
    file info (size/mtime read live when the file exists).
    """
    directory = photo_row.get("_directory") or ""
    rel_path = None
    original_path = photo_row.get("original_path")
    if original_path:
        try:
            rel_path = os.path.relpath(original_path, directory)
        except ValueError:
            rel_path = original_path

    size_bytes = None
    mtime = None
    if original_path and os.path.exists(original_path):
        try:
            size_bytes = os.path.getsize(original_path)
            mtime_iso = os.path.getmtime(original_path)
            import datetime
            mtime = datetime.datetime.fromtimestamp(
                mtime_iso).isoformat(timespec="seconds")
        except OSError:
            pass

    return {
        "filename": photo_row.get("filename"),
        "relative_path": rel_path,
        "size_bytes": size_bytes,
        "mtime": mtime,
        "camera_model": photo_row.get("camera_model"),
        "lens_model": photo_row.get("lens_model"),
        "iso": photo_row.get("iso"),
        "focal_length": photo_row.get("focal_length"),
        "date_time_original": photo_row.get("date_time_original"),
        "gps": {
            "latitude": photo_row.get("gps_latitude"),
            "longitude": photo_row.get("gps_longitude"),
        },
    }


def _build_processing_section(photo_row: dict) -> dict:
    """
    从 photos 行提取处理结果摘要（星级/对焦/质量指标/连拍）。

    Build the "processing" section: rating, focus, quality metrics, burst.
    """
    try:
        from constants import APP_VERSION
    except Exception:
        APP_VERSION = "unknown"
    return {
        "app_version": APP_VERSION,
        "db_updated_at": photo_row.get("updated_at"),
        "has_bird": bool(photo_row.get("has_bird")),
        "rating": photo_row.get("rating"),
        "picked": bool(photo_row.get("picked")) if photo_row.get("picked") is not None else False,
        "focus_status": photo_row.get("focus_status"),
        "is_flying": bool(photo_row.get("is_flying")),
        "main_yolo_confidence": photo_row.get("confidence"),
        "quality": {
            "head_sharpness": photo_row.get("head_sharp"),
            "topiq": photo_row.get("nima_score"),
            "adj_sharpness": photo_row.get("adj_sharpness"),
            "adj_topiq": photo_row.get("adj_topiq"),
        },
        "species_main": {
            "cn": photo_row.get("bird_species_cn"),
            "en": photo_row.get("bird_species_en"),
            "confidence": photo_row.get("birdid_confidence"),
        },
        "burst_id": photo_row.get("burst_id"),
        "burst_position": photo_row.get("burst_position"),
        "gbif_rarity_100": photo_row.get("gbif_rarity_100"),
    }


def _build_detection_sections(detection_rows: List[dict]) -> List[dict]:
    """
    把 bird_detections 行转换为 detections 数组（坐标已是原图像素）。

    mask_polygon 在库里是 JSON 字符串，此处解析回点阵数组，便于网站
    直接消费；解析失败时置 None（不中断导出）。

    Convert bird_detections rows to the detections array; the stored
    polygon JSON string is parsed back into a point array.
    """
    sections = []
    for row in sorted(detection_rows, key=lambda r: r.get("bird_index") or 0):
        polygon = None
        raw_poly = row.get("mask_polygon")
        if raw_poly:
            try:
                polygon = json.loads(raw_poly)
            except (TypeError, ValueError):
                polygon = None
        has_species = bool(row.get("species_cn") or row.get("species_en"))
        sections.append({
            "index": row.get("bird_index"),
            "is_selected": bool(row.get("is_selected")),
            "bbox": [row.get("bbox_x"), row.get("bbox_y"),
                     row.get("bbox_w"), row.get("bbox_h")],
            "polygon": polygon,
            "area_ratio": row.get("area_ratio"),
            "yolo_conf": row.get("yolo_conf"),
            "crop_sharpness": row.get("crop_sharpness"),
            "species": {
                "cn": row.get("species_cn"),
                "en": row.get("species_en"),
                "scientific": row.get("scientific_name"),
                "confidence": row.get("species_confidence"),
                "class_id": row.get("class_id"),
                "gbif_rarity_100": row.get("gbif_rarity_100"),
            } if has_species else None,
            "notable": bool(row.get("notable")),
            "notable_reason": row.get("notable_reason"),
            "edited": bool(row.get("edited")),
        })
    return sections


def _load_existing_edits(path: str) -> Optional[List[dict]]:
    """读取现有 sidecar 的 edits 数组（人工编辑历史永不丢失）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        edits = data.get("edits")
        return edits if isinstance(edits, list) else None
    except (OSError, ValueError):
        return None


def _needs_rewrite(path: str, stamp: str) -> bool:
    """
    增量判断：文件不存在、解析失败或导出戳变化时才需要重写。

    Incremental check: rewrite only when missing, corrupt, or the
    export stamp changed.
    """
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return existing.get("_export_stamp") != stamp
    except (OSError, ValueError):
        return True


def _atomic_write_json(path: str, payload: dict) -> None:
    """
    原子写 JSON：先写同目录 .tmp 再 os.replace。

    SMB/NAS 场景下中途断电或网络闪断只会留下完整旧文件或完整新文件，
    不会出现半截 JSON。UTF-8 无 BOM，ensure_ascii=False 保留中文可读性。

    Atomic JSON write via tmp + os.replace; UTF-8 without BOM.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def export_directory_sidecars(report_db, directory: str,
                              log=print) -> int:
    """
    把一个照片目录的 report.db 全量导出为 per-photo sidecar JSON。

    参数:
    report_db: 已打开的 ReportDB 实例（photos + bird_detections）
    directory (str): 照片目录（sidecar 写入其 .superpicky/meta/ 下）
    log: 日志函数（默认 print）

    返回:
    int: 本次实际写入（含跳过外的重写）的 JSON 数量

    Export every photo in the report DB as an incremental per-photo JSON
    sidecar. Returns the number of files written this run.
    """
    if report_db is None:
        return 0
    photos = report_db.get_all_photos()
    detections = report_db.get_all_detections()
    detections_by_filename: Dict[str, List[dict]] = {}
    for det in detections:
        detections_by_filename.setdefault(det.get("filename"), []).append(det)

    written = 0
    for photo_row in photos:
        prefix = photo_row.get("filename")
        if not prefix:
            continue
        det_rows = detections_by_filename.get(prefix, [])
        stamp = _export_stamp(photo_row, det_rows)
        path = _sidecar_path(directory, prefix)
        if not _needs_rewrite(path, stamp):
            continue

        photo_row = dict(photo_row)
        photo_row["_directory"] = directory
        payload = {
            "schema_version": SCHEMA_VERSION,
            "_export_stamp": stamp,
            "photo": _build_photo_section(photo_row),
            "processing": _build_processing_section(photo_row),
            "detections": _build_detection_sections(det_rows),
            # 人工编辑历史：重导出时从现有文件回带（二期编辑工具写入）
            "edits": _load_existing_edits(path) or [],
        }
        try:
            _atomic_write_json(path, payload)
            written += 1
        except OSError as e:
            log(f"  ⚠️ Sidecar write failed [{prefix}]: {e}")
    return written
