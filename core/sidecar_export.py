#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidecar JSON 导出模块（per-photo rich data sidecar）

批处理结束时把 report.db 的 photos 行 + bird_detections 行导出为每照片
一份 JSON，存放于 `<照片目录>/.superpicky/meta/<前缀>.json`。这是给外部
消费（用户的照片管理网站、人工审核工具）的对外数据契约层：

- RAW/JPG 原文件零接触（非破坏工作流的数据出口）；
- 增量导出：JSON 内记录 `_export_stamp`（photo 与其全部 detection 的
  updated_at 最大值），未变化的照片跳过重写；旧格式（photo.library_path
  缺失）的存量 JSON 也会被重写升级（自愈迁移，不依赖导出戳变化）；
- 人工编辑（edits 数组，二期编辑工具写入）在重导出时原样保留；
- 原子写：先写 .tmp 再 os.replace（SMB 网络盘安全）；
- UTF-8 无 BOM，中文物种名直接可读；
- V5.3 契约增强：`photo.filename` 带扩展名；`photo.library_path` 为照片
  相对库根（处理目录）的当前真实位置（整理移动/连拍重组后仍有效）；
  `photo.preview_path` 为可直接显示的 JPEG 路径（RAW 预览缓存或伴随
  JPG）。下游消费方（BirdIndex 等）应以 library_path 定位照片文件，
  不再从 sidecar 文件名推导。sidecar 文件名本身维持 `<前缀>.json`。

Exports per-photo JSON sidecars from report.db at batch end. The JSON
is the external contract for downstream consumers (management website,
review tools). Incremental, edit-preserving, atomic, UTF-8. V5.3 adds
an extension-bearing filename plus library_path/preview_path so
consumers can locate photos after organizing moves.
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


def _normalize_rel(value: Optional[str]) -> Optional[str]:
    """
    把库内相对路径规范为正斜杠（跨平台契约：Windows 侧 DB 存反斜杠，
    JSON 统一输出 "/"，macOS/Linux 天然一致）。

    参数:
    value (str): DB 中的相对路径（可能是 \\ 或 / 分隔），可为 None

    返回:
    Optional[str]: 正斜杠路径；输入 None 返回 None

    Normalize an in-library relative path to forward slashes for the
    cross-platform JSON contract.
    """
    if not value:
        return None
    return value.replace(os.sep, "/")


def _is_jpeg_ext(path: str) -> bool:
    """判断路径扩展名是否 JPEG。/ True when the path has a JPEG extension."""
    return path.lower().endswith((".jpg", ".jpeg"))


def _build_photo_section(photo_row: dict) -> dict:
    """
    从 photos 行提取照片基础信息（EXIF 摘要 + GPS + 文件信息 + 定位契约字段）。

    V5.3 定位契约（下游 BirdIndex 依赖）：
    - filename：带扩展名的照片文件名（取 original_path 的 basename；
      original_path 写库时优先 RAW，见 photo_processor 阶段3 路径记录）；
    - library_path：照片相对库根（处理目录）的**当前真实位置**，
      来源 current_path（整理移动/连拍重组后由处理流程更新），
      为空或仍指向 .superpicky/ 缓存（未整理库的临时分析 JPEG）时
      回退 original_path；指向伴随 JPG 而同目录存在 RAW（历史 JPG
      覆盖数据）时规范化为 RAW（主文件语义）；
    - relative_path：首次处理时的位置（original_path 原样，存档语义；
      旧版对相对路径再跑 relpath 会按进程 cwd 解析产生 `..\\..\\` 畸形值，
      已修复为直接输出）；
    - preview_path：可直接显示的 JPEG（RAW 预览缓存或伴随 JPG），
      可为 None；
    - size_bytes/mtime：按 library_path 相对 directory 解析到磁盘读取，
      文件不存在（如 NAS 离线）时为 None。

    参数:
    photo_row (dict): photos 表行（含 _directory 注入的库根目录）

    返回:
    dict: photo 段字典

    Build the "photo" section: EXIF summary, GPS, file info and the
    V5.3 locating contract fields (filename with extension, current
    library_path, archive relative_path, preview_path).
    """
    directory = photo_row.get("_directory") or ""
    original_path = photo_row.get("original_path")
    current_path = photo_row.get("current_path")

    # 文件名带扩展名：original_path 含扩展名且优先 RAW；缺失时回退
    # 库主键前缀（无扩展名，仅极端情况下 original_path 为空时出现）
    if original_path:
        filename = os.path.basename(original_path)
    else:
        filename = photo_row.get("filename")

    # 当前真实位置优先 current_path（整理后仍有效），回退 original_path。
    # 防御：ai_model 初始入库时 current_path 指向 YOLO 分析用的临时 JPEG
    # （RAW 转换产物，位于 .superpicky/cache/ 下）；未经阶段5 整理覆盖的库
    # 该值会一直停留在缓存目录，不是照片本体的位置——此时回退 original_path
    # （其写入时已排除 cache 路径且优先 RAW）。
    # Prefer current_path (kept fresh by organizing moves); fall back to
    # original_path. Guard: on unorganized libraries current_path still
    # points at the temp analysis JPEG under .superpicky/cache/, which is
    # not the photo itself — original_path is cache-free and RAW-first.
    library_rel = current_path or original_path
    if library_rel and current_path:
        rel_norm = _normalize_rel(current_path) or ""
        if rel_norm.startswith(".superpicky/") or "/.superpicky/" in rel_norm:
            library_rel = original_path
        elif (_is_jpeg_ext(current_path) and original_path
              and not _is_jpeg_ext(original_path)):
            # 历史数据规范化（V5.3 前旧整理循环的 JPG 覆盖问题）：
            # current_path 指向伴随 JPG 而 original_path 是 RAW 时，主文件
            # 应为 RAW——用 current_path 的目录 + original_path 的扩展名推导
            # 候选，磁盘确认存在才采用（RAW 已被单独删除时保留 JPG 原值）。
            # Normalize legacy rows whose current_path was overwritten by
            # the companion JPG: prefer the same-folder RAW (extension from
            # the RAW-first original_path) when it exists on disk.
            candidate = os.path.join(
                os.path.dirname(current_path),
                os.path.splitext(os.path.basename(current_path))[0]
                + os.path.splitext(original_path)[1])
            if directory and os.path.exists(os.path.join(directory, candidate)):
                library_rel = candidate

    size_bytes = None
    mtime = None
    if library_rel and directory:
        abs_path = os.path.join(directory, library_rel)
        try:
            if os.path.exists(abs_path):
                size_bytes = os.path.getsize(abs_path)
                mtime_iso = os.path.getmtime(abs_path)
                import datetime
                mtime = datetime.datetime.fromtimestamp(
                    mtime_iso).isoformat(timespec="seconds")
        except OSError:
            pass

    return {
        "filename": filename,
        "library_path": _normalize_rel(library_rel),
        "relative_path": _normalize_rel(original_path),
        "preview_path": _normalize_rel(photo_row.get("temp_jpeg_path")),
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
        "focus_x": photo_row.get("focus_x"),
        "focus_y": photo_row.get("focus_y"),
        "exposure_status": photo_row.get("exposure_status"),
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
        # V5.2 物种召回：含本批从未当主鸟的鸟种（与星级/精选正交）
        "notable": bool(photo_row.get("notable")),
    }


def _build_detection_sections(detection_rows: List[dict],
                              existing: Optional[dict] = None) -> List[dict]:
    """
    把 bird_detections 行转换为 detections 数组（坐标已是原图像素）。

    polygon 不导出（用户确认不需要，节省空间）——库里仍有
    mask_polygon 列，需要时可重导出或用 YOLO 确定性重算。

    existing 为当前已存在的 sidecar JSON（可 None）。人工编辑优先：
    - existing 中同 index 的 detection 若 edited=true，其 species/edited
      以 JSON 为准（数据库是机器结果，人工改动不能被重导出覆盖）；
    - 若被人工软删除（deleted=true），保留 deleted 标记。

    Convert bird_detections rows to the detections array. Manual edits
    in an existing sidecar win over DB values on re-export.
    """
    existing_dets = {}
    if existing:
        for det in existing.get("detections") or []:
            if isinstance(det, dict) and det.get("index") is not None:
                existing_dets[det["index"]] = det

    sections = []
    for row in sorted(detection_rows, key=lambda r: r.get("bird_index") or 0):
        index = row.get("bird_index")
        prev = existing_dets.get(index) or {}
        has_species = bool(row.get("species_cn") or row.get("species_en"))
        section = {
            "index": index,
            "is_selected": bool(row.get("is_selected")),
            "bbox": [row.get("bbox_x"), row.get("bbox_y"),
                     row.get("bbox_w"), row.get("bbox_h")],
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
        }
        # 人工编辑优先 / manual edits win over DB re-export
        if prev.get("edited"):
            section["species"] = prev.get("species")
            section["edited"] = True
        if prev.get("deleted"):
            section["deleted"] = True
        sections.append(section)
    return sections


def _load_existing_payload(path: str) -> Optional[dict]:
    """
    读取现有 sidecar JSON 全文（人工编辑合并的基准）；损坏返回 None。

    Load the existing sidecar JSON for edit-preserving re-export.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _needs_rewrite(path: str, stamp: str) -> bool:
    """
    增量判断：文件不存在、解析失败、导出戳变化，或**旧格式自愈**时重写。

    自愈条件：现有 JSON 的 photo.library_path 缺失（V5.3 之前的存量
    sidecar）。存量库的 DB updated_at 未变化时导出戳不变，若只比对戳，
    旧文件永远不会升级到新格式，故单独判断字段存在性。

    Incremental check: rewrite when missing, corrupt, stamp changed, or
    when the file predates the V5.3 contract (no photo.library_path) —
    the stamp alone would never trigger those upgrades.
    """
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("_export_stamp") != stamp:
            return True
        photo_section = existing.get("photo")
        if not isinstance(photo_section, dict) or "library_path" not in photo_section:
            return True
        return False
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
        existing = _load_existing_payload(path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "_export_stamp": stamp,
            "photo": _build_photo_section(photo_row),
            "processing": _build_processing_section(photo_row),
            "detections": _build_detection_sections(det_rows, existing),
            # 人工编辑历史与主鸟种选择：重导出时从现有文件回带
            # （编辑工具写入；机器重导出不覆盖人工结果）
            "edits": (existing or {}).get("edits") or [],
        }
        if existing and existing.get("main_species"):
            payload["main_species"] = existing["main_species"]
        try:
            _atomic_write_json(path, payload)
            written += 1
        except OSError as e:
            log(f"  ⚠️ Sidecar write failed [{prefix}]: {e}")
    return written
