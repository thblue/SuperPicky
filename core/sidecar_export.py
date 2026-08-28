#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidecar JSON 导出模块（per-photo rich data sidecar）

批处理结束时把 report.db 的 photos 行 + bird_detections 行导出为每照片
一份 JSON，存放于 `<照片目录>/.superpicky/meta/<前缀>.json`。这是给外部
消费（用户的照片管理网站、人工审核工具）的对外数据契约层：

- RAW/JPG 原文件零接触（非破坏工作流的数据出口）；
- 增量导出：JSON 内记录 `_export_stamp`（photo 与全部 detection 业务
  字段的内容哈希，V5.4 起），未变化的照片跳过重写；旧格式
  （photo.library_path 缺失）的存量 JSON 也会被重写升级（自愈迁移，
  不依赖导出戳变化）；
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

import hashlib
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
    计算导出戳：photo 行 + 全部 detection 行的**业务字段内容哈希**。

    V5.4 起不再用 updated_at 最大值——它只有秒级精度，"编辑器保存 →
    立即重导出"这类同秒链路（软删框/主鸟勾选/召回重标）戳不变，会漏
    重写。内容哈希任何字段变化（含同秒变化）必然变化；纯 updated_at
    变化（如召回重跑碰了值未变的行）不触发重写，语义更准。

    排除字段：id / created_at / updated_at（自增与时间戳非业务内容）；
    mask_polygon（不导出到 JSON，且全表读取时可瘦身跳过——两条路径
    ——单照片全列 / 全表瘦身——必须哈希一致）。

    Compute the export stamp as a content hash over the business fields
    of the photo row + its detection rows. Timestamps, row ids and the
    non-exported mask_polygon are excluded so full-table (slim) and
    single-photo (full) fetch paths hash identically.
    """
    volatile = ("id", "created_at", "updated_at", "mask_polygon")
    payload = {
        "photo": {k: v for k, v in photo_row.items() if k not in volatile},
        "detections": [
            {k: v for k, v in d.items() if k not in volatile}
            for d in sorted(detection_rows,
                            key=lambda r: r.get("bird_index") or 0)
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


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
        # 软删除：JSON 已有标记或 DB 行 deleted=1（V5.4 批量清理）都保留
        # Soft-delete: keep the flag from JSON or from the DB row.
        if prev.get("deleted") or row.get("deleted"):
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
                              log=print,
                              only_filenames: Optional[List[str]] = None) -> int:
    """
    把一个照片目录的 report.db 全量导出为 per-photo sidecar JSON。

    参数:
    report_db: 已打开的 ReportDB 实例（photos + bird_detections）
    directory (str): 照片目录（sidecar 写入其 .superpicky/meta/ 下）
    log: 日志函数（默认 print）
    only_filenames (Optional[List[str]]): 只导出这些照片（V5.4 编辑器
        保存路径——单照片秒级落盘，全目录重算在后台做）

    返回:
        int: 本次实际写入（含跳过外的重写）的 JSON 数量

    V5.5: 无鸟照片（has_bird=0）不导出 sidecar——BirdIndex 网站数据源只需
    有鸟照片；已存在的历史 JSON 幂等删除，保证重跑即收敛。DB 行本身不受影响
    （统计与浏览器「无鸟」筛选仍依赖 photos 表）。

    Export every photo (or only_filenames subset) as an incremental
    per-photo JSON sidecar. No-bird photos (has_bird=0) are skipped and
    their stale JSONs removed idempotently. Returns files written.
    """
    if report_db is None:
        return 0
    if only_filenames is not None and len(only_filenames) <= 50:
        # 小子集（编辑器保存路径）：逐照片取行，不做全表读取——NAS 上
        # 单照片导出保持秒级。_export_stamp 已排除 polygon，单照片全列
        # 读取与全表瘦身读取的哈希一致。
        # Small subset (editor-save path): fetch per photo instead of a
        # full-table scan; stamps stay identical across both paths.
        photos = [p for p in (report_db.get_photo(f)
                              for f in only_filenames) if p]
        detections = []
        for f in only_filenames:
            detections.extend(report_db.get_detections(f))
    else:
        photos = report_db.get_all_photos()
        # 全表瘦身读取：导出不写 polygon，NAS 上省掉每行数百字节传输
        # Slim full-table fetch: export never writes polygon anyway.
        detections = report_db.get_all_detections(include_polygon=False)
    detections_by_filename: Dict[str, List[dict]] = {}
    for det in detections:
        detections_by_filename.setdefault(det.get("filename"), []).append(det)

    # V5.4 导出戳缓存：DB 内容未变化（且文件在）的照片直接跳过，
    # 不再打开每个 JSON 比对——NAS 上 1240 次文件读取是全量导出的
    # 主要成本。缓存缺失/不匹配时回退到读文件并回填。
    # Export-stamp cache: skip photos whose DB content is unchanged
    # without opening their JSON files at all.
    try:
        cached_stamps = report_db.get_export_stamps()
    except Exception:
        cached_stamps = {}

    written = 0
    new_stamps: Dict[str, str] = {}
    for photo_row in photos:
        prefix = photo_row.get("filename")
        if not prefix:
            continue
        if only_filenames is not None and prefix not in only_filenames:
            continue
        # V5.5: 无鸟照片不导出 sidecar；历史残留 JSON 幂等删除（重跑即收敛）
        # V5.5: skip no-bird photos; remove any stale JSON idempotently.
        if not photo_row.get("has_bird"):
            stale_path = _sidecar_path(directory, prefix)
            try:
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            except OSError:
                pass
            continue
        det_rows = detections_by_filename.get(prefix, [])
        stamp = _export_stamp(photo_row, det_rows)
        path = _sidecar_path(directory, prefix)
        if (cached_stamps.get(prefix) == stamp
                and os.path.exists(path)):
            continue  # DB 未变化 → 文件不必动（人工 JSON 编辑得以保留）
        if not _needs_rewrite(path, stamp):
            new_stamps[prefix] = stamp  # 回填缓存（曾走读文件路径）
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
            new_stamps[prefix] = stamp
        except OSError as e:
            log(f"  ⚠️ Sidecar write failed [{prefix}]: {e}")
    # 写入成功的戳回填缓存（下次同内容直接跳过，不打开文件）
    if new_stamps:
        try:
            report_db.upsert_export_stamps(new_stamps)
        except Exception as e:  # noqa: BLE001（缓存失败只影响下次性能）
            log(f"  ⚠️ Export-stamp cache write failed: {e}")
    return written


def mark_species_deleted_in_sidecar(
    directory: str,
    filename: str,
    species_cn: Optional[str] = None,
    species_en: Optional[str] = None,
    scientific_name: Optional[str] = None,
) -> int:
    """
    在一张照片的 sidecar JSON 里软删除某鸟种的全部检测框（V5.4 批量
    清理入口，与多鸟编辑器的删框同格式：det.deleted=true + edits 记录）。

    名字匹配：中文名/英文名/学名任一非空相等。文件不存在或无匹配返回 0；
    全部已删（幂等重入）也返回 0。

    参数:
    directory (str): 照片目录
    filename (str): 照片前缀（无扩展名）
    species_cn / species_en / scientific_name (Optional[str]): 鸟种名

    返回:
    int: 本次新标记删除的检测框数

    Soft-delete all detections of one species in a photo's sidecar JSON
    (same format as the multi-bird editor's box deletion).
    """
    import datetime

    path = _sidecar_path(directory, filename)
    data = _load_existing_payload(path)
    if not data:
        return 0
    dets = data.get("detections")
    if not isinstance(dets, list):
        return 0

    def _match(species: Optional[dict]) -> bool:
        if not isinstance(species, dict):
            return False
        pairs = ((species.get("cn"), species_cn),
                 (species.get("en"), species_en),
                 (species.get("scientific"), scientific_name))
        return any(a and b and str(a).strip() == str(b).strip()
                   for a, b in pairs)

    removed = 0
    for det in dets:
        if not isinstance(det, dict) or det.get("deleted"):
            continue
        if _match(det.get("species")):
            det["deleted"] = True
            data.setdefault("edits", []).append({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "actor": "human",
                "action": "bbox_deleted",
                "bird_index": det.get("index"),
                "old": None,
                "new": None,
            })
            removed += 1

    # 主鸟种数组同步清理：整批删除（soft_delete_species_everywhere）时，
    # 顶层 main_species 里的同名项一并移除，否则错误的「主鸟」名会残留
    # 在 JSON 里继续导出到 BirdIndex。仅当确有框被删时才动（与 DB 侧
    # 「photos 主鸟种只在该名有检测行时清空」的粒度保持一致）。
    # Also drop the same name from the top-level main_species array when
    # boxes were deleted, so the wrong "main bird" name does not linger
    # in the JSON exports.
    main = data.get("main_species")
    if removed and isinstance(main, list):
        names = {v for v in (species_cn, species_en, scientific_name)
                 if isinstance(v, str) and v.strip()}
        if names:
            new_main = [m for m in main
                        if not (isinstance(m, str) and m.strip() in names)]
            if new_main != main:
                data["main_species"] = new_main

    if removed:
        try:
            _atomic_write_json(path, data)
        except OSError:
            return 0
    return removed


def rename_species_in_sidecar(
    directory: str,
    filename: str,
    new_cn: Optional[str] = None,
    new_en: Optional[str] = None,
    new_sci: Optional[str] = None,
    old_cn: Optional[str] = None,
    old_en: Optional[str] = None,
    old_sci: Optional[str] = None,
) -> int:
    """
    在一张照片的 sidecar JSON 里把某鸟种的检测框批量改为另一个鸟种。

    与 report_db.rename_species_everywhere 配套（整批改种的 sidecar 同步）：
    - 检测框：species 匹配旧名（cn/en/scientific 任一非空相等）的未删框
      改写为新名（未提供的新名维度写 None）；
    - main_species：数组里的旧中文名替换为新中文名（若有提供）；
    - edits 追加 species_renamed 记录（actor=human，含旧/新名）。
    文件不存在、无匹配或全部已改（幂等重入）时返回 0。

    参数:
    directory (str): 照片目录
    filename (str): 照片前缀（无扩展名）
    new_cn / new_en / new_sci (Optional[str]): 新鸟种名
    old_cn / old_en / old_sci (Optional[str]): 旧鸟种名（匹配条件）

    返回:
    int: 本次改写的检测框数

    Batch-rename one species' detections in a photo's sidecar JSON to
    another species, paired with report_db.rename_species_everywhere.
    """
    import datetime

    path = _sidecar_path(directory, filename)
    data = _load_existing_payload(path)
    if not data:
        return 0
    dets = data.get("detections")
    if not isinstance(dets, list):
        return 0

    def _match(species: Optional[dict]) -> bool:
        if not isinstance(species, dict):
            return False
        pairs = ((species.get("cn"), old_cn),
                 (species.get("en"), old_en),
                 (species.get("scientific"), old_sci))
        return any(a and b and str(a).strip() == str(b).strip()
                   for a, b in pairs)

    old_names = [v for v in (old_cn, old_en, old_sci)
                 if isinstance(v, str) and v.strip()]

    renamed = 0
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    for det in dets:
        if not isinstance(det, dict) or det.get("deleted"):
            continue
        if _match(det.get("species")):
            det["species"] = {"cn": new_cn, "en": new_en,
                              "scientific": new_sci}
            data.setdefault("edits", []).append({
                "timestamp": stamp,
                "actor": "human",
                "action": "species_renamed",
                "bird_index": det.get("index"),
                "old": {"cn": old_cn, "en": old_en, "scientific": old_sci},
                "new": {"cn": new_cn, "en": new_en, "scientific": new_sci},
            })
            renamed += 1

    # main_species 里的旧名替换为新名（数组存显示名，通常为中文名）
    # Replace old names in the top-level main_species array.
    main = data.get("main_species")
    if renamed and isinstance(main, list) and new_cn:
        old_set = {v for v in (old_cn, old_en, old_sci)
                   if isinstance(v, str) and v.strip()}
        new_main = [new_cn if (isinstance(m, str) and m.strip() in old_set)
                    else m for m in main]
        if new_main != main:
            data["main_species"] = new_main

    if renamed:
        try:
            _atomic_write_json(path, data)
        except OSError:
            return 0
    return renamed
