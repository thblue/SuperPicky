#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photo_purge — 照片「彻底删除」执行器（结果浏览器右键菜单后端）

与回收站式删除（ui/results_browser_window._move_to_trash，可恢复）不同，
本模块做**不可恢复**的删除，且把一张照片在磁盘与库里的全部痕迹一次清干净：

  1. 主文件（RAW/JPG，current_path 优先、original_path 兜底）
  2. RAW+JPG 成对的伴生 JPG（与主文件同目录同名）
  3. 预览图 temp_jpeg_path（.superpicky/cache 预览或成对 JPG）
  4. 调试图 yolo_debug_path / debug_crop_path
  5. V-cover 伴生视频（封面删而视频留会成为无记录孤儿）
  6. sidecar JSON（<照片目录>/.superpicky/meta/<前缀>.json）
  7. report.db 记录（photos + bird_detections，经 ReportDB/MergedReportDB
     的 delete_photo，V5.9 起同事务级联删检测框）

适用场景：NAS（UNC 路径）上系统回收站不可用/不可靠，用户需要从 0 星/
无鸟 pile 里批量彻底清除废片。

⚠️ 不可逆：调用方（浏览器 UI）必须先弹出明确告知「永久删除、不可恢复」
的确认对话框；本模块不做二次确认，也不提供 dry-run 参数——需要预演时
只调用 collect_purge_files 查看清单即可（纯函数，无副作用）。

photo_purge — permanent-delete executor behind the results browser's
right-click "Delete Permanently" action.

Unlike the trash-based delete (recoverable), this module irreversibly
removes every trace of a photo: main file, paired JPEG sibling, preview,
debug images, companion video, sidecar JSON, and the report.db record
(photos + bird_detections, cascaded by delete_photo since V5.9). Built
for NAS (UNC) libraries where the OS recycle bin is unavailable.

⚠️ Irreversible: the UI caller must show an explicit confirmation dialog
first. This module never re-confirms and has no dry-run flag — call
collect_purge_files (pure, side-effect free) to preview the manifest.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from tools.file_utils import sibling_jpeg

# 浏览器照片字典里指向磁盘文件的键（不含 sidecar，sidecar 由前缀推导）。
# Photo dict keys that point at files on disk (sidecar derived from prefix).
_FILE_KEYS = (
    "current_path",
    "original_path",
    "temp_jpeg_path",
    "debug_crop_path",
    "yolo_debug_path",
)


def _normalize(photo: dict, path: Optional[str]) -> Optional[str]:
    """相对路径按 _base_dir 归一为绝对路径；空值原样返回。

    Resolve a relative path against the photo's base directory;
    pass None/empty through unchanged.
    """
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    base_dir = photo.get("_base_dir") or ""
    if not base_dir:
        return None
    return os.path.normpath(os.path.join(base_dir, path))


def _sidecar_path(photo: dict) -> Optional[str]:
    """返回照片的 sidecar JSON 绝对路径（可能不存在，调用方自行判断）。

    Sidecar JSON path for the photo (may not exist; caller checks).
    """
    filename = photo.get("filename") or ""
    base_dir = photo.get("_base_dir") or ""
    if not filename or not base_dir:
        return None
    prefix = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(base_dir, ".superpicky", "meta", f"{prefix}.json")


def _bright_crop_path(photo: dict) -> Optional[str]:
    """返回 V5.9 暗框提亮重识别的亮框图路径（可能不存在，调用方判断）。

    crop_debug 里 <前缀>_bright.jpg 由 photo_processor.apply_birdid_result
    在提亮重识别命中后写入，不占用任何 DB 列——彻底删除时按前缀推导，
    防止缓存目录残留孤儿。

    Path of the V5.9 brightened-crop review image (may not exist).
    crop_debug's <prefix>_bright.jpg is written by apply_birdid_result
    after a winning brightened retry and referenced by no DB column;
    derive it from the prefix so permanent delete leaves no orphans.
    """
    filename = photo.get("filename") or ""
    base_dir = photo.get("_base_dir") or ""
    if not filename or not base_dir:
        return None
    prefix = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(base_dir, ".superpicky", "cache", "crop_debug",
                        f"{prefix}_bright.jpg")


def _dark_preview_path(photo: dict) -> Optional[str]:
    """返回 V5.9.2 原始暗渲染伴随缓存路径（可能不存在，调用方判断）。

    暗片 RAW 级提亮时原始内嵌渲染被另存为 temp_preview/<前缀>_dark.jpg
    （见 tools/find_bird_util._dark_sidecar_path），无 DB 列——彻底删除时
    按前缀推导，防缓存残留。

    Path of the V5.9.2 original dark-rendition sidecar (may not exist).
    Preserved as temp_preview/<prefix>_dark.jpg when a dark preview is
    RAW-grade brightened; referenced by no DB column, so derive it from
    the prefix to leave no orphans behind.
    """
    filename = photo.get("filename") or ""
    base_dir = photo.get("_base_dir") or ""
    if not filename or not base_dir:
        return None
    prefix = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(base_dir, ".superpicky", "cache", "temp_preview",
                        f"{prefix}_dark.jpg")


def collect_purge_files(photo: dict) -> List[str]:
    """
    计算彻底删除一张照片要删掉的全部文件（纯函数，无副作用）。

    参数:
    photo (dict): 浏览器解析后的照片记录（含 _base_dir 与绝对路径字段；
                  相对路径会按 _base_dir 归一）

    返回:
    list[str]: 去重后、当前真实存在的待删文件绝对路径列表

    Compute every file to remove for one photo (pure function).

    Parameters:
    photo (dict): resolved browser photo record (_base_dir + path fields;
                  relative paths are resolved against _base_dir).

    Return:
    list[str]: deduplicated, currently-existing absolute file paths.
    """
    candidates: List[str] = []

    # 1. 主文件：current_path 优先（整理后位置），original_path 兜底
    #    Main file: current_path first, original_path as fallback.
    main = _normalize(photo, photo.get("current_path")
                      or photo.get("original_path"))
    if main:
        candidates.append(main)
        # 2. RAW+JPG 成对伴生 JPG（同目录同名）；纯 JPG 时 sibling_jpeg
        #    返回自身路径，去重后无害。
        #    Paired JPEG sibling; for a pure-JPEG photo sibling_jpeg
        #    returns the same path — deduplicated below, harmless.
        sibling = sibling_jpeg(main)
        if sibling:
            candidates.append(os.path.normpath(sibling))
        # 5. V-cover 伴生视频（尽力而为，缺失即跳过）
        #    Companion video of a video cover (best effort).
        try:
            from core.video_cover import companion_video_for_cover
            video = companion_video_for_cover(main)
            if video:
                candidates.append(os.path.normpath(video))
        except Exception:
            pass

    # 3/4. 预览图与调试图
    #      Preview and debug images.
    for key in _FILE_KEYS:
        p = _normalize(photo, photo.get(key))
        if p:
            candidates.append(p)

    # 6. sidecar JSON
    #    Sidecar JSON.
    sidecar = _sidecar_path(photo)
    if sidecar:
        candidates.append(sidecar)

    # 7. V5.9 复核图与暗版伴随缓存（DB 无列，按前缀推导，存在才删）
    #    V5.9 review crops & dark sidecar (no DB columns, prefix-derived).
    bright_crop = _bright_crop_path(photo)
    if bright_crop:
        candidates.append(bright_crop)
        # V5.9.3: raw_stretch 重试胜出的复核图（<前缀>_raw.jpg）
        # V5.9.3: review crop for raw_stretch retry wins (<prefix>_raw.jpg)
        candidates.append(bright_crop.replace("_bright.jpg", "_raw.jpg"))
        candidates.append(bright_crop.replace("_bright.jpg", "_dark.jpg"))
    dark_preview = _dark_preview_path(photo)
    if dark_preview:
        candidates.append(dark_preview)

    # 去重 + 只保留真实存在的普通文件（绝不碰目录）
    # Deduplicate and keep only existing regular files (never directories).
    seen = set()
    existing: List[str] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            existing.append(path)
    return existing


def purge_photos(db, photos: list) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """
    彻底删除一批照片：先删磁盘文件，再删 report.db 记录。

    逐张独立处理：某张文件删除失败不影响其它张；主文件已不存在（或从未
    记录路径）时视为"磁盘侧已干净"，仍删除 DB 记录——与浏览器既有回收站
    删除的兜底语义一致。

    参数:
    db: ReportDB 或 MergedReportDB（需提供 delete_photo，兼容
        filename / (source_dir, filename) 两种键）
    photos (list[dict]): 浏览器解析后的照片记录列表

    返回:
    tuple: (purged, failed)
        purged (list[dict]): 文件与 DB 记录均已清除的照片
        failed (list[tuple[str, str]]): [(文件路径, 错误信息), ...]，
            仅收集文件删除失败项；DB 删除异常按 (filename, 错误) 计入

    Permanently delete a batch of photos: files first, then DB records.

    Each photo is handled independently; a file failure on one photo does
    not block others. A photo whose main file is already gone (or never
    recorded) is treated as clean on disk and its DB record is still
    removed — matching the browser's existing trash-delete fallback.

    Parameters:
    db: ReportDB or MergedReportDB (must expose delete_photo accepting
        filename / (source_dir, filename) keys)
    photos (list[dict]): resolved browser photo records

    Return:
    tuple: (purged, failed)
        purged (list[dict]): photos fully removed from disk and DB
        failed (list[tuple[str, str]]): [(path, error), ...] for file
            deletions that failed; DB errors are reported as
            (filename, error).
    """
    purged: List[dict] = []
    failed: List[Tuple[str, str]] = []

    for photo in photos:
        filename = photo.get("filename") or ""
        main = _normalize(photo, photo.get("current_path")
                          or photo.get("original_path"))

        # 1. 磁盘文件：逐个删除，失败收集后继续
        #    Disk files: remove one by one, collect failures, keep going.
        file_ok = True
        for path in collect_purge_files(photo):
            try:
                os.remove(path)
            except OSError as e:
                file_ok = False
                failed.append((path, str(e)))

        # 2. DB 记录：文件删干净（或本就不在）才删，避免出现
        #    "记录没了但文件还在"的半删除状态。
        #    DB record: only after files are gone (or never were), so a
        #    half-deleted state (no record, file remains) can't happen.
        if db is not None and file_ok:
            source_dir = photo.get("source_dir")
            db_key = (source_dir, filename) if source_dir else filename
            try:
                db.delete_photo(db_key)
            except Exception as e:
                failed.append((filename, str(e)))
                continue

        if file_ok:
            purged.append(photo)

    return purged, failed
