#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-cover 视频处理阶段（视频进库方案 C2）

把目录顶层每个视频的封面帧（core/video_cover.py 产物）按"封面即照片"
的思路接入照片体系：

    1. 扫描目录顶层视频（.mp4/.mov/.m4v，跳过隐藏文件与 AppleDouble）。
    2. 幂等：report.db 已有 <stem>_vcover 行的视频直接跳过（保留人工编辑）。
    3. 每个视频：生成封面 JPEG → 与照片同一套检测（detect_and_draw_birds，
       同模型同 ai_confidence 阈值）→ 有鸟则走多鸟识别
       （identify_bird 主鸟 + classify_secondary_birds 逐鸟，含 V5.1 主鸟重选）。
    4. 定星：有鸟固定 2 星（不进 V2 配额池，不走 40%/20% 分档）；
       YOLO 置信度低于 ai_confidence → 0；无鸟 → -1。与照片 gate 语义对齐。
    5. 入库：封面作为一条普通 photos 行（filename=<stem>_vcover）+
       bird_detections 行（is_selected 恰好 1 只主鸟）。
    6. organize 开启且非平铺布局时，封面 + 视频一起移入
       compute_target_folder 算出的目录（与其他照片同一布局规则），
       移动记录写入 .superpicky_video_manifest.json（reset 可还原）。
    7. 阶段尾部增量导出 sidecar（有鸟封面出 JSON，无鸟封面按 V5.5 规则跳过）。

report.db / sidecar 契约零改动：封面只是一张真实存在的 JPEG，
对所有下游（浏览库/BirdIndex/rating_mover）表现为普通照片。

The per-directory video stage (C2 of the video-into-library design):
treat each video's cover frame as a regular photo record. Detection,
multi-bird identification, DB rows, folder organization and sidecar
export all reuse the photo pipeline — no schema changes anywhere.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from constants import VIDEO_EXTENSIONS_ALL
from core.video_cover import cover_stem, generate_video_cover

# 视频封面定星（不进 V2 配额池 / fixed ratings, never in the V2 quota pool）
VIDEO_RATING_BIRD = 2        # 有鸟且过 ai_confidence 门槛 → 固定 2 星
VIDEO_RATING_LOW_CONF = 0    # 有鸟但置信度低于门槛（与照片 gate 一致）
VIDEO_RATING_NO_BIRD = -1    # 无鸟（与照片无鸟语义一致）


@dataclass
class VideoStageStats:
    """
    视频阶段统计 / Video stage statistics.

    Attributes:
        total: 目录顶层视频总数 / total top-level videos found
        skipped: 已有封面记录被幂等跳过 / already-processed, idempotently skipped
        failed: 封面生成/处理失败 / cover generation or processing failures
        covers_created: 本次新生成封面的视频数 / videos with fresh covers
        has_bird: 封面检出有鸟的视频数 / videos whose cover has a bird
        adopted: 主鸟种过采纳线的视频数 / videos with an adopted main species
        no_bird: 无鸟视频数 / bird-less videos
        low_conf: 有鸟但低置信（0 星）数 / bird-but-low-confidence (0 star)
        organized: 封面+视频完成归类的视频数 / videos moved into folders
        sidecars: 本次写出的封面 sidecar 数 / cover sidecars written
    """
    total: int = 0
    skipped: int = 0
    failed: int = 0
    covers_created: int = 0
    has_bird: int = 0
    adopted: int = 0
    no_bird: int = 0
    low_conf: int = 0
    organized: int = 0
    sidecars: int = 0


def _default_log(msg: str, level: str = "info") -> None:
    """默认日志：直接打印（CLI 场景调用方会传入自己的日志回调）。"""
    print(msg)


def find_top_level_videos(dir_path: str) -> List[str]:
    """
    扫描目录顶层（不递归）的视频文件。

    与 recursive_scanner._scan_directory_once 同规则：跳过 `.` 开头项
    （含 macOS AppleDouble `._xxx.MOV`），按文件名不区分大小写排序。

    参数:
        dir_path (str): 照片目录绝对路径

    返回:
        List[str]: 视频文件绝对路径列表（可能为空）

    List top-level (non-recursive) video files, skipping dot-prefixed
    entries (incl. AppleDouble), sorted case-insensitively by name.
    """
    videos: List[str] = []
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.name.startswith('.'):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in VIDEO_EXTENSIONS_ALL:
                    videos.append(entry.path)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    videos.sort(key=lambda p: os.path.basename(p).casefold())
    return videos


def _rating_for_cover(found_bird: bool, confidence: float,
                      ai_confidence_pct: float, rescued: bool = False) -> int:
    """
    封面定星（对齐照片 gate 语义，但固定档、不进配额池）。

    参数:
        found_bird (bool): detect_and_draw_birds 的有鸟判定
        confidence (float): 主鸟 YOLO 置信度（0-1）
        ai_confidence_pct (float): 用户 AI 置信度门槛（百分制，如 40）
        rescued (bool): 是否经补救扫描救回——两因子核验过，豁免置信度门槛
                       （与照片管线 rejected_by_detection 的 rescued 豁免一致）

    返回:
        int: 2（有鸟，含救回）/ 0（有鸟但低于门槛）/ -1（无鸟）
    """
    if not found_bird:
        return VIDEO_RATING_NO_BIRD
    if rescued:
        return VIDEO_RATING_BIRD
    if float(confidence or 0.0) < ai_confidence_pct / 100.0:
        return VIDEO_RATING_LOW_CONF
    return VIDEO_RATING_BIRD


def _identify_cover(
    cover_path: str,
    stem: str,
    found_bird: bool,
    all_birds: List[dict],
    img_dims: Optional[Tuple[int, int]],
    settings,
    config,
    identify_fn,
    log,
) -> Tuple[Optional[dict], Optional[List[dict]], Optional[dict]]:
    """
    对封面执行多鸟识别（镜像 photo_processor 的 BirdID worker 语义）。

    主鸟：identify_bird 单线程推理（封面少，无需线程池）；
    多鸟：classify_secondary_birds 逐鸟分类；对焦未命中（fallback）且
    多于 1 行时按 V5.1 稀有/画质综合重选主鸟（is_selected 恰好 1 只）。

    参数:
        cover_path (str): 封面 JPEG 路径
        stem (str): 封面主名（bird_detections.filename 用）
        found_bird (bool): 检测是否有鸟
        all_birds (List[dict]): detect_and_draw_birds 第 11 返回值
        img_dims (Optional[Tuple[int,int]]): 处理图 (w, h)
        settings: ProcessingSettings（鸟种国家/阈值/命名格式等）
        config: AdvancedConfig（multibird_* / mainbird_* 参数）
        identify_fn: identify_bird（依赖注入便于测试）
        log: 日志回调

    返回:
        (adopted_species, detection_rows, reselect_info)
        adopted_species: 过采纳线的主鸟 {cn,en,scientific,confidence,class_id,
                        gbif_rarity_100,iucn_category,aesthetic_index,
                        china_protection_level}；未过线/失败为 None
        detection_rows: bird_detections 行（无鸟/未开启多鸟时 None）
        reselect_info: {'from': idx, 'to': idx} 或 None

    Run multi-bird identification on the cover, mirroring the photo
    pipeline's BirdID worker (main bird + per-bird rows + V5.1 reselect).
    """
    if not found_bird or not settings.auto_identify:
        return None, None, None

    nf = settings.name_format if settings.name_format != "default" else None
    adopted: Optional[dict] = None
    rows: Optional[List[dict]] = None
    reselect_info: Optional[dict] = None

    try:
        result = identify_fn(
            cover_path,
            True,   # use_yolo
            True,   # use_gps（封面无 GPS，identify_bird 内部优雅回落国家码）
            settings.birdid_use_geo_filter,
            settings.birdid_country_code,
            settings.birdid_region_code,
            1,      # top_k
            nf,     # name_format
            # V5.9: 暗封面（夜鹭/晨昏视频）同样吃暗框提亮重识别。
            # V5.9: dark covers (night/dawn footage) get the brightened
            # retry as well.
            dark_retry_conf=settings.birdid_confidence_threshold,
        )
    except Exception as e:
        log(f"  ⚠️ Video cover BirdID failed [{os.path.basename(cover_path)}]: {e}",
            "warning")
        result = None

    if result and result.get('success') and result.get('results'):
        top = result['results'][0]
        if float(top.get('confidence') or 0) >= settings.birdid_confidence_threshold:
            adopted = {
                'cn': top.get('cn_name'),
                'en': top.get('en_name'),
                'scientific': top.get('scientific_name'),
                'confidence': top.get('confidence'),
                'class_id': top.get('class_id'),
                'gbif_rarity_100': top.get('gbif_rarity_100'),
                'iucn_category': top.get('iucn_category'),
                'aesthetic_index': top.get('aesthetic_index'),
                'china_protection_level': top.get('china_protection_level'),
            }

    # 多鸟逐鸟分类（单鸟也会入一行主鸟记录，与照片一致）
    # Per-bird rows (single-bird covers also get exactly one main row).
    if config.multibird_enabled and all_birds:
        try:
            from core.ai_model import read_image_bgr, read_image_dims
            from core.multi_bird import classify_secondary_birds, select_main_bird

            orig_img = read_image_bgr(cover_path)
            orig_dims = read_image_dims(cover_path) or img_dims or (0, 0)
            main_species = (None if adopted is None else {
                'cn': adopted.get('cn'),
                'en': adopted.get('en'),
                'scientific': adopted.get('scientific'),
                'confidence': adopted.get('confidence'),
                'class_id': adopted.get('class_id'),
                'gbif_rarity_100': adopted.get('gbif_rarity_100'),
                'china_protection_level': adopted.get('china_protection_level'),
            })
            rows = classify_secondary_birds(
                orig_image=orig_img,
                all_birds=all_birds,
                proc_dims=img_dims,
                orig_dims=orig_dims,
                main_species=main_species,
                filename=stem,
                photo_path=cover_path,
                min_area_ratio=config.multibird_min_area_ratio,
                use_geo_filter=settings.birdid_use_geo_filter,
                country_code=settings.birdid_country_code,
                region_code=settings.birdid_region_code,
                name_format=nf,
                identify_fn=identify_fn,
            )
            del orig_img

            # V5.1：对焦未命中（fallback）时按稀有/画质综合重选主鸟
            # V5.1: re-select the main bird when focus missed.
            sel_entry = next(
                (b for b in all_birds if b.get('is_selected')), None)
            if (rows and len(rows) > 1
                    and sel_entry is not None
                    and sel_entry.get('selection_reason') == 'fallback'):
                new_idx = select_main_bird(
                    rows,
                    rare_min_conf=config.mainbird_rare_min_conf,
                    rare_gbif=config.mainbird_rare_gbif)
                cur_idx = next(
                    (r['bird_index'] for r in rows if r.get('is_selected')), None)
                if new_idx is not None and new_idx != cur_idx:
                    for r in rows:
                        r['is_selected'] = 1 if r['bird_index'] == new_idx else 0
                    reselect_info = {'from': cur_idx, 'to': new_idx}
                    # 重选后的新主鸟过采纳线才切换 photos 表鸟种
                    # （与 _apply_mainbird_reselect 语义一致，避免低置信污染）
                    new_row = next(
                        (r for r in rows if r['bird_index'] == new_idx), None)
                    if new_row and (new_row.get('species_confidence') or 0.0) \
                            >= settings.birdid_confidence_threshold:
                        adopted = {
                            'cn': new_row.get('species_cn'),
                            'en': new_row.get('species_en'),
                            'scientific': new_row.get('scientific_name'),
                            'confidence': new_row.get('species_confidence'),
                            'class_id': new_row.get('class_id'),
                            'gbif_rarity_100': new_row.get('gbif_rarity_100'),
                            'iucn_category': None,
                            'aesthetic_index': None,
                            'china_protection_level':
                                new_row.get('china_protection_level'),
                        }
                    log(f"  🎯 主鸟重选 [{stem}]: #{cur_idx} → #{new_idx}"
                        + "（对焦未命中，按稀有/画质综合选择）", "species")
        except Exception as e:
            log(f"  ⚠️ Video cover multi-bird classify failed [{stem}]: {e}",
                "warning")
            rows = None

    return adopted, rows, reselect_info


def _organize_cover_and_video(
    dir_path: str,
    video_path: str,
    cover_path: str,
    rating: int,
    adopted: Optional[dict],
    config,
    report_db,
    stem: str,
    log,
) -> bool:
    """
    把封面 + 视频一起移入按布局规则算出的分类目录。

    与照片 _move_files_to_rating_folders 同规则：
        - rating >= 2 且鸟种过采纳线 → 用鸟种名；否则 None（走「其他鸟类」）
        - compute_target_folder 统一计算（species-first / rating-first / flat）
        - flat 布局不动文件
        - 目标已存在同名文件时跳过该文件（不覆盖，镜像照片行为）
    移动写入 .superpicky_video_manifest.json（original→video 对，
    reset 的 restore_organized_videos 会把封面和视频都移回原位）。

    参数:
        dir_path (str): 照片目录
        video_path (str): 视频绝对路径
        cover_path (str): 封面 JPEG 绝对路径
        rating (int): 封面星级
        adopted (Optional[dict]): 过采纳线的主鸟种（None=未识别）
        config: AdvancedConfig（folder_layout）
        report_db: ReportDB（更新路径）
        stem (str): 封面主名
        log: 日志回调

    返回:
        bool: 是否完成了移动（flat/目标已存在/全部失败 → False）

    Move the cover and its video together into the layout-computed folder,
    mirroring photo organizing semantics; moves are recorded in the video
    manifest so a reset restores both files.
    """
    from core.folder_layout import LAYOUT_FLAT, compute_target_folder
    from tools.video_organizer import OrganizeResult, record_organized_results

    layout = getattr(config, 'folder_layout', 'species-first')
    if layout == LAYOUT_FLAT:
        return False

    bird_name = None
    if rating >= 2 and adopted:
        try:
            from tools.i18n import get_i18n
            use_en = get_i18n().current_lang.startswith('en')
        except Exception:
            use_en = False
        bird_name = (adopted.get('en') or '').replace(' ', '_') if use_en \
            else (adopted.get('cn') or '')
        if not bird_name:
            bird_name = (adopted.get('cn') or ''
                         or (adopted.get('en') or '').replace(' ', '_'))

    try:
        from tools.i18n import get_i18n
        other_birds = get_i18n().t("logs.folder_other_birds")
    except Exception:
        other_birds = "其他鸟类"

    folder = compute_target_folder(rating, bird_name, layout, other_birds)
    if not folder:
        return False
    dst_folder = os.path.join(dir_path, folder)
    os.makedirs(dst_folder, exist_ok=True)

    move_results: List[OrganizeResult] = []
    moved_any = False
    for src in (cover_path, video_path):
        dst = os.path.join(dst_folder, os.path.basename(src))
        if os.path.exists(dst):
            # 镜像照片行为：目标已存在则不动（不覆盖）
            # Mirror photo behavior: never overwrite an existing target.
            log(f"  ⚠️ 目标已存在，跳过移动 / target exists, skip: {dst}",
                "warning")
            continue
        try:
            shutil.move(src, dst)
            moved_any = True
            move_results.append(OrganizeResult(
                source_path=src, target_video_path=dst, success=True))
        except Exception as e:
            log(f"  ⚠️ 视频归类移动失败 / move failed: {src} → {dst}: {e}",
                "warning")

    if moved_any:
        # 封面新相对路径回写 DB（current_path 语义=主文件相对路径）
        # Write the cover's new relative path back to the DB row.
        new_cover_rel = os.path.join(folder, os.path.basename(cover_path))
        try:
            report_db.update_photo(stem, {
                'current_path': new_cover_rel,
                'temp_jpeg_path': new_cover_rel,
            })
        except Exception as e:
            log(f"  ⚠️ 封面路径回写失败 / cover path DB update failed: {e}",
                "warning")
        try:
            record_organized_results(move_results)
        except Exception as e:
            log(f"  ⚠️ 视频清单写入失败 / video manifest write failed: {e}",
                "warning")
    return moved_any


def process_directory_videos(
    dir_path: str,
    settings,
    config,
    report_db,
    organize_files: bool = True,
    max_frames: int = 60,
    yolo_threshold: float = 0.5,
    yolo_model=None,
    identify_fn=None,
    log: Optional[Callable[[str, str], None]] = None,
) -> VideoStageStats:
    """
    目录级视频处理阶段主入口（照片阶段之后调用）。

    参数:
        dir_path (str): 照片目录绝对路径
        settings: ProcessingSettings（ai_confidence / auto_identify / 鸟种参数）
        config: AdvancedConfig（folder_layout / multibird_* / mainbird_*）
        report_db: 已打开的 ReportDB（照片阶段关闭后由调用方新开）
        organize_files (bool): 是否移动文件到分类文件夹（与照片 organize 同值）
        max_frames (int): 封面选帧抽帧上限（默认 60）
        yolo_threshold (float): 封面选帧 YOLO 阈值（默认 0.5）
        yolo_model: 已加载的 YOLO 模型（None 则现场加载；测试可注入）
        identify_fn: identify_bird（None 则现场导入；测试可注入）
        log: 日志回调 (msg, level)；None 则 print

    返回:
        VideoStageStats: 统计（含 sidecar 导出数）

    Main per-directory entry point, invoked after the photo stage.
    Returns stage statistics (sidecars included).
    """
    stats = VideoStageStats()
    log = log or _default_log
    videos = find_top_level_videos(dir_path)
    stats.total = len(videos)
    if not videos:
        return stats

    log(f"\n🎬 视频阶段 / Video stage: {len(videos)} 个视频（封面帧入库，默认 2 星）",
        "info")

    if yolo_model is None:
        from core.ai_model import load_yolo_model
        yolo_model = load_yolo_model()
    if identify_fn is None:
        from birdid.bird_identifier import identify_bird as identify_fn

    # 检测参数与照片完全同源（同模型同阈值）；封面不需要裁切调试产物
    # Detection uses the exact photo-stack settings; no crop debug output.
    from core.ai_model import detect_and_draw_birds
    ui_settings = [
        settings.ai_confidence,
        settings.sharpness_threshold,
        settings.nima_threshold,
        False,  # save_crop 强制关：封面不产调试裁切
        settings.normalization_mode,
    ]

    try:
        from tools.i18n import get_i18n
        i18n = get_i18n()
    except Exception:
        i18n = None

    for video_path in videos:
        stem = cover_stem(video_path)

        # 幂等：已有封面记录 → 跳过（重跑不覆盖人工编辑）
        # Idempotency: an existing cover row means "leave it alone".
        try:
            if report_db.get_photo(stem) is not None:
                stats.skipped += 1
                continue
        except Exception:
            pass

        # 1. 封面帧生成 / Cover generation
        cover_path = generate_video_cover(
            video_path, yolo_model,
            max_frames=max_frames, yolo_threshold=yolo_threshold)
        if cover_path is None:
            stats.failed += 1
            log(f"  ⚠️ 封面生成失败，跳过 / cover failed: "
                f"{os.path.basename(video_path)}", "warning")
            continue
        stats.covers_created += 1

        # 2. 检测（照片同一套）/ Detection (same stack as photos)
        det = detect_and_draw_birds(
            cover_path, yolo_model, None, dir_path, ui_settings, None,
            skip_nima=True)
        if det is None:
            stats.failed += 1
            log(f"  ⚠️ 封面检测失败 / cover detection failed: "
                f"{os.path.basename(cover_path)}", "warning")
            continue
        (found_bird, _bird_result, confidence, _sharpness, _nima,
         _bbox, img_dims, _mask, _bird_count, rescued, all_birds) = det

        rating = _rating_for_cover(found_bird, confidence, settings.ai_confidence,
                                   rescued=bool(rescued))

        # 3. 多鸟识别——镜像照片语义：只有过检测门槛（含救回豁免）的照片
        #    才提交 BirdID；低置信 0 星封面在早期拒绝块就退出，不识别。
        # 3. Multi-bird ID — mirror photo semantics: only covers that pass
        #    the detection gate (rescued exempt) reach BirdID; 0-star
        #    low-confidence covers exit early, unidentified.
        adopted = det_rows = None
        if found_bird and rating == VIDEO_RATING_BIRD:
            adopted, det_rows, _reselect = _identify_cover(
                cover_path, stem, found_bird, all_birds, img_dims,
                settings, config, identify_fn, log)

        # 4. photos 行入库 / Insert the photos row
        caption_lines = [f"视频封面 Video cover: {os.path.basename(video_path)}"]
        # 封面行写入拍摄日期（与封面帧 EXIF 同源的 get_video_capture_date
        # 链，已归一化本地墙钟），否则浏览库 sidebar 无日期——与照片救回
        # 路径曾有的 EXIF 缺口同类（2026-09-28 修复）。
        # Cover rows carry the capture date (same get_video_capture_date
        # chain embedded in the frame JPEG, normalized to local wall
        # time), otherwise the browse sidebar shows no date — same class
        # of gap the photo rescue path had (fixed 2026-09-28).
        capture_date = None
        try:
            from tools.video_organizer import get_video_capture_date
            capture_date = get_video_capture_date(video_path)
        except Exception as date_exc:
            log(f"  ⚠️ 封面日期读取失败 / capture date read failed "
                f"[{stem}]: {date_exc}", "warning")
        photo_row = {
            'filename': stem,
            'has_bird': 1 if found_bird else 0,
            'confidence': float(confidence or 0.0),   # 0-1，与照片列同单位
            'rating': rating,
            'current_path': os.path.basename(cover_path),
            'original_path': os.path.basename(cover_path),
            'temp_jpeg_path': os.path.basename(cover_path),
        }
        if capture_date is not None:
            # 与照片管线一致的 exiftool 口径（YYYY:MM:DD HH:MM:SS）
            # Same exiftool convention as photo rows (YYYY:MM:DD HH:MM:SS).
            photo_row['date_time_original'] = \
                capture_date.strftime('%Y:%m:%d %H:%M:%S')
        if adopted:
            photo_row['bird_species_cn'] = adopted.get('cn') or ''
            photo_row['bird_species_en'] = adopted.get('en') or ''
            photo_row['birdid_confidence'] = adopted.get('confidence')  # 百分制
            for col, key in (('iucn_category', 'iucn_category'),
                             ('gbif_rarity_100', 'gbif_rarity_100'),
                             ('aesthetic_index', 'aesthetic_index'),
                             ('china_protection_level', 'china_protection_level')):
                if adopted.get(key) is not None:
                    photo_row[col] = adopted[key]
            title = (adopted.get('cn') or adopted.get('en') or '')
            try:
                caption_lines.insert(
                    0, i18n.t("logs.caption_species", name=title))
            except Exception:
                caption_lines.insert(0, f"鸟种：{title}")
            stats.adopted += 1
        photo_row['caption'] = "\n".join(caption_lines)
        try:
            report_db.insert_photo(photo_row)
        except Exception as e:
            stats.failed += 1
            log(f"  ⚠️ 封面入库失败 / cover DB insert failed [{stem}]: {e}",
                "warning")
            continue

        # 5. bird_detections 行 / Detection rows
        if det_rows:
            try:
                report_db.insert_detections_batch(det_rows)
            except Exception as e:
                log(f"  ⚠️ 封面检测行入库失败 [{stem}]: {e}", "warning")

        # 统计与日志 / Stats + log line
        if not found_bird:
            stats.no_bird += 1
            log(f"  🎬 [{os.path.basename(video_path)}] 无鸟 → -1（0星_放弃，不出 sidecar）",
                "info")
        elif rating == VIDEO_RATING_LOW_CONF:
            stats.low_conf += 1
            log(f"  🎬 [{os.path.basename(video_path)}] 有鸟但低置信({confidence:.2f}) → 0",
                "info")
        else:
            stats.has_bird += 1
            name = (adopted.get('cn') or adopted.get('en') or '未识别') \
                if adopted else '未识别'
            log(f"  🎬 [{os.path.basename(video_path)}] 2星 | {name}", "species")

        # 6. 归类（封面+视频一起）/ Organize cover + video together
        if organize_files:
            if _organize_cover_and_video(
                    dir_path, video_path, cover_path, rating, adopted,
                    config, report_db, stem, log):
                stats.organized += 1

    # 7. 增量导出 sidecar（无鸟封面按 V5.5 自动跳过）
    #    Incremental sidecar export (bird-less covers skipped by V5.5 rule).
    try:
        from core.sidecar_export import export_directory_sidecars
        stats.sidecars = export_directory_sidecars(
            report_db, dir_path, log=lambda m, *a, **k: log(m, "info"))
    except Exception as e:
        log(f"  ⚠️ 封面 sidecar 导出失败 / cover sidecar export failed: {e}",
            "warning")

    log(f"🎬 视频阶段完成 / Video stage done: "
        f"{stats.covers_created} 封面, {stats.has_bird} 有鸟(2星), "
        f"{stats.adopted} 已定种, {stats.no_bird} 无鸟, "
        f"{stats.organized} 已归类, {stats.skipped} 跳过, {stats.failed} 失败",
        "info")
    return stats
