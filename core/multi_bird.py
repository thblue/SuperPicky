#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多鸟逐鸟识别编排模块（multi-bird per-bird classification）

bird_count > 1 时，对 YOLO 检出的每个鸟框单独做鸟种分类，生成
bird_detections 表的行数据。主鸟（评分对象，is_selected=1）不重复推理，
复用主流水线已有的识别结果。

设计约束：
- 与主鸟识别走完全相同的路径（birdid.bird_identifier.identify_bird，
  preloaded_crop 模式），GPS/地理过滤行为一致；
- 「看不清的就算了」由面积门槛兜底：面积 < multibird_min_area_ratio
  只入框不分类；分类结果无论置信度高低都照实入库（采纳是消费方按
  multibird_species_threshold 判断的派生概念，数据层保留完整信息）；
- 无数量上限：鸟群混稀有鸟正是目标场景，每鸟一次分类器前向
  (~10-50ms) 加一次 GPS EXIF 读取，几十只的鸟群照也可接受。

Orchestrates per-bird species classification for multi-bird photos.
The main (selected) bird reuses the pipeline's existing identification
result; every other box above the area floor is cropped and classified
through the same identify_bird path. No flock-size cap by design.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# 与主鸟裁剪一致的 padding 比例（core/photo_processor.py 主鸟 15%，
# tools/image_crop.smart_square_crop 默认值同为 0.15）
# Same padding ratio as the main-bird crop path (0.15).
_BIRDID_PADDING_RATIO = 0.15

# 轮廓多边形的最大顶点数（findContours → approxPolyDP 简化）
# Max vertices of the simplified mask polygon.
_MAX_POLYGON_POINTS = 32


def _bbox_to_orig(
    bbox_proc: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    orig_w: int,
    orig_h: int,
) -> Tuple[int, int, int, int]:
    """
    把处理图坐标系的 bbox (x1,y1,x2,y2) 换算到原图坐标并裁剪到图界。

    参数:
    bbox_proc (Tuple[int,int,int,int]): 处理图 bbox (x1, y1, x2, y2)
    scale_x / scale_y (float): 原图 / 处理图 的缩放比
    orig_w / orig_h (int): 原图尺寸

    返回:
    Tuple[int,int,int,int]: 原图坐标 bbox (x1, y1, x2, y2)，保证 x1<x2, y1<y2

    Map a processed-frame bbox onto original-image coordinates, clipped
    to the frame.
    """
    x1, y1, x2, y2 = bbox_proc
    ox1 = max(0, int(round(x1 * scale_x)))
    oy1 = max(0, int(round(y1 * scale_y)))
    ox2 = min(orig_w, int(round(x2 * scale_x)))
    oy2 = min(orig_h, int(round(y2 * scale_y)))
    if ox2 <= ox1:
        ox2 = min(orig_w, ox1 + 1)
    if oy2 <= oy1:
        oy2 = min(orig_h, oy1 + 1)
    return ox1, oy1, ox2, oy2


def _polygon_to_orig(
    polygon_proc: Optional[List[List[int]]],
    scale_x: float,
    scale_y: float,
) -> Optional[str]:
    """
    把处理图坐标的轮廓点阵换算到原图坐标，并序列化为 JSON 字符串。

    参数:
    polygon_proc: 处理图坐标 [[x,y],...]，None 原样返回
    scale_x / scale_y: 原图 / 处理图 的缩放比

    返回:
    Optional[str]: JSON 字符串（如 "[[120,80],[880,80]]"），None 表示无轮廓

    Scale polygon points from processed to original coordinates and
    serialise to a JSON string for the SQLite TEXT column.
    """
    if not polygon_proc:
        return None
    scaled = [[int(round(p[0] * scale_x)), int(round(p[1] * scale_y))]
              for p in polygon_proc]
    return json.dumps(scaled, ensure_ascii=False)


def crop_bbox_sharpness(orig_image: np.ndarray,
                        bbox_orig: Tuple[int, int, int, int]) -> float:
    """
    计算原图 bbox 矩形区域的 Tenengrad 锐度（0-1000）。

    与主鸟头部锐度同一套算法（keypoint_detector._calculate_sharpness：
    Sobel 梯度密度 → 对数归一化 → 小 ROI 尺寸补偿），量纲可直接比较。
    输入为 BGR，内部转 RGB 以匹配该函数的通道约定。

    参数:
    orig_image (np.ndarray): 原图 BGR
    bbox_orig (Tuple[int,int,int,int]): 原图坐标 bbox (x1,y1,x2,y2)

    返回:
    float: 0-1000 锐度分；区域无效时 0.0

    Tenengrad sharpness of the bbox rect on the original image, on the
    same 0-1000 scale as the main bird's head sharpness.
    """
    x1, y1, x2, y2 = bbox_orig
    if orig_image is None or orig_image.size == 0:
        return 0.0
    h_img, w_img = orig_image.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))
    crop = orig_image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    # 矩形 mask = 全选；BGR→RGB 对齐 _calculate_sharpness 的通道约定
    # Full-rect mask; BGR→RGB to match _calculate_sharpness's convention.
    mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        from core.keypoint_detector import get_keypoint_detector
        detector = get_keypoint_detector()
        return float(detector._calculate_sharpness(crop_rgb, mask))
    except Exception:
        return 0.0


def classify_secondary_birds(
    orig_image: np.ndarray,
    all_birds: List[dict],
    proc_dims: Tuple[int, int],
    orig_dims: Tuple[int, int],
    main_species: Optional[dict],
    filename: str,
    photo_path: str,
    min_area_ratio: float,
    use_geo_filter: bool = True,
    country_code: Optional[str] = None,
    region_code: Optional[str] = None,
    name_format: Optional[str] = None,
    identify_fn=None,
) -> List[dict]:
    """
    对多鸟照片的每个检测框生成 bird_detections 行（含逐鸟分类）。

    主鸟（all_birds 中 is_selected=True 的项）不重复推理：main_species
    为调用方已采纳的主鸟识别结果（低于用户阈值时传 None，则主鸟行物种
    留空）。其余每鸟：面积 ≥ min_area_ratio 才分类；分类结果无论置信度
    高低都存 top-1（物种+置信度照实入库）——「是否采纳」是展示层/统计
    层按阈值判断的派生概念，数据层保留完整信息供人工审阅调阈值。
    面积过小/分类失败仍只保留框与几何信息。

    参数:
    orig_image (np.ndarray): 原图 BGR（未缩放）
    all_birds (List[dict]): ai_model.detect_and_draw_birds 第 11 个返回值，
        每项含 idx/conf/bbox(x1,y1,x2,y2)/area_ratio/mask_polygon/
        is_selected（处理图坐标）
    proc_dims (Tuple[int,int]): 处理图 (w, h)
    orig_dims (Tuple[int,int]): 原图 (w, h)
    main_species (Optional[dict]): 主鸟已采纳识别结果
        {cn, en, scientific, confidence, class_id, gbif_rarity_100}，可为 None
    filename (str): 照片前缀（bird_detections.filename）
    photo_path (str): 原始文件路径（identify_bird 读取 GPS 用）
    min_area_ratio (float): 最小 bbox 面积占比，低于只入框
    use_geo_filter / country_code / region_code / name_format:
        与主鸟 identify_bird 相同的地理过滤与命名参数
    identify_fn: 依赖注入的 identify_bird（测试可替换；None 则现场导入）

    返回:
    List[dict]: bird_detections 行（DETECTION_COLUMNS 键），按检测 idx 排序；
        输入为空时返回空列表

    Build bird_detections rows for every detected bird. The selected bird
    (is_selected=True) reuses main_species without re-inference; others
    are cropped and classified individually, storing the raw top-1 result
    regardless of confidence — adoption is a derived, display-level
    concept so users can review and tune thresholds.
    """
    if not all_birds:
        return []
    if identify_fn is None:
        from birdid.bird_identifier import identify_bird as identify_fn

    proc_w, proc_h = proc_dims
    orig_w, orig_h = orig_dims
    if proc_w <= 0 or proc_h <= 0 or orig_w <= 0 or orig_h <= 0:
        return []
    scale_x = orig_w / float(proc_w)
    scale_y = orig_h / float(proc_h)

    # 主鸟识别结果 → 行字段（None 安全）
    # Map the adopted main-bird result to row fields (None-safe).
    main_fields = {
        'species_cn': (main_species or {}).get('cn'),
        'species_en': (main_species or {}).get('en'),
        'scientific_name': (main_species or {}).get('scientific'),
        'species_confidence': (main_species or {}).get('confidence'),
        'class_id': (main_species or {}).get('class_id'),
        'gbif_rarity_100': (main_species or {}).get('gbif_rarity_100'),
    }

    rows: List[dict] = []
    from tools.image_crop import smart_square_crop

    for bird in all_birds:
        bbox_proc = bird.get('bbox')
        if not bbox_proc:
            continue
        row = {
            'filename': filename,
            'bird_index': len(rows),
            'is_selected': 1 if bird.get('is_selected') else 0,
            'area_ratio': bird.get('area_ratio'),
            'yolo_conf': bird.get('conf'),
            # 物种/锐度字段预置 None：未分类、面积过小、置信度不足的行
            # 保持统一形状，入库时自然落 NULL / Species fields default to
            # None so every row has the same shape (NULL in SQLite).
            'crop_sharpness': None,
            'species_cn': None,
            'species_en': None,
            'scientific_name': None,
            'species_confidence': None,
            'class_id': None,
            'gbif_rarity_100': None,
        }
        rows.append(row)

        bbox_orig = _bbox_to_orig(bbox_proc, scale_x, scale_y,
                                  orig_w, orig_h)
        x1, y1, x2, y2 = bbox_orig
        row['bbox_x'] = float(x1)
        row['bbox_y'] = float(y1)
        row['bbox_w'] = float(x2 - x1)
        row['bbox_h'] = float(y2 - y1)
        row['mask_polygon'] = _polygon_to_orig(bird.get('mask_polygon'),
                                               scale_x, scale_y)

        # 全部行都记录 bbox 锐度（廉价），供综合重选/召回规则使用——
        # 包括主鸟（V5.1 重选评分需要主鸟也有锐度可比）
        # Record bbox sharpness on every row (cheap), the main bird
        # included — the comprehensive re-selection needs comparable
        # sharpness across all birds.
        row['crop_sharpness'] = crop_bbox_sharpness(orig_image, bbox_orig)

        # 主鸟：复用已采纳结果，不重复推理
        # Selected bird: reuse the adopted result, no re-inference.
        if row['is_selected']:
            row.update(main_fields)
            continue

        area_ratio = bird.get('area_ratio') or 0.0
        if area_ratio < min_area_ratio:
            # 太小看不清：只入框不分类 / Too small: keep the box, skip ID.
            continue

        try:
            square_bgr = smart_square_crop(
                orig_image, bbox_orig,
                padding_ratio=_BIRDID_PADDING_RATIO)
            pil_crop = Image.fromarray(
                cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB))
            result = identify_fn(
                photo_path,           # image_path：读 GPS/地理过滤
                False,                # use_yolo：已按框裁剪，跳过内部复检
                True,                 # use_gps
                use_geo_filter,
                country_code,
                region_code,
                1,                    # top_k
                name_format,
                pil_crop,             # preloaded_crop
            )
        except Exception:
            continue

        if not (result and result.get('success') and result.get('results')):
            continue
        # top-1 结果照实入库（无论置信度高低）——采纳与否由消费方按
        # 阈值判断，数据层保留完整信息便于人工审阅和调阈值。
        # Store the raw top-1 result; adoption is decided downstream.
        top = result['results'][0]
        row['species_cn'] = top.get('cn_name')
        row['species_en'] = top.get('en_name')
        row['scientific_name'] = top.get('scientific_name')
        row['species_confidence'] = float(top.get('confidence') or 0.0)
        row['class_id'] = top.get('class_id')
        row['gbif_rarity_100'] = top.get('gbif_rarity_100')

    return rows


def select_main_bird(rows: List[dict],
                     rare_min_conf: float = 70.0,
                     rare_gbif: float = 50.0) -> Optional[int]:
    """
    焦点未命中时的综合主鸟选择（V5.1 规则 3）。

    输入为 classify_secondary_birds 的输出行（含每鸟物种/置信度/
    crop_sharpness/面积/GBIF 稀有度）。策略：
      3a. 有「置信的稀有鸟」（分类置信 ≥ rare_min_conf 且 GBIF 稀有度
          ≥ rare_gbif）→ 取其中稀有度最高、并列取置信度最高；
      3b. 否则按「大而清晰」评分：锐度 45% + 面积 25% + 置信度 30%
          （有分类结果用分类置信，未分类用 YOLO 检测置信）。

    参数:
    rows (List[dict]): 逐鸟结果行
    rare_min_conf (float): 稀有鸟要求的分类置信度（百分比）
    rare_gbif (float): 稀有鸟要求的 GBIF 稀有度（0-100，50=罕见档）

    返回:
    Optional[int]: 选中的 bird_index；rows 为空返回 None

    Comprehensive main-bird re-selection when the AF point did not
    hit any bird: confident-rare species first, otherwise the
    big-clear-and-confident bird by a weighted score.
    """
    candidates = [r for r in rows if not r.get("deleted")]
    if not candidates:
        return None

    # 3a 稀有优先 / rare-and-confident wins
    rare = [r for r in candidates
            if (r.get("species_confidence") or 0.0) >= rare_min_conf
            and (r.get("gbif_rarity_100") or 0.0) >= rare_gbif]
    if rare:
        best = max(rare, key=lambda r: (r.get("gbif_rarity_100") or 0.0,
                                         r.get("species_confidence") or 0.0))
        return best.get("bird_index")

    # 3b 大而清晰 / big, sharp and confident
    def _score(r: dict) -> float:
        sharp = min(1.0, (r.get("crop_sharpness") or 0.0) / 500.0)
        area = min(1.0, (r.get("area_ratio") or 0.0) / 0.10)
        if r.get("species_cn") or r.get("species_en"):
            conf = min(1.0, (r.get("species_confidence") or 0.0) / 100.0)
        else:
            conf = max(0.0, min(1.0, r.get("yolo_conf") or 0.0))
        return 0.45 * sharp + 0.25 * area + 0.30 * conf

    best = max(candidates, key=_score)
    return best.get("bird_index")
