#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物种召回模块（species recall，V5.2）

批处理末尾运行：找出本批「从未成为主鸟种」的鸟种——它们只以次要鸟
的身份出现过，主鸟选择（对焦点/综合评分）可能埋没了它们。含有这类
鸟种的照片打 notable 标记（photos.notable=1 + 对应 bird_detections
行 notable=1），供结果浏览器快速筛选、人工核对/修改。

判定口径：
- 「成为过主鸟种」= 任一照片的选中检测（is_selected=1）是该鸟种，
  或 photos 表记录的主鸟种是它（含人工改种/综合重选的结果）；
- 触发召回的次要鸟必须「已采纳」：分类置信度 ≥ 采纳阈值（默认 35%，
  与逐鸟采纳同门槛），避免低置信误检刷屏；
- 标记与星级/精选完全正交（notable 独立列），不参与评分。

Batch-end species recall: flag photos containing species that never
served as the main species in this batch, so buried rarities surface
for quick review. Orthogonal to ratings/picked.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple


def _species_key(cn: Optional[str], en: Optional[str],
                 sci: Optional[str] = None) -> str:
    """
    鸟种唯一键（学名优先，回退中文名/英文名）。

    参数:
    cn / en / sci (Optional[str]): 中文名 / 英文名 / 学名

    返回:
    str: 唯一键；全空返回空串

    Unique species key: scientific name first, then cn/en.
    """
    return (sci or cn or en or "").strip()


def run_species_recall(report_db,
                       species_threshold: float = 35.0,
                       log=print) -> dict:
    """
    执行批内物种召回并落库标记。

    参数:
    report_db: 已打开的 ReportDB（photos + bird_detections 齐全）
    species_threshold (float): 触发召回要求的分类置信度（百分比）
    log: 日志函数

    返回:
    dict: {'never_main_species': [鸟种名...],
           'flagged_photos': N, 'flagged_detections': M}
        无检测结果时各项为空/0

    Run the batch-local recall and persist notable flags.
    """
    detections = report_db.get_all_detections(include_polygon=False)
    photos = report_db.get_all_photos()
    if not detections:
        return {"never_main_species": [], "flagged_photos": 0,
                "flagged_detections": 0}

    # 1) 收集「当过主鸟种」的集合：选中检测 + photos 表主鸟种
    #    （人工软删的检测行不参与任何一侧 / soft-deleted rows excluded）
    main_keys: Set[str] = set()
    for det in detections:
        if det.get("is_selected") and not det.get("deleted"):
            key = _species_key(det.get("species_cn"), det.get("species_en"),
                               det.get("scientific_name"))
            if key:
                main_keys.add(key)
    for photo in photos:
        key = _species_key(photo.get("bird_species_cn"),
                           photo.get("bird_species_en"))
        if key:
            main_keys.add(key)

    # 2) 扫次要鸟：已采纳 + 鸟种从未当主 → 标记
    never_main: List[str] = []          # 展示名（按首次出现）
    never_main_keys: Set[str] = set()
    detection_marks: List[dict] = []
    flagged_photos: Set[str] = set()
    for det in detections:
        if det.get("is_selected") or det.get("deleted"):
            continue
        cn, en = det.get("species_cn"), det.get("species_en")
        if not (cn or en):
            continue
        conf = det.get("species_confidence") or 0.0
        if conf < species_threshold:
            continue
        key = _species_key(cn, en, det.get("scientific_name"))
        if not key or key in main_keys:
            continue
        if key not in never_main_keys:
            never_main_keys.add(key)
            never_main.append(cn or en or key)
        detection_marks.append({
            "filename": det.get("filename"),
            "bird_index": det.get("bird_index"),
            "notable_reason": "never_main_species",
        })
        flagged_photos.add(det.get("filename"))

    if detection_marks:
        report_db.apply_recall_marks(sorted(flagged_photos),
                                     detection_marks)

    log(
        f"  🦜 物种召回: 本批从未成为主鸟的鸟种 {len(never_main)} 种"
        + (f"（{'、'.join(never_main[:8])}"
           + ('…' if len(never_main) > 8 else '') + "）" if never_main else "")
        + f" → 标记 {len(flagged_photos)} 张照片 / "
        f"{len(detection_marks)} 处检测（结果浏览器可按「召回」筛选）")

    return {"never_main_species": never_main,
            "flagged_photos": len(flagged_photos),
            "flagged_detections": len(detection_marks)}
