#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
照片改种对「主框种名与 photos 行不一致」的同步回归测试。

背景（2026-10 奥森黄眉柳莺键分裂）：photos 行中英名为黄眉柳莺、主鸟框
（is_selected=1）的种名却是冕柳莺（历史管线写偏）。photo 改种的旧名
匹配只命中背景框，主框被跳过——站点按（中文名, 学名）分组时留下
「黄眉柳莺 + Phylloscopus coronatus」的第二个条目（5 张 vs 24 张）。

修复后：主框无论自身种名是否命中旧名，一律随照片改种改写。

Regression tests for photo rename syncing the main (selected) detection
box even when its own species disagrees with the photos-row species —
the mismatch that split one species into two site entries.
"""
from unittest.mock import patch

import pytest

import spb_rename_species
from spb_rename_species import cmd_photo_rename

CANON = {"cn": "黄眉柳莺", "en": "Yellow-browed Warbler",
         "sci": "Phylloscopus inornatus"}
WRONG = {"cn": "冕柳莺", "en": "Eastern Crowned Warbler",
         "sci": "Phylloscopus coronatus"}


def _make_mismatch_db(d):
    """复刻 027A5213：photos 行黄眉柳莺，主框冕柳莺，背景框黄眉柳莺。

    Build one photo whose photos row says 黄眉柳莺 while the selected
    main box says 冕柳莺, plus one correctly-labeled background box.
    """
    from tools.report_db import ReportDB
    db = ReportDB(str(d))
    db.insert_photo({"filename": "027A5213", "has_bird": 1,
                     "bird_species_cn": CANON["cn"],
                     "bird_species_en": CANON["en"]})
    db.insert_detections_batch([
        {"filename": "027A5213", "bird_index": 0, "is_selected": 1,
         "bbox_x": 100.0, "bbox_y": 100.0, "bbox_w": 50.0, "bbox_h": 60.0,
         "species_cn": WRONG["cn"], "species_en": WRONG["en"],
         "scientific_name": WRONG["sci"], "species_confidence": 41.0},
        {"filename": "027A5213", "bird_index": 1, "is_selected": 0,
         "bbox_x": 200.0, "bbox_y": 200.0, "bbox_w": 40.0, "bbox_h": 50.0,
         "species_cn": CANON["cn"], "species_en": CANON["en"],
         "scientific_name": CANON["sci"], "species_confidence": 60.0},
    ])
    return db


def _boxes(db, fn):
    return {(r[0], r[1], r[2]) for r in db._conn.execute(
        "SELECT species_cn, species_en, scientific_name "
        "FROM bird_detections WHERE filename=? AND deleted=0", (fn,))}


def test_rename_syncs_mismatched_main_box(tmp_path, monkeypatch):
    """改种必须同步主框：不一致的主框随照片改写为新种。"""
    db = _make_mismatch_db(tmp_path)
    monkeypatch.setattr(spb_rename_species, "_finalize",
                        lambda db_, directory: 0)
    result = cmd_photo_rename(db, str(tmp_path), "027A5213",
                              new_cn=CANON["cn"], new_en=CANON["en"],
                              new_sci=CANON["sci"], apply=True)
    assert result["status"] == "ok"
    # 主框与背景框全部收敛到新种三元组,无冕柳莺残留
    assert _boxes(db, "027A5213") == {
        (CANON["cn"], CANON["en"], CANON["sci"])}
    photo = db._conn.execute(
        "SELECT bird_species_cn, bird_species_en FROM photos "
        "WHERE filename='027A5213'").fetchone()
    assert photo[0] == CANON["cn"] and photo[1] == CANON["en"]


def test_dry_run_leaves_mismatched_main_box_untouched(tmp_path, monkeypatch):
    """dry-run 只统计(主框计入目标),不写任何行。"""
    db = _make_mismatch_db(tmp_path)
    monkeypatch.setattr(spb_rename_species, "_finalize",
                        lambda db_, directory: 0)
    result = cmd_photo_rename(db, str(tmp_path), "027A5213",
                              new_cn=CANON["cn"], new_en=CANON["en"],
                              new_sci=CANON["sci"], apply=False)
    assert result["status"] == "ok"
    assert result["changed_detections"] == 2  # 命中背景框 1 + 主框 1
    # 主框仍为冕柳莺旧三元组
    assert (WRONG["cn"], WRONG["en"], WRONG["sci"]) in _boxes(db, "027A5213")


def test_normal_case_still_updates_both_boxes(tmp_path, monkeypatch):
    """常规态(主框与 photos 行同名)行为不变:两框都改写、无重复目标。"""
    from tools.report_db import ReportDB
    db = ReportDB(str(tmp_path))
    db.insert_photo({"filename": "027A0001", "has_bird": 1,
                     "bird_species_cn": WRONG["cn"],
                     "bird_species_en": WRONG["en"]})
    db.insert_detections_batch([
        {"filename": "027A0001", "bird_index": 0, "is_selected": 1,
         "bbox_x": 100.0, "bbox_y": 100.0, "bbox_w": 50.0, "bbox_h": 60.0,
         "species_cn": WRONG["cn"], "species_en": WRONG["en"],
         "scientific_name": WRONG["sci"], "species_confidence": 90.0},
        {"filename": "027A0001", "bird_index": 1, "is_selected": 0,
         "bbox_x": 200.0, "bbox_y": 200.0, "bbox_w": 40.0, "bbox_h": 50.0,
         "species_cn": WRONG["cn"], "species_en": WRONG["en"],
         "scientific_name": WRONG["sci"], "species_confidence": 30.0},
    ])
    monkeypatch.setattr(spb_rename_species, "_finalize",
                        lambda db_, directory: 0)
    result = cmd_photo_rename(db, str(tmp_path), "027A0001",
                              new_cn=CANON["cn"], new_en=CANON["en"],
                              new_sci=CANON["sci"], apply=True)
    assert result["status"] == "ok"
    assert result["changed_detections"] == 2
    assert _boxes(db, "027A0001") == {
        (CANON["cn"], CANON["en"], CANON["sci"])}
