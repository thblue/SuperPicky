#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整批改种/软删族函数的受影响清单回归测试。

背景（2026-09-21 玉渊潭改种时发现）：soft_delete_species_detections /
soft_delete_species_everywhere / rename_species_everywhere 三处 return 行
误写 `sorted({row[0] for r in rows})`——集合推导遍历 r，取的却是外层
`for row in rows:` 泄漏的最后一行，多张照片受影响时返回清单只剩最后
一张。浏览器整批改种据此清单同步 sidecar，会漏掉其余照片。

Regression tests for the affected-file list of the batch rename / soft
delete family: the return lines used a leaked loop variable, collapsing
the multi-photo list to its last row, so the browser's sidecar sync
missed every file but the last.
"""
from tools.report_db import ReportDB


def _make_two_photo_db(d):
    """建两库各一张照片、各一行同鸟种主鸟检测行。/ One photo each, same wrong species."""
    db = ReportDB(d)
    for fn in ("027A0001", "027A0002"):
        db.insert_photo({"filename": fn, "has_bird": 1,
                         "bird_species_cn": "大杜鹃",
                         "bird_species_en": "Common Cuckoo"})
        db.insert_detections_batch([{
            "filename": fn, "bird_index": 0, "is_selected": 1,
            "bbox_x": 100.0, "bbox_y": 100.0, "bbox_w": 50.0, "bbox_h": 60.0,
            "species_cn": "大杜鹃", "species_en": "Common Cuckoo",
            "scientific_name": "Cuculus canorus",
            "species_confidence": 50.0,
        }])
    return db


def test_rename_returns_all_affected_files(tmp_path):
    """改种 2 张时返回清单必须含两张（曾只返回最后一张）。"""
    db = _make_two_photo_db(str(tmp_path))
    try:
        files, n = db.rename_species_everywhere(
            "大杜鹃", "Common Cuckoo", None,
            "北方中杜鹃", "Oriental Cuckoo", "Cuculus optatus")
        assert n == 2
        assert sorted(files) == ["027A0001", "027A0002"]
        # 检测行与主鸟种都已改写
        for fn in ("027A0001", "027A0002"):
            det = db._conn.execute(
                "SELECT species_cn, species_en, scientific_name "
                "FROM bird_detections WHERE filename=?", (fn,)).fetchone()
            assert det["species_cn"] == "北方中杜鹃"
            photo = db._conn.execute(
                "SELECT bird_species_cn FROM photos WHERE filename=?",
                (fn,)).fetchone()
            assert photo["bird_species_cn"] == "北方中杜鹃"
    finally:
        db.close()


def test_rename_includes_photos_only_hits(tmp_path):
    """photos 命中但无检测行的照片也必须在返回清单（backfill 重检路径）。

    2026-09-21 玉渊潭改种实例：5 张经 backfill 重检采纳的照片只有
    photos 行有旧名，返回清单漏掉它们导致浏览器 sidecar 同步缺失。
    """
    db = _make_two_photo_db(str(tmp_path))
    try:
        # 制造一张「仅 photos 行」照片：删掉它的检测行，主鸟种保留旧名
        db._conn.execute(
            "DELETE FROM bird_detections WHERE filename='027A0002'")
        db._conn.commit()
        files, n = db.rename_species_everywhere(
            "大杜鹃", "Common Cuckoo", None,
            "北方中杜鹃", "Oriental Cuckoo", "Cuculus optatus")
        assert n == 1  # 检测行只改 1 行
        assert sorted(files) == ["027A0001", "027A0002"]  # 清单含两张
        photo = db._conn.execute(
            "SELECT bird_species_cn FROM photos WHERE filename='027A0002'"
        ).fetchone()
        assert photo["bird_species_cn"] == "北方中杜鹃"
    finally:
        db.close()


def test_soft_delete_species_detections_returns_all(tmp_path):
    """软删待确认框的受影响清单同样必须含全部照片。

    该函数按设计只删 is_selected=0 的待确认框（主鸟框归
    soft_delete_species_everywhere 管），故先补各一张次要行。
    """
    db = _make_two_photo_db(str(tmp_path))
    try:
        for fn in ("027A0001", "027A0002"):
            db.insert_detections_batch([
                {"filename": fn, "bird_index": 0, "is_selected": 1,
                 "bbox_x": 100.0, "bbox_y": 100.0,
                 "bbox_w": 50.0, "bbox_h": 60.0,
                 "species_cn": "大杜鹃", "species_en": "Common Cuckoo",
                 "species_confidence": 50.0},
                {"filename": fn, "bird_index": 1, "is_selected": 0,
                 "bbox_x": 300.0, "bbox_y": 200.0,
                 "bbox_w": 40.0, "bbox_h": 40.0,
                 "species_cn": "大杜鹃", "species_en": "Common Cuckoo",
                 "species_confidence": 30.0},
            ])
        files, n = db.soft_delete_species_detections("大杜鹃")
        assert n == 2
        assert sorted(files) == ["027A0001", "027A0002"]
    finally:
        db.close()


def test_soft_delete_species_everywhere_returns_all(tmp_path):
    """软删全 bird 种（含主鸟框、清主鸟种）的受影响清单必须含全部照片。"""
    db = _make_two_photo_db(str(tmp_path))
    try:
        files, n = db.soft_delete_species_everywhere("大杜鹃")
        assert n == 2
        assert sorted(files) == ["027A0001", "027A0002"]
        for fn in ("027A0001", "027A0002"):
            photo = db._conn.execute(
                "SELECT bird_species_cn FROM photos WHERE filename=?",
                (fn,)).fetchone()
            assert photo["bird_species_cn"] is None
    finally:
        db.close()
