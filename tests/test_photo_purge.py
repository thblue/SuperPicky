#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photo_purge 沙箱单元测试（临时目录，绝不触碰真实照片库）。

覆盖：
1. collect_purge_files 清单完整性（主文件/成对 JPG/预览/调试图/sidecar）
2. purge_photos 删文件 + 级联删 DB（photos + bird_detections）
3. 文件删除失败时不删 DB 记录（防半删除状态）
4. 文件已缺失时仍清 DB 记录（兜底语义）
5. MergedReportDB.delete_photo 级联删检测框
"""

import os
import sqlite3

import pytest

from core.photo_purge import collect_purge_files, purge_photos
from tools.report_db import ReportDB


def _make_photo_dir(tmp_path):
    """搭建一张 RAW+JPG 照片的完整磁盘与 DB 痕迹，返回 (目录, 照片字典)。

    Set up full on-disk + DB traces for one RAW+JPG photo.
    """
    base = tmp_path / "photos"
    (base / ".superpicky" / "cache").mkdir(parents=True)
    (base / ".superpicky" / "meta").mkdir(parents=True)

    raw = base / "IMG_001.NEF"
    raw.write_bytes(b"raw")
    jpg = base / "IMG_001.JPG"
    jpg.write_bytes(b"jpg")
    preview = base / ".superpicky" / "cache" / "IMG_001.jpg"
    preview.write_bytes(b"preview")
    crop_debug = base / ".superpicky" / "cache" / "IMG_001_crop.jpg"
    crop_debug.write_bytes(b"crop")
    yolo_debug = base / ".superpicky" / "cache" / "IMG_001_yolo.jpg"
    yolo_debug.write_bytes(b"yolo")
    sidecar = base / ".superpicky" / "meta" / "IMG_001.json"
    sidecar.write_text("{}", encoding="utf-8")

    db = ReportDB(str(base))
    db.insert_photo({
        "filename": "IMG_001",
        "has_bird": 1,
        "rating": 0,
        "current_path": str(raw),
        "temp_jpeg_path": str(preview),
        "debug_crop_path": str(crop_debug),
        "yolo_debug_path": str(yolo_debug),
    })
    db.insert_detections_batch([{
        "filename": "IMG_001",
        "bird_index": 0,
        "species_cn": "测试鸟",
        "species_en": "Test Bird",
        "is_selected": 1,
    }])

    photo = {
        "filename": "IMG_001",
        "_base_dir": str(base),
        "current_path": str(raw),
        "temp_jpeg_path": str(preview),
        "debug_crop_path": str(crop_debug),
        "yolo_debug_path": str(yolo_debug),
    }
    return base, photo, db


def test_collect_purge_files_manifest(tmp_path):
    """清单应含主文件/成对 JPG/预览/两张调试图/sidecar，共 6 项。"""
    base, photo, _db = _make_photo_dir(tmp_path)
    files = collect_purge_files(photo)
    # 相对路径 + 大小写归一（Windows 大小写不敏感，sibling_jpeg 可能
    # 以 .jpg 拼写返回实际为 .JPG 的成对文件）
    rels = {os.path.normcase(os.path.relpath(f, str(base))) for f in files}
    assert rels == {
        "img_001.nef",
        "img_001.jpg",  # 成对 JPG（同名边车）
        os.path.normcase(".superpicky/cache/IMG_001.jpg"),
        os.path.normcase(".superpicky/cache/IMG_001_crop.jpg"),
        os.path.normcase(".superpicky/cache/IMG_001_yolo.jpg"),
        os.path.normcase(".superpicky/meta/IMG_001.json"),
    }
    # 纯函数：重复调用结果一致，且不删任何文件
    assert collect_purge_files(photo) == files
    assert os.path.exists(photo["current_path"])


def test_purge_photos_deletes_files_and_db(tmp_path):
    """彻底删除后磁盘 6 项全清，photos + bird_detections 记录级联清除。"""
    base, photo, db = _make_photo_dir(tmp_path)
    purged, failed = purge_photos(db, [photo])
    assert failed == []
    assert len(purged) == 1

    # 只剩 report.db 及其 WAL/SHM 边车，照片痕迹全清
    leftovers = [str(p) for p in base.rglob("*")
                 if p.is_file() and not p.name.startswith("report.db")]
    assert leftovers == [], leftovers
    assert db.get_photo("IMG_001") is None
    assert db.get_detections("IMG_001") == []


def test_purge_failure_keeps_db_record(tmp_path, monkeypatch):
    """任一文件删除失败时：该照片不进 purged、DB 记录保留（防半删除）。"""
    _base, photo, db = _make_photo_dir(tmp_path)
    real_remove = os.remove

    def _flaky_remove(path, *args, **kwargs):
        if path.endswith("IMG_001.NEF"):
            raise OSError("mocked: file in use")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr("core.photo_purge.os.remove", _flaky_remove)
    purged, failed = purge_photos(db, [photo])

    assert purged == []
    assert len(failed) == 1 and failed[0][0].endswith("IMG_001.NEF")
    # 半删除保护：DB 记录仍在
    assert db.get_photo("IMG_001") is not None
    assert db.get_detections("IMG_001") != []


def test_purge_missing_files_still_clears_db(tmp_path):
    """文件已不在磁盘（如手动清理过）时，仍应清掉 DB 记录。"""
    base = tmp_path / "photos"
    base.mkdir()
    db = ReportDB(str(base))
    db.insert_photo({"filename": "GONE", "has_bird": 0, "rating": -1})
    photo = {"filename": "GONE", "_base_dir": str(base)}
    purged, failed = purge_photos(db, [photo])
    assert failed == [] and len(purged) == 1
    assert db.get_photo("GONE") is None


def test_merged_delete_photo_cascades_detections(tmp_path):
    """合并模式 delete_photo 也应级联删除子库的 bird_detections。"""
    from tools.merged_report_db import MergedReportDB

    root = tmp_path / "root"
    sub = root / "day1"
    sub.mkdir(parents=True)
    subdb = ReportDB(str(sub))
    subdb.insert_photo({"filename": "IMG_A", "has_bird": 1, "rating": 1})
    subdb.insert_detections_batch([{
        "filename": "IMG_A",
        "bird_index": 0,
        "species_cn": "测试鸟",
        "species_en": "Test Bird",
    }])
    subdb.close()

    merged = MergedReportDB(str(root), [str(sub)])
    assert merged.delete_photo(("day1", "IMG_A")) is True
    assert merged.get_all_photos() == []
    # 级联：子库里的检测框一并删除（report.db 位于 .superpicky/ 下）
    check = sqlite3.connect(str(sub / ".superpicky" / "report.db"))
    n = check.execute(
        "SELECT COUNT(*) FROM bird_detections WHERE filename = 'IMG_A'"
    ).fetchone()[0]
    check.close()
    assert n == 0


def test_context_menu_purge_invokes_handler():
    """右键菜单「彻底删除」项存在且正确接线到浏览器 _on_purge_photos。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QWidget

    _app = QApplication.instance() or QApplication([])
    import ui.results_browser_window as rbw
    from tools.i18n import get_i18n

    received = []

    class _FakeBrowser(QWidget):
        def _on_purge_photos(self, p):
            received.append(p)

    parent = _FakeBrowser()
    photo = {"filename": "DSC01234.ARW",
             "current_path": "/nonexistent/DSC01234.ARW"}
    menu = rbw._build_context_menu(parent, photo, "/nonexistent")
    label = get_i18n().t("browser.ctx_purge").format(count=1)
    matched = [act for act in menu.actions() if act.text() == label]
    assert matched, f"菜单应含「{label}」项"
    for act in matched:
        act.trigger()
    parent.close()
    assert received and received[0]["filename"] == "DSC01234.ARW"
