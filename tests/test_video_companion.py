#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-cover 伴生跟随测试 / Tests for the video-follows-cover companion behavior.

覆盖（C3）：
    - core/video_cover.companion_video_for_cover：命名约定解析
    - core/rating_mover：改星/改种时伴生视频随封面移动 + 视频归类清单同步
    - core/rerate_v2.load_photos：跳过 _vcover 封面行（固定 2 星不参与重定星）
    - spb_flatten._scan_companion_videos：拍平时伴生视频随封面上移的计划

全部在 pytest tmp_path 下构造最小库，绝不触碰真实照片库。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from core.rating_mover import move_photo_on_metadata_change
from core.video_cover import companion_video_for_cover

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _pin_chinese_locale():
    """钉住中文目录名（同 test_rating_mover.py 的教训）。"""
    from tools.i18n import get_i18n
    i18n = get_i18n()
    original = i18n.current_lang
    if not original.startswith("zh"):
        i18n.switch_language("zh_CN")
    yield
    if i18n.current_lang != original:
        i18n.switch_language(original)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


def _make_cover_library(tmp_path, rating_folder="2星_良好", bird="白鹭"):
    """
    构造一个最小 V-cover 库：<根>/<鸟>/<星级>/ 下放封面 jpg + 视频 mp4，
    DB 有封面行，视频归类清单记录当前位置。
    """
    from tools.report_db import ReportDB

    root = str(tmp_path)
    folder = os.path.join(bird, rating_folder)
    cover_abs = _touch(os.path.join(root, folder, "VID_1_vcover.jpg"))
    video_abs = _touch(os.path.join(root, folder, "VID_1.mp4"))

    db = ReportDB(root)
    db.insert_photo({
        "filename": "VID_1_vcover",
        "has_bird": 1,
        "rating": 2,
        "current_path": os.path.join(folder, "VID_1_vcover.jpg"),
        "original_path": "VID_1_vcover.jpg",
        "temp_jpeg_path": os.path.join(folder, "VID_1_vcover.jpg"),
        "bird_species_cn": bird,
    })
    manifest = {
        "version": 1,
        "entries": [
            {"original": os.path.join(root, "VID_1_vcover.jpg"),
             "video": cover_abs},
            {"original": os.path.join(root, "VID_1.mp4"),
             "video": video_abs},
        ],
    }
    manifest_path = os.path.join(root, ".superpicky_video_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return db, root, cover_abs, video_abs, manifest_path


# ── companion_video_for_cover ───────────────────────────────────────────────

class TestCompanionVideoForCover:
    def test_finds_video_next_to_cover(self, tmp_path):
        cover = _touch(str(tmp_path / "VID_1_vcover.jpg"))
        video = _touch(str(tmp_path / "VID_1.mp4"))
        assert companion_video_for_cover(cover) == video

    def test_uppercase_video_extension(self, tmp_path):
        cover = _touch(str(tmp_path / "GX_vcover.jpg"))
        video = _touch(str(tmp_path / "GX.MOV"))
        # Windows 文件系统大小写不敏感：命中即可，比较用 normcase
        # Windows FS is case-insensitive: match on normcase comparison.
        found = companion_video_for_cover(cover)
        assert found is not None
        assert os.path.normcase(found) == os.path.normcase(video)

    def test_plain_photo_returns_none(self, tmp_path):
        cover = _touch(str(tmp_path / "IMG_0001.jpg"))
        assert companion_video_for_cover(cover) is None

    def test_missing_video_returns_none(self, tmp_path):
        cover = _touch(str(tmp_path / "LONELY_vcover.jpg"))
        assert companion_video_for_cover(cover) is None


# ── rating_mover：视频随封面 ────────────────────────────────────────────────

class TestRatingMoverCompanion:
    def test_rating_change_moves_video_with_cover(self, tmp_path):
        """封面改星 2→3：封面与视频一起移入 {鸟种}/3星_优选，DB 回写。"""
        db, root, cover_abs, video_abs, manifest_path = \
            _make_cover_library(tmp_path)
        photo = db.get_photo("VID_1_vcover")
        photo["current_path"] = cover_abs       # 浏览器语义：绝对路径
        photo["temp_jpeg_path"] = cover_abs

        moved = move_photo_on_metadata_change(
            root, photo, 3, "白鹭", "species-first", db, "VID_1_vcover")

        assert moved is True
        new_folder = os.path.join(root, "白鹭", "3星_优选")
        assert os.path.exists(os.path.join(new_folder, "VID_1_vcover.jpg"))
        assert os.path.exists(os.path.join(new_folder, "VID_1.mp4"))
        assert not os.path.exists(cover_abs)
        assert not os.path.exists(video_abs)

        # DB 路径回写 / DB path write-back
        row = db.get_photo("VID_1_vcover")
        assert row["current_path"].replace("/", os.sep) == \
            os.path.join("白鹭", "3星_优选", "VID_1_vcover.jpg")

        # 视频归类清单同步改写（reset/flatten 的还原依据）
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)["entries"]
        videos = {e["video"] for e in entries}
        assert os.path.join(new_folder, "VID_1.mp4") in videos
        assert video_abs not in videos

        db.close()

    def test_species_change_moves_video_with_cover(self, tmp_path):
        """封面改鸟种：视频随封面进新鸟种目录。"""
        from core.rating_mover import change_bird_species
        db, root, cover_abs, video_abs, _manifest = \
            _make_cover_library(tmp_path)
        photo = db.get_photo("VID_1_vcover")
        photo["current_path"] = cover_abs
        photo["temp_jpeg_path"] = cover_abs

        changed = change_bird_species(
            root, photo, "苍鹭", "Grey Heron", "species-first", db,
            "VID_1_vcover")

        assert changed is True
        new_folder = os.path.join(root, "苍鹭", "2星_良好")
        assert os.path.exists(os.path.join(new_folder, "VID_1_vcover.jpg"))
        assert os.path.exists(os.path.join(new_folder, "VID_1.mp4"))
        db.close()


# ── rerate_v2：跳过封面行 ───────────────────────────────────────────────────

class TestRerateSkipsCovers:
    def test_load_photos_excludes_vcover_rows(self, tmp_path):
        from core.rerate_v2 import load_photos
        from tools.report_db import ReportDB

        root = str(tmp_path)
        db = ReportDB(root)
        db.insert_photo({
            "filename": "IMG_0001", "has_bird": 1, "rating": 2,
            "head_sharp": 300.0, "nima_score": 5.0, "confidence": 0.8,
        })
        db.insert_photo({
            "filename": "VID_1_vcover", "has_bird": 1, "rating": 2,
            "head_sharp": 0.0, "confidence": 0.8,
        })
        db.close()

        rows = load_photos(os.path.join(root, ".superpicky", "report.db"))
        names = {r["filename"] for r in rows}
        assert names == {"IMG_0001"}


# ── spb_flatten：伴生视频上移计划 ───────────────────────────────────────────

class TestFlattenCompanionVideos:
    def test_companion_video_planned_to_root(self, tmp_path):
        """整理子目录里的伴生视频被计划移回根（封面记录存在时）。"""
        from spb_flatten import _scan_companion_videos
        from tools.report_db import ReportDB

        root = str(tmp_path)
        db = ReportDB(root)
        db.insert_photo({
            "filename": "VID_1_vcover",
            "has_bird": 1, "rating": 2,
            "current_path": os.path.join("白鹭", "2星_良好", "VID_1_vcover.jpg"),
            "original_path": "VID_1_vcover.jpg",
        })
        _touch(os.path.join(root, "白鹭", "2星_良好", "VID_1_vcover.jpg"))
        _touch(os.path.join(root, "白鹭", "2星_良好", "VID_1.mp4"))

        plans = _scan_companion_videos(root, db)
        assert len(plans) == 1
        assert plans[0].to_root is True
        assert plans[0].new_name == "VID_1.mp4"
        assert plans[0].status == "rename"
        db.close()

    def test_root_video_and_unknown_stem_untouched(self, tmp_path):
        """根目录视频、无封面记录的视频不进计划。"""
        from spb_flatten import _scan_companion_videos
        from tools.report_db import ReportDB

        root = str(tmp_path)
        db = ReportDB(root)
        db.insert_photo({
            "filename": "VID_1_vcover",
            "has_bird": 1, "rating": 2,
            "current_path": os.path.join("白鹭", "2星_良好", "VID_1_vcover.jpg"),
            "original_path": "VID_1_vcover.jpg",
        })
        _touch(os.path.join(root, "VID_9.mp4"))                        # 根下
        _touch(os.path.join(root, "白鹭", "2星_良好", "OTHER.mp4"))    # 无封面记录

        assert _scan_companion_videos(root, db) == []
        db.close()

    def test_conflict_when_target_exists_at_root(self, tmp_path):
        """根下已有同名视频 → 计划标记冲突，不覆盖。"""
        from spb_flatten import _scan_companion_videos
        from tools.report_db import ReportDB

        root = str(tmp_path)
        db = ReportDB(root)
        db.insert_photo({
            "filename": "VID_1_vcover",
            "has_bird": 1, "rating": 2,
            "current_path": os.path.join("白鹭", "2星_良好", "VID_1_vcover.jpg"),
            "original_path": "VID_1_vcover.jpg",
        })
        _touch(os.path.join(root, "白鹭", "2星_良好", "VID_1.mp4"))
        _touch(os.path.join(root, "VID_1.mp4"))                        # 目标已存在

        plans = _scan_companion_videos(root, db)
        assert len(plans) == 1
        assert plans[0].status == "conflict"
        db.close()
