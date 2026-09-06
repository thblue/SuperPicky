# -*- coding: utf-8 -*-
"""
V-cover 视频处理阶段单测 / Unit tests for core/video_stage.py

覆盖：顶层视频扫描、定星规则（有鸟2星/低置信0/无鸟-1，不进配额池）、
photos 行 + bird_detections 入库、封面+视频伴生归类、幂等跳过、
flat 布局不动文件、sidecar 导出（有鸟出/无鸟不出）、
PhotoProcessor._scan_files 跳过 *_vcover.jpg。

重依赖全部注入/替换：YOLO 与 BirdID 用假对象，detect_and_draw_birds /
generate_video_cover 用 monkeypatch 替换——不加载 torch，跑在 tmp_path 下。

All heavy deps are injected or monkeypatched (no torch); everything runs
under pytest tmp_path, never touching real libraries.
"""
from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.photo_processor import PhotoProcessor, ProcessingSettings
from core.video_stage import (
    _rating_for_cover,
    find_top_level_videos,
    process_directory_videos,
)
from tools.report_db import ReportDB

pytestmark = pytest.mark.db

_W, _H = 64, 48


# ============================================================================
# 测试替身 / Test doubles
# ============================================================================

class _NoopYolo:
    """占位 YOLO（generate_video_cover 被替换后不会真正调用它）。"""

    def __call__(self, *args, **kwargs):
        raise AssertionError("占位 YOLO 不应被调用 / placeholder must not be called")


def _fake_generate_cover(video_path, yolo_model, max_frames=60, yolo_threshold=0.5):
    """替身封面生成：写一张纯红 JPEG（有鸟视频场景的可识别画面）。"""
    from core.video_cover import build_cover_path
    cover = build_cover_path(video_path)
    frame = np.full((_H, _W, 3), (0, 0, 220), np.uint8)
    cv2.imwrite(cover, frame)
    return cover


def _fake_detect(found_bird, confidence, all_birds=None, rescued=False):
    """构造 detect_and_draw_birds 的 11 元组返回。"""

    def _detect(image_path, model, output_path, dir, ui_settings, i18n=None,
                skip_nima=False, focus_point=None, report_db=None,
                decoded_image=None):
        return (found_bird, found_bird, confidence, 100.0, None,
                (0, 0, 32, 32), (_W, _H), None, len(all_birds or []),
                rescued, all_birds or [])

    return _detect


def _fake_identify(cn="白鹭", en="Little Egret", conf=88.0):
    def _identify(image_path, *args, **kwargs):
        return {
            'success': True,
            'results': [{
                'cn_name': cn, 'en_name': en,
                'scientific_name': 'Egretta garzetta',
                'confidence': conf, 'class_id': 123,
                'gbif_rarity_100': 12.0, 'iucn_category': 'LC',
                'aesthetic_index': 55.0, 'china_protection_level': None,
            }],
        }

    return _identify


def _make_video(directory, name: str) -> str:
    path = os.path.join(str(directory), name)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (_W, _H))
    if not writer.isOpened():
        pytest.skip("mp4v VideoWriter 在本环境不可用 / mp4v writer unavailable")
    try:
        for _ in range(15):
            writer.write(np.full((_H, _W, 3), (200, 120, 30), np.uint8))
    finally:
        writer.release()
    return path


def _fake_config(layout="species-first", multibird=False):
    return SimpleNamespace(
        folder_layout=layout,
        multibird_enabled=multibird,
        multibird_min_area_ratio=0.01,
        multibird_species_threshold=35.0,
        mainbird_rare_min_conf=70.0,
        mainbird_rare_gbif=50.0,
    )


def _settings(auto_identify=True, ai_confidence=40):
    return ProcessingSettings(
        ai_confidence=ai_confidence,
        birdid_confidence_threshold=50.0,
        auto_identify=auto_identify,
    )


# ============================================================================
# 顶层扫描 / Top-level scan
# ============================================================================

class TestFindTopLevelVideos:
    def test_finds_only_top_level_sorted(self, tmp_path):
        (tmp_path / "b.mp4").write_bytes(b"x")
        (tmp_path / "A.MOV").write_bytes(b"x")
        (tmp_path / "._evil.MOV").write_bytes(b"x")     # AppleDouble 跳过
        (tmp_path / "c.txt").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.mp4").write_bytes(b"x")            # 不递归

        videos = find_top_level_videos(str(tmp_path))
        names = [os.path.basename(v) for v in videos]
        assert names == ["A.MOV", "b.mp4"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert find_top_level_videos(str(tmp_path / "nope")) == []


# ============================================================================
# 定星 / Rating
# ============================================================================

class TestRatingForCover:
    def test_no_bird(self):
        assert _rating_for_cover(False, 0.9, 40) == -1

    def test_bird_fixed_two_stars(self):
        assert _rating_for_cover(True, 0.9, 40) == 2

    def test_bird_low_confidence_zero(self):
        assert _rating_for_cover(True, 0.3, 40) == 0


# ============================================================================
# 主流程 / Main flow
# ============================================================================

class TestProcessDirectoryVideos:
    def _run(self, tmp_path, *, detect=None, identify=None, config=None,
             organize=True, settings=None):
        db = ReportDB(str(tmp_path))
        try:
            stats = process_directory_videos(
                dir_path=str(tmp_path),
                settings=settings or _settings(),
                config=config or _fake_config(),
                report_db=db,
                organize_files=organize,
                yolo_model=_NoopYolo(),
                identify_fn=identify or _fake_identify(),
                log=lambda msg, level="info": None,
            )
        finally:
            db.close()
        # 重开核对落库结果 / reopen to assert persisted state
        check = sqlite3.connect(str(tmp_path / ".superpicky" / "report.db"))
        check.row_factory = sqlite3.Row
        try:
            yield stats, check
        finally:
            check.close()

    def test_bird_video_full_flow(self, tmp_path, monkeypatch):
        """有鸟视频：2星入库 + 定种 + 封面/视频一起归入 {鸟种}/2星_良好 + sidecar。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover", _fake_generate_cover)
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(True, 0.82))

        video = _make_video(tmp_path, "VID_0001.mp4")
        for stats, db in self._run(tmp_path):
            # 阶段统计 / stage stats
            assert stats.total == 1
            assert stats.covers_created == 1
            assert stats.has_bird == 1
            assert stats.adopted == 1
            assert stats.organized == 1
            assert stats.failed == 0

            # photos 行 / photos row
            row = db.execute(
                "SELECT * FROM photos WHERE filename='VID_0001_vcover'"
            ).fetchone()
            assert row is not None
            assert row["rating"] == 2
            assert row["has_bird"] == 1
            assert abs(row["confidence"] - 0.82) < 1e-6
            assert row["bird_species_cn"] == "白鹭"
            assert row["bird_species_en"] == "Little Egret"
            assert row["birdid_confidence"] == 88.0
            assert "视频封面" in (row["caption"] or "")
            assert row["current_path"] == os.path.join(
                "白鹭", "2星_良好", "VID_0001_vcover.jpg")

            # 文件落位 / files on disk
            assert (tmp_path / "白鹭" / "2星_良好" / "VID_0001_vcover.jpg").exists()
            assert (tmp_path / "白鹭" / "2星_良好" / "VID_0001.mp4").exists()
            assert not (tmp_path / "VID_0001.mp4").exists()

            # 归类清单（reset 还原依据）/ manifest for reset
            manifest_path = tmp_path / ".superpicky_video_manifest.json"
            assert manifest_path.exists()
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]
            assert len(entries) == 2   # 封面 + 视频 / cover + video

            # sidecar（有鸟封面导出）/ sidecar exported
            sidecar = (tmp_path / ".superpicky" / "meta"
                       / "VID_0001_vcover.json")
            assert sidecar.exists()
            assert stats.sidecars == 1

    def test_no_bird_video_rating_minus_one_no_sidecar(self, tmp_path, monkeypatch):
        """无鸟视频：-1、has_bird=0、无 sidecar、进 其他鸟类/0星_放弃。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover", _fake_generate_cover)
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(False, 0.0))

        _make_video(tmp_path, "GX01.mp4")
        for stats, db in self._run(tmp_path):
            assert stats.no_bird == 1
            assert stats.organized == 1

            row = db.execute(
                "SELECT * FROM photos WHERE filename='GX01_vcover'"
            ).fetchone()
            assert row["rating"] == -1
            assert row["has_bird"] == 0

            # 无鸟不导出 sidecar（V5.5 规则对封面同样生效）
            assert not (tmp_path / ".superpicky" / "meta"
                        / "GX01_vcover.json").exists()
            # 但视频与封面仍一起移动（与无鸟照片行为一致）
            assert (tmp_path / "其他鸟类" / "0星_放弃" / "GX01.mp4").exists()

    def test_low_confidence_bird_gets_zero_star(self, tmp_path, monkeypatch):
        """有鸟但置信度 0.3 < ai_confidence 40 → 0 星，不写鸟种列。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover", _fake_generate_cover)
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(True, 0.3))

        _make_video(tmp_path, "LOW.mp4")
        for stats, db in self._run(tmp_path):
            assert stats.low_conf == 1
            row = db.execute(
                "SELECT * FROM photos WHERE filename='LOW_vcover'"
            ).fetchone()
            assert row["rating"] == 0
            assert (row["bird_species_cn"] or "") == ""
            assert (tmp_path / "其他鸟类" / "0星_放弃" / "LOW.mp4").exists()

    def test_rescued_bird_exempt_from_confidence_gate(self, tmp_path, monkeypatch):
        """救回的鸟豁免置信度门槛：conf 0.3 也 2 星且照常识别（对齐照片语义）。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover", _fake_generate_cover)
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(True, 0.3, rescued=True))

        _make_video(tmp_path, "RSC.mp4")
        for stats, db in self._run(tmp_path):
            assert stats.has_bird == 1
            assert stats.low_conf == 0
            row = db.execute(
                "SELECT * FROM photos WHERE filename='RSC_vcover'"
            ).fetchone()
            assert row["rating"] == 2
            assert row["bird_species_cn"] == "白鹭"

    def test_species_below_adoption_threshold_not_in_columns(
            self, tmp_path, monkeypatch):
        """识别置信 30% < 采纳线 50%：不入鸟种列，rating 仍 2 星。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover", _fake_generate_cover)
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(True, 0.8))

        _make_video(tmp_path, "SP.mp4")
        for stats, db in self._run(tmp_path, identify=_fake_identify(conf=30.0)):
            assert stats.has_bird == 1
            assert stats.adopted == 0
            row = db.execute(
                "SELECT * FROM photos WHERE filename='SP_vcover'"
            ).fetchone()
            assert row["rating"] == 2
            assert (row["bird_species_cn"] or "") == ""
            # 未定种 2 星 → 其他鸟类/2星_良好
            assert (tmp_path / "其他鸟类" / "2星_良好" / "SP.mp4").exists()

    def test_idempotent_second_run_skips(self, tmp_path, monkeypatch):
        """重跑：已有封面记录的视频被跳过，不重复生成（flat 布局视频留在顶层）。"""
        import core.video_stage as vs
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(True, 0.8))

        calls = []

        def _counting_cover(video_path, yolo_model, max_frames=60, yolo_threshold=0.5):
            calls.append(video_path)
            return _fake_generate_cover(video_path, yolo_model)

        monkeypatch.setattr(vs, "generate_video_cover", _counting_cover)

        _make_video(tmp_path, "RUN.mp4")
        for _stats, _db in self._run(tmp_path, config=_fake_config(layout="flat")):
            pass
        assert len(calls) == 1
        for stats, _db in self._run(tmp_path, config=_fake_config(layout="flat")):
            assert stats.skipped == 1
            assert stats.covers_created == 0
        assert len(calls) == 1   # 第二次没有再生成 / not regenerated

    def test_flat_layout_no_move(self, tmp_path, monkeypatch):
        """flat 布局：封面生成+入库，但文件不动。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover", _fake_generate_cover)
        monkeypatch.setattr("core.ai_model.detect_and_draw_birds",
                            _fake_detect(True, 0.8))

        video = _make_video(tmp_path, "FLAT.mp4")
        for stats, _db in self._run(tmp_path, config=_fake_config(layout="flat")):
            assert stats.organized == 0
            assert (tmp_path / "FLAT.mp4").exists()
            assert (tmp_path / "FLAT_vcover.jpg").exists()

    def test_cover_generation_failure_skips_video(self, tmp_path, monkeypatch):
        """封面生成失败（返回 None）：记失败，视频原地不动、不入库。"""
        import core.video_stage as vs
        monkeypatch.setattr(vs, "generate_video_cover",
                            lambda *a, **k: None)

        _make_video(tmp_path, "BAD.mp4")
        for stats, db in self._run(tmp_path):
            assert stats.failed == 1
            row = db.execute(
                "SELECT * FROM photos WHERE filename='BAD_vcover'"
            ).fetchone()
            assert row is None
            assert (tmp_path / "BAD.mp4").exists()


# ============================================================================
# _scan_files 跳过封面 / photo pipeline skips covers
# ============================================================================

class TestScanFilesSkipsCovers:
    def test_vcover_jpg_not_scanned(self, tmp_path):
        """封面 JPEG 不进照片扫描（不进 RAW/JPG 字典，不进待删清单）。"""
        (tmp_path / "IMG_0001.jpg").write_bytes(b"photo")
        (tmp_path / "VID_0001_vcover.jpg").write_bytes(b"cover")
        (tmp_path / "VID_0002_vcover.JPG").write_bytes(b"cover upper")

        processor = PhotoProcessor(
            dir_path=str(tmp_path),
            settings=ProcessingSettings(),
            callbacks=None,
        )
        raw_dict, jpg_dict, files_tbr = processor._scan_files()

        assert set(jpg_dict) == {"IMG_0001"}
        assert "VID_0001_vcover" not in jpg_dict
        assert "VID_0002_vcover" not in jpg_dict
        assert files_tbr == ["IMG_0001.jpg"]
