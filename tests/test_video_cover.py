# -*- coding: utf-8 -*-
"""
V-cover 封面帧提取单测 / Unit tests for core/video_cover.py

覆盖：选帧（YOLO 假模型注入，不依赖 torch）、无鸟回落中间帧、
旋转矫正、封面命名约定、JPEG 写盘 + EXIF DateTimeOriginal 注入、
一站式 generate_video_cover。

合成视频用 cv2.VideoWriter（mp4v）在 pytest tmp_path 下生成，绝不触碰
真实照片库（数据安全红线）。

Synthetic clips are created via cv2.VideoWriter under pytest tmp_path —
never touching any real photo library.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from core.video_cover import (
    _apply_rotation,
    build_cover_path,
    cover_stem,
    extract_cover_frame,
    generate_video_cover,
    pick_best_cover_timestamp,
    write_cover_jpeg,
)

pytestmark = pytest.mark.db

# 帧尺寸（宽, 高）/ Frame size (width, height)
_W, _H = 64, 48
_FPS = 10

# 纯色 BGR / Solid BGR colors
_BLUE = (255, 0, 0)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)


# ============================================================================
# 合成视频工厂 / Synthetic video factory
# ============================================================================

def _make_video(path, color_of_frame, frame_count: int = 30):
    """
    生成合成视频：按索引取色的纯色帧序列。

    参数:
        path: 输出视频路径（Path）
        color_of_frame: callable(idx) -> BGR 元组
        frame_count: 总帧数（默认 30，@10fps = 3 秒）

    返回:
        视频路径字符串；mp4v 编码器不可用时 pytest.skip。
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_W, _H))
    if not writer.isOpened():
        pytest.skip("mp4v VideoWriter 在本环境不可用 / mp4v writer unavailable")
    try:
        for idx in range(frame_count):
            frame = np.full((_H, _W, 3), color_of_frame(idx), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return str(path)


# ============================================================================
# 假 YOLO 模型 / Fake YOLO model（避免测试依赖 torch）
# ============================================================================

class _CpuNumpy:
    """模拟 torch Tensor 的 .cpu().numpy() 协议 / Mimic tensor .cpu().numpy()"""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _CpuNumpy(confs)
        self._confs = list(confs)

    def __len__(self):
        return len(self._confs)


class _FakeResult:
    def __init__(self, confs):
        self.boxes = _FakeBoxes(confs)


class RedFrameYolo:
    """
    规则假模型：画面以红色为主 → 检出一只"鸟"（置信度与红通道强度正相关）；
    其余帧无检出。
    """

    def __call__(self, frame, verbose=False, conf=0.5, classes=None):
        red_mean = float(frame[:, :, 2].mean())
        if red_mean > 100:
            return [_FakeResult([min(0.95, red_mean / 255.0 + 0.2)])]
        return [_FakeResult([])]


class NeverYolo:
    """永远无检出 / Never detects anything."""

    def __call__(self, frame, verbose=False, conf=0.5, classes=None):
        return [_FakeResult([])]


# ============================================================================
# 命名与旋转 / Naming and rotation
# ============================================================================

class TestNaming:
    def test_build_cover_path(self):
        video = os.path.join("lib", "VID_1234.MOV")
        assert build_cover_path(video) == os.path.join("lib", "VID_1234_vcover.jpg")

    def test_cover_stem(self):
        assert cover_stem(os.path.join("lib", "VID_1234.mp4")) == "VID_1234_vcover"


class TestRotation:
    def test_apply_0_noop_same_object(self):
        frame = np.full((4, 6, 3), 128, np.uint8)
        assert _apply_rotation(frame, 0) is frame

    def test_apply_90_clockwise(self):
        # 6x4 → 4x6，左上像素落到右上 / 6x4 -> 4x6, top-left moves to top-right
        frame = np.zeros((4, 6, 3), np.uint8)
        frame[0, 0] = 255
        rotated = _apply_rotation(frame, 90)
        assert rotated.shape == (6, 4, 3)
        assert tuple(rotated[0, 3]) == (255, 255, 255)

    def test_apply_180(self):
        frame = np.zeros((4, 6, 3), np.uint8)
        frame[0, 0] = 255
        rotated = _apply_rotation(frame, 180)
        assert tuple(rotated[3, 5]) == (255, 255, 255)

    def test_apply_270(self):
        frame = np.zeros((4, 6, 3), np.uint8)
        frame[0, 0] = 255
        rotated = _apply_rotation(frame, 270)
        assert rotated.shape == (6, 4, 3)
        assert tuple(rotated[5, 0]) == (255, 255, 255)

    def test_exiftool_rotation_negative_normalizes(self, tmp_path):
        """QuickTime 的 -90（逆时针）应归一化为 270（顺时针）。"""
        from core.video_cover import _read_rotation_with_exiftool
        video = _make_video(tmp_path / "v.mp4", lambda i: _BLUE)

        class _FakeCompleted:
            returncode = 0
            stdout = "-90\n"

        with patch.object(subprocess, "run", return_value=_FakeCompleted()):
            assert _read_rotation_with_exiftool("exiftool", video) == 270


# ============================================================================
# 写盘 / Cover writing
# ============================================================================

class TestWriteCoverJpeg:
    def test_writes_file_with_exif_datetime(self, tmp_path):
        frame = np.full((_H, _W, 3), _RED, np.uint8)
        cover = tmp_path / "VID_1_vcover.jpg"
        dt = datetime(2026, 9, 1, 8, 30, 0)

        assert write_cover_jpeg(frame, str(cover), dt) is True
        assert cover.exists()

        with Image.open(str(cover)) as img:
            exif = img.getexif()
            # 36867 = DateTimeOriginal
            assert exif[36867] == "2026:09:01 08:30:00"
            assert img.size == (_W, _H)

    def test_writes_without_date(self, tmp_path):
        frame = np.full((_H, _W, 3), _GREEN, np.uint8)
        cover = tmp_path / "VID_2_vcover.jpg"
        assert write_cover_jpeg(frame, str(cover), None) is True
        assert cover.exists()


# ============================================================================
# 选帧 / Picking
# ============================================================================

class TestPickBestCover:
    def test_picks_highest_conf_bird_frame(self, tmp_path):
        # 3 秒视频：0-1s 蓝、1-2s 绿、2-3s 红（红帧有"鸟"）
        video = _make_video(
            tmp_path / "bird.mp4",
            lambda i: _RED if i >= 20 else (_GREEN if i >= 10 else _BLUE),
        )
        result = pick_best_cover_timestamp(video, RedFrameYolo())
        assert result.sampled_frames == 3        # 0s/1s/2s 各一帧
        assert result.best_timestamp_sec == pytest.approx(2.0)
        assert result.best_conf > 0.5

    def test_no_bird_returns_none(self, tmp_path):
        video = _make_video(tmp_path / "empty.mp4", lambda i: _BLUE)
        result = pick_best_cover_timestamp(video, NeverYolo())
        assert result.best_timestamp_sec is None
        assert result.best_conf == 0.0
        assert result.duration_sec == pytest.approx(3.0, abs=0.2)

    def test_unopenable_raises_ioerror(self, tmp_path):
        with pytest.raises(IOError):
            pick_best_cover_timestamp(str(tmp_path / "nope.mp4"), NeverYolo())


# ============================================================================
# 提帧 / Extraction
# ============================================================================

class TestExtractCoverFrame:
    def test_extracts_frame_at_timestamp(self, tmp_path):
        video = _make_video(
            tmp_path / "seq.mp4",
            lambda i: _RED if i >= 20 else (_GREEN if i >= 10 else _BLUE),
        )
        frame = extract_cover_frame(video, 2.0)
        assert frame is not None
        assert frame.shape[:2] == (_H, _W)
        # mp4v 有压缩偏色，按通道主导性断言 / lossy codec: assert dominant channel
        assert frame[:, :, 2].mean() > frame[:, :, 0].mean()
        assert frame[:, :, 2].mean() > frame[:, :, 1].mean()

    def test_rotation_applied_on_extract(self, tmp_path):
        video = _make_video(tmp_path / "rot.mp4", lambda i: _BLUE)
        frame = extract_cover_frame(video, 0.0, rotation_deg=90)
        assert frame is not None
        assert frame.shape[:2] == (_W, _H)   # 48x64 → 64x48

    def test_bad_timestamp_returns_none(self, tmp_path):
        video = _make_video(tmp_path / "s.mp4", lambda i: _BLUE)
        assert extract_cover_frame(video, 9999.0) is None


# ============================================================================
# 一站式 / End-to-end generation
# ============================================================================

class TestGenerateVideoCover:
    def test_bird_video_cover_created(self, tmp_path):
        video = _make_video(
            tmp_path / "GX01.mp4",
            lambda i: _RED if i >= 20 else _BLUE,
        )
        cover = generate_video_cover(video, RedFrameYolo())
        assert cover is not None
        assert os.path.basename(cover) == "GX01_vcover.jpg"
        assert os.path.exists(cover)
        # 封面内容 = 红帧（2s 处）
        with Image.open(cover) as img:
            arr = np.asarray(img)
        assert arr[:, :, 0].mean() > arr[:, :, 1].mean()   # PIL 是 RGB

    def test_no_bird_video_falls_back_to_middle_frame(self, tmp_path):
        # 0-1.5s 蓝，1.5-3s 红：中间帧（1.5s）落在红区
        video = _make_video(
            tmp_path / "mid.mp4",
            lambda i: _RED if i >= 15 else _BLUE,
        )
        cover = generate_video_cover(video, NeverYolo())
        assert cover is not None
        with Image.open(cover) as img:
            arr = np.asarray(img)
        assert arr[:, :, 0].mean() > arr[:, :, 1].mean()

    def test_unreadable_video_returns_none(self, tmp_path):
        assert generate_video_cover(str(tmp_path / "bad.mov"), NeverYolo()) is None
