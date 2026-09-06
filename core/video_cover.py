#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V-cover 视频封面帧提取（视频进库方案 C1）

设计要点 / Design:
    - 每个视频抽一帧"代表性画面"落成真实 JPEG（<stem>_vcover.jpg），
      作为一条普通 photos 记录进入 report.db；视频文件作为伴生文件跟随封面移动。
    - 代表帧 = 全视频 YOLO 置信度最高的有鸟帧（复用 video_analyzer 的采样骨架）；
      无鸟视频取中间帧。
    - 帧必须做旋转矫正（竖拍 MOV/MP4 的 Rotation 元数据，cv2 解码不自动转正），
      并注入 EXIF DateTimeOriginal（取视频拍摄日期），保证浏览库按时间排序正确。
    - 本模块不做检测入库/文件移动（那是 core/video_stage.py 的职责），
      只负责"选帧 + 提帧 + 写封面文件"，纯函数化便于单测。

V-cover cover-frame extraction (stage C1 of the video-into-library design).
Picks the highest-confidence bird frame (middle frame for bird-less clips),
extracts it at full resolution with rotation correction, and writes the
<stem>_vcover.jpg cover file with EXIF DateTimeOriginal.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from constants import VIDEO_COVER_EXT, VIDEO_COVER_SUFFIX
from core.video_analyzer import (
    COCO_BIRD_CLASS_ID,
    _CaptureState,
    _decode_frame_grab,
    _decode_frame_seek,
    compute_frame_interval,
    pick_strategy,
)


# ============================================================================
# 数据结构 / Data structures
# ============================================================================

@dataclass(slots=True)
class CoverPickResult:
    """
    封面选帧结果 / Cover-frame picking result.

    Attributes:
        video_path: 视频绝对路径 / absolute video path
        duration_sec: 视频时长（秒）/ duration in seconds
        best_timestamp_sec: 置信度最高有鸟帧的时间戳；无有鸟帧时为 None
                            timestamp of the best bird frame; None if no bird frame
        best_conf: 最佳帧的 YOLO 置信度（0.0 当无有鸟帧）/ best YOLO confidence
        sampled_frames: 实际解码并推理的帧数 / number of decoded + inferred frames
    """
    video_path: str
    duration_sec: float
    best_timestamp_sec: Optional[float] = None
    best_conf: float = 0.0
    sampled_frames: int = 0


# ============================================================================
# 选帧 / Picking the representative frame
# ============================================================================

def pick_best_cover_timestamp(
    video_path: str,
    yolo_model,
    max_frames: int = 60,
    yolo_threshold: float = 0.5,
) -> CoverPickResult:
    """
    扫描视频抽帧并选取"代表性封面帧"：全视频 YOLO 置信度最高的有鸟帧。

    复用 video_analyzer 的自适应采样骨架（compute_frame_interval +
    seek/grab 混合解码），但只做 YOLO 检测，不做段合并/鸟种识别——
    鸟种识别由封面 JPEG 走照片管线完成（与照片同一模型同一阈值）。

    参数:
        video_path (str): 视频文件绝对路径
        yolo_model: 已加载的 YOLO 模型（ultralytics.YOLO 实例，或兼容其调用协议的对象）
        max_frames (int): 单视频抽帧上限（默认 60，与视频分析器一致）
        yolo_threshold (float): YOLO 置信度阈值（默认 0.5）

    返回:
        CoverPickResult: 选帧结果；best_timestamp_sec=None 表示无有鸟帧
        （调用方应回落到中间帧），duration_sec=0 表示视频完全无法解码。

    异常:
        IOError: 无法打开视频文件

    Scan sampled frames and pick the highest-YOLO-confidence bird frame as the
    cover. Reuses the analyzer's adaptive sampling skeleton but runs YOLO only
    (species ID happens later on the cover JPEG via the photo pipeline).

    Parameters:
        video_path: absolute video file path
        yolo_model: pre-loaded YOLO model (ultralytics.YOLO or compatible)
        max_frames: per-video sampling cap (default 60)
        yolo_threshold: YOLO confidence threshold (default 0.5)

    Return:
        CoverPickResult; best_timestamp_sec=None means no bird frame (caller
        falls back to the middle frame); duration_sec=0 means nothing decoded.

    Raises:
        IOError: cannot open the video file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频 / Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration_sec = frame_count / fps if fps > 0 else 0.0

        interval_sec = compute_frame_interval(duration_sec, max_frames)
        sample_times = np.arange(0.0, max(duration_sec, interval_sec), interval_sec)

        strategy = pick_strategy(fps, interval_sec)
        decode_fn = _decode_frame_grab if strategy == 'grab' else _decode_frame_seek
        state = _CaptureState(fps=fps, cursor=0)

        best_ts: Optional[float] = None
        best_conf = 0.0
        sampled = 0

        for t_sec in sample_times:
            ok, frame = decode_fn(cap, float(t_sec), state)
            if not ok or frame is None:
                continue
            sampled += 1

            results = yolo_model(
                frame,
                verbose=False,
                conf=yolo_threshold,
                classes=[COCO_BIRD_CLASS_ID],
            )
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                frame_max_conf = float(confs.max())
                # 严格大于：并列时保留更早的帧（视频叙事上更自然）
                # Strict > keeps the earlier frame on ties.
                if frame_max_conf > best_conf:
                    best_conf = frame_max_conf
                    best_ts = float(t_sec)
            del results

        return CoverPickResult(
            video_path=video_path,
            duration_sec=duration_sec,
            best_timestamp_sec=best_ts,
            best_conf=best_conf,
            sampled_frames=sampled,
        )
    finally:
        cap.release()


# ============================================================================
# 旋转 / Rotation handling
# ============================================================================

def _read_rotation_with_exiftool(exiftool_path: str, video_path: str) -> Optional[int]:
    """
    用 ExifTool 读视频 Rotation 复合标签，返回顺时针旋转角度（0/90/180/270）。

    QuickTime 容器可能给出负值（如 -90 表示逆时针 90°，等价于顺时针 270°），
    统一归一化到 [0, 270]。

    参数:
        exiftool_path (str): ExifTool 二进制路径
        video_path (str): 视频文件路径

    返回:
        Optional[int]: 旋转角度；读不到/无法解析时返回 None

    Read the Rotation composite tag via ExifTool, normalized to 0/90/180/270
    (clockwise). Negative QuickTime values (e.g. -90) map to 270.
    Returns None when unreadable.
    """
    cmd = [exiftool_path, '-s3', '-Rotation', video_path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, encoding='utf-8'
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = float(line)
        except ValueError:
            continue
        deg = int(raw) % 360
        return deg
    return None


def get_video_rotation(video_path: str, exiftool_path: Optional[str] = None) -> int:
    """
    获取视频的显示旋转角度（顺时针，0/90/180/270）。

    优先级：
        1. ExifTool Rotation 复合标签（最可靠，覆盖 QuickTime/MP4 metadata）
        2. OpenCV CAP_PROP_ORIENTATION_META（FFMPEG 后端，ExifTool 缺失时兜底）
        3. 0（不旋转）

    参数:
        video_path (str): 视频文件路径
        exiftool_path (Optional[str]): ExifTool 路径；None 自动定位

    返回:
        int: 0 / 90 / 180 / 270

    Get the video's display rotation (clockwise, one of 0/90/180/270).
    Priority: ExifTool Rotation tag → cv2 CAP_PROP_ORIENTATION_META → 0.
    """
    if exiftool_path is None:
        # 复用 video_organizer 的定位逻辑（内部优先 ExifToolManager 单例）
        # Reuse the locator from video_organizer (prefers the manager singleton).
        from tools.video_organizer import _locate_exiftool
        try:
            exiftool_path = _locate_exiftool()
        except Exception:
            exiftool_path = None

    if exiftool_path and os.path.exists(exiftool_path):
        deg = _read_rotation_with_exiftool(exiftool_path, video_path)
        if deg is not None and deg != 0:
            return deg

    # Fallback: OpenCV 自带的方向元数据（部分构建/后端支持）
    # Fallback: cv2's orientation meta (supported by some builds/backends).
    try:
        cap = cv2.VideoCapture(video_path)
        try:
            if cap.isOpened():
                meta = cap.get(getattr(cv2, 'CAP_PROP_ORIENTATION_META', 0))
                if meta:
                    return int(meta) % 360
        finally:
            cap.release()
    except Exception:
        pass

    return 0


def _apply_rotation(frame: np.ndarray, rotation_deg: int) -> np.ndarray:
    """
    按顺时针旋转角度矫正帧。

    参数:
        frame (np.ndarray): BGR 帧
        rotation_deg (int): 顺时针角度（0/90/180/270，其他值按就近的合法值处理）

    返回:
        np.ndarray: 矫正后的帧（0° 时原样返回，不复制）

    Rotate a frame clockwise by the given display rotation.
    Returns the input unchanged (no copy) for 0°.
    """
    if rotation_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============================================================================
# 提帧与写盘 / Frame extraction and cover writing
# ============================================================================

def extract_cover_frame(
    video_path: str,
    timestamp_sec: float,
    rotation_deg: int = 0,
) -> Optional[np.ndarray]:
    """
    在指定时间点提取一帧全分辨率画面并做旋转矫正。

    参数:
        video_path (str): 视频文件路径
        timestamp_sec (float): 目标时间点（秒）
        rotation_deg (int): 显示旋转角度（顺时针，0/90/180/270）

    返回:
        Optional[np.ndarray]: BGR 帧；解码失败时 None

    Extract the full-resolution frame at the given timestamp with rotation
    applied. Returns None on decode failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return _apply_rotation(frame, rotation_deg)
    finally:
        cap.release()


def build_cover_path(video_path: str) -> str:
    """
    构建封面 JPEG 路径：<视频所在目录>/<视频名>_vcover.jpg

    参数:
        video_path (str): 视频文件路径

    返回:
        str: 封面文件绝对/相对路径（与入参同基调）

    Build the cover JPEG path: <video dir>/<stem>_vcover.jpg.
    """
    directory = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(directory, stem + VIDEO_COVER_SUFFIX + VIDEO_COVER_EXT)


def cover_stem(video_path: str) -> str:
    """
    视频对应的封面主名（无扩展名），即 report.db photos.filename 的取值。

    参数:
        video_path (str): 视频文件路径

    返回:
        str: 如 "VID_1234_vcover"

    The cover stem (no extension) used as report.db photos.filename.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return stem + VIDEO_COVER_SUFFIX


def companion_video_for_cover(cover_abs_path: str) -> Optional[str]:
    """
    取封面 JPEG 的伴生视频路径（同目录、去 _vcover 后缀 + 视频扩展名）。

    封面 <stem>_vcover.jpg 的视频本体是同目录的 <stem>.mp4/.mov/.m4v。
    这是全仓库"视频跟随封面"约定的唯一实现——rating_mover 移动、
    浏览器删除/打开视频、拍平还原都经由此函数定位伴生视频。

    参数:
        cover_abs_path (str): 封面 JPEG 绝对路径（或其他照片主文件路径）

    返回:
        Optional[str]: 伴生视频绝对路径；入参不是封面或找不到视频时 None

    Return the companion video path for a cover JPEG (same directory,
    stem minus the _vcover suffix plus a video extension), or None.
    """
    basename = os.path.basename(cover_abs_path)
    stem, ext = os.path.splitext(basename)
    if ext.lower() not in ('.jpg', '.jpeg'):
        return None
    if not stem.endswith(VIDEO_COVER_SUFFIX):
        return None
    video_stem = stem[:-len(VIDEO_COVER_SUFFIX)]
    directory = os.path.dirname(cover_abs_path)
    from constants import VIDEO_EXTENSIONS_ALL
    for vext in VIDEO_EXTENSIONS_ALL:
        candidate = os.path.join(directory, video_stem + vext)
        if os.path.exists(candidate):
            return candidate
    return None


def write_cover_jpeg(
    frame_bgr: np.ndarray,
    cover_path: str,
    capture_date: Optional[datetime],
    jpeg_quality: int = 95,
) -> bool:
    """
    把封面帧写成 JPEG 文件，并注入 EXIF DateTimeOriginal（拍摄日期）。

    用 PIL 编码（cv2.imwrite 不支持写 EXIF）；日期取视频拍摄日期
    （video_organizer.get_video_capture_date 的 ExifTool→mtime→now 链），
    保证封面在浏览库/排序中与视频同时代。

    参数:
        frame_bgr (np.ndarray): BGR 帧（已做旋转矫正）
        cover_path (str): 输出 JPEG 路径
        capture_date (Optional[datetime]): 拍摄日期；None 则不注入 EXIF
        jpeg_quality (int): JPEG 质量（默认 95）

    返回:
        bool: 是否成功写盘

    Write the cover frame as a JPEG with EXIF DateTimeOriginal injected
    (PIL encoding, since cv2.imwrite cannot write EXIF). Returns True on
    success.
    """
    try:
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        save_kwargs = {"format": "JPEG", "quality": jpeg_quality}
        if capture_date is not None:
            exif = Image.Exif()
            # 36867 = DateTimeOriginal, 306 = DateTime（部分读取方只认其中一个）
            # 36867 = DateTimeOriginal, 306 = DateTime (some readers check either).
            dt_str = capture_date.strftime("%Y:%m:%d %H:%M:%S")
            exif[36867] = dt_str
            exif[306] = dt_str
            save_kwargs["exif"] = exif

        img.save(cover_path, **save_kwargs)
        return True
    except Exception:
        return False


def generate_video_cover(
    video_path: str,
    yolo_model,
    max_frames: int = 60,
    yolo_threshold: float = 0.5,
) -> Optional[str]:
    """
    一站式生成视频封面：选帧 →（无鸟时回落中间帧）→ 提帧矫正 → 写盘。

    参数:
        video_path (str): 视频文件绝对路径
        yolo_model: 已加载的 YOLO 模型
        max_frames (int): 抽帧上限
        yolo_threshold (float): YOLO 置信度阈值

    返回:
        Optional[str]: 封面 JPEG 路径；任何环节失败（打不开/解不出帧/写盘失败）
        返回 None，由调用方记日志跳过该视频。

    One-shot cover generation: pick frame (middle frame for bird-less clips)
    → extract with rotation → write JPEG. Returns the cover path, or None on
    any failure (caller logs and skips the video).
    """
    from tools.video_organizer import get_video_capture_date

    try:
        pick = pick_best_cover_timestamp(
            video_path, yolo_model, max_frames=max_frames, yolo_threshold=yolo_threshold)
    except IOError:
        # 视频打不开（损坏/编码不支持）：按契约返回 None，调用方记日志跳过
        # Unopenable video: return None per contract; caller logs and skips.
        return None

    timestamp = pick.best_timestamp_sec
    if timestamp is None:
        # 无有鸟帧：取中间帧（视频内容上最有"代表性"的近似）
        # No bird frame: fall back to the middle of the clip.
        if pick.duration_sec <= 0:
            return None
        timestamp = pick.duration_sec / 2.0

    rotation = get_video_rotation(video_path)
    frame = extract_cover_frame(video_path, timestamp, rotation)
    if frame is None:
        return None

    try:
        capture_date = get_video_capture_date(video_path)
    except Exception:
        capture_date = None

    cover_path = build_cover_path(video_path)
    if not write_cover_jpeg(frame, cover_path, capture_date):
        return None
    return cover_path
