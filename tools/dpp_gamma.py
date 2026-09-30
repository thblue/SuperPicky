# -*- coding: utf-8 -*-
"""
DPP 伽马编辑的共享读取与应用（W1 / V5.9.4）。

Canon DPP「工具调色板 → 伽马调整」把 recipe 以 CanonVRD/DR4 二进制
trailer 写回 CR3（exiftool -CanonVRD:all 可读）。本模块把「批量读
recipe → 幂律 LUT → 应用到缓存预览」收敛为单一实现，供两处消费：

1. 跑批内 M1（core/photo_processor._convert_raws → raw_to_jpeg）：
   本次新抽取的预览按「人工编辑优先」先套 DPP LUT，并跳过自动提亮
   （人工与自动绝不叠加）；
2. 批后脚本 scripts_dev/refresh_gamma_previews.py：老批次回补与跑批
   之后新做的 DPP 编辑的幂等刷新（每次从 CR3 重抽原始预览再套 LUT）。

数学：GammaMidPoint m（负值=提亮，实测 21 张全负）→ 幂律
gamma g = 2^m，LUT: y = (x/255)^g；GammaWhitePoint 正值（高光端扩展）
暂忽略——主视觉量来自中点。未编辑的文件没有 CanonVRD 伽马区，
不会出现在读取结果里。

Shared reading and application of DPP gamma edits (W1 / V5.9.4).

Canon DPP's "tool palette -> gamma adjustment" writes the recipe back
into the CR3 as a CanonVRD/DR4 binary trailer (readable via
exiftool -CanonVRD:all). This module is the single home for
"batch-read recipes -> power-law LUT -> apply to cached preview",
consumed by both the in-batch M1 path (human edit first, auto-brighten
never stacks on top) and the post-batch refresh script (idempotent
re-extract + LUT for old batches and edits made after processing).

Math: GammaMidPoint m (negative = brighter) maps to gamma g = 2^m,
LUT: y = (x/255)^g; the positive GammaWhitePoint (highlight stretch) is
ignored for now — the visual bulk comes from the midpoint. Unedited
files have no CanonVRD gamma section and never appear in the results.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def read_gamma_params(cr3_paths: List[str],
                      exiftool: str) -> Dict[str, Tuple[str, str]]:
    """
    批量读 CR3 的 CanonVRD 伽马参数。

    大目录（如 640 张 UNC 路径）会超出 Windows 32K 命令行上限
    （WinError 206），因此按块分批调用 exiftool 再合并结果。

    参数:
        cr3_paths (List[str]): CR3 绝对路径（非 CR3 会被 exiftool 过滤）
        exiftool (str): exiftool 可执行文件路径

    返回:
        Dict[str, Tuple[str, str]]: {文件名前缀: (中点值, 白点值)}；
        未编辑（无 CanonVRD 伽马区）的文件不在字典里

    Batch-read CanonVRD gamma parameters from CR3 files.

    Parameters:
        cr3_paths (List[str]): absolute CR3 paths (non-CR3 entries are
            filtered out by exiftool itself).
        exiftool (str): path to the exiftool executable.

    Returns:
        Dict[str, Tuple[str, str]]: {file prefix: (midpoint, whitepoint)};
        unedited files (no CanonVRD gamma section) are absent.
    """

    def _read_batch(batch: List[str]) -> Dict[str, Tuple[str, str]]:
        """读单批文件，返回 {前缀: (中点, 白点)}。/ Read one batch."""
        creationflags = subprocess.CREATE_NO_WINDOW \
            if sys.platform.startswith('win') else 0
        out = subprocess.run(
            [exiftool, '-s3', '-csv', '-FileName',
             '-CanonVRD:GammaMidPoint', '-CanonVRD:GammaWhitePoint',
             '-ext', 'CR3', *batch],
            capture_output=True, timeout=600, creationflags=creationflags)
        lines = out.stdout.decode('utf-8', 'replace').strip().splitlines()
        if len(lines) < 2:
            return {}
        header = lines[0].split(',')
        ix_fn = header.index('FileName')
        ix_mid = header.index('GammaMidPoint') if 'GammaMidPoint' in header else -1
        ix_wp = header.index('GammaWhitePoint') if 'GammaWhitePoint' in header else -1
        batch_result: Dict[str, Tuple[str, str]] = {}
        for ln in lines[1:]:
            cols = ln.split(',')
            fn = cols[ix_fn] if ix_fn < len(cols) else ''
            mid = cols[ix_mid].strip() if 0 <= ix_mid < len(cols) else ''
            wp = cols[ix_wp].strip() if 0 <= ix_wp < len(cols) else ''
            if mid:
                prefix = os.path.splitext(fn)[0]
                batch_result[prefix] = (mid, wp)
        return batch_result

    # 每批 100 个文件：单条命令约 8-9K 字符，远低于 Windows 32K 上限
    # 100 files per batch: ~8-9K chars per command, well under the 32K limit
    result: Dict[str, Tuple[str, str]] = {}
    chunk_size = 100
    for i in range(0, len(cr3_paths), chunk_size):
        result.update(_read_batch(cr3_paths[i:i + chunk_size]))
    return result


def read_dpp_gamma_map(cr3_paths: List[str],
                       exiftool: str) -> Dict[str, float]:
    """
    跑批预读入口：{前缀: 伽马中点浮点值}，只含可解析的已编辑文件。

    是 read_gamma_params 的管线友好包装：中点值解析失败或为空的条目
    直接丢弃（不会让单个坏值阻断整批转换）。

    参数:
        cr3_paths (List[str]): CR3 绝对路径列表
        exiftool (str): exiftool 可执行文件路径

    返回:
        Dict[str, float]: {前缀: GammaMidPoint 浮点值}；无编辑=空字典

    Pipeline-friendly wrapper: {prefix: parsed midpoint float}, dropping
    entries whose midpoint cannot be parsed so a single bad value never
    blocks the whole conversion batch. Empty dict when nothing is edited.
    """
    gamma_map: Dict[str, float] = {}
    for prefix, (mid, _wp) in read_gamma_params(cr3_paths, exiftool).items():
        try:
            gamma_map[prefix] = float(mid)
        except (TypeError, ValueError):
            continue
    return gamma_map


def build_lut(mid_point: float) -> np.ndarray:
    """
    DPP 伽马中点 → 256 级 LUT。

    DPP 中点滑块负值=提亮（实测 21 张全负、用户描述「亮度提高」）。
    幂律近似：g = 2^m，y = (x/255)^g。m=-1.65 → g≈0.32，中灰 0.5 → 0.80。

    参数:
        mid_point (float): CanonVRD GammaMidPoint 原始值

    返回:
        np.ndarray: uint8[256] 查找表

    DPP midpoint slider -> 256-entry LUT. Negative mid = brighter
    (measured: all 21 edited samples negative). Power-law approximation:
    g = 2^m, y = (x/255)^g. m=-1.65 gives g≈0.32, mid-gray 0.5 -> 0.80.

    Parameters:
        mid_point (float): raw CanonVRD GammaMidPoint value.

    Returns:
        np.ndarray: uint8[256] lookup table.
    """
    g = 2.0 ** mid_point
    x = np.arange(256, dtype=np.float64) / 255.0
    y = np.clip(x ** g, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def read_bgr(path: str) -> Optional[np.ndarray]:
    """
    中文/UNC 安全读图。/ UNC-safe image read. """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def apply_lut_to_jpeg(jpg_path: str, lut: np.ndarray) -> Optional[float]:
    """
    对缓存 JPEG 应用 LUT 并写回（原地刷新，JPEG 质量参数 95）。

    参数:
        jpg_path (str): temp_preview 缓存 JPEG 路径
        lut (np.ndarray): uint8[256] 查找表

    返回:
        Optional[float]: 应用后的平均亮度（0-255）；读写失败返回 None

    Apply a LUT to a cached JPEG and write it back in place (JPEG
    quality 95).

    Parameters:
        jpg_path (str): temp_preview cache JPEG path.
        lut (np.ndarray): uint8[256] lookup table.

    Returns:
        Optional[float]: post-application mean luma (0-255), or None on
        any read/write failure.
    """
    img = read_bgr(jpg_path)
    if img is None:
        return None
    out = cv2.LUT(img, lut)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None
    buf.tofile(jpg_path)
    return float(out.mean())
