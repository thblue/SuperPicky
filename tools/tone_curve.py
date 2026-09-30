#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tone_curve — 暗图提亮的幂律伽马数学（单一实现，多消费方共用）。

SuperPicky 有两处需要对暗图做「按目标均值提亮」：
  1. tools/find_bird_util：RAW 转换出的预览缓存整体提亮（管线源头自愈）；
  2. birdid/bird_identifier：识鸟暗框重识别时对鸟框裁剪图提亮。
两者必须使用同一套 gamma 数学（只提亮不压暗、强度封顶），否则行为漂移、
幂等性破坏，故收敛到本模块。

tone_curve — power-law brightening math shared by all consumers.

Two places brighten dark images toward a target mean:
  1. tools/find_bird_util: whole-preview brightening at RAW conversion;
  2. birdid/bird_identifier: dark-crop brightening for BirdID retry.
They must share one gamma implementation (brighten-only, strength-capped)
so behavior cannot drift and idempotency holds; hence this module.

约定 / Conventions:
  - 亮度口径：Rec.601 爻度（OpenCV BGR2GRAY / PIL "L"），0-255。
    Luma convention: Rec.601 gray, 0-255.
  - 幂律映射 y = (x/255)^γ：γ<1 提亮，γ=1 恒等，永不返回 γ>1（不压暗）。
    Mapping y = (x/255)^gamma: gamma < 1 brightens; never > 1 (no darkening).
"""

from typing import Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # 极少数纯 CPU 精简环境 / rare CPU-only builds
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

# 默认参数：暗片判定线 / 提亮目标 / 最大提亮强度（γ 下限）。
# Defaults: dark threshold / target mean / max lift (gamma floor).
DEFAULT_DARK_MEAN = 90.0
DEFAULT_TARGET_MEAN = 115.0
DEFAULT_MIN_GAMMA = 0.45  # γ=0.45 ≈ 最亮 2.2 倍，防噪声过度放大 / ≈2.2x lift cap

# V5.9.1: 暗框重识别的「压死证据」判据与翻盘裕度——用真实库校准
# （2026-09-30 奥森/南堡样本，详见 scripts_dev 沙盒校准脚本）：
#   健康黑鸟（正确曝光的乌鸫/乌鸦）：crop p95 ≈ 173-203（羽轴高光拖尾）；
#   压死暗片：crop p95 ≈ 87-91（高光饥荒）。
# 只有「均值暗 且 p95 低」才值得提亮重试；黑鸟本身黑不欠曝，不该重试。
# 实测另证：伽马提亮在压死片上可能把 66% 的正确判定崩到 16%，故提亮
# 结果必须净胜 BRIGHTEN_WIN_MARGIN 个点才允许替换原判定。
# V5.9.1: "crush evidence" gate and flip margin for the dark-crop retry,
# calibrated on real library samples: healthy black birds keep a plumage
# highlight tail (crop p95 ~173-203) while crushed frames starve (p95
# ~87-91). Retry only when BOTH dark mean and low p95. Measurements also
# showed gamma can collapse a correct 66% ID to 16% on crushed frames,
# so a brightened result must win by BRIGHTEN_WIN_MARGIN points.
DEFAULT_CRUSH_P95 = 120.0
BRIGHTEN_WIN_MARGIN = 3.0

# V5.9.1: 预览整体提亮的高光护栏：画面里已有大片 ≥235 高光（背光天空/
# 舞台灯）时全局提亮只会削掉高光，鸟区域交给暗框重识别（Layer B）处理。
# Guard for whole-preview brightening: frames already carrying large
# highlight areas (backlit sky / stage light) must not be globally lifted
# (clipping); the bird itself is handled by the dark-crop retry.
DEFAULT_HIGHLIGHT_GUARD_FRAC = 0.15
DEFAULT_HIGHLIGHT_LEVEL = 235


def mean_luma_bgr(img_bgr: "np.ndarray") -> float:
    """
    计算 BGR 图像的平均亮度（Rec.601 灰度均值，0-255）。

    Compute the mean luma (Rec.601 gray mean, 0-255) of a BGR image.

    参数 / Parameters:
        img_bgr (np.ndarray): BGR 图像 / BGR image array.

    返回 / Returns:
        float: 平均亮度；输入无效返回 -1.0 / mean luma, or -1.0 if invalid.
    """
    if img_bgr is None or not hasattr(img_bgr, "shape") or img_bgr.size == 0:
        return -1.0
    if _CV2_AVAILABLE:
        return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    rgb = np.asarray(img_bgr, dtype=np.float32)[..., ::-1]
    return float((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                  + 0.114 * rgb[..., 2]).mean())


def mean_luma_pil(pil_image) -> float:
    """
    计算 PIL 图像的平均亮度（转 L 模式后取均值，0-255）。

    Compute the mean luma of a PIL image via its "L" mode (0-255).

    参数 / Parameters:
        pil_image (PIL.Image.Image): RGB/RGBA/L 图像 / PIL image.

    返回 / Returns:
        float: 平均亮度；输入无效返回 -1.0 / mean luma, or -1.0 if invalid.
    """
    if pil_image is None:
        return -1.0
    try:
        gray = pil_image.convert("L")
        return float(np.asarray(gray, dtype=np.float32).mean())
    except Exception:
        return -1.0


def compute_brighten_gamma(
    mean_luma: float,
    target_mean: float = DEFAULT_TARGET_MEAN,
    min_gamma: float = DEFAULT_MIN_GAMMA,
) -> float:
    """
    由当前均值与目标均值求等效幂律 gamma（只提亮不压暗，强度封顶）。

    Compute the power-law gamma bringing a mean luma to the target.
    Brighten-only: means at/above target return 1.0; lift strength is
    capped at min_gamma so noise in very dark frames stays bounded.

    参数 / Parameters:
        mean_luma (float): 当前平均亮度 / current mean luma (0-255).
        target_mean (float): 目标均值 / target mean luma.
        min_gamma (float): gamma 下限（最大提亮）/ lower bound of gamma.

    返回 / Returns:
        float: [min_gamma, 1.0] 内的 gamma / gamma within [min_gamma, 1.0].
    """
    mean_luma = max(float(mean_luma), 1.0)
    if mean_luma >= target_mean:
        return 1.0
    gamma = float(np.log(target_mean / 255.0) / np.log(mean_luma / 255.0))
    return float(np.clip(gamma, min_gamma, 1.0))


def build_gamma_lut(gamma: float) -> "np.ndarray":
    """
    构建幂律 gamma 的 256 项 uint8 查找表。

    Build the 256-entry uint8 LUT for a power-law gamma.

    参数 / Parameters:
        gamma (float): 幂指数 / power-law exponent.

    返回 / Returns:
        np.ndarray: 形状 (256,) 的 uint8 LUT / uint8 LUT of shape (256,).
    """
    return np.array(
        [((i / 255.0) ** float(gamma)) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )


def brighten_bgr_to_target(
    img_bgr: "np.ndarray",
    dark_mean: float = DEFAULT_DARK_MEAN,
    target_mean: float = DEFAULT_TARGET_MEAN,
    min_gamma: float = DEFAULT_MIN_GAMMA,
) -> Optional["np.ndarray"]:
    """
    暗图提亮：均值低于 dark_mean 时套目标均值伽马 LUT，返回提亮后的副本。

    幂等性：提亮后均值落在目标附近（约 105-120），对结果再调
    brighten_bgr_to_target 会因均值 ≥ dark_mean 直接返回 None，不会叠加。
    亮度达标时返回 None 表示「无需处理」，调用方沿用原图。

    Brighten a dark image toward the target mean via a gamma LUT and
    return the brightened copy. Idempotent: the result's mean sits near
    the target, so re-applying returns None (no change). Returns None
    when the input is already bright enough, meaning "keep as is".

    参数 / Parameters:
        img_bgr (np.ndarray): BGR 图像 / BGR image array.
        dark_mean (float): 暗图判定均值 / dark threshold on mean luma.
        target_mean (float): 提亮目标均值 / target mean luma.
        min_gamma (float): gamma 下限 / lower bound of gamma.

    返回 / Returns:
        Optional[np.ndarray]: 提亮后的 BGR 图；无需提亮或输入无效返回
                              None / brightened BGR, or None if not needed.
    """
    if not _CV2_AVAILABLE:
        return None
    mean = mean_luma_bgr(img_bgr)
    if mean < 0 or mean >= dark_mean:
        return None
    gamma = compute_brighten_gamma(mean, target_mean, min_gamma)
    if gamma >= 1.0:
        return None
    return cv2.LUT(img_bgr, build_gamma_lut(gamma))
