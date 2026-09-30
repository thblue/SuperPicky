#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tone_curve 提白数学与 find_bird_util 预览提亮的幂等性测试。

Covers the shared brightening math (brighten-only, strength cap) and the
preview-file brightening wrapper's luma-guard idempotency.
"""

import os

import numpy as np
import pytest

from tools.tone_curve import (
    DEFAULT_DARK_MEAN,
    DEFAULT_MIN_GAMMA,
    DEFAULT_TARGET_MEAN,
    brighten_bgr_to_target,
    build_gamma_lut,
    compute_brighten_gamma,
    mean_luma_bgr,
    mean_luma_pil,
)


def _random_bgr(rng, mean_target):
    """生成指定均值附近的随机 BGR 图（uint8）。"""
    arr = rng.normal(mean_target, 25, (48, 64, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)


class TestComputeBrightenGamma:
    def test_bright_image_returns_identity(self):
        """亮于目标均值 → γ=1 恒等（只提亮不压暗）。"""
        assert compute_brighten_gamma(165) == 1.0
        assert compute_brighten_gamma(DEFAULT_TARGET_MEAN) == 1.0

    def test_dark_image_brightens(self):
        """暗图 → γ<1。"""
        assert compute_brighten_gamma(60) < 1.0

    def test_strength_capped(self):
        """极暗图 γ 不低于下限（最大提亮封顶）。"""
        assert compute_brighten_gamma(15) == DEFAULT_MIN_GAMMA

    def test_result_within_bounds(self):
        rng = np.random.default_rng(7)
        for mean in rng.uniform(1, 254, 50):
            g = compute_brighten_gamma(float(mean))
            assert DEFAULT_MIN_GAMMA <= g <= 1.0


class TestBrightenBgrToTarget:
    def test_dark_image_brightened_toward_target(self):
        rng = np.random.default_rng(1)
        dark = _random_bgr(rng, 40)
        out = brighten_bgr_to_target(dark)
        assert out is not None
        assert mean_luma_bgr(out) > mean_luma_bgr(dark)

    def test_idempotent(self):
        """提亮结果再调一次返回 None（亮度护栏，不叠加）。"""
        rng = np.random.default_rng(2)
        dark = _random_bgr(rng, 35)
        once = brighten_bgr_to_target(dark)
        assert once is not None
        # γ 封顶时极暗图可能到不了目标，但必然越过暗片判定线
        # (capped lifts may stay under target yet above the dark line)
        assert mean_luma_bgr(once) >= DEFAULT_DARK_MEAN or \
            brighten_bgr_to_target(once) is not None
        if mean_luma_bgr(once) >= DEFAULT_DARK_MEAN:
            assert brighten_bgr_to_target(once) is None

    def test_bright_image_untouched(self):
        rng = np.random.default_rng(3)
        bright = _random_bgr(rng, 170)
        assert brighten_bgr_to_target(bright) is None


class TestPilMeanLuma:
    def test_matches_numpy_gray_mean(self):
        from PIL import Image
        arr = np.tile(np.linspace(0, 200, 64, dtype=np.uint8), (32, 1))
        img = Image.fromarray(arr).convert("RGB")
        assert abs(mean_luma_pil(img) - arr.mean()) < 1.0

    def test_invalid_inputs(self):
        assert mean_luma_pil(None) == -1.0
        assert mean_luma_bgr(None) == -1.0


class TestBuildGammaLut:
    def test_identity_gamma(self):
        lut = build_gamma_lut(1.0)
        assert (lut == np.arange(256, dtype=np.uint8)).all()

    def test_brightening_lut_monotonic(self):
        lut = build_gamma_lut(0.55)
        assert (np.diff(lut.astype(int)) >= 0).all()
        assert lut[128] > 128  # 中点被提亮 / midpoint lifted


class TestPreviewFileBrighten:
    def test_dark_preview_brightened_once(self, tmp_path):
        """暗预览被提亮到暗线以上；再次调用为 no-op（幂等）。"""
        from tools.find_bird_util import brighten_preview_if_dark, preview_mean_luma
        import cv2

        rng = np.random.default_rng(4)
        dark = _random_bgr(rng, 45)
        path = str(tmp_path / "preview.jpg")
        cv2.imwrite(path, dark, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert preview_mean_luma(path) < DEFAULT_DARK_MEAN

        new_mean = brighten_preview_if_dark(path, str(tmp_path))
        assert new_mean is not None
        assert new_mean >= DEFAULT_DARK_MEAN
        mtime1 = os.path.getmtime(path)

        # 第二次：均值已过暗线 → 不再改写（幂等）
        # Second call: above the dark line → untouched (idempotent).
        assert brighten_preview_if_dark(path, str(tmp_path)) is None
        assert os.path.getmtime(path) == mtime1

    def test_bright_preview_untouched(self, tmp_path):
        from tools.find_bird_util import brighten_preview_if_dark
        import cv2

        rng = np.random.default_rng(5)
        path = str(tmp_path / "bright.jpg")
        cv2.imwrite(path, _random_bgr(rng, 150),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert brighten_preview_if_dark(path, str(tmp_path)) is None

    def test_missing_file(self, tmp_path):
        from tools.find_bird_util import brighten_preview_if_dark
        assert brighten_preview_if_dark(
            str(tmp_path / "nope.jpg"), str(tmp_path)) is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
