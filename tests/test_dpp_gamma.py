#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DPP 人工伽马（W1 / V5.9.4）的仓库级单元测试。

tools/dpp_gamma.py 是跑批内「人工编辑优先于自动提亮」的核心模块，
此前只有 gitignore 沙盒里的端到端验证（scripts_dev/_backfill_sandbox/
w1_dpp_sandbox_test.py）；本文件把关键语义钉进仓库：

- build_lut 幂律数学（含真实样张锚点：玉渊潭 mid -2.16 → LUT[128]=219）；
- read_dpp_gamma_map 容错解析（坏值丢弃、空值不入表）；
- apply_lut_to_jpeg 原地提亮往返；
- raw_to_jpeg 的 DPP 优先分支：仅新抽取套 LUT、缓存命中不动、
  有人工编辑绝不叠加自动提亮（纯逻辑测试，monkeypatch 提取与套表）。

Unit tests for the DPP human-gamma module (W1 / V5.9.4) — LUT math with
real-sample anchors, tolerant recipe parsing, in-place LUT application,
and the raw_to_jpeg human-edit-first branch (fresh-only, cache-hit
untouched, never stacked with auto-brightening).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from tools import dpp_gamma  # noqa: E402
import tools.find_bird_util as fbu  # noqa: E402


class TestBuildLut(unittest.TestCase):
    """幂律 LUT 数学 / power-law LUT math."""

    def test_identity_at_zero(self):
        """m=0（g=1）→ 恒等 LUT。/ m=0 gives the identity LUT."""
        lut = dpp_gamma.build_lut(0.0)
        self.assertEqual(lut.tolist(), list(range(256)))

    def test_real_sample_anchors(self):
        """真实样张锚点（refresh --dry 实测值）：提亮单调、端点不变。"""
        lut = dpp_gamma.build_lut(-2.16)
        self.assertEqual(int(lut[128]), 219)   # 玉渊潭 027A5640 实测
        lut2 = dpp_gamma.build_lut(-1.65)
        self.assertEqual(int(lut2[128]), 205)  # 玉渊潭 027A5642 实测
        for lut in (lut, lut2):
            self.assertEqual(int(lut[0]), 0)
            self.assertEqual(int(lut[255]), 255)
            diffs = np.diff(lut.astype(int))
            self.assertTrue((diffs >= 0).all())  # 单调不减 / monotonic

    def test_positive_mid_darkens(self):
        """m>0 → g>1 → 变暗（人工压暗同样被尊重）。/ positive mid darkens."""
        lut = dpp_gamma.build_lut(1.0)
        self.assertLess(int(lut[128]), 128)


class TestReadDppGammaMap(unittest.TestCase):
    """管线预读入口的容错解析 / tolerant parsing of the pipeline entry."""

    def test_bad_values_dropped(self):
        """解析失败/空值丢弃，好值保留——单条坏值不阻断整批。"""
        with mock.patch.object(
                dpp_gamma, "read_gamma_params",
                return_value={"good": ("-1.5", "+0.0"),
                              "bad": ("not-a-number", ""),
                              "empty": ("", "")}):
            m = dpp_gamma.read_dpp_gamma_map(["x/any.CR3"], "exiftool")
        self.assertEqual(m, {"good": -1.5})


class TestApplyLutToJpeg(unittest.TestCase):
    """LUT 原地应用 / in-place LUT application."""

    def test_brightens_and_returns_mean(self):
        import cv2  # noqa: F401
        import tempfile
        fd, jpg = tempfile.mkstemp(suffix=".jpg", prefix="sp_lut_")
        os.close(fd)
        try:
            dark = np.full((64, 64, 3), 60, dtype=np.uint8)
            cv2.imwrite(jpg, dark)
            after = dpp_gamma.apply_lut_to_jpeg(
                jpg, dpp_gamma.build_lut(-2.16))
            self.assertIsNotNone(after)
            self.assertGreater(after, 80.0)   # 提亮生效 / brightened
            self.assertLessEqual(after, 255.0)
        finally:
            if os.path.exists(jpg):
                os.remove(jpg)


class TestRawToJpegDppBranch(unittest.TestCase):
    """raw_to_jpeg 的「人工编辑优先」分支逻辑（monkeypatch 提取与套表）。

    Human-edit-first branch logic with the extraction and LUT application
    patched out — pure dispatch semantics, no real RAW involved.
    """

    def setUp(self):
        self.jpg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_dpp_branch_probe.jpg")
        self.applied = []
        self.brightened = []

    def _patch(self, fresh):
        extract = mock.patch.object(
            fbu, "_raw_to_jpeg_extract", return_value=(self.jpg, fresh))
        apply_lut = mock.patch.object(
            fbu, "_apply_dpp_gamma_lut",
            side_effect=lambda p, m: self.applied.append((p, m)))
        brighten = mock.patch.object(
            fbu, "_brighten_dark_preview",
            side_effect=lambda r, p: self.brightened.append(r))
        for p in (extract, apply_lut, brighten):
            p.start()
            self.addCleanup(p.stop)

    def test_fresh_extraction_applies_dpp_only(self):
        """新抽取 + 人工编辑 → 套 LUT，绝不叠加自动提亮。"""
        self._patch(fresh=True)
        out = fbu.raw_to_jpeg("X:/a.CR3", auto_brighten=True,
                              dpp_gamma_mid=-2.16)
        self.assertEqual(out, self.jpg)
        self.assertEqual(self.applied, [(self.jpg, -2.16)])
        self.assertEqual(self.brightened, [])  # 人工优先，自动不介入

    def test_cache_hit_untouched(self):
        """缓存命中 + 人工编辑 → 原样返回（幂等，不重复套 LUT）。"""
        self._patch(fresh=False)
        out = fbu.raw_to_jpeg("X:/a.CR3", auto_brighten=True,
                              dpp_gamma_mid=-2.16)
        self.assertEqual(out, self.jpg)
        self.assertEqual(self.applied, [])
        self.assertEqual(self.brightened, [])

    def test_no_recipe_takes_auto_path(self):
        """无人工编辑 → 自动提亮路径照旧（回归验证）。"""
        self._patch(fresh=True)
        out = fbu.raw_to_jpeg("X:/a.CR3", auto_brighten=True)
        self.assertEqual(out, self.jpg)
        self.assertEqual(self.applied, [])
        self.assertEqual(self.brightened, ["X:/a.CR3"])


if __name__ == "__main__":
    unittest.main()
