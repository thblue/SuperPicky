#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""identify_bird 暗框提亮重识别（dark_retry_conf）逻辑测试。

不加载任何模型：_identify_with_tiers / extract_gps_from_exif /
load_species_whitelist 全部替换为可控替身，只验证重试的触发条件、
择优语义与结果元数据。
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import birdid.bird_identifier as bi
from tools.tone_curve import mean_luma_pil


def _crop(mean: float, size: tuple = (64, 64)) -> Image.Image:
    """生成指定均值的 RGB 框图（模拟暗框/亮框）。"""
    rng = np.random.default_rng(11)
    arr = np.clip(rng.normal(mean, 20, (size[1], size[0], 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _candidate(conf: float, name: str = "普通翠鸟") -> dict:
    return {
        "cn_name": name, "en_name": "Common Kingfisher",
        "scientific_name": "Alcedo atthis",
        "confidence": conf, "class_id": 1,
    }


@pytest.fixture()
def patched_identify_env(monkeypatch):
    """替换 GPS/白名单，收集 _identify_with_tiers 调用序列。"""
    calls = []

    def _fake_tiers(img, **kwargs):
        calls.append(img)
        queued = getattr(_fake_tiers, "queue")
        return list(queued.pop(0)), 1, 100

    def _fake_gps(path):
        return None, None, "no gps"

    monkeypatch.setattr(bi, "_identify_with_tiers", _fake_tiers)
    monkeypatch.setattr(bi, "extract_gps_from_exif", _fake_gps)
    monkeypatch.setattr(bi, "load_species_whitelist", lambda: {})
    _fake_tiers.queue = calls  # 便捷：list 本体，pop(0) 依序取出
    return calls


class TestDarkRetry:
    def test_retry_wins_on_dark_crop(self, patched_identify_env):
        """暗框 + 首跑低置信 + 提亮后更高 → 采用提亮结果并带元数据。"""
        patched_identify_env.append(
            ([_candidate(30.0, "乌鸫")]))    # 首跑：低于采纳线
        patched_identify_env.append(
            ([_candidate(62.0)]))            # 提亮重跑：胜出

        dark = _crop(45)
        result = bi.identify_bird(
            "X:/fake.jpg", use_yolo=False, preloaded_crop=dark,
            dark_retry_conf=40.0)

        assert result["success"] is True
        assert result["results"][0]["confidence"] == pytest.approx(62.0)
        assert len(patched_identify_env) == 2  # 恰好多跑一次
        retry = result["brightened_retry"]
        assert retry["orig_conf"] == pytest.approx(30.0)
        assert retry["bright_conf"] == pytest.approx(62.0)
        bright = result["brightened_crop"]
        assert bright is not None
        # 提亮图确实更亮，且原图未被修改 / brightened is brighter; original untouched
        assert mean_luma_pil(bright) > mean_luma_pil(dark)

    def test_retry_keeps_better_original(self, patched_identify_env):
        """提亮后置信度反而更低 → 保留原结果，不标注重试。"""
        patched_identify_env.append([_candidate(38.0, "乌灰鸫")])
        patched_identify_env.append([_candidate(20.0, "紫啸鸫")])

        result = bi.identify_bird(
            "X:/fake.jpg", use_yolo=False, preloaded_crop=_crop(50),
            dark_retry_conf=40.0)

        assert result["results"][0]["confidence"] == pytest.approx(38.0)
        assert "brightened_retry" not in result
        assert len(patched_identify_env) == 2  # 重跑了但没采纳

    def test_no_retry_when_confident(self, patched_identify_env):
        """首跑已过采纳线 → 不做第二次推理。"""
        patched_identify_env.append([_candidate(85.0)])

        result = bi.identify_bird(
            "X:/fake.jpg", use_yolo=False, preloaded_crop=_crop(40),
            dark_retry_conf=40.0)

        assert result["results"][0]["confidence"] == pytest.approx(85.0)
        assert len(patched_identify_env) == 1
        assert "brightened_retry" not in result

    def test_no_retry_on_bright_crop(self, patched_identify_env):
        """框不暗（均值 ≥ 90）→ 即使低置信也不重试。"""
        patched_identify_env.append([_candidate(30.0, "麻雀")])

        result = bi.identify_bird(
            "X:/fake.jpg", use_yolo=False, preloaded_crop=_crop(140),
            dark_retry_conf=40.0)

        assert result["results"][0]["confidence"] == pytest.approx(30.0)
        assert len(patched_identify_env) == 1

    def test_disabled_by_default(self, patched_identify_env):
        """不传 dark_retry_conf → 永远单次分类（向后兼容）。"""
        patched_identify_env.append([_candidate(10.0, "噪鹃")])

        result = bi.identify_bird(
            "X:/fake.jpg", use_yolo=False, preloaded_crop=_crop(30))

        assert result["results"][0]["confidence"] == pytest.approx(10.0)
        assert len(patched_identify_env) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
