# -*- coding: utf-8 -*-
"""
ai_model._rescue_scan 单测：用 FakeModel 模拟 ultralytics 结果，
不加载真实 YOLO/BirdID 模型。

覆盖 4 条路径：直接接受 / 弱候选识鸟通过 / 弱候选识鸟拒绝 / 无候选。

_rescue_scan unit tests with a FakeModel mimicking ultralytics results;
no real YOLO/BirdID model is loaded. Covers direct-accept, gate-accept,
gate-reject and no-candidate paths.
"""
import numpy as np
import torch

from core import ai_model


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.masks = None


class FakeModel:
    """返回预设检测结果的假 YOLO / Fake YOLO returning canned detections."""

    def __init__(self, xyxy, conf, cls):
        self._r = FakeResult(FakeBoxes(xyxy, conf, cls))

    def __call__(self, image, **kwargs):
        return [self._r]


IMG = np.zeros((683, 1024, 3), dtype=np.uint8)


def test_direct_accept_bird_above_threshold():
    model = FakeModel([[10, 10, 60, 60]], [0.8], [14])
    r = ai_model._rescue_scan(model, IMG, 0.5, 10, ".", None)
    assert r is not None and r["source"] == "bird"
    assert abs(r["conf"] - 0.8) < 1e-6


def test_weak_bird_gate_accept(monkeypatch):
    model = FakeModel([[10, 10, 60, 60]], [0.12], [14])
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy: ("红脚鹬", 81.7))
    r = ai_model._rescue_scan(model, IMG, 0.5, 10, ".", None)
    assert r is not None and r["species"] == "红脚鹬"


def test_weak_bird_gate_prefers_fullres_crop(tmp_path, monkeypatch):
    """提供 image_path 且原图更高分辨率时，守门裁剪用缩放后的原图框。
    With image_path and a higher-res source, the gate confirm must receive
    the box scaled into full-resolution coordinates (tiny-target fix)."""
    import cv2

    # 预览图 2048x1366（预处理图 1024 的 2 倍），白色块标记预期裁剪区
    big = np.full((1366, 2048, 3), 200, dtype=np.uint8)
    big[20:120, 40:120] = 255
    jpg = str(tmp_path / "full.jpg")
    cv2.imwrite(jpg, big)

    captured = {}

    def fake_confirm(image, xyxy, full_image=None, xyxy_full=None):
        captured["image_shape"] = image.shape[:2]
        captured["xyxy"] = xyxy
        captured["full_shape"] = (None if full_image is None
                                  else full_image.shape[:2])
        captured["xyxy_full"] = xyxy_full
        return ("红脚鹬", 81.7)

    monkeypatch.setattr(ai_model, "_birdid_confirm", fake_confirm)

    # 候选框 (10, 10, 30, 50) 位于 1024 图 → 原图坐标系应为 (20, 20, 60, 100)
    model = FakeModel([[10, 10, 30, 50]], [0.12], [14])
    r = ai_model._rescue_scan(model, IMG, 0.5, 10, ".", None, image_path=jpg)

    assert r is not None and r["species"] == "红脚鹬"
    assert captured["xyxy_full"] == (20, 20, 60, 100)
    assert captured["full_shape"] == (1366, 2048)


def test_weak_bird_gate_falls_back_without_fullres(monkeypatch):
    """无 image_path（或原图不更大）时保持旧 2 参数守门调用。
    Without image_path the legacy two-argument confirm call is kept."""
    model = FakeModel([[10, 10, 60, 60]], [0.12], [14])
    captured = {}

    def fake_confirm(image, xyxy):
        captured["called"] = True
        return ("红脚鹬", 81.7)

    monkeypatch.setattr(ai_model, "_birdid_confirm", fake_confirm)
    r = ai_model._rescue_scan(model, IMG, 0.5, 10, ".", None, image_path=None)
    assert r is not None and captured["called"]


def test_kite_candidate_gate_reject(monkeypatch):
    model = FakeModel([[10, 10, 60, 60]], [0.85], [33])  # kite
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy: ("某鸟", 4.0))
    assert ai_model._rescue_scan(model, IMG, 0.5, 10, ".", None) is None


def test_no_candidate_returns_none():
    model = FakeModel([[10, 10, 60, 60]], [0.9], [0])  # person
    assert ai_model._rescue_scan(model, IMG, 0.5, 10, ".", None) is None


def test_detect_returns_10_tuple_no_bird(tmp_path, monkeypatch):
    """空检测 + 补救关闭 → 10 元组，末位 rescued=False。
    Empty detections with rescue disabled → 10-tuple ending rescued=False."""
    import cv2

    jpg = str(tmp_path / "t.jpg")
    cv2.imwrite(jpg, np.zeros((64, 64, 3), dtype=np.uint8))

    class _Cfg:
        rescue_scan_enabled = False
        rescue_birdid_gate = 10

    monkeypatch.setattr(ai_model, "get_advanced_config", lambda: _Cfg())
    model = FakeModel(np.zeros((0, 4)), [], [])
    result = ai_model.detect_and_draw_birds(
        jpg, model, None, str(tmp_path), [50, 300, 5.0, False], None)
    assert len(result) == 11  # V5.0: 追加 all_birds
    assert result[0] is False and result[9] is False
    assert result[10] == []


def test_detect_rescue_success_path(tmp_path, monkeypatch):
    """补救成功 → rescued=True，bbox 来自救回候选，bird_count=1。
    Rescue success → rescued=True, bbox from rescued candidate, bird_count=1."""
    import cv2

    jpg = str(tmp_path / "t.jpg")
    cv2.imwrite(jpg, np.zeros((64, 64, 3), dtype=np.uint8))

    class _Cfg:
        rescue_scan_enabled = True
        rescue_birdid_gate = 10

    monkeypatch.setattr(ai_model, "get_advanced_config", lambda: _Cfg())
    monkeypatch.setattr(ai_model, "_rescue_scan",
                        lambda *a, **k: {
                            "xyxy": np.array([4.0, 5.0, 40.0, 50.0]),
                            "conf": 0.31, "mask": None,
                            "source": "kite", "species": "红脚鹬",
                            "species_conf": 81.7,
                        })
    model = FakeModel(np.zeros((0, 4)), [], [])  # 第一遍无任何检测
    result = ai_model.detect_and_draw_birds(
        jpg, model, None, str(tmp_path), [50, 300, 5.0, False], None)
    assert len(result) == 11  # V5.0: 追加 all_birds
    assert result[0] is True          # found_bird
    assert result[9] is True          # rescued
    assert result[8] == 1             # bird_count
    x, y, w, h = result[5]            # bbox (x, y, w, h)
    assert (x, y) == (4, 5) and (w, h) == (36, 45)
    assert abs(result[2] - 0.31) < 1e-6  # confidence 来自救回候选
