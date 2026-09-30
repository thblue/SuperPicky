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


# ---------------------------------------------------------------------------
# V5.8 常规瓦片检测（每张必跑 + 三重门槛）
# V5.8 universal tile detection pass (runs for every photo, three gates)
# ---------------------------------------------------------------------------

# 预处理图 1024x768；原图 2560x1920（同 4:3，> RESCUE_TILE_SIZE 触发瓦片）
IMG43 = np.zeros((768, 1024, 3), dtype=np.uint8)
FULL_W, FULL_H = 2560, 1920
# 白色目标块（全分辨率坐标），只落在第二列瓦片（x0=512）内
TARGET_FULL = (2200, 800, 2350, 950)
# 第二瓦起点 x0=512 → 目标在瓦内局部坐标
TARGET_TILE_LOCAL = (TARGET_FULL[0] - 512, TARGET_FULL[1],
                     TARGET_FULL[2] - 512, TARGET_FULL[3])
# 映射回预处理图坐标（×0.4）
TARGET_PROC = tuple(v * 0.4 for v in TARGET_FULL)


class _ContentFakeModel:
    """
    内容驱动的假 YOLO：1024 整图调用返回空（模拟目标在整图分辨率下
    不可见——本组测试的场景设定）；只有含白色目标块的瓦片（均值 > 0.5）
    才报鸟框。
    Content-driven fake YOLO: the whole-1024-image call sees nothing (the
    scenario's premise); only tiles containing the white patch report it.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, image, **kwargs):
        self.calls += 1
        h, w = image.shape[:2]
        if w <= 1024:
            # 整图分辨率下不可见 / invisible at whole-image resolution
            return [FakeResult(FakeBoxes(np.zeros((0, 4)), [], []))]
        if float(image.mean()) > 0.5:
            return [FakeResult(FakeBoxes([list(TARGET_TILE_LOCAL)],
                                         [0.45], [14]))]
        return [FakeResult(FakeBoxes(np.zeros((0, 4)), [], []))]


def _write_fullres_jpg(tmp_path):
    import cv2

    big = np.zeros((FULL_H, FULL_W, 3), dtype=np.uint8)
    x1, y1, x2, y2 = TARGET_FULL
    big[y1:y2, x1:x2] = 255
    jpg = str(tmp_path / "full.jpg")
    cv2.imwrite(jpg, big)
    return jpg


def test_tile_pass_accepts_new_bird(tmp_path, monkeypatch):
    """瓦片新框过 BirdID 门槛 → 返回框（proc 坐标）与守门种名。
    A tile box passing the BirdID gate is returned in proc coordinates."""
    jpg = _write_fullres_jpg(tmp_path)
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy, full_image=None,
                        xyxy_full=None: ("栗耳鹀", 93.0))
    boxes = ai_model._tile_detect_pass(_ContentFakeModel(), jpg, IMG43,
                                       None, 25, ".", None)
    assert len(boxes) == 1
    got = [float(v) for v in boxes[0]["xyxy"]]
    assert all(abs(g - e) < 1.0 for g, e in zip(got, TARGET_PROC))
    assert boxes[0]["species"] == "栗耳鹀"
    assert abs(boxes[0]["conf"] - 0.45) < 1e-6


def test_tile_pass_drops_duplicate_of_existing(tmp_path):
    """与已有框重叠/被包含的瓦片框 → 丢弃（同一只鸟/大鸟的半鸟框）。
    Tile boxes overlapping existing detections are dropped."""
    jpg = _write_fullres_jpg(tmp_path)
    existing = np.array([list(TARGET_PROC)], dtype=np.float64)
    boxes = ai_model._tile_detect_pass(_ContentFakeModel(), jpg, IMG43,
                                       existing, 25, ".", None)
    assert boxes == []


def test_tile_pass_conf_floor_skips_classifier(tmp_path, monkeypatch):
    """conf < RESCUE_TILE_MIN_CONF → 直接丢弃，不调分类器。
    Below the conf floor the box is dropped without a classifier call."""
    from config import config as app_config

    jpg = _write_fullres_jpg(tmp_path)
    calls = []

    def fake_confirm(image, xyxy, full_image=None, xyxy_full=None):
        calls.append(1)
        return ("某鸟", 99.0)

    monkeypatch.setattr(ai_model, "_birdid_confirm", fake_confirm)
    monkeypatch.setattr(app_config.ai, "RESCUE_TILE_MIN_CONF", 0.5)
    boxes = ai_model._tile_detect_pass(_ContentFakeModel(), jpg, IMG43,
                                       None, 25, ".", None)
    assert boxes == [] and calls == []


def test_tile_pass_birdid_gate_rejects_leaf(tmp_path, monkeypatch):
    """BirdID top1 < 门槛 → 框保留为 unconfirmed（0★ 仅框待人工回捞）。

    V5.9.5 契约：守门未过不再丢弃——返回框带 unconfirmed=True，由批量
    链路落库为仅几何行；不触发救回语义（见 detect_and_draw_birds）。
    Gate-failed candidates are KEPT as unconfirmed geometry-only boxes
    (V5.9.5) instead of being dropped — human rescue has a box to work
    with; they never flip the rescued flag."""
    jpg = _write_fullres_jpg(tmp_path)
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy, full_image=None,
                        xyxy_full=None: ("某鸟", 12.0))
    boxes = ai_model._tile_detect_pass(_ContentFakeModel(), jpg, IMG43,
                                       None, 25, ".", None)
    assert len(boxes) == 1
    assert boxes[0].get("unconfirmed") is True
    assert boxes[0]["species_conf"] == 12.0


def test_tile_pass_unconfirmed_floor_drops_low_conf(tmp_path, monkeypatch):
    """V5.9.5: 守门未过且 conf < 仅框保留地板 → 丢弃（植被杂物降噪）。

    地板与守门调用地板（0.1）解耦：候选仍过 BirdID 守门，只是守门未过
    且低于 0.2 的不再落 0★ 仅框行。假模型固定 conf 0.45，把地板抬到
    0.5 即可模拟低置信场景。
    Gate failures below the geometry-keep floor (decoupled from the
    0.1 gate-call floor) are dropped; the candidate still gets its
    BirdID confirm call first. The fake model reports conf 0.45, so
    raising the floor to 0.5 simulates the low-conf scenario."""
    from config import config as app_config

    jpg = _write_fullres_jpg(tmp_path)
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy, full_image=None,
                        xyxy_full=None: ("某鸟", 12.0))
    monkeypatch.setattr(app_config.ai, "RESCUE_TILE_UNCONFIRMED_MIN_CONF", 0.5)
    boxes = ai_model._tile_detect_pass(_ContentFakeModel(), jpg, IMG43,
                                       None, 25, ".", None)
    assert boxes == []


def test_tile_pass_disabled(tmp_path, monkeypatch):
    """RESCUE_TILE_ENABLED=False → 完全不跑瓦片推理。
    Disabled → no tile inference at all."""
    from config import config as app_config

    monkeypatch.setattr(app_config.ai, "RESCUE_TILE_ENABLED", False)
    jpg = _write_fullres_jpg(tmp_path)
    model = _ContentFakeModel()
    assert ai_model._tile_detect_pass(model, jpg, IMG43, None, 25,
                                      ".", None) == []
    assert model.calls == 0


def test_tile_pass_skips_small_image(tmp_path):
    """原图长边 <= 瓦片边长时不跑瓦片（1024 整图已覆盖）。
    No tiling when the source is no larger than one tile."""
    import cv2

    jpg = str(tmp_path / "small.jpg")
    cv2.imwrite(jpg, np.zeros((768, 1024, 3), dtype=np.uint8))
    model = _ContentFakeModel()
    assert ai_model._tile_detect_pass(model, jpg, IMG43, None, 25,
                                      ".", None) == []
    assert model.calls == 0


def test_detect_tile_pass_finds_hidden_bird(tmp_path, monkeypatch):
    """pass-1 无鸟 + 瓦片检出 → found_bird=True、rescued=True、bird_count=1。
    Empty pass-1 plus a tile hit → found_bird/rescued True, bird_count 1."""
    jpg = _write_fullres_jpg(tmp_path)

    class _Cfg:
        rescue_scan_enabled = True
        rescue_birdid_gate = 25

    monkeypatch.setattr(ai_model, "get_advanced_config", lambda: _Cfg())
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy, full_image=None,
                        xyxy_full=None: ("栗耳鹀", 93.0))
    result = ai_model.detect_and_draw_birds(
        jpg, _ContentFakeModel(), None, str(tmp_path), [50, 300, 5.0, False],
        None)
    assert result[0] is True          # found_bird
    assert result[8] == 1             # bird_count
    assert result[9] is True          # rescued（原判无鸟 → 瓦片救回）
    assert result[10][0]["mask_polygon"] is None
    x, y, w, h = result[5]
    assert abs(x - TARGET_PROC[0]) < 1 and abs(y - TARGET_PROC[1]) < 1


def test_detect_tile_pass_enriches_big_bird_photo(tmp_path, monkeypatch):
    """pass-1 检出大鸟（过 UI 阈值）时瓦片仍运行，且新增小鸟不与大鸟重复。
    Tiles still run for photos that passed; the added bird is separate."""
    jpg = _write_fullres_jpg(tmp_path)

    class _DispatchModel:
        """整图调用 → 大鸟；含目标块的瓦片 → 小鸟。/ whole-image vs tile."""

        def __call__(self, image, **kwargs):
            h, w = image.shape[:2]
            if w <= 1024:
                # 整图调用：大鸟已过 UI 阈值（proc 坐标框）
                return [FakeResult(FakeBoxes([[40, 40, 360, 360]],
                                             [0.9], [14]))]
            if float(image.mean()) > 0.5:
                return [FakeResult(FakeBoxes([list(TARGET_TILE_LOCAL)],
                                             [0.45], [14]))]
            return [FakeResult(FakeBoxes(np.zeros((0, 4)), [], []))]

    class _Cfg:
        rescue_scan_enabled = True
        rescue_birdid_gate = 25

    monkeypatch.setattr(ai_model, "get_advanced_config", lambda: _Cfg())
    monkeypatch.setattr(ai_model, "_birdid_confirm",
                        lambda image, xyxy, full_image=None,
                        xyxy_full=None: ("栗耳鹀", 93.0))
    result = ai_model.detect_and_draw_birds(
        jpg, _DispatchModel(), None, str(tmp_path), [50, 300, 5.0, False],
        None)
    assert result[0] is True
    assert result[8] == 2             # 大鸟 + 瓦片新增小鸟
    assert result[9] is False         # 照片本身过线，不算被救回
    birds = result[10]
    assert birds[0]['bbox'] == (40, 40, 360, 360)   # 大鸟仍为主鸟（conf 最高）
    assert abs(birds[1]['conf'] - 0.45) < 1e-6      # 瓦片新框入列
