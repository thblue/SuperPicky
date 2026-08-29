"""
稀有度国家解析链单测：GPS 反解 → 手选国家 → 放弃（全球分）。
无证据时不得默认中国——伦敦/新加坡的无 GPS 照片会被持久化错误分数。
Unit tests for the rarity country chain; a silent CN default would
persist wrong scores for GPS-less overseas shots.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from PIL import Image

import birdid.bird_identifier as bi


class _NoDB:
    """绝不该被调用的桩：断言 predict_bird 被正确替换后 DB 不参与。"""

    def __getattr__(self, name):
        raise AssertionError(f"DB 不应被触碰: {name}")


def _run_identify(monkeypatch, captured, *, gps, resolved, country_arg):
    """stub 掉模型与 GPS，捕获 predict_bird 收到的 photo_country_code。"""
    monkeypatch.setattr(bi, "get_database_manager", lambda: _NoDB())
    monkeypatch.setattr(
        bi, "extract_gps_from_exif",
        lambda p: (gps[0], gps[1], "stub") if gps else (None, None, "no gps"))
    monkeypatch.setattr(
        bi, "_resolve_country_code_from_gps",
        lambda lat, lon: resolved)
    monkeypatch.setattr(bi, "_gps_coords_present",
                        lambda lat, lon: gps is not None)

    def fake_predict(image, **kw):
        captured.append(kw.get("photo_country_code"))
        return []

    monkeypatch.setattr(bi, "predict_bird", fake_predict)
    bi.identify_bird(
        "stub.jpg", use_yolo=False, use_gps=True, use_geo_filter=False,
        country_code=country_arg, preloaded_crop=Image.new("RGB", (8, 8)))


def test_gps_absent_no_selection_stays_global(monkeypatch):
    """无 GPS + 未手选 → None（全球分），绝不默认中国。"""
    captured = []
    _run_identify(monkeypatch, captured, gps=None, resolved=None,
                  country_arg=None)
    assert captured == [None]


def test_gps_absent_user_selected_country_used(monkeypatch):
    """无 GPS + 手选 GB → 用 GB（伦敦照片不受中国口径影响）。"""
    captured = []
    _run_identify(monkeypatch, captured, gps=None, resolved=None,
                  country_arg="GB")
    assert captured == ["GB"]


def test_gps_wins_over_selection(monkeypatch):
    """GPS 反解优先于手选：在伦敦拍摄（GPS→GB）即使手选 CN 也用 GB。"""
    captured = []
    _run_identify(monkeypatch, captured, gps=(51.5, -0.1), resolved="GB",
                  country_arg="CN")
    assert captured == ["GB"]


def test_gps_resolved_failure_falls_to_selection(monkeypatch):
    """有 GPS 但反解失败（海上等）→ 回退手选。"""
    captured = []
    _run_identify(monkeypatch, captured, gps=(0.0, 0.0), resolved=None,
                  country_arg="CN")
    assert captured == ["CN"]
