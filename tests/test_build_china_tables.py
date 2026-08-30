"""
中国国别稀有度 + 国家保护等级 构建脚本单测。
用合成夹具验证归一化数学、tier 映射和保护表解析/匹配/去重逻辑，
不访问网络（解析器吃本地 HTML 片段，匹配吃内存索引）。
Builder tests for the CN rarity and national-protection tables using
synthetic fixtures — no network access.
"""
import json
import os
import sqlite3

import pytest

from scripts_dev.build_china_protection import BirdTableParser, match_entries
from scripts_dev.build_china_rarity import normalize


# ---------------------------------------------------------------------------
# 中国稀有度：log 归一化 / CN rarity normalization
# ---------------------------------------------------------------------------

def test_normalize_bounds_and_monotonic():
    """最高观察数→0 分，最低→100 分，且分数随观察数单调下降。"""
    counts = {1: 1, 2: 10, 3: 100, 4: 100_000}
    scores = normalize(counts)
    assert scores[4] == pytest.approx(0.0, abs=0.01)   # 最多 → 0 分
    assert scores[1] == pytest.approx(100.0, abs=0.01)  # 最少 → 100 分
    assert scores[1] > scores[2] > scores[3] > scores[4]


def test_normalize_skips_zero_counts():
    """count=0 的物种不参与归一化（无 CN 行 → 运行时回退全球分）。"""
    scores = normalize({1: 0, 2: 5})
    assert 1 not in scores
    assert 2 in scores


def test_normalize_single_species_midpoint():
    """只有一个物种时退化处理为 50 分（span=0 分支）。"""
    assert normalize({7: 42}) == {7: 50.0}


# ---------------------------------------------------------------------------
# 保护等级：维基表格解析 / protection table parsing
# ---------------------------------------------------------------------------

BIRD_HTML = """
<table>
<tr><td colspan="4"><b>鸟纲 Aves</b></td></tr>
<tr><td><b>鸡形目</b></td><td><b>Galliformes</b></td><td></td><td></td></tr>
<tr><td>&#160;<a>雉科</a></td><td>Phasianidae</td><td></td><td></td></tr>
<tr><td>&#160;&#160;金雕</td><td><i>Aquila chrysaetos</i></td>
    <td align="center">Ⅰ</td><td></td></tr>
<tr><td>&#160;&#160;大天鹅<sup>[1]</sup></td><td><i>Cygnus cygnus</i></td>
    <td align="center">Ⅱ</td><td>仅限野外种群</td></tr>
</table>
"""


def _parse(html: str):
    p = BirdTableParser()
    p.feed(html)
    return p.rows


def test_parser_extracts_bird_rows_with_level():
    rows = _parse(BIRD_HTML)
    species = [r for r in rows if len(r) >= 3 and r[2] in ("Ⅰ", "Ⅱ")]
    assert ("金雕", "Aquila chrysaetos", "Ⅰ") == (
        species[0][0], species[0][1], species[0][2])
    # 脚注引用与备注列被剥离 / footnotes stripped, note kept in col 4
    assert species[1][0] == "大天鹅"
    assert species[1][3] == "仅限野外种群"


def test_parser_ignores_non_species_rows():
    """纲/目/科标题行没有级别格，不应被当成物种行。"""
    rows = _parse(BIRD_HTML)
    species = [r for r in rows if len(r) >= 3 and r[2] in ("Ⅰ", "Ⅱ")]
    assert len(species) == 2
    # 标题行全部留在非物种集合里 / header rows never classified as species
    names = [r[0] for r in species]
    assert not any(k in n for n in names for k in ("纲", "目", "科"))


# ---------------------------------------------------------------------------
# 保护等级：匹配管线 / protection match pipeline
# ---------------------------------------------------------------------------

class _E:
    """轻量 WikiEntry 替身 / lightweight WikiEntry stand-in."""

    def __init__(self, zh, sci, level=1):
        self.chinese_name = zh
        self.scientific_name = sci
        self.level = level
        self.note = ""
        self.taxon_class = "鸟纲"
        self.order = "试验目"


def test_match_pipeline_sci_and_chinese_and_manual():
    sci_to_cid = {"aquila chrysaetos": 1, "falco tinnunculus": 2}
    unique_zh = {"试验鸟": 3}
    info = {1: ("Aquila chrysaetos", "金雕"),
            2: ("Falco tinnunculus", "红隼"),
            3: ("Testus testus", "试验鸟")}
    manual = {}
    import scripts_dev.build_china_protection as mod
    saved = dict(mod.MANUAL_NAME_MAP)
    saved_gbif = mod.gbif_match
    mod.MANUAL_NAME_MAP = {
        "falco alauda": ("Falco tinnunculus", "测试映射"),
    }
    mod.gbif_match = lambda q: None  # 断网测试 / keep the test offline
    try:
        entries = [
            _E("金雕", "Aquila chrysaetos"),          # 学名直中 / exact sci
            _E("试验鸟", "Unknownus birdus"),          # 学名未中 → 中文名
            _E("红隼", "Falco alauda"),                # 人工映射 / manual
            _E("无影鸟", "Ghostus ghostus"),           # 未决 / unresolved
        ]
        results = match_entries(entries, sci_to_cid, unique_zh, info)
    finally:
        mod.MANUAL_NAME_MAP = saved
        mod.gbif_match = saved_gbif
    by_zh = {r.entry.chinese_name: r for r in results}
    assert by_zh["金雕"].method == "sci" and by_zh["金雕"].class_id == 1
    assert by_zh["试验鸟"].method == "chinese" and by_zh["试验鸟"].class_id == 3
    assert by_zh["红隼"].method == "manual" and by_zh["红隼"].class_id == 2
    assert by_zh["无影鸟"].class_id is None
