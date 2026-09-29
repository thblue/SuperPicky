# -*- coding: utf-8 -*-
"""
人工定种白名单单测 / Unit tests for the manual species whitelist.

验证 _apply_species_whitelist 的替换/去重/透传规则与 load_species_whitelist
的文件装载，不依赖模型推理。

Covers substitution, dedup and passthrough rules of _apply_species_whitelist
plus the file loading of load_species_whitelist, without any inference.
"""
import json

from birdid.bird_identifier import (
    _apply_species_whitelist,
    load_species_whitelist,
)

# 玩具白名单：欧亚喜鹊(6819) → 喜鹊(10868)，模拟真实鸟类库映射
WL = {
    6819: {
        "class_id": 10868,
        "cn_name": "喜鹊",
        "en_name": "Oriental Magpie",
        "scientific_name": "Pica serica",
        "iucn_category": None,
        "gbif_rarity_100": None,
    }
}


def _res(cid: int, cn: str, conf: float) -> dict:
    return {
        "class_id": cid,
        "cn_name": cn,
        "en_name": cn,
        "confidence": conf,
        "region_match": True,
    }


def test_demote_top1_substituted_in_place():
    """模型 top-1 是 demote 物种 → 原位替换为 keep，置信度保留"""
    out = _apply_species_whitelist(
        [_res(6819, "欧亚喜鹊", 77.7), _res(10868, "喜鹊", 5.9)], WL
    )
    assert len(out) == 1
    assert out[0]["class_id"] == 10868
    assert out[0]["cn_name"] == "喜鹊"
    assert out[0]["confidence"] == 77.7
    assert out[0]["whitelist_substituted"] is True


def test_keep_already_top_stays_untouched():
    """keep 已自然排第一 → 保持原条目，只丢弃后面的 demote"""
    out = _apply_species_whitelist(
        [_res(10868, "喜鹊", 30.0), _res(6819, "欧亚喜鹊", 20.0)], WL
    )
    assert len(out) == 1
    assert out[0]["class_id"] == 10868
    assert out[0]["confidence"] == 30.0
    assert "whitelist_substituted" not in out[0]


def test_keep_not_in_results_still_substituted():
    """keep 不在 top-k 也能替换（插入 keep 的库信息）"""
    out = _apply_species_whitelist(
        [_res(6819, "欧亚喜鹊", 50.1), _res(9999, "董鸡", 1.0)], WL
    )
    assert out[0]["class_id"] == 10868
    assert out[0]["confidence"] == 50.1
    assert out[1]["class_id"] == 9999


def test_no_demote_present_unchanged():
    """结果里没有 demote 物种 → 原样返回"""
    src = [_res(100, "麻雀", 60.0), _res(200, "白头鹎", 30.0)]
    out = _apply_species_whitelist(src, WL)
    assert out == src


def test_empty_whitelist_noop():
    """空白名单 → 不做任何处理"""
    src = [_res(6819, "欧亚喜鹊", 77.7)]
    assert _apply_species_whitelist(src, {}) == src


def test_rarity_fn_receives_keep_and_country():
    """稀有度回调拿到 keep 的 class_id 与拍摄国代码"""
    seen = []

    def rarity_fn(cid, cc):
        seen.append((cid, cc))
        return 42.0

    out = _apply_species_whitelist(
        [_res(6819, "欧亚喜鹊", 77.7)], WL,
        photo_country_code="CN", rarity_fn=rarity_fn,
    )
    assert seen == [(10868, "CN")]
    assert out[0]["gbif_rarity_100"] == 42.0


def test_load_whitelist_from_file(tmp_path, monkeypatch):
    """文件装载：pairs 解析为 demote_cid → keep 条目；坏行跳过"""
    from birdid import bird_identifier as bi

    # 用真实鸟类库反查（开发环境自带 bird_reference.sqlite）
    path = tmp_path / "species_whitelist.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"keep_scientific": "Pica serica",
                     "demote_scientific": "Pica pica"},
                    {"keep_scientific": "不存在的种", "demote_scientific": "Pica pica"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    wl = load_species_whitelist(path=str(path), force=True)
    assert 6819 in wl
    assert wl[6819]["class_id"] == 10868
    assert wl[6819]["cn_name"] == "喜鹊"

    # mtime 缓存：未变化时二次读取直接命中缓存对象
    wl2 = load_species_whitelist(path=str(path))
    assert wl2 is wl


def test_load_whitelist_demote_array(tmp_path):
    """demote 数组形态：一个 keep 收编多个易混种，单字符串形态仍兼容"""
    path = tmp_path / "species_whitelist.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "keep_scientific": "Saxicola maurus",
                        # 东亚/欧洲/非洲石䳭全部归并到黑喉石䳭
                        "demote_scientific": [
                            "Saxicola stejnegeri",
                            "Saxicola rubicola",
                            "Saxicola torquatus",
                            "解析不了的种",
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    wl = load_species_whitelist(path=str(path), force=True)
    # 三个可解析的 demote 都指向同一个 keep 条目，坏名被跳过
    assert {9091, 9089, 9092} <= set(wl)
    assert all(wl[cid]["class_id"] == 9090 for cid in (9091, 9089, 9092))
    assert wl[9091]["cn_name"] == "黑喉石䳭"
