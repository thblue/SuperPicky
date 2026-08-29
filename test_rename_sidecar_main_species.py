#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_species 对象条目改写/删除单测（修复 2026-08 前字符串-only 的遗漏）。

多鸟编辑器把主鸟写成对象条目 {cn, en, scientific, bird_index}；批量改种/
整种删除的 sidecar 同步此前只处理字符串条目，对象条目里的旧鸟名残留，
BirdIndex（main_species 取种优先级最高）一直显示旧名。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sidecar_export import (mark_species_deleted_in_sidecar,
                                 rename_species_in_sidecar)

OLD = {"cn": "美洲鹈鹕", "en": "American White Pelican",
       "scientific": "Pelecanus erythrorhynchos"}
NEW = {"cn": "澳洲鹈鹕", "en": "Australian Pelican",
       "scientific": "Pelecanus conspicillatus"}


def _write(payload: dict) -> tuple:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, ".superpicky", "meta", "IMG_1.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return tmp, path


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _payload(main_species, det_species=None, deleted=None):
    """构造最小 sidecar：1-2 个检测框 + 指定形态的 main_species。"""
    dets = []
    for i, sp in enumerate(det_species or [OLD]):
        det = {"index": i, "is_selected": i == 0, "bbox": [1, 2, 3, 4],
               "species": dict(sp, confidence=0.9),
               "deleted": bool(deleted and i in deleted)}
        dets.append(det)
    return {"schema_version": 1, "_export_stamp": "s",
            "photo": {"filename": "IMG_1.JPG"},
            "processing": {"rating": 2},
            "detections": dets,
            "main_species": main_species}


class TestRenameMainSpecies(unittest.TestCase):
    """rename_species_in_sidecar 对两种 main_species 条目形态的改写。"""

    def test_object_entry_renamed_with_aux_kept(self):
        """对象条目：名字维度换新，bird_index 等辅助键保留。"""
        main = [{"cn": OLD["cn"], "en": OLD["en"],
                 "scientific": OLD["scientific"], "bird_index": 0}]
        tmp, path = _write(_payload(main))
        n = rename_species_in_sidecar(
            tmp, "IMG_1", new_cn=NEW["cn"], new_en=NEW["en"],
            new_sci=NEW["scientific"], old_cn=OLD["cn"], old_en=OLD["en"])
        self.assertEqual(n, 1)
        entry = _load(path)["main_species"][0]
        self.assertEqual(entry["cn"], NEW["cn"])
        self.assertEqual(entry["en"], NEW["en"])
        self.assertEqual(entry["scientific"], NEW["scientific"])
        self.assertEqual(entry["bird_index"], 0)  # 辅助键保留

    def test_object_entry_matched_by_scientific(self):
        """对象条目按学名/英文名命中同样改写。"""
        main = [{"cn": OLD["cn"], "en": OLD["en"],
                 "scientific": OLD["scientific"], "bird_index": 0}]
        tmp, path = _write(_payload(main))
        rename_species_in_sidecar(
            tmp, "IMG_1", new_cn=NEW["cn"], new_en=NEW["en"],
            new_sci=NEW["scientific"], old_sci=OLD["scientific"])
        self.assertEqual(_load(path)["main_species"][0]["cn"], NEW["cn"])

    def test_string_entry_still_renamed(self):
        """字符串条目（旧形态）行为不变：换成新中文名。"""
        tmp, path = _write(_payload([OLD["cn"]]))
        rename_species_in_sidecar(
            tmp, "IMG_1", new_cn=NEW["cn"], new_en=NEW["en"],
            old_cn=OLD["cn"], old_en=OLD["en"])
        self.assertEqual(_load(path)["main_species"], [NEW["cn"]])

    def test_unrelated_entries_untouched(self):
        """非命中条目（别的种、其他键）原样保留。"""
        other = {"cn": "白鹈鹕", "en": "Great White Pelican",
                 "scientific": "Pelecanus onocrotalus", "bird_index": 1}
        main = [{"cn": OLD["cn"], "en": OLD["en"],
                 "scientific": OLD["scientific"], "bird_index": 0},
                dict(other), "白鹈鹕"]
        tmp, path = _write(_payload(main, det_species=[OLD, dict(other)]))
        rename_species_in_sidecar(
            tmp, "IMG_1", new_cn=NEW["cn"], new_en=NEW["en"],
            new_sci=NEW["scientific"], old_cn=OLD["cn"], old_en=OLD["en"])
        got = _load(path)["main_species"]
        self.assertEqual(got[1], other)      # 未命中对象条目不动
        self.assertEqual(got[2], "白鹈鹕")   # 未命中字符串条目不动


class TestWipeMainSpecies(unittest.TestCase):
    """mark_species_deleted_in_sidecar 对两种条目形态的移除。"""

    def test_object_entry_removed(self):
        """对象条目同名移除，异名条目保留。"""
        keep = {"cn": "白鹈鹕", "en": "Great White Pelican",
                "scientific": "Pelecanus onocrotalus", "bird_index": 1}
        main = [{"cn": OLD["cn"], "en": OLD["en"],
                 "scientific": OLD["scientific"], "bird_index": 0},
                dict(keep)]
        tmp, path = _write(_payload(main, det_species=[OLD, dict(keep)]))
        n = mark_species_deleted_in_sidecar(
            tmp, "IMG_1", species_cn=OLD["cn"], species_en=OLD["en"])
        self.assertEqual(n, 1)
        self.assertEqual(_load(path)["main_species"], [keep])

    def test_string_entry_removed(self):
        """字符串条目同名移除（旧行为回归）。"""
        tmp, path = _write(_payload([OLD["cn"]]))
        mark_species_deleted_in_sidecar(tmp, "IMG_1", species_cn=OLD["cn"])
        self.assertEqual(_load(path)["main_species"], [])


if __name__ == "__main__":
    unittest.main()
