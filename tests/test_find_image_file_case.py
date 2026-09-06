# -*- coding: utf-8 -*-
"""
post_adjustment_engine.find_image_file 单测：
大小写不敏感文件系统上必须返回磁盘真实存储名，而非拼造名。

find_image_file unit tests: on case-insensitive filesystems the lookup must
return the on-disk stored name, not the constructed candidate path
(2026-08-31 restar renamed 44 extensions via the constructed-path flaw).
"""
import os

import pytest

from core.post_adjustment_engine import PostAdjustmentEngine


def test_returns_real_stored_case_for_uppercase_file(tmp_path):
    """磁盘存 .CR3（大写）→ 按 RAW 优先级返回真实大写名。
    Stored .CR3 must resolve to the real uppercase name."""
    (tmp_path / "IMG_0001.CR3").write_bytes(b"x")
    engine = PostAdjustmentEngine(str(tmp_path))
    path = engine.find_image_file("IMG_0001")
    assert path is not None
    assert os.path.basename(path) == "IMG_0001.CR3"


def test_returns_real_stored_case_for_lowercase_file(tmp_path):
    """磁盘存 .cr3（小写）→ 即使候选列表大写在前（RAW 顺序），返回真实小写名。
    Stored .cr3 must resolve to the real lowercase name regardless of
    candidate priority order."""
    (tmp_path / "IMG_0002.cr3").write_bytes(b"x")
    engine = PostAdjustmentEngine(str(tmp_path))
    path = engine.find_image_file("IMG_0002")
    assert path is not None
    assert os.path.basename(path) == "IMG_0002.cr3"


def test_finds_file_in_subdirectory_with_real_case(tmp_path):
    """递归分支同样返回磁盘真实存储名。
    The recursive branch must also return the stored name."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "IMG_0003.NEF").write_bytes(b"x")
    engine = PostAdjustmentEngine(str(tmp_path))
    path = engine.find_image_file("IMG_0003")
    assert path is not None
    assert os.path.basename(path) == "IMG_0003.NEF"
    assert os.path.dirname(path) == str(sub)


def test_missing_file_returns_none(tmp_path):
    """无匹配文件返回 None。
    No match → None."""
    engine = PostAdjustmentEngine(str(tmp_path))
    assert engine.find_image_file("NOPE") is None


@pytest.mark.parametrize("name", ["IMG_0004.CR3", "IMG_0004.cr3"])
def test_query_case_does_not_leak_into_result(tmp_path, name):
    """无论磁盘真实大小写如何，返回名始终等于磁盘名（查询侧大小写不敏感）。
    Either stored casing resolves; the returned name always equals disk."""
    (tmp_path / name).write_bytes(b"x")
    engine = PostAdjustmentEngine(str(tmp_path))
    path = engine.find_image_file("img_0004")
    assert path is not None
    assert os.path.basename(path) == name
