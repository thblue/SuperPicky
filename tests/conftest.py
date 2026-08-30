# -*- coding: utf-8 -*-
"""SuperPicky 测试共享夹具。

当前仅提供「临时照片库」工厂：在 pytest tmp_path 下建一个最小可用的
照片库（.superpicky/report.db + 若干 photo 行），供契约测试与后续迁入
的 DB 类测试复用。**绝不指向任何真实照片目录**。

Shared fixtures. The `photo_library` factory builds a minimal throwaway
photo library under pytest's tmp_path — it never touches real libraries.
"""
from __future__ import annotations

import pytest

from tools.report_db import ReportDB


@pytest.fixture()
def photo_library(tmp_path):
    """
    创建一个临时照片库并预置一鸟一无鸟两张照片。

    参数:
    tmp_path: pytest 提供的临时目录

    返回:
    tuple: (ReportDB 实例, 照片库目录 Path)。用例结束后自动关闭 DB。

    Creates a temp photo library preloaded with one bird photo and one
    no-bird photo. Returns (ReportDB, library Path); DB auto-closed.
    """
    library = tmp_path / "library"
    library.mkdir()
    db = ReportDB(str(library))
    # current_path/original_path 与真实管线一致：未移动前都等于文件名，
    # sidecar 的 V5.3 定位锚点（library_path/relative_path）由它们导出。
    db.insert_photo({
        "filename": "IMG_0001.CR3",
        "has_bird": 1,
        "rating": 3,
        "current_path": "IMG_0001.CR3",
        "original_path": "IMG_0001.CR3",
    })
    db.insert_photo({"filename": "IMG_0002.CR3", "has_bird": 0})
    yield db, library
    db.close()
