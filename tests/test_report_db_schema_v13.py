# -*- coding: utf-8 -*-
"""
report.db schema 契约测试 / report.db Schema Contract Test

守护 dev-docs/INTERFACE_CONTRACTS.md §2：schema 版本、表集合、photos 列集
全部冻结。任何改动（加列、改类型、改迁移链）都会让本测试失败——这是**有意
的**：schema 是跨进程消费的持久契约（BirdIndex fixer、scripts_dev 回填、
merged_report_db ATTACH 联查），改动必须先走契约评审，再同步更新本文件与
冻结清单。

Guards §2 of the frozen-interface list: schema version, table set and the
photos column list are frozen on purpose. A failure here means a persistent
cross-process contract changed — review it against downstream consumers
first, then update this test together with INTERFACE_CONTRACTS.md.
"""
from __future__ import annotations

import sqlite3

import pytest

from tools.report_db import PHOTO_COLUMNS, SCHEMA_VERSION, ReportDB

pytestmark = pytest.mark.contract

# 冻结的表集合 / Frozen table set
EXPECTED_TABLES = {"photos", "meta", "corrections", "bird_detections", "export_stamps"}

# 冻结的 schema 版本 / Frozen schema version
EXPECTED_SCHEMA_VERSION = "13"


def _open_raw(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def test_schema_version_constant_frozen():
    """SCHEMA_VERSION 常量必须保持在冻结版本 / Version constant stays frozen."""
    assert SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION


def test_meta_records_schema_version(photo_library):
    """新建库的 meta 表必须记录冻结版本号 / meta table records the frozen version."""
    db, library = photo_library
    db_path = library / ".superpicky" / "report.db"
    assert db_path.exists(), "ReportDB 应在 <库>/.superpicky/report.db 建库"
    for conn in _open_raw(db_path):
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
    assert row == (EXPECTED_SCHEMA_VERSION,)


def test_table_set_frozen(photo_library):
    """表集合冻结：不得删表、不得未经评审加表 / Table set is frozen."""
    db, library = photo_library
    # export_stamps 为懒建表（首次 sidecar 导出时创建），先物化再断言。
    # export_stamps is created lazily on first sidecar export — materialize first.
    from core.sidecar_export import export_directory_sidecars
    export_directory_sidecars(db, str(library), log=lambda *_: None)
    db_path = library / ".superpicky" / "report.db"
    for conn in _open_raw(db_path):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    tables = {r[0] for r in rows}
    missing = EXPECTED_TABLES - tables
    extra = tables - EXPECTED_TABLES
    assert not missing, f"冻结表缺失: {missing}"
    assert not extra, f"出现冻结清单之外的表（请先评审 dev-docs/INTERFACE_CONTRACTS.md §2）: {extra}"


def test_photos_columns_frozen(photo_library):
    """
    photos 列集冻结：实际建库列必须与 PHOTO_COLUMNS 声明逐列一致。

    The physical photos columns must match the declared PHOTO_COLUMNS
    one-for-one (order included) — keeps declaration and reality honest.
    """
    db, library = photo_library
    db_path = library / ".superpicky" / "report.db"
    # 物理表首列是 CREATE TABLE 里声明的自增主键 id，PHOTO_COLUMNS 不含它。
    # The physical table leads with the autoincrement PK `id`, declared in
    # CREATE TABLE but absent from PHOTO_COLUMNS.
    expected = ["id"] + [c[0] for c in PHOTO_COLUMNS]
    for conn in _open_raw(db_path):
        actual = [r[1] for r in conn.execute("PRAGMA table_info(photos)").fetchall()]
    assert actual == expected, (
        "photos 列集偏离冻结契约——请先评审 INTERFACE_CONTRACTS.md §2 与全部下游消费者"
        "（BirdIndex fixer / scripts_dev 回填 / merged_report_db ATTACH），再同步更新本测试"
    )


def test_migration_chain_preserves_legacy_rows(photo_library):
    """
    迁移链行为冒烟：新建 v13 库后插入行可读回，schema 版本不漂移。

    Smoke for the migration chain contract: rows inserted into a fresh
    v13 library read back with the schema version intact.
    """
    db, _ = photo_library
    row = db.get_photo("IMG_0001.CR3")
    assert row is not None
    assert row["has_bird"] == 1
    assert SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION
