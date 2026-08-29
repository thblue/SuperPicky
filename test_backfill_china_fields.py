"""
回填脚本单测：合成 report.db + 合成字典库，验证保护等级连接、
CN 稀有度改写、非中国照片不动、dry-run 零写入。
Builder tests for the backfill script using synthetic DBs — offline.
"""
import os
import sqlite3
import sys

import pytest

from scripts_dev.backfill_china_fields import backfill


@pytest.fixture
def env(tmp_path, monkeypatch):
    """合成 report.db（v13 形状的最小列集）+ 合成字典库。"""
    ref = os.path.join(tmp_path, "ref.sqlite")
    conn = sqlite3.connect(ref)
    conn.executescript(
        """
        CREATE TABLE BirdCountInfo (model_class_id INTEGER PRIMARY KEY,
            scientific_name TEXT, chinese_simplified TEXT);
        CREATE TABLE china_protection (model_class_id INTEGER PRIMARY KEY,
            scientific_name TEXT, level INTEGER, chinese_name TEXT, source TEXT);
        CREATE TABLE gbif_rarity_by_country (model_class_id INTEGER,
            countrycode TEXT, cc_cn_count INTEGER, gbif_rarity_100 REAL,
            snapshot_date TEXT, source TEXT,
            PRIMARY KEY (model_class_id, countrycode));
        INSERT INTO BirdCountInfo VALUES (1041, 'Aquila chrysaetos', '金雕');
        INSERT INTO BirdCountInfo VALUES (9380, 'Passer montanus', '麻雀');
        INSERT INTO china_protection VALUES (1041, 'Aquila chrysaetos', 1, '金雕', 't');
        INSERT INTO gbif_rarity_by_country VALUES
            (1041, 'CN', 2378, 34.43, '2026-08-29', 't');
        """
    )
    conn.commit()
    conn.close()

    rpt = os.path.join(tmp_path, "photo_dir", ".superpicky", "report.db")
    os.makedirs(os.path.dirname(rpt))
    conn = sqlite3.connect(rpt)
    conn.executescript(
        """
        CREATE TABLE photos (filename TEXT PRIMARY KEY, bird_species_cn TEXT,
            gps_latitude REAL, gps_longitude REAL,
            gbif_rarity_100 REAL, china_protection_level INTEGER);
        CREATE TABLE bird_detections (filename TEXT, bird_index INTEGER,
            class_id INTEGER, gbif_rarity_100 REAL,
            china_protection_level INTEGER);
        -- 金雕在青海（中国，应有 CN 分+一级）；麻雀在北京；灰 spy 无GPS；
        -- 人工改名行连不上保护表应保持 NULL
        INSERT INTO photos VALUES
            ('A', '金雕',   36.0,  96.0,  4.17, NULL),
            ('B', '麻雀',   39.9, 116.4,  1.2,  NULL),
            ('C', '金雕',   NULL,  NULL,  4.17, NULL),
            ('D', '自定义鸟', 30.0, 120.0, 9.9,  NULL);
        INSERT INTO bird_detections VALUES
            ('A', 0, 1041, 4.17, NULL),
            ('A', 1, 9380, 1.2,  NULL),
            ('C', 0, 1041, 4.17, NULL);
        """
    )
    conn.commit()
    conn.close()

    import scripts_dev.backfill_china_fields as mod
    monkeypatch.setattr(mod, "REF_DB", ref)
    # 中国坐标 → CN，其余 → AU（断网离线 / offline）
    monkeypatch.setattr(mod, "_load_country_lookup",
                        lambda coords: {fn: ("CN" if lat > 20 else "AU")
                                        for fn, lat, _ in coords})
    return rpt


def test_backfill_protection_and_cn_rarity(env):
    backfill(env, dry_run=False, do_rarity=True)
    conn = sqlite3.connect(env)
    # A 金雕（中国）：CN 分 34.43 + 一级；检测行同享
    a = conn.execute("SELECT gbif_rarity_100, china_protection_level "
                     "FROM photos WHERE filename='A'").fetchone()
    assert a == (34.43, 1)
    det = {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT bird_index, gbif_rarity_100, china_protection_level "
        "FROM bird_detections WHERE filename='A'").fetchall()}
    assert det[0] == (34.43, 1)   # 金雕检测行
    assert det[1] == (1.2, None)  # 麻雀：无 CN 行 → 保留全球分、无保护
    # B 麻雀在中国但无 CN 记录：全球分保留，中文名连不上保护表 → NULL
    b = conn.execute("SELECT gbif_rarity_100, china_protection_level "
                     "FROM photos WHERE filename='B'").fetchone()
    assert b == (1.2, None)
    # C 无 GPS：保护等级仍按物种回填，稀有度保持全球分
    c = conn.execute("SELECT gbif_rarity_100, china_protection_level "
                     "FROM photos WHERE filename='C'").fetchone()
    assert c == (4.17, 1)
    cd = conn.execute("SELECT gbif_rarity_100 FROM bird_detections "
                      "WHERE filename='C'").fetchone()
    assert cd == (4.17,)  # 照片非 CN（无GPS→空国别）→ 检测行不动
    # D 人工改名：保护连不上 → NULL，非中国 → 稀有度不动
    d = conn.execute("SELECT gbif_rarity_100, china_protection_level "
                     "FROM photos WHERE filename='D'").fetchone()
    assert d == (9.9, None)
    conn.close()


def test_dry_run_writes_nothing(env):
    backfill(env, dry_run=True, do_rarity=True)
    conn = sqlite3.connect(env)
    n = conn.execute("SELECT COUNT(*) FROM photos WHERE "
                     "china_protection_level IS NOT NULL").fetchone()[0]
    r = conn.execute("SELECT gbif_rarity_100 FROM photos "
                     "WHERE filename='A'").fetchone()
    conn.close()
    assert n == 0
    assert r == (4.17,)  # 原值未动
