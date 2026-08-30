# -*- coding: utf-8 -*-
"""
回填历史 report.db 的中国稀有度与国家保护等级 / Backfill the China-scoped
rarity and national protection level into existing .superpicky/report.db.

背景 / Background:
    bird_reference.sqlite（字典层）更新后，详情面板显示的仍是各照片目录
    .superpicky/report.db（结果层）里识别时刻冻结的值：历史照片的
    gbif_rarity_100 是当时的全球分，china_protection_level 全为 NULL。
    本脚本不重跑识别模型，纯数据库回填：
      - schema v12→v13 迁移：历史库没有 china_protection_level 列，先按
        report_db.py 的语义补列并升 meta.schema_version（应用之后打开
        同一库时迁移链保持空转，不会冲突）。
      - 国家保护等级：photos 按 bird_species_cn、bird_detections 按
        class_id 连接 china_protection（物种级属性，与拍摄地无关）。
      - 中国稀有度：photos 已存 GPS 经纬度，离线 reverse-geocode 出国家；
        仅中国照片改写为 gbif_rarity_by_country 的 CN 分（非中国照片保持
        全球分——全球分与国家无关，无需改动）。无 GPS 的照片用 --country
        显式指定国家（如 --country CN）参与回填；不指定则跳过——无证据时
        猜国家会把伦敦/新加坡的照片误标成中国口径。detections 跟随所属
        照片的国家按 class_id 回填。
    After the reference-DB update, the detail panel still shows values
    frozen at identification time in each directory's report.db. This
    script backfills without re-running the model: it first migrates the
    v12 schema (adding china_protection_level exactly like report_db.py
    would), then fills protection by species/class_id joins and CN rarity
    by offline reverse-geocoding of the stored GPS (non-CN photos keep
    their global score, which is country-independent). GPS-less photos
    participate only when --country is given explicitly — guessing a
    country would mislabel London or Singapore shots as CN-scoped.

事务与备份 / Transaction & backup:
    「迁移 + 回填」在同一个显式事务里执行：--dry-run 回滚时连 ALTER
    TABLE 一起撤销（文件零写入）；正式跑则原子提交，不会留下半迁移的库。
    改写前用 SQLite backup API 做一致性整库备份（report.db.bak-<时间戳>，
    --no-backup 跳过）——比文件拷贝可靠，能正确捕获 WAL 中的内容。
    Migration and backfill share one explicit transaction: --dry-run
    rolls the ALTERs back too (zero bytes written); a real run commits
    atomically. A consistent snapshot backup is taken first via the
    SQLite backup API (report.db.bak-<timestamp>, --no-backup to skip) —
    reliable where a plain file copy would miss WAL content.

用法 / Usage:
    .venv/Scripts/python scripts_dev/backfill_china_fields.py <照片目录>
    .venv/Scripts/python scripts_dev/backfill_china_fields.py <目录> --dry-run
    .venv/Scripts/python scripts_dev/backfill_china_fields.py <目录> --country CN
    多个目录逐个执行 / run once per photo directory.

注意 / Notes:
    只改 report.db；已导出的 sidecar JSON / 嵌入 XMP 需要在应用里重新
    导出才会带新值。
    Only report.db is touched; re-export sidecars/XMP from the app to
    refresh them.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DB = os.path.join(PROJ, "birdid", "data", "bird_reference.sqlite")


def _load_country_lookup(coords: List[Tuple[str, float, float]]) -> Dict[str, str]:
    """
    离线反解一批 (filename, lat, lon) → ISO 国家码。

    Offline reverse-geocode (filename, lat, lon) pairs to ISO country
    codes via the same reverse_geocoder the app uses at identification
    time. Unresolvable coordinates map to "".

    参数 / Parameters:
        coords (list): [(filename, lat, lon), ...]

    返回 / Returns:
        dict[str, str]: {filename: 国家码大写} / {filename: uppercase CC}
    """
    if not coords:
        return {}
    import reverse_geocoder as rg

    results = rg.search([(lat, lon) for _, lat, lon in coords],
                        mode=2, verbose=False)
    out: Dict[str, str] = {}
    for (fn, _lat, _lon), r in zip(coords, results):
        out[fn] = str((r or {}).get("cc") or "").upper()
    return out


def _ensure_v13_columns(conn: sqlite3.Connection) -> None:
    """
    v12 → v13 迁移：给 photos / bird_detections 补 china_protection_level 列。

    NAS 上的历史库全部是 v12（列不存在），而本脚本用裸 sqlite3 直连，不会
    触发应用内的迁移链。这里逐字复刻 report_db.py v12→v13 的语义：列已
    存在则跳过（OperationalError 容忍），meta.schema_version 升到 '13'，
    保证应用之后打开同一库时不会再重复迁移。不在此处提交——由调用方的
    显式事务统一提交/回滚（dry-run 时 ALTER 也会被撤销）。

    v12 → v13 migration mirroring report_db.py verbatim: add the column
    to both tables when missing (tolerating "already exists") and raise
    meta.schema_version to '13', so the app's own migration chain stays a
    no-op afterwards. No commit here; the caller's explicit transaction
    commits or rolls everything back together.

    参数 / Parameters:
        conn (sqlite3.Connection): 已 ATTACH 字典库的连接 / Connection
            with the reference DB attached.

    返回 / Returns:
        bool: True 表示本次执行了迁移 / True when a migration ran.
    """
    def _has_col(table: str) -> bool:
        return any(row[1] == "china_protection_level"
                   for row in conn.execute(f"PRAGMA table_info({table})"))

    if _has_col("photos") and _has_col("bird_detections"):
        return False
    for stmt in (
        "ALTER TABLE photos ADD COLUMN china_protection_level INTEGER",
        "ALTER TABLE bird_detections ADD COLUMN china_protection_level INTEGER",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # 列已存在 / column already exists
    conn.execute(
        "UPDATE meta SET value = '13' WHERE key = 'schema_version'")
    print("[backfill] schema v12→v13：补 china_protection_level 列 / column added")
    return True


def backfill(
    db_path: str,
    dry_run: bool = False,
    do_rarity: bool = True,
    default_country: Optional[str] = None,
    do_backup: bool = True,
) -> Dict[str, int]:
    """
    对单个 report.db 执行「迁移 + 回填」。

    Backfill one report.db. Protection joins are pure SQL; CN rarity
    needs offline reverse-geocoding of stored GPS. Migration and data
    updates share one explicit transaction — rolled back entirely under
    --dry-run (including the schema change), committed atomically
    otherwise. A consistent snapshot backup is taken first via the
    SQLite backup API (unlike a file copy it also captures WAL content).

    参数 / Parameters:
        db_path (str): .superpicky/report.db 路径 / Path to report.db.
        dry_run (bool): True 只统计不写（含 schema 迁移一并回滚）/
            Count only; the schema migration rolls back too.
        do_rarity (bool): False 跳过稀有度、只回填保护等级 / Skip the
            rarity pass when False.
        default_country (Optional[str]): 无 GPS 照片的国家兜底（ISO 大写，
            如 'CN'）。仅当用户显式传入时生效——无证据不猜国家 / Country
            fallback for GPS-less photos (ISO uppercase). Only applied when
            explicitly provided; never guess.
        do_backup (bool): False 跳过整库备份 / Skip the snapshot backup.

    返回 / Returns:
        dict[str, int]: 统计（prot_photos/prot_detections/cn_photos/
            rarity_photos/rarity_detections/migrated），dry-run 与正式跑
            数值一致 / Stats identical between dry-run and real run.

    异常 / Exceptions:
        FileNotFoundError: report.db 或字典库缺失时抛出 / Raised when
            the report DB or the reference DB is missing.
    """
    stats: Dict[str, int] = {
        "migrated": 0,
        "prot_photos": 0,
        "prot_detections": 0,
        "cn_photos": 0,
        "rarity_photos": 0,
        "rarity_detections": 0,
    }
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"report.db 不存在 / missing: {db_path}")
    if not os.path.exists(REF_DB):
        raise FileNotFoundError(f"字典库不存在 / missing: {REF_DB}")

    conn = sqlite3.connect(db_path)
    conn.execute("ATTACH DATABASE ? AS ref", (REF_DB,))
    # 显式事务：isolation_level=None 关掉隐式事务，BEGIN IMMEDIATE 先拿
    # 写锁（NAS 上若有应用正持有写锁可即刻失败而不是写到一半失败）。
    # Explicit transaction: implicit txn handling off; BEGIN IMMEDIATE
    # grabs the write lock upfront so a busy writer fails fast.
    conn.isolation_level = None
    try:
        if do_backup and not dry_run:
            bak = db_path + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = sqlite3.connect(bak)
            try:
                conn.backup(dest)
            finally:
                dest.close()
            print(f"[backfill] 备份 / backup → {bak}")

        conn.execute("BEGIN IMMEDIATE")
        try:
            stats["migrated"] = 1 if _ensure_v13_columns(conn) else 0

            # ---------- 1) 国家保护等级 ----------
            # detections 有 class_id，直接连表；photos 无 class_id，按库内
            # 中文名连接（photos 的鸟种名来自 BirdCountInfo，可与
            # china_protection.chinese_name 精确匹配；人工改名的历史行保持 NULL）。
            before_det = conn.execute(
                "SELECT COUNT(*) FROM bird_detections WHERE china_protection_level "
                "IS NOT NULL").fetchone()[0]
            cur = conn.execute(
                "UPDATE bird_detections SET china_protection_level = ("
                "  SELECT p.level FROM ref.china_protection p"
                "  WHERE p.model_class_id = bird_detections.class_id)"
                "WHERE class_id IS NOT NULL")
            n_det = cur.rowcount
            after_det = conn.execute(
                "SELECT COUNT(*) FROM bird_detections WHERE china_protection_level "
                "IS NOT NULL").fetchone()[0]

            before_ph = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE china_protection_level "
                "IS NOT NULL").fetchone()[0]
            cur = conn.execute(
                "UPDATE photos SET china_protection_level = ("
                "  SELECT p.level FROM ref.china_protection p"
                "  WHERE p.chinese_name = photos.bird_species_cn)"
                "WHERE bird_species_cn IS NOT NULL")
            n_ph = cur.rowcount
            after_ph = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE china_protection_level "
                "IS NOT NULL").fetchone()[0]
            stats["prot_photos"] = after_ph - before_ph
            stats["prot_detections"] = after_det - before_det
            print(f"[backfill] 保护等级 / protection: photos +{after_ph - before_ph}"
                  f"（扫描 {n_ph}），detections +{after_det - before_det}"
                  f"（扫描 {n_det}）")

            # ---------- 2) 中国稀有度 ----------
            if do_rarity:
                gps_rows = conn.execute(
                    "SELECT filename, gps_latitude, gps_longitude FROM photos "
                    "WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL"
                ).fetchall()
                country_of = _load_country_lookup(
                    [(fn, lat, lon) for fn, lat, lon in gps_rows])
                # 无 GPS 照片：仅当用户显式给 --country 时才赋予国家；
                # 显式传入的兜底不覆盖 GPS 反解结果（GPS 永远优先）。
                # GPS-less photos get the explicit --country fallback only;
                # an explicit fallback never overrides a GPS-derived code.
                if default_country:
                    all_files = [r[0] for r in conn.execute(
                        "SELECT filename FROM photos")]
                    for fn in all_files:
                        country_of.setdefault(fn, default_country)
                cn_files = [fn for fn, cc in country_of.items() if cc == "CN"]
                stats["cn_photos"] = len(cn_files)
                print(f"[backfill] GPS 反解 / geocoded: {len(country_of)} 张，"
                      f"中国 {len(cn_files)} 张"
                      + (f"（含 --country {default_country} 兜底）" if default_country else ""))

                # 中国照片：按中文名 → model_class_id → CN 分改写 photos
                # detections 跟随照片国家按 class_id 改写
                n_photo_r = n_det_r = 0
                for fn in cn_files:
                    row = conn.execute(
                        "SELECT g.gbif_rarity_100 FROM ref.gbif_rarity_by_country g"
                        " JOIN ref.BirdCountInfo b ON b.model_class_id = g.model_class_id"
                        " WHERE g.countrycode = 'CN' AND b.chinese_simplified = ("
                        "   SELECT bird_species_cn FROM photos WHERE filename = ?)",
                        (fn,)).fetchone()
                    if row is None:
                        continue  # 中国无记录 → 保留全球分（运行时同语义）
                    conn.execute(
                        "UPDATE photos SET gbif_rarity_100 = ? WHERE filename = ?",
                        (row[0], fn))
                    n_photo_r += 1
                if cn_files:
                    qmarks = ",".join("?" * len(cn_files))
                    # 只改写「有 CN 分」的检测行；中国无记录的物种（如照片里
                    # 顺带的麻雀）保留全球分——与运行时回退语义一致
                    # Only rewrite detections that actually have a CN row;
                    # species without CN records keep their global score,
                    # mirroring the runtime fallback semantics.
                    cur = conn.execute(
                        f"UPDATE bird_detections SET gbif_rarity_100 = ("
                        f"  SELECT g.gbif_rarity_100 FROM ref.gbif_rarity_by_country g"
                        f"  WHERE g.countrycode = 'CN'"
                        f"    AND g.model_class_id = bird_detections.class_id)"
                        f"WHERE filename IN ({qmarks}) AND class_id IS NOT NULL"
                        f"  AND EXISTS (SELECT 1 FROM ref.gbif_rarity_by_country g2"
                        f"              WHERE g2.countrycode = 'CN'"
                        f"                AND g2.model_class_id = bird_detections.class_id)",
                        cn_files)
                    n_det_r = cur.rowcount
                stats["rarity_photos"] = n_photo_r
                stats["rarity_detections"] = n_det_r
                print(f"[backfill] 中国稀有度 / CN rarity: photos {n_photo_r}，"
                      f"detections {n_det_r}"
                      f"（非中国照片保持全球分 / non-CN photos untouched）")

            if dry_run:
                conn.execute("ROLLBACK")
                print("[backfill] --dry-run：未写入任何更改 / no changes written")
            else:
                conn.execute("COMMIT")
                print(f"[backfill] 已提交 / committed: {db_path}")
            return stats
        except Exception:
            # 任何一步失败：整体回滚，不留下半迁移/半回填的库。
            # Any failure rolls the whole thing back — no half-migrated DB.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Backfill CN rarity + national protection into report.db")
    p.add_argument("directory", help="照片目录（含 .superpicky/report.db）")
    p.add_argument("--dry-run", action="store_true",
                   help="只统计不写入 / report only, no writes")
    p.add_argument("--no-rarity", action="store_true",
                   help="跳过稀有度，只回填保护等级 / protection only")
    p.add_argument("--no-backup", action="store_true",
                   help="跳过整库备份 / skip the DB backup")
    p.add_argument("--country", default=None, metavar="CC",
                   help="无 GPS 照片的国家兜底（ISO 大写，如 CN）；不传则"
                        "无 GPS 照片不改稀有度 / country fallback for "
                        "GPS-less photos; omit to skip them entirely")
    a = p.parse_args()

    db_path = os.path.join(a.directory, ".superpicky", "report.db")
    country = a.country.strip().upper() if a.country else None
    backfill(db_path, dry_run=a.dry_run, do_rarity=not a.no_rarity,
             default_country=country, do_backup=not a.no_backup)


if __name__ == "__main__":
    main()
