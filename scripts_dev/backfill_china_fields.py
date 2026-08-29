# -*- coding: utf-8 -*-
"""
回填历史 report.db 的中国稀有度与国家保护等级 / Backfill the China-scoped
rarity and national protection level into existing .superpicky/report.db.

背景 / Background:
    bird_reference.sqlite（字典层）更新后，详情面板显示的仍是各照片目录
    .superpicky/report.db（结果层）里识别时刻冻结的值：历史照片的
    gbif_rarity_100 是当时的全球分，china_protection_level 全为 NULL。
    本脚本不重跑识别模型，纯数据库回填：
      - 国家保护等级：photos 按 bird_species_cn、bird_detections 按
        class_id 连接 china_protection（物种级属性，与拍摄地无关）。
      - 中国稀有度：photos 已存 GPS 经纬度，离线 reverse-geocode 出国家；
        仅中国照片改写为 gbif_rarity_by_country 的 CN 分（非中国照片保持
        全球分——全球分与国家无关，无需改动）。detections 跟随所属照片
        的国家按 class_id 回填。
    After the reference-DB update, the detail panel still shows values
    frozen at identification time in each directory's report.db. This
    script backfills without re-running the model: protection by
    species/class_id joins, CN rarity by offline reverse-geocoding of the
    stored GPS (non-CN photos keep their global score, which is
    country-independent).

用法 / Usage:
    .venv/Scripts/python scripts_dev/backfill_china_fields.py <照片目录>
    .venv/Scripts/python scripts_dev/backfill_china_fields.py <目录> --dry-run
    多个目录逐个执行 / run once per photo directory.

注意 / Notes:
    只改 report.db；已导出的 sidecar JSON / 嵌入 XMP 需要在应用里重新
    导出才会带新值。改写前脚本自动做一次 WAL 安全的整库备份
    （report.db.bak-<时间戳>，--no-backup 跳过）。
    Only report.db is touched; re-export sidecars/XMP from the app to
    refresh them. A timestamped DB backup is made first unless
    --no-backup.
"""
from __future__ import annotations

import argparse
import os
import shutil
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


def backfill(db_path: str, dry_run: bool = False,
             do_rarity: bool = True) -> None:
    """
    对单个 report.db 执行回填。

    Backfill one report.db. Protection joins are pure SQL; CN rarity
    needs offline reverse-geocoding of stored GPS. Everything runs in
    one transaction (skipped under --dry-run, which only reports).

    参数 / Parameters:
        db_path (str): .superpicky/report.db 路径 / Path to report.db.
        dry_run (bool): True 只统计不写 / Count only, no writes.
        do_rarity (bool): False 跳过稀有度、只回填保护等级 / Skip the
            rarity pass when False.

    异常 / Exceptions:
        FileNotFoundError: report.db 或字典库缺失时抛出 / Raised when the
            report DB or the reference DB is missing.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"report.db 不存在 / missing: {db_path}")
    if not os.path.exists(REF_DB):
        raise FileNotFoundError(f"字典库不存在 / missing: {REF_DB}")

    conn = sqlite3.connect(db_path)
    conn.execute("ATTACH DATABASE ? AS ref", (REF_DB,))
    try:
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
            cn_files = [fn for fn, cc in country_of.items() if cc == "CN"]
            print(f"[backfill] GPS 反解 / geocoded: {len(country_of)} 张，"
                  f"中国 {len(cn_files)} 张")

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
                if not dry_run:
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
            print(f"[backfill] 中国稀有度 / CN rarity: photos {n_photo_r}，"
                  f"detections {n_det_r}"
                  f"（非中国照片保持全球分 / non-CN photos untouched）")

        if dry_run:
            conn.rollback()
            print("[backfill] --dry-run：未写入任何更改 / no changes written")
        else:
            conn.commit()
            print(f"[backfill] 已提交 / committed: {db_path}")
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
    a = p.parse_args()

    db_path = os.path.join(a.directory, ".superpicky", "report.db")
    if not a.dry_run and not a.no_backup and os.path.exists(db_path):
        bak = db_path + ".bak-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(db_path, bak)
        print(f"[backfill] 备份 / backup → {bak}")
    backfill(db_path, dry_run=a.dry_run, do_rarity=not a.no_rarity)


if __name__ == "__main__":
    main()
