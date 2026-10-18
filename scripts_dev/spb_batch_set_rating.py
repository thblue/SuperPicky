#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_batch_set_rating — 按鸟种批量人工改星（结果浏览器单张改星链路的批量版）。

场景：某鸟种整批被 V2 定星压到 0★/低星，人工判断整体应提档（如雉鸡）。
逐张在结果浏览器里打星要重复 N 次，本工具把与浏览器单张改星
（ResultsBrowser._on_rating_changed）完全同构的写链路批量执行：

1. report.db  photos.rating（经 ReportDB.update_photo，自动刷新 updated_at）；
2. sidecar JSON（core.sidecar_export.export_directory_sidecars 增量重导出，
   只重写内容变化的 meta/*.json——浏览器改星后的 sidecar 补导出同款）；
3. XMP Rating（tools.exiftool_manager.set_rating_and_pick，专有 RAW 自动
   落 .xmp 侧车，遵守全局 metadata_write_mode：none 时跳过不碰文件）。

零接触照片本体：不移动、不重命名、不重写 RAW（单一写者原则——本脚本
即 SuperPicky 运维上下文，与 rerate_v2 等运维工具同一立场）。

安全设计 / Safety:
- 默认 dry-run：只列出命中文件与星级分布，不写任何数据；
- --execute 前自动备份 report.db（先 TRUNCATE checkpoint 再复制，
  确保 WAL 中已提交事务进入备份，文件名含操作描述与时间戳）；
- 幂等：已处于目标星级的行自动跳过，可安全重跑。

用法 / Usage:
    python scripts_dev/spb_batch_set_rating.py "<照片目录>" \
        --species 雉鸡 --from 0 --to 1            # dry-run 预览
    python scripts_dev/spb_batch_set_rating.py "<照片目录>" \
        --species 雉鸡 --from 0 --to 1 --execute  # 实际写入

参数:
    directory: 批处理根目录（其下应有 .superpicky/report.db）
    --species: 鸟种名（中文名或英文名，子串匹配、英文不区分大小写）
    --from:    只改当前处于该星级的行（-1/0/1/2/3）；缺省=不限
    --to:      目标星级（-1/0/1/2/3，必填）
    --execute: 实际写入（缺省 dry-run）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import VIDEO_COVER_SUFFIX  # noqa: E402
from core.sidecar_export import export_directory_sidecars  # noqa: E402
from tools.exiftool_manager import get_exiftool_manager  # noqa: E402
from tools.report_db import ReportDB  # noqa: E402

# SuperPicky 内部星级取值 / internal rating scale
VALID_RATINGS = {-1, 0, 1, 2, 3}


def backup_report_db(directory: str, tag: str) -> str:
    """
    备份 report.db：先 TRUNCATE checkpoint 把 WAL 并入主库再复制。

    参数:
    directory (str): 照片目录（report.db 位于其 .superpicky/ 下）
    tag (str): 备份文件名中的操作描述（如「批量改星」）

    返回:
    str: 备份文件完整路径

    Back up report.db. A TRUNCATE checkpoint first folds committed WAL
    transactions into the main file so the plain-copy backup is complete
    even if the library was last written through WAL.
    """
    db_path = os.path.join(directory, ".superpicky", "report.db")
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001（checkpoint 失败仅降级为普通复制）
        print(f"⚠️ WAL checkpoint 失败（备份可能缺最新事务）: {e}")
    bak = db_path + f".bak_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(db_path, bak)
    return bak


def select_targets(photos: List[dict], species: str,
                   from_rating: Optional[int], to_rating: int) -> List[dict]:
    """
    从 photos 全表中筛出本次要改星的行。

    规则：鸟种子串匹配（中/英文名，英文不区分大小写）；--from 限定当前
    星级；跳过 V-cover 行（固定 2 星，不参与人工定档）与已处于目标
    星级的行（幂等重跑）。

    参数:
    photos (List[dict]): ReportDB.get_all_photos() 的返回
    species (str): 鸟种名关键字
    from_rating (Optional[int]): 当前星级限定（None=不限）
    to_rating (int): 目标星级

    返回:
    List[dict]: 命中的照片行

    Pick the rows to re-rate: substring species match on CN/EN names
    (EN case-insensitive), optional current-rating gate, skipping V-cover
    rows and rows already at the target rating (idempotent reruns).
    """
    key = species.strip().lower()
    targets = []
    for row in photos:
        fn = row.get("filename") or ""
        if fn.endswith(VIDEO_COVER_SUFFIX):
            continue
        if int(row.get("rating") or 0) == to_rating:
            continue
        if from_rating is not None and int(row.get("rating") or 0) != from_rating:
            continue
        cn = (row.get("bird_species_cn") or "").lower()
        en = (row.get("bird_species_en") or "").lower()
        if key in cn or key in en:
            targets.append(row)
    return targets


def resolve_photo_path(directory: str, row: dict) -> str:
    """
    解析一张照片的当前磁盘路径（current_path > original_path > 根目录拼接）。

    参数:
    directory (str): 批处理根目录
    row (dict): photos 行

    返回:
    str: 绝对路径（不保证存在，调用方需检查）

    Resolve a photo's current on-disk path from its DB row.
    """
    for col in ("current_path", "original_path"):
        p = row.get(col) or ""
        if p and os.path.exists(p):
            return p
    return os.path.join(directory, row.get("filename") or "")


def write_xmp_ratings(targets: List[dict], directory: str,
                      to_rating: int) -> int:
    """
    逐张写 XMP Rating（与浏览器改星同款：set_rating_and_pick，Pick=0）。

    全局 metadata_write_mode == "none" 时整体跳过（星级只进 DB/sidecar，
    不碰照片文件——与浏览器行为一致）。

    参数:
    targets (List[dict]): 命中的照片行
    directory (str): 批处理根目录
    to_rating (int): 目标星级

    返回:
    int: 写入（或按配置跳过）成功的张数

    Write XMP Rating per photo via the browser-identical helper. Honors
    metadata_write_mode=none by skipping the whole pass. Returns the
    number of photos written (config-skip counts as success).
    """
    from advanced_config import get_advanced_config
    mode = str(get_advanced_config().get_metadata_write_mode() or "").lower()
    if mode == "none":
        print("[XMP] metadata_write_mode=none，按配置跳过 XMP 写入"
              "（星级只落 report.db 与 sidecar）")
        return len(targets)

    mgr = get_exiftool_manager()
    ok = 0
    for row in targets:
        path = resolve_photo_path(directory, row)
        if not os.path.exists(path):
            print(f"⚠️ 文件缺失，跳过 XMP: {row.get('filename')}")
            continue
        if mgr.set_rating_and_pick(path, to_rating):
            ok += 1
        else:
            print(f"❌ XMP 写入失败: {row.get('filename')}")
    return ok


def verify(report_db: ReportDB, directory: str, species: str,
           to_rating: int, sample: List[dict]) -> bool:
    """
    写后自校验：DB 分布复核 + sidecar JSON 抽查 + XMP Rating 抽查。

    参数:
    report_db (ReportDB): 已打开的库连接
    directory (str): 批处理根目录
    species (str): 鸟种名关键字
    to_rating (int): 目标星级
    sample (List[dict]): 抽查的照片行（最多 3 张）

    返回:
    bool: 全部通过为 True

    Post-write verification: DB distribution recheck plus sidecar/XMP
    spot checks on a small sample.
    """
    ok = True
    dist: Dict[int, int] = {}
    for row in select_targets_all(report_db, species):
        dist[int(row.get("rating") or 0)] = dist.get(int(row.get("rating") or 0), 0) + 1
    print(f"[校验] DB 中该鸟种星级分布: {dict(sorted(dist.items()))}")
    if to_rating not in dist:
        print("❌ DB 复核失败：没有行处于目标星级")
        ok = False

    for row in sample:
        fn = row.get("filename") or ""
        stem = os.path.splitext(fn)[0]
        meta_path = os.path.join(directory, ".superpicky", "meta", stem + ".json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            got = (data.get("processing") or {}).get("rating")
            if got != to_rating:
                print(f"❌ sidecar 复核失败 {stem}.json: processing.rating={got}")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"❌ sidecar 读取失败 {stem}.json: {e}")
            ok = False

    from advanced_config import get_advanced_config
    mode = str(get_advanced_config().get_metadata_write_mode() or "").lower()
    if mode != "none":
        mgr = get_exiftool_manager()
        for row in sample:
            # 专有 RAW 的 Rating 落在 .xmp 侧车，exiftool 读 RAW 本体不会
            # 带出侧车内容——抽查必须显式读侧车文件。
            # Proprietary RAWs keep Rating in the .xmp sidecar; reading the
            # RAW body would not surface it, so probe the sidecar directly.
            stem = os.path.splitext(row.get("filename") or "")[0]
            sidecar = os.path.join(directory, stem + ".xmp")
            path = sidecar if os.path.exists(sidecar) \
                else resolve_photo_path(directory, row)
            tags = mgr.read_metadata(path) or {}
            got = tags.get("Rating")
            if got is None:
                # 兼容带组名返回（XMP:Rating）
                got = tags.get("XMP:Rating")
            if str(got) != str(to_rating):
                print(f"❌ XMP 复核失败 {os.path.basename(path)}: Rating={got}")
                ok = False
    return ok


def select_targets_all(report_db: ReportDB, species: str) -> List[dict]:
    """
    取该鸟种当前全部行（含各星级），供分布复核。

    参数:
    report_db (ReportDB): 已打开的库连接
    species (str): 鸟种名关键字

    返回:
    List[dict]: 命中的照片行

    Fetch every row of the species (all ratings) for distribution checks.
    """
    return select_targets(report_db.get_all_photos(), species,
                          from_rating=None, to_rating=-99)


def main() -> int:
    """
    命令行入口：dry-run 预览 / --execute 三段写入 + 自校验。

    返回:
    int: 0 成功；1 参数或前置条件错误；2 自校验失败

    CLI entry: dry-run preview by default; --execute performs the
    DB → sidecar → XMP write chain followed by verification.
    """
    ap = argparse.ArgumentParser(description="按鸟种批量人工改星（默认 dry-run）")
    ap.add_argument("directory", help="照片目录（含 .superpicky/report.db）")
    ap.add_argument("--species", required=True, help="鸟种名（中/英文，子串匹配）")
    ap.add_argument("--from", dest="from_rating", type=int, default=None,
                    help="只改当前处于该星级的行（缺省不限）")
    ap.add_argument("--to", dest="to_rating", type=int, required=True,
                    help="目标星级（-1/0/1/2/3）")
    ap.add_argument("--execute", action="store_true", help="实际写入（缺省 dry-run）")
    args = ap.parse_args()

    directory = os.path.abspath(args.directory)
    if args.to_rating not in VALID_RATINGS:
        print(f"❌ 非法目标星级: {args.to_rating}")
        return 1
    if args.from_rating is not None and args.from_rating not in VALID_RATINGS:
        print(f"❌ 非法 --from 星级: {args.from_rating}")
        return 1
    if not os.path.isdir(directory):
        print(f"❌ 目录不存在: {directory}")
        return 1

    db = ReportDB(directory)
    try:
        targets = select_targets(db.get_all_photos(), args.species,
                                 args.from_rating, args.to_rating)
        print(f"命中 {len(targets)} 张（species={args.species!r}, "
              f"from={args.from_rating}, to={args.to_rating}）")
        for row in targets:
            print(f"  {row.get('filename')}  "
                  f"{int(row.get('rating') or 0)}★ → {args.to_rating}★  "
                  f"[{row.get('bird_species_cn')} / {row.get('bird_species_en')}]")
        if not targets:
            print("没有需要修改的行，结束。")
            return 0
        if not args.execute:
            print("\n[dry-run] 未写入。加 --execute 执行。")
            return 0

        bak = backup_report_db(directory, tag="批量改星")
        print(f"已备份 report.db → {os.path.basename(bak)}")

        # 1) DB：与浏览器 update_photo 同款（自动刷新 updated_at）
        done = 0
        for row in targets:
            if db.update_photo(row["filename"], {"rating": args.to_rating}):
                done += 1
            else:
                print(f"❌ DB 更新失败: {row.get('filename')}")
        print(f"[1/3] DB rating 已更新 {done}/{len(targets)}")

        # 2) sidecar：增量重导出（只重写内容变化的 meta/*.json）
        n = export_directory_sidecars(db, directory, log=lambda *_: None)
        print(f"[2/3] sidecar 增量重导出完成，本次写入 {n} 个 JSON")

        # 3) XMP：与浏览器同款写入口
        ok = write_xmp_ratings(targets, directory, args.to_rating)
        print(f"[3/3] XMP Rating 写入 {ok}/{len(targets)}")

        sample = targets[:3]
        if verify(db, directory, args.species, args.to_rating, sample):
            print("✅ 自校验通过（DB 分布 + sidecar 抽查 + XMP 抽查）")
            return 0
        print("⚠️ 自校验未完全通过，请检查上方 ❌ 条目（可从备份恢复）")
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
