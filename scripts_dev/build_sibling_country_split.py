# -*- coding: utf-8 -*-
"""
构建合并 key 姊妹种的国家级拆分表 / Build the country-level sibling split table.

背景 / Background:
  154 组碰撞组的姊妹类在 geo_distribution.db 里共享同一份（合并的）观察
  计数，候选集无法区分它们——模型若偏爱美洲「普通水鸡」，中国的黑水鸡
  照片就会定错种。GBIF 虽在种级（speciesKey）分不开，但每条记录保留了
  数据集原始标签的 usageKey：美洲记录全带 galeata 异名标签（CN=0），中国
  记录全带 chloropus 标签（CN=45,737）——国家级完全可以拆开（2026-09-13
  实测验证）。

  本脚本对每个碰撞组成员做学名匹配拿到 usageKey，再按国家 facet 拉该
  标签的记录数，写入 geo_distribution.db 两张表：

  - merged_key_classes(specieskey, class_id)：组员名册（组 id 即
    bird_reference 里的共享 specieskey）；
  - merged_group_country_split(specieskey, country, class_id, n_usage)：
    仅存 n>0 的行（无行=0）。运行时（geo_filter.iter_candidates）在国家
    已知时把「该组在该国 n=0 的成员」从各层候选集中剔除；某组在某国
    一行都没有（数据缺口）则整组放行，不误伤。

  全灭组（fix_geo_key_collisions.py 已按各自种级 key 单独入库）不参与
  拆分——它们已不再共享计数，无需甄别。

  Though GBIF merges these sibling species under one speciesKey, each
  occurrence keeps its dataset-original usage label: American moorhen
  records all carry the galeata synonym usage (CN=0) while Chinese ones
  carry chloropus (CN=45,737), so the split works at country level. This
  script resolves each member's usageKey and fetches per-country label
  counts into two tables that geo_filter consumes to demote zero-count
  siblings per country. Dead groups, now keyed individually, are excluded.

用法 / Usage:
  python scripts_dev/build_sibling_country_split.py             # dry-run
  python scripts_dev/build_sibling_country_split.py --execute   # 写入
  python scripts_dev/build_sibling_country_split.py --execute --resume  # 断点续跑
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_geo_key_collisions import (  # noqa: E402
    GEO_DB,
    REF_DB,
    _clean_name,
    _gbif_get,
    fetch_country_counts,
    load_groups,
)

CACHE_PATH = os.path.join(
    PROJ, "scripts_dev", "data_sources", "sibling_usage_cache.json"
)
WORKERS = 4
LICENSES = ("CC0_1_0", "CC_BY_4_0")


def match_usage_key(sci: str) -> Optional[int]:
    """
    学名 → usageKey（数据集标签键，非接受键）/ Name to usage (label) key.

    拆分表关心的是「记录被打成什么标签」，因此 SYNONYM 时直接用 usageKey
    （如 galeata 的 7340222），不像 fix 脚本那样跳到 acceptedKey。只接受
    rank=SPECIES 的 EXACT 命中；落到属级/存疑用法或无命中时返回 None
    （该类无标签数据，运行时按数据缺口放行）。

    The split cares about the LABEL records carry, so a SYNONYM usage key is
    used as-is (e.g. 7340222 for galeata), unlike the fix script which jumps
    to the accepted key. Only EXACT species-rank matches count; genus-level
    or doubtful fallbacks return None (no label data; runtime keeps the
    class under the data-gap rule).

    参数 / Parameters:
        sci (str): 学名 / Scientific name.

    返回 / Returns:
        Optional[int]: usageKey 或 None / The usage key or None.
    """
    data = _gbif_get("species/match", [("name", _clean_name(sci))])
    if not data:
        return None
    if data.get("rank") != "SPECIES" or data.get("matchType") != "EXACT":
        return None
    usage = data.get("usageKey")
    return int(usage) if usage else None


def fetch_usage_country_counts(usage_key: int) -> Optional[Dict[str, int]]:
    """
    usageKey 标签记录的国家级计数（不过滤 license，标签语义不受影响）。

    Per-country counts of records carrying this usage label. No license
    filter here: the label semantics do not depend on it.

    参数 / Parameters:
        usage_key (int): usageKey / The label key.

    返回 / Returns:
        Optional[dict]: {country: n}；失败 None / Counts or None on failure.
    """
    params = [
        ("taxonKey", str(usage_key)),
        ("hasCoordinate", "true"),
        ("hasGeospatialIssue", "false"),
        ("facet", "country"),
        ("facetLimit", "300"),
        ("limit", "0"),
    ]
    for lic in LICENSES:
        params.append(("license", lic))
    data = _gbif_get("occurrence/search", params)
    if data is None:
        return None
    out: Dict[str, int] = {}
    for f in data.get("facets") or []:
        if f.get("field") == "COUNTRY":
            for c in f.get("counts", []):
                out[str(c["name"]).upper()] = int(c["count"])
    return out


def merged_groups_with_cells() -> Dict[int, List[Dict]]:
    """
    取「仍在共享合并计数」的组 / Groups still sharing merged cell counts.

    fix_geo_key_collisions.py 执行后，胜者组全员持有（复制的）格级数据，
    无法再用「唯一胜者」判定；改用「组内任一成员有格级行」：有胜者组
    恒真，全灭组（含已按新 key 单独入库者）恒假，正好是参与拆分的集合。

    After the collision fix, every member of a winner group holds (copied)
    cell rows, so the unique-winner test no longer applies. A group joins
    when any member has cell rows: always true for winner groups, always
    false for dead groups (which are keyed individually now).

    返回 / Returns:
        dict[int, list[dict]]: {specieskey: 成员行} / Group id to members.
    """
    groups = load_groups()
    geo = sqlite3.connect(GEO_DB)
    try:
        out: Dict[int, List[Dict]] = {}
        for skey, members in groups.items():
            ids = [m["class_id"] for m in members]
            marks = ",".join("?" * len(ids))
            hit = geo.execute(
                f"SELECT 1 FROM cell_species WHERE class_id IN ({marks}) LIMIT 1",
                ids,
            ).fetchone()
            if hit:
                out[skey] = members
        return out
    finally:
        geo.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build merged_group_country_split in geo_distribution.db"
    )
    p.add_argument("--execute", action="store_true",
                   help="写入（默认 dry-run）/ write (dry-run by default)")
    p.add_argument("--resume", action="store_true",
                   help="用 sibling_usage_cache.json 续跑 / resume from cache")
    a = p.parse_args()

    groups = merged_groups_with_cells()
    print(f"[split] 参与拆分的合并组 {len(groups)}"
          f"（全灭组已单独入库，不参与）")

    # 组员名册：specieskey(组id) → [class_id]
    roster: Dict[int, List[int]] = {
        skey: [m["class_id"] for m in members] for skey, members in groups.items()
    }

    cache: Dict[str, dict] = {}
    if a.resume and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[split] 缓存命中 / cached: {len(cache)}")

    # ---- 1) 逐类解析 usageKey + 国家 facet（缓存优先）----
    member_of: Dict[int, int] = {
        m["class_id"]: skey
        for skey, members in groups.items()
        if skey in roster
        for m in members
    }
    flat: Dict[int, str] = {
        cid: m["sci"]
        for skey, members in groups.items()
        if skey in roster
        for m in members
        for cid in (m["class_id"],)
    }

    def _work(cid: int) -> Optional[Dict[str, int]]:
        key = str(cid)
        if key in cache:
            entry = cache[key]
            return entry["countries"] if entry.get("usage_key") else None
        usage = match_usage_key(flat[cid])
        if usage is None:
            cache[key] = {"usage_key": None, "countries": {}}
            return None
        counts = fetch_usage_country_counts(usage)
        if counts is None:
            return None            # 查询失败不写缓存，留待续跑
        cache[key] = {"usage_key": usage, "countries": counts}
        return counts

    print(f"[split] 解析 {len(flat)} 个类名的 usageKey + 国家计数 ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        per_class = list(ex.map(_work, sorted(flat)))
    print(f"[split] 解析完成，用时 {time.time() - t0:.0f}s")

    # ---- 2) 组装行 + 摘要 ----
    split_rows: List[Tuple[int, str, int, int]] = []
    roster_rows: List[Tuple[int, int]] = []
    labeled = 0
    for skey, cids in roster.items():
        for cid in cids:
            roster_rows.append((skey, cid))
            entry = cache.get(str(cid))
            if not entry or not entry.get("usage_key"):
                continue
            labeled += 1
            for cc, n in entry["countries"].items():
                if n > 0:
                    split_rows.append((skey, cc, cid, n))
    print(f"[split] 组员名册 {len(roster_rows)} 行 | 有标签数据的类 {labeled} | "
          f"拆分行 {len(split_rows):,}")

    # 摘要：旗舰组的中国/美国拆分
    ref = sqlite3.connect(f"file:{REF_DB}?mode=ro", uri=True)
    names = dict(ref.execute(
        "SELECT model_class_id, chinese_simplified FROM BirdCountInfo"
    ).fetchall())
    ref.close()
    geo = sqlite3.connect(GEO_DB)
    split_lookup = {(sk, cc, c): n for sk, cc, c, n in split_rows}
    for cid in (1388, 1389, 6506, 10876, 1119, 1120, 8175, 8176):
        skey = member_of.get(cid)
        if skey is None:
            continue
        cn_n = split_lookup.get((skey, "CN", cid), 0)
        us_n = split_lookup.get((skey, "US", cid), 0)
        print(f"  {names.get(cid, cid)}({cid}): usage CN={cn_n} US={us_n}")

    if not a.execute:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"[split] dry-run 结束（缓存已存 {CACHE_PATH}）。加 --execute 写库。")
        geo.close()
        return

    # ---- 3) 写库（geo_distribution.db 内部表，不在对外契约内）----
    geo.executescript(
        """
        CREATE TABLE IF NOT EXISTS merged_key_classes (
            specieskey INTEGER NOT NULL,
            class_id   INTEGER NOT NULL,
            PRIMARY KEY (specieskey, class_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS merged_group_country_split (
            specieskey INTEGER NOT NULL,
            country    TEXT NOT NULL,
            class_id   INTEGER NOT NULL,
            n_usage    INTEGER NOT NULL,
            PRIMARY KEY (specieskey, country, class_id)
        ) WITHOUT ROWID;
        """
    )
    geo.execute("DELETE FROM merged_key_classes")
    geo.execute("DELETE FROM merged_group_country_split")
    geo.executemany(
        "INSERT OR REPLACE INTO merged_key_classes (specieskey, class_id) VALUES (?,?)",
        roster_rows,
    )
    geo.executemany(
        "INSERT OR REPLACE INTO merged_group_country_split "
        "(specieskey, country, class_id, n_usage) VALUES (?,?,?,?)",
        split_rows,
    )
    geo.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
        (
            "sibling_split",
            json.dumps(
                {
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "groups": len(roster),
                    "classes": len(roster_rows),
                    "labeled_classes": labeled,
                    "split_rows": len(split_rows),
                },
                ensure_ascii=False,
            ),
        ),
    )
    geo.commit()
    geo.close()

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"[split] 完成 / done：roster {len(roster_rows)} 行，"
          f"split {len(split_rows):,} 行（缓存 {CACHE_PATH}）")


if __name__ == "__main__":
    sys.exit(main())
