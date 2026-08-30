# -*- coding: utf-8 -*-
"""
构建中国国别稀有度表 gbif_rarity_by_country / Build the country-scoped rarity
table gbif_rarity_by_country (countrycode='CN') in bird_reference.sqlite.

分数语义 / Score semantics:
    对模型库 10,963 个鸟种中「中国境内有 GBIF 开放许可记录」的物种，按
    GBIF country=CN + CC0/CC-BY 记录数做 log 归一化，得到 0-100 分
    （记录越少分越高）。运行时 bird_database_manager.get_gbif_rarity_by_class_id()
    对 GPS 在中国的照片优先取本表分数，无行的物种自动回退全球分。

为什么计数直接走 API 而不是 geo_distribution.db 的 country_species：
    country_species 是「1° 网格中心反解国家再汇总」的几何近似，跨边境格网
    会被整格涂抹、海岸/海洋物种与 GBIF country 语义偏差大。2026-08-29 实测
    6 个样本中本地计数系统性偏高 10%-40%（白玄鸥 Sterna sumatrana 偏高 10
    倍），超过校验阈值，故本脚本只把 country_species 用作「中国有记录」的
    候选清单，计数全部通过 GBIF Occurrence Search API 以与全球表完全相同的
    许可口径重新获取（约 1,500 次请求，8 并发约 10 分钟）。
    China candidate list comes from country_species (grid rollup), but every
    count is re-fetched from the GBIF API with the exact license filter used
    by the global table, because measured grid-rollup bias was +10..40%
    (10x for coastal species).

归一化 / Normalization:
    score = 100 × (1 − (log10(n+1) − log10(nmin+1)) / (log10(nmax+1) − log10(nmin+1)))
    在「中国有记录」子集内归一化；不做 IUCN 下限（与「保护等级独立标注、
    不并入分数」的决策一致，中国分保持纯观察密度口径）。
    Normalized within the CN-present subset only; no IUCN floor is applied —
    the national protection level is a separate label by design.

用法 / Usage:
    .venv/Scripts/python scripts_dev/build_china_rarity.py
    .venv/Scripts/python scripts_dev/build_china_rarity.py --resume   # 用计数缓存续跑
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DB = os.path.join(PROJ, "birdid", "data", "bird_reference.sqlite")
GEO_DB = os.path.join(PROJ, "birdid", "data", "geo_distribution.db")
CACHE_PATH = os.path.join(PROJ, "scripts_dev", "data_sources", "cn_occurrence_counts.json")

LICENSES = ("CC0_1_0", "CC_BY_4_0")
WORKERS = 8
BATCH = 200
MAX_RETRY = 5
COUNTRY = "CN"
SOURCE_NOTE = "GBIF.org Occurrence Search API country=CN, CC0+CC-BY-4.0 only"


def fetch_cn_count(specieskey: int) -> Optional[int]:
    """
    查询单个物种在中国的 CC0+CC-BY 记录数，带 429/网络错误退避重试。

    Fetch the CC0+CC-BY occurrence count for one species in China (country=CN),
    with backoff retries on HTTP 429 and transient network errors.

    参数 / Parameters:
        specieskey (int): GBIF speciesKey / GBIF speciesKey of the species.

    返回 / Returns:
        Optional[int]: 记录数；重试耗尽仍失败时返回 None / The count, or
            None when all retries are exhausted.
    """
    params = [
        ("taxonKey", str(specieskey)),
        ("country", COUNTRY),
        ("limit", "0"),
    ]
    for lic in LICENSES:
        params.append(("license", lic))
    url = "https://api.gbif.org/v1/occurrence/search?" + urllib.parse.urlencode(params)

    for attempt in range(MAX_RETRY):
        req = urllib.request.Request(url, headers={"User-Agent": "SuperPicky-build/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            return int(data.get("count", 0))
        except urllib.error.HTTPError as e:
            if e.code != 429 and attempt == MAX_RETRY - 1:
                print(f"[build] speciesKey={specieskey} HTTP {e.code}，放弃")
                return None
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == MAX_RETRY - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def load_candidates() -> List[Tuple[int, int, str]]:
    """
    从本地库取「中国有记录」的候选物种清单及其 specieskey。

    Load the CN-present candidate species from the local reference DB
    (candidate list from geo_distribution.db country_species, keys from
    gbif_rarity_100).

    返回 / Returns:
        list[tuple[int, int, str]]: [(model_class_id, specieskey, scientific_name), ...]

    异常 / Exceptions:
        FileNotFoundError: geo_distribution.db 缺失时抛出 / Raised when the
            geo database is missing.
    """
    if not os.path.exists(GEO_DB):
        raise FileNotFoundError(f"地理分布库缺失 / geo db missing: {GEO_DB}")

    ref = sqlite3.connect(REF_DB)
    geo = sqlite3.connect(GEO_DB)
    try:
        key_of: Dict[int, Tuple[int, str]] = {}
        for cid, skey, name in ref.execute(
            "SELECT model_class_id, specieskey, scientific_name "
            "FROM gbif_rarity_100 WHERE specieskey IS NOT NULL"
        ):
            key_of[int(cid)] = (int(skey), str(name))
        candidates: List[Tuple[int, int, str]] = []
        for (cid,) in geo.execute(
            "SELECT class_id FROM country_species WHERE country=? ORDER BY class_id",
            (COUNTRY,),
        ):
            entry = key_of.get(int(cid))
            if entry is not None:
                candidates.append((int(cid), entry[0], entry[1]))
        return candidates
    finally:
        ref.close()
        geo.close()


def harvest_counts(
    candidates: List[Tuple[int, int, str]], resume: bool
) -> Dict[int, int]:
    """
    并发拉取所有候选物种的中国记录数，分批写入 JSON 缓存。

    Fetch CN occurrence counts for all candidate species concurrently,
    flushing results into the JSON cache every batch so a rerun with
    --resume skips finished keys.

    参数 / Parameters:
        candidates (list): load_candidates() 的候选清单 / Candidate species.
        resume (bool): True 时先加载缓存、只补缺失项 / Load cache first and
            only fetch missing keys when True.

    返回 / Returns:
        dict[int, int]: {specieskey: count}；获取失败的 key 不在返回值中 /
            {specieskey: count}; keys whose fetch failed are absent.

    异常 / Exceptions:
        RuntimeError: 失败条目超过允许上限时抛出 / Raised when too many
            fetches fail.
    """
    cache: Dict[str, int] = {}
    if resume and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[build] 缓存命中 / cached counts: {len(cache)}")

    todo = [(cid, skey, name) for cid, skey, name in candidates
            if str(skey) not in cache]
    print(f"[build] 待查询 / to fetch: {len(todo)}")

    failed: List[Tuple[int, str]] = []
    t0 = time.time()
    processed = 0
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda c: (c, fetch_cn_count(c[1])), chunk))

        for (cid, skey, name), count in results:
            if count is None:
                failed.append((cid, name))
                continue
            cache[str(skey)] = count

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        processed += len(chunk)

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        remain = (len(todo) - processed) / rate / 60 if rate > 0 else 0
        print(
            f"[build] {processed}/{len(todo)}  {rate:.1f} 次/秒  剩余约 {remain:.0f} 分钟",
            flush=True,
        )

    if len(failed) > len(candidates) * 0.05:
        raise RuntimeError(
            f"失败条目过多 / too many fetch failures: {len(failed)}，"
            f"示例 / e.g. {failed[:5]}（--resume 续跑）"
        )
    if failed:
        print(f"[build] ⚠️ {len(failed)} 个物种查询失败，本批跳过 / skipped: "
              f"{[n for _, n in failed[:10]]}")
    return {int(k): v for k, v in cache.items()}


def normalize(counts: Dict[int, int]) -> Dict[int, float]:
    """
    在「中国有记录」子集内做 log 归一化（记录越少分越高）。

    Log-normalize counts within the CN-present subset (fewer records → higher
    score), mirroring the global table's formula but with CN-only bounds.

    参数 / Parameters:
        counts (dict): {specieskey: count}，只保留 count >= 1 的物种 /
            species counts; only count >= 1 entries participate.

    返回 / Returns:
        dict[int, float]: {specieskey: score 0-100}
    """
    positive = {k: v for k, v in counts.items() if v >= 1}
    lo = math.log10(min(positive.values()) + 1)
    hi = math.log10(max(positive.values()) + 1)
    span = hi - lo
    scores: Dict[int, float] = {}
    for key, n in positive.items():
        if span <= 0:
            scores[key] = 50.0
            continue
        scores[key] = round(100.0 * (1.0 - (math.log10(n + 1) - lo) / span), 2)
    return scores


def write_table(scores: Dict[int, float], counts: Dict[int, int]) -> None:
    """
    DROP+重建 bird_reference.sqlite 的 gbif_rarity_by_country 表并写入 CN 行。

    Recreate gbif_rarity_by_country in bird_reference.sqlite and insert the
    CN rows. Column names must stay compatible with the runtime query in
    bird_database_manager.get_gbif_rarity_by_class_id().

    参数 / Parameters:
        scores (dict): {specieskey: score} / normalized scores.
        counts (dict): {specieskey: count} / raw counts for provenance.
    """
    ref = sqlite3.connect(REF_DB)
    try:
        ref.execute("DROP TABLE IF EXISTS gbif_rarity_by_country")
        ref.execute(
            """
            CREATE TABLE gbif_rarity_by_country (
                model_class_id  INTEGER NOT NULL,
                countrycode     TEXT NOT NULL,
                cc_cn_count     INTEGER,
                gbif_rarity_100 REAL NOT NULL,
                snapshot_date   TEXT,
                source          TEXT,
                PRIMARY KEY (model_class_id, countrycode)
            ) WITHOUT ROWID
            """
        )
        rows: List[Tuple[int, str, int, float, str, str]] = []
        for cid, skey, _name in load_candidates():
            score = scores.get(skey)
            if score is None:
                continue  # count=0 或查询失败 → 不写行，运行时回退全球分
            rows.append(
                (cid, COUNTRY, counts[skey], score, date.today().isoformat(), SOURCE_NOTE)
            )
        ref.executemany(
            "INSERT INTO gbif_rarity_by_country "
            "(model_class_id, countrycode, cc_cn_count, gbif_rarity_100, "
            " snapshot_date, source) VALUES (?,?,?,?,?,?)",
            rows,
        )
        ref.commit()
        print(f"[build] gbif_rarity_by_country 写入 {len(rows)} 行")
    finally:
        ref.close()


def report(scores: Dict[int, float], counts: Dict[int, int]) -> None:
    """
    打印构建审计摘要：物种数、tier 分布、旗舰种对照。

    Print a build audit summary: CN species count, tier distribution and
    flagship-species comparison against the global table.

    参数 / Parameters:
        scores (dict): {specieskey: score} / normalized scores.
        counts (dict): {specieskey: count} / raw counts.
    """
    sys.path.insert(0, PROJ)
    from core.rarity_tier import TIER_NAMES_ZH, gbif_score_to_tier

    ref = sqlite3.connect(REF_DB)
    try:
        rows = list(
            ref.execute(
                "SELECT model_class_id, scientific_name, gbif_rarity_100 "
                "FROM gbif_rarity_100"
            )
        )
        names = {
            cid: (name, skey)
            for cid, name, skey in ref.execute(
                "SELECT model_class_id, scientific_name, specieskey FROM gbif_rarity_100"
            )
        }
    finally:
        ref.close()

    tier_hist = [0] * 5
    for s in scores.values():
        tier_hist[gbif_score_to_tier(s) or 0] += 1
    print(f"[audit] 中国有记录物种 / CN-present species: {len(scores)}")
    print("[audit] tier 分布 / distribution: " + " | ".join(
        f"{TIER_NAMES_ZH[i]}: {n}" for i, n in enumerate(tier_hist)
    ))

    print("[audit] 旗舰种对照 / flagship check (global → CN):")
    for cid, label in [(1041, "金雕 Golden Eagle"), (7201, "白头鹎 Light-vented Bulbul"),
                       (9380, "麻雀 Eurasian Tree Sparrow")]:
        name, skey = names[cid]
        gs = next((r[2] for r in rows if r[0] == cid), None)
        cs = scores.get(skey)
        print(f"[audit]   {label:28s} global={gs} → CN={cs} (count={counts.get(skey)})")

    golden = names[1041][1]
    cn_score = scores.get(golden)
    assert cn_score is not None and cn_score > 25.0, (
        f"金雕 CN 分数异常 / unexpected Golden Eagle CN score: {cn_score}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build gbif_rarity_by_country (CN) in bird_reference.sqlite"
    )
    p.add_argument("--resume", action="store_true",
                   help="使用 cn_occurrence_counts.json 缓存续跑 / resume from cache")
    a = p.parse_args()

    candidates = load_candidates()
    print(f"[build] 中国候选物种 / CN candidates: {len(candidates)}")
    counts = harvest_counts(candidates, a.resume)
    scores = normalize(counts)
    write_table(scores, counts)
    report(scores, counts)
    print("[build] 完成 / done")


if __name__ == "__main__":
    main()
