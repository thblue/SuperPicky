# -*- coding: utf-8 -*-
"""
从 GBIF Occurrence API 生成地理分布库 / Build the geo-distribution DB from the GBIF API.

对每个 1°网格调用 GBIF 的 speciesKey facet，拿到「该格每个物种的观察记录数」，
映射到 OSEA class_id 后写入 SQLite，再按国家汇总。服务端完成聚合，无需下载原始记录。

For each 1-degree cell, call the GBIF speciesKey facet to get per-species
occurrence counts, map them to OSEA class ids, write them to SQLite, and roll up
by country. The aggregation happens server-side; no raw records are downloaded.

支持断点续传：已处理的网格记录在 _build_progress 表，重跑时自动跳过。
Resumable: processed cells are tracked in _build_progress and skipped on re-run.

用法 / Usage:
    .venv/bin/python scripts_dev/build_geo_distribution.py --tier1 cumulative:0.999
    .venv/bin/python scripts_dev/build_geo_distribution.py --resume   # 续跑
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJ, "birdid", "data", "geo_distribution.db")
AVES_CLASS_KEY = 212
LICENSES = ("CC0_1_0", "CC_BY_4_0")
WORKERS = 8
BATCH = 200
MAX_RETRY = 5


def cell_id_of(lat_bin: int, lon_bin: int) -> int:
    """
    网格编号 / Encode a 1-degree cell as a single integer.

    参数 / Parameters:
        lat_bin (int): floor(纬度)，-90..89 / floor(latitude).
        lon_bin (int): floor(经度)，-180..179 / floor(longitude).

    返回 / Returns:
        int: 0..64799 的网格编号 / Cell id in 0..64799.
    """
    lat_bin = max(-90, min(89, lat_bin))
    lon_bin = max(-180, min(179, lon_bin))
    return (lat_bin + 90) * 360 + (lon_bin + 180)


def fetch_cell(lat_bin: int, lon_bin: int) -> Optional[Dict[int, int]]:
    """
    拉取单个网格内的鸟种及观察记录数，带 429/网络错误退避重试。

    Fetch per-species occurrence counts for one cell, with backoff retries on
    HTTP 429 and transient network errors.

    参数 / Parameters:
        lat_bin (int): 网格南边界纬度 / Southern latitude of the cell.
        lon_bin (int): 网格西边界经度 / Western longitude of the cell.

    返回 / Returns:
        Optional[dict]: {gbif_species_key: count}；重试耗尽仍失败时返回 None /
            {gbif_species_key: count}, or None when all retries are exhausted.
    """
    params = [
        ("classKey", str(AVES_CLASS_KEY)),
        ("decimalLatitude", f"{lat_bin},{lat_bin + 1}"),
        ("decimalLongitude", f"{lon_bin},{lon_bin + 1}"),
        ("hasCoordinate", "true"),
        ("hasGeospatialIssue", "false"),
        ("facet", "speciesKey"),
        ("facetLimit", "1200"),
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
            out: Dict[int, int] = {}
            for f in data.get("facets", []):
                if f.get("field") == "SPECIES_KEY":
                    for c in f.get("counts", []):
                        out[int(c["name"])] = int(c["count"])
            return out
        except urllib.error.HTTPError as e:
            if e.code != 429:
                if attempt == MAX_RETRY - 1:
                    print(f"[build] 网格 ({lat_bin},{lon_bin}) HTTP {e.code}，放弃")
                    return None
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == MAX_RETRY - 1:
                return None
            time.sleep(2 ** attempt)
    return None


OVERRIDES_PATH = os.path.join(
    PROJ, "scripts_dev", "data_sources", "specieskey_overrides.json"
)


def apply_key_overrides(cls_to_key: Dict[int, int]) -> Dict[int, int]:
    """
    应用人工核定的 specieskey 覆盖表 / Apply the audited specieskey overrides.

    gbif_rarity_100 的学名匹配对部分近年拆分种落到了非种级/存疑用法上
    （如 Tachyspiza 属级 DOUBTFUL key、鸟纲 key），speciesKey facet 永远
    不返回这类 key，对应类别在全球候选集中整组清零。覆盖表由
    fix_geo_key_collisions.py 经 GBIF match API 逐类核定生成（含学名、
    解析状态与时间戳，便于人工复核）。

    The gbif_rarity_100 name matcher resolved some recent splits to
    non-species / doubtful usages (e.g. the DOUBTFUL genus key for
    Tachyspiza, or the class key for Aves). The speciesKey facet never
    returns such keys, starving those classes of all occurrence data. The
    override file is produced by fix_geo_key_collisions.py via the GBIF
    match API, with names / status / timestamps for auditing.

    参数 / Parameters:
        cls_to_key (dict): {model_class_id: specieskey}，待修正的映射 /
            The class-to-key mapping to correct.

    返回 / Returns:
        dict[int, int]: 应用覆盖后的映射 / The mapping after overrides.
    """
    if not os.path.exists(OVERRIDES_PATH):
        return cls_to_key
    with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    changed = 0
    for row in rows:
        try:
            cid = int(row["model_class_id"])
            new_key = int(row["specieskey"])
        except (KeyError, TypeError, ValueError):
            continue
        if cls_to_key.get(cid) != new_key:
            cls_to_key[cid] = new_key
            changed += 1
    if changed:
        print(f"[build] 应用 specieskey 覆盖 / overrides applied: {changed}")
    return cls_to_key


def load_key_to_class() -> Dict[int, Set[int]]:
    """
    GBIF specieskey → model_class_id 集合（一对多，覆盖 10963/10964）。

    GBIF 骨干未跟上的分类学拆分会让多个模型类共用同一个 speciesKey
    （如黑水鸡/普通水鸡共用 5228199）。旧实现用 {key: class} 字典装载，
    同 key 后写覆盖先写，导致 191 个类在全球候选集中被静默清零
    （黑水鸡在中国候选集里完全消失，美洲普通水鸡反成「中国常见种」）。
    现改为一对多：同一 key 的观察计数全量累加到组内每个类，由下游
    （模型排序 + 国家级姊妹拆分表 merged_group_country_split）自行甄别。

    One-to-many mapping from GBIF speciesKey to model class ids. Taxonomic
    splits the GBIF backbone has not adopted make several model classes share
    one speciesKey (e.g. Common Moorhen/Common Gallinule both map to 5228199).
    The old {key: class} dict silently starved 191 classes of all occurrence
    data; counts now accumulate in full onto every class of a shared group,
    leaving disambiguation to the downstream country-level sibling split.

    返回 / Returns:
        dict[int, set[int]]: {specieskey: {model_class_id, ...}}
    """
    db = sqlite3.connect(os.path.join(PROJ, "birdid", "data", "bird_reference.sqlite"))
    cls_to_key: Dict[int, int] = {}
    for cid, skey in db.execute(
        "SELECT model_class_id, specieskey FROM gbif_rarity_100 WHERE specieskey IS NOT NULL"
    ):
        try:
            cls_to_key[int(cid)] = int(skey)
        except (TypeError, ValueError):
            continue
    db.close()
    cls_to_key = apply_key_overrides(cls_to_key)

    m: Dict[int, Set[int]] = {}
    for cid, skey in cls_to_key.items():
        m.setdefault(skey, set()).add(cid)
    shared = sum(1 for v in m.values() if len(v) > 1)
    if shared:
        print(
            f"[build] 共用 specieskey 的姊妹组 / shared-key groups: {shared}"
            "（计数将全量累加到组内每类 / counts accrue to every sibling in full）"
        )
    return m


def land_cells() -> List[Tuple[int, int]]:
    """
    枚举待扫描的陆地网格 / Enumerate the land cells to scan.

    从 `birdid/data/land_cells.json` 读取 16,882 个网格编号并还原为
    (lat_bin, lon_bin)。该列表最初由已退役的 avonet.db 分布网格一次性导出，
    此后本脚本不再依赖 avonet.db——它已随 GBIF 迁移被删除。

    Reads the 16,882 cell ids from `birdid/data/land_cells.json` and decodes
    them back to (lat_bin, lon_bin). The list was exported once from the retired
    avonet.db distribution grid; this script no longer depends on avonet.db,
    which was removed as part of the GBIF migration.

    返回 / Returns:
        list[tuple[int, int]]: [(lat_bin, lon_bin), ...]，按编号升序 /
            sorted by cell id.

    异常 / Exceptions:
        FileNotFoundError: 网格清单缺失时抛出 / Raised when the list is missing.
    """
    path = os.path.join(PROJ, "birdid", "data", "land_cells.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"网格清单缺失 / cell list missing: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cells: List[Tuple[int, int]] = []
    for cid in data.get("cell_ids", []):
        lat_bin, lon_bin = divmod(int(cid), 360)
        cells.append((lat_bin - 90, lon_bin - 180))
    return sorted(cells)


def init_db(path: str, resume: bool) -> sqlite3.Connection:
    """
    打开（必要时重建）目标库并确保表结构存在。

    Open (recreating when not resuming) the target database and ensure the schema.

    参数 / Parameters:
        path (str): 数据库路径 / Database path.
        resume (bool): True 时保留已有数据续跑 / Keep existing data when True.

    返回 / Returns:
        sqlite3.Connection: 已就绪的连接 / A ready connection.
    """
    if not resume and os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript(
        """
        -- WITHOUT ROWID + 复合主键：主键 B 树即是表本身，省掉隐藏 rowid 与
        -- 一份独立的 cell_id 索引；同时主键约束天然防止同一网格被重复写入。
        -- 普通 rowid 表实测 86.3 MB，改此结构后显著缩小。
        -- WITHOUT ROWID with a composite primary key: the key B-tree *is* the
        -- table, dropping both the hidden rowid and a separate cell_id index,
        -- and the constraint prevents duplicate rows for a cell outright.
        -- A plain rowid table measured 86.3 MB for the same data.
        CREATE TABLE IF NOT EXISTS cell_species (
            cell_id  INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            n        INTEGER NOT NULL,
            PRIMARY KEY (cell_id, class_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS country_species (
            country  TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            n        INTEGER NOT NULL,
            PRIMARY KEY (country, class_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS _build_progress (cell_id INTEGER PRIMARY KEY);
        """
    )
    db.commit()
    return db


def harvest(
    db: sqlite3.Connection,
    key2cls: Dict[int, Set[int]],
    cells: List[Tuple[int, int]],
) -> int:
    """
    并发拉取所有网格并分批写入，跳过已完成的网格。

    Fetch all cells concurrently and write in batches, skipping completed cells.

    参数 / Parameters:
        db (sqlite3.Connection): 目标库连接 / Target database connection.
        key2cls (dict): specieskey → class_id 集合（一对多，见 load_key_to_class）
            / speciesKey to class-id sets (one-to-many, see load_key_to_class).
        cells (list): 待扫描网格 / Cells to scan.

    返回 / Returns:
        int: 本次新处理的网格数 / Number of cells processed in this run.
    """
    done: Set[int] = {r[0] for r in db.execute("SELECT cell_id FROM _build_progress")}
    todo = [c for c in cells if cell_id_of(c[0], c[1]) not in done]
    print(f"[build] 待扫描 / to scan: {len(todo)}（已完成 / done: {len(done)}）")

    t0 = time.time()
    processed = 0
    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda c: (c, fetch_cell(c[0], c[1])), chunk))

        rows: List[Tuple[int, int, int]] = []
        progress: List[Tuple[int]] = []
        for (lat_bin, lon_bin), counts in results:
            if counts is None:
                continue                      # 失败的格不标记完成，留待续跑重试
            cid = cell_id_of(lat_bin, lon_bin)
            acc: Dict[int, int] = {}
            for skey, n in counts.items():
                # 一对多：合并 key 的计数全量累加到组内每个类（GBIF 在种级
                # 无法区分姊妹种，候选集应全保留，甄别交给下游）
                # One-to-many: the merged key's count accrues in full to every
                # sibling class; disambiguation is left to downstream stages.
                for cls in key2cls.get(skey, ()):
                    acc[cls] = acc.get(cls, 0) + n
            rows.extend((cid, cls, n) for cls, n in acc.items())
            progress.append((cid,))

        db.executemany(
            "INSERT OR REPLACE INTO cell_species (cell_id, class_id, n) VALUES (?,?,?)", rows
        )
        db.executemany("INSERT OR IGNORE INTO _build_progress (cell_id) VALUES (?)", progress)
        db.commit()
        processed += len(progress)

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        remain = (len(todo) - processed) / rate / 60 if rate > 0 else 0
        print(f"[build] {processed}/{len(todo)} 格  {rate:.1f} 格/秒  剩余约 {remain:.0f} 分钟", flush=True)

    return processed


def rollup_countries(db: sqlite3.Connection) -> int:
    """
    按国家汇总网格数据 / Roll up cell data by country.

    每个网格中心用 reverse_geocoder 反查 ISO 国家代码后聚合。

    Each cell centre is reverse-geocoded to an ISO country code, then aggregated.

    参数 / Parameters:
        db (sqlite3.Connection): 目标库连接 / Target database connection.

    返回 / Returns:
        int: 写入的国家级行数 / Number of country-level rows written.
    """
    import reverse_geocoder as rg

    cell_ids = [r[0] for r in db.execute("SELECT DISTINCT cell_id FROM cell_species")]
    if not cell_ids:
        return 0
    coords = []
    for cid in cell_ids:
        lat_bin, lon_bin = divmod(cid, 360)
        coords.append((lat_bin - 90 + 0.5, lon_bin - 180 + 0.5))
    print(f"[build] 反查国家 / reverse-geocoding {len(coords)} cells ...", flush=True)
    results = rg.search(coords, mode=2, verbose=False)
    cell_country = {
        cid: str(r.get("cc", "")).upper()
        for cid, r in zip(cell_ids, results)
        if r.get("cc")
    }

    acc: Dict[Tuple[str, int], int] = {}
    for cid, cls, n in db.execute("SELECT cell_id, class_id, n FROM cell_species"):
        cc = cell_country.get(cid)
        if not cc:
            continue
        acc[(cc, cls)] = acc.get((cc, cls), 0) + n

    db.execute("DELETE FROM country_species")
    db.executemany(
        "INSERT OR REPLACE INTO country_species (country, class_id, n) VALUES (?,?,?)",
        [(cc, cls, n) for (cc, cls), n in acc.items()],
    )
    db.commit()
    return len(acc)


def finalize(db: sqlite3.Connection, tier1: str) -> None:
    """
    建索引、写 meta、清理进度表并压实。

    Create indexes, write meta, drop the progress table, and vacuum.

    参数 / Parameters:
        db (sqlite3.Connection): 目标库连接 / Target database connection.
        tier1 (str): Task 1 标定的 L1 方案 / The calibrated L1 strategy.
    """
    # 两表均为 WITHOUT ROWID + 复合主键，主键 B 树已按 cell_id / country 前缀
    # 排序，范围查询直接走主键，无需额外索引。
    # Both tables are WITHOUT ROWID with composite keys, so the primary-key
    # B-tree already orders by cell_id / country; no extra index is needed.
    db.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
        [
            ("snapshot_date", date.today().isoformat()),
            ("gbif_doi", "GBIF.org Occurrence Search API (facet aggregation)"),
            ("license", "CC0-1.0 / CC-BY-4.0"),
            ("attribution", "GBIF.org occurrence data; CC0 and CC-BY-4.0 records only"),
            ("builder_version", "2"),
            ("tier1_threshold", tier1),
        ],
    )
    db.execute("DROP TABLE IF EXISTS _build_progress")
    db.commit()
    db.execute("VACUUM")


def main() -> None:
    p = argparse.ArgumentParser(description="Build geo_distribution.db from the GBIF API")
    p.add_argument("--tier1", default="cumulative:0.999",
                   help="Task 1 标定的 L1 方案 / calibrated L1 strategy")
    p.add_argument("--resume", action="store_true",
                   help="保留已有数据续跑 / keep existing data and resume")
    a = p.parse_args()

    key2cls = load_key_to_class()
    print(f"[build] specieskey→class_id 映射: {len(key2cls)}")
    cells = land_cells()
    print(f"[build] 陆地网格 / land cells: {len(cells)}")

    db = init_db(OUT_PATH, a.resume)
    processed = harvest(db, key2cls, cells)
    remaining = len(cells) - db.execute("SELECT COUNT(*) FROM _build_progress").fetchone()[0]
    if remaining > 0:
        print(f"[build] ⚠️ 仍有 {remaining} 格未完成，用 --resume 续跑")
        db.close()
        sys.exit(1)

    n_country = rollup_countries(db)
    finalize(db, a.tier1)
    db.close()
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"[build] 完成 / done: 本次 {processed} 格，国家行 {n_country}，{size_mb:.1f} MB")


if __name__ == "__main__":
    main()
