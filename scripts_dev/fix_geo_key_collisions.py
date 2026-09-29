# -*- coding: utf-8 -*-
"""
修复 geo_distribution.db 的 specieskey 碰撞伤损 / Repair specieskey-collision damage.

背景 / Background:
  bird_reference.sqlite 的 gbif_rarity_100 表里 172 个 GBIF speciesKey 被多个
  模型类共享（GBIF 骨干未跟上的分类学拆分）。旧版 build_geo_distribution.py
  用 {key: class} 字典装载映射，同 key 后写覆盖先写，造成两类伤损：

  1) 154 组「有胜者」：胜者独吞合并 key 的全部观察计数，191 个姊妹类在全球
     候选集中被静默清零（黑水鸡在中国名下零记录，美洲普通水鸡反成中国常见种）。
     修法：把胜者的 cell_species / country_species 行复制给组内每个被清零类
     （GBIF 在种级分不开的，候选集就全保留，甄别交给下游模型排序与国家级
     姊妹拆分表）。
  2) 18 组「整组全灭」：学名匹配落到了非种级/存疑用法（如 Tachyspiza 属级
     DOUBNTFUL key、Anarhynchus 属级 key），speciesKey facet 永远不返回这类
     key。骨干库（backbone d7dddbf4）往往还把物种挂在旧属组合下（褐耳鹰在
     骨干里仍是 Accipiter badius）。修法：两级解析——先学名精确匹配，再
     「科范围 + 种加词」在骨干库内检索（词干归一化处理拉丁性别词尾
     badia/badius）；解析到种级 key 后按国家 facet 拉记录写入 country_species，
     并把 key 记入 specieskey_overrides.json 供下次全量重建应用。解析失败的
     类保持现状并报告。

  bird_reference.sqlite 本身只读不写（跨仓只读契约资产）。

  gbif_rarity_100 shares 172 GBIF speciesKeys across multiple model classes.
  The old dict-based loader starved 191 sibling classes (154 groups) of all
  occurrence data, and left 18 further groups dead because their keys fell
  back to genus-rank usages while the backbone still files the species under
  legacy genera. This script copies winner rows to starved siblings, resolves
  dead-group names to species-rank backbone keys (exact match first, then a
  family-scoped epithet search with Latin gender-stem normalization), writes
  country rows, and records overrides for the next full rebuild. The
  reference DB is opened read-only.

用法 / Usage:
  python scripts_dev/fix_geo_key_collisions.py             # dry-run，只打印
  python scripts_dev/fix_geo_key_collisions.py --execute   # 备份后写库
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_DB = os.path.join(PROJ, "birdid", "data", "geo_distribution.db")
REF_DB = os.path.join(PROJ, "birdid", "data", "bird_reference.sqlite")
OVERRIDES_PATH = os.path.join(
    PROJ, "scripts_dev", "data_sources", "specieskey_overrides.json"
)
GBIF = "https://api.gbif.org/v1"
BACKBONE = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"
LICENSES = ("CC0_1_0", "CC_BY_4_0")
WORKERS = 4
MAX_RETRY = 5


def _gbif_get(path: str, params: Optional[List[Tuple[str, str]]] = None) -> Optional[dict]:
    """
    带 429/网络错误退避的 GBIF GET / GBIF GET with 429/network backoff.

    参数 / Parameters:
        path (str): API 路径（如 species/match）/ API path.
        params (list): 查询参数 / Query parameters.

    返回 / Returns:
        Optional[dict]: 解析后的 JSON；重试耗尽仍失败时 None / Parsed JSON
            or None when all retries fail.
    """
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"{GBIF}/{path}{qs}"
    for attempt in range(MAX_RETRY):
        req = urllib.request.Request(url, headers={"User-Agent": "SuperPicky-fix/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception:
            if attempt == MAX_RETRY - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def _clean_name(name: str) -> str:
    """
    去掉灭绝标记等非字母字符 / Strip extinction daggers and noise.

    参数 / Parameters:
        name (str): 库里的学名（可能带 †）/ Scientific name (may carry †).

    返回 / Returns:
        str: 干净的学名 / The cleaned name.
    """
    return " ".join(t for t in name.replace("†", " ").split() if t.isalpha())


def _epithet_stem(name: str) -> str:
    """
    取种加词并归一化拉丁性别词尾 / Extract the specific epithet stem.

    骨干库常保留旧拼写的性别词尾（AviList 的 badia 对骨干的 badius、
    wilsonia 对 wilsonius）。词尾长度不同：a/e 砍 1 字符，us/um/is 砍
    2 字符——只砍 1 字符会把 badius 砍成 badiu、与 badia 的 badi 永不
    相等（褐耳鹰 Tachyspiza badia 曾因此漏配）。两侧统一用本函数即可
    对齐；误配由「科内唯一接受种 + key 占用守卫 + 俗名交叉验证」兜底。

    The backbone often keeps legacy gender endings (badia vs badius,
    wilsonia vs wilsonius). Endings differ in length: drop one char for
    a/e and two for us/um/is -- dropping a single char turned badius into
    badiu, which never equals badia's badi (this once hid Tachyspiza
    badia). Both sides use this function so they align; false hits are
    bounded by the unique-accepted guard, the key-claim guard and the
    vernacular cross-check.

    参数 / Parameters:
        name (str): 学名 / Scientific name.

    返回 / Returns:
        str: 词干（短加词返回原样）/ The stem (short epithets as-is).
    """
    tokens = _clean_name(name).split()
    if len(tokens) < 2:
        return ""
    epi = tokens[-1].lower()
    if len(epi) > 4 and epi.endswith(("us", "um", "is")):
        return epi[:-2]
    if len(epi) > 4 and epi.endswith(("a", "e")):
        return epi[:-1]
    return epi


class Resolver:
    """
    死组学名 → 骨干种级 key 解析器 / Dead-group name to backbone species-key resolver.

    两级策略 / Two-stage strategy:
      1) species/match 学名精确匹配（EXACT，或 FUZZY 且 confidence ≥ 95），
         只接受 rank=SPECIES 的 ACCEPTED/SYNONYM 用法；
      2) 骨干库「科范围 + 种加词」检索：科的 usageKey 经 match 解析并缓存，
         种加词按词干归一化在科内比对，要求命中唯一的接受种。

    The match API accepts EXACT (or FUZZY with confidence >= 95) species-rank
    usages only; otherwise a backbone family-scoped search matches the
    stem-normalized epithet and requires a unique accepted species.
    """

    def __init__(self) -> None:
        self._family_keys: Dict[str, Optional[int]] = {}

    def _from_match(self, name: str) -> Optional[Dict]:
        data = _gbif_get(
            "species/match", [("name", name), ("verbose", "true")]
        )
        if not data:
            return None
        for c in [data] + list(data.get("alternatives") or []):
            if c.get("rank") != "SPECIES":
                continue
            if c.get("matchType") == "EXACT" or (
                c.get("matchType") == "FUZZY" and (c.get("confidence") or 0) >= 95
            ):
                status = str(c.get("status") or "")
                if status == "SYNONYM" and c.get("acceptedUsageKey"):
                    return {
                        "key": int(c["acceptedUsageKey"]),
                        "status": f"SYNONYM→{c.get('accepted')}",
                        "via": f"match:{c.get('scientificName')}",
                    }
                if status == "ACCEPTED" and c.get("usageKey"):
                    return {
                        "key": int(c["usageKey"]),
                        "status": "ACCEPTED",
                        "via": f"match:{c.get('scientificName')}",
                    }
        return None

    def _family_key(self, family: Optional[str]) -> Optional[int]:
        if not family:
            return None
        if family not in self._family_keys:
            data = _gbif_get("species/match", [("name", family)])
            ok = (
                data
                and data.get("rank") == "FAMILY"
                and data.get("usageKey")
            )
            self._family_keys[family] = int(data["usageKey"]) if ok else None
        return self._family_keys[family]

    def _from_family_epithet(
        self, sci: str, family: Optional[str]
    ) -> Optional[Dict]:
        fam_key = self._family_key(family)
        stem = _epithet_stem(sci)
        if not fam_key or not stem:
            return None
        # species/search 的 q 只支持整词（前缀/通配符查不到），且按词形分词：
        # q=fasciata 查不到 fasciatus。骨干库的加词还可能保留另一套拉丁
        # 性别词尾（badia vs badius、wilsonia vs wilsonius）。因此把「原词 +
        # 各词尾变体」的命中**聚合**后统一判断——只有聚合后恰好一个接受种
        # 才采纳（避免 A属 fasciata 与 B属 fasciatus 各自整词可查、单变体
        # 假唯一导致错配到隔壁属的同加词种）。
        # species/search q matches whole words only (no prefix/wildcard) and
        # tokenizes by word form: q=fasciata never sees fasciatus. The
        # backbone may also keep the epithet under a different gender ending.
        # Aggregate hits across the original epithet and every ending variant,
        # and accept only when the combined set holds exactly one accepted
        # species -- a single variant can look unique while a sibling genus
        # carries the same epithet under another ending.
        epithet = _clean_name(sci).split()[-1].lower()
        variants = [epithet]
        for suffix in ("a", "e", "us", "um", "is"):
            cand = stem + suffix
            if cand not in variants:
                variants.append(cand)
        hits: Dict[int, str] = {}
        for q in variants:
            data = _gbif_get(
                "species/search",
                [
                    ("datasetKey", BACKBONE),
                    ("highertaxonKey", str(fam_key)),
                    ("rank", "SPECIES"),
                    ("q", q),
                    ("limit", "50"),
                ],
            )
            if data is None:
                # 变体查询失败必须放弃而非跳过：跳过会让「多义词干」退化成
                # 「假唯一」，曾因此把褐鹰 Tachyspiza fasciata 错配到隔壁属
                # 的 Aquila fasciata（fasciatus 变体查询恰好失败时）。
                # A failed variant query must abort, not skip: skipping turns
                # an ambiguous stem into a fake-unique hit (this once mapped
                # Tachyspiza fasciata onto Aquila fasciata when the
                # fasciatus-variant query transiently failed).
                return None
            for r in data.get("results") or []:
                if r.get("taxonomicStatus") != "ACCEPTED":
                    continue
                canon = (r.get("canonicalName") or "").split()
                if len(canon) >= 2 and _epithet_stem(" ".join(canon[-2:])) == stem:
                    hits[int(r["key"])] = r.get("canonicalName")
        if len(hits) == 1:
            key, canon = next(iter(hits.items()))
            return {
                "key": key,
                "status": "ACCEPTED",
                "via": f"family:{family}+epithet:{stem}→{canon}",
            }
        return None

    def _from_backbone_name(self, name: str) -> Optional[Dict]:
        """
        骨干库全名检索（可命中骨干已索引的异名）/ Backbone full-name search.

        match 接口对异名的覆盖不全；species/search 直接在骨干库里按全名
        检索，能拿到骨干已索引为 SYNONYM 的组合（其 acceptedKey 即正解）。

        The match API misses some synonyms; a direct backbone search by the
        full name catches combinations the backbone indexes as SYNONYM, whose
        acceptedKey is the answer.

        参数 / Parameters:
            name (str): 学名 / Scientific name.

        返回 / Returns:
            Optional[dict]: {key, status, via} 或 None / Resolution or None.
        """
        data = _gbif_get(
            "species/search",
            [
                ("datasetKey", BACKBONE),
                ("q", name),
                ("rank", "SPECIES"),
                ("limit", "10"),
            ],
        )
        if not data:
            return None
        for r in data.get("results") or []:
            sci = " ".join(
                t for t in (r.get("scientificName") or "").split() if t.isalpha()
            )
            if sci.lower() != name.lower():
                continue
            status = str(r.get("taxonomicStatus") or "")
            if status == "ACCEPTED" and r.get("key"):
                return {
                    "key": int(r["key"]),
                    "status": "ACCEPTED",
                    "via": f"backbone:{sci}",
                }
            if status.endswith("SYNONYM") and r.get("acceptedKey"):
                return {
                    "key": int(r["acceptedKey"]),
                    "status": f"SYNONYM→{r.get('accepted')}",
                    "via": f"backbone:{sci}",
                }
        return None

    def resolve(
        self, sci: str, ioc_name: Optional[str], family: Optional[str]
    ) -> Optional[Dict]:
        """
        依次尝试模型名 / IOC 名 / 骨干全名 / 科内加词检索。

        Try the model name, the IOC name, a backbone full-name search, then
        the family-scoped epithet search.

        参数 / Parameters:
            sci (str): 模型学名 / Model scientific name.
            ioc_name (Optional[str]): IOC 属+种加词（与模型名不同时有价值）/
                IOC genus + epithet (valuable when it differs).
            family (Optional[str]): IOC 科名 / IOC family name.

        返回 / Returns:
            Optional[dict]: {key, status, via} 或 None / Resolution or None.
        """
        name = _clean_name(sci)
        for cand in (name, ioc_name):
            if not cand:
                continue
            hit = self._from_match(cand)
            if hit:
                return hit
        for cand in (name, ioc_name):
            if not cand:
                continue
            hit = self._from_backbone_name(cand)
            if hit:
                return hit
        return self._from_family_epithet(name, family)


def load_groups() -> Dict[int, List[Dict]]:
    """
    载入全部碰撞组（附 IOC 属/加词/科名）/ Load shared-key groups with IOC hints.

    返回 / Returns:
        dict[int, list[dict]]: {specieskey: [{class_id, sci, cn, ioc, family}]}，
        仅含被多个类共享的 key / Only keys shared by multiple classes.
    """
    conn = sqlite3.connect(f"file:{REF_DB}?mode=ro", uri=True)
    try:
        groups: Dict[int, List[Dict]] = {}
        for skey, cid, sci in conn.execute(
            "SELECT specieskey, model_class_id, scientific_name "
            "FROM gbif_rarity_100 WHERE specieskey IS NOT NULL"
        ):
            cn_row = conn.execute(
                "SELECT chinese_simplified, english_name FROM BirdCountInfo "
                "WHERE model_class_id=?",
                (cid,),
            ).fetchone()
            ioc_row = conn.execute(
                "SELECT DISTINCT genus, species_scientific, family_scientific "
                "FROM bird_ioc WHERE birdcount_info_id="
                "(SELECT id FROM BirdCountInfo WHERE model_class_id=?) "
                "AND genus IS NOT NULL AND species_scientific IS NOT NULL "
                "LIMIT 1",
                (cid,),
            ).fetchone()
            ioc = (
                f"{ioc_row[0]} {ioc_row[1]}".strip()
                if ioc_row and ioc_row[0] and ioc_row[1]
                else None
            )
            family = ioc_row[2] if ioc_row else None
            groups.setdefault(int(skey), []).append(
                {
                    "class_id": int(cid),
                    "sci": sci,
                    "cn": cn_row[0] if cn_row else "?",
                    "en": cn_row[1] if cn_row else None,
                    "ioc": ioc if (ioc and ioc != _clean_name(sci)) else None,
                    "family": family,
                }
            )
    finally:
        conn.close()
    return {k: v for k, v in groups.items() if len(v) > 1}


def classify_groups(
    groups: Dict[int, List[Dict]],
) -> Tuple[List[Tuple[int, int, List[int]]], List[Tuple[int, List[Dict]]]]:
    """
    把碰撞组分成「有胜者」与「整组全灭」两类 / Split groups into winner/dead.

    参数 / Parameters:
        groups (dict): load_groups() 的输出 / Output of load_groups().

    返回 / Returns:
        tuple: (有胜者组 [(key, winner_id, 被清零ids)], 全灭组 [(key, 成员)]) /
            (winner groups, dead groups).

    异常 / Exceptions:
        RuntimeError: 组内多个类持有数据（历史验证为不可达状态）时抛出 /
            Raised when a group has multiple classes holding rows.
    """
    geo = sqlite3.connect(GEO_DB)
    winner_groups: List[Tuple[int, int, List[int]]] = []
    dead_groups: List[Tuple[int, List[Dict]]] = []
    try:
        for skey, members in sorted(groups.items()):
            ids = [m["class_id"] for m in members]
            marks = ",".join("?" * len(ids))
            with_rows = [
                r[0]
                for r in geo.execute(
                    f"SELECT class_id FROM cell_species WHERE class_id IN ({marks}) "
                    "GROUP BY class_id",
                    ids,
                )
            ]
            if len(with_rows) == 1:
                winner_groups.append(
                    (skey, with_rows[0], [i for i in ids if i != with_rows[0]])
                )
            elif not with_rows:
                dead_groups.append((skey, members))
            else:
                raise RuntimeError(
                    f"组 {skey} 有多个类持有数据 {with_rows}，人工核查后再执行"
                )
    finally:
        geo.close()
    return winner_groups, dead_groups


def fetch_country_counts(species_key: int) -> Optional[Dict[str, int]]:
    """
    一个物种在各国（ISO alpha-2）的记录数，与建库脚本同样过滤 license。

    Per-country occurrence counts for one species, license-filtered the same
    way as the build script.

    参数 / Parameters:
        species_key (int): GBIF 种级 taxonKey / Species-rank taxon key.

    返回 / Returns:
        Optional[dict]: {country: n}；失败 None / Counts or None on failure.
    """
    params = [
        ("taxonKey", str(species_key)),
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


def _vernacular_ok(species_key: int, class_en: Optional[str]) -> bool:
    """
    俗名交叉验证（仅科内加词路径用）/ Vernacular cross-check (epithet path).

    词干检索命中「科内唯一」也可能是隔壁属的同加词异种（非洲乌鹟
    Artomyias fuliginosa 命中红尾水鸲 Rhyacornis fuliginosa，且后者 key
    未被类表直接占用、key 守卫拦不住）。取骨干物种的俗名与类表英文名
    比较：双方都有俗名且无任何共有实词 → 判定错配。俗名缺失时不拦
    （无法验证就放行给既有守卫）。

    A unique stem hit can still be a same-epithet different species whose
    key is not directly claimed. Compare the backbone species' vernacular
    with the class's English name: reject when both exist and share no
    significant word. Missing vernaculars pass through to the other guards.

    参数 / Parameters:
        species_key (int): 解析出的骨干 key / The resolved backbone key.
        class_en (Optional[str]): 类表英文名 / The class's English name.

    返回 / Returns:
        bool: True=通过 / passes.
    """
    if not class_en:
        return True
    data = _gbif_get(f"species/{species_key}")
    if not data:
        return True
    vern = str(data.get("vernacularName") or "").lower().strip()
    if not vern:
        return True
    ours = {w for w in class_en.lower().replace("-", " ").split() if len(w) > 3}
    theirs = {w for w in vern.replace("-", " ").split() if len(w) > 3}
    return bool(ours & theirs) or vern == class_en.lower().strip()


def claimed_keys() -> Dict[int, set]:
    """
    已被模型类占用的 specieskey → 学名集合 / Specieskeys already claimed.

    词干检索可能把 A 属物种错配到同加词的 B 属物种（非洲乌鹟 Artomyias
    fuliginosa 曾命中红尾水鸲 Rhyacornis fuliginosa）。但 B 若本来就在
    模型类表里，它的 key 一定已被 B 自己的 gbif_rarity_100 行占用——
    「目标 key 已被**不同学名**的类占用」即可判定错配并拒绝。同种异属的
    正解不受影响：旧组合名（如 Ixobrychus sinensis）不在现代类表里，
    key 无人占用。

    Stem matching can map species of genus A onto a same-epithet species of
    genus B. When B itself is a model class, its key is already claimed by
    B's own gbif_rarity_100 row -- a target key claimed by a *different*
    scientific name therefore proves a mismatch and is rejected. Legitimate
    genus-renamed resolutions survive: legacy combinations (e.g. Ixobrychus
    sinensis) are not modern class names, so their keys are unclaimed.

    返回 / Returns:
        dict[int, set[str]]: {specieskey: {干净学名集合}} / Keys to names.
    """
    conn = sqlite3.connect(f"file:{REF_DB}?mode=ro", uri=True)
    try:
        claimed: Dict[int, set] = {}
        for skey, sci in conn.execute(
            "SELECT specieskey, scientific_name FROM gbif_rarity_100 "
            "WHERE specieskey IS NOT NULL"
        ):
            claimed.setdefault(int(skey), set()).add(_clean_name(sci).lower())
        return claimed
    finally:
        conn.close()


def resolve_dead_group(
    dead_groups: List[Tuple[int, List[Dict]]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    并发解析全灭组成员并拉国家级记录 / Resolve dead-group classes concurrently.

    解析结果经 claimed_keys 守卫过滤：目标 key 已被不同学名的模型类占用
    视为错配，降级为未解析并注明原因。

    Resolutions pass the claimed-key guard: a target key claimed by a
    different scientific name is a mismatch and downgrades to unresolved.

    参数 / Parameters:
        dead_groups (list): classify_groups() 的全灭组 / Dead groups.

    返回 / Returns:
        tuple: (resolved 列表 [{class_id, sci, cn, key, status, via, countries}],
        unresolved 列表) / (resolved, unresolved).
    """
    flat: List[Dict] = [m for _, members in dead_groups for m in members]
    resolver = Resolver()
    claimed = claimed_keys()

    def _work(m: Dict) -> Dict:
        hit = resolver.resolve(m["sci"], m.get("ioc"), m.get("family"))
        if hit is None:
            return {**m, "resolved": False, "reason": "未匹配到骨干种级 key"}
        names = claimed.get(hit["key"])
        mine = _clean_name(m["sci"]).lower()
        if names and mine not in names:
            owner = "/".join(sorted(names)[:3])
            return {
                **m,
                "resolved": False,
                "reason": f"key={hit['key']} 已被不同学名占用（{owner}），疑同加词错配",
            }
        if hit["via"].startswith("family:") and not _vernacular_ok(
            hit["key"], m.get("en")
        ):
            return {
                **m,
                "resolved": False,
                "reason": f"key={hit['key']} 俗名与类表英文名不符，疑同加词错配",
            }
        counts = fetch_country_counts(hit["key"])
        return {**m, "resolved": counts is not None, **hit, "countries": counts or {}}

    print(f"[fix] 解析全灭组 {len(flat)} 个类名（match / 骨干全名 / 科内加词 + 守卫 + country facet）...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(_work, flat))
    resolved = [r for r in results if r.get("resolved")]
    unresolved = [r for r in results if not r.get("resolved")]
    return resolved, unresolved


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fix specieskey-collision damage in geo_distribution.db"
    )
    p.add_argument("--execute", action="store_true",
                   help="写库（默认 dry-run）/ write (dry-run by default)")
    a = p.parse_args()

    groups = load_groups()
    winner_groups, dead_groups = classify_groups(groups)
    print(
        f"[fix] 碰撞组 {len(groups)} | 有胜者 {len(winner_groups)}（被清零 "
        f"{sum(len(x[2]) for x in winner_groups)} 类）| 全灭 {len(dead_groups)}"
        f"（{sum(len(m) for _, m in dead_groups)} 类）"
    )

    # ---- 1) 有胜者组：统计待复制行 ----
    geo = sqlite3.connect(GEO_DB)
    cell_add: List[Tuple[int, int, int]] = []
    country_add: List[Tuple[str, int, int]] = []
    for skey, winner, starved in winner_groups:
        cell_rows = geo.execute(
            "SELECT cell_id, n FROM cell_species WHERE class_id=?", (winner,)
        ).fetchall()
        country_rows = geo.execute(
            "SELECT country, n FROM country_species WHERE class_id=?", (winner,)
        ).fetchall()
        for cls in starved:
            cell_add.extend((cid, cls, n) for cid, n in cell_rows)
            country_add.extend((cc, cls, n) for cc, n in country_rows)
    print(f"[fix] 复制胜者行: cell_species +{len(cell_add):,} 行, "
          f"country_species +{len(country_add):,} 行")

    # ---- 2) 全灭组：解析 + 国家记录 ----
    resolved, unresolved = resolve_dead_group(dead_groups)
    for r in resolved:
        country_add.extend(
            (cc, r["class_id"], n) for cc, n in r["countries"].items()
        )
    print(f"[fix] 全灭组解析成功 {len(resolved)} / 失败 {len(unresolved)}")
    for r in sorted(resolved, key=lambda x: -x["countries"].get("CN", 0)):
        cn_n = r["countries"].get("CN", 0)
        print(f"  ✓ {r['cn']} {_clean_name(r['sci'])} → key={r['key']} "
              f"[{r['via']}] 国家数={len(r['countries'])} CN={cn_n}")
    for r in unresolved:
        reason = f"（{r['reason']}）" if r.get("reason") else "（保持现状）"
        print(f"  ✗ 未能解析: {r['cn']} {_clean_name(r['sci'])}"
              f"（class {r['class_id']}）{reason}")

    if not a.execute:
        print("[fix] dry-run 结束（未写库）。加 --execute 执行。")
        geo.close()
        return

    # ---- 3) 备份 + 写库 ----
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{GEO_DB}.bak_specieskey_fix_{stamp}"
    shutil.copy2(GEO_DB, backup)
    print(f"[fix] 已备份 / backup: {backup}")

    try:
        geo.executemany(
            "INSERT OR IGNORE INTO cell_species (cell_id, class_id, n) VALUES (?,?,?)",
            cell_add,
        )
        geo.executemany(
            "INSERT OR IGNORE INTO country_species (country, class_id, n) VALUES (?,?,?)",
            country_add,
        )
        geo.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (
                "collision_fix",
                json.dumps(
                    {
                        "date": datetime.now().isoformat(timespec="seconds"),
                        "winner_groups": len(winner_groups),
                        "cell_rows_copied": len(cell_add),
                        "country_rows_written": len(country_add),
                        "dead_resolved": len(resolved),
                        "dead_unresolved": len(unresolved),
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        geo.commit()
    except sqlite3.Error as e:
        geo.close()
        raise SystemExit(f"[fix] 写库失败，原库未受影响（备份在 {backup}）: {e}")

    # ---- 4) 覆盖表（供 build_geo_distribution.py 下次全量重建应用）----
    overrides = [
        {
            "model_class_id": r["class_id"],
            "specieskey": r["key"],
            "scientific_name": _clean_name(r["sci"]),
            "status": r["status"],
            "resolved_via": r["via"],
            "resolved_at": datetime.now().isoformat(timespec="seconds"),
            "source": "fix_geo_key_collisions.py / GBIF backbone",
        }
        for r in resolved
    ]
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=1)
    print(f"[fix] 覆盖表写入 / overrides: {OVERRIDES_PATH}（{len(overrides)} 条）")

    # ---- 5) 旗舰抽查 ----
    checks = {
        1388: "黑水鸡 G.chloropus",
        1119: "白尾鹞 C.cyaneus",
        8175: "山鹛 R.pekinensis",
        811: "黄苇鳽 B.sinensis",
    }
    print("[fix] 旗舰抽查（CN 记录数 / 全球格数）:")
    for cid, label in checks.items():
        row = geo.execute(
            "SELECT n FROM country_species WHERE country='CN' AND class_id=?",
            (cid,),
        ).fetchone()
        cn = row[0] if row else 0
        cells, = geo.execute(
            "SELECT COUNT(*) FROM cell_species WHERE class_id=?", (cid,)
        ).fetchone()
        print(f"  {label}: CN={cn} cells={cells}")
    geo.close()
    print("[fix] 完成 / done")


if __name__ == "__main__":
    sys.exit(main())
