# -*- coding: utf-8 -*-
"""
构建国家保护等级表 china_protection / Build the national protection level
table china_protection in bird_reference.sqlite.

数据源 / Data source:
    zh.wikipedia《国家重点保护野生动物名录》条目（2021 年第 3 号公告的全量
    转录，CC BY-SA 4.0）。页面由 MediaWiki API 以简体变体（variant=zh-cn）
    保存到 scripts_dev/data_sources/wiki_protected_2021.json，本脚本只解析
    本地文件，不依赖运行时网络（zh.wikipedia.org 在中国大陆不可直连）。
    The page (a full transcription of the official 2021 No.3 announcement)
    is saved locally via the MediaWiki API; this script parses the local
    snapshot only. Wikipedia text is CC BY-SA 4.0 — attribution is recorded
    in the table's source column.

已知数据质量风险 / Known quality risks (measured 2026-08-29):
    维基条目存在学名笔误（实证：红隼被写成 Falco vespertinus，应为
    F. tinnunculus）。因此匹配采用多级管线：①学名精确匹配模型库 →
    ②唯一中文名匹配 → ③GBIF /species/match 归一（仅兜底，全部列入审计）→
    ④脚本内人工映射表 → ⑤未决项写入审计 CSV 人工把关。学名匹配成功但中
    文名不一致的条目也会进审计清单。Wiki has known scientific-name typos,
    so a layered match pipeline plus a mandatory audit CSV is used.

校验 / Validation:
    双名法格式校验、重复学名检测、「所有种/spp.」聚合行硬失败（须显式补
    展开规则）、条目总数 sanity（350-430）、旗舰种级别硬断言。
    Binomial-format check, duplicate detection, hard failure on aggregate
    rows, total-count sanity, and flagship-level assertions.

用法 / Usage:
    .venv/Scripts/python scripts_dev/build_china_protection.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DB = os.path.join(PROJ, "birdid", "data", "bird_reference.sqlite")
WIKI_JSON = os.path.join(PROJ, "scripts_dev", "data_sources", "wiki_protected_2021.json")
AUDIT_CSV = os.path.join(PROJ, "scripts_dev", "data_sources", "china_protection_audit.csv")

SOURCE_NOTE = "国家重点保护野生动物名录(2021年第3号公告), via zh.wikipedia (CC BY-SA 4.0)"

# 内置人工映射表：维基学名笔误/异名 → 模型库学名。
# 每条必须附理由，改动走审计 CSV 复核。
# Manual fixes: wiki scientific-name typo / synonym → model library name.
MANUAL_NAME_MAP: Dict[str, Tuple[str, str]] = {
    "falco vespertinus": (
        "Falco tinnunculus",
        "维基把红隼误写为 F. vespertinus（红脚隼）；红隼学名应为 F. tinnunculus，"
        "中文名匹配亦印证。Wiki typo: 红隼 mis-written as F. vespertinus. "
        "[zcode 2026-08-29]",
    ),
    "bubo blakistoni": (
        "Ketupa blakistoni",
        "毛腿雕鸮/毛腿渔鸮：渔鸮类已由 Bubo 拆入 Ketupa，模型库用 Ketupa "
        "blakistoni (class 2464)。Genus split Bubo → Ketupa. [zcode 2026-08-29]",
    ),
    "garrulax courtoisi": (
        "Pterorhinus courtoisi",
        "蓝冠噪鹛（库中文名「靛冠噪鹛」故中文名未命中）：噪鹛类已由 Garrulax "
        "拆入 Pterorhinus，模型库用 Pterorhinus courtoisi (class 8062)。"
        "Genus split Garrulax → Pterorhinus. [zcode 2026-08-29]",
    ),
}

# 旗舰种级别断言（学名小写 → 期望级别 1/2），不符即中止构建。
# Flagship assertions on protection level; abort the build on mismatch.
FLAGSHIP_ASSERTS: Dict[str, int] = {
    "aquila chrysaetos": 1,   # 金雕
    "nipponia nippon": 1,     # 朱鹮
    "pavo muticus": 1,        # 绿孔雀
    "emberiza aureola": 1,    # 黄胸鹀
    "cygnus cygnus": 2,       # 大天鹅
    "asio otus": 2,           # 长耳鸮
}


@dataclass
class WikiEntry:
    """维基表格中的一个物种行 / One species row from the wiki table."""

    chinese_name: str
    scientific_name: str
    level: int          # 1 / 2
    note: str           # 备注列原文 / raw note column
    taxon_class: str    # 所属纲（应恒为 鸟纲）/ containing class, always 鸟纲
    order: str          # 所属目 / containing order


class BirdTableParser(HTMLParser):
    """
    解析维基名录单张巨型表格，抽取鸟纲物种行。

    表格结构（实测）：纲标题行 colspan=4；目/科行为 4 格且级别格为空；
    物种行 4 格 = 中文名 | 斜体学名 | Ⅰ/Ⅱ | 备注。以「纲标题行」切换
    收集范围，只保留鸟纲内的物种行。

    Table layout (measured): class-header rows use colspan=4; order/family
    rows have an empty level cell; species rows are
    chinese | italic sci | Ⅰ/Ⅱ | note. Collection scope follows the current
    class-header row and only bird rows are kept.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag == "td" and self._row is not None:
            text = "".join(self._cell_parts)
            text = re.sub(r"\[\d+\]", "", text)      # 去脚注引用 / strip footnotes
            text = text.replace("\u00a0", " ").strip(" *")
            text = re.sub(r"\s+", " ", text).strip()
            self._row.append(text)
            self._cell_parts = []
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._row is not None and self._cell_parts is not None:
            self._cell_parts.append(data)


def _norm_sci(name: str) -> str:
    """学名归一化：小写、去作者与括号 / Normalize a scientific name."""
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.lower()


def load_bird_entries() -> List[WikiEntry]:
    """
    从本地维基快照解析鸟纲物种行，并做结构级校验。

    Parse bird species rows from the local wiki snapshot with structural
    validation: aggregate rows (所有种/spp.) abort the build; totals must
    fall in 350-430.

    返回 / Returns:
        list[WikiEntry]: 鸟纲全部物种行 / All bird species rows.

    异常 / Exceptions:
        RuntimeError: 出现聚合行或总数超出 sanity 区间时抛出 / Raised on
            aggregate rows or an out-of-range total.
        FileNotFoundError: 本地快照缺失时抛出 / Raised when the snapshot is
            missing (re-download it via the MediaWiki API with variant=zh-cn).
    """
    if not os.path.exists(WIKI_JSON):
        raise FileNotFoundError(
            f"维基快照缺失 / wiki snapshot missing: {WIKI_JSON}\n"
            "用 MediaWiki API 重新下载 / re-download via: "
            "https://zh.wikipedia.org/w/api.php?action=parse&page="
            "国家重点保护野生动物名录&prop=text&format=json&formatversion=2&variant=zh-cn"
        )
    with open(WIKI_JSON, "r", encoding="utf-8") as f:
        html = json.load(f)["parse"]["text"]

    parser = BirdTableParser()
    parser.feed(html)

    entries: List[WikiEntry] = []
    aggregates: List[str] = []
    current_class = ""
    current_order = ""
    seen: Dict[str, int] = {}
    for cells in parser.rows:
        if not cells:
            continue
        first = cells[0]
        if first.endswith("纲") or "纲 " in first:
            current_class = first.split()[0]
            continue
        if first.endswith("目") and len(first) <= 8:
            current_order = first
            continue
        level = -1
        if len(cells) >= 3:
            m = re.search(r"[ⅠⅡ]", cells[2])
            if m:
                level = 1 if m.group() == "Ⅰ" else 2
        if level < 0 or current_class != "鸟纲":
            continue
        raw_sci = cells[1]
        if "所有种" in first or "所有种" in raw_sci or "spp" in raw_sci.lower():
            aggregates.append(f"{first} | {raw_sci}")
            continue
        # 双名法硬校验：属名+种名各一个词；亚种名/含作者等异常会被拦下人工处理
        # Hard binomial gate: exactly genus + species; trinomials or names
        # with author strings are surfaced for manual handling instead of
        # silently entering the table.
        sci_words = _norm_sci(raw_sci).split()
        if (len(sci_words) != 2
                or not all(re.fullmatch(r"[a-z][a-z\-]+", w) for w in sci_words)):
            aggregates.append(f"非双名法 / non-binomial: {first} | {raw_sci}")
            continue
        sci = sci_words[0].capitalize() + " " + sci_words[1]
        seen[sci] = seen.get(sci, 0) + 1
        entries.append(WikiEntry(first, sci, level, cells[3] if len(cells) > 3 else "",
                                 current_class, current_order))

    if aggregates:
        raise RuntimeError(
            "出现「所有种/非双名法」聚合行，需要显式展开规则，禁止静默处理 / "
            f"aggregate rows require explicit expansion rules: {aggregates[:10]}"
        )
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        print(f"[build] ⚠️ 维基重复学名 / duplicate sci names: {dups}")
    if not 350 <= len(entries) <= 430:
        raise RuntimeError(
            f"鸟纲条目数异常 / unexpected bird entry count: {len(entries)}（期望 ~390）"
        )
    print(f"[build] 鸟纲条目 / bird entries: {len(entries)}")
    return entries


def load_library() -> Tuple[Dict[str, int], Dict[str, List[int]], Dict[int, Tuple[str, str]]]:
    """
    加载模型库的学名/中文名索引。

    Load scientific-name and unique-Chinese-name indexes of the model library
    from BirdCountInfo.

    返回 / Returns:
        tuple: (sci→class_id, 唯一中文→class_id, class_id→(学名, 中文名))
    """
    conn = sqlite3.connect(REF_DB)
    try:
        sci_to_cid: Dict[str, int] = {}
        zh_count: Dict[str, int] = {}
        zh_to_cid: Dict[str, List[int]] = {}
        info: Dict[int, Tuple[str, str]] = {}
        for cid, sci, zh in conn.execute(
            "SELECT model_class_id, scientific_name, chinese_simplified FROM BirdCountInfo"
        ):
            sci_to_cid[_norm_sci(sci)] = cid
            info[cid] = (sci, zh)
            if zh:
                zh_count[zh] = zh_count.get(zh, 0) + 1
                zh_to_cid.setdefault(zh, []).append(cid)
        unique_zh = {z: ids[0] for z, ids in zh_to_cid.items() if len(ids) == 1}
        return sci_to_cid, unique_zh, info
    finally:
        conn.close()


def gbif_match(query: str) -> Optional[str]:
    """
    GBIF /species/match 把维基学名归一到接受的学名。

    Resolve a wiki scientific name via the GBIF name-match API and return the
    accepted canonical name (lowercased), or None on any failure.

    参数 / Parameters:
        query (str): 待归一学名 / scientific name to resolve.

    返回 / Returns:
        Optional[str]: 接受学名（小写）；失败返回 None / accepted name or None.
    """
    url = ("https://api.gbif.org/v1/species/match?strict=false&name="
           + urllib.parse.quote(query))
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "SuperPicky-build/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            canon = data.get("canonicalName")
            if data.get("matchType") in ("EXACT", "HIGHRANK", "FUZZY") and canon:
                return str(canon).lower()
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


@dataclass
class MatchResult:
    """单个条目的匹配结果 / Match outcome for one wiki entry."""

    entry: WikiEntry
    class_id: Optional[int]
    method: str          # sci / chinese / gbif / manual / unresolved
    model_sci: str = ""
    model_zh: str = ""
    reason: str = ""


def match_entries(
    entries: List[WikiEntry],
    sci_to_cid: Dict[str, int],
    unique_zh: Dict[str, List[int]],
    info: Dict[int, Tuple[str, str]],
) -> List[MatchResult]:
    """
    五级匹配管线：学名 → 唯一中文名 → GBIF 归一 → 人工映射 → 未决。

    Layered match pipeline: exact sci → unique Chinese → GBIF canonical →
    manual map → unresolved. Sci-matched entries whose Chinese name disagrees
    with the library are downgraded to the audit list for human review.

    参数 / Parameters:
        entries (list): 鸟纲条目 / Parsed wiki entries.
        sci_to_cid / unique_zh / info: load_library() 的索引 / Library indexes.

    返回 / Returns:
        list[MatchResult]: 与 entries 等长 / Same length as entries.
    """
    results: List[MatchResult] = []
    for e in entries:
        key = _norm_sci(e.scientific_name)
        cid = sci_to_cid.get(key)
        if cid is not None:
            m_sci, m_zh = info[cid]
            if e.chinese_name != m_zh:
                results.append(MatchResult(
                    e, cid, "sci", m_sci, m_zh,
                    f"学名命中但中文名不一致（维基「{e.chinese_name}」vs 库「{m_zh}」），请复核"))
            else:
                results.append(MatchResult(e, cid, "sci", m_sci, m_zh))
            continue

        zh_ids = unique_zh.get(e.chinese_name)
        if zh_ids is not None:
            m_sci, m_zh = info[zh_ids]
            results.append(MatchResult(
                e, zh_ids, "chinese", m_sci, m_zh,
                f"经唯一中文名匹配（维基学名 {e.scientific_name} 未命中库，"
                f"库学名 {m_sci}），请复核学名笔误"))
            continue

        if key in MANUAL_NAME_MAP:
            fixed, reason = MANUAL_NAME_MAP[key]
            cid = sci_to_cid.get(_norm_sci(fixed))
            if cid is not None:
                m_sci, m_zh = info[cid]
                results.append(MatchResult(e, cid, "manual", m_sci, m_zh, reason))
                continue

        canon = gbif_match(e.scientific_name)
        if canon:
            cid = sci_to_cid.get(canon)
            if cid is not None:
                m_sci, m_zh = info[cid]
                results.append(MatchResult(
                    e, cid, "gbif", m_sci, m_zh,
                    f"GBIF 归一 {e.scientific_name} → {canon}；学名笔误可能指向错误鸟种，请务必复核"))
                continue
        results.append(MatchResult(e, None, "unresolved", reason="未匹配，见审计 CSV"))
    return results


def run_flagship_asserts(results: List[MatchResult]) -> None:
    """
    校验旗舰种的保护级别，不符即中止构建。

    Assert protection levels of flagship species; abort the build on any
    mismatch (guards against wiki transcription errors).

    参数 / Parameters:
        results (list): match_entries() 的结果 / Match results.

    异常 / Exceptions:
        AssertionError: 旗舰种缺失或级别不符时抛出 / Raised on missing or
            mismatched flagship levels.
    """
    by_key = {_norm_sci(r.model_sci or r.entry.scientific_name): r for r in results}
    for sci, want in FLAGSHIP_ASSERTS.items():
        r = by_key.get(sci)
        assert r is not None, f"旗舰种缺失 / flagship missing: {sci}"
        assert r.class_id is not None, f"旗舰种未匹配 / flagship unmatched: {sci}"
        assert r.entry.level == want, (
            f"旗舰种级别不符 / flagship level mismatch: {sci} "
            f"期望/want={want} 维基/wiki={r.entry.level}"
        )
    print(f"[build] 旗舰种断言通过 / flagship asserts passed: {len(FLAGSHIP_ASSERTS)}")


def write_audit_csv(results: List[MatchResult]) -> None:
    """
    把需要人工复核/未决的条目写入审计 CSV。

    Write every entry that needs human review (chinese/gbif/manual matches,
    name mismatches) or is unresolved into the audit CSV.

    参数 / Parameters:
        results (list): match_entries() 的结果 / Match results.
    """
    os.makedirs(os.path.dirname(AUDIT_CSV), exist_ok=True)
    with open(AUDIT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["维基中文名", "维基学名", "级别", "匹配方式",
                    "库class_id", "库学名", "库中文名", "说明", "维基目", "备注"])
        for r in results:
            if r.method in ("sci",) and not r.reason:
                continue
            w.writerow([r.entry.chinese_name, r.entry.scientific_name, r.entry.level,
                        r.method, r.class_id or "", r.model_sci, r.model_zh,
                        r.reason, r.entry.order, r.entry.note])
    n_review = sum(1 for r in results
                   if r.class_id is None or r.reason or r.method != "sci")
    print(f"[build] 审计 CSV / audit rows: {n_review} → {AUDIT_CSV}")


def write_table(results: List[MatchResult]) -> int:
    """
    DROP+重建 china_protection 表并写入匹配成功的行。

    Recreate china_protection in bird_reference.sqlite and insert matched
    rows only. Unresolved entries are excluded (they fail loudly in the
    build output and audit CSV instead of silently polluting the table).

    参数 / Parameters:
        results (list): match_entries() 的结果 / Match results.

    返回 / Returns:
        tuple[int, int]: (写入行数, 其中一级条数) / (rows written, level-1 rows).
    """
    conn = sqlite3.connect(REF_DB)
    try:
        conn.execute("DROP TABLE IF EXISTS china_protection")
        conn.execute(
            """
            CREATE TABLE china_protection (
                model_class_id  INTEGER PRIMARY KEY,
                scientific_name TEXT NOT NULL,
                level           INTEGER NOT NULL CHECK (level IN (1, 2)),
                chinese_name    TEXT,
                source          TEXT
            )
            """
        )
        # 维基同一物种可能重复出现（如 原鸡/红原鸡 两条同名行），按 class_id
        # 去重：级别一致则合并，冲突则中止交人工裁决。
        # The wiki table may list one species twice (e.g. 原鸡/红原鸡 rows);
        # dedupe by class_id, merging when levels agree and aborting on
        # conflicts so a human can adjudicate.
        grouped: Dict[int, MatchResult] = {}
        for r in results:
            if r.class_id is None:
                continue
            prev = grouped.get(r.class_id)
            if prev is None:
                grouped[r.class_id] = r
            elif prev.entry.level != r.entry.level:
                raise RuntimeError(
                    "同一物种级别冲突 / conflicting levels for "
                    f"{r.model_sci}: {prev.entry.level} vs {r.entry.level}"
                )
        rows = [
            (cid, r.model_sci, r.entry.level, r.model_zh, SOURCE_NOTE)
            for cid, r in grouped.items()
        ]
        conn.executemany(
            "INSERT INTO china_protection "
            "(model_class_id, scientific_name, level, chinese_name, source) "
            "VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
        lv1 = sum(1 for r in grouped.values() if r.entry.level == 1)
        return len(rows), lv1
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build china_protection in bird_reference.sqlite from the wiki snapshot"
    )
    a = p.parse_args()

    entries = load_bird_entries()
    sci_to_cid, unique_zh, info = load_library()
    results = match_entries(entries, sci_to_cid, unique_zh, info)
    run_flagship_asserts(results)
    write_audit_csv(results)

    n_by_method: Dict[str, int] = {}
    for r in results:
        n_by_method[r.method] = n_by_method.get(r.method, 0) + 1
    print("[build] 匹配方式分布 / match methods:", n_by_method)
    unresolved = [r for r in results if r.class_id is None]
    if len(unresolved) > 15:
        raise RuntimeError(
            f"未决条目过多 / too many unresolved: {len(unresolved)}，"
            f"先补 MANUAL_NAME_MAP 再构建 / extend MANUAL_NAME_MAP first"
        )
    n, lv1 = write_table(results)
    print(f"[build] china_protection 写入 {n} 行（一级 {lv1} / 二级 {n - lv1}）")
    print("[build] 完成 / done")


if __name__ == "__main__":
    main()
