#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 配额定星重算（不重跑检测/识鸟）— 只用 report.db 现成指标重建定星输入。

背景 / Background:
CLI `restar` 走的是 PostAdjustmentEngine V1 绝对阈值逻辑，且 0 星下限读
advanced_config 的 min_sharpness/min_nima（而非 -s/-n 参数），无法复现
process 主流程的 V2 批内配额定星。本脚本按 photo_processor V2 收尾阶段
的同一算法（core.rating_quota.assign_ratings）从 DB 现有字段重建指标：

- norm_sharpness = head_sharp × ISO 归一化系数（ISO_BASE=800，每翻倍 -5%，下限 0.5）
- topiq          = nima_score（鸟裁剪区 TOPIQ 原始分）
- best_eye       = max(left_eye, right_eye)；beak_vis = beak
- species        = bird_species_en 或 bird_species_cn（与收尾阶段 en_name or cn_name 一致）
- 连拍/相似簇    = DB burst_id + 对已识鸟、无连拍组的照片重跑 pHash 相似聚类
                  （与收尾阶段同一函数 core.burst_detector.cluster_similar_by_phash）

自校验 / Self-check:
先用当前配置参数（min_confidence=0.7 + 配额 20/30）复算一遍，应精确复现
DB 现存评级；复现失败则说明重建有误，退出码 2，不进入任何写库分支。

用途 / Usage:
    python scripts_dev/rerate_v2_conf_gate.py <照片目录> [--dry-run] [--execute]
        [--min-conf 0.5] [--quota3 20] [--quota2 30|40]

默认 dry-run：只打印分布对比，不写库。--execute 时先备份 report.db 再更新
rating 字段（单一写者原则：本脚本即 SuperPicky 运维上下文，不碰照片文件）。
"""

import argparse
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rating_quota import (  # noqa: E402
    PhotoMetricsV2,
    RatingV2Result,
    assign_ratings,
    gate_photo,
)
from tools.i18n import get_i18n  # noqa: E402

# 与 photo_processor.PhotodProcessor 类常量一致（ISO 归一化）
ISO_BASE = 800
ISO_PENALTY_FACTOR = 0.05
ISO_MIN_FACTOR = 0.5

# 本次跑批实际生效的参数（用于自校验复现）
CURRENT_MIN_CONF = 0.7
CURRENT_QUOTA3 = 20.0
CURRENT_QUOTA2 = 30.0


def iso_factor(iso_value: Optional[int]) -> float:
    """
    ISO 锐度归一化系数，与 PhotoProcessor._get_iso_sharpness_factor 一致。

    参数:
        iso_value (Optional[int]): 照片 ISO 值，None 或 ≤800 视为 1.0

    返回:
        float: 归一化系数（0.5 ~ 1.0）

    ISO sharpness normalization factor, identical to the processor's method.
    """
    if iso_value is None or iso_value <= ISO_BASE:
        return 1.0
    penalty = ISO_PENALTY_FACTOR * math.log2(iso_value / ISO_BASE)
    return max(ISO_MIN_FACTOR, 1.0 - penalty)


def load_photos(db_path: str) -> List[Dict]:
    """
    从 report.db 读取重建 V2 指标所需的全部字段。

    参数:
        db_path (str): report.db 路径

    返回:
        List[Dict]: 每张照片一行（dict）

    Load all fields needed to rebuild V2 metrics from the report DB.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT filename, has_bird, confidence, head_sharp, left_eye, right_eye,"
            " beak, nima_score, is_flying, focus_status, burst_id, iso, rating,"
            " bird_species_en, bird_species_cn, adj_sharpness, adj_topiq"
            " FROM photos ORDER BY filename"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 匹配识别日志行：🐦 Bird ID [027A4921.CR3]: 白头鹎 (97%)...
#               🐦 Low confidence [027A4878.CR3]: 蓝歌鸲 (45% < 50.0%)
# 半角括号+百分号保证不误匹配 Multi-bird / 主鸟重选（全角括号）行。
BIRDID_LINE_RE = re.compile(r"Bird ID \[([^\]]+)\]: (\S+) \(\d+%(?:\s*<[^)]*)?\)")
LOWCONF_LINE_RE = re.compile(
    r"Low confidence \[([^\]]+)\]: (\S+) \(\d+%(?:\s*<[^)]*)?\)")


def parse_species_from_log(log_path: str) -> Dict[str, str]:
    """
    从 superpicky.log 解析每张照片的识别鸟种（含低置信候选）。

    机制：低置信识别（< birdid 阈值）的鸟种会进内存 file_bird_species 参与
    V2 按种分组配额，但不写 photos 表鸟种列（仅高置信写入），也不可靠地
    落入 caption。唯一完整持久化载体是运行日志的两类行；同函数打印，
    与收尾分组读到的标签一一对应。中文本地化下记录的是中文名，用作分组
    键与 en_name 等价（同一分类一对一同映射）。

    参数:
        log_path (str): superpicky.log 路径

    返回:
        Dict[str, str]: 文件名前缀 → 鸟种名；同一前缀多行时取最后一次

    Parse per-photo species labels (incl. low-confidence candidates) from
    the run log; the only complete persisted source for quota grouping.
    """
    labels: Dict[str, str] = {}
    if not os.path.exists(log_path):
        return labels
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            plain = ANSI_RE.sub("", line)
            m = (LOWCONF_LINE_RE.search(plain) or BIRDID_LINE_RE.search(plain))
            if m:
                prefix = os.path.splitext(m.group(1))[0]
                labels[prefix] = m.group(2)
    return labels


def build_metrics(rows: List[Dict],
                  species_labels: Optional[Dict[str, str]] = None
                  ) -> Dict[str, PhotoMetricsV2]:
    """
    按 process 收尾阶段的口径，把 DB 行重建为 PhotoMetricsV2（不含相似簇）。

    参数:
        rows (List[Dict]): load_photos 的输出
        species_labels (Optional[Dict[str, str]]): 日志解析的鸟种标签；
            优先于 DB 鸟种列（DB 缺低置信候选，而分组恰恰需要它们）

    返回:
        Dict[str, PhotoMetricsV2]: key=filename 前缀 → 指标

    Rebuild per-photo V2 metrics from DB rows (without similarity clusters).
    """
    metrics: Dict[str, PhotoMetricsV2] = {}
    for r in rows:
        if not r["has_bird"]:
            continue
        best_eye = max(r["left_eye"] or 0.0, r["right_eye"] or 0.0)
        m = PhotoMetricsV2(
            key=r["filename"],
            detected=True,
            confidence=float(r["confidence"] or 0.0),
            norm_sharpness=float(r["head_sharp"] or 0.0)
            * iso_factor(r["iso"]),
            topiq=(float(r["nima_score"]) if r["nima_score"] is not None else None),
            best_eye=best_eye,
            beak_vis=float(r["beak"] or 0.0),
            is_flying=bool(r["is_flying"]),
            focus_status=r["focus_status"] or "",
            has_exposure_issue=False,  # 本次与沙河批次 exposure_check 均为关
            burst_id=r["burst_id"],
            species=(species_labels or {}).get(r["filename"])
            or r["bird_species_en"] or r["bird_species_cn"],
        )
        metrics[m.key] = m
    return metrics


def attach_similar_clusters(pool: List[PhotoMetricsV2], root: str) -> None:
    """
    对入池照片做 pHash 相似聚类并写回伪 burst_id（原地修改）。

    与 process 收尾阶段完全一致：只对「过门槛入池」的照片聚类（v2_pending
    口径），且仅限已识鸟、无真实连拍组、有 temp_preview 预览者。伪组 id 接
    在真实连拍组之后，只影响打星封顶。

    参数:
        pool (List[PhotoMetricsV2]): 过硬门槛的入池照片
        root (str): 照片目录（定位预览 JPG）

    Cluster pool photos by pHash similarity (post-pass semantics: pool only).
    """
    from core.burst_detector import cluster_similar_by_phash

    sim_items = []
    preview_dir = os.path.join(root, ".superpicky", "cache", "temp_preview")
    for m in pool:
        if m.species and m.burst_id is None:
            preview = os.path.join(preview_dir, m.key + ".jpg")
            if os.path.exists(preview):
                sim_items.append((m.key, m.species, preview))
    clusters = cluster_similar_by_phash(sim_items)
    if not clusters:
        return
    next_id = (max((m.burst_id for m in pool if m.burst_id is not None),
                   default=0)) + 1
    by_key = {m.key: m for m in pool}
    for cl in clusters:
        for key in cl:
            by_key[key].burst_id = next_id
        next_id += 1


def simulate(metrics: Dict[str, PhotoMetricsV2], root: str, min_conf: float,
             quota3: float, quota2: float
             ) -> Dict[str, "RatingV2Result"]:
    """
    按给定门槛/配额跑一遍 V2 定星（门槛 → 入池聚类 → 配额定星）。

    注意相似簇依赖门槛结果（只聚入池照片），因此每次模拟都要重新聚类，
    不能跨 min_conf 复用。

    参数:
        metrics (Dict[str, PhotoMetricsV2]): 全部有鸟照片指标
        root (str): 照片目录
        min_conf (float): 置信度硬门槛
        quota3 / quota2 (float): 3★ / 2★ 配额百分比

    返回:
        Dict[str, RatingV2Result]: key → 定星结果（星级+原因键，用于 caption）

    Full V2 simulation: gate, then cluster the pool, then assign by quota.
    """
    results: Dict[str, RatingV2Result] = {}
    pool: List[PhotoMetricsV2] = []
    for m in metrics.values():
        gated = gate_photo(m, min_confidence=min_conf)
        if gated is not None:
            results[m.key] = gated
        else:
            pool.append(m)
    attach_similar_clusters(pool, root)
    results.update(assign_ratings(pool, quota3=quota3, quota2=quota2,
                                  min_confidence=min_conf))
    return results


def dist(ratings: Dict[str, int], total: int) -> str:
    """
    格式化星级分布（含无鸟 -1）。

    参数:
        ratings (Dict[str, int]): key → 星级（-1=无鸟）
        total (int): 总张数（用于交叉核对）

    返回:
        str: 一行分布文本

    Format a one-line star distribution including no-bird count.
    """
    c = {3: 0, 2: 0, 1: 0, 0: 0, -1: 0}
    for v in ratings.values():
        c[v] = c.get(v, 0) + 1
    return (f"3★={c[3]}  2★={c[2]}  1★={c[1]}  0★={c[0]}  无鸟={c[-1]}"
            f"  （合计 {sum(c.values())}/{total}）")


def main() -> int:
    """
    命令行入口：自校验 → 模拟新分布 →（可选）写库。

    返回:
        int: 0 成功；2 自校验失败；参数错误 1

    CLI entry: self-check first, then simulate; write only with --execute.
    """
    ap = argparse.ArgumentParser(description="V2 配额定星重算（不重跑识别）")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="置信度门槛（默认 0.4，与批量识别工作流定档一致）")
    ap.add_argument("--quota3", type=float, default=CURRENT_QUOTA3,
                    help="3★ 配额 %%（默认 20，与当前配置一致）")
    ap.add_argument("--quota2", type=float, default=CURRENT_QUOTA2,
                    help="2★ 配额 %%（默认 30，当前配置；沙河快照为 40）")
    ap.add_argument("--execute", action="store_true", help="写库（默认只模拟）")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    db_path = os.path.join(root, ".superpicky", "report.db")
    if not os.path.exists(db_path):
        print(f"❌ 未找到 {db_path}")
        return 1

    rows = load_photos(db_path)
    total = len(rows)
    old = {r["filename"]: int(r["rating"]) for r in rows}
    print(f"📁 {root}")
    print(f"📊 DB 现存分布: {dist(old, total)}")

    # 从日志解析鸟种标签（含低置信候选），并解析真实跑批的
    # 「每鸟种 3★/池内」分布行作为分组校验和
    log_path = os.path.join(root, "superpicky.log")
    labels = parse_species_from_log(log_path)
    print(f"🏷️ 日志解析到 {len(labels)} 张照片的鸟种标签")

    metrics = build_metrics(rows, labels)
    print(f"🐦 有鸟照片 {len(metrics)} 张（重建 V2 指标）")

    # ---- 分组校验和：0.7 门槛下的池内分组应与真实跑批明细一致 ----
    real_breakdown: Dict[str, int] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            plain = ANSI_RE.sub("", line)
            if "/池内" in plain or ("·" in plain and re.search(r"\d+/\d+", plain)):
                for name, cnt in re.findall(r"(\S+) \d+/(\d+)", plain):
                    real_breakdown[name] = int(cnt)
    pool070 = [m for m in metrics.values()
               if gate_photo(m, min_confidence=CURRENT_MIN_CONF) is None]
    sim_breakdown: Dict[str, int] = {}
    for m in pool070:
        sim_breakdown[m.species or "未识别"] = \
            sim_breakdown.get(m.species or "未识别", 0) + 1
    print(f"🔍 池内分组: 模拟={dict(sorted(sim_breakdown.items(), key=lambda x: -x[1]))}")
    print(f"           真实={dict(sorted(real_breakdown.items(), key=lambda x: -x[1]))}")
    if real_breakdown and sim_breakdown != real_breakdown:
        print("❌ 分组校验和不一致（鸟种标签还原有误，禁止写库）")
        return 2
    print("✅ 分组校验和一致")

    # ---- 自校验：用当前配置参数应精确复现存库评级 ----
    check_res = simulate(metrics, root, CURRENT_MIN_CONF,
                         CURRENT_QUOTA3, CURRENT_QUOTA2)
    check = {k: r.rating for k, r in check_res.items()}
    mismatch = [(k, old[k], check.get(k)) for k in metrics
                if old[k] != check.get(k)]
    if mismatch:
        print(f"\n❌ 自校验失败：{len(mismatch)} 张与现存评级不符（重建有误，禁止写库）")
        for k, o, n in mismatch[:10]:
            print(f"   {k}: DB={o}  复算={n}")
        return 2
    print("✅ 自校验通过：conf=0.7 + 配额20/30 精确复现存库评级")

    # ---- 目标方案模拟 ----
    new_res = simulate(metrics, root, args.min_conf, args.quota3, args.quota2)
    new = {k: r.rating for k, r in new_res.items()}
    changed = [(k, old[k], new[k]) for k in sorted(new)
               if old[k] != new[k]]
    print(f"\n🎯 方案 conf={args.min_conf} + 配额 {args.quota3:.0f}/{args.quota2:.0f}:")
    full_new = dict(old)
    full_new.update(new)
    print(f"   新分布: {dist(full_new, total)}")
    print(f"   变化 {len(changed)} 张:")
    for k, o, n in changed:
        m = metrics[k]
        sp = m.species or "?"
        print(f"   {k}: {o}★ → {n}★  (conf={m.confidence:.0%}, {sp})")

    if not args.execute:
        print("\n（dry-run，未写库。确认后加 --execute 执行）")
        return 0

    # ---- 写库：先备份，再更新 rating + caption 首行（保持浏览器一致）----
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path + f".bak_重定星conf{args.min_conf}_{ts}"
    shutil.copy(db_path, bak)
    print(f"📦 已备份: {os.path.basename(bak)}")

    i18n = get_i18n("zh_CN")  # 现存 caption 为 zh_CN 文案，head 模板保持同语言

    def rewrite_caption(caption: str, key: str) -> str:
        """
        把 caption 里的「最终评分」首行替换为新评级对应文案。

        参数:
            caption (str): 现存 caption（可能为空）
            key (str): 照片前缀（定位定星结果）

        返回:
            str: 重写后的 caption

        Replace the stale rating head line of a caption with the new one.
        """
        res = new_res[key]
        reason = i18n.t(res.reason_key, **res.reason_args)
        head = i18n.t("logs.caption_final", rating=res.rating, reason=reason)
        lines = (caption or "").split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith(("最终评分:", "Final Rating:")):
                lines[i] = head
                return "\n".join(lines)
        return head + (("\n" + caption) if caption else "")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for k, _o, n in changed:
            row = cur.execute("SELECT caption FROM photos WHERE filename=?",
                              (k,)).fetchone()
            new_cap = rewrite_caption(row[0] if row else "", k)
            cur.execute(
                "UPDATE photos SET rating=?, caption=?,"
                " updated_at=CURRENT_TIMESTAMP WHERE filename=?",
                (n, new_cap, k))
        conn.commit()
        print(f"✅ 已更新 {len(changed)} 张的评级与 caption 首行")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
