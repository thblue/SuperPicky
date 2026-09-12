#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 配额定星重算引擎（不重跑检测/识鸟）。

CLI `rerate-v2` 子命令与 scripts_dev/rerate_v2_conf_gate.py 共用的唯一实现。
按 photo_processor V2 收尾阶段的同一算法（core.rating_quota），从 report.db
现成字段 + superpicky.log 鸟种标签重建定星输入，对新门槛/配额做全批重算：

- norm_sharpness = head_sharp × ISO 归一化系数（ISO_BASE=800，每翻倍 -5%，下限 0.5）
- topiq          = nima_score（鸟裁剪区 TOPIQ 原始分）
- best_eye       = max(left_eye, right_eye)；beak_vis = beak
- species        = 日志解析的识别标签（含低置信候选，分组恰恰需要它们），
                  回退 DB 鸟种列
- 连拍/相似簇    = DB burst_id + 对入池照片重跑 pHash 相似聚类（只聚入池者，
                  与收尾阶段口径一致）

安全设计 / Safety:
1. 默认 dry-run，只打印分布对比；
2. 写库前双自校验：a) 用「现存评级所用参数」复算必须逐张复现存库评级，
   b) 0.7/当前门槛下池内按鸟种分组必须与真实跑批明细一致——任一不过即拒绝；
3. `--execute` 前自动备份 report.db；
4. 只写 report.db rating/caption 与 sidecar `processing.rating`，零接触照片文件。

用法（CLI）:
    python superpicky_cli.py rerate-v2 <照片目录> [--min-conf 0.4] [--execute]
"""

import json
import math
import os
import re
import shutil
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from core.rating_quota import (
    PhotoMetricsV2,
    RatingV2Result,
    assign_ratings,
    gate_photo,
    get_quota2_for_skill,
    get_quota3_for_skill,
)
from tools.i18n import get_i18n

# 与 photo_processor.PhotoProcessor 类常量一致（ISO 归一化）
ISO_BASE = 800
ISO_PENALTY_FACTOR = 0.05
ISO_MIN_FACTOR = 0.5

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 识别日志行：🐦 Bird ID [027A4921.CR3]: 白头鹎 (97%)...
#             🐦 Low confidence [027A4878.CR3]: 蓝歌鸲 (45% < 50.0%)
# 半角括号+百分号保证不误匹配 Multi-bird / 主鸟重选（全角括号）行。
BIRDID_LINE_RE = re.compile(r"Bird ID \[([^\]]+)\]: (\S+) \(\d+%(?:\s*<[^)]*)?\)")
LOWCONF_LINE_RE = re.compile(
    r"Low confidence \[([^\]]+)\]: (\S+) \(\d+%(?:\s*<[^)]*)?\)")
# V2 定星汇总行里的 3★ 配额：🎯 V2 定星完成: 排序池 44 张, 3星 8 张 (配额20%, ...)
QUOTA3_LINE_RE = re.compile(r"V2 定星完成.*配额(\d+)%")
# 置信守门拒绝原因里的门槛。只匹配**无空格**的紧凑形态「NN%<MM%」：
# 评级守门行是紧凑的（…0★ (置信度26%<40%)，MM = -c 门槛，是想要的）；
# 识鸟低置信行带空格（…Low confidence…(44% < 50.0%)，MM = birdid 采纳
# 阈值，不是想要的）。若放宽为 \s*<\s* 会双语义匹配、结果取决于行序运气
# （2026-09-12 review 发现的潜伏 bug，此前三个日志均碰巧落在正确值）。
CONF_GATE_RE = re.compile(r"(\d+)%<(\d+)%")

# process 收尾写入 report.db meta 表的「本次生效参数」键 / run-param keys
# written by the process post-pass。rerate-v2 按 meta 表 > 日志 > 当前配置
# 的优先级读取：meta 无解析歧义，且 quota2 不入日志、只有 meta 能精确记录。
META_KEY_MIN_CONF = "last_run_min_confidence"
META_KEY_QUOTA3 = "last_run_quota3"
META_KEY_QUOTA2 = "last_run_quota2"


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

    V-cover 视频封面行（filename 以 _vcover 结尾）被排除：封面是固定
    2 星、不参与 V2 配额定星，也无锐度/TOPIQ 指标可重建。

    参数:
        db_path (str): report.db 路径

    返回:
        List[Dict]: 每张照片一行（dict）

    Load all fields needed to rebuild V2 metrics from the report DB.
    V-cover rows (filename ending with _vcover) are excluded: covers carry
    a fixed 2-star rating with no V2 metrics to rebuild.
    """
    from constants import VIDEO_COVER_SUFFIX

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT filename, has_bird, confidence, head_sharp, left_eye, right_eye,"
            " beak, nima_score, is_flying, focus_status, burst_id, iso, rating,"
            " bird_species_en, bird_species_cn, adj_sharpness, adj_topiq"
            " FROM photos ORDER BY filename"
        ).fetchall()
        return [dict(r) for r in rows
                if not (r["filename"] or "").endswith(VIDEO_COVER_SUFFIX)]
    finally:
        conn.close()


def parse_species_from_log(log_path: str) -> Dict[str, str]:
    """
    从 superpicky.log 解析每张照片的识别鸟种（含低置信候选）。

    机制：低置信识别（< birdid 阈值）的鸟种会进内存 file_bird_species 参与
    V2 按种分组配额，但不写 photos 表鸟种列（仅高置信写入）。唯一完整
    持久化载体是运行日志的两类行；中文名作为分组键与 en_name 等价。

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


def load_run_params_from_meta(db_path: str) -> Dict[str, float]:
    """
    从 report.db meta 表读取上次 process 写入的生效参数。

    参数:
        db_path (str): report.db 路径

    返回:
        Dict[str, float]: {"min_conf"/"quota3"/"quota2": 值}，仅含存在的键；
        旧批次（2026-09-12 前）无这些键，返回空 dict，调用方回退日志解析

    Read the effective parameters of the last process run from the meta
    table; empty dict for pre-2026-09-12 batches (caller falls back to
    log parsing).
    """
    params: Dict[str, float] = {}
    try:
        conn = sqlite3.connect(db_path)
        try:
            for key, name in ((META_KEY_MIN_CONF, "min_conf"),
                              (META_KEY_QUOTA3, "quota3"),
                              (META_KEY_QUOTA2, "quota2")):
                row = conn.execute(
                    "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                if row and row[0] not in (None, ""):
                    params[name] = float(row[0])
        finally:
            conn.close()
    except Exception:
        pass
    return params


def parse_last_run_params(log_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    从日志最近一次跑批解析置信门槛与 3★ 配额（纯日志解析，不做回退）。

    解析来源：V2 定星汇总行的 3★ 配额；评级守门行「NN%<MM%」的 MM
    （紧凑形态，见 CONF_GATE_RE 注释——识鸟低置信行带空格不会误匹配）。
    quota2 不入日志，恒为 None，由调用方回退 meta/配置。

    参数:
        log_path (str): superpicky.log 路径

    返回:
        Tuple[Optional[float], Optional[float]]: (min_conf, quota3)；
        日志不存在或未命中为 (None, None)

    Parse (min_conf, quota3) from the most recent run in the log; None
    when absent — fallbacks are the caller's job.
    """
    min_conf: Optional[float] = None
    quota3: Optional[float] = None
    if not os.path.exists(log_path):
        return min_conf, quota3
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            plain = ANSI_RE.sub("", line)
            m = QUOTA3_LINE_RE.search(plain)
            if m:
                quota3 = float(m.group(1))
            for cm in CONF_GATE_RE.finditer(plain):
                # 文件按时间追加，最后一次命中即最近一次跑批的门槛
                min_conf = float(cm.group(2)) / 100.0
    return min_conf, quota3


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
            has_exposure_issue=False,  # 复现前提：批次跑批时曝光检测为关
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
             ) -> Dict[str, RatingV2Result]:
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


def _parse_real_breakdown(log_path: str) -> Dict[str, int]:
    """
    从日志解析真实跑批的「每鸟种 池内」分布行（分组校验和基准）。

    参数:
        log_path (str): superpicky.log 路径

    返回:
        Dict[str, int]: 鸟种名 → 池内张数；解析不到返回空 dict

    Parse the per-species pool breakdown line from the run log.
    """
    breakdown: Dict[str, int] = {}
    if not os.path.exists(log_path):
        return breakdown
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            plain = ANSI_RE.sub("", line)
            if "·" in plain and re.search(r"\d+/\d+", plain):
                found = re.findall(r"(\S+) \d+/(\d+)", plain)
                if found:
                    breakdown = {name: int(cnt) for name, cnt in found}
    return breakdown


def _sync_sidecar_ratings(root: str, changes: Dict[str, int]) -> Tuple[int, int]:
    """
    把变更照片的新星级同步进 sidecar JSON 的 processing.rating（编辑层一致）。

    参数:
        root (str): 照片目录
        changes (Dict[str, int]): 前缀 → 新星级

    返回:
        Tuple[int, int]: (同步成功数, sidecar 缺失数)

    Mirror new ratings into sidecar JSON processing.rating for changed photos.
    """
    synced = missing = 0
    meta_dir = os.path.join(root, ".superpicky", "meta")
    for prefix, rating in changes.items():
        path = os.path.join(meta_dir, prefix + ".json")
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("processing", {})["rating"] = rating
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            synced += 1
        except Exception:
            missing += 1
    return synced, missing


def rerate_directory(directory: str, min_conf: float = 0.4,
                     quota3: Optional[float] = None,
                     quota2: Optional[float] = None,
                     execute: bool = False,
                     current_min_conf: Optional[float] = None,
                     current_quota3: Optional[float] = None,
                     current_quota2: Optional[float] = None,
                     log=print) -> int:
    """
    V2 重定星主入口：自校验 → 模拟目标方案 →（可选）写库。

    参数:
        directory (str): 照片目录
        min_conf (float): 目标置信度门槛（默认 0.4，工作流定档）
        quota3 / quota2 (Optional[float]): 目标配额百分比；None = 跟随配置
        execute (bool): 写库（默认 dry-run）
        current_min_conf / current_quota3 / current_quota2:
            显式指定「现存评级所用参数」覆盖日志解析（自校验复核用）
        log: 打印回调

    返回:
        int: 0 成功；1 前置错误；2 自校验失败

    Full V2 re-rating entry: self-check, simulate, optionally write.
    """
    from advanced_config import get_advanced_config

    root = os.path.normpath(directory)
    db_path = os.path.join(root, ".superpicky", "report.db")
    if not os.path.exists(db_path):
        log(f"❌ 未找到 {db_path}")
        return 1

    adv_config = get_advanced_config()
    rows = load_photos(db_path)
    total = len(rows)
    old = {r["filename"]: int(r["rating"]) for r in rows}
    log(f"📁 {root}")
    log(f"📊 DB 现存分布: {dist(old, total)}")

    log_path = os.path.join(root, "superpicky.log")
    labels = parse_species_from_log(log_path)
    log(f"🏷️ 日志解析到 {len(labels)} 张照片的鸟种标签")

    metrics = build_metrics(rows, labels)
    log(f"🐦 有鸟照片 {len(metrics)} 张（重建 V2 指标）")

    # ---- 分组校验和：现存门槛下的池内分组应与真实跑批明细一致 ----
    real_breakdown = _parse_real_breakdown(log_path)

    # 「现存评级所用参数」解析，逐键取最优来源：
    # CLI 显式覆盖 > meta 表（process 收尾写入，无歧义）> 日志解析 > 当前配置。
    # 旧批次（2026-09-12 前）无 meta，依赖日志/配置回退并告警。
    meta_params = load_run_params_from_meta(db_path)
    log_conf, log_q3 = parse_last_run_params(log_path)

    skill_level = getattr(adv_config, "skill_level", "custom")
    last_conf = float(getattr(adv_config, "min_confidence", 0.5))
    last_q3 = float(get_quota3_for_skill(skill_level, adv_config))
    last_q2 = float(get_quota2_for_skill(skill_level, adv_config))
    src_conf = src_q3 = src_q2 = "配置回退"
    if log_conf is not None:
        last_conf, src_conf = log_conf, "日志"
    if log_q3 is not None:
        last_q3, src_q3 = log_q3, "日志"
    if "min_conf" in meta_params:
        last_conf, src_conf = meta_params["min_conf"], "meta表"
    if "quota3" in meta_params:
        last_q3, src_q3 = meta_params["quota3"], "meta表"
    if "quota2" in meta_params:
        last_q2, src_q2 = meta_params["quota2"], "meta表"
    elif "quota3" not in meta_params:
        log(f"⚠️ quota2 未入日志且无 meta（2026-09-12 前的批次），"
            f"按当前配置 {last_q2:.0f}% 自校验；不符时用 --current-quota2 指定历史值")
    if src_conf == "配置回退":
        log(f"⚠️ meta/log 均无置信门槛，回退当前配置 {last_conf}")
    if src_q3 == "配置回退":
        log(f"⚠️ meta/log 均无 3★ 配额，回退当前配置 {last_q3:.0f}%")
    if current_min_conf is not None:
        last_conf, src_conf = current_min_conf, "CLI覆盖"
    if current_quota3 is not None:
        last_q3, src_q3 = current_quota3, "CLI覆盖"
    if current_quota2 is not None:
        last_q2, src_q2 = current_quota2, "CLI覆盖"
    log(f"🔍 现存评级参数（来源 conf={src_conf} q3={src_q3} q2={src_q2}）: "
        f"conf={last_conf} quota3={last_q3:.0f}% quota2={last_q2:.0f}%")

    pool070 = [m for m in metrics.values()
               if gate_photo(m, min_confidence=last_conf) is None]
    sim_breakdown: Dict[str, int] = {}
    for m in pool070:
        sim_breakdown[m.species or "未识别"] = \
            sim_breakdown.get(m.species or "未识别", 0) + 1
    log(f"🔍 池内分组: 模拟={dict(sorted(sim_breakdown.items(), key=lambda x: -x[1]))}")
    log(f"           真实={dict(sorted(real_breakdown.items(), key=lambda x: -x[1]))}")
    if real_breakdown and sim_breakdown != real_breakdown:
        log("❌ 分组校验和不一致（鸟种标签还原有误，禁止写库）")
        return 2
    log("✅ 分组校验和一致")

    # ---- 自校验：现存参数应精确复现存库评级（人工改星作为覆盖层豁免） ----
    check_res = simulate(metrics, root, last_conf, last_q3, last_q2)
    check = {k: r.rating for k, r in check_res.items()}
    mismatch = [(k, old[k], check.get(k)) for k in metrics
                if old[k] != check.get(k)]
    # 浏览器里的人工改星（_on_rating_changed 直写 DB）是有意覆盖管线输出
    # 的合法操作，少量不符按「人工评级」处理：重算时保留、不重写 caption/
    # sidecar；超过容忍度则更可能是重建缺陷，拒绝写库。
    manual_tolerance = max(3, int(len(metrics) * 0.1))
    manual_keys: set = set()
    if mismatch:
        if len(mismatch) <= manual_tolerance:
            manual_keys = {k for k, _o, _n in mismatch}
            log(f"ℹ️ {len(mismatch)} 张与管线复算不符（≤容忍度 {manual_tolerance}），"
                "判定为人工改星，重算将保留这些人工评级:")
            for k, o, n in mismatch:
                log(f"   {k}: 管线复算={n}，保留人工评级 {o}★")
        else:
            log(f"\n❌ 自校验失败：{len(mismatch)} 张与现存评级不符"
                f"（超过人工改星容忍度 {manual_tolerance}，疑似重建缺陷，禁止写库）")
            for k, o, n in mismatch[:10]:
                log(f"   {k}: DB={o}  复算={n}")
            log("   若现存评级确由其他参数产生，请用 --current-conf/--current-quota3/"
                "--current-quota2 显式指定后重试。")
            return 2
    else:
        log(f"✅ 自校验通过：conf={last_conf} + 配额 {last_q3:.0f}/{last_q2:.0f} "
            "精确复现存库评级")

    # ---- 目标方案模拟 ----
    eff_q3 = quota3 if quota3 is not None else float(get_quota3_for_skill(
        getattr(adv_config, "skill_level", "custom"), adv_config))
    eff_q2 = quota2 if quota2 is not None else float(get_quota2_for_skill(
        getattr(adv_config, "skill_level", "custom"), adv_config))
    new_res = simulate(metrics, root, min_conf, eff_q3, eff_q2)
    new = {k: r.rating for k, r in new_res.items()}
    # 人工改星的照片保留存量评级，不参与重算变更
    changed = [(k, old[k], new[k]) for k in sorted(new)
               if k not in manual_keys and old[k] != new[k]]
    log(f"\n🎯 方案 conf={min_conf} + 配额 {eff_q3:.0f}/{eff_q2:.0f}:")
    full_new = dict(old)
    for k, v in new.items():
        if k not in manual_keys:  # 人工改星按存量评级计入展示
            full_new[k] = v
    log(f"   新分布: {dist(full_new, total)}")
    log(f"   变化 {len(changed)} 张:")
    for k, o, n in changed:
        m = metrics[k]
        sp = m.species or "?"
        log(f"   {k}: {o}★ → {n}★  (conf={m.confidence:.0%}, {sp})")

    if not execute:
        log("\n（dry-run，未写库。确认后加 --execute 执行）")
        return 0

    # ---- 写库：先备份，再更新 rating/caption/sidecar ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path + f".bak_rerate-v2_conf{min_conf}_{ts}"
    shutil.copy(db_path, bak)
    log(f"📦 已备份: {os.path.basename(bak)}")

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
        log(f"✅ 已更新 {len(changed)} 张的评级与 caption 首行")
    finally:
        conn.close()

    synced, missing = _sync_sidecar_ratings(
        root, {k: n for k, _o, n in changed})
    log(f"✅ sidecar processing.rating 同步 {synced} 张"
        + (f"（{missing} 张无 sidecar 跳过）" if missing else ""))
    return 0
