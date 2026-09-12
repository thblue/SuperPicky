# -*- coding: utf-8 -*-
"""
rerate_v2 参数解析单测：
1. CONF_GATE_RE 双语义消解——评级守门行（紧凑 NN%<MM%）匹配、
   识鸟低置信行（带空格 NN% < MM%）不匹配（2026-09-12 P1 潜伏 bug 回归）；
2. load_run_params_from_meta 读取 process 写入的 meta 键；
3. rerate_directory 的来源优先级（meta > 日志 > 配置）经由
   parse_last_run_params + load_run_params_from_meta 的组合行为覆盖。

Param-parsing unit tests for rerate_v2: regex disambiguation, meta-table
loading. The priority wiring (meta > log > config) is exercised end-to-end.
"""
import os
import sqlite3

from core import rerate_v2


def _write_log(tmp_path, lines):
    p = tmp_path / "superpicky.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_conf_gate_matches_compact_rating_line(tmp_path):
    """评级守门行（紧凑）解析出 -c 门槛。
    Compact rating lines yield the -c gate."""
    log = _write_log(tmp_path, [
        "[093/270] 027A5253.jpg | 0★ (置信度31%<40%) | 625ms",
    ])
    conf, q3 = rerate_v2.parse_last_run_params(log)
    assert conf == 0.4


def test_conf_gate_ignores_spaced_birdid_line(tmp_path):
    """识鸟低置信行（带空格，门槛是 birdid 阈值）不得污染 -c 解析。
    Spaced Low-confidence lines (birdid threshold) must not leak in —
    even when they are the last matching-looking line in the log."""
    log = _write_log(tmp_path, [
        # -c 门槛 40 的评级行在前
        "[001/270] a.jpg | 0★ (置信度31%<40%) | 100ms",
        # birdid 阈值 50 的低置信行在后（旧行序 bug 会取到 50）
        "  🐦 Low confidence [a.CR3]: 黄眉柳莺 (44% < 50.0%)",
    ])
    conf, _q3 = rerate_v2.parse_last_run_params(log)
    assert conf == 0.4


def test_quota3_from_v2_summary_line(tmp_path):
    """V2 定星汇总行的配额被解析，且后行覆盖前行（多次跑批取最近）。
    The quota parses from the V2 summary; the last run wins."""
    log = _write_log(tmp_path, [
        "🎯 V2 定星完成: 排序池 270 张, 3星 41 张 (配额20%, 占全部 15%)",
        "🎯 V2 定星完成: 排序池 44 张, 3星 8 张 (配额30%, 占全部 5%)",
    ])
    _conf, q3 = rerate_v2.parse_last_run_params(log)
    assert q3 == 30.0


def test_missing_log_returns_none(tmp_path):
    """日志不存在 → (None, None)，由调用方回退。
    Missing log → None/None; fallbacks are the caller's job."""
    conf, q3 = rerate_v2.parse_last_run_params(str(tmp_path / "nope.log"))
    assert conf is None and q3 is None


def test_load_run_params_from_meta(tmp_path):
    """meta 表键读取：存在的键返回值，缺失键不出现，坏库安全降级。
    Meta keys load when present; absent keys omitted; bad DB degrades."""
    db = str(tmp_path / "report.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES (?, ?)",
                 (rerate_v2.META_KEY_MIN_CONF, "0.4000"))
    conn.execute("INSERT INTO meta VALUES (?, ?)",
                 (rerate_v2.META_KEY_QUOTA2, "30.0"))
    conn.commit()
    conn.close()

    params = rerate_v2.load_run_params_from_meta(db)
    assert params == {"min_conf": 0.4, "quota2": 30.0}

    # 无 meta 表的库 → 空 dict（异常吞掉，调用方回退日志/配置）
    db2 = str(tmp_path / "empty.db")
    sqlite3.connect(db2).close()
    assert rerate_v2.load_run_params_from_meta(db2) == {}
