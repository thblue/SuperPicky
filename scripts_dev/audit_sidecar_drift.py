#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计 sidecar JSON 与 report.db 的漂移（只读，不写任何文件）。

背景：SPBBrowse 浏览器里手工改星（V5.9.7 之前的版本）只写 report.db
+ EXIF + 移动文件，不触发 sidecar 重导出；BirdIndex 只认 sidecar，
于是看到旧星级/旧 library_path。本工具批量找出这类历史遗留漂移。

判定口径与生产导出完全一致：对每张有鸟照片用
core.sidecar_export._export_stamp 计算 DB 行内容哈希，与 sidecar
JSON 落盘的 _export_stamp 比对。不一致 ⇔ "该照片的 DB 行在导出之后
发生过变化"（改星、移动、编辑等），与逐字段比对等价且无遗漏。
DB 严格只读（sqlite mode=ro），照片原文件零接触；修复请用
scripts_dev/reexport_sidecars.py（同款生产导出逻辑，增量重写）。

Usage:
    python scripts_dev/audit_sidecar_drift.py <库目录> [<库目录> ...]

Audit drift between report.db and sidecar JSONs, strictly read-only.
A photo is drifted iff its recomputed export stamp differs from the
one stored in its sidecar JSON — i.e. the DB row changed after the
JSON was written. Fix with reexport_sidecars.py.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.sidecar_export import _export_stamp, _normalize_rel, _sidecar_path
from scripts_dev.reexport_sidecars import ReadOnlySidecarSource

# 每目录最多列出的漂移样例数（全部数量仍会统计）/ max examples per dir
MAX_EXAMPLES = 30


def _classify(row: dict, payload: dict) -> List[str]:
    """
    对一条漂移照片做人类可读的原因归类（星级 / 路径 / 其他字段）。

    library_path 的比对只是提示性归类（生产导出对历史行有 JPG→RAW
    等回退规范化，可能个别误标"其他"）；stamp 不一致本身即权威判定。

    Parameters:
    row (dict): photos 表行
    payload (dict): 现有 sidecar JSON 内容

    Return:
    List[str]: 漂移原因列表（中文描述）
    """
    reasons: List[str] = []
    proc = payload.get("processing") or {}
    photo_sec = payload.get("photo") or {}
    if proc.get("rating") != row.get("rating"):
        reasons.append(
            f"星级 {proc.get('rating')} → {row.get('rating')}")
    db_lib = _normalize_rel(row.get("current_path")
                            or row.get("original_path"))
    if photo_sec.get("library_path") != db_lib:
        reasons.append(
            f"路径 {photo_sec.get('library_path')} → {db_lib}")
    if not reasons:
        reasons.append("其他字段（检测框/指标/元数据）")
    return reasons


def audit_directory(directory: str, examples_out: List[str]) -> dict:
    """
    审计单个照片库目录，把漂移样例追加进 examples_out。

    参数:
    directory (str): 照片库根目录（需含 .superpicky/report.db）
    examples_out (List[str]): 输出参数，样例行追加于此

    返回:
    dict: 各类漂移计数（drifted / missing / stale_nobird / ok / total）

    Audit one library directory; append human-readable examples and
    return drift counters.
    """
    db_path = os.path.join(directory, ".superpicky", "report.db")
    source = ReadOnlySidecarSource(db_path)
    try:
        photos = source.get_all_photos()
        detections = source.get_all_detections(include_polygon=False)
    finally:
        source.close()

    dets_by_filename: dict = {}
    for det in detections:
        dets_by_filename.setdefault(det.get("filename"), []).append(det)

    stats = {"drifted": 0, "missing": 0, "stale_nobird": 0,
             "ok": 0, "total": 0}
    n_shown = 0
    for row in photos:
        prefix = row.get("filename")
        if not prefix:
            continue
        stats["total"] += 1
        path = _sidecar_path(directory, prefix)

        # 无鸟照片本就不该有 sidecar（V5.5 契约），有则属历史残留
        if not row.get("has_bird"):
            if os.path.exists(path):
                stats["stale_nobird"] += 1
                if n_shown < MAX_EXAMPLES:
                    examples_out.append(
                        f"    [无鸟残留] {prefix}（JSON 应删除）")
                    n_shown += 1
            continue

        if not os.path.exists(path):
            stats["missing"] += 1
            if n_shown < MAX_EXAMPLES:
                examples_out.append(f"    [缺 JSON] {prefix}")
                n_shown += 1
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            stats["drifted"] += 1
            if n_shown < MAX_EXAMPLES:
                examples_out.append(f"    [JSON 损坏] {prefix}: {e}")
                n_shown += 1
            continue

        stamp = _export_stamp(row, dets_by_filename.get(prefix, []))
        if payload.get("_export_stamp") == stamp:
            stats["ok"] += 1
            continue
        stats["drifted"] += 1
        if n_shown < MAX_EXAMPLES:
            reasons = "；".join(_classify(row, payload))
            examples_out.append(f"    [漂移] {prefix}：{reasons}")
            n_shown += 1
    return stats


def main(argv: List[str] = None) -> int:
    """
    入口：逐目录审计并汇总打印。

    参数:
    argv (List[str]): 命令行参数（库目录列表）

    返回:
    int: 0 无漂移；1 有漂移（便于脚本化）；2 输入错误
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    if not argv:
        print("用法: python scripts_dev/audit_sidecar_drift.py <库目录> ...")
        return 2

    total_drift = 0
    for directory in argv:
        # 只读 URI 拼接要求绝对路径（相对路径会被当成盘符根）/
        # the read-only URI builder expects an absolute path
        directory = os.path.abspath(directory.replace("\\", "/").rstrip("/"))
        print(f"\n📂 {directory}")
        if not os.path.exists(
                os.path.join(directory, ".superpicky", "report.db")):
            print("   ❌ 未找到 report.db，跳过")
            continue
        examples: List[str] = []
        stats = audit_directory(directory, examples)
        total_drift += stats["drifted"] + stats["missing"] \
            + stats["stale_nobird"]
        print(f"   照片 {stats['total']} 张：一致 {stats['ok']}，"
              f"漂移 {stats['drifted']}，缺 JSON {stats['missing']}，"
              f"无鸟残留 {stats['stale_nobird']}")
        for line in examples:
            print(line)
        if (stats["drifted"] + stats["missing"] + stats["stale_nobird"]
                > len(examples)):
            print(f"    …（其余 {total_drift - len(examples)} 条略）")

    print(f"\n{'✅ 全库一致，无需修复' if total_drift == 0 else f'⚠️ 共 {total_drift} 张需重导出（用 scripts_dev/reexport_sidecars.py 修复）'}")
    return 0 if total_drift == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
