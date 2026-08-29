#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_fix_mainspecies — 把 sidecar 的 main_species 数组对齐检测框真值

场景：多鸟编辑器把主鸟写进 sidecar 顶层 main_species（对象条目
{cn, en, scientific, bird_index}），而批量改种/整种删除的 sidecar 同步
（rename_species_in_sidecar / mark_species_deleted_in_sidecar）在 2026-08
之前只处理字符串条目——对象条目里的旧鸟名残留，BirdIndex（取种优先级
main_species 最高）会一直显示旧名，表现为「站内改种没生效」。

本工具逐照片校验 main_species 与同文件 detections 的一致性（不变式：
主鸟条目必须等于其引用的存活检测框的物种）：

- 条目引用的检测框已软删（deleted=true）→ 移除该条目；
- 条目名字与存活检测框的物种不一致 → 改写为检测框的当前物种；
- 条目（含字符串形态）在存活检测框中找不到任何同名框 → 移除该条目。

只改 sidecar JSON（原子写），不动 report.db、不动照片文件；幂等可重复
执行。默认 dry-run 只报告，--apply 才落盘；修完在 BirdIndex 触发一次
扫描即可回显。

用法:
    python spb_fix_mainspecies.py <照片库目录> [目录2 ...] [--apply]
    python spb_fix_mainspecies.py -f 清单.txt [--apply]

Align sidecar main_species arrays with their detection boxes (repairs
stale object-form entries left by pre-fix bulk renames/deletes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sidecar_export import _atomic_write_json


def _entry_names(entry: object) -> Optional[Tuple[str, str, str]]:
    """
    取 main_species 对象条目的三个名字维度；非对象返回 None。

    返回:
    Optional[Tuple[str, str, str]]: (cn, en, scientific)，缺维度为空串
    """
    if not isinstance(entry, dict):
        return None
    return (str(entry.get("cn") or ""), str(entry.get("en") or ""),
            str(entry.get("scientific") or ""))


def _det_species(det: dict) -> Tuple[str, str, str]:
    """取检测框 species 的三个名字维度（缺为空串）。"""
    sp = det.get("species") or {}
    return (str(sp.get("cn") or ""), str(sp.get("en") or ""),
            str(sp.get("scientific") or ""))


def _same_name(a: Tuple[str, str, str], b: Tuple[str, str, str]) -> bool:
    """两组名字任一非空维度相等即视为同种。"""
    return any(x and x == y for x, y in zip(a, b))


def _plan_fixes(payload: dict) -> List[str]:
    """
    计算一张照片 main_species 的修正动作（直接改 payload，不写盘）。

    参数:
    payload (dict): sidecar JSON 全文

    返回:
    List[str]: 动作描述列表（空列表 = 无需修正）
    """
    main = payload.get("main_species")
    if not isinstance(main, list) or not main:
        return []
    dets = [d for d in payload.get("detections") or []
            if isinstance(d, dict)]
    live_by_index = {d.get("index"): d for d in dets if not d.get("deleted")}
    live_dets = list(live_by_index.values())

    actions: List[str] = []
    new_main = []
    for entry in main:
        if isinstance(entry, str):
            ref = (entry.strip(), "", "")
        else:
            ref = _entry_names(entry)
            if ref is None:
                new_main.append(entry)
                continue
        # 优先按 bird_index 定位引用框；索引指向的框已是别的种或已删
        # 时，回退按名字在存活框里找
        det = None
        if isinstance(entry, dict) and entry.get("bird_index") is not None:
            det = live_by_index.get(entry.get("bird_index"))
            if det is not None and not _same_name(ref, _det_species(det)):
                det = None
        if det is None:
            det = next((d for d in live_dets
                        if _same_name(ref, _det_species(d))), None)
        if det is None:
            actions.append(f"移除条目（无存活同名框）: {ref[0] or ref[1]}")
            continue
        det_names = _det_species(det)
        if isinstance(entry, dict):
            if ref != det_names:
                actions.append(
                    f"改名条目: {ref[0] or ref[1]} → "
                    f"{det_names[0] or det_names[1]}")
                fresh = {k: v for k, v in entry.items()
                         if k not in ("cn", "en", "scientific")}
                fresh.update({"cn": det_names[0] or None,
                              "en": det_names[1] or None,
                              "scientific": det_names[2] or None})
                new_main.append(fresh)
            else:
                new_main.append(entry)
        else:  # 字符串条目指向存活框：保持字符串，改写为该框中文名
            if entry.strip() != (det_names[0] or entry.strip()):
                actions.append(f"改名字符串条目: {entry} → {det_names[0]}")
                new_main.append(det_names[0])
            else:
                new_main.append(entry)
    if actions:
        payload["main_species"] = new_main
    return actions


def _fix_library(directory: str, apply: bool) -> int:
    """
    校验并修正一个照片库的全部 sidecar。

    参数:
    directory (str): 照片库目录（含 .superpicky/meta/）
    apply (bool): True 落盘，False 仅报告

    返回:
    int: 修正（或待修正）的照片数
    """
    meta_dir = os.path.join(directory, ".superpicky", "meta")
    if not os.path.isdir(meta_dir):
        print(f"  ⚠️ 跳过（无 .superpicky/meta）: {directory}")
        return 0
    fixed = 0
    for name in sorted(os.listdir(meta_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(meta_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  ⚠️ JSON 读取失败 {name}: {e}")
            continue
        if not isinstance(payload, dict):
            continue
        actions = _plan_fixes(payload)
        if not actions:
            continue
        fixed += 1
        print(f"  {name}:")
        for a in actions:
            print(f"    - {a}")
        if apply:
            try:
                _atomic_write_json(path, payload)
            except OSError as e:
                print(f"    ⚠️ 写入失败: {e}")
    return fixed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 sidecar main_species 对齐检测框真值"
                    "（默认 dry-run，--apply 执行）")
    parser.add_argument("directories", nargs="*", help="照片库目录（可多个）")
    parser.add_argument("-f", "--file", help="目录清单文件（每行一个目录）")
    parser.add_argument("--apply", action="store_true",
                        help="真正执行（默认只报告）")
    args = parser.parse_args()

    dirs: List[str] = list(args.directories)
    if args.file:
        with open(args.file, "r", encoding="utf-8-sig") as f:
            dirs.extend(line.strip() for line in f if line.strip())
    if not dirs:
        parser.print_help()
        return 1

    mode = "执行" if args.apply else "dry-run（只报告，不改任何文件）"
    print(f"=== spb_fix_mainspecies: {len(dirs)} 个目录, 模式: {mode} ===")
    total = 0
    for d in dirs:
        print(f"[{d}]")
        total += _fix_library(d, args.apply)
    tail = "已修正" if args.apply else "待修正（加 --apply 执行）"
    print(f"=== 完成: {total} 张照片{tail} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
