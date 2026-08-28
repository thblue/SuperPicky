#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_dedupe_jpg — 删除与 RAW 同名的 JPG（RAW+JPG 双格式拍摄的冗余副本清理）。

动机 / Motivation:
    相机以 RAW+JPG 双格式拍摄时，每个快门会产出 `IMG_xxxx.CR3` 与 `IMG_xxxx.JPG`
    两个文件。当筛选/修图流程只依赖 RAW 时，同名 JPG 属于冗余副本，占用双倍
    空间。本脚本递归扫描根目录下的"叶子目录"（不含子目录的目录节点），凡同一
    文件名前缀同时存在 JPG 与 CR2/CR3 的，将 JPG 列入待删清单。

行为 / Behavior:
    1. 递归扫描根目录；隐藏目录（如 .superpicky）不进入，也不计作子目录
       （即只含隐藏子目录的目录仍视为叶子）；
    2. 默认仅处理叶子目录，--all-dirs 可改为处理所有目录；
    3. 前缀与扩展名比较均不区分大小写：.jpg/.jpeg 对 .cr2/.cr3；
    4. 先打印完整待删清单（含每个 JPG 对应保留的 RAW 与体积），交互输入 y
       确认后才删除；--dry-run 只看清单不询问，--yes 跳过询问直接删；
    5. 删除前复查同前缀 RAW 仍在位，RAW 缺失则跳过该 JPG 并告警，绝不误删。

用法 / Usage:
    python spb_dedupe_jpg.py <根目录>             # 预览清单 + 交互确认删除
    python spb_dedupe_jpg.py <根目录> --dry-run   # 仅预览，不询问不删除
    python spb_dedupe_jpg.py <根目录> --yes       # 免确认（脚本化批处理，慎用）

安全设计 / Safety:
    - 默认不输入 y 不删任何文件；
    - 只删与 CR2/CR3 同前缀的 .jpg/.jpeg，其余文件一概不动；
    - 清单生成到实际删除之间若对应 RAW 被移走，该 JPG 自动跳过。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

# 待删的 JPEG 扩展名（小写比较）/ JPEG extensions eligible for deletion
JPG_EXTENSIONS = ('.jpg', '.jpeg')
# 视为 RAW 主文件的 Canon 扩展名（小写比较）/ RAW extensions that keep JPG alive
RAW_EXTENSIONS = ('.cr2', '.cr3')


@dataclass
class DeletePlan:
    """
    一个待删 JPG 的计划项 / plan entry for one JPG to delete.

    属性 / Attributes:
        jpg_path (str): 待删 JPG 绝对路径 / absolute path of the JPG
        raw_names (List[str]): 同前缀 RAW 文件名 / basenames of matching RAWs
        size (int): JPG 体积（字节）/ JPG size in bytes
    """
    jpg_path: str
    raw_names: List[str] = field(default_factory=list)
    size: int = 0


def _force_utf8_stdio() -> None:
    """
    在 Windows 控制台强制 stdout/stderr 使用 UTF-8，避免中文清单乱码。

    Force UTF-8 on stdout/stderr so the Chinese plan list prints without
    mojibake on Windows consoles.
    """
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8')
            except (OSError, ValueError):
                pass


def _format_size(num_bytes: int) -> str:
    """
    字节数转人类可读大小。

    参数:
    num_bytes (int): 字节数

    返回:
    str: 如 "3.2 MB" 的可读大小

    Human-readable byte size, e.g. "3.2 MB".
    """
    value = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1024.0:
            return f"{int(value)} B" if unit == 'B' else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _scan_directory(directory: str) -> List[DeletePlan]:
    """
    扫描单个目录，找出与 CR2/CR3 同前缀的 JPG（只读，不写任何东西）。

    参数:
    directory (str): 待扫描目录绝对路径

    返回:
    List[DeletePlan]: 该目录内所有"存在同前缀 RAW"的 JPG 计划项

    Scan one directory for JPGs sharing a stem with a CR2/CR3 (read-only).
    """
    stem_jpgs: Dict[str, List[str]] = {}
    stem_raws: Dict[str, List[str]] = {}
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        print(f"[跳过] 无法读取目录 {directory}: {exc}")
        return []
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        stem, ext = os.path.splitext(entry.name)
        key = stem.lower()
        if ext.lower() in JPG_EXTENSIONS:
            stem_jpgs.setdefault(key, []).append(entry.name)
        elif ext.lower() in RAW_EXTENSIONS:
            stem_raws.setdefault(key, []).append(entry.name)

    plans: List[DeletePlan] = []
    for stem_key, jpg_names in sorted(stem_jpgs.items()):
        raw_names = stem_raws.get(stem_key)
        if not raw_names:
            continue
        for jpg_name in sorted(jpg_names):
            jpg_path = os.path.join(directory, jpg_name)
            try:
                size = os.path.getsize(jpg_path)
            except OSError:
                size = 0
            plans.append(DeletePlan(jpg_path=jpg_path,
                                    raw_names=sorted(raw_names),
                                    size=size))
    return plans


def _iter_target_dirs(root: str, all_dirs: bool) -> Iterator[str]:
    """
    遍历根目录下需要扫描的目录（生成器）。

    默认只产出叶子目录（不含可见子目录的目录）；隐藏目录（如 .superpicky）
    既不下钻、也不计作子目录。all_dirs=True 时产出除隐藏目录外的所有目录。

    参数:
    root (str): 根目录绝对路径
    all_dirs (bool): True 时扫描所有可见目录而非仅叶子

    Yield directories to scan: leaf-only by default (hidden subdirs are
    neither entered nor counted as children), all visible dirs if all_dirs.
    """
    for dirpath, dirnames, _filenames in os.walk(root):
        # 原地剪掉隐藏目录：不下钻，也不让它们阻断"叶子"判定
        # Prune hidden dirs in place: never descend, and don't let them
        # disqualify a directory from being a leaf.
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        if all_dirs or not dirnames:
            yield dirpath


def _collect_plans(root: str, all_dirs: bool) -> Tuple[List[DeletePlan], int]:
    """
    汇总所有目标目录的待删计划。

    参数:
    root (str): 根目录绝对路径
    all_dirs (bool): 是否扫描所有目录（而非仅叶子）

    返回:
    Tuple[List[DeletePlan], int]: (按路径排序的待删清单, 实际扫描目录数)

    Collect delete plans from every target directory under root.
    """
    plans: List[DeletePlan] = []
    scanned = 0
    for directory in _iter_target_dirs(root, all_dirs):
        scanned += 1
        plans.extend(_scan_directory(directory))
    plans.sort(key=lambda p: p.jpg_path.lower())
    return plans, scanned


def _print_plans(plans: List[DeletePlan], scanned: int) -> None:
    """
    按目录分组打印完整待删清单。

    参数:
    plans (List[DeletePlan]): 待删清单
    scanned (int): 实际扫描的目录数

    Print the full deletion plan, grouped by directory.
    """
    by_dir: Dict[str, List[DeletePlan]] = {}
    for plan in plans:
        by_dir.setdefault(os.path.dirname(plan.jpg_path), []).append(plan)

    print("=" * 72)
    print(f"待删除 JPG 清单（共 {len(plans)} 个，分布在 {len(by_dir)} 个目录）：")
    for directory in sorted(by_dir, key=str.lower):
        group = by_dir[directory]
        group_bytes = sum(p.size for p in group)
        print(f"\n[{directory}]  {len(group)} 个 / {_format_size(group_bytes)}")
        for plan in group:
            raw_list = ', '.join(plan.raw_names)
            print(f"  - {os.path.basename(plan.jpg_path)}"
                  f"  ({_format_size(plan.size)})  ← 保留 RAW: {raw_list}")
    total = sum(p.size for p in plans)
    print()
    print("=" * 72)
    print(f"扫描目录 {scanned} 个；待删 JPG {len(plans)} 个，"
          f"可释放 {_format_size(total)}。")


def _execute_deletions(plans: List[DeletePlan]) -> Tuple[int, int, int, List[str]]:
    """
    逐条删除清单中的 JPG，删除前复查同前缀 RAW 仍在位。

    参数:
    plans (List[DeletePlan]): 已确认的待删清单

    返回:
    Tuple[int, int, int, List[str]]: (成功数, 跳过/失败数, 释放字节数, 问题明细)

    Delete each planned JPG after re-verifying its RAW still exists.
    """
    deleted = 0
    skipped = 0
    freed = 0
    problems: List[str] = []
    for plan in plans:
        directory = os.path.dirname(plan.jpg_path)
        raw_alive = any(os.path.exists(os.path.join(directory, name))
                        for name in plan.raw_names)
        if not raw_alive:
            skipped += 1
            problems.append(f"[跳过] 同前缀 RAW 已不在位，不动 JPG: {plan.jpg_path}")
            continue
        try:
            os.remove(plan.jpg_path)
            deleted += 1
            freed += plan.size
        except OSError as exc:
            skipped += 1
            problems.append(f"[失败] {plan.jpg_path}: {exc}")
    return deleted, skipped, freed, problems


def main(argv: Optional[List[str]] = None) -> int:
    """
    入口：解析参数 → 扫描 → 打印清单 → 确认 → 删除。

    参数:
    argv (Optional[List[str]]): 命令行参数（缺省取 sys.argv[1:]）

    返回:
    int: 退出码（0 正常结束或无事可做；1 存在跳过/失败；2 参数/路径错误）

    Entry point: parse args, scan, print plan, confirm, then delete.
    """
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog='spb_dedupe_jpg',
        description='递归删除叶子目录中与 CR2/CR3 同名的 JPG（先预览清单，确认后才删）。')
    parser.add_argument('root', help='要处理的根目录')
    parser.add_argument('--all-dirs', action='store_true',
                        help='扫描所有目录（默认仅处理叶子目录）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印待删清单，不询问、不删除')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='跳过交互确认直接删除（批处理慎用）')
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[错误] 根目录不存在或不是目录: {root}")
        return 2

    plans, scanned = _collect_plans(root, args.all_dirs)
    if not plans:
        print(f"扫描 {scanned} 个目录，未发现与 RAW 同名的 JPG，无需处理。")
        return 0

    _print_plans(plans, scanned)

    if args.dry_run:
        print("--dry-run：仅预览，未删除任何文件。")
        return 0

    if not args.yes:
        try:
            answer = input(f"\n确认删除以上 {len(plans)} 个 JPG? (y=删除 / 其他=取消): ")
        except (EOFError, KeyboardInterrupt):
            answer = ''
        if answer.strip().lower() not in ('y', 'yes'):
            print("已取消，未删除任何文件。")
            return 0

    deleted, skipped, freed, problems = _execute_deletions(plans)
    for line in problems:
        print(line)
    print(f"完成：删除 {deleted} 个，跳过/失败 {skipped} 个，"
          f"释放 {_format_size(freed)}。")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
