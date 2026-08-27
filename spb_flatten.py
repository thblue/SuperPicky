#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_flatten — 把已按「鸟种/星级/burst_」整理过的照片库恢复为扁平结构。

动机 / Motivation:
    识别、评分、召回、多鸟编辑的结果都完整落在 report.db + sidecar JSON 里，
    文件系统的鸟种/星级目录只是展示形态之一。切到 flat 布局后，希望把历史
    整理过的库也还原回处理前的扁平结构：照片回到首次处理时所在目录
    （photos.original_path），文件名不变，识别结果全部保留。

行为 / Behavior（对每个含 report.db 的"原子目录"独立执行）:
    1. 读取 photos 表，凡 current_path 与 original_path 目录不同的照片，
       连同同前缀伴生文件（.xmp / 伴随 .jpg 等）一并移回 original_path 所在目录；
    2. 回写 DB：current_path / temp_jpeg_path 指向新位置，original_path 不动；
    3. 清空 burst_id / burst_position（burst_ 目录消失后浏览器本来也会重算）；
    4. 增量重导出 sidecar JSON（library_path 随 current_path 更新，人工 edits 原样保留）；
    5. 视频恢复原名：`鸟种_YYYYMMDD_原名.MP4` → `原名.MP4`（原名本就保留在尾部），
       位于鸟种子目录里的视频同时移回原子目录根；
    6. 清掉因移动而变空的整理目录（只删本次移出过文件的目录及其空父级）；
    7. 全程先 dry-run 预览，--execute 才真正执行；执行前写撤销清单
       .superpicky/flatten_undo_<时间戳>.json（含每条 移动前→移动后 路径）。

用法 / Usage:
    python spb_flatten.py <目录> [<目录>...]            # dry-run 预览
    python spb_flatten.py <目录> --execute              # 正式执行
    python spb_flatten.py "//NAS/BAK2/2024.1 江西" --execute

安全设计 / Safety:
    - 目标已存在同名文件时跳过该照片并上报（绝不覆盖）；
    - 源文件缺失但目标已在位 → 视为已恢复，仅补写 DB（幂等可重跑）；
    - .superpicky/ 隐藏目录与其中的缓存、meta 永不移动删除；
    - 仅在 --execute 时写文件系统与 DB。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 允许脚本从仓库根直接运行 / allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.report_db import ReportDB                       # noqa: E402
from tools.merged_report_db import find_processed_subdirs  # noqa: E402
from core.sidecar_export import export_directory_sidecars  # noqa: E402
from constants import RATING_FOLDER_NAMES, RATING_FOLDER_NAMES_EN  # noqa: E402

# 视频/伴生文件扩展名（小写比较）/ extensions matched case-insensitively
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mpg', '.mpeg', '.m4v',
                    '.mts', '.m2ts', '.wmv')
# 视频改名模板 `鸟种(_鸟种)*_YYYYMMDD_原名.ext` 的日期段 / date segment in
# the video rename template produced by tools/video_organizer.py
VIDEO_DATE_SEG = re.compile(r'^(?P<head>.+)_(?P<date>\d{8})_(?P<orig>.+)$')

# 整理器会创建的星级目录名集合（中英）/ rating folder names (zh + en)
_RATING_DIR_NAMES = set(RATING_FOLDER_NAMES.values()) | set(RATING_FOLDER_NAMES_EN.values())


@dataclass
class PhotoPlan:
    """单张照片的还原计划 / restore plan for one photo."""
    prefix: str                       # 库主键（文件名去扩展名）/ DB key
    src_dir: str                      # 现位置目录（相对原子目录）/ current dir (relative)
    dst_dir: str                      # 目标目录（相对原子目录）/ target dir (relative)
    master_cur: str                   # 现主文件相对路径 / current master rel path
    files: List[str] = field(default_factory=list)   # 现存待移文件名（含伴生）
    status: str = 'move'              # move | already | conflict | missing
    detail: str = ''


@dataclass
class VideoPlan:
    """单个视频的还原计划 / restore plan for one video file."""
    path: str                         # 现相对路径 / current rel path
    new_name: str                     # 恢复后的文件名（不含目录）/ restored basename
    to_root: bool                     # 是否需要移回原子目录根 / move up to atomic root
    status: str = 'rename'            # rename | conflict
    detail: str = ''


def _rel_parts(rel: str) -> List[str]:
    """把库内相对路径按两种分隔符拆成目录段。/ Split rel path on \\/."""
    return re.split(r'[\\/]+', rel) if rel else []


def _join_rel(*parts: str) -> str:
    """用当前系统分隔符拼接库内相对路径。/ Join rel path with os.sep."""
    return os.path.join(*parts) if parts else ''


def _plan_photos(directory: str, db: ReportDB, log) -> List[PhotoPlan]:
    """
    为一个原子目录生成照片还原计划（不写任何东西）。

    参数:
    directory (str): 原子目录绝对路径
    db (ReportDB): 已打开的库
    log: 日志函数

    返回:
    List[PhotoPlan]: 每张需要处理的照片的计划（含异常状态项）

    Build the per-photo restore plan for one atomic directory (no writes).
    """
    plans: List[PhotoPlan] = []
    for row in db.get_all_photos():
        prefix = row.get('filename') or ''
        orig = row.get('original_path') or ''
        cur = row.get('current_path') or ''
        if not prefix or not orig:
            continue
        # current_path 停留在 .superpicky 缓存（未经整理的库）→ 照片本来就没动过
        if not cur or any(p == '.superpicky' for p in _rel_parts(cur)):
            continue

        src_dir = os.path.dirname(cur)
        dst_dir = os.path.dirname(orig)
        if src_dir == dst_dir:
            continue  # 已扁平 / already flat

        stem = os.path.splitext(os.path.basename(cur))[0]
        # 原始主文件名兜底：original_path 优先 RAW，stem 应一致
        plan = PhotoPlan(prefix=prefix, src_dir=src_dir, dst_dir=dst_dir,
                         master_cur=cur)

        src_abs_dir = os.path.join(directory, src_dir)
        dst_abs_dir = os.path.join(directory, dst_dir)
        dst_master_abs = os.path.join(directory, orig)

        if os.path.isfile(dst_master_abs):
            # 目标已存在：源也在 → 冲突不覆盖；源没了 → 已恢复，带走滞留伴生
            src_master_abs = os.path.join(directory, cur)
            if os.path.isfile(src_master_abs):
                plan.status = 'conflict'
                plan.detail = f'目标已存在 {orig} 且源仍在，跳过'
            else:
                plan.status = 'already'
                plan.detail = '目标在位、源缺失，视为已恢复'
                if os.path.isdir(src_abs_dir):
                    for fn in os.listdir(src_abs_dir):
                        if os.path.splitext(fn)[0] == stem and not fn.startswith('.'):
                            plan.files.append(fn)
            plans.append(plan)
            continue

        # 收集同前缀的全部现存文件（主文件 + .xmp / 伴随 jpg 等）
        if os.path.isdir(src_abs_dir):
            for fn in os.listdir(src_abs_dir):
                if os.path.splitext(fn)[0] == stem and not fn.startswith('.'):
                    plan.files.append(fn)
        if not any(os.path.isfile(os.path.join(src_abs_dir, f))
                   for f in (os.path.basename(cur),)):
            plan.status = 'missing'
            plan.detail = f'源主文件不存在: {cur}'
        plans.append(plan)
    return plans


def _scan_videos(directory: str) -> List[VideoPlan]:
    """
    扫描原子目录内的已改名视频，生成还原原名计划。

    仅匹配 `..._YYYYMMDD_原名.ext` 模板（video_organizer 的改名产物）；
    未被改名的视频原样不动。位于子目录里的视频同时计划移回原子目录根
    （video_organizer 的落点就是 源目录/鸟种名/，源目录即原子目录根）。

    Scan renamed videos and plan name restoration (in-place, or back to
    the atomic root when nested in a sub-directory).
    """
    plans: List[VideoPlan] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            stem = os.path.splitext(fn)[0]
            m = VIDEO_DATE_SEG.match(stem)
            if not m or not m.group('orig'):
                continue  # 未按模板改名 / not a renamed video
            rel = os.path.relpath(dirpath, directory)
            vp = VideoPlan(path=os.path.join(rel, fn) if rel != '.' else fn,
                           new_name=f"{m.group('orig')}{os.path.splitext(fn)[1]}",
                           to_root=(rel != '.'))
            dst_abs = os.path.join(directory, vp.new_name) if vp.to_root \
                else os.path.join(dirpath, vp.new_name)
            if os.path.exists(dst_abs):
                vp.status = 'conflict'
                vp.detail = f'目标已存在 {os.path.basename(dst_abs)}'
            plans.append(vp)
    return plans


def _quarantine(directory: str, src_abs: str, src_rel: str,
                undo: List[dict], log) -> None:
    """
    把与目标重名的旧伴生文件隔离到 .superpicky/flatten_orphans/。

    多轮整理/改星会在历史目录留下同名 .xmp 旧副本（现行副本随主文件
    已在新位置）。直接删除不可逆，隔离区保留原相对路径，可人工找回。

    Quarantine a stale same-name companion (e.g., old .xmp copies left in
    earlier rating folders) under .superpicky/flatten_orphans/, keeping
    its relative path, instead of overwriting or deleting.
    """
    dest = os.path.join(directory, '.superpicky', 'flatten_orphans', src_rel)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            os.remove(src_abs)  # 隔离区已有同内容副本时才直接删 / dedupe
            return
        shutil.move(src_abs, dest)
        undo.append({'quarantined': True, 'from': src_rel,
                     'to': os.path.relpath(dest, directory)})
        log(f"    🔒 旧副本已隔离: {src_rel}")
    except OSError as e:
        log(f"    ⚠️ 隔离失败（保留原位）{src_rel}: {e}")


def _execute_photos(directory: str, db: ReportDB, plans: List[PhotoPlan],
                    undo: List[dict], log) -> Tuple[int, int]:
    """
    执行照片移动并回写 DB。

    返回:
    (moved_files, updated_rows): 实际移动的文件数 / 回写 DB 的照片行数

    Perform the photo moves and DB write-backs.
    """
    moved_files = updated = 0
    for plan in plans:
        if plan.status in ('move', 'already') and plan.files:
            src_abs_dir = os.path.join(directory, plan.src_dir)
            dst_abs_dir = os.path.join(directory, plan.dst_dir)
            os.makedirs(dst_abs_dir, exist_ok=True)
            ok = True
            for fn in plan.files:
                src = os.path.join(src_abs_dir, fn)
                dst = os.path.join(dst_abs_dir, fn)
                if os.path.exists(dst):
                    # 目标已有同名文件：主文件冲突在计划期已排除，这里只会
                    # 是伴生旧副本 → 隔离而不是覆盖
                    # Dest occupied (companion only): quarantine, never overwrite
                    _quarantine(directory, src,
                                _join_rel(plan.src_dir, fn), undo, log)
                    continue
                try:
                    shutil.move(src, dst)
                    undo.append({'prefix': plan.prefix,
                                 'from': _join_rel(plan.src_dir, fn),
                                 'to': _join_rel(plan.dst_dir, fn)})
                    moved_files += 1
                except OSError as e:
                    ok = False
                    log(f"    ❌ 移动失败 {fn}: {e}")
            if plan.status == 'move' and not ok:
                continue  # 主文件有失败项时不动 DB，保持可诊断
        elif plan.status not in ('move', 'already'):
            continue

        # 回写：current_path 指回原始位置；temp_jpeg_path 同步换目录
        # write-back: current_path back to original; temp_jpeg_path re-based
        row = db.get_photo(plan.prefix) or {}
        updates = {'current_path': row.get('original_path')}
        tmp = row.get('temp_jpeg_path') or ''
        if tmp:
            tmp_parts = _rel_parts(tmp)
            src_parts = _rel_parts(plan.master_cur)
            # 旧 temp_jpeg_path 与旧主文件同目录时，把目录前缀换成新目录
            if len(tmp_parts) > 1 and tmp_parts[:-1] == src_parts[:-1]:
                updates['temp_jpeg_path'] = _join_rel(
                    *_rel_parts(row.get('original_path'))[:-1], tmp_parts[-1])
        try:
            db.update_photo(plan.prefix, updates)
            updated += 1
        except Exception as e:  # noqa: BLE001
            log(f"    ❌ DB 回写失败 {plan.prefix}: {e}")
    return moved_files, updated


def _find_stale_companions(directory: str, db: ReportDB) -> List[str]:
    """
    找出整理目录树里 DB 已不再引用的同前缀旧文件（多轮整理/改星的残留）。

    判定：文件位于子目录（非原子根、非隐藏），stem 命中 photos 主键，
    但其相对路径不在 DB 记录的有效路径集合（original/current/temp_jpeg）
    中——即"DB 认识这个前缀，但该副本已不被引用"。DB 完全不认识的前缀
    不在此列（可能用户新放的照片，只报告不处理）。

    Find same-stem stale copies (from repeated organize/re-rate rounds)
    that the DB no longer references. Unknown stems are NOT included.
    """
    valid_paths: set = set()
    prefixes: set = set()
    for row in db.get_all_photos():
        if row.get('filename'):
            prefixes.add(row['filename'])
        for key in ('original_path', 'current_path', 'temp_jpeg_path'):
            v = row.get(key)
            if v:
                valid_paths.add(v.replace('/', os.sep))
    stale: List[str] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        rel_dir = os.path.relpath(dirpath, directory)
        if rel_dir == '.':
            continue  # 原子根下的现行文件 / files at the atomic root are current
        for fn in filenames:
            stem = os.path.splitext(fn)[0]
            rel = os.path.join(rel_dir, fn)
            if stem in prefixes and rel not in valid_paths:
                stale.append(rel)
    return sorted(stale)


def _execute_videos(directory: str, plans: List[VideoPlan],
                    undo: List[dict], log) -> int:
    """执行视频还原改名/上移。返回成功数。/ Restore video names; return count."""
    done = 0
    for vp in plans:
        if vp.status != 'rename':
            continue
        src_abs = os.path.join(directory, vp.path)
        dst_dir_abs = directory if vp.to_root else os.path.dirname(src_abs)
        dst_abs = os.path.join(dst_dir_abs, vp.new_name)
        try:
            if os.path.exists(dst_abs):
                continue
            shutil.move(src_abs, dst_abs)
            undo.append({'video': True, 'from': vp.path,
                         'to': os.path.relpath(dst_abs, directory)})
            done += 1
        except OSError as e:
            log(f"    ❌ 视频还原失败 {vp.path}: {e}")
    return done


def _is_organize_dir(name: str) -> bool:
    """判断目录名是否整理器产物（星级/burst_）。/ Organizer-produced folder?"""
    return name in _RATING_DIR_NAMES or name.startswith('burst_')


def _cleanup_empty_dirs(directory: str, touched_dirs: set, log) -> int:
    """
    删除因移动/隔离而变空的整理目录及其空父级（绝不碰 .superpicky 与根）。

    判定：目录为空，且（本次移出过文件，或 是整理器产物目录名，或
    它是已删目录的父级）。鸟种目录（如 `白翅浮鸥/`）由整理器创建，
    其星级子目录清空后按父级规则一并回收；用户自建的非空目录不受影响。

    Remove now-empty organizer folders and their emptied parents (never
    .superpicky or the root). A dir qualifies when empty AND (we moved
    files out of it, OR it is an organizer-produced folder name, OR it
    is a parent of an already-removed dir).
    """
    removed: set = set()
    all_dirs: List[str] = []
    for dirpath, dirnames, _filenames in os.walk(directory):
        keep = [d for d in dirnames if not d.startswith('.')]
        dirnames[:] = keep
        for d in keep:
            rel = os.path.relpath(os.path.join(dirpath, d), directory)
            all_dirs.append(rel)
    # 自底向上：先删深层（burst_/星级），父级随之变为可删
    for rel in sorted(all_dirs, key=lambda p: -p.count(os.sep)):
        abs_d = os.path.join(directory, rel)
        if not os.path.isdir(abs_d) or os.listdir(abs_d):
            continue
        parent_of_removed = any(r.startswith(rel + os.sep) for r in removed)
        if (rel not in touched_dirs and not parent_of_removed
                and not _is_organize_dir(os.path.basename(rel))):
            continue
        try:
            os.rmdir(abs_d)
            removed.add(rel)
            log(f"    🧹 已删除空目录: {rel}")
        except OSError:
            pass  # 被占用 → 保留 / keep when busy
    return len(removed)


def flatten_directory(root: str, execute: bool = False, log=print) -> bool:
    """
    把一个根目录下所有已处理库恢复扁平。

    参数:
    root (str): 根目录（可含多层子目录，逐原子目录处理）
    execute (bool): False=dry-run；True=执行并写撤销清单
    log: 日志函数

    返回:
    bool: 无冲突/无缺失（或已全部处理）返回 True

    Flatten every processed library under root (per atomic directory).
    """
    if not os.path.isdir(root):
        log(f"❌ 目录不可达: {root}")
        return False

    atomic_dirs = find_processed_subdirs(root)
    if not atomic_dirs:
        log(f"❌ 未发现任何已处理目录（缺少 .superpicky/report.db）: {root}")
        return False

    log(f"📂 发现 {len(atomic_dirs)} 个已处理目录" +
        ("" if len(atomic_dirs) == 1 else f"（{root}）"))
    all_ok = True
    for d in atomic_dirs:
        log(f"\n── {d} " + "─" * max(4, 60 - len(d)))
        ok = _flatten_atomic(d, execute=execute, log=log)
        all_ok = all_ok and ok
    return all_ok


def _flatten_atomic(directory: str, execute: bool, log) -> bool:
    """对单个原子目录执行 dry-run 或正式还原。/ Flatten one atomic dir."""
    db = ReportDB(directory)
    try:
        photo_plans = _plan_photos(directory, db, log)
        video_plans = _scan_videos(directory)
    finally:
        pass

    n_move = sum(1 for p in photo_plans if p.status == 'move')
    n_already = sum(1 for p in photo_plans if p.status == 'already')
    n_conflict = sum(1 for p in photo_plans if p.status == 'conflict')
    n_missing = sum(1 for p in photo_plans if p.status == 'missing')
    n_files = sum(len(p.files) for p in photo_plans if p.status == 'move')
    n_vid = sum(1 for v in video_plans if v.status == 'rename')
    n_vid_conflict = sum(1 for v in video_plans if v.status == 'conflict')

    log(f"  照片: 待移 {n_move} 张（含伴生共 {n_files} 个文件）｜"
        f"已在位 {n_already}｜冲突 {n_conflict}｜源缺失 {n_missing}")
    log(f"  视频: 待还原原名 {n_vid} 个｜冲突 {n_vid_conflict}")
    stale = _find_stale_companions(directory, db)
    log(f"  陈旧伴生副本: {len(stale)} 个（执行时隔离到 .superpicky/flatten_orphans/）")
    for s in stale[:5]:
        log(f"    示例: {s}")
    for p in photo_plans:
        if p.status in ('conflict', 'missing'):
            log(f"    ⚠️ [{p.status}] {p.prefix}: {p.detail}")
    for v in video_plans:
        if v.status == 'conflict':
            log(f"    ⚠️ [video-conflict] {v.path}: {v.detail}")
    for p in photo_plans[:5]:
        if p.status == 'move':
            log(f"    示例: {p.src_dir}{os.sep}{p.files[0] if p.files else '?'} "
                f"→ {p.dst_dir or '.'}{os.sep}{p.files[0] if p.files else '?'}")

    if not execute:
        log("  （dry-run 未做任何改动，加 --execute 执行）")
        return n_conflict == 0 and n_missing == 0

    undo: List[dict] = []
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    moved_files, updated = _execute_photos(directory, db, photo_plans, undo, log)
    vids = _execute_videos(directory, video_plans, undo, log)
    stale_quarantined = 0
    for rel in stale:
        before = len(undo)
        _quarantine(directory, os.path.join(directory, rel), rel, undo, log)
        stale_quarantined += len(undo) - before
    log(f"  ✅ 移动文件 {moved_files} 个，DB 回写 {updated} 行，视频还原 {vids} 个，"
        f"隔离旧副本 {stale_quarantined} 个")

    # burst_ 目录消失 → 清空分组列；浏览器打开时本就会按路径重算
    try:
        db.clear_burst_ids()
    except Exception as e:  # noqa: BLE001
        log(f"  ⚠️ clear_burst_ids 失败（不影响结果）: {e}")

    # 增量重导出 sidecar：current_path 变化触发重写，人工 edits 保留
    try:
        written = export_directory_sidecars(db, directory, log=lambda *_: None)
        log(f"  ✅ sidecar 重导出 {written} 个 JSON")
    except Exception as e:  # noqa: BLE001
        log(f"  ⚠️ sidecar 重导出失败: {e}")

    touched = {p.src_dir for p in photo_plans if p.status == 'move'}
    touched |= {os.path.dirname(v.path)
                for v in video_plans if v.status == 'rename' and v.to_root}
    cleaned = _cleanup_empty_dirs(directory, {t for t in touched if t and t != '.'},
                                  log)
    log(f"  🧹 清理空目录 {cleaned} 个")

    undo_path = os.path.join(directory, '.superpicky',
                             f'flatten_undo_{stamp}.json')
    try:
        with open(undo_path, 'w', encoding='utf-8') as f:
            json.dump({'created': stamp, 'directory': directory,
                       'entries': undo}, f, ensure_ascii=False, indent=2)
        log(f"  📝 撤销清单: {undo_path}")
    except OSError as e:
        log(f"  ⚠️ 撤销清单写入失败: {e}")

    db.close()
    return True


def verify_directory(root: str, log=print) -> bool:
    """
    事后验证：每个原子目录的 current_path 应等于 original_path（或未整理
    的缓存路径保持原样），且文件确实存在于该位置。

    Post-run verification: DB paths match original_path and files exist.
    """
    ok = True
    for d in find_processed_subdirs(root):
        db = ReportDB(d)
        bad = 0
        for row in db.get_all_photos():
            cur, orig = row.get('current_path') or '', row.get('original_path') or ''
            if not orig:
                continue
            if any(p == '.superpicky' for p in _rel_parts(cur)):
                continue  # 未整理库的缓存路径 / untouched cache path
            if cur != orig or not os.path.isfile(os.path.join(d, cur)):
                bad += 1
                if bad <= 5:
                    log(f"  ❌ {d}: {row.get('filename')} cur={cur} orig={orig}")
        db.close()
        log(f"  {'✅' if bad == 0 else '❌'} {d}: "
            f"{'路径全部一致' if bad == 0 else f'{bad} 行异常'}")
        ok = ok and bad == 0
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口 / CLI entry point."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description='把已按鸟种/星级/burst_ 整理的照片库恢复扁平（识别结果保留）')
    parser.add_argument('directories', nargs='+', help='要恢复的根目录（可多个）')
    parser.add_argument('--execute', action='store_true',
                        help='正式执行（默认 dry-run 只预览）')
    parser.add_argument('--verify-only', action='store_true',
                        help='只做事后验证，不做任何改动')
    args = parser.parse_args(argv)

    if args.verify_only:
        ok = all(verify_directory(d) for d in args.directories)
        return 0 if ok else 1

    ok = True
    for d in args.directories:
        ok = flatten_directory(d, execute=args.execute) and ok
        if args.execute:
            ok = verify_directory(d) and ok
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
