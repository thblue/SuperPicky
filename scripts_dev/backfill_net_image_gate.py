#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非相机原图门槛存量清洗：把 EXIF「三全空」的有鸟行归为无鸟态。

面向 2026-10-21 网图门槛（core/photo_processor.py 主循环 +
constants.is_non_camera_original_exif）上线前已跑出的成品库：QQ/网页
转存下载图常伪装相机命名（如 IMG_8475.JPG 煤山雀高清网图、IMG_1689-91
雪鸮网图），管线当时按真鸟入库并在 BirdIndex 鸟种页造成污染。

清洗语义与 spb_rename_species.py `photo --wipe`（对外契约）逐字段一致，
即 2026-10-20 网图清理（scripts_dev/_backfill_sandbox/batch_wipe_netimg.py）
的同一套手法：

  1. bird_detections 全部活性行软删（deleted=1, is_selected=0，
     物种数据保留在行内，可追溯可恢复）；
  2. photos 归无鸟态（spb_rename_species._NO_BIRD_FIELDS +
     temp_jpeg_path=None）；
  3. .superpicky/meta/<前缀>.json sidecar 幂等删除（BirdIndex 唯一
     数据源，下次 `python -m indexer scan` 按消失文件自动出清）；
  4. wal_checkpoint(TRUNCATE) 收尾。

防护与边界：
- 候选判据与管线门槛完全一致（constants.is_non_camera_original_exif），
  V-cover 行（filename 以 _vcover 结尾）跳过——归视频工作流；
- 归态前对文件本体**重新读一次 EXIF**：只有现读仍三全空才降级，防止
  当年跑批时 NAS 瞬时故障导致真照片 EXIF 读空被误杀；文件已不存在时
  以 DB 行为准降级（无从复核，但软删行保留可回滚）；
- 单一写者原则：本脚本属 SuperPicky 运维上下文，只写 report.db 与
  .superpicky/meta/*.json，照片文件本体零接触（预览删除仅限
  .superpicky/cache/ 之下，与 spb_rename_species 同款硬闸门）；
- 默认 dry-run，--execute 才落盘；--execute 前自动备份 report.db；
- 可回滚：软删检测行翻回 deleted=0 / is_selected=1 + photos 重建鸟种
  字段后重导 sidecar（2026-10-20 误伤 184 张当日全量恢复同路径）。

Existing-library cleanup for the 2026-10-21 net-image gate: demote
has_bird rows whose camera model / capture time / GPS are ALL empty
(the signature of transferred net images masquerading as camera names).
Field-for-field identical to spb_rename_species.py `photo --wipe`
(soft-delete detections, reset photos to the no-bird state, delete
sidecar JSONs — BirdIndex purges them as stale on the next scan).
Guardrails: same gate criterion; V-cover rows skipped; each candidate's
file is re-read via exiftool before demotion; missing files demote on
DB evidence; dry-run by default, --execute backs up report.db first.

用法 / Usage:
    python scripts_dev/backfill_net_image_gate.py <照片库目录>... [--execute]
    python scripts_dev/backfill_net_image_gate.py //NAS-server/zhengyuan/photo --execute
"""

import argparse
import os
import sqlite3
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import is_non_camera_original_exif, VIDEO_COVER_SUFFIX  # noqa: E402
from spb_rename_species import _NO_BIRD_FIELDS  # noqa: E402
from tools.exiftool_manager import get_exiftool_manager  # noqa: E402
from tools.report_db import ReportDB  # noqa: E402

# 文件本体复核时读取的 EXIF 标签（与三全空判据一一对应）
# EXIF tags re-read from the file for verification (mirrors the gate keys).
_VERIFY_TAGS = {
    'camera_model': '-Model',
    'date_time_original': '-DateTimeOriginal',
    'gps_latitude': '-GPSLatitude',
}


def find_library_dirs(roots: List[str]) -> List[str]:
    """
    递归枚举根目录下全部含 .superpicky/report.db 的照片库目录。

    参数:
    roots (List[str]): 待扫描的库根目录列表（文件或目录均可，文件则取其所在库）

    返回:
    List[str]: 去重后的照片库目录列表（即 .superpicky 的父目录）

    Enumerate every photo-library directory holding a
    .superpicky/report.db under the given roots.
    """
    libs: List[str] = []
    seen = set()
    for root in roots:
        if os.path.isfile(root):
            lib = os.path.dirname(os.path.dirname(os.path.abspath(root)))
            if lib not in seen:
                seen.add(lib)
                libs.append(lib)
            continue
        if not os.path.isdir(root):
            print(f"⚠️ 跳过不存在的路径: {root}")
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if (os.path.basename(dirpath) == '.superpicky'
                    and 'report.db' in filenames):
                lib = os.path.dirname(dirpath)
                if lib not in seen:
                    seen.add(lib)
                    libs.append(lib)
    return libs


def _candidate_file(photo_dir: str, row: Dict) -> str:
    """
    返回某照片行用于 EXIF 复核的文件本体路径（原文件优先，退回当前路径）。

    参数:
    photo_dir (str): 照片库目录
    row (Dict): photos 行（含 original_path / current_path）

    返回:
    str: 存在的文件路径；都不存在时返回空串

    Resolve the on-disk file of a photo row for EXIF re-verification.
    """
    for key in ('original_path', 'current_path'):
        rel = row.get(key)
        if not rel:
            continue
        p = os.path.normpath(os.path.join(photo_dir, rel))
        if os.path.exists(p):
            return p
    return ''


def _remove_cache_previews(photo_dir: str, rel_paths: List[object]) -> int:
    """
    删除被归无鸟照片的生成预览（仅限 .superpicky/cache/ 之下的硬闸门内，
    与 spb_rename_species._remove_previews 同款——temp_jpeg_path 指向
    原片/伴随 JPG 时绝不删）。

    参数:
    photo_dir (str): 照片库目录
    rel_paths (List[object]): photos.temp_jpeg_path 相对路径列表

    返回:
    int: 实际删除的文件数

    Remove generated previews under .superpicky/cache/ only (hard gate
    identical to spb_rename_species._remove_previews).
    """
    cache_root = os.path.normpath(os.path.join('.superpicky', 'cache'))
    removed = 0
    for rel in rel_paths:
        if not rel:
            continue
        norm = os.path.normpath(str(rel))
        if not norm.startswith(cache_root + os.sep):
            continue
        abs_path = os.path.join(photo_dir, norm)
        try:
            if os.path.exists(abs_path):
                os.remove(abs_path)
                removed += 1
        except OSError as e:
            print(f"  ⚠️ 预览删除失败 {rel}: {e}")
    return removed


def clean_library(photo_dir: str, execute: bool,
                  manager=None) -> Tuple[int, int, int]:
    """
    清洗单个照片库：三全空有鸟行归无鸟态 + 删除 sidecar JSON。

    参数:
    photo_dir (str): 照片库目录（含 .superpicky/report.db）
    execute (bool): True=实际写库删文件；False=dry-run 只报告
    manager: 可复用的 ExifTool 常驻进程管理器（None=内部创建）

    返回:
    Tuple[int, int, int]: (归无鸟行数, 复核豁免行数, 文件缺失归无鸟行数)

    Clean one library: reset all-empty-EXIF has_bird rows to the no-bird
    state and remove their sidecar JSONs.
    """
    db = ReportDB(photo_dir)
    try:
        candidates = db.get_all_photos()
    finally:
        db.close()

    demoted: List[Dict] = []
    spared: List[str] = []
    missing: List[Dict] = []
    if manager is None:
        manager = get_exiftool_manager()

    for row in candidates:
        if not row.get('has_bird'):
            continue
        filename = row.get('filename') or ''
        if not filename or filename.endswith(VIDEO_COVER_SUFFIX):
            continue
        # 判据与管线一致：三字段全空（DB 侧）
        if not is_non_camera_original_exif(row):
            continue
        file_path = _candidate_file(photo_dir, row)
        if not file_path:
            missing.append(row)
            continue
        # 文件本体复核：现读仍三全空才降级
        meta = manager.read_metadata(
            file_path,
            extra_args=['-n', '-charset', 'utf8']
            + list(_VERIFY_TAGS.values()))
        if meta is None:
            # 读取失败（非文件缺失）：保守豁免，报告人工排查
            spared.append(f"{filename} (exiftool 读取失败)")
            continue
        if not is_non_camera_original_exif({
            'camera_model': meta.get('Model'),
            'date_time_original': meta.get('DateTimeOriginal'),
            'gps_latitude': meta.get('GPSLatitude'),
        }):
            spared.append(f"{filename} (文件现读有相机 EXIF)")
            continue
        demoted.append(row)

    if not demoted and not missing:
        if spared:
            for name in spared:
                print(f"  ! 豁免 [{os.path.basename(photo_dir)}]: {name}")
        print(f"  ✅ {photo_dir}: 无三全空有鸟行，跳过")
        return 0, len(spared), 0

    print(f"  📋 {photo_dir}\n"
          f"     候选 {len(demoted) + len(missing)} 行"
          f"（文件复核确认 {len(demoted)}，文件缺失按 DB 归态 {len(missing)}，"
          f"复核豁免 {len(spared)}）")
    for row in demoted[:20]:
        print(f"    - {row['filename']} "
              f"[{row.get('bird_species_cn') or '?'}] → 无鸟态（文件复核确认）")
    if len(demoted) > 20:
        print(f"    - …共 {len(demoted)} 行")
    for row in missing[:20]:
        print(f"    - {row['filename']} "
              f"[{row.get('bird_species_cn') or '?'}] → 无鸟态（文件缺失，按 DB）")
    if len(missing) > 20:
        print(f"    - …共 {len(missing)} 行")
    for name in spared:
        print(f"    ! 豁免: {name}")

    if not execute:
        print("  （dry-run，未写库；加 --execute 生效）")
        return len(demoted), len(spared), len(missing)

    # --execute：先用 sqlite backup API 备份 report.db（对 WAL 安全，
    # 直接 copy2 可能落下一份不含 -wal 内容的旧快照）
    stamp = time.strftime('%Y%m%d-%H%M%S')
    db_path = os.path.join(photo_dir, '.superpicky', 'report.db')
    bak_path = f"{db_path}.bak-netgate-{stamp}"
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(bak_path)
    src.backup(dst)
    dst.close()
    src.close()
    print(f"  💾 已备份 report.db → {os.path.basename(bak_path)}")

    db = ReportDB(photo_dir)
    sidecar_dir = os.path.join(photo_dir, '.superpicky', 'meta')
    removed_sidecars = 0
    removed_previews = 0
    n_done = 0
    try:
        for row in demoted + missing:
            filename = row['filename']
            live = [d for d in db.get_detections(filename)
                    if not d.get('deleted')]
            if live:
                db.soft_delete_detections(
                    filename, [d['bird_index'] for d in live])
            db.update_photo(filename, dict(_NO_BIRD_FIELDS,
                                           temp_jpeg_path=None))
            removed_previews += _remove_cache_previews(
                photo_dir, [row.get('temp_jpeg_path')])
            sidecar = os.path.join(sidecar_dir, f"{filename}.json")
            if os.path.exists(sidecar):
                os.remove(sidecar)
                removed_sidecars += 1
            n_done += 1
        try:
            db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
    finally:
        db.close()
    print(f"  ✍️ 已归无鸟态 {n_done} 行，删除 sidecar {removed_sidecars} 个，"
          f"清理缓存预览 {removed_previews} 个")
    return len(demoted), len(spared), len(missing)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='非相机原图门槛存量清洗（三全空 EXIF 有鸟行归无鸟态，'
                    '等价 spb_rename_species photo --wipe）')
    parser.add_argument('roots', nargs='+',
                        help='照片库根目录（递归查找 .superpicky/report.db）')
    parser.add_argument('--execute', action='store_true',
                        help='实际写库（默认 dry-run；执行前自动备份 report.db）')
    args = parser.parse_args()

    libs = find_library_dirs(args.roots)
    if not libs:
        print("❌ 未找到任何含 .superpicky/report.db 的照片库")
        return 1
    mode = 'EXECUTE' if args.execute else 'DRY-RUN'
    print(f"{'🚀' if args.execute else '🔍'} 非相机原图门槛存量清洗 [{mode}]，"
          f"共 {len(libs)} 个库\n")
    total = [0, 0, 0]
    manager = get_exiftool_manager()
    try:
        for lib in libs:
            d, s, m = clean_library(lib, args.execute, manager)
            total[0] += d
            total[1] += s
            total[2] += m
    finally:
        try:
            manager.close_persistent_session('backfill_net_image_gate')
        except Exception:
            pass
    print(f"\n🏁 完成：归无鸟态 {total[0]} 行（其中文件缺失 {total[2]}），"
          f"复核豁免 {total[1]} 行")
    if not args.execute:
        print("（dry-run 未写任何数据；确认无误后加 --execute 重跑）")
    else:
        print("下一步：在 BirdIndex 目录执行 `python -m indexer scan`，"
              "出清已删除 sidecar 对应的索引记录。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
