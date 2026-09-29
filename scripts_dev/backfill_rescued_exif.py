#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回填"救回/改种"照片缺失的 EXIF 元数据（历史链路缺口修复工具）。

Backfill missing EXIF metadata for photos that were rescued / manually
re-identified back to has_bird=1 by older tool versions.

背景：V5.6 识鸟门控拦掉的照片（has_bird=0）不提取 EXIF；早期的无鸟
救回 / 手动改种路径把它们写回 has_bird=1 时没有回填 EXIF（拍摄时间、
机身、ISO 等），导致浏览库 sidebar 与 sidecar 的 photo 块日期等字段
为空。本工具扫描 photos 表中 has_bird=1 且 date_time_original 为 NULL
的行（即该历史缺口的典型特征），用 ExifTool 从照片原片读回标准字段，
仅回填当前仍为 NULL 的列（COALESCE 语义，绝不覆盖已有值）。

写库安全：
- 默认 dry-run，--execute 才写；写前自动备份 report.db；
- 单一写者原则：零接触照片文件，只写 report.db；
- 参数化 SQL，无字符串拼接；
- sidecar 交 reexport_sidecars.py 重导（本脚本不改 sidecar）。

Safety: dry-run by default, auto-backup before --execute, parameterized
SQL only, photo files never touched; re-export sidecars afterwards via
reexport_sidecars.py.

用法 / Usage:
    # dry-run：列出待回填照片与将写入的字段值
    python scripts_dev/backfill_rescued_exif.py "<照片目录>"
    # 实际写库（自动备份）
    python scripts_dev/backfill_rescued_exif.py "<照片目录>" --execute
    # 指定文件前缀（默认自动扫描 has_bird=1 且日期为 NULL 的行）
    python scripts_dev/backfill_rescued_exif.py "<照片目录>" --files 027A5584,027A5585 --execute
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

# 确保能 import 项目模块（本脚本在 scripts_dev/ 下，项目根是上一级）。
# Ensure the project root is importable (this script lives in scripts_dev/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts_dev.backfill_species import find_raw, find_plain_image  # noqa: E402

# photos 列 → ExifTool 标签。字段口径与
# core/photo_processor.PhotoProcessor._read_all_exif_metadata 完全一致，
# 读取参数（-n -charset utf8）也与 core/focus_point_detector._read_exif
# 相同，保证回填值与主管线写库格式一致。
# photos column -> ExifTool tag. Mirrors the main pipeline's field set
# and read flags so backfilled values match pipeline-written formats.
EXIF_FIELD_MAP: Dict[str, str] = {
    "iso": "ISO",
    "shutter_speed": "ShutterSpeed",
    "aperture": "Aperture",
    "focal_length": "FocalLength",
    "focal_length_35mm": "FocalLengthIn35mmFormat",
    "camera_model": "Model",
    "lens_model": "LensModel",
    "gps_latitude": "GPSLatitude",
    "gps_longitude": "GPSLongitude",
    "gps_altitude": "GPSAltitude",
    "title": "Title",
    "caption": "Caption-Abstract",
    "city": "City",
    "state_province": "State",
    "country": "Country",
    "date_time_original": "DateTimeOriginal",
}

# 整数列与浮点列（其余列一律按字符串回填）。
# Integer columns and float columns (all other columns backfill as str).
_INT_COLUMNS = {"iso", "focal_length_35mm"}
_FLOAT_COLUMNS = {"shutter_speed", "aperture", "focal_length",
                  "gps_latitude", "gps_longitude", "gps_altitude"}


def resolve_exiftool() -> Optional[str]:
    """
    定位可用的 ExifTool 可执行文件。

    优先用仓库自带的 exiftools_win / exiftools_mac 目录，找不到再退回
    PATH 上的 exiftool。

    返回:
    Optional[str]: 可执行文件路径；找不到返回 None

    Locate the ExifTool executable: prefer the bundled exiftools_win /
    exiftools_mac directory, then fall back to `exiftool` on PATH.

    Return:
    Optional[str]: Executable path, or None when not found.
    """
    if sys.platform.startswith("win"):
        bundled = os.path.join(_PROJECT_ROOT, "exiftools_win", "exiftool.exe")
    else:
        bundled = os.path.join(_PROJECT_ROOT, "exiftools_mac", "exiftool")
    if os.path.isfile(bundled):
        return bundled
    return shutil.which("exiftool")


def _convert_column(column: str, raw) -> Optional[object]:
    """
    按 photos 列的类型把 ExifTool 原始值转换成可入库的 Python 值。

    参数:
    column (str): photos 表列名
    raw: ExifTool -j 输出的原始值（int/float/str）

    返回:
    Optional[object]: 转换后的值；缺失或转换失败返回 None

    Convert a raw ExifTool value to the photos-column Python type.

    Parameters:
    column (str): photos table column name
    raw: Raw value from ExifTool -j output (int/float/str)

    Return:
    Optional[object]: Converted value; None when missing or conversion
    fails.
    """
    if raw is None:
        return None
    try:
        if column in _INT_COLUMNS:
            return int(float(raw))
        if column in _FLOAT_COLUMNS:
            return float(raw)
        text = str(raw).strip()
        return text or None
    except (TypeError, ValueError):
        return None


def read_exif_fields(exiftool_path: str, filepath: str) -> Optional[dict]:
    """
    一次性读取标准 EXIF 字段（与主管线同一套标签与参数）。

    参数:
    exiftool_path (str): ExifTool 可执行文件路径
    filepath (str): 照片原片路径

    返回:
    Optional[dict]: {ExifTool 标签: 值}；读取失败返回 None

    Read the standard EXIF fields in one shot (same tags and flags as
    the main pipeline).

    Parameters:
    exiftool_path (str): ExifTool executable path
    filepath (str): Original photo file path

    Return:
    Optional[dict]: {tag: value}; None when the read fails.
    """
    cmd = [exiftool_path, "-j", "-n", "-charset", "utf8"]
    cmd += [f"-{tag}" for tag in EXIF_FIELD_MAP.values()]
    cmd.append(filepath)
    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                              errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    return data[0]


def collect_fill_values(conn: sqlite3.Connection, exiftool_path: str,
                        root: str, prefix: str
                        ) -> Tuple[Dict[str, object], Optional[str]]:
    """
    计算单张照片需要回填的列与值（只含当前为 NULL 且原片有值的列）。

    参数:
    conn (sqlite3.Connection): 已打开的 report.db 连接
    exiftool_path (str): ExifTool 可执行文件路径
    root (str): 照片库根目录
    prefix (str): 照片文件名前缀（photos.filename）

    返回:
    Tuple[Dict[str, object], Optional[str]]: ({列: 值}, 跳过原因)；
    正常可回填时跳过原因为 None

    Compute the columns/values to backfill for one photo (only columns
    that are currently NULL and readable from the original file).

    Parameters:
    conn (sqlite3.Connection): Open report.db connection
    exiftool_path (str): ExifTool executable path
    root (str): Library root directory
    prefix (str): Photo filename prefix (photos.filename)

    Return:
    Tuple[Dict[str, object], Optional[str]]: ({column: value}, skip
    reason); the skip reason is None when backfill is possible.
    """
    cols = list(EXIF_FIELD_MAP)
    row = conn.execute(
        f"SELECT {', '.join(cols)} FROM photos WHERE filename = ?",
        (prefix,)).fetchone()
    if row is None:
        return {}, "photos 表无此行"
    filepath = find_raw(root, prefix) or find_plain_image(root, prefix)
    if filepath is None:
        return {}, "找不到原片文件"
    exif = read_exif_fields(exiftool_path, filepath)
    if not exif:
        return {}, "EXIF 读取失败"
    fill: Dict[str, object] = {}
    for column, current in zip(cols, row):
        if current is not None:
            continue  # 已有值一律不覆盖 / Never overwrite existing values
        value = _convert_column(column, exif.get(EXIF_FIELD_MAP[column]))
        if value is not None:
            fill[column] = value
    return fill, (None if fill else "原片无可回填的新字段")


def backfill_photos_exif(conn: sqlite3.Connection, exiftool_path: str,
                         root: str, prefixes: List[str],
                         log: Callable[[str], None] = print) -> int:
    """
    对指定前缀逐张回填 EXIF（UPDATE 后不 commit，由调用方统一提交）。

    参数:
    conn (sqlite3.Connection): 已打开的 report.db 连接
    exiftool_path (str): ExifTool 可执行文件路径
    root (str): 照片库根目录
    prefixes (List[str]): 照片文件名前缀列表
    log (Callable[[str], None]): 日志输出函数

    返回:
    int: 实际回填的行数

    Backfill EXIF for the given prefixes row by row (UPDATE without
    commit; the caller owns the transaction).

    Parameters:
    conn (sqlite3.Connection): Open report.db connection
    exiftool_path (str): ExifTool executable path
    root (str): Library root directory
    prefixes (List[str]): Photo filename prefixes
    log (Callable[[str], None]): Log sink

    Return:
    int: Number of rows actually backfilled.
    """
    updated = 0
    for prefix in prefixes:
        fill, reason = collect_fill_values(conn, exiftool_path, root, prefix)
        if reason is not None and not fill:
            log(f"  ⏭️ {prefix}: 跳过（{reason}）")
            continue
        set_parts = [f"{column} = ?" for column in fill]
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        params = list(fill.values()) + [prefix]
        conn.execute(
            f"UPDATE photos SET {', '.join(set_parts)} WHERE filename = ?",
            params)
        updated += 1
        brief = ", ".join(
            f"{k}={fill[k]!r}" for k in
            ("date_time_original", "camera_model", "iso") if k in fill)
        log(f"  ✅ {prefix}: 回填 {len(fill)} 列（{brief} …）")
    return updated


def main() -> int:
    """
    主流程：定位待回填行 →（dry-run 预览 / 备份后写库）→ 提示重导 sidecar。

    返回:
    int: 0 成功；1 输入错误

    Main flow: locate rows to backfill -> (dry-run preview / backup then
    write) -> hint to re-export sidecars.

    Return:
    int: 0 on success; 1 on input errors.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description="回填救回/改种照片缺失的 EXIF 元数据")
    ap.add_argument("directory", help="照片库根目录（需含 .superpicky/report.db）")
    ap.add_argument("--files", default=None,
                    help="逗号分隔的文件名前缀；默认自动扫描 has_bird=1 "
                         "且 date_time_original 为 NULL 的行")
    ap.add_argument("--execute", action="store_true",
                    help="写库（默认 dry-run；写前自动备份 report.db）")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    db_path = os.path.join(root, ".superpicky", "report.db")
    if not os.path.exists(db_path):
        print(f"❌ 未找到 report.db: {db_path}")
        return 1
    exiftool_path = resolve_exiftool()
    if exiftool_path is None:
        print("❌ 未找到 ExifTool（exiftools_win/exiftools_mac 或 PATH）")
        return 1

    if args.files:
        prefixes = [p.strip() for p in args.files.split(",") if p.strip()]
    else:
        conn = sqlite3.connect(db_path)
        try:
            prefixes = [r[0] for r in conn.execute(
                "SELECT filename FROM photos "
                "WHERE has_bird = 1 AND date_time_original IS NULL "
                "ORDER BY filename")]
        finally:
            conn.close()
    if not prefixes:
        print("✅ 无待回填行（has_bird=1 且日期为 NULL 的照片不存在）")
        return 0

    print(f"📂 库目录: {root}")
    print(f"🕵️ 待回填 {len(prefixes)} 张: {','.join(prefixes)}")

    if not args.execute:
        conn = sqlite3.connect(db_path)
        try:
            for prefix in prefixes:
                fill, reason = collect_fill_values(
                    conn, exiftool_path, root, prefix)
                if reason is not None and not fill:
                    print(f"  ⏭️ {prefix}: 跳过（{reason}）")
                    continue
                preview = ", ".join(
                    f"{k}={fill[k]!r}" for k in sorted(fill))
                print(f"  📋 {prefix}: 将回填 {len(fill)} 列 → {preview}")
        finally:
            conn.close()
        print("（dry-run，未写库。确认后加 --execute 执行）")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = db_path + f".bak_EXIF回填_{stamp}"
    shutil.copy(db_path, bak)
    print(f"📦 已备份: {os.path.basename(bak)}")

    conn = sqlite3.connect(db_path)
    try:
        updated = backfill_photos_exif(conn, exiftool_path, root, prefixes)
        conn.commit()
    finally:
        conn.close()
    print(f"→ 完成 {updated} 张；sidecar 用 reexport_sidecars.py 重导")
    return 0


if __name__ == "__main__":
    sys.exit(main())
