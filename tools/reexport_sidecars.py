#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对已有照片库重导出 per-photo sidecar JSON（存量库升级工具）。

用途：SuperPicky 只在批处理收尾导出 sidecar；当导出格式升级
（如 V5.3 新增 photo.library_path / preview_path、filename 带扩展名）
而已有成品库不重新跑批时，用本工具按当前导出逻辑把
`<照片目录>/.superpicky/report.db` 全量重写 `.superpicky/meta/*.json`。

特点：
- DB 严格只读（sqlite mode=ro URI 打开，不创建 wal/shm、不写任何行）；
- 复用 core.sidecar_export.export_directory_sidecars 的全部生产逻辑：
  增量判断（含 V5.3 前旧格式的 library_path 自愈重写）、人工
  edits/main_species 保留、tmp+replace 原子写；
- 照片原文件零接触。

Re-export per-photo sidecar JSONs for an existing library without
re-running the batch. Opens report.db strictly read-only (sqlite URI
mode=ro) and reuses the production export path (incremental,
edit-preserving, atomic writes).

用法 / Usage:
    python tools/reexport_sidecars.py "G:/PHOTO/某库"
    python tools/reexport_sidecars.py "//NAS-server/BAK2/90d/2024.1 江西"
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import urllib.parse
from typing import List

# 确保能 import 生产模块（本脚本在 tools/ 下，项目根是上一级）。
# Ensure the project root is importable (this script lives in tools/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.sidecar_export import export_directory_sidecars


def _sqlite_ro_uri(db_path: str) -> str:
    """
    构造 sqlite 只读 URI（兼容 Windows 本地盘与 UNC 网络路径）。

    注意：Path.as_uri() 对 UNC 路径产生 `file://NAS-server/...`，SQLite 会
    把 NAS-server 解析成不支持的 URI authority 而报错；正确形式是空
    authority + 双斜杠路径 `file:////NAS-server/...`。故统一手动拼接并
    百分号编码非 ASCII 字符（中文库目录名）。

    参数:
    db_path (str): report.db 路径（G:/... 或 //NAS-server/... 均可）

    返回:
    str: sqlite URI（含 ?mode=ro）

    Build a read-only sqlite URI that works for both local Windows drives
    and UNC share paths (where Path.as_uri() yields an invalid authority).
    """
    norm = db_path.replace("\\", "/")
    if not norm.startswith("//"):
        # 本地绝对路径（G:/x）规范为 /G:/x → file:///G:/x
        norm = "/" + norm.lstrip("/")
    return "file://" + urllib.parse.quote(norm, safe="/:") + "?mode=ro"


class ReadOnlySidecarSource:
    """
    只读数据源：duck-type ReportDB 的 get_all_photos / get_all_detections，
    供 export_directory_sidecars 消费。

    连接以 sqlite URI mode=ro 打开——绝不写库、不建 wal/shm，
    可安全用于 NAS 生产库。

    参数:
    db_path (str): report.db 路径（本地盘或 UNC 均可）

    异常:
    sqlite3.Error: 打开失败（文件不存在/被锁/损坏）

    Read-only duck-typed source for export_directory_sidecars; the
    connection uses mode=ro so the DB is never modified.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(_sqlite_ro_uri(db_path), uri=True)
        self._conn.row_factory = sqlite3.Row

    def get_all_photos(self) -> List[dict]:
        """全量 photos 行（与 ReportDB.get_all_photos 同序）。/ All photos."""
        cur = self._conn.execute("SELECT * FROM photos ORDER BY filename")
        return [dict(r) for r in cur.fetchall()]

    def get_all_detections(self) -> List[dict]:
        """全量 bird_detections 行。/ All detections."""
        cur = self._conn.execute(
            "SELECT * FROM bird_detections ORDER BY filename, bird_index")
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def main(argv: List[str] = None) -> int:
    """
    入口：解析库目录 → 只读打开 report.db → 全量重导出 sidecar。

    参数:
    argv (List[str]): 命令行参数（默认取 sys.argv[1:]）

    返回:
    int: 0 成功；1 输入错误
    """
    # Windows 控制台中文输出保护（Git Bash / cmd 下强制 UTF-8）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description="重导出照片库的 per-photo sidecar JSON（DB 只读，不改照片）")
    parser.add_argument(
        "directory", help="照片库根目录（需含 .superpicky/report.db）")
    args = parser.parse_args(argv)

    directory = args.directory
    db_path = os.path.join(directory, ".superpicky", "report.db")
    if not os.path.exists(db_path):
        print(f"❌ 未找到 report.db: {db_path}")
        return 1

    print(f"📂 库目录: {directory}")
    print(f"🗄️  只读打开: {db_path}")

    source = ReadOnlySidecarSource(db_path)
    try:
        written = export_directory_sidecars(source, directory, log=print)
    finally:
        source.close()

    meta_dir = os.path.join(directory, ".superpicky", "meta")
    total = 0
    if os.path.isdir(meta_dir):
        total = sum(1 for f in os.listdir(meta_dir) if f.endswith(".json"))
    print(f"✅ 完成：本次写入/升级 {written} 个，meta 目录现存 {total} 个 JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
