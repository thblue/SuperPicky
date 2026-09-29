#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 BirdIndex config.json 里的全部照片目录，找出「照片已入库但视频未处理」的目录。

判定口径（与 core/video_stage 的幂等语义一致）：
  - 目录存在 .superpicky/report.db → 照片阶段跑过；
  - 目录顶层视频数（.mp4/.mov/.m4v，跳过点开头文件）> 0 且
    report.db photos 表里 <stem>_vcover 行数 < 视频数 → 该目录需要补跑视频阶段。

只读操作：report.db 以只读方式打开，不写任何数据。

Scan every photo root listed in BirdIndex's config.json and report
directories whose photos are already in report.db but whose videos were
never processed (vcover rows < top-level videos). Strictly read-only.

参数 / Args:
    --config : BirdIndex config.json 路径（默认 G:/code/BirdIndex/config.json）
    --json   : 以 JSON 输出需要补跑的目录清单（供补跑脚本消费）
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import List

VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


def count_top_level_videos(dir_path: str) -> int:
    """统计目录顶层视频数（跳过 `.` 开头项，与 find_top_level_videos 同规则）。"""
    n = 0
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if os.path.splitext(entry.name)[1].lower() in VIDEO_EXTS:
                    n += 1
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return -1  # -1 = 目录不可达
    return n


def vcover_stats(db_path: str) -> tuple:
    """
    只读打开 report.db，返回 (photos 总行数, vcover 行数)。

    只读打开（mode=ro），避免误写老库；schema 不兼容时抛异常由调用方捕获。

    Returns (total photos rows, vcover rows), opening the DB read-only.
    """
    # UNC 路径（//NAS-server/...）的只读 URI 需要 4 个斜杠：
    # "file://" + "//NAS-server/..." → authority 为空，路径保留 //NAS-server/...
    # A UNC read-only URI needs four slashes so the host part stays a path.
    uri = "file://" + db_path.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        vcover = cur.execute(
            "SELECT COUNT(*) FROM photos WHERE filename LIKE '%\\_vcover' ESCAPE '\\'"
        ).fetchone()[0]
        return total, vcover
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描视频补跑缺口")
    parser.add_argument("--config", default="G:/code/BirdIndex/config.json")
    parser.add_argument("--json", action="store_true",
                        help="仅输出待补跑目录的 JSON 清单")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pending: List[dict] = []
    rows: List[dict] = []

    for root in cfg["photo_roots"]:
        path = root["path"]
        name = root["name"]
        db_path = os.path.join(path, ".superpicky", "report.db")
        has_db = os.path.isfile(db_path)
        videos = count_top_level_videos(path)

        # 目录不可达 → 单独标记，不算缺口
        if videos < 0:
            rows.append({"name": name, "path": path, "status": "UNREACHABLE"})
            continue

        if not has_db:
            rows.append({"name": name, "path": path, "status": "NO_DB",
                         "videos": videos})
            continue

        try:
            total_photos, vcovers = vcover_stats(db_path)
        except Exception as e:
            rows.append({"name": name, "path": path, "status": f"DB_ERROR: {e}",
                         "videos": videos})
            continue

        backlog = videos - vcovers
        rows.append({
            "name": name, "path": path, "status": "OK",
            "videos": videos, "vcovers": vcovers, "photos": total_photos,
        })
        if backlog > 0:
            pending.append({
                "name": name, "path": path,
                "videos": videos, "vcovers": vcovers, "backlog": backlog,
            })

    if args.json:
        print(json.dumps(pending, ensure_ascii=False, indent=2))
        return 0

    print(f"{'目录':<40} {'状态':<12} {'视频':>4} {'vcover':>6} {'photos':>7}")
    print("-" * 80)
    for r in rows:
        if r["status"] == "OK":
            print(f"{r['name']:<40} {'已入库':<12} {r['videos']:>4} "
                  f"{r['vcovers']:>6} {r['photos']:>7}")
        else:
            print(f"{r['name']:<40} {r['status']:<12}")
    print("-" * 80)
    print(f"待补跑目录数: {len(pending)}")
    for p in pending:
        print(f"  - {p['name']}: 视频 {p['videos']} / 已有封面 {p['vcovers']}"
              f"（缺 {p['backlog']}）\n    {p['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
