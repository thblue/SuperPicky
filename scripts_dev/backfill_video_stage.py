#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频阶段补跑 runner——只补「照片已入库但视频未处理」的目录，不重跑照片。

背景：视频阶段（V-cover）上线前跑过的目录，report.db 里没有 *_vcover 封面行。
本脚本对这些目录**只执行 core.video_stage.process_directory_videos**：
  - 天然幂等：已有封面记录的视频自动跳过（video_stage 内置）；
  - 照片零接触：不跑检测/评分/定星/organize，已入库照片与文件位置完全不动；
  - 跑前把该目录 report.db 备份为 .superpicky/report.db.bak_video_backfill_<时间戳>；
  - flat 布局下封面与视频不移动文件（与目录内照片的 flat 现状一致）。

参数与 batch-identify skill 的标准命令对齐：
  process -i <目录> --birdid-country <CC> -c 40
  → ai_confidence=40、auto_identify=True、采纳线=birdid_confidence(50)、
    其余全部回落 advanced_config（GUI 同源）。

Backfill the per-directory video stage for folders whose photos are already
in report.db but whose videos were never processed. Idempotent by design
(existing *_vcover rows are skipped inside video_stage); photos are never
re-processed. Each target's report.db is backed up before any write.

参数 / Args:
    --config   : BirdIndex config.json（目录清单来源）
    --ids      : 逗号分隔的 photo_roots id 子集（默认=自动探测全部有缺口的目录）
    --execute  : 真正写库；缺省为 dry-run（只打印将处理的目录与视频数）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 海外目录的鸟种国家码（封面无 GPS，地理过滤回落到该国家层）；
# 未列出的目录一律 CN。香港 HK（34 种）、日本 JP（698 种）在 geo_distribution.db
# 里均有国家层，层空时 geo_filter 自动放宽到不过滤，安全。
# Overseas folders get their real ISO country code; everything else defaults CN.
COUNTRY_BY_ID = {
    "xianggang-2014": "HK",   # 2014.3 香港
    "dongjing-2024": "JP",    # 2024.7 东京
}
DEFAULT_COUNTRY = "CN"

# 与 batch-identify skill 定档一致的 AI 置信度门槛（百分制）
AI_CONFIDENCE = 40


def find_backlog(config_path: str) -> list:
    """
    复用 scan_video_backlog 的只读扫描，返回有视频缺口的 photo_roots 条目。

    参数:
        config_path (str): BirdIndex config.json 路径

    返回:
        list[dict]: 带 backlog 的目录（name/path/videos/vcovers/backlog/id）
    """
    from scripts_dev.scan_video_backlog import count_top_level_videos, vcover_stats

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pending = []
    for root in cfg["photo_roots"]:
        path = root["path"]
        db_path = os.path.join(path, ".superpicky", "report.db")
        if not os.path.isfile(db_path):
            continue
        videos = count_top_level_videos(path)
        if videos <= 0:
            continue
        _, vcovers = vcover_stats(db_path)
        if videos - vcovers > 0:
            pending.append({**root, "videos": videos, "vcovers": vcovers})
    return pending


def build_settings(country_code: str):
    """
    构造与 CLI 标准命令等价的 ProcessingSettings + 内存 advanced_config。

    复用 tools.cli_settings 的两条解析通道（A: settings / B: config 内存覆盖），
    等价于：process -i <目录> --birdid-country <CC> -c 40

    参数:
        country_code (str): ISO 国家码（封面无 GPS 时的地理过滤兜底）

    返回:
        (ProcessingSettings, AdvancedConfig)
    """
    from advanced_config import get_advanced_config
    from tools.cli_settings import apply_config_overrides, resolve_processing_settings

    adv_config = get_advanced_config()
    # 与 superpicky_cli.process 的 argparse 命名空间对齐：只显式给
    # auto_identify / confidence / birdid_country，其余 None → 回落配置。
    args = SimpleNamespace(
        sharpness=None, nima_threshold=None, confidence=AI_CONFIDENCE,
        skill_level=None, flight=None, burst=None, exposure=None,
        exposure_threshold=None, min_sharpness=None, min_nima=None,
        picked_top=None, xmp=None, arw_write_mode=None, metadata_mode=None,
        folder_layout=None, name_format=None, auto_identify=True,
        ebird=None, birdid_country=country_code, birdid_region=None,
        birdid_threshold=None, keep_temp_files=None, cleanup=True,
        cleanup_days=None,
    )
    apply_config_overrides(args, adv_config)
    settings = resolve_processing_settings(args, adv_config)
    return settings, adv_config


def backup_report_db(dir_path: str) -> str:
    """
    把目录的 report.db 备份为 .superpicky/report.db.bak_video_backfill_<时间戳>。

    参数:
        dir_path (str): 照片目录

    返回:
        str: 备份文件路径
    """
    db_path = os.path.join(dir_path, ".superpicky", "report.db")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f"{db_path}.bak_video_backfill_{ts}"
    shutil.copy2(db_path, bak)
    return bak


def main() -> int:
    parser = argparse.ArgumentParser(description="补跑视频阶段（幂等，不碰照片）")
    parser.add_argument("--config", default="G:/code/BirdIndex/config.json")
    parser.add_argument("--ids", default=None,
                        help="逗号分隔 photo_roots id 子集；缺省=全部有缺口目录")
    parser.add_argument("--execute", action="store_true",
                        help="真正写库（缺省 dry-run）")
    args = parser.parse_args()

    pending = find_backlog(args.config)
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        pending = [p for p in pending if p["id"] in wanted]

    if not pending:
        print("没有需要补跑的目录。/ No video backlog found.")
        return 0

    total_backlog = sum(p["videos"] - p["vcovers"] for p in pending)
    print(f"待补跑 {len(pending)} 个目录，共 {total_backlog} 个视频"
          f"{'（DRY-RUN，加 --execute 写库）' if not args.execute else ''}")
    for p in pending:
        cc = COUNTRY_BY_ID.get(p["id"], DEFAULT_COUNTRY)
        print(f"  - {p['name']}: 视频 {p['videos']}（缺 {p['videos'] - p['vcovers']}）"
              f" country={cc}\n    {p['path']}")
    if not args.execute:
        return 0

    from core.video_stage import process_directory_videos
    from tools.report_db import ReportDB

    summary = []
    t_all = time.time()
    for p in pending:
        dir_path = p["path"]
        cc = COUNTRY_BY_ID.get(p["id"], DEFAULT_COUNTRY)
        print(f"\n{'=' * 70}\n▶ {p['name']}（country={cc}）\n  {dir_path}")

        settings, adv_config = build_settings(cc)
        bak = backup_report_db(dir_path)
        print(f"  已备份 report.db → {os.path.basename(bak)}")

        db = ReportDB(dir_path)
        t0 = time.time()
        try:
            stats = process_directory_videos(
                dir_path=dir_path,
                settings=settings,
                config=adv_config,
                report_db=db,
                organize_files=True,   # flat 布局下为 no-op，不动文件
                max_frames=adv_config.config.get("video_max_frames", 60),
                yolo_threshold=adv_config.config.get("video_yolo_threshold", 0.5),
                log=lambda msg, level="info": print(msg),
            )
        finally:
            db.close()

        took = time.time() - t0
        summary.append({
            "id": p["id"], "name": p["name"], "path": dir_path,
            "videos": p["videos"], "backup": os.path.basename(bak),
            "covers_created": stats.covers_created, "skipped": stats.skipped,
            "failed": stats.failed, "has_bird": stats.has_bird,
            "adopted": stats.adopted, "no_bird": stats.no_bird,
            "low_conf": stats.low_conf, "organized": stats.organized,
            "sidecars": stats.sidecars, "seconds": round(took, 1),
        })

    print(f"\n{'=' * 70}\n✅ 全部完成，总耗时 {time.time() - t_all:.0f}s")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
