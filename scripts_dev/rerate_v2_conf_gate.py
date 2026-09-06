#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 配额定星重算（薄壳）— 逻辑已收编为 CLI `rerate-v2` 子命令的唯一实现
core/rerate_v2.py，本脚本仅保留独立入口便于在 scripts_dev 场景直呼。

前置：目录跑过 process；无鸟照片预览被清理时 pHash 聚类覆盖会下降
（核心实现会用日志鸟种标签 + DB 现成指标重建定星输入）。

用途 / Usage:
    python scripts_dev/rerate_v2_conf_gate.py <照片目录> [--min-conf 0.4]
        [--quota3 20] [--quota2 30] [--execute]

默认 dry-run：只打印分布对比，不写库。--execute 时先备份 report.db 再更新
rating/caption/sidecar（单一写者原则：本脚本即 SuperPicky 运维上下文，
零接触照片文件）。自校验失败（复现存库评级/分组校验和不一致）拒绝写库。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rerate_v2 import rerate_directory  # noqa: E402


def main() -> int:
    """
    命令行入口：参数解析后委托 core.rerate_v2.rerate_directory。

    返回:
        int: core 实现的退出码（0 成功；1 前置错误；2 自校验失败）

    Thin CLI shell delegating to the single implementation in core.
    """
    ap = argparse.ArgumentParser(description="V2 配额定星重算（不重跑识别）")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="置信度门槛（默认 0.4，与批量识别工作流定档一致）")
    ap.add_argument("--quota3", type=float, default=None,
                    help="3★ 配额%%（默认跟随配置 custom_quota3）")
    ap.add_argument("--quota2", type=float, default=None,
                    help="2★ 配额%%（默认跟随配置 custom_quota2）")
    ap.add_argument("--execute", action="store_true",
                    help="写库（默认 dry-run；写前自动备份）")
    args = ap.parse_args()
    return rerate_directory(args.directory, min_conf=args.min_conf,
                            quota3=args.quota3, quota2=args.quota2,
                            execute=args.execute, log=print)


if __name__ == "__main__":
    sys.exit(main())
