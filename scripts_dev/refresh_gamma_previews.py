#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPP 伽马调整预览刷新：解析 CR3 内 CanonVRD recipe 的伽马参数，
把等效提亮应用到 .superpicky 缓存缩略图。

背景（2026-09-21 玉渊潭）：用户在 Canon DPP「工具调色板 → 伽马调整」
拉亮了 21 张飞版暗片。DPP 把 recipe 以 CanonVRD/DR4 二进制 trailer
写回了 CR3（exiftool -CanonVRD:all 可读），但 SuperPicky 的缩略图
抽的是机身内嵌 JPEG（拍摄时的原始暗图），两者互不相通——浏览库里
缩略图仍然黑。

本脚本把 DPP recipe 还原为近似 LUT 应用到缓存预览：
- GammaMidPoint m（负值=提亮）→ 幂律 gamma g = 2^m（m<1 → g<1 → 提亮），
  LUT: y = (x/255)^g；
- GammaWhitePoint 正值（高光端扩展）在 v1 先忽略——主视觉量来自中点；
- 只写 temp_preview 缓存（可再生的应用自管文件），不碰 CR3/DB/sidecar；
  回退 = 删除缓存 jpg，raw_to_jpeg 会重新生成原始版。

读取/LUT/应用的实现在 tools/dpp_gamma.py（V5.9.4 起与跑批内 M1 共享
同一实现）；本脚本保留 CLI 编排：找有编辑的 CR3 → 重抽原始预览 →
套 LUT。V5.9.4 起跑批时已对「本次新抽取」的预览先套 DPP LUT（人工
优先、跳过自动提亮），本脚本的定位收窄为：老批次回补 + 跑批之后
才做的 DPP 编辑。

用法:
    python scripts_dev/refresh_gamma_previews.py <照片目录> [--dry]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dpp_gamma import (  # noqa: E402
    apply_lut_to_jpeg,
    build_lut,
    read_bgr,
    read_gamma_params,
)
from tools.find_bird_util import raw_to_jpeg  # noqa: E402


def main() -> int:
    """
    主流程：找有 DPP 伽马编辑的 CR3 → 生成 LUT → 刷新缓存预览。

    返回:
        int: 0 成功；1 无待处理
    """
    ap = argparse.ArgumentParser(description="DPP 伽马调整预览刷新")
    ap.add_argument("directory", help="照片目录")
    ap.add_argument("--dry", action="store_true",
                    help="只报告每张的伽马值与理论提亮，不写缓存")
    ap.add_argument("--only", default=None,
                    help="只处理指定前缀（逗号分隔），调试用")
    args = ap.parse_args()

    root = os.path.normpath(args.directory)
    exiftool = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'exiftools_win', 'exiftool.exe')
    cr3s = sorted(
        os.path.join(root, f) for f in os.listdir(root)
        if f.lower().endswith('.cr3'))
    if not cr3s:
        print("目录内无 CR3")
        return 1

    params = read_gamma_params(cr3s, exiftool)
    if args.only:
        keep = set(args.only.split(','))
        params = {k: v for k, v in params.items() if k in keep}
    if not params:
        print("未发现 DPP 伽马编辑（无 CanonVRD GammaMidPoint）")
        return 1

    print(f"📁 {root}\n   DPP 伽马编辑 {len(params)} 张")
    for prefix, (mid, wp) in params.items():
        m = float(mid)
        lut = build_lut(m)
        cache = os.path.join(root, '.superpicky', 'cache',
                             'temp_preview', prefix + '.jpg')
        raw = os.path.join(root, prefix + '.CR3')
        # dry 只读（V5.9.4/W1 修正）：不删缓存、不重抽——老实现「先重抽
        # 原始预览再 continue」会把已套 LUT 的缓存（W1 跑批内产出或本脚本
        # 上次真跑的结果）静默回退成暗图。dry 仅报告 recipe 与当前缓存亮度。
        # Read-only dry (fixed in V5.9.4/W1): never delete/re-extract the
        # cache — the old "re-extract then continue" silently regressed an
        # already-LUTed preview (from in-batch W1 or a previous real run).
        if args.dry:
            cur = read_bgr(cache)
            c_mean = float(cur.mean()) if cur is not None else -1.0
            print(f"  {prefix}: 中点{mid} 白点{wp} → "
                  f"LUT[128]={lut[128]}（当前缓存亮度 {c_mean:.0f}）")
            continue
        # 幂等保证（真跑）：先删缓存从 CR3 重抽原始预览，避免在已提亮的图
        # 上二次应用 LUT（重复运行不会叠加提亮）。auto_brighten=False 必须
        # 显式传：V5.9 起 raw_to_jpeg 默认会给暗预览做目标均值提亮，这里
        # 的语义是「原始渲染 + DPP LUT」，叠加自动提亮会双重变亮。
        # Idempotency (real run): always re-extract the original preview
        # from the CR3 so re-running never stacks the brightening on an
        # edited cache. auto_brighten=False is REQUIRED: since V5.9
        # raw_to_jpeg auto-brightens dark previews by default, while this
        # script's contract is "original rendition + DPP LUT" — stacking
        # the auto gamma would double-brighten.
        if os.path.exists(cache):
            os.remove(cache)
        if raw_to_jpeg(raw, auto_brighten=False) is None:
            print(f"  {prefix}: 预览生成失败，跳过")
            continue
        before = read_bgr(cache)
        b_mean = float(before.mean()) if before is not None else -1.0
        after = apply_lut_to_jpeg(cache, lut)
        if after is None:
            print(f"  {prefix}: LUT 应用失败")
        else:
            print(f"  {prefix}: 中点{mid} → 亮度 {b_mean:.0f} → {after:.0f}")
    if args.dry:
        print("（dry 模式，只读未动缓存）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
