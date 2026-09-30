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

用法:
    python scripts_dev/refresh_gamma_previews.py <照片目录> [--dry]
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.find_bird_util import raw_to_jpeg  # noqa: E402


def read_gamma_params(cr3_paths: List[str],
                      exiftool: str) -> Dict[str, Tuple[str, str]]:
    """
    批量读 CR3 的 CanonVRD 伽马参数。

    大目录（如 640 张 UNC 路径）会超出 Windows 32K 命令行上限
    （WinError 206），因此按块分批调用 exiftool 再合并结果。

    参数:
        cr3_paths (List[str]): CR3 绝对路径
        exiftool (str): exiftool 可执行文件路径

    返回:
        Dict[str, Tuple[str, str]]: {文件名前缀: (中点值, 白点值)}；
        未编辑（无 CanonVRD 伽马区）的文件不在字典里
    """

    def _read_batch(batch: List[str]) -> Dict[str, Tuple[str, str]]:
        """读单批文件，返回 {前缀: (中点, 白点)}。Read one batch."""
        out = subprocess.run(
            [exiftool, '-s3', '-csv', '-FileName',
             '-CanonVRD:GammaMidPoint', '-CanonVRD:GammaWhitePoint',
             '-ext', 'CR3', *batch],
            capture_output=True, timeout=600)
        lines = out.stdout.decode('utf-8', 'replace').strip().splitlines()
        if len(lines) < 2:
            return {}
        header = lines[0].split(',')
        ix_fn = header.index('FileName')
        ix_mid = header.index('GammaMidPoint') if 'GammaMidPoint' in header else -1
        ix_wp = header.index('GammaWhitePoint') if 'GammaWhitePoint' in header else -1
        batch_result: Dict[str, Tuple[str, str]] = {}
        for ln in lines[1:]:
            cols = ln.split(',')
            fn = cols[ix_fn] if ix_fn < len(cols) else ''
            mid = cols[ix_mid].strip() if 0 <= ix_mid < len(cols) else ''
            wp = cols[ix_wp].strip() if 0 <= ix_wp < len(cols) else ''
            if mid:
                prefix = os.path.splitext(fn)[0]
                batch_result[prefix] = (mid, wp)
        return batch_result

    # 每批 100 个文件：单条命令约 8-9K 字符，远低于 Windows 32K 上限
    # 100 files per batch: ~8-9K chars per command, well under the 32K limit
    result: Dict[str, Tuple[str, str]] = {}
    chunk_size = 100
    for i in range(0, len(cr3_paths), chunk_size):
        result.update(_read_batch(cr3_paths[i:i + chunk_size]))
    return result


def build_lut(mid_point: float) -> np.ndarray:
    """
    DPP 伽马中点 → 256 级 LUT。

    DPP 中点滑块负值=提亮（实测 21 张全负、用户描述「亮度提高」）。
    幂律近似：g = 2^m，y = (x/255)^g。m=-1.65 → g≈0.32，中灰 0.5 → 0.80。

    参数:
        mid_point (float): CanonVRD GammaMidPoint 原始值

    返回:
        np.ndarray: uint8[256] 查找表
    """
    g = 2.0 ** mid_point
    x = np.arange(256, dtype=np.float64) / 255.0
    y = np.clip(x ** g, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def read_bgr(path: str) -> Optional[np.ndarray]:
    """中文/UNC 安全读图。/ UNC-safe read."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def apply_lut_to_jpeg(jpg_path: str, lut: np.ndarray) -> Optional[float]:
    """
    对缓存 JPEG 应用 LUT 并写回（原地刷新，JPEG 质量参数 95）。

    参数:
        jpg_path (str): temp_preview 缓存 JPEG 路径
        lut (np.ndarray): uint8[256] 查找表

    返回:
        Optional[float]: 应用后的平均亮度（0-255）；读写失败返回 None
    """
    img = read_bgr(jpg_path)
    if img is None:
        return None
    out = cv2.LUT(img, lut)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return None
    buf.tofile(jpg_path)
    return float(out.mean())


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
        # 幂等保证：先删缓存从 CR3 重抽原始预览，避免在已提亮的图上
        # 二次应用 LUT（重复运行不会叠加提亮）。auto_brighten=False 必须
        # 显式传：V5.9 起 raw_to_jpeg 默认会给暗预览做目标均值提亮，这里
        # 的语义是「原始渲染 + DPP LUT」，叠加自动提亮会双重变亮。
        # Idempotency: always re-extract the original preview from the CR3
        # so re-running never stacks the brightening on an edited cache.
        # auto_brighten=False is REQUIRED: since V5.9 raw_to_jpeg
        # auto-brightens dark previews by default, while this script's
        # contract is "original rendition + DPP LUT" — stacking the auto
        # gamma would double-brighten.
        if os.path.exists(cache):
            os.remove(cache)
        if raw_to_jpeg(raw, auto_brighten=False) is None:
            print(f"  {prefix}: 预览生成失败，跳过")
            continue
        before = read_bgr(cache)
        b_mean = float(before.mean()) if before is not None else -1.0
        if args.dry:
            print(f"  {prefix}: 中点{mid} 白点{wp} → "
                  f"LUT[128]={lut[128]}（原图亮度 {b_mean:.0f}）")
            continue
        after = apply_lut_to_jpeg(cache, lut)
        if after is None:
            print(f"  {prefix}: LUT 应用失败")
        else:
            print(f"  {prefix}: 中点{mid} → 亮度 {b_mean:.0f} → {after:.0f}")
    if args.dry:
        print("（dry 模式，未写缓存）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
