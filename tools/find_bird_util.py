import os
import sqlite3
import subprocess
import sys
import threading
from typing import Optional

import numpy as np
import rawpy
import imageio
from .utils import log_message
from .exiftool_manager import get_exiftool_manager
import glob
import shutil

from .file_utils import ensure_hidden_directory, clear_readonly_attribute


_EXIFTOOL_CLI_INFO = None
_EXIFTOOL_CLI_INFO_LOCK = threading.Lock()


def _get_exiftool_cli_info():
    global _EXIFTOOL_CLI_INFO
    if _EXIFTOOL_CLI_INFO is None:
        with _EXIFTOOL_CLI_INFO_LOCK:
            if _EXIFTOOL_CLI_INFO is None:
                manager = get_exiftool_manager()
                exiftool_path = manager.exiftool_path
                exiftool_cwd = os.path.dirname(os.path.abspath(exiftool_path))
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith('win') else 0
                _EXIFTOOL_CLI_INFO = (exiftool_path, exiftool_cwd, creationflags)
    return _EXIFTOOL_CLI_INFO


def _extract_binary_via_exiftool_cli(raw_file_path, tag):
    exiftool_path, exiftool_cwd, creationflags = _get_exiftool_cli_info()
    result = subprocess.run(
        [exiftool_path, '-b', tag, raw_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        cwd=exiftool_cwd,
        creationflags=creationflags,
        check=False
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(stderr_text or f"ExifTool exited with code {result.returncode}")
    return result.stdout

# V5.9: 伽马数学收敛到 tools/tone_curve 单一实现（find_bird_util 预览提亮与
# birdid 暗框重识别共用），此处按旧名再导出保持兼容。
# V5.9: the gamma math lives in tools/tone_curve as the single
# implementation (shared by preview brightening here and the BirdID
# dark-crop retry); re-exported under the historical names.
from tools.tone_curve import (  # noqa: E402,F401
    DEFAULT_DARK_MEAN as PREVIEW_DARK_MEAN,
    DEFAULT_MIN_GAMMA as PREVIEW_MIN_GAMMA,
    DEFAULT_TARGET_MEAN as PREVIEW_TARGET_MEAN,
    compute_brighten_gamma,
)


def preview_tone_stats(jpg_path: str) -> dict:
    """
    一次 draft 解码同时取均值 / p95 / 高光占比（提亮判据的全部输入）。

    Single draft-mode decode returning mean / p95 / highlight fraction —
    everything the brightening gates need, decoded once.

    参数 / Parameters:
        jpg_path (str): JPEG 文件路径 / path to the JPEG file.

    返回 / Returns:
        dict: {mean, p95, frac_highlight}；解码失败各项为 -1.0 / stats,
              all -1.0 on failure.
    """
    try:
        from PIL import Image
        with Image.open(jpg_path) as im:
            im.draft("L", (512, 512))  # 1/2..1/8 解码，仅影响精度不影响方向
            gray = im.convert("L")
            import numpy as _np
            arr = _np.asarray(gray, dtype=_np.float32)
            return {
                "mean": float(arr.mean()),
                "p95": float(_np.percentile(arr, 95)),
                "frac_highlight": float((arr >= 235).mean()),
            }
    except Exception:
        return {"mean": -1.0, "p95": -1.0, "frac_highlight": -1.0}


def preview_mean_luma(jpg_path: str) -> float:
    """
    快速估算预览 JPEG 的平均亮度（Rec.601 灰度均值，0-255）。

    用 PIL draft 模式按 1/8 解码（libjpeg DCT 缩放），3200x2400 预览
    只需约 10ms，避免整幅解码拖慢大批量转换。

    Estimate the mean luma (Rec.601 gray mean, 0-255) of a preview JPEG
    quickly via PIL's draft mode (1/8 DCT downscale), ~10ms for a
    3200x2400 preview instead of a full decode.

    参数 / Parameters:
        jpg_path (str): JPEG 文件路径 / path to the JPEG file.

    返回 / Returns:
        float: 平均亮度；解码失败返回 -1.0 / mean luma, or -1.0 on failure.
    """
    return preview_tone_stats(jpg_path)["mean"]


def brighten_preview_if_dark(
    jpg_path: str,
    directory_path: Optional[str] = None,
    dark_mean: float = PREVIEW_DARK_MEAN,
    target_mean: float = PREVIEW_TARGET_MEAN,
) -> Optional[float]:
    """
    暗预览原地提亮：均值低于 dark_mean 时套目标均值伽马 LUT 后重写同一文件。

    幂等性由亮度护栏保证：提亮后均值落在 [target 附近, 上限] 区间（约
    105-120），再跑时均值 ≥ dark_mean 直接跳过，不会叠加。可回退性：只写
    .superpicky/cache 里的可再生预览，删除缓存 jpg 即自动重生原始版；绝不
    接触 RAW 本体或成对 JPG 等用户文件。

    Brighten a dark preview in place: when its mean luma is below
    dark_mean, a target-mean gamma LUT is applied and the same file is
    rewritten. Idempotent by the luma guard — a brightened preview lands
    around the target mean and is skipped on re-runs, so lifts never
    stack. Reversible: only the regenerable .superpicky/cache preview is
    touched; deleting the cached JPEG regenerates the original rendition.
    RAW files and paired user JPEGs are never modified.

    参数 / Parameters:
        jpg_path (str): 预览 JPEG 路径（应位于 .superpicky/cache 内）
                        / preview JPEG path (expected inside .superpicky/cache).
        directory_path (Optional[str]): 日志归属目录 / dir for log context.
        dark_mean (float): 暗片判定均值 / dark threshold on mean luma.
        target_mean (float): 提亮目标均值 / target mean after brightening.

    返回 / Returns:
        Optional[float]: 应用了提亮时返回提亮后的均值；未触发或失败返回
                         None / post-brighten mean when applied, else None.
    """
    if not jpg_path or not os.path.exists(jpg_path):
        return None
    from tools.tone_curve import (
        DEFAULT_HIGHLIGHT_GUARD_FRAC, DEFAULT_HIGHLIGHT_LEVEL,
    )
    tone = preview_tone_stats(jpg_path)
    mean = tone["mean"]
    if mean < 0 or mean >= dark_mean:
        return None
    # V5.9.1 高光护栏：已有大片高光的背光/高反差画面不做全局提亮
    # （高光会被削掉），鸟区域由暗框重识别负责。
    # V5.9.1 highlight guard: frames with large highlight areas (backlit /
    # high-contrast) are not lifted globally — highlights would clip; the
    # bird region is the dark-crop retry's job.
    if 0 <= tone["frac_highlight"] and \
            tone["frac_highlight"] > DEFAULT_HIGHLIGHT_GUARD_FRAC:
        return None
    gamma = compute_brighten_gamma(mean, target_mean)
    if gamma >= 1.0:
        return None
    try:
        import cv2

        from tools.tone_curve import build_gamma_lut
        img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        cv2.imwrite(jpg_path, cv2.LUT(img, build_gamma_lut(gamma)),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        new_mean = preview_tone_stats(jpg_path)["mean"]
        log_message(
            f"BRIGHTEN, {os.path.basename(jpg_path)}: mean {mean:.0f} -> "
            f"{new_mean:.0f} (gamma {gamma:.2f})",
            directory_path)
        return new_mean
    except Exception as e:
        log_message(f"ERROR, brighten preview failed [{jpg_path}]: {e}",
                    directory_path)
        return None


def _auto_brighten_enabled(flag: Optional[bool]) -> bool:
    """
    解析 auto_brighten 参数：显式指定优先，否则读高级配置开关。

    Resolve the auto_brighten flag: an explicit argument wins; otherwise
    fall back to the advanced-config switch (preview_auto_brighten).

    参数 / Parameters:
        flag (Optional[bool]): 调用方显式开关（None=跟随配置）/ explicit
                               flag, None to follow config.

    返回 / Returns:
        bool: 是否启用自动提亮 / whether auto brightening is enabled.
    """
    if flag is not None:
        return bool(flag)
    try:
        from advanced_config import get_advanced_config
        return bool(get_advanced_config().preview_auto_brighten)
    except Exception:
        return True


def _dark_sidecar_path(jpg_path: str) -> str:
    """
    暗片原始渲染的伴随缓存路径（<前缀>_dark.jpg）。

    V5.9.2 双渲染架构：暗片提亮改在 RAW 开发级进行，原内嵌渲染另存为
    _dark.jpg，供分类「亮版优先、暗版重试」对比使用（实测暗图直判在
    部分样本上本就有 65-72% 正确率，单一亮版会丢掉这些判定）。

    Companion cache path of the original dark rendition (<prefix>_dark.jpg).

    V5.9.2 dual-rendition architecture: dark previews are brightened at RAW
    development level while the original embedded rendition is preserved as
    _dark.jpg for the classifier's bright-first/dark-retry compare (the dark
    image alone already IDs correctly at 65-72% on some real samples).
    """
    base, _ext = os.path.splitext(jpg_path)
    return base + "_dark.jpg"


# V5.9.3: 平坦雾片判据与 RAW 开发探测的扩展名集。
# V5.9.3: flat-fog threshold and RAW extension set for develop probes.
RAW_DEV_EXTS = (".cr3", ".nef", ".arw", ".raf", ".orf", ".dng")
FLAT_MAX_SPAN = 45.0   # 全图 p99-p1 灰阶跨度低于此值视为平坦雾片 / fog span
FLAT_MIN_MEAN = 90.0   # 均值低于此走暗片路径，不重复处理 / dark path owns <90


def find_raw_sibling(directory: str, prefix: str,
                     max_depth: int = 2) -> Optional[str]:
    """
    按前缀探测 RAW 文件路径（浅层子目录递归，兼容整理过的目录布局）。

    Probe a RAW file by prefix, searching shallow subdirectories to
    tolerate organized (species-first) layouts.

    参数 / Parameters:
        directory (str): 起始目录 / starting directory.
        prefix (str): 文件名前缀（不含扩展名）/ filename prefix.
        max_depth (int): 子目录搜索深度上限 / max subdirectory depth.

    返回 / Returns:
        Optional[str]: RAW 文件完整路径；未找到返回 None / path or None.
    """
    base_depth = directory.rstrip(os.sep).count(os.sep)
    try:
        for dirpath, dirnames, _files in os.walk(directory):
            if dirpath.rstrip(os.sep).count(os.sep) - base_depth \
                    >= max_depth:
                dirnames[:] = []  # 剪枝：不超深 / prune deeper
                continue
            for ext in RAW_DEV_EXTS:
                p = os.path.join(dirpath, prefix + ext)
                if os.path.exists(p):
                    return p
    except OSError:
        pass
    return None


def frame_is_flat(img_bgr: "np.ndarray",
                  max_span: float = FLAT_MAX_SPAN,
                  min_mean: float = FLAT_MIN_MEAN) -> bool:
    """
    判定整幅画面是否为「平坦雾片」：灰阶跨度极窄且不是暗片。

    实测（2026-09-30 奥森阴天批）：平坦剪影片全图挤在 132-154（跨度
    22-31），相机渲染 JPEG 时压掉了雾带内的真实层次；正常片跨度普遍
    >130。均值 < 90 的暗片走暗片路径（_dark 双渲染），此处排除以免双重
    处理。

    Classify a frame as "flat fog": an extremely narrow gray span while not
    being dark. Measured on an overcast batch: flat silhouettes span 22-31
    levels (132-154) with the tonal detail crushed by the camera JPEG
    rendering, while healthy frames span 130+. Frames darker than 90 belong
    to the dark-frame path and are excluded here.

    参数 / Parameters:
        img_bgr (np.ndarray): 全幅 BGR / full-frame BGR image.
        max_span (float): 平坦判定跨度上限 / span threshold.
        min_mean (float): 暗片排除线 / dark-path exclusion line.

    返回 / Returns:
        bool: 是否平坦雾片 / whether the frame is flat fog.
    """
    try:
        import cv2
        sub = img_bgr[::8, ::8]
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        p1, p99 = np.percentile(gray, [1, 99])
        return (p99 - p1) < max_span and float(gray.mean()) >= min_mean
    except Exception:
        return False


def _raw_dev_linear(raw_path: str) -> Optional["np.ndarray"]:
    """
    RAW 中性线性开发：postprocess 16-bit 线性输出（gamma=(1,1)）。

    返回 float32 [0,1] RGB 线性数组；rawpy 不可开发的格式（HEIF/X3F
    等）返回 None。线性域是所有后续运算（提亮乘法/拉伸）的精确基础，
    避免 bare bright 参数的传递偏差（实测 34→148 过冲）。

    Neutral linear RAW develop via postprocess with 16-bit linear output
    (gamma=(1,1)). Returns a float32 [0,1] RGB linear array, or None for
    formats rawpy cannot develop (HEIF/X3F...). The linear domain is the
    exact basis for later math (lift multiply / contrast stretch), avoiding
    the bare bright parameter's measured transfer deviation.

    参数 / Parameters:
        raw_path (str): RAW 文件路径 / RAW file path.

    返回 / Returns:
        Optional[np.ndarray]: HxWx3 float32 线性数组 / linear array or None.
    """
    try:
        with rawpy.imread(raw_path) as raw:
            lin = raw.postprocess(
                use_camera_wb=True,     # 机内白平衡 / camera white balance
                no_auto_bright=True,
                output_bps=16,
                gamma=(1, 1),           # 线性输出 / linear output
            )
        return lin.astype(np.float32) / 65535.0
    except Exception:
        return None


def raw_dev_stretched_crop(raw_path: str, box, ref_dims,
                           pad_ratio: float = 0.15):
    """
    平坦雾片的 RAW 级救援裁剪：线性开发 → 框内百分位拉伸 → 伽马编码。

    实测（奥森 5 张平坦剪影）：JPEG 框内拉伸放大带状伪影（有害），但
    RAW 线性域内雾带有 ~1600 个真实灰阶（14-bit vs JPEG 的 ~22 个），
    框内 [p1,p99] 线性拉伸后重编码可救回定种（11%→40%、24%→51%、
    36%→42% 三张跨过采纳线）。全图拉伸仍然有害（夜鹰垃圾桶），拉伸
    必须限制在鸟框内。

    RAW-grade rescue crop for flat fog frames: linear develop →
    in-box percentile stretch → gamma encode. Measured on five flat
    silhouettes: stretching the JPEG amplifies banding (harmful), but the
    RAW linear domain holds ~1600 real levels inside the fog band (14-bit
    vs ~22 in JPEG), and an in-box [p1,p99] linear stretch recovers IDs
    (11%→40%, 24%→51%, 36%→42% crossing the adoption line). GLOBAL stretch
    stays harmful; the stretch must stay inside the bird box.

    参数 / Parameters:
        raw_path (str): RAW 文件路径 / RAW file path.
        box (tuple): 预览坐标 (x, y, w, h)（含 padding 语义由调用方决定）
                    / box in preview coords.
        ref_dims (tuple): box 参照系 (w, h)（预览尺寸）/ reference dims.
        pad_ratio (float): 额外留边比例 / extra padding ratio.

    返回 / Returns:
        PIL.Image.Image 或 None（开发失败/框无效）/ PIL RGB crop or None.
    """
    try:
        import cv2
        from PIL import Image
        lin_f = _raw_dev_linear(raw_path)
        if lin_f is None:
            return None
        dh, dw = lin_f.shape[:2]
        rw, rh = ref_dims
        if rw <= 0 or rh <= 0:
            return None
        sx, sy = dw / float(rw), dh / float(rh)
        x, y, w, h = box
        pad = int(max(w, h) * pad_ratio)
        x1 = max(0, int(x * sx) - pad)
        y1 = max(0, int(y * sy) - pad)
        x2 = min(dw, int((x + w) * sx) + pad)
        y2 = min(dh, int((y + h) * sy) + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = lin_f[y1:y2, x1:x2]
        luma = (0.299 * crop[..., 0] + 0.587 * crop[..., 1]
                + 0.114 * crop[..., 2])
        lo, hi = np.percentile(luma, [1, 99])
        stretched = np.clip((crop - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        bgr = (stretched[:, :, ::-1] ** (1.0 / 2.2) * 255.0 + 0.5) \
            .astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return None


def _raw_dev_brighten(raw_path: str, jpg_path: str, preview_mean: float) -> bool:
    """
    RAW 开发级提亮：postprocess 线性域 bright 倍率重开发传感器数据。

    8-bit JPEG 暗部仅十几个离散灰阶，伽马拉曲线在放大量化噪声；RAW 是
    12/14-bit 线性数据，去马赛克/白平衡后在线性域乘 bright 再伽马编码，
    是真正的信息提取（实测：伽马崩溃样本 RAW 级恢复 25%→70%）。倍率由
    预览均值估算 m=(目标/均值)^2.2，上限 8（约 +3 挡）；开发后仍暗于
    判定线时用伽马补足一次（同时保证下次运行的幂等跳过）。

    RAW-development-grade brightening: re-develop the sensor data with a
    linear-domain bright multiplier in postprocess. An 8-bit JPEG has only
    a dozen usable shadow codes (gamma lifting amplifies quantization noise)
    while RAW is 12/14-bit linear — a real information gain (measured: a
    gamma-collapsed sample recovered 25%→70% at RAW grade). The multiplier
    is estimated from the preview mean, m=(target/mean)^2.2, capped at 8
    (~+3 stops); if the result still lands below the dark line, one gamma
    top-up is applied (which also restores idempotent skipping next run).

    参数 / Parameters:
        raw_path (str): RAW 文件路径（rawpy 可解的 Bayer 格式）
                       / RAW path (rawpy-decodable Bayer format).
        jpg_path (str): 预览缓存路径（将被提亮版覆盖）
                       / preview cache path (overwritten with the lift).
        preview_mean (float): 原始渲染的平均亮度 / original rendition mean.

    返回 / Returns:
        bool: 开发成功并已写回返回 True；任何失败返回 False（调用方回退
              伽马路径） / True when re-developed and written; False on any
              failure (caller falls back to the gamma path).
    """
    import time as _time

    from tools.tone_curve import DEFAULT_TARGET_MEAN
    t0 = _time.perf_counter()
    try:
        import cv2
        import numpy as _np

        # 共享的 RAW 中性线性开发（见 _raw_dev_linear）；提亮乘法与伽马
        # 编码在本函数完成，数学关系精确可控。
        # Shared neutral linear develop (see _raw_dev_linear); the lift
        # multiply and gamma encode happen here with exact math.
        lin_f = _raw_dev_linear(raw_path)
        if lin_f is None:
            raise RuntimeError("rawpy develop unavailable")
        # 用 1/8 子采样测基准编码均值（Rec.601 亮度口径，与
        # preview_tone_stats 的 PIL "L" 一致——三通道简单平均会因绿权重
        # 差异偏 10+ 点）
        # Measure the base encoded mean on a 1/8 subsample with the
        # Rec.601 luma convention, matching preview_tone_stats (PIL "L");
        # a plain channel average drifts 10+ points because green weighs
        # heavier in luma.
        sub = lin_f[::8, ::8] ** (1.0 / 2.2) * 255.0
        base_enc = float((0.299 * sub[..., 0] + 0.587 * sub[..., 1]
                          + 0.114 * sub[..., 2]).mean())
        m = float(_np.clip(
            (DEFAULT_TARGET_MEAN / max(base_enc, 1.0)) ** 2.2, 1.0, 8.0))
        lifted = _np.clip(lin_f * m, 0.0, 1.0)
        del lin_f
        bgr = (lifted[:, :, ::-1] ** (1.0 / 2.2) * 255.0 + 0.5) \
            .astype(_np.uint8)
        del lifted
        cv2.imwrite(jpg_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        del bgr
        new_mean = preview_tone_stats(jpg_path)["mean"]
        if 0 <= new_mean < PREVIEW_DARK_MEAN:
            # 极暗帧提升到上限仍偏暗：伽马补足到判定线以上，保证幂等
            # Extreme frame still dark at the cap: gamma top-up past the
            # dark line so subsequent runs skip.
            brighten_preview_if_dark(jpg_path, os.path.dirname(raw_path))
            new_mean = preview_tone_stats(jpg_path)["mean"]
        log_message(
            f"DEVBRIGHTEN, {os.path.basename(jpg_path)}: mean "
            f"{preview_mean:.0f} -> {new_mean:.0f} (raw-dev m={m:.2f}, "
            f"{_time.perf_counter() - t0:.1f}s)",
            os.path.dirname(raw_path))
        return True
    except Exception as e:
        log_message(
            f"DEBUG: raw-dev brighten unavailable for {raw_path} "
            f"({type(e).__name__}: {e}), falling back to gamma",
            os.path.dirname(raw_path))
        return False


def _brighten_dark_preview(raw_path: str, jpg_path: str) -> None:
    """
    暗预览提亮调度：原始渲染另存 _dark.jpg → RAW 开发级提亮（伽马回退）。

    门控与 brighten_preview_if_dark 相同（均值暗 + 高光护栏）。原始渲染
    必须在覆盖前另存——分类侧的暗版重试依赖它；HEIF/X3F 等 rawpy 无法
    postprocess 的格式自动落伽马路径（_dark 同样保留）。

    Dispatch for dark-preview brightening: preserve the original rendition
    to _dark.jpg → RAW-development lift (gamma fallback). Gates match
    brighten_preview_if_dark (dark mean + highlight guard). The original
    must be saved BEFORE overwriting — the classifier's dark retry depends
    on it; formats rawpy cannot postprocess (HEIF/X3F) fall back to gamma
    (the _dark copy is kept either way).

    参数 / Parameters:
        raw_path (str): RAW 文件路径 / RAW file path.
        jpg_path (str): 预览缓存路径 / preview cache path.
    """
    from tools.tone_curve import DEFAULT_HIGHLIGHT_GUARD_FRAC

    tone = preview_tone_stats(jpg_path)
    mean = tone["mean"]
    if mean < 0 or mean >= PREVIEW_DARK_MEAN:
        return
    if 0 <= tone["frac_highlight"] and \
            tone["frac_highlight"] > DEFAULT_HIGHLIGHT_GUARD_FRAC:
        return
    dark_path = _dark_sidecar_path(jpg_path)
    if not os.path.exists(dark_path):
        try:
            shutil.copy2(jpg_path, dark_path)
        except Exception:
            pass  # 另存失败只损失暗版重试，不阻断提亮 / best effort
    if _raw_dev_brighten(raw_path, jpg_path, mean):
        return
    brighten_preview_if_dark(jpg_path, os.path.dirname(raw_path))


def raw_to_jpeg(raw_file_path, auto_brighten: Optional[bool] = None):
    """
    RAW → 预览 JPEG 转换（缓存优先），并可按配置对暗预览做自动提亮。

    V5.9.2: 暗片提亮为 RAW 开发级（传感器数据线性域提升，原内嵌渲染另
    存 <前缀>_dark.jpg 供分类双渲染对比）；rawpy 无法开发时回退伽马。
    提亮只作用于 .superpicky/cache 预览缓存（可再生的应用自管文件），由
    亮度护栏保证幂等。DPP 伽马工作流（refresh_gamma_previews）需要
    「原始渲染 + DPP LUT」，必须传 auto_brighten=False，否则会在提亮版
    上叠加 LUT 造成双重提亮。

    Convert a RAW file to a preview JPEG (cache-first), optionally
    auto-brightening dark previews per config. V5.9.2: brightening is
    RAW-development grade (linear-domain lift of sensor data, with the
    original embedded rendition preserved as <prefix>_dark.jpg for the
    classifier's dual-rendition compare); gamma is the fallback when rawpy
    cannot develop the format. Only the regenerable .superpicky/cache
    preview is touched; idempotent via the luma guard. The DPP-gamma
    workflow (refresh_gamma_previews) needs the ORIGINAL rendition and must
    pass auto_brighten=False, otherwise the LUT would stack on an
    already-brightened preview.

    参数 / Parameters:
        raw_file_path (str): RAW 文件路径 / path to the RAW file.
        auto_brighten (Optional[bool]): 显式开关（None=跟随高级配置
            preview_auto_brighten）/ explicit flag, None to follow config.

    返回 / Returns:
        str: 预览 JPEG 完整路径；失败返回 None / preview path, or None.
    """
    jpg_file_path = _raw_to_jpeg_extract(raw_file_path)
    if jpg_file_path and _auto_brighten_enabled(auto_brighten):
        try:
            _brighten_dark_preview(raw_file_path, jpg_file_path)
        except Exception:
            pass  # 提亮是增强步骤，绝不阻断转换主流程 / best effort only
    return jpg_file_path


def _raw_to_jpeg_extract(raw_file_path):
    """RAW → 预览 JPEG 的原始提取实现（不含提亮，见 raw_to_jpeg）。

    Raw preview extraction implementation without brightening; see
    raw_to_jpeg for the public entry.
    """
    filename = os.path.basename(raw_file_path)
    file_prefix, file_ext = os.path.splitext(filename)
    directory_path = os.path.dirname(raw_file_path)

    # 在初步生成预览图前先移除原文件只读属性，避免后续元数据写入或移动阶段失败
    clear_readonly_attribute(raw_file_path)
    
    # V4.1.0: 使用 .superpicky/cache 目录存储临时 JPEG
    superpicky_dir = os.path.join(directory_path, ".superpicky")
    cache_dir = os.path.join(superpicky_dir, "cache", "temp_preview")
    
    # 确保目录存在并隐藏
    ensure_hidden_directory(superpicky_dir)
    ensure_hidden_directory(cache_dir)
    
    # 文件名不带 tmp_ 前缀，直接使用原名前缀
    jpg_file_path = os.path.join(cache_dir, f"{file_prefix}.jpg")
    
    if os.path.exists(jpg_file_path) and os.path.getsize(jpg_file_path) >= 128 * 1024:
        return jpg_file_path  # 返回完整路径（缓存命中且 ≥128KB，无需重新生成）
        
    if not os.path.exists(raw_file_path):
        log_message(f"ERROR, file [{filename}] cannot be found in RAW form", directory_path)
        return None

    # HEIF/HIF 格式（rawpy 不支持）：用 pillow-heif 解码全分辨率图
    heif_exts = {'.hif', '.heif', '.heic'}
    if file_ext.lower() in heif_exts:
        return _raw_to_jpeg_via_heif(raw_file_path, jpg_file_path, directory_path)

    try:
        with rawpy.imread(raw_file_path) as raw:
            thumbnail = raw.extract_thumb()
            if thumbnail is None:
                log_message(f"DEBUG: rawpy extract_thumb is None for {filename}", directory_path)
                return None
            if thumbnail.format == rawpy.ThumbFormat.JPEG:
                with open(jpg_file_path, 'wb') as f:
                    f.write(thumbnail.data)
            elif thumbnail.format == rawpy.ThumbFormat.BITMAP:
                imageio.imsave(jpg_file_path, thumbnail.data)
                # 成功转换——已由 photo_processor 的批量日志统计，无需逐文件记录
            return jpg_file_path
    except rawpy._rawpy.LibRawFileUnsupportedError:
        # LibRaw 不支持的格式（如 Sony A7M5 的已压缩 ARW）
        log_message(f"DEBUG: rawpy unsupported format for {filename}, falling back to ExifTool", directory_path)
        return _raw_to_jpeg_via_exiftool(raw_file_path, jpg_file_path, directory_path)
    except Exception as e:
        log_message(f"Error occurred while converting the RAW file:{raw_file_path}, Error: {e}", directory_path)
        # 即使是普通异常，也尝试走一次 ExifTool 回退（增加容错）
        return _raw_to_jpeg_via_exiftool(raw_file_path, jpg_file_path, directory_path)


def _raw_to_jpeg_via_heif(raw_file_path, jpg_file_path, directory_path):
    """使用 pillow-heif 解码 HEIF/HIF 文件并保存为 JPEG。"""
    try:
        import pillow_heif
        from PIL import Image as _Image
        heif_file = pillow_heif.read_heif(raw_file_path)
        img = _Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw").convert("RGB")
        img.save(jpg_file_path, "JPEG", quality=92)
        log_message(f"[HEIF] pillow-heif 解码成功: {img.size[0]}x{img.size[1]}", directory_path)
        return jpg_file_path
    except ImportError:
        log_message("pillow-heif 未安装，回退到 ExifTool", directory_path)
        return _raw_to_jpeg_via_exiftool(raw_file_path, jpg_file_path, directory_path)
    except Exception as e:
        log_message(f"HEIF 解码失败 ({os.path.basename(raw_file_path)}): {e}", directory_path)
        return _raw_to_jpeg_via_exiftool(raw_file_path, jpg_file_path, directory_path)


def _raw_to_jpeg_via_exiftool(raw_file_path, jpg_file_path, directory_path):
    """
    使用 ExifTool 从 RAW 提取内嵌 JPEG (V4.2.1: 使用统一的 ExifToolManager)
    用于 LibRaw 不支持的格式（如 Sony A7M5 的已压缩 ARW）。
    """
    # Use standalone CLI calls here because binary extraction through the
    # persistent ExifToolManager path is significantly slower for A7M5 compressed ARWs.
    
    # 按优先级尝试提取不同的内嵌图
    for tag in ["-JpgFromRaw", "-PreviewImage", "-ThumbnailImage"]:
        try:
            # 使用常驻进程提取二进制
            stdout_bytes = _extract_binary_via_exiftool_cli(raw_file_path, tag)
            
            if stdout_bytes and len(stdout_bytes) > 1000:
                with open(jpg_file_path, "wb") as f:
                    f.write(stdout_bytes)
                log_message(f"ExifTool {tag} fallback OK: {os.path.basename(raw_file_path)}", directory_path)
                return jpg_file_path
        except Exception as e:
            log_message(f"ExifTool {tag} fallback failed for {os.path.basename(raw_file_path)}: {e}", directory_path)
            continue

    # 所有方法均失败——记录友好信息，不 raise 让流程继续
    log_message(
        f"暂不支持此 RAW 格式 ({os.path.basename(raw_file_path)})，"
        "将在后续版本修复。建议使用无压缩 RAW 或 JPEG 拍摄。",
        directory_path
    )
    return None

def reset(directory, log_callback=None, i18n=None):
    """
    重置工作目录：
    1. 清理临时文件和日志
    2. 重置所有照片的EXIF元数据（Rating、Pick、Label）

    Args:
        directory: 工作目录
        log_callback: 日志回调函数（可选，用于UI显示）
        i18n: I18n instance for internationalization (optional)
    """
    def log(msg):
        """统一日志输出"""
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if not os.path.exists(directory):
        if i18n:
            log(i18n.t("errors.dir_not_exist", directory=directory))
        else:
            log(f"ERROR: {directory} does not exist")
        return False

    if i18n:
        log(i18n.t("logs.reset_start"))
        log(i18n.t("logs.reset_dir", directory=directory))
    else:
        log(f"🔄 开始重置目录: {directory}")

    # 1. 清理临时文件、日志和Crop图片
    if i18n:
        log("\n" + i18n.t("logs.clean_tmp"))
    else:
        log("\n📁 清理临时文件...")

    # 1.1 清理 _tmp 目录（包含所有临时文件、日志、crop图片等）
    tmp_dir = os.path.join(directory, ".superpicky")
    if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
        try:
            # 先逐文件清空（含 ExFAT 上的 ._* 资源分叉文件），再删目录
            import stat
            for dirpath, dirnames, filenames in os.walk(tmp_dir, topdown=False):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        try:
                            os.chmod(fpath, stat.S_IWRITE | stat.S_IREAD)
                            os.remove(fpath)
                        except Exception:
                            pass
                for dname in dirnames:
                    dpath = os.path.join(dirpath, dname)
                    try:
                        os.rmdir(dpath)
                    except Exception:
                        pass
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if i18n:
                log(i18n.t("logs.tmp_deleted"))
            else:
                log(f"  ✅ 已删除 _tmp 目录及其所有内容")
        except Exception as e:
            if i18n:
                log(i18n.t("logs.tmp_delete_failed", error=str(e)))
            else:
                log(f"  ❌ 删除 _tmp 目录失败: {e}")
            # 尝试使用系统命令强制删除（macOS/Linux）
            try:
                import subprocess
                if os.name == 'nt':
                     subprocess.run(['cmd', '/c', 'rd', '/s', '/q', tmp_dir], check=True)
                else:
                    subprocess.run(['rm', '-rf', tmp_dir], check=True)
                if i18n:
                    log(i18n.t("logs.tmp_force_delete"))
                else:
                    log(f"  ✅ 使用系统命令强制删除 _tmp 成功")
            except Exception as e2:
                if i18n:
                    log(i18n.t("logs.tmp_force_failed", error=str(e2)))
                else:
                    log(f"  ❌ 强制删除也失败: {e2}")

    # 1.2 清理旧版本的日志和CSV文件（如果存在于根目录）
    files_to_clean = [".report.csv", ".report.db", ".process_log.txt", "superpicky.log"]
    for name in files_to_clean:
        path = os.path.join(directory, name)
        if os.path.exists(path) and os.path.isfile(path):
            try:
                os.remove(path)
                if i18n:
                    log(i18n.t("logs.file_deleted", name=name))
                else:
                    log(f"  ✅ 已删除: {name}")
            except Exception as e:
                if i18n:
                    log(i18n.t("logs.delete_failed", filename=name, error=e))
                else:
                    log(f"  ❌ 删除失败 {name}: {e}")

    # 1.3 清理临时JPEG文件（tmp_*.jpg，如果有遗留在根目录的）
    tmp_jpg_pattern = os.path.join(directory, "tmp_*.jpg")
    tmp_jpg_files = glob.glob(tmp_jpg_pattern)
    tmp_jpg_files = [f for f in tmp_jpg_files if not os.path.basename(f).startswith('.')]
    if tmp_jpg_files:
        if i18n:
            log(i18n.t("logs.tmp_jpeg_found", count=len(tmp_jpg_files)))
        else:
            log(f"  发现 {len(tmp_jpg_files)} 个临时JPEG文件（tmp_*.jpg），正在删除...")
        deleted_tmp = 0
        for tmp_file in tmp_jpg_files:
            try:
                os.remove(tmp_file)
                deleted_tmp += 1
            except Exception as e:
                if i18n:
                    log(i18n.t("logs.delete_failed", filename=os.path.basename(tmp_file), error=e))
                else:
                    log(f"  ❌ 删除失败 {os.path.basename(tmp_file)}: {e}")
        if deleted_tmp > 0:
            if i18n:
                log(i18n.t("logs.tmp_jpeg_done", count=deleted_tmp))
            else:
                log(f"  ✅ 临时JPEG删除完成: {deleted_tmp} 成功")

    # 2. 删除所有XMP侧车文件（Lightroom会优先读取XMP）
    if i18n:
        log("\n" + i18n.t("logs.delete_xmp"))
    else:
        log("\n🗑️  删除XMP侧车文件...")
    xmp_pattern = os.path.join(directory, "**/*.xmp")
    xmp_files = glob.glob(xmp_pattern, recursive=True)
    # 过滤掉隐藏文件
    xmp_files = [f for f in xmp_files if not os.path.basename(f).startswith('.')]
    if xmp_files:
        if i18n:
            log(i18n.t("logs.xmp_found", count=len(xmp_files)))
        else:
            log(f"  发现 {len(xmp_files)} 个XMP文件，正在删除...")
        deleted_xmp = 0
        for xmp_file in xmp_files:
            try:
                os.remove(xmp_file)
                deleted_xmp += 1
            except Exception as e:
                if i18n:
                    log(i18n.t("logs.delete_failed", filename=os.path.basename(xmp_file), error=e))
                else:
                    log(f"  ❌ 删除失败 {os.path.basename(xmp_file)}: {e}")
        if i18n:
            log(i18n.t("logs.xmp_deleted", count=deleted_xmp))
        else:
            log(f"  ✅ XMP文件删除完成: {deleted_xmp} 成功")
    else:
        if i18n:
            log(i18n.t("logs.xmp_not_found"))
        else:
            log("  ℹ️  未找到XMP文件")

    # 3. 重置所有图片文件的EXIF元数据
    if i18n:
        log("\n" + i18n.t("logs.reset_exif"))
    else:
        log("\n🏷️  重置EXIF元数据...")

    # 支持的图片格式
    image_extensions = ['*.NEF', '*.nef', '*.CR2', '*.cr2', '*.ARW', '*.arw',
                       '*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.DNG', '*.dng']

    # 收集所有图片文件（跳过隐藏文件）
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        files = glob.glob(pattern)
        # 过滤掉隐藏文件（以.开头的文件）
        files = [f for f in files if not os.path.basename(f).startswith('.')]
        image_files.extend(files)

    # V3.9.4: 对文件列表执行去重（Windows 下 *.NEF 和 *.nef 匹配结果相同，会导致计数翻倍）
    image_files = sorted(list(set(os.path.abspath(f) for f in image_files)))

    if image_files:
        if i18n:
            log(i18n.t("logs.images_found", count=len(image_files)))
        else:
            log(f"  发现 {len(image_files)} 个图片文件")

        try:
            # 使用批量重置功能（传递log_callback和i18n）
            manager = get_exiftool_manager()
            stats = manager.batch_reset_metadata(image_files, log_callback=log_callback, i18n=i18n)

            if i18n:
                log(i18n.t("logs.batch_complete", success=stats['success'], skipped=stats.get('skipped', 0), failed=stats['failed']))
            else:
                log(f"  ✅ EXIF重置完成: {stats['success']} 成功, {stats.get('skipped', 0)} 跳过(4-5星), {stats['failed']} 失败")

        except Exception as e:
            if i18n:
                log(i18n.t("logs.exif_reset_failed", error=str(e)))
            else:
                log(f"  ❌ EXIF重置失败: {e}")
            return False
    else:
        if i18n:
            log(i18n.t("logs.no_images"))
        else:
            log("  ⚠️  未找到图片文件")

    if i18n:
        log("\n" + i18n.t("logs.reset_complete"))
    else:
        log("\n✅ 目录重置完成！")
    return True


# ============================================================================
# 高级重置 / Advanced reset（无 manifest 时按「SuperPicky 生成目录」识别并摊平）
# Advanced reset: when no manifest exists, recognize SuperPicky-generated folders
# (by name) and recursively flatten their files back to the root, leaving any
# user-created folders untouched so it can never move the wrong thing.
# ============================================================================

# 「其他鸟类」目录名（照片端 logs.folder_other_birds 的中英取值）
# "Other Birds" folder labels (zh/en values of logs.folder_other_birds).
_OTHER_BIRDS_LABELS = ("其他鸟类", "Other_Birds")
# 旧版遗留评分目录名 / legacy rating folder names
_LEGACY_RATING_FOLDERS = ("2星_良好_锐度", "2星_良好_美学")

_SUPERPICKY_FOLDER_SET = None  # 缓存：所有「SuperPicky 生成目录名」集合


def _bird_reference_db_path():
    """
    定位鸟种参考库 bird_reference.sqlite（dev + 打包均可）。
    Locate bird_reference.sqlite for both dev and frozen builds.
    """
    try:
        import birdid
        candidate = os.path.join(os.path.dirname(birdid.__file__), "data", "bird_reference.sqlite")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    # 兜底：相对项目根 / fallback relative to project root
    candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "birdid", "data", "bird_reference.sqlite")
    return candidate if os.path.exists(candidate) else None


def _build_superpicky_folder_set():
    """
    构建「SuperPicky 生成目录名」匹配集：
        鸟名(中文 chinese_simplified + 英文 english_name[空格→下划线])
        ∪ 评分目录名(中文 RATING_FOLDER_NAMES + 英文 RATING_FOLDER_NAMES_EN + legacy)
        ∪ 其他鸟类(中/英)
    burst_ 前缀单独判断（不入集合）。结果缓存。

    Build the set of folder names SuperPicky generates, so advanced reset only
    touches those (never user folders).
    """
    global _SUPERPICKY_FOLDER_SET
    if _SUPERPICKY_FOLDER_SET is not None:
        return _SUPERPICKY_FOLDER_SET

    names = set()
    # 评分目录（中英两套 + legacy）/ rating folders (zh + en + legacy)
    try:
        from constants import RATING_FOLDER_NAMES, RATING_FOLDER_NAMES_EN
        names.update(RATING_FOLDER_NAMES.values())
        names.update(RATING_FOLDER_NAMES_EN.values())
    except Exception:
        pass
    names.update(_LEGACY_RATING_FOLDERS)
    names.update(_OTHER_BIRDS_LABELS)

    # 鸟名（中文原样 + 英文空格转下划线，与照片端命名一致）
    db_path = _bird_reference_db_path()
    if db_path:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT chinese_simplified, english_name FROM BirdCountInfo")
                for cn, en in cur.fetchall():
                    if cn:
                        names.add(str(cn).strip())
                    if en:
                        names.add(str(en).strip().replace(" ", "_"))
        except Exception:
            pass

    names.discard("")
    _SUPERPICKY_FOLDER_SET = names
    return names


def _is_superpicky_folder(name: str) -> bool:
    """目录名是否为 SuperPicky 生成（鸟名/评分/其他鸟类/burst_）。"""
    if name.startswith("burst_"):
        return True
    return name in _build_superpicky_folder_set()


def is_ignorable_reset_residue(filename: str) -> bool:
    """
    判断 reset 后是否可以忽略/清理的系统元数据文件。

    参数:
    filename (str): 文件名，不需要包含完整路径。

    返回:
    bool: True 表示这是可安全删除的系统元数据残留。

    Determine whether a post-reset file is ignorable OS metadata.

    Parameters:
    filename (str): File name only; a full path is not required.

    Return:
    bool: True when the file is safe-to-remove OS metadata residue.
    """
    lower_name = filename.lower()
    return (
        filename.startswith("._")
        or lower_name in {".ds_store", "thumbs.db", "desktop.ini"}
    )


def cleanup_ignorable_reset_residue(directory: str) -> int:
    """
    递归清理 reset 目录中的系统元数据残留文件。

    参数:
    directory (str): 需要清理的目录路径。

    返回:
    int: 成功删除的残留文件数量。

    Recursively remove ignorable OS metadata residue from a reset directory.

    Parameters:
    directory (str): Directory to clean.

    Return:
    int: Number of residue files successfully removed.
    """
    removed = 0
    if not os.path.isdir(directory):
        return removed

    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if not is_ignorable_reset_residue(filename):
                continue
            path = os.path.join(root, filename)
            try:
                if os.path.islink(path) or os.path.isfile(path):
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
    return removed


def force_flatten_directory(directory, log_callback=None, i18n=None) -> dict:
    """
    高级重置核心：把「SuperPicky 生成的顶层目录」内的文件递归移回 directory 根。

    仅处理顶层目录名匹配（鸟名 中+英 / 评分名 中+英 / 其他鸟类 / burst_）的目录，
    用户自建目录原样不动。同名文件跳过、不覆盖、只移动不删除；移完删除清空的目录。
    隐藏文件 / AppleDouble(._*) / .superpicky 内部目录一律跳过（不污染根目录；
    .superpicky 由随后的 reset() 统一删除）。

    Args:
        directory: 目标目录（用户选中的根目录）
        log_callback: 日志回调
        i18n: I18n 实例

    Returns:
        dict: {'moved': int, 'skipped': int, 'dirs_removed': int, 'folders_matched': int}

    Advanced-reset core: recursively move files out of SuperPicky-generated top-level
    folders back to the root; user folders are left untouched. Conflicts are skipped
    (never overwritten); nothing is deleted except now-empty matched folders.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    stats = {"moved": 0, "skipped": 0, "dirs_removed": 0, "folders_matched": 0}
    if not os.path.isdir(directory):
        return stats

    if i18n:
        log(i18n.t("logs.adv_flatten_start"))
    else:
        log("\n🧹 高级重置：识别 SuperPicky 目录并摊平文件...")

    try:
        top_entries = sorted(os.listdir(directory))
    except Exception:
        return stats

    matched_dirs = []
    for entry in top_entries:
        entry_path = os.path.join(directory, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("."):  # 隐藏 / .superpicky 等内部目录，跳过
            continue
        if _is_superpicky_folder(entry):
            matched_dirs.append(entry_path)

    stats["folders_matched"] = len(matched_dirs)

    for top_dir in matched_dirs:
        # 递归把该匹配目录内的所有文件移回根（任意深度）
        for root, dirs, files in os.walk(top_dir):
            for fname in files:
                if fname.startswith("."):  # ._* / .DS_Store 等
                    continue
                src = os.path.join(root, fname)
                dst = os.path.join(directory, fname)
                if os.path.exists(dst):
                    stats["skipped"] += 1
                    if i18n:
                        log(i18n.t("logs.restore_skipped_exists", filename=fname))
                    else:
                        log(f"  ⏭️  同名跳过（不覆盖）: {fname}")
                    continue
                try:
                    shutil.move(src, dst)
                    stats["moved"] += 1
                except Exception as e:
                    if i18n:
                        log(i18n.t("logs.move_failed", filename=fname, error=e))
                    else:
                        log(f"  ❌ 移动失败 {fname}: {e}")

        # 删除清空的子目录（最深优先），再删顶层匹配目录
        for root, dirs, files in os.walk(top_dir, topdown=False):
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    stats["dirs_removed"] += 1
            except Exception:
                pass

    if i18n:
        log(i18n.t("logs.adv_flatten_done",
                   moved=stats["moved"], folders=stats["folders_matched"],
                   skipped=stats["skipped"]))
    else:
        log(f"  ✅ 高级重置完成：识别 {stats['folders_matched']} 个目录，"
            f"移回 {stats['moved']} 个文件，同名跳过 {stats['skipped']} 个")
    return stats
