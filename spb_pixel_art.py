#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_pixel_art — 像素风鸟图生成 CLI（输入驱动，供 BirdIndex 子进程调用）

场景：BirdIndex 网站已有鸟图文件和 bbox 检测框，想把框出的鸟生成像素风
图片。本工具不读 report.db / sidecar——输入完全来自调用方（JSON 清单或
单张参数），执行「按 bbox 方形裁剪 → (可选) 送 ComfyUI 工作流 → 收回
像素图」，结果以 JSON 回传，便于网站后台解析。

  单张模式:
    python spb_pixel_art.py --image D:/photos/IMG_1234.jpg \
        --bbox 3421 1893 886 932 --label 白鹭 \
        [--workflow pixel_art_api.json] [--out DIR] [--json]

  批量模式（BirdIndex 主用，tasks.json 见 _parse_batch_tasks）:
    python spb_pixel_art.py --batch tasks.json \
        --workflow pixel_art_api.json --out D:/pixel --json
bbox 坐标格式与 sidecar detections[].bbox 完全一致：原图像素 [x, y, w, h]
（左上角 + 宽高，EXIF 转正后的显示坐标系）；省略 bbox = 整图作源图。

输出协议（与 spb_rename_species 相同的约定）: --json 时人类日志走 stderr，
stdout 只有一行结果 JSON（ensure_ascii=False）：

  {"status": "ok|partial|error", "out_dir": "...",
   "results": [{"id", "image", "status": "ok|crop_only|error",
                "crop_path", "pixel_path", "pixel_paths",
                "seed", "prompt_id", "error"}],
   "summary": {"total", "ok", "crop_only", "failed"}}

退出码: 0=全部成功  1=部分/全部单项失败  2=致命（参数/文件/ComfyUI 不可达）。

安全设计: 只新增输出文件非破坏性（默认执行, --overwrite 才覆盖同名产物）；
不修改输入图片; 不读写 report.db; 不管理 ComfyUI 进程（调用方自启）。

Pixel-art bird image CLI driven entirely by caller input (image path +
bbox or a JSON task manifest), designed to be invoked as a subprocess by
the BirdIndex web backend. Crops a square around each bbox, optionally
runs a ComfyUI workflow on the crop, and reports results as one JSON
line on stdout (logs go to stderr).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 允许直接 `python spb_pixel_art.py` 运行（仓库根即 CWD）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.comfy_client import (
    ComfyClient,
    ComfyUIError,
    collect_output_images,
)
from tools.image_crop import smart_square_crop

# Windows 保留设备名，不能单独用作目录名 / reserved device names
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# 文件名/目录名非法字符（Windows 侧最严集合）/ illegal filename chars
_ILLEGAL_FS_CHARS = '<>:"/\\|?*'


def _force_utf8_stdio() -> None:
    """
    在 Windows 控制台强制 stdout/stderr 使用 UTF-8，避免中文输出乱码。

    Force UTF-8 on stdout/stderr so Chinese output never mojibakes on
    Windows consoles.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def _log(message: str) -> None:
    """
    人类可读日志走 stderr（--json 模式下 stdout 只留给结果 JSON）。

    参数:
    message (str): 日志行

    Human-readable log line to stderr; stdout stays pure JSON for callers.
    """
    print(message, file=sys.stderr)


# ── 名字清理 / name sanitizing ────────────────────────────────────────────

def _sanitize_name(name: str, fallback: str, max_len: int = 80) -> str:
    """
    清理成安全的文件/目录名：去 Windows 非法字符与控制字符、压缩空白、
    限长、规避保留设备名；空结果回退 fallback。

    参数:
    name (str): 原始名字（鸟种名或输出文件主干）
    fallback (str): 清理后为空时使用的回退名
    max_len (int): 最大长度，默认 80

    返回:
    str: 可安全用作目录/文件名的字符串

    Sanitize into a filesystem-safe name (Windows rules, which are the
    strictest); empty result falls back to `fallback`.
    """
    cleaned = "".join(
        ch if (ch not in _ILLEGAL_FS_CHARS and ord(ch) >= 0x20) else "_"
        for ch in str(name).strip()
    )
    cleaned = cleaned.rstrip(" .")          # Windows 目录名不能以点/空格结尾
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in _WIN_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


# ── 工作流处理 / workflow handling ────────────────────────────────────────

def _load_workflow(path: str) -> Dict[str, Any]:
    """
    加载并做基础结构校验的 ComfyUI API 格式工作流 JSON。

    参数:
    path (str): 工作流 JSON 文件路径（ComfyUI「Save (API Format)」导出）

    返回:
    Dict[str, Any]: {node_id: {"class_type": ..., "inputs": {...}}}

    异常:
    ValueError: 文件不可读、JSON 非法或结构不是 dict-of-nodes。

    Load and structurally validate an API-format workflow JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            workflow = json.load(fh)
    except OSError as exc:
        raise ValueError(f"工作流文件不可读 / cannot read workflow: {path} ({exc})")
    except json.JSONDecodeError as exc:
        raise ValueError(f"工作流 JSON 非法 / invalid workflow JSON: {path} ({exc})")
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError(
            f"工作流结构异常（应为 API 格式的非空 dict）: {path}")
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ValueError(
                f"工作流节点 {node_id} 缺少 class_type（请用 ComfyUI 的 "
                f"Save (API Format) 导出）")
    return workflow


def _resolve_loadimage_node(workflow: Dict[str, Any],
                            prefer: Optional[str]) -> str:
    """
    定位接收输入图片的 LoadImage 节点。

    参数:
    workflow (Dict[str, Any]): API 格式工作流
    prefer (Optional[str]): 调用方指定的节点 ID；None 则自动探测

    返回:
    str: 节点 ID

    异常:
    ValueError: 指定节点不存在/类型不符，或自动探测时找到 0 个/多个
                LoadImage 节点（多个时提示用 --image-node 指定）。

    Find the LoadImage node to inject the uploaded image into; either
    the caller-specified node or auto-detect (must be exactly one).
    """
    loadimage_ids = [nid for nid, node in workflow.items()
                     if node.get("class_type") == "LoadImage"]
    if prefer is not None:
        if prefer not in workflow:
            raise ValueError(f"--image-node 指定的节点不存在: {prefer}")
        if workflow[prefer].get("class_type") != "LoadImage":
            raise ValueError(
                f"--image-node 指定的节点 {prefer} 不是 LoadImage "
                f"(实际为 {workflow[prefer].get('class_type')})")
        return prefer
    if len(loadimage_ids) == 1:
        return loadimage_ids[0]
    if not loadimage_ids:
        raise ValueError(
            "工作流中没有 LoadImage 节点 / no LoadImage node in workflow")
    raise ValueError(
        f"工作流含多个 LoadImage 节点 {loadimage_ids}，"
        f"请用 --image-node 指定 / multiple LoadImage nodes, pick one")


def _apply_seed(workflow: Dict[str, Any], seed: int) -> int:
    """
    把 seed 写进所有采样类节点的 seed/noise_seed 输入并返回该值。

    多个采样节点共用同一 seed（同一 item 内可复现）；--seed 固定时批量
    内每张都用同值，缺省时每张调用前随机。

    参数:
    workflow (Dict[str, Any]): API 格式工作流（原地修改）
    seed (int): 要写入的种子值

    返回:
    int: 实际使用的 seed（记录进结果 JSON 便于复现）

    Write `seed` into every sampler node's seed/noise_seed input so each
    generation is reproducible from the reported seed value.
    """
    for node in workflow.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key in ("seed", "noise_seed"):
            if key in inputs:
                inputs[key] = seed
    return seed


# ── 裁剪 / cropping ───────────────────────────────────────────────────────

_REMBG_SESSION_CACHE: Dict[str, Any] = {}


def _rembg_session(model: str = "u2net") -> Any:
    """
    进程内复用 rembg session（按模型名缓存）。

    实测：本环境 onnxruntime 为 CPU 版，每次 new_session 的图优化要
    ~40s，而单次推理只要 ~1s；rembg 2.0.8x 的 remove() 不带 session 时
    每次调用都会新建 session。批量场景必须显式持有 session 复用，否则
    两级白底处理的耗时完全失控（BirdIndex 全鸟种批量实测发现）。

    参数:
    model (str): rembg 模型名（默认 u2net，与调参记录一致）

    返回:
    rembg session 对象

    Cache the rembg session per model; creating an onnxruntime session
    costs ~40s of graph optimization on CPU, versus ~1s per inference.
    """
    if model not in _REMBG_SESSION_CACHE:
        from rembg import new_session
        _REMBG_SESSION_CACHE[model] = new_session(model)
    return _REMBG_SESSION_CACHE[model]


def _keep_subject_alpha(alpha: Any, min_ratio: float = 0.08) -> Any:
    """
    只保留 alpha 的主体连通域，二值阈值后清掉零散碎片。

    生产实测（BirdIndex 全鸟种批量）：源图背景里有其他鸟/浪花时，
    rembg 的软 mask 会保留背景碎片（alpha>8 即保留），白底化后成为
    杂色斑点，img2img 还会将其二次固化进像素图。这里按 127 二值化后
    保留最大连通域与面积 >= min_ratio×最大域 的伴生域（翅/脚偶尔与
    躯干断开），其余归零。

    参数:
    alpha (np.ndarray): rembg 输出的 alpha 通道（0-255）
    min_ratio (float): 伴生域保留的最小面积比例

    返回:
    np.ndarray: 清理后的 alpha（软边缘保留，碎片归零）

    Keep only the main subject's connected components of the alpha mask;
    background fragments (other birds / spray) survive rembg's soft mask
    and otherwise end up baked into the pixel art as speckles.
    """
    import cv2
    import numpy as np
    hard = (alpha > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if n <= 2:  # 至多一个前景域，无需清理 / at most one blob
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = float(areas.max())
    keep_ids = [1 + i for i, a in enumerate(areas)
                if a >= max(min_ratio * biggest, 1.0)]
    keep = np.isin(labels, keep_ids)
    out = np.array(alpha, copy=True)
    out[~keep] = 0
    return out


def _whiten_background(img_bgr: Any) -> Any:
    """
    用 rembg(U2Net) 抠出鸟身后贴到纯白背景。

    实测（见 docs/specs/2026-08-29-spb-pixel-art-design.md 调参记录）：
    直接对原始裁剪图做 img2img，白底提示词压不过照片背景（低 denoise
    漏杂色、高 denoise 丢特征）；先把源图背景变白，高 denoise 重绘时
    模型才能放开发挥 Q 版构图，杂色源头被消除。

    参数:
    img_bgr (np.ndarray): BGR 裁剪图

    返回:
    np.ndarray: 白底 BGR 图

    异常:
    ImportError: rembg 未安装（调用方捕获后降级为不白化）。

    Cut the bird out with rembg and composite onto solid white; removes
    photo-background noise so high-denoise chibi conversion stays clean.
    """
    import cv2
    import numpy as np
    from rembg import remove
    from PIL import Image
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = np.array(remove(Image.fromarray(rgb),
                          session=_rembg_session()))  # RGBA，背景透明
    alpha = _keep_subject_alpha(arr[:, :, 3]) > 127
    out = np.full_like(arr[:, :, :3], 255)
    out[alpha] = arr[:, :, :3][alpha]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _crop_square(image_path: str, bbox: Optional[List[int]],
                 padding: float) -> Any:
    """
    读取输入图并裁出鸟身方形图。

    读图走 core.crop_advisor._load_image_exif_aware（PIL 路径：中文/NAS
    非 ASCII 路径安全，且自动应用 EXIF 方向——与 bbox 的坐标系约定一致，
    同 core.crop_export.export_crop 的既有组合）。有 bbox 时用
    smart_square_crop 方形化并加 padding 余量；无 bbox 时整图直用。

    参数:
    image_path (str): 输入图片路径（JPEG/PNG 等 PIL 可解码格式）
    bbox (Optional[List[int]]): [x, y, w, h] 原图像素；None=整图
    padding (float): 方形裁剪的环境余量比例（smart_square_crop 参数）

    返回:
    np.ndarray: BGR 图像（有 bbox 时为方形）

    异常:
    ValueError: 图片无法解码，或 bbox 宽高非正。
    """
    from core.crop_advisor import _load_image_exif_aware
    # CropAdvisor 的诊断日志打到 stdout，会污染 --json 协议；重定向到
    # stderr（与 spb_rename_species 对噪音库调用的处理一致）。
    # CropAdvisor logs to stdout; redirect so stdout stays pure JSON.
    with contextlib.redirect_stdout(sys.stderr):
        img = _load_image_exif_aware(image_path)
    if img is None:
        raise ValueError(
            f"图片无法解码（本工具 v1 仅支持 JPEG/PNG 等 PIL 可解码格式，"
            f"不支持 RAW）/ cannot decode image: {image_path}")
    if bbox is None:
        return img
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox 宽高必须为正 / invalid bbox: {bbox}")
    return smart_square_crop(img, (x, y, x + w, y + h),
                             padding_ratio=padding)


def _save_jpeg_pil(img_bgr: Any, out_path: str, quality: int = 95) -> None:
    """
    用 PIL 把 BGR ndarray 写成 JPEG（路径含中文/UNC 时 cv2.imwrite 不可靠，
    PIL 走 Python 层文件打开，跨平台安全）。

    参数:
    img_bgr (np.ndarray): BGR 图像
    out_path (str): 输出 .jpg 路径
    quality (int): JPEG 质量，默认 95

    异常:
    OSError: 写盘失败。

    Write a BGR ndarray as JPEG via PIL (safe for non-ASCII/UNC paths,
    where cv2.imwrite is unreliable on Windows).
    """
    import cv2  # 局部导入，避免无 cv2 环境下 --help 都失败
    from PIL import Image
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path, "JPEG", quality=quality)


# ── 任务解析 / task parsing ───────────────────────────────────────────────

@dataclass
class TaskItem:
    """
    一个待处理任务项（批量清单的一项或单张模式构造）。

    One unit of work: one image + optional bbox + label + output stem.

    属性 / Attributes:
        item_id (str): 调用方追踪 ID（缺省空串）
        image (str): 输入图片路径
        bbox (Optional[List[int]]): [x,y,w,h] 原图像素；None=整图
        label (str): 输出子目录名（鸟种名，已清理）
        name (str): 输出文件主干（已清理）
        prompt (str): 追加到正向提示词的鸟种特征描述（可为空）
    """
    item_id: str
    image: str
    bbox: Optional[List[int]]
    label: str
    name: str
    prompt: str = ""


@dataclass
class RunContext:
    """
    单次运行的共享配置与状态（避免全局变量）。

    Shared per-run configuration and state (no module-level globals).

    属性 / Attributes:
        out_dir (str): 输出根目录
        padding (float): 方形裁剪余量比例
        min_crop_px (int): 裁剪结果最小边长（像素），小于则判失败
        workflow (Optional[Dict[str, Any]]): 工作流；None=只导出裁剪图
        image_node (Optional[str]): 指定 LoadImage 节点 ID
        fixed_seed (Optional[int]): 固定 seed；None=每张随机
        timeout (float): 单张 ComfyUI 等待超时秒数
        overwrite (bool): 是否覆盖同名产物
        white_bg (bool): 白底可爱风开关：源图预白底化 + 生成后抠鸟贴纯白
        pixel_grid (int): 像素格边数（白底 alpha 对齐 + 提示词无关，须与
                          工作流缩小节点的宽高一致）
        comfy (Optional[ComfyClient]): ComfyUI 客户端；无工作流时为 None
        used_rel_paths (set): 已占用的输出相对路径，做重名自动加序号
    """
    out_dir: str
    padding: float
    min_crop_px: int
    workflow: Optional[Dict[str, Any]] = None
    image_node: Optional[str] = None
    fixed_seed: Optional[int] = None
    timeout: float = 300.0
    overwrite: bool = False
    white_bg: bool = False
    pixel_grid: int = 64
    comfy: Optional[ComfyClient] = None
    used_rel_paths: set = field(default_factory=set)


def _parse_bbox(raw: Any, where: str) -> Optional[List[int]]:
    """
    校验并归一 bbox 字段为 4 个非负整数 [x, y, w, h]。

    参数:
    raw (Any): tasks.json 里的原始字段或 CLI 已转好的列表
    where (str): 出错时定位用的描述（如 item 序号）

    返回:
    Optional[List[int]]: 归一后的 bbox；raw 为 None 时返回 None

    异常:
    ValueError: 不是 4 元列表或数值不可转整数。

    Validate/normalize a bbox field into 4 ints [x, y, w, h].
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"{where}: bbox 必须是 4 元数组 [x, y, w, h]，收到: {raw!r}")
    try:
        return [int(round(float(v))) for v in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox 含非数值 / non-numeric bbox: {raw!r}") from exc


def _parse_batch_tasks(path: str) -> Tuple[List[TaskItem], Optional[str]]:
    """
    解析批量任务清单 JSON。

    清单格式（字段与 sidecar 对齐，便于 BirdIndex 直接拼装）::

        {"out_dir": "可选，输出根目录",
         "items": [{"id": "可选", "image": "图片路径",
                    "bbox": [x, y, w, h],  # 可选，缺省整图
                    "label": "可选，鸟种名，作输出子目录",
                    "name": "可选，输出文件主干",
                    "prompt": "可选，鸟种英文特征词，追加到正向提示词"}]}

    参数:
    path (str): 清单 JSON 路径

    返回:
    Tuple[List[TaskItem], Optional[str]]: (任务列表, 清单里的 out_dir)

    异常:
    ValueError: 文件不可读、JSON 非法、items 缺失/为空或某项缺 image。

    Parse the batch manifest JSON into TaskItems plus its optional out_dir.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise ValueError(f"清单文件不可读 / cannot read tasks: {path} ({exc})")
    except json.JSONDecodeError as exc:
        raise ValueError(f"清单 JSON 非法 / invalid tasks JSON: {path} ({exc})")
    if not isinstance(data, dict):
        raise ValueError(f"清单顶层应为对象 / manifest must be an object: {path}")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"清单缺少非空 items 数组 / manifest lacks items[]: {path}")
    items: List[TaskItem] = []
    for idx, raw in enumerate(raw_items):
        where = f"items[{idx}]"
        if not isinstance(raw, dict) or not raw.get("image"):
            raise ValueError(f"{where}: 缺少 image 路径 / missing image")
        stem = os.path.splitext(os.path.basename(str(raw["image"])))[0]
        items.append(TaskItem(
            item_id=str(raw.get("id", "")),
            image=str(raw["image"]),
            bbox=_parse_bbox(raw.get("bbox"), where),
            label=_sanitize_name(str(raw.get("label", "")), "unlabeled"),
            name=_sanitize_name(str(raw.get("name", "")), stem),
            prompt=str(raw.get("prompt", "")),
        ))
    out_dir = data.get("out_dir")
    return items, (str(out_dir) if out_dir else None)


# ── 单项执行 / per-item pipeline ──────────────────────────────────────────

def _allocate_rel_path(ctx: RunContext, item: TaskItem, suffix: str) -> str:
    """
    为输出文件分配「目录内唯一」的相对路径（重名自动追加 _2/_3…）。

    参数:
    ctx (RunContext): 运行上下文（记录已占用路径）
    item (TaskItem): 任务项（取 label 子目录与 name 主干）
    suffix (str): 含扩展名的后缀，如 "_src.jpg"

    返回:
    str: 相对 out_dir 的路径，如 "白鹭/IMG_1234_src.jpg"

    Allocate a unique-in-run output relative path, appending _2/_3… on
    collision (same image reused by multiple bboxes etc.).
    """
    base = f"{item.label}/{item.name}{suffix}"
    candidate, n = base, 2
    while candidate in ctx.used_rel_paths:
        candidate = f"{item.label}/{item.name}_{n}{suffix}"
        n += 1
    ctx.used_rel_paths.add(candidate)
    return candidate


def _guard_existing(path: str, overwrite: bool) -> None:
    """
    输出已存在且未开 --overwrite 时拒绝覆盖（非破坏性默认）。

    参数:
    path (str): 目标输出路径
    overwrite (bool): 是否允许覆盖

    异常:
    ValueError: 文件已存在且未允许覆盖。

    Refuse to clobber an existing output unless --overwrite.
    """
    if os.path.exists(path) and not overwrite:
        raise ValueError(
            f"输出已存在（用 --overwrite 覆盖）/ output exists: {path}")


def _find_positive_node(workflow: Dict[str, Any]) -> Optional[str]:
    """
    定位正向提示词节点：找采样节点的 positive 输入所链接的 CLIPTextEncode。

    参数:
    workflow (Dict[str, Any]): API 格式工作流

    返回:
    Optional[str]: 正向提示词节点 ID；找不到返回 None

    Find the CLIPTextEncode node wired into a sampler's "positive" input.
    """
    for node in workflow.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or "positive" not in inputs:
            continue
        link = inputs["positive"]
        if isinstance(link, list) and len(link) >= 1 and str(link[0]) in workflow:
            return str(link[0])
    return None


def _white_bg_on_png(png_bytes: bytes, grid: int = 64) -> bytes:
    """
    像素图白底后处理：rembg 抠出鸟 → 贴纯白底 → 按像素格重对齐。

    生成端提示词无法保证纯白背景（实测总有噪点/杂色残留），白底只能在
    生成后确定性地保证：rembg 分割出主体后贴白，再把整图降采样到
    grid×grid（吸收 alpha 边缘的过渡像素）、alpha 二值化、最近邻放大回
    原尺寸——生成图本身是 grid 像素格的最近邻放大，该往返对颜色无损，
    只把边缘对齐到整数格。

    参数:
    png_bytes (bytes): 生成的 PNG 图字节
    grid (int): 像素格边数（与工作流里缩小节点的宽高一致，默认 64）

    返回:
    bytes: 处理后的 PNG 字节

    异常:
    ImportError: rembg 未安装（调用方应捕获并降级为不处理）。

    Post-process a generated pixel PNG to a guaranteed solid white
    background: rembg cutout, white composite, then grid-aligned
    down/up-sample roundtrip so mask edges snap to the pixel grid.
    """
    import io
    import cv2
    import numpy as np
    from rembg import remove
    from PIL import Image

    img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))
    cut = np.array(remove(Image.fromarray(img),
                          session=_rembg_session()))  # RGBA
    alpha = _keep_subject_alpha(cut[:, :, 3]).astype(np.float32)
    rgb = cut[:, :, :3].astype(np.float32) * (alpha / 255.0)[..., None] \
        + 255.0 * (1.0 - alpha / 255.0)[..., None]
    h, w = rgb.shape[:2]
    g = max(1, min(grid, h, w))
    small_rgb = cv2.resize(rgb, (g, g), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(alpha, (g, g), interpolation=cv2.INTER_AREA)
    small_a = np.where(small_a > 127, 255.0, 0.0)   # alpha 二值化到整数格
    out = cv2.resize(small_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(small_a, (w, h), interpolation=cv2.INTER_NEAREST)
    out[mask == 0] = 255.0
    buf = io.BytesIO()
    Image.fromarray(out.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _run_comfy_pixel(ctx: RunContext, crop_path: str, stem: str,
                    item: TaskItem) -> Tuple[List[str], Optional[int], Optional[str]]:
    """
    把裁剪图送进 ComfyUI 工作流并收回全部生成图。

    流程：上传（独立 UUID 名防多任务串图）→ 注入 LoadImage 节点 →
    写 seed → 排队 → 轮询 → 逐张 /view 下载。生成图统一落
    <label>/<name>_pixel<N><ext>，主输出取第一张。

    参数:
    ctx (RunContext): 运行上下文（工作流/节点/seed/超时/客户端）
    crop_path (str): 已落盘的源裁剪图路径
    stem (str): 上传用的 ASCII 文件主干（UUID，规避编码问题）
    item (TaskItem): 任务项（输出命名）

    返回:
    Tuple[List[str], Optional[int], Optional[str]]:
        (生成图路径列表, 使用的 seed, prompt_id)；无工作流时 ([], None, None)

    异常:
    ComfyUIError / ValueError / OSError: 任一环节失败。

    Upload the crop, run the workflow, download every output image.
    """
    if ctx.workflow is None or ctx.comfy is None:
        return [], None, None
    with open(crop_path, "rb") as fh:
        crop_bytes = fh.read()
    upload = ctx.comfy.upload_image(f"spb_pixel_{stem}.jpg", crop_bytes)

    # 每张深拷贝一份工作流，避免 seed/图片名在批量间互相污染
    import copy
    workflow = copy.deepcopy(ctx.workflow)
    node_id = _resolve_loadimage_node(workflow, ctx.image_node)
    workflow[node_id].setdefault("inputs", {})["image"] = upload["name"]
    # 追加调用方注入的鸟种提示词（如 common kingfisher）强化特征保留
    if item.prompt:
        pos = _find_positive_node(workflow)
        if pos is not None:
            text = str(workflow[pos].get("inputs", {}).get("text", ""))
            workflow[pos].setdefault("inputs", {})["text"] = \
                f"{text}, {item.prompt}" if text else item.prompt
        else:
            _log("[警告] 工作流无正向提示词节点，忽略 --prompt / "
                 "no positive node, prompt ignored")
    seed = _apply_seed(workflow,
                       ctx.fixed_seed if ctx.fixed_seed is not None
                       else random.randint(0, 2 ** 32 - 1))
    prompt_id = ctx.comfy.queue_prompt(workflow)
    entry = ctx.comfy.wait_result(prompt_id, timeout=ctx.timeout)

    out_paths: List[str] = []
    for i, img in enumerate(collect_output_images(entry), start=1):
        ext = os.path.splitext(img["filename"])[1].lower() or ".png"
        suffix = f"_pixel{ext}" if i == 1 else f"_pixel_{i}{ext}"
        out_path = os.path.join(ctx.out_dir, _allocate_rel_path(ctx, item, suffix))
        _guard_existing(out_path, ctx.overwrite)
        data = ctx.comfy.fetch_output(img["filename"], img["subfolder"],
                                      img["type"])
        if ctx.white_bg:
            try:
                data = _white_bg_on_png(data, grid=ctx.pixel_grid)
            except ImportError:
                _log("[警告] --white-bg 需要 rembg（pip install \"rembg[cpu]\"），"
                     "本次跳过白底化 / rembg not installed, skipping")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(data)
        out_paths.append(out_path)
    if not out_paths:
        raise ComfyUIError("工作流执行成功但未产出图片 / no images in outputs")
    return out_paths, seed, prompt_id


def process_item(item: TaskItem, ctx: RunContext) -> Dict[str, Any]:
    """
    执行单个任务项：裁剪 → 存源图 → (可选) ComfyUI 像素化。

    单项失败不抛出，转为 status=error 的结果行（批量不被一项卡死）。

    参数:
    item (TaskItem): 任务项
    ctx (RunContext): 运行上下文

    返回:
    Dict[str, Any]: 结果行 {id, image, status, crop_path, pixel_path,
                    pixel_paths, seed, prompt_id, error}
    """
    result: Dict[str, Any] = {
        "id": item.item_id, "image": item.image, "status": "error",
        "crop_path": None, "pixel_path": None, "pixel_paths": [],
        "seed": None, "prompt_id": None, "error": None,
    }
    try:
        img = _crop_square(item.image, item.bbox, ctx.padding)
        side = min(img.shape[0], img.shape[1])
        if side < ctx.min_crop_px:
            raise ValueError(
                f"裁剪结果边长 {side}px 低于下限 {ctx.min_crop_px}px "
                f"(可用 --min-crop-px 调低) / crop too small")
        if ctx.white_bg:
            # 白底风格前置步骤：源图先白底化，生成端才有干净构图基础
            try:
                img = _whiten_background(img)
            except ImportError:
                _log("[警告] --white-bg 需要 rembg（pip install \"rembg[cpu]\"），"
                     "本次跳过白底处理 / rembg not installed, skipping")

        src_rel = _allocate_rel_path(ctx, item, "_src.jpg")
        src_path = os.path.join(ctx.out_dir, src_rel)
        _guard_existing(src_path, ctx.overwrite)
        _save_jpeg_pil(img, src_path, quality=95)
        result["crop_path"] = src_path
        _log(f"[裁剪 OK] {item.image} → {src_rel} ({side}px)")

        if ctx.workflow is not None:
            paths, seed, prompt_id = _run_comfy_pixel(
                ctx, src_path, uuid.uuid4().hex[:10], item)
            result.update({
                "pixel_paths": paths,
                "pixel_path": paths[0] if paths else None,
                "seed": seed,
                "prompt_id": prompt_id,
            })
            result["status"] = "ok"
            _log(f"[像素图 OK] {src_rel} → {len(paths)} 张 / seed={seed}")
        else:
            result["status"] = "crop_only"
    except (ValueError, ComfyUIError, OSError) as exc:
        result["error"] = str(exc)
        _log(f"[失败] {item.image}: {exc}")
    return result


# ── 入口 / entry ──────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    构造命令行参数解析器（单张/批量二选一，详见模块 docstring）。

    Build the argparse parser; see module docstring for full usage.
    """
    parser = argparse.ArgumentParser(
        prog="spb_pixel_art",
        description="按 bbox 裁剪鸟图并可选送 ComfyUI 生成像素风图片；"
                    "输入驱动，供 BirdIndex 等外部系统子进程调用。",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch", metavar="TASKS_JSON",
                     help="批量任务清单 JSON 路径（items[] 每项含 image/bbox/label）")
    src.add_argument("--image", metavar="PATH",
                     help="单张模式：输入图片路径")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="单张模式 bbox：原图像素 [x, y, w, h]；缺省=整图")
    parser.add_argument("--label", default="",
                        help="输出子目录名（鸟种名），缺省 unlabeled")
    parser.add_argument("--name", default="",
                        help="输出文件主干，缺省取图片文件名")
    parser.add_argument("--workflow", metavar="API_JSON",
                        help="ComfyUI API 格式工作流；缺省只导出裁剪图不连 ComfyUI")
    parser.add_argument("--image-node", default=None,
                        help="注入图片的 LoadImage 节点 ID；缺省自动探测（须唯一）")
    parser.add_argument("--host", default="127.0.0.1:8188",
                        help="ComfyUI 地址 host:port（默认 127.0.0.1:8188）")
    parser.add_argument("--out", metavar="DIR",
                        help="输出根目录（批量模式也可写在清单 out_dir 字段；"
                             "单张模式缺省为 <图片目录>/pixel_art）")
    parser.add_argument("--padding", type=float, default=0.25,
                        help="方形裁剪的环境余量比例（默认 0.25）")
    parser.add_argument("--min-crop-px", type=int, default=320,
                        help="裁剪结果最小边长像素，低于判失败（默认 320）")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="单张 ComfyUI 等待超时秒数（默认 300）")
    parser.add_argument("--seed", type=int, default=None,
                        help="固定采样 seed（复现用）；缺省每张随机")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的同名输出（默认拒绝覆盖）")
    parser.add_argument("--white-bg", action="store_true",
                        help="白底可爱像素风：生成前源图白底化 + 生成后抠鸟贴纯"
                             "白并按像素格对齐（需 pip install \"rembg[cpu]\"；"
                             "配合 0.8+ denoise 的模板使用）")
    parser.add_argument("--pixel-grid", type=int, default=64,
                        help="像素格边数，须与工作流缩小节点宽高一致（默认 64）")
    parser.add_argument("--prompt", default="",
                        help="追加到正向提示词的特征描述（如 common kingfisher）；"
                             "批量模式写在每个 item 的 prompt 字段")
    parser.add_argument("--json", action="store_true",
                        help="结果 JSON 走 stdout（日志转 stderr），供子进程解析")
    return parser


def _fatal(message: str, as_json: bool) -> int:
    """
    致命错误统一出口：--json 时 stdout 输出 {"status":"error"} 载荷。

    参数:
    message (str): 错误描述
    as_json (bool): 是否处于 --json 模式

    返回:
    int: 退出码 2
    """
    _log(f"[致命] {message}")
    if as_json:
        print(json.dumps({"status": "error", "error": message},
                         ensure_ascii=False))
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI 主入口：解析参数 → 汇集任务 → 逐项执行 → 汇总输出结果 JSON。

    参数:
    argv (Optional[List[str]]): 命令行参数；None 时取 sys.argv[1:]

    返回:
    int: 退出码（0 全部成功 / 1 有单项失败 / 2 致命错误）

    Main entry: parse args, resolve tasks and output dir, optionally
    connect to ComfyUI, process items serially, emit the result JSON.
    """
    _force_utf8_stdio()
    args = _build_arg_parser().parse_args(argv)
    as_json = bool(args.json)

    # 1) 汇集任务与输出目录 / gather tasks & output dir
    try:
        if args.batch:
            items, manifest_out = _parse_batch_tasks(args.batch)
            out_dir = args.out or manifest_out
            if not out_dir:
                raise ValueError(
                    "批量模式必须用 --out 或清单 out_dir 指定输出目录 / "
                    "batch mode needs an output dir")
        else:
            if args.bbox is not None and (args.bbox[2] <= 0 or args.bbox[3] <= 0):
                raise ValueError("--bbox 的 W/H 必须为正 / bbox w/h must be positive")
            stem = os.path.splitext(os.path.basename(args.image))[0]
            items = [TaskItem(
                item_id="", image=args.image, bbox=args.bbox,
                label=_sanitize_name(args.label, "unlabeled"),
                name=_sanitize_name(args.name, stem),
                prompt=str(args.prompt or ""),
            )]
            out_dir = args.out or os.path.join(
                os.path.dirname(os.path.abspath(args.image)), "pixel_art")
    except ValueError as exc:
        return _fatal(str(exc), as_json)
    out_dir = os.path.abspath(out_dir)

    # 2) 工作流与 ComfyUI 连接 / workflow & ComfyUI connection
    workflow: Optional[Dict[str, Any]] = None
    comfy: Optional[ComfyClient] = None
    if args.workflow:
        try:
            workflow = _load_workflow(args.workflow)
            _resolve_loadimage_node(workflow, args.image_node)  # 提前校验
        except ValueError as exc:
            return _fatal(str(exc), as_json)
        comfy = ComfyClient(host=args.host)
        if not comfy.is_alive():
            comfy.close()
            return _fatal(
                f"ComfyUI 不可达 http://{args.host}（先用 --host 指对地址，"
                f"并确认服务已启动）/ ComfyUI unreachable", as_json)

    ctx = RunContext(
        out_dir=out_dir, padding=args.padding, min_crop_px=args.min_crop_px,
        workflow=workflow, image_node=args.image_node, fixed_seed=args.seed,
        timeout=args.timeout, overwrite=args.overwrite,
        white_bg=bool(args.white_bg), pixel_grid=int(args.pixel_grid),
        comfy=comfy,
    )

    # 3) 逐项串行执行（ComfyUI 自身即队列；单项失败不中断）/ process serially
    results: List[Dict[str, Any]] = []
    try:
        for item in items:
            results.append(process_item(item, ctx))
    except KeyboardInterrupt:
        # 剩余任务按原 item 补占位行（保留 id/image 便于调用方对账）
        for item in items[len(results):]:
            results.append({
                "id": item.item_id, "image": item.image, "status": "error",
                "crop_path": None, "pixel_path": None, "pixel_paths": [],
                "seed": None, "prompt_id": None,
                "error": "interrupted (Ctrl+C)",
            })
        _log("[中断] 用户中断，剩余任务标记为 interrupted")
    finally:
        if comfy is not None:
            comfy.close()

    # 4) 汇总输出 / summarize
    ok = sum(1 for r in results if r["status"] == "ok")
    crop_only = sum(1 for r in results if r["status"] == "crop_only")
    failed = sum(1 for r in results if r["status"] == "error")
    status = "error" if failed == len(results) and failed > 0 else \
        ("partial" if failed else "ok")
    summary = {"total": len(results), "ok": ok,
               "crop_only": crop_only, "failed": failed}
    payload = {"status": status, "out_dir": out_dir,
               "results": results, "summary": summary}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        _log(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
