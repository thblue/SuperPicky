#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_review — 多鸟分类结果人工审阅工具

在原图上叠加 bird_detections 的 bbox/轮廓/物种/置信度，生成一张审阅图，
便于人工检查逐鸟分类是否正确。

数据来源（自动降级）：
    1. <照片目录>/.superpicky/meta/<前缀>.json   —— sidecar（首选）
    2. <照片目录>/.superpicky/report.db           —— 数据库回退
       （批处理在 sidecar 导出功能之前跑出的旧结果也能审）

用法:
    python spb_review.py 照片.jpg [更多照片...]
    python spb_review.py 照片目录
    python spb_review.py 照片.NEF --out D:/review --max-side 2400

输出: 默认 <照片目录>/.superpicky/review/<前缀>_review.jpg
标注: 红框=主鸟(评分对象) · 绿框=已识别 · 灰框=未识别/置信度不足
      每框标注 序号 物种名 置信度%；中文用系统字体渲染（无 CJK 字体时
      回退英文物种名）。

Overlay per-bird detection/classification results onto the original
photo for manual review. Reads sidecar JSON first, falls back to
report.db. Originals are never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 审阅图长边上限（原图 30MP+ 直接看不动，统一缩放）
# Max long side of the annotated review image.
DEFAULT_MAX_SIDE = 2400

# 支持的图片扩展（RAW 走 birdid.load_image 解码）
# Supported extensions; RAW files decode via birdid.load_image.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff",
              ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".raf",
              ".orf", ".rw2", ".pef", ".srw", ".heic", ".heif", ".avif"}

# 通用 CJK 字体候选（Windows/macOS/Linux 各自的系统字体）
# Common CJK font candidates across platforms.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

_font_cache: dict = {"pil_font": None, "checked": False}


def _get_cjk_font(size: int = 22):
    """
    加载一个可渲染中文的 PIL TrueType 字体，找不到返回 None。

    参数:
    size (int): 字号（像素）

    返回:
    ImageFont 对象或 None（调用方需回退英文标注）

    Load a PIL font capable of rendering CJK text, else None.
    """
    if not _font_cache["checked"]:
        _font_cache["checked"] = True
        try:
            from PIL import ImageFont
            for path in _FONT_CANDIDATES:
                if os.path.exists(path):
                    _font_cache["pil_font"] = (path, ImageFont)
                    break
        except ImportError:
            _font_cache["pil_font"] = None
    entry = _font_cache["pil_font"]
    if entry is None:
        return None
    path, ImageFont = entry
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def _read_image(path: str) -> Optional[np.ndarray]:
    """
    读取图片为 BGR（兼容中文路径；RAW/HEIF 回退 birdid.load_image）。

    参数:
    path (str): 图片路径

    返回:
    Optional[np.ndarray]: BGR 图像；失败返回 None

    Read an image as BGR, Chinese-path safe; RAW/HEIF via birdid.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except OSError:
        pass
    try:
        from birdid.bird_identifier import load_image
        import PIL.Image
        pil = load_image(path)
        if pil is not None:
            return cv2.cvtColor(np.array(pil.convert("RGB")),
                                cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return None


def _load_detections(photo_path: str
                     ) -> Tuple[Optional[List[dict]], Optional[dict]]:
    """
    加载某照片的多鸟检测数据：sidecar JSON 优先，report.db 回退。

    参数:
    photo_path (str): 照片文件路径

    返回:
    Tuple[Optional[List[dict]], Optional[dict]]:
        (detections, sidecar_json)。detections 每项含
        bbox[x,y,w,h](原图坐标)/polygon/is_selected/species{...}；
        找不到数据时 (None, None)

    Load detections for one photo: sidecar JSON first, then report.db.
    """
    directory = os.path.dirname(os.path.abspath(photo_path))
    prefix = os.path.splitext(os.path.basename(photo_path))[0]

    # 1) sidecar JSON（批处理导出的对外契约）
    sidecar = os.path.join(directory, ".superpicky", "meta",
                           f"{prefix}.json")
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            dets = data.get("detections") or []
            if dets:
                return dets, data
        except (OSError, ValueError):
            pass

    # 2) report.db 回退（sidecar 出现之前的批结果）
    db_path = os.path.join(directory, ".superpicky", "report.db")
    if os.path.exists(db_path):
        try:
            from tools.report_db import ReportDB
            db = ReportDB(directory)
            rows = db.get_detections(prefix)
            db._conn.close()
            if rows:
                dets = []
                for row in rows:
                    polygon = None
                    if row.get("mask_polygon"):
                        try:
                            polygon = json.loads(row["mask_polygon"])
                        except (TypeError, ValueError):
                            pass
                    has_species = bool(row.get("species_cn")
                                       or row.get("species_en"))
                    dets.append({
                        "index": row.get("bird_index"),
                        "is_selected": bool(row.get("is_selected")),
                        "bbox": [row.get("bbox_x"), row.get("bbox_y"),
                                 row.get("bbox_w"), row.get("bbox_h")],
                        "polygon": polygon,
                        "species": ({"cn": row.get("species_cn"),
                                     "en": row.get("species_en"),
                                     "confidence": row.get("species_confidence")}
                                    if has_species else None),
                    })
                if dets:
                    return dets, None
        except Exception:
            pass
    return None, None


def _draw_label(img: np.ndarray, xy: Tuple[int, int], text: str,
                color: Tuple[int, int, int], has_cjk: bool) -> None:
    """
    在图上画一条带背景的标注文字（中文走 PIL，否则 cv2）。

    参数:
    img (np.ndarray): BGR 图像（原地修改）
    xy (Tuple[int,int]): 文字左下角位置
    text (str): 标注内容
    color (Tuple[int,int,int]): BGR 文字色
    has_cjk (bool): 是否已确认有 CJK 字体（决定渲染路径）

    Draw a label with a solid background; PIL when CJK is available.
    """
    x, y = xy
    if has_cjk:
        from PIL import Image, ImageDraw
        font = _get_cjk_font(22)
        if font is not None:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            bbox = draw.textbbox((x, y - 24), text, font=font)
            draw.rectangle([bbox[0] - 4, bbox[1] - 2,
                            bbox[2] + 4, bbox[3] + 2],
                           fill=(color[2], color[1], color[0]))
            draw.text((x, y - 24), text, font=font,
                      fill=(color[2], color[1], color[0]))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
    # 无 CJK 字体：cv2 原生（仅 ASCII 可靠）
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x - 2, y - th - 8), (x + tw + 2, y), color, -1)
    cv2.putText(img, text, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def annotate_photo(photo_path: str, out_path: str,
                   max_side: int = DEFAULT_MAX_SIDE) -> bool:
    """
    生成一张照片的审阅叠加图。

    参数:
    photo_path (str): 原照片路径（不被修改）
    out_path (str): 输出 JPEG 路径
    max_side (int): 输出图长边上限

    返回:
    bool: 是否成功生成

    Render the annotated review image for one photo.
    """
    detections, _sidecar = _load_detections(photo_path)
    if not detections:
        print(f"  ⚠️ 无检测数据（未处理过或多鸟关闭）: {photo_path}")
        return False

    img = _read_image(photo_path)
    if img is None:
        print(f"  ⚠️ 无法读取图片: {photo_path}")
        return False
    h0, w0 = img.shape[:2]

    # 缩放到审阅尺寸；bbox/多边形坐标按同比例换算
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_AREA)

    has_cjk = _get_cjk_font() is not None
    identified = sum(1 for d in detections if d.get("species"))

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or any(v is None for v in bbox):
            continue
        x, y, w, h = [v * scale for v in bbox]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        is_sel = bool(det.get("is_selected"))
        species = det.get("species")
        if is_sel:
            color = (0, 0, 255)          # 红：主鸟
        elif species:
            color = (0, 180, 0)          # 绿：已识别
        else:
            color = (160, 160, 160)      # 灰：未识别
        thickness = max(1, int(round(3 * scale)) if is_sel
                        else round(1.5 * scale))

        # 轮廓多边形（细线，同色）
        poly = det.get("polygon")
        if poly:
            pts = np.array([[int(p[0] * scale), int(p[1] * scale)]
                            for p in poly], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color,
                          thickness=1, lineType=cv2.LINE_AA)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 标注文字：#序号 [主] 物种 置信度%
        idx = det.get("index", "?")
        if species:
            name = species.get("cn") or species.get("en") or "?"
            if not has_cjk and species.get("en"):
                name = species["en"]     # 无中文字体时用英文名
            conf = species.get("confidence")
            label = f"#{idx} {name}"
            if conf is not None:
                label += f" {conf:.0f}%"
        else:
            label = f"#{idx} 未识别" if has_cjk else f"#{idx} unID"
        if is_sel:
            label += " ★" if has_cjk else " [main]"
        _draw_label(img, (x1, max(y1, 26)), label, color, has_cjk)

    # 顶部图例条
    legend = (f"{os.path.basename(photo_path)}  "
              f"detected={len(detections)} identified={identified}")
    if has_cjk:
        _draw_label(img, (10, 30), legend, (0, 0, 0), has_cjk)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        print(f"  ⚠️ 编码失败: {out_path}")
        return False
    buf.tofile(out_path)   # np.ndarray.tofile 兼容中文路径
    return True


def _collect_photos(inputs: List[str]) -> List[str]:
    """展开输入参数为图片文件列表（目录取一级图片文件）。"""
    photos: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    photos.append(os.path.join(item, name))
        elif os.path.exists(item):
            photos.append(item)
        else:
            print(f"  ⚠️ 路径不存在: {item}")
    return photos


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口 / CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spb_review",
        description="多鸟分类结果人工审阅：原图叠加 bbox/物种/置信度")
    parser.add_argument("inputs", nargs="+",
                        help="照片文件或目录（可多个）")
    parser.add_argument("--out", default=None,
                        help="输出目录（默认 <照片目录>/.superpicky/review）")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"审阅图长边上限（默认 {DEFAULT_MAX_SIDE}）")
    args = parser.parse_args(argv)

    photos = _collect_photos(args.inputs)
    if not photos:
        print("没有找到可处理的图片 / no images found")
        return 1

    done = 0
    for photo in photos:
        directory = os.path.dirname(os.path.abspath(photo))
        prefix = os.path.splitext(os.path.basename(photo))[0]
        out_dir = args.out or os.path.join(directory, ".superpicky", "review")
        out_path = os.path.join(out_dir, f"{prefix}_review.jpg")
        if annotate_photo(photo, out_path, max_side=args.max_side):
            print(f"  ✅ {out_path}")
            done += 1
    print(f"完成: {done}/{len(photos)}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
