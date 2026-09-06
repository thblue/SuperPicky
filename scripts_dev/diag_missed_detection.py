#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
漏检诊断：对单张 RAW 复现「跑批主检 / 补救重扫 / 单张识别」三条 YOLO 推理链路，
输出每条的原始检测分数，定位漏检环节。只读诊断，不写任何库。

用法:
    python scripts_dev/diag_missed_detection.py <照片.RAW> [照片2.RAW ...]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import config, get_best_device  # noqa: E402
from core.ai_model import preprocess_image, load_yolo_model  # noqa: E402
from tools.find_bird_util import raw_to_jpeg  # noqa: E402


def yolo_boxes(model, img, imgsz, conf):
    """
    跑一次 YOLO，返回 (类id, 置信度, 框) 列表（不过滤类别）。

    参数:
        model: YOLO 实例
        img: BGR ndarray
        imgsz (int): 推理分辨率
        conf (float): 置信度地板

    返回:
        list[tuple[int, float, tuple]]: (cls, conf, (x1, y1, x2, y2))
    """
    res = model(img, imgsz=imgsz, conf=conf,
                device=get_best_device().type, verbose=False)[0]
    out = []
    if res.boxes is not None:
        for b in res.boxes:
            out.append((int(b.cls[0]), float(b.conf[0]),
                        tuple(int(v) for v in b.xyxy[0])))
    return out


def main() -> int:
    """
    对每张输入照片依次复现三条链路并打印对比。

    返回:
        int: 0 成功
    """
    raw_path = sys.argv[1]
    model = load_yolo_model()

    # 链路0：跑批预览（extract_thumb 内嵌 JPEG，与单张 load_image 同源）
    preview = raw_to_jpeg(raw_path)
    from PIL import Image
    full = np.array(Image.open(preview).convert("RGB"))
    full_bgr = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
    fh, fw = full_bgr.shape[:2]
    print(f"内嵌预览分辨率: {fw}x{fh}")

    # 链路1：跑批主检 —— 预览 JPG preprocess 到 1024 长边，默认 imgsz(640)
    # （跑批的 detect_and_draw_birds 即以转换后的预览 JPG 为输入）
    img1024 = preprocess_image(preview)
    h, w = img1024.shape[:2]
    print(f"跑批预处理图: {w}x{h}")
    boxes = yolo_boxes(model, img1024, 640, 0.01)
    birds = [(c, f, bb) for c, f, bb in boxes if c == config.ai.BIRD_CLASS_ID]
    print(f"[主检 imgsz=640 conf>=0.01] 检出 {len(boxes)} 框, 鸟类 {len(birds)}: "
          + ", ".join(f"{f:.3f}@{bb}" for c, f, bb in sorted(birds, key=lambda x: -x[1])[:5]))

    # 链路2：补救重扫 —— 同一张 1024 图，imgsz=1024 conf=0.05
    boxes = yolo_boxes(model, img1024, config.ai.RESCUE_IMGSZ, 0.01)
    birds = [(c, f, bb) for c, f, bb in boxes if c == config.ai.BIRD_CLASS_ID]
    print(f"[补救 imgsz=1024 conf>=0.01] 检出 {len(boxes)} 框, 鸟类 {len(birds)}: "
          + ", ".join(f"{f:.3f}@{bb}" for c, f, bb in sorted(birds, key=lambda x: -x[1])[:5]))

    # 链路3：单张识别 —— 全分辨率内嵌 JPEG，imgsz=1024
    boxes = yolo_boxes(model, full_bgr, 1024, 0.01)
    birds = [(c, f, bb) for c, f, bb in boxes if c == config.ai.BIRD_CLASS_ID]
    print(f"[单张 full {fw}px→imgsz1024] 检出 {len(boxes)} 框, 鸟类 {len(birds)}: "
          + ", ".join(f"{f:.3f}@{bb}" for c, f, bb in sorted(birds, key=lambda x: -x[1])[:5]))
    if birds:
        c, f, (x1, y1, x2, y2) = max(birds, key=lambda x: x[1])
        print(f"最佳鸟框: {(x2-x1)}x{(y2-y1)}px = 全图面积 "
              f"{(x2-x1)*(y2-y1)/(fw*fh)*100:.2f}% / 线性 {(x2-x1)/fw*100:.1f}% 宽度")
    return 0


if __name__ == "__main__":
    sys.exit(main())
