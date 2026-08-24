#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多鸟编辑对话框（multi-bird editor dialog）

在结果浏览器中人工审阅/修正一张照片的多鸟识别结果。所有编辑只写
图片同目录的 sidecar JSON（.superpicky/meta/<前缀>.json），不碰照片
EXIF/XMP；物种修改会同步 report.db 的 bird_detections 行（应用内视图
一致）。

功能：
- 左侧原图叠加全部 bbox（红=主鸟/绿=已采纳/橙=低置信/灰=未分类，
  与 spb_review 同一配色）；点击框选中
- 右侧显示选中 bbox 的放大裁切 + 亮度调节（调亮/调暗，纯查看不落盘）；
  可重新标记鸟种（复用 IOC 搜索对话框，标记后该框记为人工）或删除框
  （软删除，JSON 中 deleted=true，数据保留）
- 主鸟种选择：默认电脑选中的主鸟；可从已识别鸟种里复选（最多 3 个），
  写入 JSON 顶层 main_species 数组；人工改过分类的框会即时反映到候选

Multi-bird manual review/edit dialog. Edits go to the per-photo sidecar
JSON (never EXIF); species changes are mirrored into bird_detections.
"""

from __future__ import annotations

import copy
import datetime
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from ui.styles import COLORS

# 主鸟种最多可选数 / Max selectable main species.
MAX_MAIN_SPECIES = 3

# 亮度滑块范围（±100，内部放大 2.5 倍作用于像素值）
# Brightness slider range (scaled 2.5x onto pixel values).
_BRIGHTNESS_RANGE = 100

# 框配色（QColor: 红主鸟/绿采纳/橙低置信/灰未分类）
# Box colors matching spb_review.
def _box_color(det: dict, threshold: float) -> QColor:
    """按检测状态返回框色 / Return the box color for a detection."""
    species = det.get("species")
    conf = (species or {}).get("confidence")
    if det.get("is_selected"):
        return QColor(255, 0, 0)
    if species and conf is not None and conf >= threshold:
        return QColor(0, 180, 0)
    if species:
        return QColor(255, 165, 0)
    return QColor(160, 160, 160)


def _now_iso() -> str:
    """编辑日志时间戳 / Edit-log timestamp."""
    return datetime.datetime.now().isoformat(timespec="seconds")


class _BirdCanvas(QWidget):
    """
    左侧画布：绘制缩放后的原图 + 全部 bbox，点击选中一只鸟。

    信号:
    bird_selected(int) — 选中的 detection index；点击空白发射 -1

    Canvas painting the photo with bbox overlays; click to select.
    """

    bird_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimage: Optional[QImage] = None
        self._detections: List[dict] = []
        self._selected = -1
        self._threshold = 35.0
        self._orig_w = 0   # bbox 坐标系（原图）尺寸 / bbox coord space
        self._orig_h = 0
        self.setMinimumSize(520, 400)
        self.setMouseTracking(False)

    def set_data(self, qimage: QImage, detections: List[dict],
                 threshold: float,
                 orig_w: int = 0, orig_h: int = 0) -> None:
        """
        设置底图与检测列表（刷新绘制）。

        orig_w/orig_h 是 bbox 所在的原图坐标系尺寸——底图通常为省内存
        而降采样过（如 6960→1800），不传原图尺寸会把框画到错误位置。

        Set image + detections. orig_w/orig_h give the coordinate space
        of the bboxes (the canvas image is usually downscaled).
        """
        self._qimage = qimage
        self._detections = detections
        self._threshold = threshold
        # bbox 坐标系 = 原图；缺省时视为与底图同尺寸
        self._orig_w = orig_w or (qimage.width() if qimage else 0)
        self._orig_h = orig_h or (qimage.height() if qimage else 0)
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = index
        self.update()

    # ---- 几何换算 / geometry mapping ----

    def _fit_rect(self) -> Tuple[int, int, int, int]:
        """底图等比居中的显示区域 (x, y, w, h)。/ Fitted image rect."""
        if self._qimage is None:
            return (0, 0, 0, 0)
        iw, ih = self._qimage.width(), self._qimage.height()
        if iw <= 0 or ih <= 0:
            return (0, 0, 0, 0)
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        return ((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    def _bbox_to_display(self, bbox: List[float]) -> Tuple[int, int, int, int]:
        """原图坐标 bbox → 画布坐标矩形。/ Map an original-coord bbox."""
        if self._qimage is None or self._orig_w <= 0 or self._orig_h <= 0:
            return (0, 0, 0, 0)
        ox, oy, dw, dh = self._fit_rect()
        sx = dw / self._orig_w
        sy = dh / self._orig_h
        x, y, w, h = bbox
        return (int(ox + x * sx), int(oy + y * sy),
                max(2, int(w * sx)), max(2, int(h * sy)))

    def _point_to_orig(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        """画布坐标 → 原图坐标（底图区域内）。/ Canvas → original coords."""
        if self._qimage is None or self._orig_w <= 0 or self._orig_h <= 0:
            return None
        ox, oy, dw, dh = self._fit_rect()
        if not (ox <= pos.x() < ox + dw and oy <= pos.y() < oy + dh):
            return None
        sx = self._orig_w / dw
        sy = self._orig_h / dh
        return ((pos.x() - ox) * sx, (pos.y() - oy) * sy)

    # ---- 绘制与交互 / paint & interact ----

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLORS["bg_void"]))
        if self._qimage is None:
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "无图片数据 / no image")
            return
        ox, oy, dw, dh = self._fit_rect()
        # 关键：底图必须缩放进适配区域（QRect 目标重载），
        # 与 bbox 的映射使用同一缩放比——否则框和鸟必然错位。
        # The image MUST be drawn scaled into the fit rect (QRect
        # overload) so it shares the exact mapping used for bboxes.
        from PySide6.QtCore import QRect
        painter.drawImage(QRect(int(ox), int(oy), int(dw), int(dh)),
                          self._qimage)
        # 先画未选中的细框，选中框最后画（覆盖在上）
        # Draw unselected first so the selected box stays on top.
        selected_rect = None
        for det in self._detections:
            if det.get("deleted"):
                continue
            bbox = det.get("bbox")
            if not bbox or any(v is None for v in bbox):
                continue
            rect = self._bbox_to_display(bbox)
            is_sel = det.get("index") == self._selected
            color = _box_color(det, self._threshold)
            pen = QPen(color)
            pen.setWidth(3 if is_sel else 1)
            painter.setPen(pen)
            if is_sel:
                selected_rect = (rect, det)
                continue
            painter.drawRect(*rect)
        if selected_rect is not None:
            rect, det = selected_rect
            pen = QPen(QColor(255, 220, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(*rect)
            # 选中序号角标 / index badge
            label = f"#{det.get('index')}"
            painter.fillRect(rect[0], max(0, rect[1] - 16),
                             10 + 8 * len(label), 16,
                             QColor(255, 220, 0))
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(rect[0] + 4, max(11, rect[1] - 4), label)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = self._point_to_orig(event.position().toPoint())
        if pos is None:
            self.bird_selected.emit(-1)
            return
        # 命中测试：取包含该点、面积最小的框（密集鸟群里更精确）
        # Hit-test: smallest containing bbox wins in dense flocks.
        best, best_area = -1, float("inf")
        for det in self._detections:
            if det.get("deleted"):
                continue
            bbox = det.get("bbox")
            if not bbox or any(v is None for v in bbox):
                continue
            x, y, w, h = bbox
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h:
                area = w * h
                if area < best_area:
                    best_area = area
                    best = det.get("index", -1)
        self.bird_selected.emit(best)


class MultibirdEditorDialog(QDialog):
    """
    多鸟识别人工编辑对话框。

    构造后 exec()；Accepted 时编辑已写入 sidecar JSON（并同步
    bird_detections 的物种）。编辑内容：
    - 重新标记某框鸟种（标记后 edited=true，视为人工结果）
    - 软删除误检框（deleted=true，画布不再显示）
    - 主鸟种复选（≤3，写入顶层 main_species）

    Manual multi-bird editor dialog; edits persist to the sidecar JSON.
    """

    def __init__(self, photo: dict, directory: str, parent=None):
        """
        参数:
        photo (dict): 浏览器的 photo 行（report.db 字段）
        directory (str): 照片所在目录（定位 JSON/DB/文件）
        parent: Qt 父窗口
        """
        super().__init__(parent)
        self._photo = dict(photo)
        self._directory = directory
        self._prefix = photo.get("filename") or ""
        self.setWindowTitle(
            f"多鸟编辑 - {self._prefix}  |  Multi-bird Editor")
        self.resize(1180, 760)
        self.setWindowState(Qt.WindowMaximized)  # 默认最大化，鸟群照更需要空间

        # ---- 数据加载 / load data ----
        from core.sidecar_export import _sidecar_path, _atomic_write_json  # noqa: F401 (写回用)
        self._json_path = _sidecar_path(directory, self._prefix)
        self._data = None
        if os.path.exists(self._json_path):
            import json
            try:
                with open(self._json_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (OSError, ValueError):
                self._data = None
        self._img_bgr = self._load_image()

        # 主鸟种选择状态（species key 列表）
        self._main_keys: List[str] = []
        self._main_checkboxes: List[QCheckBox] = []
        # 物种修改（同步 DB 用）：bird_index → (cn, en, scientific)
        self._species_updates = {}
        # 选中鸟的裁切基图（亮度调节的基准）
        self._crop_base: Optional[np.ndarray] = None

        self._build_ui()
        self._init_selection()

    # ------------------------------------------------------------------
    # 数据加载 / data loading
    # ------------------------------------------------------------------

    def _photo_file(self) -> Optional[str]:
        """解析照片实际文件路径（current/original → 前缀扫描）。"""
        for key in ("current_path", "original_path"):
            rel = self._photo.get(key)
            if not rel:
                continue
            p = rel if os.path.isabs(rel) else os.path.join(self._directory, rel)
            if os.path.exists(p):
                return p
        # 兜底：按前缀在目录里找（jpg 优先，可显示性最好）
        for ext in (".jpg", ".jpeg", ".cr3", ".nef", ".arw", ".dng",
                    ".cr2", ".png"):
            p = os.path.join(self._directory, self._prefix + ext)
            if os.path.exists(p):
                return p
        return None

    def _load_image(self) -> Optional[np.ndarray]:
        """读原图（兼容 RAW/中文路径），返回 BGR ndarray。"""
        path = self._photo_file()
        if not path:
            return None
        from spb_review import _read_image
        return _read_image(path)

    # ------------------------------------------------------------------
    # UI 构建 / UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QScrollArea, QGroupBox
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        body = QHBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        root.addLayout(body, 1)

        # 左：画布 + 底部图例条（图例不画在照片上，避免遮挡）
        left_area = QWidget()
        left_lay = QVBoxLayout(left_area)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)
        self._canvas = _BirdCanvas()
        self._canvas.bird_selected.connect(self._on_bird_selected)
        left_lay.addWidget(self._canvas, 1)
        self._legend_label = QLabel(
            "■ <span style='color:#ff3b30'>主鸟</span>  "
            "■ <span style='color:#28a745'>已采纳</span>  "
            "■ <span style='color:#ffa500'>低置信</span>  "
            "■ <span style='color:#a0a0a0'>未分类</span>  "
            "· 点击框选中")
        self._legend_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; padding: 2px;")
        left_lay.addWidget(self._legend_label)
        body.addWidget(left_area, 1)
        if self._img_bgr is not None:
            qimg = self._ndarray_to_qimage(self._img_bgr, max_side=1800)
            dets = (self._data or {}).get("detections") or []
            _oh, _ow = self._img_bgr.shape[:2]
            self._canvas.set_data(qimg, dets, threshold=35.0,
                                  orig_w=_ow, orig_h=_oh)

        # 右：属性面板 = 固定容器（滚动区 + 底部固定保存条），
        # 保存条不进滚动区，永远可见可点
        from PySide6.QtWidgets import QScrollArea, QGroupBox
        right_container = QWidget()
        right_container.setMinimumWidth(360)
        right_container.setMaximumWidth(420)
        right_outer = QVBoxLayout(right_container)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(8)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        # 内容一律纵向排布，禁止横向滚动（此前 600px 裁切图撑出横条）
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right = QWidget()
        right_scroll.setWidget(right)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(10, 10, 10, 10)
        right_lay.setSpacing(14)
        right_outer.addWidget(right_scroll, 1)
        body.addWidget(right_container)

        if self._data is None:
            tip = QLabel("未找到 sidecar JSON（照片未处理或多鸟功能未启用）\n"
                         "No sidecar JSON found.")
            tip.setWordWrap(True)
            right_lay.addWidget(tip)
            right_lay.addStretch(1)
            close_bar = QHBoxLayout()
            close = QPushButton("关闭")
            close.setObjectName("secondary")
            close.clicked.connect(self.reject)
            close_bar.addStretch(1)
            close_bar.addWidget(close)
            right_outer.addLayout(close_bar)
            return

        # -- 使用提示（首行，帮助理解交互）--
        usage = QLabel("① 点击左侧图中的框选中一只鸟\n"
                       "② 右侧放大查看，可调亮度、改正鸟种或删框\n"
                       "③ 底部勾选主鸟种后保存")
        usage.setWordWrap(True)
        usage.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px;"
            f"padding: 4px;")
        right_lay.addWidget(usage)

        # -- 选中鸟区域 --
        box = QGroupBox("选中鸟")
        box_lay = QVBoxLayout(box)
        box_lay.setSpacing(8)
        self._crop_label = QLabel("点击左侧图中的框选择一只鸟\n"
                                  "Click a box on the photo")
        self._crop_label.setAlignment(Qt.AlignCenter)
        self._crop_label.setMinimumHeight(210)
        from PySide6.QtWidgets import QSizePolicy
        self._crop_label.setSizePolicy(
            QSizePolicy.Ignored, self._crop_label.sizePolicy().verticalPolicy())
        self._crop_label.setStyleSheet(
            f"background-color: {COLORS['bg_void']}; border-radius: 6px;")
        box_lay.addWidget(self._crop_label)

        bright_row = QHBoxLayout()
        bright_row.addWidget(QLabel("亮度"))
        self._bright_slider = QSlider(Qt.Horizontal)
        self._bright_slider.setRange(-_BRIGHTNESS_RANGE, _BRIGHTNESS_RANGE)
        self._bright_slider.setValue(0)
        self._bright_slider.valueChanged.connect(self._on_brightness)
        bright_row.addWidget(self._bright_slider, 1)
        reset_btn = QPushButton("重置")
        reset_btn.setObjectName("tertiary")
        reset_btn.setFixedWidth(48)
        reset_btn.clicked.connect(
            lambda: self._bright_slider.setValue(0))
        bright_row.addWidget(reset_btn)
        box_lay.addLayout(bright_row)

        self._info_label = QLabel("-")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            f"color: {COLORS['text_secondary']};")
        box_lay.addWidget(self._info_label)

        btn_row = QHBoxLayout()
        self._mark_btn = QPushButton("改鸟种")
        self._mark_btn.setObjectName("secondary")
        self._mark_btn.clicked.connect(self._on_remark)
        btn_row.addWidget(self._mark_btn)
        self._delete_btn = QPushButton("删框")
        self._delete_btn.setObjectName("tertiary")
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._delete_btn)
        box_lay.addLayout(btn_row)
        right_lay.addWidget(box)

        # -- 主鸟种区域（内层限高滚动，长列表不再撑爆面板）--
        main_box = QGroupBox(f"主鸟种（最多 {MAX_MAIN_SPECIES} 个）")
        main_outer = QVBoxLayout(main_box)
        self._main_scroll = QScrollArea()
        self._main_scroll.setWidgetResizable(True)
        self._main_scroll.setMaximumHeight(220)
        self._main_scroll.setFrameShape(QScrollArea.NoFrame)
        self._main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_host = QWidget()
        self._main_lay = QVBoxLayout(main_host)
        self._main_lay.setContentsMargins(0, 0, 0, 0)
        self._main_scroll.setWidget(main_host)
        main_outer.addWidget(self._main_scroll)
        self._main_keys = self._default_main_keys()
        self._rebuild_main_checkboxes()
        right_lay.addWidget(main_box)

        right_lay.addStretch(1)

        # -- 底部保存条（固定在滚动区之外，始终可见）--
        save_bar = QHBoxLayout()
        hint = QLabel("保存 → 同目录 JSON（不动照片）")
        hint.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        hint.setWordWrap(True)
        save_bar.addWidget(hint, 1)
        cancel = QPushButton("取消")
        cancel.setObjectName("tertiary")
        cancel.clicked.connect(self.reject)
        save_bar.addWidget(cancel)
        save = QPushButton("保存")
        save.setObjectName("secondary")
        save.clicked.connect(self._on_save)
        save_bar.addWidget(save)
        right_outer.addLayout(save_bar)

    def _ndarray_to_qimage(self, arr: np.ndarray,
                           max_side: int = 1800) -> QImage:
        """BGR ndarray → QImage（超长边先降采样，RGB888 通道序）。"""
        img = arr
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h2, w2 = rgb.shape[:2]
        return QImage(rgb.data, w2, h2, 3 * w2, QImage.Format_RGB888).copy()

    # ------------------------------------------------------------------
    # 选中与预览 / selection & preview
    # ------------------------------------------------------------------

    def _selected_det(self) -> Optional[dict]:
        idx = self._canvas._selected
        for det in (self._data or {}).get("detections") or []:
            if det.get("index") == idx and not det.get("deleted"):
                return det
        return None

    def _init_selection(self) -> None:
        """默认选中主鸟。/ Initially select the main bird."""
        for det in (self._data or {}).get("detections") or []:
            if det.get("is_selected") and not det.get("deleted"):
                self._canvas.set_selected(det.get("index", -1))
                break
        self._refresh_selection_ui()

    def _on_bird_selected(self, index: int) -> None:
        self._canvas.set_selected(index)
        self._bright_slider.setValue(0)
        self._refresh_selection_ui()

    def _refresh_selection_ui(self) -> None:
        det = self._selected_det()
        if det is None:
            self._crop_base = None
            self._crop_label.setPixmap(QPixmap())  # 清空
            self._crop_label.setText("点击左侧图中的框选择一只鸟\n"
                                     "Click a box on the photo")
            self._info_label.setText("-")
            self._mark_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            return
        self._mark_btn.setEnabled(True)
        self._delete_btn.setEnabled(True)
        self._show_crop(det)
        # 信息行：物种/置信度/来源(人工|AI)/面积
        species = det.get("species")
        if species:
            name = species.get("cn") or species.get("en") or "?"
            conf = species.get("confidence")
            source = "人工 manual" if det.get("edited") else "AI"
            conf_txt = f"{conf:.0f}%" if conf is not None else "-"
            info = (f"#{det.get('index')} {name} · {conf_txt} · {source}")
        else:
            info = f"#{det.get('index')} 未分类 unclassified"
        area = det.get("area_ratio")
        if area is not None:
            info += f"\n面积 area: {area * 100:.2f}%"
        yolo_conf = det.get("yolo_conf")
        if yolo_conf is not None:
            info += f" · YOLO {yolo_conf:.2f}"
        self._info_label.setText(info)

    def _show_crop(self, det: dict) -> None:
        """显示选中 bbox 的放大裁切（含 10% padding）。"""
        if self._img_bgr is None:
            return
        bbox = det.get("bbox")
        if not bbox or any(v is None for v in bbox):
            return
        x, y, w, h = [int(v) for v in bbox]
        ih, iw = self._img_bgr.shape[:2]
        pad = int(max(w, h) * 0.10) + 2
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(iw, x + w + pad), min(ih, y + h + pad)
        crop = self._img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        # 放大到预览区（长边 ~600），小 bbox 即为放大镜效果
        scale = 600 / max(crop.shape[:2])
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        self._crop_base = crop
        self._render_crop(self._bright_slider.value())

    def _render_crop(self, brightness: int) -> None:
        """按亮度偏移渲染裁切预览（纯查看，不落盘）。"""
        crop = self._crop_base
        if crop is None:
            return
        if brightness != 0:
            crop = np.clip(crop.astype(np.int16)
                           + int(brightness * 2.5), 0, 255).astype(np.uint8)
        qimg = self._ndarray_to_qimage(crop, max_side=600)
        from PySide6.QtGui import QPixmap
        pm = QPixmap.fromImage(qimg).scaled(
            self._crop_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self._crop_label.setPixmap(pm)

    def _on_brightness(self, value: int) -> None:
        self._render_crop(value)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        """窗口尺寸变化时按新面板宽度重渲染裁切缩略图。"""
        super().resizeEvent(event)
        if getattr(self, "_crop_base", None) is not None:
            self._render_crop(self._bright_slider.value())

    # ------------------------------------------------------------------
    # 编辑动作 / edit actions
    # ------------------------------------------------------------------

    def _on_remark(self) -> None:
        """重新标记选中鸟的物种（IOC 搜索；标记后视为人工结果）。"""
        det = self._selected_det()
        if det is None:
            return
        from ui.bird_species_edit_dialog import BirdSpeciesEditDialog
        dialog = BirdSpeciesEditDialog(parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        cn = dialog.selected_cn or ""
        en = dialog.selected_en or ""
        latin = dialog.selected_latin or ""
        if not (cn or en or latin):
            return
        old = copy.deepcopy(det.get("species"))
        det["species"] = {
            "cn": cn, "en": en, "scientific": latin,
            "confidence": None, "class_id": None,
            "gbif_rarity_100": det.get("species", {}).get("gbif_rarity_100"),
        }
        det["edited"] = True
        self._species_updates[det.get("index")] = (cn, en, latin)
        self._append_edit("species_set", det.get("index"),
                          old=old, new=det["species"])
        self._refresh_selection_ui()
        self._canvas.update()
        self._rebuild_main_checkboxes()

    def _on_delete(self) -> None:
        """软删除选中框（JSON deleted=true，数据保留可手工恢复）。"""
        det = self._selected_det()
        if det is None:
            return
        det["deleted"] = True
        self._append_edit("bbox_deleted", det.get("index"),
                          old=None, new=None)
        self._canvas.set_selected(-1)
        self._refresh_selection_ui()
        self._canvas.update()
        self._rebuild_main_checkboxes()

    def _append_edit(self, action: str, bird_index: Optional[int],
                     old=None, new=None) -> None:
        """追加一条人工编辑日志（保存时随 JSON 落盘）。"""
        self._data.setdefault("edits", []).append({
            "timestamp": _now_iso(),
            "actor": "human",
            "action": action,
            "bird_index": bird_index,
            "old": old,
            "new": new,
        })

    # ------------------------------------------------------------------
    # 主鸟种复选 / main-species checkboxes
    # ------------------------------------------------------------------

    @staticmethod
    def _species_key(species: dict) -> str:
        """鸟种唯一键（学名优先，回退中文名）。/ Unique species key."""
        return (species.get("scientific")
                or species.get("cn")
                or species.get("en") or "?")

    def _main_candidates(self) -> List[dict]:
        """主鸟种候选：未删除、有物种的鸟（按首次出现去重）。"""
        seen = {}
        for det in (self._data or {}).get("detections") or []:
            if det.get("deleted"):
                continue
            species = det.get("species")
            if not species:
                continue
            key = self._species_key(species)
            if key not in seen:
                seen[key] = {
                    "key": key,
                    "cn": species.get("cn") or species.get("en") or key,
                    "en": species.get("en") or "",
                    "scientific": species.get("scientific") or "",
                    "bird_index": det.get("index"),
                }
        return list(seen.values())

    def _default_main_keys(self) -> List[str]:
        """默认主鸟种：JSON 已有人工选择 → 沿用；否则电脑主鸟。"""
        existing = (self._data or {}).get("main_species") or []
        if existing:
            return [e.get("scientific") or e.get("cn") or ""
                    for e in existing if e]
        for det in (self._data or {}).get("detections") or []:
            if det.get("is_selected") and not det.get("deleted") \
                    and det.get("species"):
                return [self._species_key(det["species"])]
        # 回退 photos 表的主鸟种（单鸟照）
        cn = self._photo.get("bird_species_cn")
        en = self._photo.get("bird_species_en")
        if cn or en:
            return [f"{cn or en}"]
        return []

    def _rebuild_main_checkboxes(self) -> None:
        """重建主鸟种复选框（人工改分类后即时刷新）。"""
        # 清旧
        while self._main_lay.count():
            item = self._main_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._main_checkboxes = []
        for cand in self._main_candidates():
            cb = QCheckBox(f"{cand['cn']}  #{cand['bird_index']}")
            cb.setChecked(cand["key"] in self._main_keys)
            cb.stateChanged.connect(self._on_main_toggled)
            self._main_checkboxes.append(cb)
            self._main_lay.addWidget(cb)
        if not self._main_checkboxes:
            tip = QLabel("（暂无已识别鸟种）")
            tip.setStyleSheet(f"color: {COLORS['text_muted']};")
            self._main_lay.addWidget(tip)

    def _on_main_toggled(self) -> None:
        """复选状态变化：收集勾选并强制 ≤ MAX_MAIN_SPECIES。"""
        sender = self.sender()
        checked = [cb for cb in self._main_checkboxes if cb.isChecked()]
        if len(checked) > MAX_MAIN_SPECIES and sender is not None:
            # 超限：撤销本次勾选的框并复位
            sender.blockSignals(True)
            sender.setChecked(False)
            sender.blockSignals(False)
            return
        cands = self._main_candidates()
        self._main_keys = [
            cand["key"] for cand, cb in zip(cands, self._main_checkboxes)
            if cb.isChecked()]

    # ------------------------------------------------------------------
    # 保存 / save
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """写入 sidecar JSON + 同步 bird_detections 物种，关闭对话框。"""
        from core.sidecar_export import _atomic_write_json
        # 主鸟种选择写入顶层（候选映射回完整字段）
        cands = {c["key"]: c for c in self._main_candidates()}
        main_species = []
        for key in self._main_keys:
            c = cands.get(key)
            if c:
                main_species.append({
                    "cn": c["cn"], "en": c["en"],
                    "scientific": c["scientific"],
                    "bird_index": c["bird_index"],
                })
        old_main = (self._data or {}).get("main_species")
        if main_species and main_species != old_main:
            self._data["main_species"] = main_species
            self._append_edit("main_species_set", None,
                              old=old_main, new=main_species)
        try:
            _atomic_write_json(self._json_path, self._data)
        except OSError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "保存失败 Save failed", str(e))
            return
        # 同步 report.db 的物种（应用内视图一致；删除软标记不入库）
        if self._species_updates:
            try:
                from tools.report_db import ReportDB
                db = ReportDB(self._directory)
                for idx, (cn, en, sci) in self._species_updates.items():
                    db.update_detection_species(
                        self._prefix, idx, cn, en, sci or None)
                db._conn.close()
            except Exception:
                pass
        self.accept()
