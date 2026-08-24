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


from PySide6.QtCore import QPoint, QRect, QSize, Qt as _Qt2
from PySide6.QtWidgets import QLayout, QSizePolicy as _QSizePolicy


class _FlowLayout(QLayout):
    """
    流式布局：子控件按行排列、自动换行（鸟种筛选 chips 用）。

    Qt 没有内置流式布局，这是官方 FlowLayout 示例的精简移植。
    A minimal port of the Qt FlowLayout example for species chips.
    """

    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return _Qt2.Orientations(_Qt2.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect, test):
        x, y = rect.x(), rect.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not test:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class _ZoomCropView(QWidget):
    """
    固定尺寸的裁切预览：滚轮缩放（1-8x）+ 拖拽平移 + 双击复位。

    视图窗口大小始终不变，缩放/平移都在视图内部进行（找鸟核对用：
    看清羽毛细节不需要另开窗口）。亮度由上游应用到图像后 set_image。

    Fixed-size crop preview with wheel zoom and drag pan; the view
    itself never resizes. Brightness is baked into the input image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qimg = None
        self._zoom = 1.0
        self._off_x = 0.0   # 平移偏移（图像像素坐标）
        self._off_y = 0.0
        self._drag_pos = None
        self.setMinimumHeight(210)
        self.setSizePolicy(_QSizePolicy.Ignored,
                           _QSizePolicy.Preferred)
        self.setStyleSheet("background-color: #101014; border-radius: 6px;")

    def set_image(self, qimg):
        """设置新图像并复位缩放。/ Set a new image, reset the view."""
        self._qimg = qimg
        self._zoom = 1.0
        self._off_x = self._off_y = 0.0
        self.update()

    # ---- 视口几何 / viewport math ----

    def _base_scale(self) -> float:
        if self._qimg is None:
            return 1.0
        iw, ih = self._qimg.width(), self._qimg.height()
        if iw <= 0 or ih <= 0:
            return 1.0
        return min(self.width() / iw, self.height() / ih)

    def _clamp_offset(self):
        """把平移偏移限制在图像范围内（不拉出黑边过多）。"""
        if self._qimg is None:
            return
        iw, ih = self._qimg.width(), self._qimg.height()
        s = self._base_scale() * self._zoom
        vw, vh = self.width() / s, self.height() / s   # 视口在图像坐标的尺寸
        max_x = max(0.0, iw - vw)
        max_y = max(0.0, ih - vh)
        self._off_x = min(max(0.0, self._off_x), max_x)
        self._off_y = min(max(0.0, self._off_y), max_y)
        if iw <= vw:
            self._off_x = (iw - vw) / 2.0   # 图小于视口时居中
        if ih <= vh:
            self._off_y = (ih - vh) / 2.0

    def paintEvent(self, event):  # noqa: N802
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101014"))
        if self._qimg is None:
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "-")
            painter.end()
            return
        self._clamp_offset()
        s = self._base_scale() * self._zoom
        vw, vh = self.width() / s, self.height() / s
        painter.drawImage(QRect(0, 0, self.width(), self.height()),
                          self._qimg,
                          QRect(int(self._off_x), int(self._off_y),
                                max(1, int(vw)), max(1, int(vh))))
        if self._zoom > 1.05:
            painter.setPen(QPen(QColor(255, 220, 0), 1))
            painter.drawText(8, 16, f"{self._zoom:.1f}x")
        painter.end()

    def wheelEvent(self, event):  # noqa: N802
        """滚轮缩放：光标处的图像点保持在原位。"""
        if self._qimg is None:
            return
        old_zoom = self._zoom
        self._zoom = max(1.0, min(8.0, self._zoom *
                                  (1.25 if event.angleDelta().y() > 0 else 0.8)))
        if self._zoom == old_zoom:
            return
        # 以光标为锚点调整偏移
        mx = (event.position().x() / self.width()) *              (self.width() / (self._base_scale() * old_zoom)) + self._off_x
        my = (event.position().y() / self.height()) *              (self.height() / (self._base_scale() * old_zoom)) + self._off_y
        s = self._base_scale() * self._zoom
        self._off_x = mx - (event.position().x() / self.width()) *             (self.width() / s)
        self._off_y = my - (event.position().y() / self.height()) *             (self.height() / s)
        self._clamp_offset()
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.position()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_pos is None or self._qimg is None:
            return
        s = self._base_scale() * self._zoom
        delta = event.position() - self._drag_pos
        self._drag_pos = event.position()
        self._off_x -= delta.x() / s
        self._off_y -= delta.y() / s
        self._clamp_offset()
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self._zoom = 1.0
        self._off_x = self._off_y = 0.0
        self.update()


from PySide6.QtCore import QThread as _QThread, Signal as _Signal


class _EditorImageLoader(_QThread):
    """
    后台解码原图的加载线程（RAW 全尺寸解码需数秒，不能阻塞窗口）。

    信号:
    ready(object) — 解码完成的 BGR ndarray（失败为 None）

    Background image loader so the dialog opens instantly; a full-res
    RAW decode takes seconds and must not block the UI.
    """

    ready = _Signal(object)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        from spb_review import _read_image
        try:
            self.ready.emit(_read_image(self._path))
        except Exception:
            self.ready.emit(None)


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
        self._status_text = ""  # 无图时的提示文案（如「加载中」）
        self._highlight_keys: Optional[set] = None  # 鸟种筛选高亮（None=全部正常）
        self._focus_point = None   # (fx, fy) 归一化对焦点 / AF point
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

    def set_status_text(self, text: str) -> None:
        """设置无图占位文案（加载中/加载失败）。"""
        self._status_text = text
        self.update()

    def set_highlight_keys(self, keys) -> None:
        """设置鸟种筛选高亮集（None 清除；非匹配框半透明弱化）。"""
        self._highlight_keys = keys if keys else None
        self.update()

    def set_focus_point(self, fx: float, fy: float) -> None:
        """设置归一化对焦点 (0-1)，画布上画十字标记。"""
        self._focus_point = (fx, fy)
        self.update()

    def _point_to_display(self, x: float, y: float) -> Tuple[int, int]:
        """原图坐标点 → 画布坐标（与 bbox 同一映射）。"""
        if self._qimage is None or self._orig_w <= 0 or self._orig_h <= 0:
            return (0, 0)
        ox, oy, dw, dh = self._fit_rect()
        return (int(ox + x * dw / self._orig_w),
                int(oy + y * dh / self._orig_h))

    def _det_species_key(self, det: dict) -> str:
        """检测项的鸟种键（与对话框 _species_key 同规则）。"""
        species = det.get("species") or {}
        return (species.get("scientific") or species.get("cn")
                or species.get("en") or "")

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
                             self._status_text or "无图片数据 / no image")
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
            color = QColor(_box_color(det, self._threshold))
            # 鸟种筛选激活时，非匹配框半透明弱化
            if self._highlight_keys is not None                     and self._det_species_key(det) not in self._highlight_keys:
                color.setAlpha(70)
            pen = QPen(color)
            pen.setWidth(3 if is_sel else 1)
            painter.setPen(pen)
            if is_sel:
                selected_rect = (rect, det)
                continue
            painter.drawRect(*rect)
        # 对焦点标记（十字+圆圈，青色）
        if self._focus_point is not None:
            fx, fy = self._focus_point
            cx, cy = self._point_to_display(fx * self._orig_w,
                                            fy * self._orig_h)
            pen = QPen(QColor(0, 255, 210))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawEllipse(cx - 7, cy - 7, 14, 14)
            painter.drawLine(cx - 11, cy, cx + 11, cy)
            painter.drawLine(cx, cy - 11, cx, cy + 11)
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

    def __init__(self, photo: dict, directory: str, parent=None,
                 load_async: bool = True):
        """
        参数:
        photo (dict): 浏览器的 photo 行（report.db 字段）
        directory (str): 照片所在目录（定位 JSON/DB/文件）
        parent: Qt 父窗口
        load_async (bool): 后台线程加载原图（默认开；窗口秒开，
            RAW 解码数秒不阻塞 UI。测试传 False 走同步路径）
        """
        super().__init__(parent)
        self._load_async = load_async
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
        # 原图解码改为延迟：异步模式下后台加载（RAW 解码数秒），
        # 同步模式（测试）在 _build_ui 后立即加载
        self._img_bgr = None
        self._loader: Optional[_EditorImageLoader] = None
        self._closing = False

        # 主鸟种选择状态（species key 列表）
        self._main_keys: List[str] = []
        self._main_checkboxes: List[QCheckBox] = []
        # 物种修改（同步 DB 用）：bird_index → (cn, en, scientific)
        self._species_updates = {}
        # 选中鸟的裁切基图（亮度调节的基准）
        self._crop_base: Optional[np.ndarray] = None

        self._build_ui()
        # 默认选中等图片就绪后执行（_on_image_ready → _init_selection）；
        # 构造函数立即返回，窗口先出现、图片后台加载。

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
            "◎ <span style='color:#00ffd2'>对焦点</span>  · 点击框选中")
        self._legend_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; padding: 2px;")
        left_lay.addWidget(self._legend_label)
        # 照片信息栏：星级/对焦/锐度/美学等（找鸟核对的上下文）
        self._info_bar = QLabel("-")
        self._info_bar.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 13px;"
            f"padding: 2px;")
        left_lay.addWidget(self._info_bar)
        # 鸟种筛选 chips：点击高亮该鸟种的所有框（快速找鸟）
        self._chips_host = QWidget()
        self._chips_lay = _FlowLayout(self._chips_host, margin=2, spacing=6)
        left_lay.addWidget(self._chips_host)
        body.addWidget(left_area, 1)
        dets = (self._data or {}).get("detections") or []
        self._canvas.set_data(None, dets, threshold=35.0)
        self._canvas.set_status_text("图片加载中… / loading image…")

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
        # 逐只导航：◀ 上一只 / n/N / 下一只 ▶（筛选激活时只在匹配鸟里循环）
        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀")
        prev_btn.setObjectName("tertiary")
        prev_btn.setFixedWidth(40)
        prev_btn.clicked.connect(lambda: self._select_next(-1))
        next_btn = QPushButton("▶")
        next_btn.setObjectName("tertiary")
        next_btn.setFixedWidth(40)
        next_btn.clicked.connect(lambda: self._select_next(1))
        self._nav_counter = QLabel("-")
        self._nav_counter.setAlignment(Qt.AlignCenter)
        self._nav_counter.setStyleSheet(
            f"color: {COLORS['text_secondary']};")
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(self._nav_counter, 1)
        nav_row.addWidget(next_btn)
        box_lay.addLayout(nav_row)
        # 缩略图：滚轮缩放/拖拽平移（视图尺寸不变），亮度由上游烘焙
        self._crop_label = _ZoomCropView()
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
        self._info_label.setMinimumHeight(44)  # 固定两行高度，防切换时布局跳动
        self._info_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; padding: 2px;")
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
        # UI 就绪后启动图片加载（异步不阻塞窗口出现）
        self._start_image_load()

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

    def _start_image_load(self) -> None:
        """按构造参数启动图片加载（异步线程或同步）。"""
        if self._load_async:
            path = self._photo_file()
            if path is None:
                self._canvas.set_status_text("找不到图片文件 / photo not found")
                return
            self._loader = _EditorImageLoader(path, parent=self)
            self._loader.ready.connect(self._on_image_ready)
            self._loader.start()
        else:
            self._on_image_ready(self._load_image())

    def _on_image_ready(self, arr) -> None:
        """图片解码完成：填充画布、恢复默认选中。UI 线程执行。"""
        if self._closing:
            return
        if arr is None:
            self._canvas.set_status_text("图片加载失败 / failed to load")
            return
        self._img_bgr = arr
        dets = (self._data or {}).get("detections") or []
        _oh, _ow = arr.shape[:2]
        qimg = self._ndarray_to_qimage(arr, max_side=1800)
        self._canvas.set_data(qimg, dets, threshold=35.0,
                              orig_w=_ow, orig_h=_oh)
        self._init_selection()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        """关闭时置停接收加载结果，避免线程信号触达已销毁控件。"""
        self._closing = True
        super().closeEvent(event)

    def _init_selection(self) -> None:
        """默认选中主鸟；构建左下信息栏/chips/对焦点标记。"""
        for det in (self._data or {}).get("detections") or []:
            if det.get("is_selected") and not det.get("deleted"):
                self._canvas.set_selected(det.get("index", -1))
                break
        # 对焦点标记（JSON 有坐标时画十字）
        proc = (self._data or {}).get("processing") or {}
        if proc.get("focus_x") is not None and proc.get("focus_y") is not None:
            self._canvas.set_focus_point(float(proc["focus_x"]),
                                         float(proc["focus_y"]))
        self._refresh_info_bar()
        self._rebuild_species_chips()
        self._refresh_selection_ui()

    def _on_bird_selected(self, index: int) -> None:
        self._canvas.set_selected(index)
        self._bright_slider.setValue(0)
        self._refresh_selection_ui()

    def _refresh_selection_ui(self) -> None:
        det = self._selected_det()
        if det is None:
            self._crop_base = None
            self._crop_label.set_image(None)  # 清空视图
            self._nav_counter.setText("-")
            self._info_label.setText("-")
            self._mark_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            return
        # 导航计数：当前序号/可见总数（筛选时为匹配数）
        visible = self._visible_dets()
        try:
            pos = [d.get("index") for d in visible].index(det.get("index"))
        except ValueError:
            pos = 0
        self._nav_counter.setText(f"{pos + 1}/{len(visible)}")
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
        self._crop_label.set_image(qimg)

    def _on_brightness(self, value: int) -> None:
        self._render_crop(value)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        """窗口尺寸变化时按新面板宽度重渲染裁切缩略图。"""
        super().resizeEvent(event)
        if getattr(self, "_crop_base", None) is not None:
            self._render_crop(self._bright_slider.value())

    # ------------------------------------------------------------------
    # 找鸟导航与筛选 / find & verify navigation
    # ------------------------------------------------------------------

    def _visible_dets(self) -> List[dict]:
        """当前可见的检测项：未删除；筛选激活时仅匹配鸟种。"""
        keys = self._canvas._highlight_keys
        out = []
        for det in (self._data or {}).get("detections") or []:
            if det.get("deleted"):
                continue
            if keys is not None and \
                    self._canvas._det_species_key(det) not in keys:
                continue
            out.append(det)
        return out

    def _select_next(self, delta: int) -> None:
        """导航到上/下一只可见的鸟（循环）。/ Step to prev/next bird."""
        visible = self._visible_dets()
        if not visible:
            return
        indexes = [d.get("index") for d in visible]
        cur = self._canvas._selected
        if cur in indexes:
            pos = (indexes.index(cur) + delta) % len(visible)
        else:
            pos = 0 if delta > 0 else len(visible) - 1
        self._canvas.set_selected(indexes[pos])
        self._bright_slider.setValue(0)
        self._refresh_selection_ui()

    def _rebuild_species_chips(self) -> None:
        """重建左下鸟种筛选 chips（含数量；人工改种后即时刷新）。"""
        while self._chips_lay.count():
            item = self._chips_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        counts = {}
        for det in (self._data or {}).get("detections") or []:
            if det.get("deleted"):
                continue
            key = self._canvas._det_species_key(det)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        # 已勾选状态保留
        checked = set()
        for key in counts:
            for det in (self._data or {}).get("detections") or []:
                if (not det.get("deleted")
                        and self._canvas._det_species_key(det) == key):
                    sp = det.get("species") or {}
                    checked_key = key
                    break
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            label = key if len(key) <= 14 else key[:13] + "…"
            chip = QPushButton(f"{label} ×{n}")
            chip.setObjectName("tertiary")
            chip.setCheckable(True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(
                "QPushButton { padding: 2px 10px; font-size: 12px; }")
            chip.setProperty("species_key", key)
            chip.toggled.connect(
                lambda checked, _k=key: self._on_chip_toggled(_k, checked))
            self._chips_lay.addWidget(chip)

    def _on_chip_toggled(self, key: str, checked: bool) -> None:
        """chips 勾选变化：收集全部勾选鸟种 → 画布高亮 + 跳到首只匹配鸟。"""
        keys = {k for k, v in self._chips_state().items() if v}
        self._canvas.set_highlight_keys(keys or None)
        if keys:
            visible = self._visible_dets()
            if visible:
                self._canvas.set_selected(visible[0].get("index", -1))
                self._bright_slider.setValue(0)
                self._refresh_selection_ui()

    def _chips_state(self) -> dict:
        """当前各 chip 的勾选状态（键 → bool）。"""
        state = {}
        for i in range(self._chips_lay.count()):
            w = self._chips_lay.itemAt(i).widget()
            if w is not None and w.property("species_key"):
                state[w.property("species_key")] = w.isChecked()
        return state

    def _refresh_info_bar(self) -> None:
        """左下照片信息栏：星级/对焦/锐度/美学/飞版/主鸟。"""
        proc = (self._data or {}).get("processing") or {}
        q = proc.get("quality") or {}
        rating = proc.get("rating")
        stars = {3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐", 0: "0★"}.get(rating, "❌")
        parts = [f"<b>{stars}</b>"]
        focus = proc.get("focus_status")
        if focus:
            parts.append(f"对焦 {focus}")
        sharp = q.get("head_sharpness")
        if sharp is not None:
            parts.append(f"锐度 {sharp:.0f}")
        topiq = q.get("topiq")
        if topiq is not None:
            parts.append(f"美学 {topiq:.1f}")
        if proc.get("is_flying"):
            parts.append("飞版")
        exposure = proc.get("exposure_status")
        if exposure:
            parts.append(str(exposure))
        sm = proc.get("species_main") or {}
        main_name = sm.get("cn") or sm.get("en")
        if main_name:
            conf = sm.get("confidence")
            conf_txt = f" {conf:.0f}%" if conf is not None else ""
            parts.append(f"主鸟 {main_name}{conf_txt}")
        self._info_bar.setText("  |  ".join(parts))

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
        self._rebuild_species_chips()
        self._refresh_info_bar()

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
        self._rebuild_species_chips()

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
