# -*- coding: utf-8 -*-
"""
spb_browse 启动器对话框 / Launcher dialog for spb_browse

双击 SPBBrowse.exe（或 `python spb_browse.py` 不带参数）时弹出：
最近目录列表 + 「浏览」按钮，用户选定目录后进入结果浏览器。
最近目录与主程序共用同一份 advanced_config.json
（get_advanced_config() 单例），主程序里处理过的目录会直接出现在这里。

Launcher dialog for spb_browse. Shown when SPBBrowse.exe is launched
without arguments: a recent-directories list plus a Browse button.
The user picks a photo folder and the results browser opens on it.
Recent folders are shared with the main app via advanced_config.json.

使用方法 / Usage:
    dialog = BrowseLauncherDialog(parent=None)
    if dialog.exec() == QDialog.Accepted and dialog.selected_directory:
        window.open_directory(dialog.selected_directory)
"""

import os

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, QStandardPaths

from advanced_config import get_advanced_config
from tools.i18n import get_i18n
from ui.styles import COLORS


class BrowseLauncherDialog(QDialog):
    """
    spb_browse 无参数启动时的「打开照片目录」启动器。

    界面构成：
      - 最近目录列表（不可访问的目录加「(脱机)」前缀并禁用）
      - 「浏览」按钮（QFileDialog，起始目录取最近可用目录，回退系统图片目录）
      - 「打开 / 取消 / 清除历史」

    选定目录通过 selected_directory 属性返回；用户取消时返回空字符串。

    Launcher shown when spb_browse starts without arguments.
    Consists of a recent-directories list (offline entries disabled with an
    "(Offline)" prefix), a Browse button (starts at the most recent available
    folder, falling back to the system Pictures location), and
    Open / Cancel / Clear-history buttons. The chosen folder is exposed via
    the selected_directory attribute; an empty string means the user cancelled.
    """

    _WIDTH = 560          # 固定宽度，路径较长时仍可读 / fixed width for long paths
    _LIST_MAX_HEIGHT = 260  # 列表最大高度（最近目录最多 10 条）/ list max height

    def __init__(self, parent=None):
        super().__init__(parent)
        self.i18n = get_i18n()
        self.config = get_advanced_config()

        #: 用户最终选定的目录；空字符串表示取消 / chosen dir, "" on cancel
        self.selected_directory: str = ""

        self.setWindowTitle(self.i18n.t("browser.launcher_title"))
        self.setFixedWidth(self._WIDTH)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        self.setStyleSheet(
            f"QDialog {{ background-color: {COLORS['bg_elevated']}; }}"
        )

        self._build_ui()
        self._refresh_recent_list()

    # ------------------------------------------------------------------
    #  UI 构建 / UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """构建对话框 UI：标题提示 + 最近目录列表 + 按钮行。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        hint = QLabel(self.i18n.t("browser.launcher_recent_hint"))
        hint.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; "
            f"background: transparent;"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._group_label = QLabel(self.i18n.t("menu.recent_dirs"))
        self._group_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 13px; "
            f"font-weight: bold; background: transparent;"
        )
        root.addWidget(self._group_label)

        self._recent_list = QListWidget()
        self._recent_list.setMaximumHeight(self._LIST_MAX_HEIGHT)
        self._recent_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 4px;
                font-size: 12px;
            }}
            QListWidget::item {{
                padding: 6px 8px;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['accent_dim']};
                color: {COLORS['text_primary']};
            }}
            QListWidget::item:disabled {{
                color: {COLORS['text_muted']};
            }}
            """
        )
        self._recent_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._recent_list.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._recent_list)

        # 空列表提示（有历史时隐藏）/ placeholder shown when history is empty
        self._empty_label = QLabel(self.i18n.t("browser.launcher_no_recent"))
        self._empty_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; "
            f"background: transparent;"
        )
        self._empty_label.setWordWrap(True)
        root.addWidget(self._empty_label)

        # 按钮行 / button row
        buttons = QHBoxLayout()
        buttons.setSpacing(8)

        self._clear_button = QPushButton(self.i18n.t("menu.recent_dirs_clear"))
        self._clear_button.clicked.connect(self._on_clear_history)
        buttons.addWidget(self._clear_button)

        buttons.addStretch(1)

        self._open_button = QPushButton(self.i18n.t("browser.launcher_open"))
        self._open_button.setEnabled(False)
        self._open_button.clicked.connect(self._on_open_clicked)
        buttons.addWidget(self._open_button)

        cancel_button = QPushButton(self.i18n.t("buttons.cancel"))
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button)

        browse_button = QPushButton(self.i18n.t("buttons.select_dir"))
        browse_button.setDefault(True)
        browse_button.clicked.connect(self._on_browse_clicked)
        buttons.addWidget(browse_button)

        root.addLayout(buttons)

    # ------------------------------------------------------------------
    #  最近目录列表 / recent list
    # ------------------------------------------------------------------

    def _refresh_recent_list(self):
        """根据配置重建最近目录列表（脱机项禁用并加前缀）。"""
        self._recent_list.clear()
        offline_prefix = self.i18n.t("menu.recent_dirs_offline")
        for directory in self.config.get_recent_directories():
            available = os.path.isdir(directory)
            label = directory if available else f"{offline_prefix} {directory}"
            item = QListWidgetItem(label)
            if not available:
                # 脱机目录置灰且不可选，与主窗口「最近目录」菜单行为一致
                # Offline folders are greyed out and unselectable, matching
                # the main window's recent-directories menu.
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
            item.setData(Qt.UserRole, directory)
            item.setData(Qt.UserRole + 1, available)
            self._recent_list.addItem(item)

        has_history = self._recent_list.count() > 0
        self._recent_list.setVisible(has_history)
        self._group_label.setVisible(has_history)
        self._empty_label.setVisible(not has_history)
        self._clear_button.setEnabled(has_history)
        self._open_button.setEnabled(False)

    # ------------------------------------------------------------------
    #  事件处理 / event handlers
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        """列表选中变化时同步「打开」按钮可用状态。"""
        item = self._recent_list.currentItem()
        self._open_button.setEnabled(bool(item) and item.data(Qt.UserRole + 1))

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """双击可用目录直接打开 / double-click an available folder to open it."""
        if item.data(Qt.UserRole + 1):
            self._accept_directory(item.data(Qt.UserRole))

    def _on_open_clicked(self):
        """「打开」按钮：打开当前选中的可用目录。"""
        item = self._recent_list.currentItem()
        if item and item.data(Qt.UserRole + 1):
            self._accept_directory(item.data(Qt.UserRole))

    def _on_browse_clicked(self):
        """
        「浏览」按钮：弹系统目录选择框。

        起始目录优先取最近列表中第一个仍可访问的目录，
        否则回退系统「图片」目录（与主窗口 _browse_directory 策略一致）。
        """
        start_dir = ""
        for directory in self.config.get_recent_directories():
            if os.path.isdir(directory):
                start_dir = directory
                break
        if not start_dir:
            start_dir = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.PicturesLocation
            ) or ""

        directory = QFileDialog.getExistingDirectory(
            self,
            self.i18n.t("labels.select_photo_dir"),
            start_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self._accept_directory(os.path.normpath(directory))

    def _on_clear_history(self):
        """清空最近目录历史（确认后写入配置并刷新列表）。"""
        answer = QMessageBox.question(
            self,
            self.i18n.t("menu.recent_dirs_clear"),
            self.i18n.t("browser.launcher_clear_confirm"),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.config.config["recent_directories"] = []
        self.config.save()
        self._refresh_recent_list()

    # ------------------------------------------------------------------
    #  结果返回 / result handling
    # ------------------------------------------------------------------

    def _accept_directory(self, directory: str):
        """记录用户选定的目录并关闭对话框（Accepted）。"""
        self.selected_directory = directory
        self.accept()
