# -*- coding: utf-8 -*-
"""
spb_browse 启动器对话框 / Launcher dialog for spb_browse

双击 SPBBrowse.exe（或 `python spb_browse.py` 不带参数）时弹出，
提供两种选目录方式：
  1. 从清单直选：「已处理目录」（跑过 process 的目录，自动记录，
     与主程序共用 advanced_config.json）+「最近打开」（历史）
  2. 「浏览」按钮：系统目录选择框自己翻文件夹
另有「扫描导入」按钮：选一个根目录（如 NAS 观鸟根目录），自动发现
所有含 .superpicky/report.db 的子目录灌入已处理清单（存量初始化用）。

Launcher dialog for spb_browse, shown when SPBBrowse.exe starts without
arguments. Two ways to pick a folder: straight from the lists (processed
folders recorded automatically whenever process runs, plus recently opened
folders — both shared with the main app via advanced_config.json), or via
the system browse dialog. A Scan & Import button discovers all processed
sub-folders under a chosen root (e.g. the NAS birding root) and merges
them into the processed list in one shot.

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
from PySide6.QtGui import QColor, QFont

from advanced_config import get_advanced_config
from tools.i18n import get_i18n
from ui.styles import COLORS


class BrowseLauncherDialog(QDialog):
    """
    spb_browse 无参数启动时的「打开照片目录」启动器。

    界面构成：
      - 分组列表：「已处理目录」（processed_directories 全量）+
        「最近打开」（recent_directories，最多 10 条）
      - 不可访问的目录加「(脱机)」前缀并禁用
      - 按钮：扫描导入…（批量发现已处理子目录）/ 清除历史 /
        浏览（QFileDialog）/ 打开 / 取消

    选定目录通过 selected_directory 属性返回；用户取消时返回空字符串。

    Launcher shown when spb_browse starts without arguments. Shows a
    sectioned list — processed folders (full registry) and recently opened
    folders — with offline entries disabled, plus Scan & Import (bulk
    discovery of processed sub-folders), Clear-history, Browse, Open and
    Cancel buttons. The chosen folder is exposed via selected_directory;
    an empty string means the user cancelled.
    """

    _WIDTH = 560          # 固定宽度，路径较长时仍可读 / fixed width for long paths
    _LIST_MAX_HEIGHT = 320  # 列表最大高度（两组目录共用滚动区）/ shared list max height

    #: 目录项数据键：路径 / data role for a folder entry's path
    _ROLE_PATH = Qt.UserRole
    #: 目录项数据键：当前是否可访问 / data role for a folder entry's availability
    _ROLE_AVAILABLE = Qt.UserRole + 1

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
        self._refresh_folder_lists()

    # ------------------------------------------------------------------
    #  UI 构建 / UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """构建对话框 UI：标题提示 + 分组目录列表 + 按钮行。"""
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

        self._folder_list = QListWidget()
        self._folder_list.setMaximumHeight(self._LIST_MAX_HEIGHT)
        self._folder_list.setStyleSheet(
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
        self._folder_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._folder_list.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._folder_list)

        # 空列表提示（两组都为空时显示）/ placeholder when both lists are empty
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

        self._scan_button = QPushButton(self.i18n.t("browser.launcher_scan"))
        self._scan_button.clicked.connect(self._on_scan_clicked)
        buttons.addWidget(self._scan_button)

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
    #  分组目录列表 / sectioned folder list
    # ------------------------------------------------------------------

    def _add_section_header(self, title: str):
        """追加一个不可选中的分组标题行。"""
        item = QListWidgetItem(title)
        item.setFlags(Qt.NoItemFlags)  # 标题行不可选/不可用 / headers are inert
        font = QFont()
        font.setBold(True)
        item.setFont(font)
        item.setForeground(QColor(COLORS['accent']))
        self._folder_list.addItem(item)

    def _add_folder_item(self, directory: str):
        """追加一个目录行（脱机目录禁用并加前缀）。"""
        offline_prefix = self.i18n.t("menu.recent_dirs_offline")
        available = os.path.isdir(directory)
        label = directory if available else f"{offline_prefix} {directory}"
        item = QListWidgetItem(label)
        if not available:
            # 脱机目录置灰且不可选，与主窗口「最近目录」菜单行为一致
            # Offline folders are greyed out and unselectable, matching
            # the main window's recent-directories menu.
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
        item.setData(self._ROLE_PATH, directory)
        item.setData(self._ROLE_AVAILABLE, available)
        self._folder_list.addItem(item)

    def _refresh_folder_lists(self):
        """
        根据配置重建分组列表：「已处理目录」+「最近打开」。

        已处理组始终显示（哪怕为空也保留标题，配合扫描导入引导）；
        两组都无条目时显示空态提示。
        """
        self._folder_list.clear()

        self._add_section_header(self.i18n.t("browser.launcher_processed_group"))
        processed = self.config.get_processed_directories()
        for directory in processed:
            self._add_folder_item(directory)

        self._add_section_header(self.i18n.t("menu.recent_dirs"))
        recents = self.config.get_recent_directories()
        for directory in recents:
            self._add_folder_item(directory)

        has_entries = bool(processed or recents)
        self._folder_list.setVisible(has_entries)
        self._empty_label.setVisible(not has_entries)
        self._clear_button.setEnabled(bool(recents))
        self._open_button.setEnabled(False)

    # ------------------------------------------------------------------
    #  事件处理 / event handlers
    # ------------------------------------------------------------------

    def _current_folder_item(self) -> QListWidgetItem | None:
        """返回当前选中的可用目录项（标题行/脱机项/无选中 → None）。"""
        item = self._folder_list.currentItem()
        if item is None or not item.flags() & Qt.ItemIsSelectable:
            return None
        return item

    def _on_selection_changed(self):
        """列表选中变化时同步「打开」按钮可用状态。"""
        item = self._current_folder_item()
        self._open_button.setEnabled(bool(item) and item.data(self._ROLE_AVAILABLE))

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """双击可用目录直接打开 / double-click an available folder to open it."""
        if item.data(self._ROLE_PATH) and item.data(self._ROLE_AVAILABLE):
            self._accept_directory(item.data(self._ROLE_PATH))

    def _on_open_clicked(self):
        """「打开」按钮：打开当前选中的可用目录。"""
        item = self._current_folder_item()
        if item and item.data(self._ROLE_AVAILABLE):
            self._accept_directory(item.data(self._ROLE_PATH))

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

    def _on_scan_clicked(self):
        """
        「扫描导入」：选一个根目录，自动发现所有已处理子目录并合并进清单。

        用于把存量目录（如 NAS 观鸟根目录下已跑过 process 的十几个目录）
        一次性灌入「已处理目录」清单。起始目录优先取已知目录的父目录
        （大概率就是 NAS 根），否则回退系统「图片」目录。
        """
        start_dir = ""
        for candidates in (
            self.config.get_recent_directories(),
            self.config.get_processed_directories(),
        ):
            for directory in candidates:
                if os.path.isdir(directory):
                    start_dir = os.path.dirname(directory)
                    break
            if start_dir and os.path.isdir(start_dir):
                break
        if not (start_dir and os.path.isdir(start_dir)):
            start_dir = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.PicturesLocation
            ) or ""

        root_dir = QFileDialog.getExistingDirectory(
            self,
            self.i18n.t("browser.launcher_scan"),
            start_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not root_dir:
            return

        from tools.merged_report_db import find_processed_subdirs

        found = find_processed_subdirs(root_dir)
        if not found:
            QMessageBox.information(
                self,
                self.i18n.t("browser.launcher_scan"),
                self.i18n.t("browser.launcher_scan_none"),
            )
            return

        added = self.config.add_processed_directories(found)
        self._refresh_folder_lists()
        QMessageBox.information(
            self,
            self.i18n.t("browser.launcher_scan"),
            self.i18n.t("browser.launcher_scan_done", n=added),
        )

    def _on_clear_history(self):
        """清空「最近打开」历史（确认后写入配置并刷新列表）。"""
        answer = QMessageBox.question(
            self,
            self.i18n.t("menu.recent_dirs_clear"),
            self.i18n.t("browser.launcher_clear_confirm"),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.config.config["recent_directories"] = []
        self.config.save()
        self._refresh_folder_lists()

    # ------------------------------------------------------------------
    #  结果返回 / result handling
    # ------------------------------------------------------------------

    def _accept_directory(self, directory: str):
        """记录用户选定的目录并关闭对话框（Accepted）。"""
        self.selected_directory = directory
        self.accept()
