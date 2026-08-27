#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_browse — 独立结果浏览器启动器

直接打开某个已处理照片目录的结果浏览器（缩略图 + 详情面板 + 多鸟编辑
右键菜单），不经过主界面。目录需含 .superpicky/report.db（跑过
process 即有）；传父目录时自动发现所有已处理子目录进入合并模式。

双轨入口 / dual entry modes:
    python spb_browse.py                # 无参数：弹出「打开照片目录」启动器
                                        # （最近目录 + 浏览按钮，与主程序互通）
    python spb_browse.py <照片目录>      # 带参数：直接打开指定目录
                                        # （目录本身或其子目录含处理结果即可）

打包为 SPBBrowse.exe 后双击即走无参数启动器路径。

Standalone launcher for the results browser on an already-processed
photo directory. Supports a launcher dialog (recent folders + browse)
when started without arguments, and direct open with a directory argument.
"""

from __future__ import annotations

import os
import sys


def _find_processed_directory_or_none(directory: str) -> str | None:
    """
    校验目录是否含处理结果（report.db），返回可用目录或 None。

    认可两种情况 / accepts either of:
      1. 目录本身含 .superpicky/report.db（单目录模式）
      2. 目录的子目录中含处理结果（父目录合并模式，
         与 ResultsBrowserWindow.open_directory 的判定一致）

    参数:
    directory (str): 候选照片目录

    返回:
    str | None: 校验通过返回原目录，否则 None

    Validate that the directory holds processed results and return it,
    or None when nothing is found.
    """
    from tools.merged_report_db import find_processed_subdirs

    if find_processed_subdirs(directory):
        return directory
    return None


def _pick_directory_via_launcher(app) -> str:
    """
    无参数启动路径：弹出启动器对话框让用户选目录。

    参数:
    app: 已创建的 QApplication 实例

    返回:
    str: 用户选定的目录；取消时返回空字符串

    Launcher-dialog path: let the user pick a folder interactively.
    Returns "" when the user cancels.
    """
    from PySide6.QtWidgets import QDialog

    from ui.browse_launcher_dialog import BrowseLauncherDialog

    dialog = BrowseLauncherDialog(parent=None)
    if dialog.exec() == QDialog.Accepted and dialog.selected_directory:
        return dialog.selected_directory
    return ""


def main(argv=None) -> int:
    """命令行入口 / CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)

    from PySide6.QtWidgets import QApplication

    from ui.results_browser_window import ResultsBrowserWindow

    app = QApplication.instance() or QApplication(argv)

    if argv:
        # 带参数：CLI 直开 / with argument: open directly
        directory = argv[0]
        if not os.path.isdir(directory):
            print(f"目录不存在 / not a directory: {directory}")
            return 1
        if _find_processed_directory_or_none(directory) is None:
            db_path = os.path.join(directory, ".superpicky", "report.db")
            print(f"未找到处理结果 ({db_path}，子目录中也没有)，"
                  f"请先跑 process / no processed results found, "
                  f"run process first")
            return 1
    else:
        # 无参数：启动器对话框 / no argument: launcher dialog
        directory = _pick_directory_via_launcher(app)
        if not directory:
            return 0

    # 最近目录历史由 ResultsBrowserWindow.open_directory 统一记录
    # （CLI / 启动器 / Ctrl+O / 最近菜单全部入口共用同一逻辑）。
    # Recent-folder history is recorded inside open_directory, covering
    # every entry path (CLI / launcher / Ctrl+O / recent menu).
    win = ResultsBrowserWindow(parent=None)
    win.closed.connect(app.quit)
    win.open_directory(directory)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
