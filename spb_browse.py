#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_browse — 独立结果浏览器启动器

直接打开某个已处理照片目录的结果浏览器（缩略图 + 详情面板 + 多鸟编辑
右键菜单），不经过主界面。目录需含 .superpicky/report.db（跑过
process 即有）。

用法:
    python spb_browse.py <照片目录>

Standalone launcher for the results browser on an already-processed
photo directory.
"""

from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    """命令行入口 / CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法 / usage: python spb_browse.py <照片目录>")
        return 1
    directory = argv[0]
    if not os.path.isdir(directory):
        print(f"目录不存在 / not a directory: {directory}")
        return 1

    from PySide6.QtWidgets import QApplication

    from ui.results_browser_window import ResultsBrowserWindow

    app = QApplication.instance() or QApplication(argv)
    db_path = os.path.join(directory, ".superpicky", "report.db")
    if not os.path.exists(db_path):
        print(f"未找到处理结果 ({db_path})，请先跑 process / "
              f"run process first")
        return 1
    win = ResultsBrowserWindow(parent=None)
    win.closed.connect(app.quit)
    win.open_directory(directory)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
