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


def _run_selftest() -> int:
    """
    打包冒烟自测（隐藏参数 --selftest，仅用于 exe 回归验证）。

    在临时目录构造最小数据（sidecar JSON + 假 RAW + 同名 JPG），
    完整走一遍多鸟编辑器的图片回退链与鸟种搜索对话框的懒加载链——
    这两条是 SPBBrowse.exe 最脆弱的链路（cv2/rawpy/懒加载模块缺失时
    只会静默失败）。全部通过返回 0，任一失败打印异常返回 1。

    Packaged smoke test (hidden --selftest). Builds throwaway data and
    exercises the multibird editor's image fallback chain plus the species
    dialog's lazy-import chain — the two paths that fail silently in the
    exe when a module is missing. Returns 0 on success, 1 on failure.
    """
    import json
    import shutil
    import tempfile
    import traceback

    import numpy as np

    tmp = os.path.join(tempfile.gettempdir(), 'spb_browse_selftest')
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        os.makedirs(os.path.join(tmp, '.superpicky', 'meta'))

        # 假 RAW + 同名 JPG 边车 + sidecar JSON
        import cv2
        cv2.imwrite(os.path.join(tmp, 'SELFTEST.jpg'),
                    np.full((64, 64, 3), (30, 160, 220), np.uint8))
        with open(os.path.join(tmp, 'SELFTEST.cr3'), 'wb') as f:
            f.write(b'not-a-real-raw')
        with open(os.path.join(tmp, '.superpicky', 'meta', 'SELFTEST.json'),
                  'w', encoding='utf-8') as f:
            json.dump({"detections": [
                {"index": 0, "is_selected": True, "x": 8, "y": 8,
                 "w": 40, "h": 40,
                 "species": {"cn": "测试鸟", "en": "Test Bird",
                             "scientific": "Testus testus",
                             "confidence": 0.9}},
            ], "main_species": ["测试鸟"]}, f, ensure_ascii=False)

        # ① 多鸟编辑：RAW 应回退同名 JPG 并解码成功
        from ui.multibird_editor_dialog import MultibirdEditorDialog
        dlg = MultibirdEditorDialog(
            {'filename': 'SELFTEST',
             'current_path': os.path.join(tmp, 'SELFTEST.cr3'),
             'original_path': os.path.join(tmp, 'SELFTEST.cr3')},
            tmp, load_async=False)
        assert dlg._img_bgr is not None, 'multibird image fallback failed'
        dlg.close()
        print('selftest: multibird fallback OK')

        # ② 鸟种搜索对话框（懒加载链）
        from ui.bird_species_edit_dialog import BirdSpeciesEditDialog
        sd = BirdSpeciesEditDialog()
        sd.close()
        print('selftest: species dialog OK')

        # ③ rawpy 可导入（纯 RAW 兜底解码器；缺失时仅同名 JPG 场景可用）
        import rawpy  # noqa: F401
        print('selftest: rawpy OK')
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    """命令行入口 / CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)

    from PySide6.QtWidgets import QApplication

    from ui.results_browser_window import ResultsBrowserWindow

    app = QApplication.instance() or QApplication(argv)

    if argv and argv[0] == '--selftest':
        return _run_selftest()

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
