# -*- coding: utf-8 -*-
"""
多鸟编辑器布局探针 v2：等待图片与布局稳定后，用全局坐标转换遍历右侧
面板所有可见控件，输出几何矩形并检测非父子控件间的真实重叠。

用法：python scripts_dev/probe_multibird_ui.py <照片目录> <文件前缀>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QTimer

PHOTO_DIR = sys.argv[1] if len(sys.argv) > 1 else "G:/photo/20260822_merged"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "027A3684"


def dump_overlaps(win):
    """用全局坐标遍历可见控件，报告非父子关系的兄弟重叠。"""
    win_rect = win.rect()
    widgets = [w for w in win.findChildren(QWidget) if w.isVisible()]
    rows = []
    for w in widgets:
        g = w.geometry()
        glo = w.mapToGlobal(g.topLeft())
        loc = win.mapFromGlobal(glo)
        # 标注父类链，方便定位
        chain = []
        p = w.parentWidget()
        while p is not None and p is not win:
            chain.append(type(p).__name__)
            p = p.parentWidget()
        rows.append((w, f"{type(w).__name__}<-{'<-'.join(chain[:3])}",
                     loc.x(), loc.y(), g.width(), g.height()))
    print(f"窗口客户区: {win_rect.width()}x{win_rect.height()}")
    print("== 控件几何（客户区坐标，仅可视区内） ==")
    vis = [r for r in rows if r[4] > 0 and r[5] > 0
           and r[2] < win_rect.width() and r[3] < win_rect.height()]
    for w, name, x, y, w_, h_ in vis:
        print(f"{name:52s} x={x:5d} y={y:5d} w={w_:5d} h={h_:5d}")
    print("== 非父子重叠对 ==")
    n = 0
    for i in range(len(vis)):
        for j in range(i + 1, len(vis)):
            a, b = vis[i], vis[j]
            wa, wb = a[0], b[0]
            if wa.isAncestorOf(wb) or wb.isAncestorOf(wa):
                continue
            ox = max(0, min(a[2] + a[4], b[2] + b[4]) - max(a[2], b[2]))
            oy = max(0, min(a[3] + a[5], b[3] + b[5]) - max(a[3], b[3]))
            if ox > 4 and oy > 4:
                n += 1
                print(f"OVERLAP [{a[1]}] x [{b[1]}] 交集{ox}x{oy} "
                      f"@({max(a[2], b[2])},{max(a[3], b[3])})")
    if n == 0:
        print("（无）")
    # 额外：QLabel 文字是否超出自身矩形（潜在被截断/压线）
    print("== 文字超宽的 QLabel ==")
    fm = win.fontMetrics()
    for w, name, x, y, w_, h_ in vis:
        if type(w).__name__ == "QLabel" and w.text() and w.wordWrap() is False:
            need = fm.horizontalAdvance(w.text())
            if need > w_ + 6:
                print(f"WIDE {name} 需{need}px 实际{w_}px: {w.text()[:40]!r}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    from tools.report_db import ReportDB
    from ui.multibird_editor_dialog import MultibirdEditorDialog

    db = ReportDB(PHOTO_DIR)
    photo = db.get_photo(PREFIX)
    db._conn.close()
    if not photo:
        print(f"找不到照片 {PREFIX}")
        return
    photo["current_path"] = os.path.join(PHOTO_DIR, photo["current_path"])
    dlg = MultibirdEditorDialog(photo, PHOTO_DIR, parent=None)

    def _after():
        QApplication.processEvents()
        dlg.layout().activate()
        QApplication.processEvents()
        dump_overlaps(dlg)
        dlg.grab().save(r"C:\Users\yuhao\AppData\Local\Temp\mb_repro.png")
        print("截图已保存 mb_repro.png")
        dlg.close()
        app.quit()

    QTimer.singleShot(9000, _after)  # 等 RAW 解码 + 布局稳定
    dlg.show()
    app.exec()


if __name__ == "__main__":
    main()
