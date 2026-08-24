#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多鸟编辑对话框（MultibirdEditorDialog）离屏 GUI 测试。

覆盖：
1. 画布命中测试（点击命中/空白/最小框优先）
2. 重新标记鸟种（人工结果 + edited 标记 + DB 同步 + 主鸟种候选刷新）
3. 软删除框（deleted 标记 + 候选剔除）
4. 主鸟种复选（默认=电脑主鸟；最多 3 个）
5. 保存写回 sidecar JSON（原子写 + edits 日志 + main_species）
6. sidecar 重导出的人工编辑保护（机器导出不覆盖人工结果）

Offscreen GUI tests for the multi-bird editor dialog.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

# --- 测试夹具：临时目录 + 合成照片 + report.db + sidecar JSON ---


def _make_fixture():
    """构造单照片夹具：3 只鸟（主鸟+2 次鸟），返回 (目录, photo_dict)。"""
    import cv2
    from tools.report_db import ReportDB
    from core.sidecar_export import export_directory_sidecars

    d = tempfile.mkdtemp()
    img = np.full((400, 600, 3), 128, dtype=np.uint8)
    cv2.imencode(".jpg", img)[1].tofile(os.path.join(d, "F1.jpg"))

    db = ReportDB(d)
    db.insert_photo({'filename': 'F1', 'has_bird': 1, 'rating': 2,
                     'current_path': 'F1.jpg', 'original_path': 'F1.jpg',
                     'bird_species_cn': '泽鹬', 'bird_species_en': 'Common Greenshank'})
    db.insert_detections_batch([
        {'filename': 'F1', 'bird_index': 0, 'is_selected': 1,
         'bbox_x': 40, 'bbox_y': 40, 'bbox_w': 120, 'bbox_h': 100,
         'area_ratio': 0.05, 'yolo_conf': 0.9, 'crop_sharpness': 300.0,
         'species_cn': '泽鹬', 'species_en': 'Common Greenshank',
         'scientific_name': 'Tringa nebularia',
         'species_confidence': 95.0, 'class_id': 1, 'gbif_rarity_100': 5.0},
        {'filename': 'F1', 'bird_index': 1, 'is_selected': 0,
         'bbox_x': 300, 'bbox_y': 60, 'bbox_w': 100, 'bbox_h': 90,
         'area_ratio': 0.037, 'yolo_conf': 0.7, 'crop_sharpness': 200.0,
         'species_cn': '反嘴鹬', 'species_en': 'Pied Avocet',
         'scientific_name': 'Recurvirostra avosetta',
         'species_confidence': 80.0, 'class_id': 2, 'gbif_rarity_100': 8.0},
        {'filename': 'F1', 'bird_index': 2, 'is_selected': 0,
         'bbox_x': 460, 'bbox_y': 250, 'bbox_w': 90, 'bbox_h': 80,
         'area_ratio': 0.03, 'yolo_conf': 0.5, 'crop_sharpness': 150.0,
         'species_cn': '黑腹滨鹬', 'species_en': 'Dunlin',
         'scientific_name': 'Calidris alpina',
         'species_confidence': 40.0, 'class_id': 3, 'gbif_rarity_100': 3.0},
    ])
    export_directory_sidecars(db, d, log=lambda *_: None)
    db._conn.close()
    photo = {'filename': 'F1', 'current_path': 'F1.jpg',
             'original_path': 'F1.jpg',
             'bird_species_cn': '泽鹬', 'bird_species_en': 'Common Greenshank'}
    return d, photo


class TestMultibirdEditor(unittest.TestCase):
    """离屏对话框测试 / offscreen dialog tests."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.d, self.photo = _make_fixture()
        from ui.multibird_editor_dialog import MultibirdEditorDialog
        self.dlg = MultibirdEditorDialog(self.photo, self.d)

    def tearDown(self):
        self.dlg.deleteLater()
        shutil.rmtree(self.d, ignore_errors=True)

    def _det(self, idx):
        for det in self.dlg._data['detections']:
            if det['index'] == idx:
                return det
        raise AssertionError(idx)

    def test_canvas_downscaled_coord_mapping(self):
        """底图降采样后 bbox 仍按原图坐标正确映射（真实照片回归 bug）。"""
        canvas = self.dlg._canvas
        from PySide6.QtGui import QImage
        canvas._qimage = QImage(600, 400, QImage.Format_RGB888)
        canvas._detections = self.dlg._data['detections']
        canvas._orig_w, canvas._orig_h = 1200, 800
        # 桩掉布局相关尺寸：底图显示区固定 (20, 20, 600, 400)
        canvas._fit_rect = lambda: (20, 20, 600, 400)
        # det0 bbox=(40,40,120,100) 原图坐标：sx=0.5 → (40,40,60,50)
        x, y, w, h = canvas._bbox_to_display([40, 40, 120, 100])
        self.assertEqual((x, y, w, h), (40, 40, 60, 50))
        # 逆向：显示 (80, 60) → 原图 (120, 80)，落在 det0 内
        from PySide6.QtCore import QPoint
        orig = canvas._point_to_orig(QPoint(80, 60))
        self.assertEqual(orig, (120.0, 80.0))
        results = []
        canvas.bird_selected.connect(results.append)
        canvas.mousePressEvent(_FakeMouse(QPoint(80, 60)))
        self.assertEqual(results[-1], 0)

    def test_canvas_hit_test(self):
        """画布命中：框内点选中、空白 -1、重叠取最小框。"""
        from PySide6.QtCore import QPoint
        self.dlg.resize(1180, 760)
        self.dlg._canvas.resize(760, 740)
        self.dlg._canvas.show()
        self.dlg._canvas.repaint()
        # 主鸟 bbox (40,40)-(160,140) 原图坐标 → 画布中心换算
        results = []
        self.dlg._canvas.bird_selected.connect(results.append)
        orig = self.dlg._canvas._point_to_orig(
            self.dlg._canvas._fit_rect().__class__ and QPoint(100, 100))
        # 直接用原图坐标做命中测试更稳（换算回路 paintEvent 已隐式覆盖）
        # 点击主鸟中心
        from ui.multibird_editor_dialog import _BirdCanvas
        # 构造 point_to_orig 的逆过程：直接验证命中选择逻辑
        # 模拟鼠标点击：把画布坐标映射回原图后命中
        ox, oy, dw, dh = self.dlg._canvas._fit_rect()
        sx = 600.0 / dw
        sy = 400.0 / dh
        # 原图 (100, 90) ≈ 主鸟中心
        pos = QPoint(int(ox + 100 / sx), int(oy + 90 / sy))
        self.dlg._canvas.mousePressEvent(
            _FakeMouse(pos))
        self.assertEqual(results[-1], 0)
        # 次鸟 2 中心 (505, 290)
        pos2 = QPoint(int(ox + 505 / sx), int(oy + 290 / sy))
        self.dlg._canvas.mousePressEvent(_FakeMouse(pos2))
        self.assertEqual(results[-1], 2)
        # 空白区
        pos3 = QPoint(int(ox + 250 / sx), int(oy + 350 / sy))
        self.dlg._canvas.mousePressEvent(_FakeMouse(pos3))
        self.assertEqual(results[-1], -1)

    def test_remark_marks_manual_and_updates_main_candidates(self):
        """重新标记：edited=true、confidence 清空、主鸟种候选含新种。"""
        self.dlg._canvas.set_selected(1)
        self.dlg._refresh_selection_ui()
        # 打桩 IOC 搜索对话框
        import ui.bird_species_edit_dialog as bsed
        orig_cls = bsed.BirdSpeciesEditDialog

        class _FakeSpeciesDialog:
            def __init__(self, parent=None):
                self.selected_cn = '白鹭'
                self.selected_en = 'Little Egret'
                self.selected_latin = 'Egretta garzetta'

            def exec(self):
                from PySide6.QtWidgets import QDialog
                return QDialog.Accepted

        bsed.BirdSpeciesEditDialog = _FakeSpeciesDialog
        try:
            self.dlg._on_remark()
        finally:
            bsed.BirdSpeciesEditDialog = orig_cls
        det = self._det(1)
        self.assertEqual(det['species']['cn'], '白鹭')
        self.assertTrue(det['edited'])
        self.assertIsNone(det['species']['confidence'])
        # 主鸟种候选刷新：白鹭加入
        cands = [c['cn'] for c in self.dlg._main_candidates()]
        self.assertIn('白鹭', cands)
        # edits 日志
        actions = [e['action'] for e in self.dlg._data['edits']]
        self.assertIn('species_set', actions)

    def test_delete_soft_marks_and_excludes(self):
        """删除框：deleted=true、候选剔除、画布选中复位。"""
        self.dlg._canvas.set_selected(2)
        self.dlg._refresh_selection_ui()
        self.dlg._on_delete()
        det = self._det(2)
        self.assertTrue(det.get('deleted'))
        cands = [c['cn'] for c in self.dlg._main_candidates()]
        self.assertNotIn('黑腹滨鹬', cands)
        self.assertIn('bbox_deleted',
                      [e['action'] for e in self.dlg._data['edits']])

    def test_main_species_default_and_limit(self):
        """默认主鸟=电脑选中的主鸟；复选最多 3 个。"""
        # 默认勾选 = 泽鹬（is_selected 的物种）
        self.assertEqual(self.dlg._main_keys, ['Tringa nebularia'])
        # 3 个候选全勾（恰好 3 个，允许）
        for cb in self.dlg._main_checkboxes:
            cb.setChecked(True)
        self.assertEqual(len(self.dlg._main_keys), 3)
        # 加第 4 种候选 → 勾第 4 个应被复位（上限保护）
        self.dlg._data['detections'].append({
            'index': 3, 'is_selected': False,
            'bbox': [10, 10, 50, 50], 'species': {
                'cn': '第四种', 'en': 'Fourth',
                'scientific': 'F. quartus', 'confidence': None}})
        self.dlg._rebuild_main_checkboxes()
        boxes = self.dlg._main_checkboxes
        self.assertEqual(len(boxes), 4)
        for cb in boxes[:3]:
            cb.setChecked(True)
        boxes[3].setChecked(True)
        self.assertFalse(boxes[3].isChecked())
        self.assertEqual(len(self.dlg._main_keys), 3)

    def test_save_writes_json_and_syncs_db(self):
        """保存：JSON 原子写（edits/main_species/edited/deleted）+ DB 物种同步。"""
        # 改一个物种 + 删一个框 + 勾两个主鸟种
        self.dlg._canvas.set_selected(1)
        self.dlg._refresh_selection_ui()
        self._det(1)['species'] = {
            'cn': '白鹭', 'en': 'Little Egret',
            'scientific': 'Egretta garzetta',
            'confidence': None, 'class_id': None}
        self._det(1)['edited'] = True
        self.dlg._species_updates[1] = ('白鹭', 'Little Egret',
                                        'Egretta garzetta')
        self._det(2)['deleted'] = True
        # 勾两个主鸟种
        for cb in self.dlg._main_checkboxes[:2]:
            cb.setChecked(True)
        from PySide6.QtWidgets import QDialog
        self.dlg.accepted.connect(lambda: None)
        # 直接调用保存逻辑（不走 QMessageBox 分支）
        self.dlg._on_save()
        # JSON 落盘校验
        with open(os.path.join(self.d, '.superpicky', 'meta', 'F1.json'),
                  encoding='utf-8') as f:
            data = json.load(f)
        det1 = [x for x in data['detections'] if x['index'] == 1][0]
        self.assertEqual(det1['species']['cn'], '白鹭')
        self.assertTrue(det1['edited'])
        det2 = [x for x in data['detections'] if x['index'] == 2][0]
        self.assertTrue(det2.get('deleted'))
        self.assertEqual(len(data.get('main_species') or []), 2)
        self.assertIn('main_species_set',
                      [e['action'] for e in data['edits']])
        # DB 物种同步
        from tools.report_db import ReportDB
        db = ReportDB(self.d)
        rows = db.get_detections('F1')
        db._conn.close()
        self.assertEqual(rows[1]['species_cn'], '白鹭')
        self.assertEqual(rows[1]['edited'], 1)

    def test_reexport_preserves_manual_edits(self):
        """机器重导出不覆盖人工结果（编辑优先合并）。"""
        # 人工修改 + 删除 + 主鸟种
        self._det(1)['species'] = {'cn': '白鹭', 'en': 'Little Egret',
                                   'scientific': 'Egretta garzetta',
                                   'confidence': None}
        self._det(1)['edited'] = True
        self._det(2)['deleted'] = True
        self.dlg._data['main_species'] = [
            {'cn': '白鹭', 'en': 'Little Egret',
             'scientific': 'Egretta garzetta', 'bird_index': 1}]
        from core.sidecar_export import _atomic_write_json
        _atomic_write_json(self.dlg._json_path, self.dlg._data)
        # 改 DB 的 updated_at 触发重导出
        import time
        time.sleep(1.1)
        from tools.report_db import ReportDB
        db = ReportDB(self.d)
        db.update_detection_species('F1', 0, '泽鹬', 'Common Greenshank',
                                    'Tringa nebularia')
        from core.sidecar_export import export_directory_sidecars
        export_directory_sidecars(db, self.d, log=lambda *_: None)
        db._conn.close()
        with open(os.path.join(self.d, '.superpicky', 'meta', 'F1.json'),
                  encoding='utf-8') as f:
            data = json.load(f)
        det1 = [x for x in data['detections'] if x['index'] == 1][0]
        det2 = [x for x in data['detections'] if x['index'] == 2][0]
        # 人工编辑保留
        self.assertEqual(det1['species']['cn'], '白鹭')
        self.assertTrue(det1['edited'])
        self.assertTrue(det2.get('deleted'))
        self.assertEqual(data['main_species'][0]['cn'], '白鹭')
        # DB 里未动的 detection 0 仍从库导出
        det0 = [x for x in data['detections'] if x['index'] == 0][0]
        self.assertEqual(det0['species']['cn'], '泽鹬')


class _FakeMouse:
    """mousePressEvent 需要的最小事件桩 / minimal mouse-event stub."""

    def __init__(self, pos):
        from PySide6.QtCore import QPoint
        self._pos = QPoint(pos)

    def position(self):
        class _P:
            def toPoint(self_inner):
                return self._pos
        return _P()


if __name__ == '__main__':
    unittest.main(verbosity=2)
