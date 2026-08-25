#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多鸟逐鸟识别（multibird）单元测试。

覆盖三块：
1. report.db v9→v10 升级：旧库数据不丢、bird_detections 自动创建；
2. bird_detections 入库幂等：重跑（先删后插）不产生重复行；
3. core/multi_bird.classify_secondary_birds 的门槛逻辑：
   面积边界（<min_area 只入框不分类）、置信度边界（<threshold 物种留空）、
   主鸟复用（is_selected 行不重复推理）。

Unit tests for the multi-bird per-bird classification feature:
schema v9→v10 migration, idempotent detection persistence, and the
min-area / confidence thresholds of classify_secondary_birds.
"""
import json
import os
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np

from tools.report_db import ReportDB, SCHEMA_VERSION


def _make_rows(filename='DSC_0001', n=2, with_species=(True, False)):
    """构造 n 行检测数据（中文物种名用于 UTF-8 回读验证）。"""
    names = [('鸡尾鹦鹉', 'Cockatiel', 'Nymphicus hollandicus'),
             ('虎皮鹦鹉', 'Budgerigar', 'Melopsittacus undulatus')]
    rows = []
    for i in range(n):
        row = {
            'filename': filename,
            'bird_index': i,
            'is_selected': 1 if i == 0 else 0,
            'bbox_x': 100.0 * (i + 1), 'bbox_y': 200.0,
            'bbox_w': 300.0, 'bbox_h': 400.0,
            'mask_polygon': json.dumps([[100, 200], [400, 200], [400, 600]]),
            'area_ratio': 0.05 * (i + 1),
            'yolo_conf': 0.9 - 0.1 * i,
            'crop_sharpness': 500.0 - 100.0 * i,
        }
        if with_species[i]:
            cn, en, sci = names[i % len(names)]
            row.update({
                'species_cn': cn, 'species_en': en,
                'scientific_name': sci,
                'species_confidence': 88.5,
                'class_id': 100 + i,
                'gbif_rarity_100': 12.0,
            })
        rows.append(row)
    return rows


class TestSchemaV10Migration(unittest.TestCase):
    """v9 旧库打开后自动升级 v10，photos 数据保留。"""

    def _create_v9_db(self, db_dir):
        """手工建一个 v9 库：photos 表带一行数据、meta 版本号 9。"""
        sp_dir = os.path.join(db_dir, '.superpicky')
        os.makedirs(sp_dir, exist_ok=True)
        db_path = os.path.join(sp_dir, 'report.db')
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE photos (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "filename TEXT UNIQUE, has_bird INTEGER, confidence REAL, rating INTEGER)")
        conn.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '9')")
        conn.execute(
            "INSERT INTO photos (filename, has_bird, confidence, rating) "
            "VALUES ('OLD_0001', 1, 0.87, 3)")
        conn.commit()
        conn.close()
        return db_path

    def test_v9_upgrades_and_keeps_data(self):
        db_dir = tempfile.mkdtemp()
        try:
            self._create_v9_db(db_dir)
            db = ReportDB(db_dir)
            # 版本连续升级到当前 schema（V5.2 起为 11）
            ver = db._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
            self.assertEqual(ver, SCHEMA_VERSION)
            self.assertEqual(SCHEMA_VERSION, "11")
            # 旧 photos 数据仍在
            photo = db.get_photo('OLD_0001')
            self.assertIsNotNone(photo)
            self.assertEqual(photo['rating'], 3)
            # bird_detections 表已创建且可写
            db.insert_detections_batch(_make_rows('OLD_0001'))
            self.assertEqual(len(db.get_detections('OLD_0001')), 2)
            db._conn.close()
        finally:
            shutil.rmtree(db_dir, ignore_errors=True)


class TestDetectionsPersistence(unittest.TestCase):
    """bird_detections 入库与幂等重跑。"""

    def test_insert_idempotent_and_utf8(self):
        db_dir = tempfile.mkdtemp()
        try:
            db = ReportDB(db_dir)
            rows = _make_rows()
            db.insert_detections_batch(rows)
            self.assertEqual(len(db.get_detections('DSC_0001')), 2)
            # 幂等：同照片重插仍是 2 行（先删后插）
            db.insert_detections_batch(rows)
            self.assertEqual(len(db.get_detections('DSC_0001')), 2)
            # 中文物种名 UTF-8 回读无损
            got = db.get_detections('DSC_0001')
            self.assertEqual(got[0]['species_cn'], '鸡尾鹦鹉')
            self.assertIsNone(got[1]['species_cn'])
            # polygon JSON 可解析
            poly = json.loads(got[0]['mask_polygon'])
            self.assertEqual(poly[0], [100, 200])
            # 混入不同 filename 应拒绝（防脏数据）
            bad = [dict(rows[0]), dict(rows[1], filename='OTHER')]
            with self.assertRaises(ValueError):
                db.insert_detections_batch(bad)
            db._conn.close()
        finally:
            shutil.rmtree(db_dir, ignore_errors=True)


class TestClassifySecondaryBirds(unittest.TestCase):
    """classify_secondary_birds 的门槛逻辑（注入假识别器，不加载模型）。"""

    def setUp(self):
        # 200x300 原图 / 100x150 处理图 → 缩放比 2.0
        rng = np.random.default_rng(42)
        self.orig = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        self.calls = []

    def _fake_identify(self, confidence=88.5):
        def fake(path, use_yolo, use_gps, use_geo, cc, rc, top_k, nf, crop):
            self.calls.append({'path': path, 'crop_size': crop.size})
            return {'success': True, 'results': [{
                'cn_name': '虎皮鹦鹉', 'en_name': 'Budgerigar',
                'scientific_name': 'Melopsittacus undulatus',
                'confidence': confidence, 'class_id': 7,
                'gbif_rarity_100': 5.0}]}
        return fake

    def _run(self, all_birds, main_species=None, min_area=0.001,
             identify=None):
        from core.multi_bird import classify_secondary_birds
        return classify_secondary_birds(
            self.orig, all_birds,
            proc_dims=(150, 100), orig_dims=(300, 200),
            main_species=main_species, filename='DSC_0001',
            photo_path='X:/DSC_0001.NEF',
            min_area_ratio=min_area,
            identify_fn=identify or self._fake_identify())

    def test_small_bird_boxed_but_not_classified(self):
        """面积 < min_area_ratio：只入框不分类（识别器不被调用）。"""
        birds = [
            {'idx': 0, 'conf': 0.9, 'bbox': (10, 10, 60, 60),
             'area_ratio': 0.09, 'mask_polygon': None, 'is_selected': True},
            {'idx': 1, 'conf': 0.7, 'bbox': (80, 80, 86, 84),
             'area_ratio': 0.0009, 'mask_polygon': None, 'is_selected': False},
        ]
        rows = self._run(birds)
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.calls, [])  # 小鸟没触发分类
        self.assertIsNone(rows[1]['species_cn'])
        self.assertIsNotNone(rows[1]['bbox_w'])  # 框照入
        self.assertIsNotNone(rows[1]['crop_sharpness'])

    def test_area_boundary_exact_equal_classifies(self):
        """面积恰好等于阈值 → 分类（>= 语义）。"""
        birds = [
            {'idx': 0, 'conf': 0.9, 'bbox': (10, 10, 60, 60),
             'area_ratio': 0.09, 'mask_polygon': None, 'is_selected': True},
            {'idx': 1, 'conf': 0.7, 'bbox': (60, 60, 110, 90),
             'area_ratio': 0.001, 'mask_polygon': None, 'is_selected': False},
        ]
        rows = self._run(birds, min_area=0.001)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(rows[1]['species_cn'], '虎皮鹦鹉')

    def test_low_confidence_result_still_stored(self):
        """置信度低于阈值也照实入库（采纳是消费方派生概念）。"""
        birds = [
            {'idx': 0, 'conf': 0.9, 'bbox': (10, 10, 60, 60),
             'area_ratio': 0.09, 'mask_polygon': None, 'is_selected': True},
            {'idx': 1, 'conf': 0.7, 'bbox': (60, 60, 110, 90),
             'area_ratio': 0.03, 'mask_polygon': None, 'is_selected': False},
        ]
        rows = self._run(birds,
                         identify=self._fake_identify(confidence=30.0))
        self.assertEqual(len(self.calls), 1)
        # 低置信结果保留：数据层完整，采纳由展示层按阈值判断
        self.assertEqual(rows[1]['species_cn'], '虎皮鹦鹉')
        self.assertEqual(rows[1]['species_confidence'], 30.0)
        self.assertIsNotNone(rows[1]['crop_sharpness'])

    def test_selected_bird_reuses_main_result(self):
        """主鸟行复用 main_species，不触发识别器。"""
        birds = [
            {'idx': 0, 'conf': 0.9, 'bbox': (10, 10, 60, 60),
             'area_ratio': 0.09, 'mask_polygon': None, 'is_selected': True},
            {'idx': 1, 'conf': 0.7, 'bbox': (60, 60, 110, 90),
             'area_ratio': 0.03, 'mask_polygon': None, 'is_selected': False},
        ]
        main = {'cn': '鸡尾鹦鹉', 'en': 'Cockatiel',
                'scientific': 'Nymphicus hollandicus',
                'confidence': 99.0, 'class_id': 123, 'gbif_rarity_100': 12.0}
        rows = self._run(birds, main_species=main)
        # 只有次鸟调用了一次识别器
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(rows[0]['species_cn'], '鸡尾鹦鹉')
        self.assertEqual(rows[0]['is_selected'], 1)
        self.assertEqual(rows[1]['species_cn'], '虎皮鹦鹉')
        # 主鸟的 main_species=None 时行物种留空（低于用户阈值场景）
        rows2 = self._run(birds, main_species=None)
        self.assertIsNone(rows2[0]['species_cn'])
        self.assertEqual(rows2[0]['is_selected'], 1)

    def test_square_crop_geometry(self):
        """次鸟裁剪为方形（最长边 × 1.15 padding）。"""
        birds = [
            {'idx': 0, 'conf': 0.9, 'bbox': (10, 10, 60, 60),
             'area_ratio': 0.09, 'mask_polygon': None, 'is_selected': True},
            # 处理图 (60,60)-(110,90) → 原图 (120,120)-(220,180)：100x60 → 方 115
            {'idx': 1, 'conf': 0.7, 'bbox': (60, 60, 110, 90),
             'area_ratio': 0.03, 'mask_polygon': None, 'is_selected': False},
        ]
        self._run(birds)
        w, h = self.calls[0]['crop_size']
        self.assertEqual(w, h)
        self.assertEqual(w, 115)
        self.assertEqual(self.calls[0]['path'], 'X:/DSC_0001.NEF')

    def test_polygon_scaled_to_orig(self):
        """轮廓点从处理图坐标缩放到原图坐标（比例 2.0），JSON 序列化。"""
        birds = [
            {'idx': 0, 'conf': 0.9, 'bbox': (10, 10, 60, 60),
             'area_ratio': 0.09, 'mask_polygon': None, 'is_selected': True},
            {'idx': 1, 'conf': 0.7, 'bbox': (60, 60, 110, 90),
             'area_ratio': 0.03,
             'mask_polygon': [[60, 60], [110, 60], [110, 90]],
             'is_selected': False},
        ]
        rows = self._run(birds)
        self.assertEqual(json.loads(rows[1]['mask_polygon']),
                         [[120, 120], [220, 120], [220, 180]])

    def test_empty_input(self):
        """空输入返回空列表，不抛异常。"""
        self.assertEqual(self._run([]), [])
        self.assertEqual(self._run(None), [])


class TestBirdBoxDedup(unittest.TestCase):
    """_dedupe_bird_boxes：一鸟多框去重（保留最高置信度框）。"""

    def _dedup(self, boxes, confs, iou_thresh=0.55):
        import numpy as np
        from ai_model import _dedupe_bird_boxes
        dets = np.array(boxes, dtype=np.float64)
        cf = np.array(confs, dtype=np.float64)
        cls = np.array([14] * len(boxes), dtype=np.float64)
        d, c, k, m = _dedupe_bird_boxes(dets, cf, cls, None,
                                        iou_thresh=iou_thresh)
        return d.tolist(), c.tolist()

    def test_duplicate_box_suppressed(self):
        """同一只鸟的两个高重叠框 → 只留置信度高的。"""
        boxes = [[100, 100, 200, 200], [105, 102, 198, 205]]  # IoU≈0.9
        dets, confs = self._dedup(boxes, [0.5, 0.8])
        self.assertEqual(len(dets), 1)
        self.assertEqual(confs[0], 0.8)  # 保留高置信度框

    def test_adjacent_birds_kept(self):
        """相邻但不重叠的两只鸟都保留（IoU 低）。"""
        boxes = [[100, 100, 200, 200], [220, 100, 320, 200]]  # IoU=0
        dets, _ = self._dedup(boxes, [0.5, 0.8])
        self.assertEqual(len(dets), 2)

    def test_partial_overlap_boundary(self):
        """IoU 在阈值附近的框：高于阈值抑制、低于保留。"""
        # 200x200 与右移 100 的框：交 100x200=20000，并 60000 → IoU=0.33
        boxes = [[100, 100, 300, 300], [200, 100, 400, 300]]
        dets, _ = self._dedup(boxes, [0.8, 0.5], iou_thresh=0.3)
        self.assertEqual(len(dets), 1)
        dets, _ = self._dedup(boxes, [0.8, 0.5], iou_thresh=0.5)
        self.assertEqual(len(dets), 2)


class TestRescueMultibirdConfFloor(unittest.TestCase):
    """rescue 带回多鸟列表的 0.2 置信度地板（碎小误检框不带回）。"""

    def test_low_conf_birds_filtered_from_bringback(self):
        """rescue 结果的 detections 数组只含 conf≥0.2 的鸟 + 救回候选。"""
        import numpy as np
        import torch
        import ai_model

        class _FakeBoxes:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
                self.conf = torch.tensor(conf, dtype=torch.float32)
                self.cls = torch.tensor(cls, dtype=torch.float32)

            def __len__(self):
                return len(self.conf)

        class _FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes
                self.masks = None

        class _FakeModel:
            def __init__(self, xyxy, conf, cls):
                self._r = _FakeResult(_FakeBoxes(xyxy, conf, cls))

            def __call__(self, image, **kwargs):
                return [self._r]

        model = _FakeModel(
            [[10, 10, 60, 60],      # 0.42 主候选（直接救回）
             [100, 100, 150, 150],  # 0.25 ≥0.2 保留
             [200, 200, 240, 240]],  # 0.08 <0.2 过滤掉
            [0.42, 0.25, 0.08], [14, 14, 14])
        r = ai_model._rescue_scan(model, np.zeros((683, 1024, 3), np.uint8),
                                  0.3, 10, ".", None)
        self.assertIsNotNone(r)
        confs = sorted(float(c) for c in r["detection_confs"])
        # 0.08 的碎框被过滤，只剩主候选 + 0.25 那只（float32 精度用近似）
        self.assertEqual(len(confs), 2)
        self.assertAlmostEqual(confs[0], 0.25, places=5)
        self.assertAlmostEqual(confs[1], 0.42, places=5)


class TestMainBirdSelection(unittest.TestCase):
    """ai_model._select_main_bird 三级规则（单鸟/polygon/bbox/兜底）。"""

    def _bird(self, idx, bbox, conf=0.5, poly=None):
        return {'idx': idx, 'conf': conf, 'bbox': bbox,
                'mask_polygon': poly, 'is_selected': False}

    def test_single_bird(self):
        from ai_model import _select_main_bird
        idx, reason = _select_main_bird(
            [self._bird(0, (0, 0, 100, 100))], None, 200, 100)
        self.assertEqual((idx, reason), (0, 'single'))

    def test_polygon_hit_beats_bbox(self):
        """对焦点在大框的 bbox 内但在其多边形外 → 跳过它选真正命中的鸟。

        旧 bbox 逻辑会误选鸟0（bbox 包含但点其实在鸟身体外的空白），
        polygon 精度避免了这种误选。
        """
        from ai_model import _select_main_bird
        birds = [
            # 鸟0：bbox 覆盖右下区域，但身体多边形只占上半部
            self._bird(0, (100, 100, 200, 200), conf=0.9,
                       poly=[[110, 110], [190, 110], [190, 150], [110, 150]]),
            # 鸟1：右下角小 bbox（无多边形）
            self._bird(1, (140, 140, 200, 200), conf=0.3, poly=None),
        ]
        # 点 (170,180)：在鸟0的 bbox 内、多边形外（身体下方的空白），
        # 在鸟1的 bbox 内 → 应选鸟1；旧 bbox 逻辑会误选鸟0
        idx, reason = _select_main_bird(birds, (0.85, 0.9), 200, 200)
        self.assertEqual((idx, reason), (1, 'focus'))

    def test_bbox_fallback_when_no_polygon(self):
        """无多边形时 bbox 命中仍有效。"""
        from ai_model import _select_main_bird
        birds = [self._bird(0, (10, 10, 90, 90), conf=0.9, poly=None),
                 self._bird(1, (110, 10, 190, 90), conf=0.3, poly=None)]
        idx, reason = _select_main_bird(birds, (0.6, 0.3), 200, 100)
        self.assertEqual((idx, reason), (1, 'focus'))

    def test_focus_miss_falls_back_to_conf(self):
        """对焦点不在任何鸟上 → fallback + 最高置信度。"""
        from ai_model import _select_main_bird
        birds = [self._bird(0, (10, 10, 60, 60), conf=0.4,
                            poly=[[10, 10], [60, 10], [60, 60], [10, 60]]),
                 self._bird(1, (110, 10, 180, 90), conf=0.8,
                            poly=[[110, 10], [180, 10], [180, 90], [110, 90]])]
        idx, reason = _select_main_bird(birds, (0.5, 0.95), 200, 100)
        self.assertEqual((idx, reason), (1, 'fallback'))

    def test_no_focus_falls_back(self):
        """无对焦点 → fallback。"""
        from ai_model import _select_main_bird
        birds = [self._bird(0, (0, 0, 50, 50), conf=0.9),
                 self._bird(1, (60, 0, 120, 50), conf=0.8)]
        idx, reason = _select_main_bird(birds, None, 200, 100)
        self.assertEqual((idx, reason), (0, 'fallback'))


class TestComprehensiveMainBird(unittest.TestCase):
    """core/multi_bird.select_main_bird 综合重选（稀有优先/大而清晰）。"""

    def test_rare_confident_wins(self):
        """置信的稀有鸟（conf≥70 且 gbif≥50）优先于大而清晰。"""
        from core.multi_bird import select_main_bird
        rows = [
            {'bird_index': 0, 'species_cn': '常见大鸟', 'species_confidence': 99.0,
             'gbif_rarity_100': 5.0, 'crop_sharpness': 900.0, 'area_ratio': 0.3},
            {'bird_index': 1, 'species_cn': '稀有小鸟', 'species_confidence': 75.0,
             'gbif_rarity_100': 80.0, 'crop_sharpness': 100.0, 'area_ratio': 0.01},
        ]
        self.assertEqual(select_main_bird(rows), 1)

    def test_rare_requires_confidence(self):
        """稀有但置信不足（<70）不算「置信的稀有鸟」→ 走评分。"""
        from core.multi_bird import select_main_bird
        rows = [
            {'bird_index': 0, 'species_cn': '常见大鸟', 'species_confidence': 99.0,
             'gbif_rarity_100': 5.0, 'crop_sharpness': 900.0, 'area_ratio': 0.3},
            {'bird_index': 1, 'species_cn': '稀有但存疑', 'species_confidence': 40.0,
             'gbif_rarity_100': 80.0, 'crop_sharpness': 100.0, 'area_ratio': 0.01},
        ]
        self.assertEqual(select_main_bird(rows), 0)

    def test_score_prefers_big_sharp(self):
        """无稀有鸟时：锐度+面积+置信度加权，大而清晰者胜。"""
        from core.multi_bird import select_main_bird
        rows = [
            {'bird_index': 0, 'species_cn': '小糊鸟', 'species_confidence': 90.0,
             'gbif_rarity_100': 3.0, 'crop_sharpness': 80.0, 'area_ratio': 0.005},
            {'bird_index': 1, 'species_cn': '大清晰鸟', 'species_confidence': 85.0,
             'gbif_rarity_100': 10.0, 'crop_sharpness': 700.0, 'area_ratio': 0.20},
        ]
        self.assertEqual(select_main_bird(rows), 1)

    def test_unclassified_uses_yolo_conf(self):
        """未分类的鸟用 YOLO 置信度参与评分。"""
        from core.multi_bird import select_main_bird
        rows = [
            {'bird_index': 0, 'species_cn': None, 'species_confidence': None,
             'gbif_rarity_100': None, 'crop_sharpness': 400.0,
             'area_ratio': 0.05, 'yolo_conf': 0.9},
            {'bird_index': 1, 'species_cn': '低分鸟', 'species_confidence': 36.0,
             'gbif_rarity_100': 2.0, 'crop_sharpness': 200.0,
             'area_ratio': 0.05, 'yolo_conf': 0.4},
        ]
        self.assertEqual(select_main_bird(rows), 0)

    def test_deleted_excluded(self):
        """软删除的鸟不参与重选。"""
        from core.multi_bird import select_main_bird
        rows = [
            {'bird_index': 0, 'deleted': True, 'species_cn': '被删的稀有鸟',
             'species_confidence': 99.0, 'gbif_rarity_100': 90.0,
             'crop_sharpness': 900.0, 'area_ratio': 0.3},
            {'bird_index': 1, 'species_cn': '普通鸟', 'species_confidence': 80.0,
             'gbif_rarity_100': 5.0, 'crop_sharpness': 500.0, 'area_ratio': 0.1},
        ]
        self.assertEqual(select_main_bird(rows), 1)


class TestSpeciesRecall(unittest.TestCase):
    """core/species_recall.run_species_recall 批内物种召回。"""

    def _setup_db(self, d):
        """三张照片：A 主鸟=泽鹬、次要=反嘴鹬；B 主鸟=泽鹬；C 无主鸟种、次要=黑腹滨鹬。"""
        from tools.report_db import ReportDB
        db = ReportDB(d)
        db.insert_photo({'filename': 'A', 'has_bird': 1, 'rating': 3,
                         'bird_species_cn': '泽鹬'})
        db.insert_photo({'filename': 'B', 'has_bird': 1, 'rating': 2,
                         'bird_species_cn': '泽鹬'})
        db.insert_photo({'filename': 'C', 'has_bird': 1, 'rating': 1})
        db.insert_detections_batch([
            {'filename': 'A', 'bird_index': 0, 'is_selected': 1,
             'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 10, 'bbox_h': 10,
             'species_cn': '泽鹬', 'species_en': 'Common Greenshank',
             'scientific_name': 'Tringa nebularia',
             'species_confidence': 95.0},
            {'filename': 'A', 'bird_index': 1, 'is_selected': 0,
             'bbox_x': 50, 'bbox_y': 50, 'bbox_w': 10, 'bbox_h': 10,
             'species_cn': '反嘴鹬', 'species_en': 'Pied Avocet',
             'scientific_name': 'Recurvirostra avosetta',
             'species_confidence': 80.0},
        ])
        db.insert_detections_batch([
            # 低置信的反嘴鹬（C 照片）：不该触发召回
            {'filename': 'C', 'bird_index': 0, 'is_selected': 0,
             'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 10, 'bbox_h': 10,
             'species_cn': '反嘴鹬', 'species_confidence': 20.0},
            {'filename': 'C', 'bird_index': 1, 'is_selected': 0,
             'bbox_x': 20, 'bbox_y': 20, 'bbox_w': 10, 'bbox_h': 10,
             'species_cn': '黑腹滨鹬', 'species_confidence': 60.0},
        ])
        return db

    def test_recall_flags_never_main_species(self):
        """从未当主鸟的鸟种（反嘴鹬[高置信]、黑腹滨鹬）→ 照片打标。"""
        import os, shutil, tempfile
        from core.species_recall import run_species_recall
        d = tempfile.mkdtemp()
        try:
            db = self._setup_db(d)
            logs = []
            stats = run_species_recall(db, species_threshold=35.0,
                                       log=logs.append)
            # 泽鹬当过主鸟（A/B）→ 不召回；反嘴鹬(80%)与黑腹滨鹬(60%)召回
            self.assertEqual(sorted(stats['never_main_species']),
                             ['反嘴鹬', '黑腹滨鹬'])
            self.assertEqual(stats['flagged_photos'], 2)      # A 和 C
            self.assertEqual(stats['flagged_detections'], 2)  # A#1 与 C#1
            # photos.notable 落库 + notable_only 筛选
            flagged = db.get_photos_by_filters({'notable_only': True})
            self.assertEqual(sorted(p['filename'] for p in flagged),
                             ['A', 'C'])
            # detection 级标记与原因
            rows = db.get_detections('A')
            self.assertEqual(rows[1]['notable'], 1)
            self.assertEqual(rows[1]['notable_reason'],
                             'never_main_species')
            self.assertIn(rows[0]['notable'], (None, 0))
            # 低置信的 C#0 反嘴鹬未标记
            rows_c = db.get_detections('C')
            self.assertIn(rows_c[0]['notable'], (None, 0))
            self.assertEqual(rows_c[1]['notable'], 1)
            db._conn.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_recall_noop_when_all_species_main(self):
        """所有鸟种都当过主鸟 → 无标记。"""
        import os, shutil, tempfile
        from core.species_recall import run_species_recall
        from tools.report_db import ReportDB
        d = tempfile.mkdtemp()
        try:
            db = ReportDB(d)
            db.insert_photo({'filename': 'A', 'has_bird': 1,
                             'bird_species_cn': '泽鹬'})
            db.insert_detections_batch([
                {'filename': 'A', 'bird_index': 0, 'is_selected': 1,
                 'species_cn': '泽鹬', 'species_confidence': 90.0},
                {'filename': 'A', 'bird_index': 1, 'is_selected': 0,
                 'species_cn': '泽鹬', 'species_confidence': 85.0},
            ])
            stats = run_species_recall(db, species_threshold=35.0,
                                       log=lambda *_: None)
            self.assertEqual(stats['flagged_photos'], 0)
            db._conn.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
