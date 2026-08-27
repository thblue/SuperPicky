#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky ReportDB - SQLite 报告数据库封装
替代原有的 CSV 报告存储，提供更高效的查询和更新操作。

Usage:
    db = ReportDB("/path/to/photos")
    db.insert_photo({"filename": "IMG_1234", "has_bird": 1, ...})
    photo = db.get_photo("IMG_1234")
    db.close()
"""

import os
import sqlite3
import time
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from .file_utils import ensure_hidden_directory


# Schema 版本，用于未来升级
SCHEMA_VERSION = "12"

# 所有列定义（有序），用于 CREATE TABLE 和数据验证
PHOTO_COLUMNS = [
    # (列名, SQLite 类型, 默认值)
    ("filename",      "TEXT NOT NULL UNIQUE", None),
    ("has_bird",      "INTEGER", 0),          # 0=no, 1=yes
    ("confidence",    "REAL", 0.0),
    ("head_sharp",    "REAL", None),
    ("left_eye",      "REAL", None),
    ("right_eye",     "REAL", None),
    ("beak",          "REAL", None),
    ("nima_score",    "REAL", None),
    ("is_flying",     "INTEGER", 0),          # 0=no, 1=yes
    ("flight_conf",   "REAL", None),
    ("rating",        "INTEGER", 0),          # -1/0/1/2/3
    ("picked",        "INTEGER", 0),          # 精选旗标:选鸟时 3★ 中美学∩锐度 top% 的交集(0/1)
    ("notable",       "INTEGER", 0),          # V5.2 物种召回旗标:含本批从未当主鸟的鸟种(0/1)
    ("focus_status",  "TEXT", None),           # BEST/GOOD/BAD/WORST
    ("focus_x",       "REAL", None),
    ("focus_y",       "REAL", None),
    ("adj_sharpness", "REAL", None),
    ("adj_topiq",     "REAL", None),
    
    # V2: 相机设置
    ("iso",              "INTEGER", None),
    ("shutter_speed",    "TEXT", None),
    ("aperture",         "TEXT", None),
    ("focal_length",     "REAL", None),
    ("focal_length_35mm","INTEGER", None),
    ("camera_model",     "TEXT", None),
    ("lens_model",       "TEXT", None),
    
    # V2: GPS
    ("gps_latitude",     "REAL", None),
    ("gps_longitude",    "REAL", None),
    ("gps_altitude",     "REAL", None),
    
    # V2: IPTC 元数据
    ("title",            "TEXT", None),
    ("caption",          "TEXT", None),
    ("city",             "TEXT", None),
    ("state_province",   "TEXT", None),
    ("country",          "TEXT", None),
    
    # V2: 时间
    ("date_time_original", "TEXT", None),
    
    # V2: 鸟种识别
    ("bird_species_cn",  "TEXT", None),
    ("bird_species_en",  "TEXT", None),
    ("birdid_confidence","REAL", None),
    
    # V2: 曝光状态
    ("exposure_status",  "TEXT", None),
    
    # V3: 文件路径（相对路径）
    ("original_path",    "TEXT", None),
    ("current_path",     "TEXT", None),
    ("temp_jpeg_path",   "TEXT", None),
    ("debug_crop_path",  "TEXT", None),   # 裁切鸟+mask (crop_debug/)
    ("yolo_debug_path",  "TEXT", None),   # 全图+YOLO框 (yolo_debug/)
    
    # V5: 连拍分组
    ("burst_id",         "INTEGER", None),
    ("burst_position",   "INTEGER", None),

    # V6: 懂鸟罕见指数 (0-10，越大越罕见)
    # V6: BirdID rarity index (0-10, higher = rarer)
    ("rarity_index",     "REAL", None),

    # V7: IUCN 红色名录保护级别 (LC/NT/VU/EN/CR/CR(PE)/CR(PEW)/EW/EX/DD/NE)
    # V7: IUCN Red List category
    ("iucn_category",    "TEXT", None),

    # V8: GBIF 全球罕见度 (0-100 分制，越大越罕见，CC0+CC-BY 4.0 子集派生)
    # V8: GBIF-derived global rarity score (0-100, higher = rarer)
    ("gbif_rarity_100",  "REAL", None),

    # V9: iRateBird 鸟种美学(颜值)指数 (0-100，越大越好看，CC-BY 4.0 派生)
    # V9: iRateBird species aesthetic score (0-100, higher = prettier)
    ("aesthetic_index",  "REAL", None),

    ("created_at",    "TEXT", None),
    ("updated_at",    "TEXT", None),
]

# 列名集合，用于快速查找
COLUMN_NAMES = {col[0] for col in PHOTO_COLUMNS}

# bird_detections 表的全部业务列（不含 id/created_at/updated_at）
# All business columns of bird_detections (id/timestamps excluded).
DETECTION_COLUMNS = (
    "filename", "bird_index", "is_selected",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "mask_polygon", "area_ratio", "yolo_conf", "crop_sharpness",
    "species_cn", "species_en", "scientific_name",
    "species_confidence", "class_id", "gbif_rarity_100",
    "notable", "notable_reason", "edited",
    # V5.4 人工软删除标记（0=正常 1=已删框；删除只隐藏，不物理删行）
    "deleted",
)


class ReportDB:
    """SQLite 报告数据库封装。

    每个照片处理目录拥有一个独立的数据库文件：
        <directory>/.superpicky/report.db

    线程安全：设置 check_same_thread=False，支持工作线程写入。
    WAL 模式：支持读写并发。
    """

    DB_FILENAME = "report.db"

    def __init__(self, directory: str):
        """
        初始化数据库连接。

        Args:
            directory: 照片目录路径（数据库存储在 .superpicky/ 子目录下）
        """
        self.directory = directory
        self._superpicky_dir = os.path.join(directory, ".superpicky")
        self.db_path = os.path.join(self._superpicky_dir, self.DB_FILENAME)
        # 同一连接会被主线程和后台线程复用，需要串行化访问避免事务冲突
        self._lock = threading.RLock()

        # 确保 .superpicky 目录存在并隐藏（Windows 下设置 Hidden 属性）
        ensure_hidden_directory(self._superpicky_dir)

        # 连接数据库
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row  # 支持按列名访问

        # 启用 WAL 模式和外键
        self._conn.execute("PRAGMA journal_mode=WAL")
        # WAL 搭配 NORMAL:每次 commit 不再单独 fsync(默认 FULL 每 commit 一次),
        # 仅 checkpoint 时同步。主处理循环每张照片 2-6 次 commit,在 SD 卡/ExFAT/
        # HDD 上每次 fsync 10-30ms;NORMAL 断电最多丢最后一批事务,库不会损坏。
        # WAL + NORMAL: commits no longer fsync individually (default FULL
        # syncs every commit); only checkpoints do. The main loop commits 2-6
        # times per photo, and on SD/ExFAT/HDD each fsync costs 10-30ms.
        # NORMAL may lose the last batch on power loss but never corrupts.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # 初始化 Schema
        self._init_schema()

    def _init_schema(self):
        """创建表和索引（如果不存在）。"""
        # 构建 CREATE TABLE 语句
        col_defs = []
        for name, type_def, _ in PHOTO_COLUMNS:
            col_defs.append(f"    {name} {type_def}")

        create_sql = (
            "CREATE TABLE IF NOT EXISTS photos (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            + ",\n".join(col_defs)
            + "\n)"
        )

        with self._lock:
            with self._conn:
                self._conn.execute(create_sql)

                # 索引
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_photos_filename "
                    "ON photos(filename)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_photos_rating "
                    "ON photos(rating)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_photos_has_bird "
                    "ON photos(has_bird)"
                )

                # 元数据表
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)

                # 纠错样本表（correction submission）：随项目持久化。
                # 记录一次「改鸟种」事件的原预测(wrong_*)与改正结果(corrected_*)。
                # Corrections table for the correction-submission feature.
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        wrong_cn TEXT,
                        wrong_en TEXT,
                        corrected_model_class_id INTEGER,
                        corrected_cn TEXT,
                        corrected_en TEXT,
                        birdid_confidence REAL,
                        created_at TEXT
                    )
                """)
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_corrections_filename "
                    "ON corrections(filename)"
                )

                # 多鸟检测表（multi-bird detection）：
                # 每张照片的每个检测框一行，记录逐鸟分类结果。
                # filename 与 photos.filename 同键（文件名前缀，无扩展名）。
                # is_selected=1 的行是主鸟（现有评分链路的对象），其余为次要鸟。
                # mask_polygon 为简化轮廓 [[x,y],...]（JSON 字符串，原图坐标）。
                # Multi-bird detections table: one row per detected bird box.
                # filename matches photos.filename (prefix without extension).
                # is_selected=1 marks the main bird used by the rating pipeline.
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS bird_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        bird_index INTEGER NOT NULL,
                        is_selected INTEGER DEFAULT 0,
                        bbox_x REAL,
                        bbox_y REAL,
                        bbox_w REAL,
                        bbox_h REAL,
                        mask_polygon TEXT,
                        area_ratio REAL,
                        yolo_conf REAL,
                        crop_sharpness REAL,
                        species_cn TEXT,
                        species_en TEXT,
                        scientific_name TEXT,
                        species_confidence REAL,
                        class_id INTEGER,
                        gbif_rarity_100 REAL,
                        notable INTEGER DEFAULT 0,
                        notable_reason TEXT,
                        edited INTEGER DEFAULT 0,
                        deleted INTEGER DEFAULT 0,
                        created_at TEXT,
                        updated_at TEXT
                    )
                """)
                self._conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_bird_detections_filename_idx "
                    "ON bird_detections(filename, bird_index)"
                )

                # 初始化元数据
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                    ("schema_version", SCHEMA_VERSION)
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                    ("directory_path", self.directory)
                )

        # Schema 升级在独立事务中执行，避免嵌套 commit 冲突
        self._upgrade_schema_if_needed()
    
    def _upgrade_schema_if_needed(self):
        """检查并升级数据库 Schema（支持连续升级 v1 -> v2 -> v3 -> v4）"""
        with self._lock:
            # 获取当前 schema 版本
            cursor = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
            current_version = row[0] if row else "1"

            # ----------------------------------------------------------------------
            #  Upgrade: v1 -> v2 (EXIF metadata)
            # ----------------------------------------------------------------------
            if current_version == "1":
                print("🔄 Upgrading database schema from v1 to v2...")
                new_columns = [
                    ("iso", "INTEGER"),
                    ("shutter_speed", "TEXT"),
                    ("aperture", "TEXT"),
                    ("focal_length", "REAL"),
                    ("focal_length_35mm", "INTEGER"),
                    ("camera_model", "TEXT"),
                    ("lens_model", "TEXT"),
                    ("gps_latitude", "REAL"),
                    ("gps_longitude", "REAL"),
                    ("gps_altitude", "REAL"),
                    ("title", "TEXT"),
                    ("caption", "TEXT"),
                    ("city", "TEXT"),
                    ("state_province", "TEXT"),
                    ("country", "TEXT"),
                    ("date_time_original", "TEXT"),
                    ("bird_species_cn", "TEXT"),
                    ("bird_species_en", "TEXT"),
                    ("birdid_confidence", "REAL"),
                    ("exposure_status", "TEXT"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("2")
                current_version = "2"
                print("✅ Database schema upgraded to v2")

            # ----------------------------------------------------------------------
            #  Upgrade: v2 -> v3 (File paths)
            # ----------------------------------------------------------------------
            if current_version == "2":
                print("🔄 Upgrading database schema from v2 to v3...")
                new_columns_v3 = [
                    ("original_path", "TEXT"),
                    ("current_path", "TEXT"),
                    ("temp_jpeg_path", "TEXT"),
                    ("debug_crop_path", "TEXT"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v3:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("3")
                current_version = "3"
                print("✅ Database schema upgraded to v3")

            # ----------------------------------------------------------------------
            #  Upgrade: v3 -> v4 (Check debug images)
            # ----------------------------------------------------------------------
            if current_version == "3":
                print("🔄 Upgrading database schema from v3 to v4...")
                new_columns_v4 = [
                    ("yolo_debug_path", "TEXT"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v4:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("4")
                current_version = "4"
                print("✅ Database schema upgraded to v4")

            # ----------------------------------------------------------------------
            #  Upgrade: v4 -> v5 (Burst id and position)
            # ----------------------------------------------------------------------
            if current_version == "4":
                print("🔄 Upgrading database schema from v4 to v5...")
                new_columns_v5 = [
                    ("burst_id", "INTEGER"),
                    ("burst_position", "INTEGER"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v5:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("5")
                current_version = "5"
                print("✅ Database schema upgraded to v5")

            # ----------------------------------------------------------------------
            #  Upgrade: v5 -> v6 (BirdID rarity index)
            # ----------------------------------------------------------------------
            if current_version == "5":
                print("🔄 Upgrading database schema from v5 to v6...")
                new_columns_v6 = [
                    ("rarity_index", "REAL"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v6:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("6")
                current_version = "6"
                print("✅ Database schema upgraded to v6")

            # ----------------------------------------------------------------------
            #  Upgrade: v6 -> v7 (IUCN Red List category)
            # ----------------------------------------------------------------------
            if current_version == "6":
                print("🔄 Upgrading database schema from v6 to v7...")
                new_columns_v7 = [
                    ("iucn_category", "TEXT"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v7:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("7")
                current_version = "7"
                print("✅ Database schema upgraded to v7")

            # ----------------------------------------------------------------------
            #  Upgrade: v7 -> v8 (GBIF 0-100 rarity score)
            # ----------------------------------------------------------------------
            if current_version == "7":
                print("🔄 Upgrading database schema from v7 to v8...")
                new_columns_v8 = [
                    ("gbif_rarity_100", "REAL"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v8:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("8")
                current_version = "8"
                print("✅ Database schema upgraded to v8")

            # ----------------------------------------------------------------------
            #  Upgrade: v8 -> v9 (iRateBird species aesthetic index)
            # ----------------------------------------------------------------------
            if current_version == "8":
                print("🔄 Upgrading database schema from v8 to v9...")
                new_columns_v9 = [
                    ("aesthetic_index", "REAL"),
                ]
                with self._conn:
                    for col_name, col_type in new_columns_v9:
                        try:
                            self._conn.execute(
                                f"ALTER TABLE photos ADD COLUMN {col_name} {col_type}"
                            )
                        except sqlite3.OperationalError:
                            pass  # 列已存在，跳过
                    self._update_schema_version("9")
                current_version = "9"
                print("✅ Database schema upgraded to v9")

            # ----------------------------------------------------------------------
            #  Upgrade: v9 -> v10 (Multi-bird detections table)
            #  新增 bird_detections 表：每照片多鸟逐鸟分类结果。
            #  纯新增表，photos 表无变化；CREATE IF NOT EXISTS 对新库幂等。
            #  Adds bird_detections table for per-bird classification results.
            #  Additive only; no photos-table change.
            # ----------------------------------------------------------------------
            if current_version == "9":
                print("🔄 Upgrading database schema from v9 to v10...")
                with self._conn:
                    self._conn.execute("""
                        CREATE TABLE IF NOT EXISTS bird_detections (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filename TEXT NOT NULL,
                            bird_index INTEGER NOT NULL,
                            is_selected INTEGER DEFAULT 0,
                            bbox_x REAL,
                            bbox_y REAL,
                            bbox_w REAL,
                            bbox_h REAL,
                            mask_polygon TEXT,
                            area_ratio REAL,
                            yolo_conf REAL,
                            crop_sharpness REAL,
                            species_cn TEXT,
                            species_en TEXT,
                            scientific_name TEXT,
                            species_confidence REAL,
                            class_id INTEGER,
                            gbif_rarity_100 REAL,
                            notable INTEGER DEFAULT 0,
                            notable_reason TEXT,
                            edited INTEGER DEFAULT 0,
                            created_at TEXT,
                            updated_at TEXT
                        )
                    """)
                    self._conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "idx_bird_detections_filename_idx "
                        "ON bird_detections(filename, bird_index)"
                    )
                    self._update_schema_version("10")
                current_version = "10"
                print("✅ Database schema upgraded to v10")

            # ----------------------------------------------------------------------
            #  Upgrade: v10 -> v11 (Species recall flag on photos)
            #  V5.2 物种召回：photos 加 notable 列（批内从未当主鸟的鸟种标记）。
            #  纯加列，bird_detections 的 notable 列 v10 已有。
            #  Adds photos.notable for the species-recall feature.
            # ----------------------------------------------------------------------
            if current_version == "10":
                print("🔄 Upgrading database schema from v10 to v11...")
                with self._conn:
                    try:
                        self._conn.execute(
                            "ALTER TABLE photos ADD COLUMN notable INTEGER"
                        )
                    except sqlite3.OperationalError:
                        pass  # 列已存在，跳过
                    self._update_schema_version("11")
                current_version = "11"
                print("✅ Database schema upgraded to v11")

            # ----------------------------------------------------------------------
            #  Upgrade: v11 -> v12 (Manual soft-delete flag on detections)
            #  V5.4 多鸟编辑/批量清理：bird_detections 加 deleted 列
            #  （0=正常 1=人工软删除）。软删除只隐藏框（筛选/召回/编辑器/
            #  sidecar 导出均跳过），不物理删行，误删可手工恢复。
            #  Adds bird_detections.deleted for manual soft-deletes.
            # ----------------------------------------------------------------------
            if current_version == "11":
                print("🔄 Upgrading database schema from v11 to v12...")
                with self._conn:
                    try:
                        self._conn.execute(
                            "ALTER TABLE bird_detections ADD COLUMN "
                            "deleted INTEGER DEFAULT 0"
                        )
                    except sqlite3.OperationalError:
                        pass  # 列已存在，跳过
                    self._update_schema_version("12")
                current_version = "12"
                print("✅ Database schema upgraded to v12")

    def _update_schema_version(self, version):
        """更新数据库中的版本号（由调用方负责提交事务）"""
        with self._lock:
            self._conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                (version,)
            )

    # ==========================================================================
    #  写入操作
    # ==========================================================================

    def insert_photo(self, data: dict) -> None:
        """
        插入或更新一条照片记录。

        如果 filename 已存在则更新，否则插入新记录。
        自动处理 CSV 兼容的数据格式转换（如 "yes"/"no" → 1/0）。

        Args:
            data: 照片数据字典，键为列名
        """
        cleaned = self._clean_data(data)
        now = _now_iso()
        cleaned.setdefault("created_at", now)
        cleaned["updated_at"] = now

        # 仅保留合法列
        columns = [k for k in cleaned if k in COLUMN_NAMES]
        values = [cleaned[k] for k in columns]

        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)

        # INSERT OR REPLACE
        update_clause = ", ".join(
            f"{c} = excluded.{c}" for c in columns if c != "filename"
        )

        sql = (
            f"INSERT INTO photos ({col_str}) VALUES ({placeholders}) "
            f"ON CONFLICT(filename) DO UPDATE SET {update_clause}"
        )

        with self._lock:
            self._conn.execute(sql, values)
            self._safe_commit()

    def insert_photos_batch(self, photos: List[dict]) -> int:
        """
        批量插入或更新照片记录。

        使用事务包裹，性能优于逐条插入。

        Args:
            photos: 照片数据字典列表

        Returns:
            成功插入/更新的记录数
        """
        if not photos:
            return 0

        now = _now_iso()
        count = 0

        with self._lock:
            with self._conn:
                for data in photos:
                    cleaned = self._clean_data(data)
                    cleaned.setdefault("created_at", now)
                    cleaned["updated_at"] = now

                    columns = [k for k in cleaned if k in COLUMN_NAMES]
                    values = [cleaned[k] for k in columns]

                    placeholders = ", ".join(["?"] * len(columns))
                    col_str = ", ".join(columns)

                    update_clause = ", ".join(
                        f"{c} = excluded.{c}" for c in columns if c != "filename"
                    )

                    sql = (
                        f"INSERT INTO photos ({col_str}) VALUES ({placeholders}) "
                        f"ON CONFLICT(filename) DO UPDATE SET {update_clause}"
                    )

                    self._conn.execute(sql, values)
                    count += 1

        return count

    # ==========================================================================
    #  查询操作
    # ==========================================================================

    def get_photo(self, filename: str) -> Optional[dict]:
        """
        按 filename 查询单条记录。

        Args:
            filename: 照片文件名（不含扩展名）

        Returns:
            照片数据字典，未找到返回 None
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM photos WHERE filename = ?", (filename,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_photos(self) -> List[dict]:
        """
        获取所有照片记录。

        Returns:
            照片数据字典列表
        """
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM photos ORDER BY filename")
            return [dict(row) for row in cursor.fetchall()]

    def get_bird_photos(self) -> List[dict]:
        """
        获取所有有鸟的照片记录（has_bird=1）。

        Returns:
            有鸟照片数据字典列表
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM photos WHERE has_bird = 1 ORDER BY filename"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_photos_by_rating(self, rating: int) -> List[dict]:
        """
        按评分查询照片。

        Args:
            rating: 评分 (-1/0/1/2/3)

        Returns:
            照片数据字典列表
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM photos WHERE rating = ? ORDER BY filename",
                (rating,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_photos_by_burst_id(self, burst_id: int, **kwargs) -> List[dict]:
        """
        按 burst_id 查询同组所有连拍照片。
        **kwargs 仅用于与 MergedReportDB 签名兼容，在此忽略。

        Args:
            burst_id: 连拍组 ID

        Returns:
            该组所有照片的数据字典列表，按 burst_position / filename 排序
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM photos WHERE burst_id = ? ORDER BY burst_position, filename",
                (burst_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_distinct_species(self, use_en: bool = False, ratings: list = None) -> List[str]:
        """
        获取数据库中去重后的鸟种名称列表（用于结果浏览器筛选下拉框）。

        V5.4 起一张照片可有多个人工勾选的主鸟种（bird_detections
        is_selected=1 且未软删），这些鸟种也并入下拉列表——多主鸟照片
        在每个主鸟种下都能被筛出。photos 表单字段仍是第一主鸟。

        Args:
            use_en: True 使用英文鸟种列，False 使用中文鸟种列
            ratings: 若提供，只返回在这些星级下有照片的鸟种

        Returns:
            鸟种名称列表（已去重、去空值）

        Distinct species names for the filter dropdown. Since V5.4 a
        photo may carry multiple human-selected main species
        (bird_detections is_selected=1, not soft-deleted); those names
        are UNIONed in so multi-main photos surface under each of them.
        """
        column = "bird_species_en" if use_en else "bird_species_cn"
        det_column = "species_en" if use_en else "species_cn"

        # 星级约束两侧共用；列非空约束各用各的列
        # Shared rating gate; each side keeps its own non-empty checks.
        common: List[str] = ["rating != -1"]
        params: List[Any] = []
        det_params: List[Any] = []

        if isinstance(ratings, list):
            valid = [r for r in ratings if r != -1]
            if valid:
                placeholders = ", ".join(["?"] * len(valid))
                common.append(f"rating IN ({placeholders})")
                params.extend(valid)
                det_params.extend(valid)

        common_sql = " AND ".join(common)
        sql = (
            f"SELECT {column} AS name FROM photos "
            f"WHERE {common_sql} AND {column} IS NOT NULL "
            f"AND TRIM({column}) != '' "
            f"UNION "
            f"SELECT d.{det_column} AS name FROM bird_detections d "
            f"JOIN photos p ON p.filename = d.filename "
            f"WHERE d.is_selected = 1 AND d.deleted = 0 "
            f"AND d.{det_column} IS NOT NULL "
            f"AND TRIM(d.{det_column}) != '' "
            f"AND p.{common_sql} "
            f"ORDER BY name COLLATE NOCASE"
        )
        # photos 侧占位符与检测侧 JOIN 的星级占位符各一份
        # Placeholders appear twice: photos side + JOINed detection side.
        with self._lock:
            cursor = self._conn.execute(sql, params + det_params)
            return [row[0] for row in cursor.fetchall()]

    def get_photos_by_filters(self, filters: Optional[dict] = None) -> List[dict]:
        """
        按结果浏览器筛选条件查询照片。

        支持的 filters 键：
            - ratings: List[int]
            - focus_statuses: List[str]
            - is_flying: List[int]
            - bird_species_cn / bird_species_en: str
            - sort_by: filename | sharpness_desc | aesthetic_desc
            - picked_only: bool (如果为 True，则在结果集中筛选出 adj_topiq 和 adj_sharpness 均排名前 25% 的照片)
        """
        filters = filters or {}

        where_clauses = []
        params: List[Any] = []

        ratings = filters.get("ratings")

        if isinstance(ratings, list):
            if not ratings:
                return []
            placeholders = ", ".join(["?"] * len(ratings))
            where_clauses.append(f"rating IN ({placeholders})")
            params.extend(ratings)

        # 是否包含低评分（0 星），这类照片 focus_status/is_flying 可能是 NULL
        has_low_rating = ratings is None or (isinstance(ratings, list) and any(r <= 0 for r in ratings))

        focus_statuses = filters.get("focus_statuses")
        if isinstance(focus_statuses, list):
            if not focus_statuses:
                return []
            placeholders = ", ".join(["?"] * len(focus_statuses))
            condition = f"focus_status IN ({placeholders})"
            if has_low_rating:
                condition = f"({condition} OR focus_status IS NULL)"
            where_clauses.append(condition)
            params.extend(focus_statuses)

        is_flying = filters.get("is_flying")
        if isinstance(is_flying, list):
            if not is_flying:
                return []
            placeholders = ", ".join(["?"] * len(is_flying))
            condition = f"is_flying IN ({placeholders})"
            if has_low_rating:
                condition = f"({condition} OR is_flying IS NULL)"
            where_clauses.append(condition)
            params.extend(is_flying)

        species_col = None
        species_val = None
        if "bird_species_en" in filters:
            species_col = "bird_species_en"
            species_val = filters.get("bird_species_en")
        elif "bird_species_cn" in filters:
            species_col = "bird_species_cn"
            species_val = filters.get("bird_species_cn")

        if isinstance(species_val, str) and species_val.strip():
            assert species_col in {"bird_species_en", "bird_species_cn"}, f"Invalid column: {species_col}"
            # V5.4 多主鸟：photos 单字段（第一主鸟）之外，凡有该鸟种的
            # is_selected 检测行的照片也算命中——多主鸟照片在每个主鸟种
            # 下都能被筛出（星级等其余条件照常 AND 组合）。
            # Multi-main photos match via their is_selected detections as
            # well as the single photos-table species field.
            det_col = ("species_en" if species_col == "bird_species_en"
                       else "species_cn")
            name = species_val.strip()
            where_clauses.append(
                f"({species_col} = ? OR filename IN ("
                f"SELECT filename FROM bird_detections "
                f"WHERE is_selected = 1 AND deleted = 0 "
                f"AND {det_col} = ?))"
            )
            params.append(name)
            params.append(name)

        # 精选(picked):直接用选鸟时写入的持久旗标列(3★ 中美学∩锐度 top% 的交集)。
        # 旧目录(未重跑选鸟)该列全为 0,需重跑后才有结果。
        if filters.get("picked_only", False):
            where_clauses.append("picked = 1")

        # V5.2 物种召回筛选：只看含「本批从未当主鸟」鸟种的照片
        if filters.get("notable_only", False):
            where_clauses.append("notable = 1")

        # V5.4 待确认鸟种定位：只看含指定召回鸟种（notable=1 未删检测）
        # 的照片——召回清单点击某鸟种时的网格过滤
        notable_sp = filters.get("notable_species")
        if isinstance(notable_sp, str) and notable_sp.strip():
            where_clauses.append(
                "filename IN (SELECT filename FROM bird_detections "
                "WHERE notable = 1 AND deleted = 0 AND species_cn = ?)")
            params.append(notable_sp.strip())

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Determine sort order
        sort_by = filters.get("sort_by") or "filename"
        if sort_by == "sharpness_desc":
            order_sql = "ORDER BY COALESCE(adj_sharpness, head_sharp, -1e99) DESC, filename ASC"
        elif sort_by == "aesthetic_desc":
            order_sql = "ORDER BY COALESCE(adj_topiq, nima_score, -1e99) DESC, filename ASC"
        elif sort_by == "rarity_desc":
            # V4.2.7: 按 GBIF 罕见度降序（最罕见在前）— 无 GBIF 数据的排最后
            order_sql = "ORDER BY COALESCE(gbif_rarity_100, -1e99) DESC, filename ASC"
        elif sort_by == "species_beauty_desc":
            # V9: 按鸟种颜值(iRateBird)降序 — 无数据排最后
            # V9: sort by species beauty (iRateBird) desc — missing data last
            order_sql = "ORDER BY COALESCE(aesthetic_index, -1e99) DESC, filename ASC"
        elif sort_by == "capture_time":
            # 按拍摄时间升序（= 原始拍摄顺序）：无拍摄时间的排最后；
            # 同秒按文件名 tiebreak —— 相机计数序号即真实先后。
            # Sort by capture time asc (= original shooting order); photos
            # without EXIF time go last; same-second tiebreak by filename
            # (camera counter names reflect the real order).
            order_sql = ("ORDER BY (date_time_original IS NULL) ASC, "
                         "date_time_original ASC, filename ASC")
        else:
            order_sql = "ORDER BY filename ASC"

        sql = f"SELECT * FROM photos {where_sql} {order_sql}"

        with self._lock:
            cursor = self._conn.execute(sql, params)
            results = [dict(row) for row in cursor.fetchall()]

        return results

    # ==========================================================================
    #  纠错样本（correction submission）
    # ==========================================================================

    def insert_correction(self, data: dict) -> None:
        """
        插入一条纠错记录（每次「改鸟种」事件一条）。

        参数 / Args:
            data: 键含 filename, wrong_cn, wrong_en, corrected_model_class_id,
                  corrected_cn, corrected_en, birdid_confidence。created_at 自动填。
        """
        cols = ("filename", "wrong_cn", "wrong_en", "corrected_model_class_id",
                "corrected_cn", "corrected_en", "birdid_confidence", "created_at")
        row = {k: data.get(k) for k in cols}
        row["created_at"] = _now_iso()
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        sql = f"INSERT INTO corrections ({col_str}) VALUES ({placeholders})"
        with self._lock:
            self._conn.execute(sql, [row[c] for c in cols])
            self._safe_commit()

    def get_corrections(self) -> List[dict]:
        """返回全部纠错记录，按 created_at 升序。"""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM corrections ORDER BY created_at, id"
            )
            return [dict(r) for r in cursor.fetchall()]

    # ==========================================================================
    #  多鸟检测（multi-bird detections）
    # ==========================================================================

    def insert_detections_batch(self, rows: List[dict]) -> int:
        """
        整体替换一张照片的多鸟检测记录（先删后插，幂等，支持重跑）。

        参数:
        rows (List[dict]): 每鸟一行的数据字典，键为 DETECTION_COLUMNS
            中的列名；filename 必填。缺失列写 NULL。

        返回:
        int: 实际写入的行数

        Replace all detections of a photo (delete-then-insert, idempotent).
        Rows must share the same filename; missing keys become NULL.

        Raises:
            ValueError: rows 为空或 filename 缺失/不一致时。
        """
        if not rows:
            raise ValueError("insert_detections_batch: rows 为空 / rows is empty")
        filename = rows[0].get("filename")
        if not filename:
            raise ValueError("insert_detections_batch: filename 缺失 / missing filename")
        for r in rows:
            if r.get("filename") != filename:
                raise ValueError(
                    "insert_detections_batch: rows 必须同一照片 / "
                    f"rows must share one filename, got {r.get('filename')} vs {filename}"
                )

        now = _now_iso()
        cols = DETECTION_COLUMNS
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        sql = (
            f"INSERT INTO bird_detections ({col_str}, created_at, updated_at) "
            f"VALUES ({placeholders}, ?, ?)"
        )
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM bird_detections WHERE filename = ?",
                    (filename,)
                )
                for data in rows:
                    values = [data.get(c) for c in cols]
                    # deleted 列必须落 0/1（缺失按 0），避免 NULL 导致
                    # `deleted = 0` 条件漏匹配
                    # deleted must be 0/1 (default 0); NULL would break
                    # `deleted = 0` filtering.
                    if not values[cols.index("deleted")]:
                        values[cols.index("deleted")] = 0
                    self._conn.execute(sql, values + [now, now])
            self._safe_commit()
        return len(rows)

    def get_detections(self, filename: str) -> List[dict]:
        """
        返回一张照片的全部检测记录，按 bird_index 升序。

        参数:
        filename (str): 照片前缀（与 photos.filename 同键）

        返回:
        List[dict]: 检测记录列表；无记录时为空列表

        Return all detections of one photo ordered by bird_index.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM bird_detections WHERE filename = ? "
                "ORDER BY bird_index",
                (filename,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_all_detections(self, include_polygon: bool = True) -> List[dict]:
        """
        返回全表检测记录，按 filename、bird_index 升序。

        参数:
        include_polygon (bool): False 时跳过 mask_polygon 列（每行数百
            字节的轮廓 JSON）。召回/导出等只需要物种与标记字段，NAS 等
            网络盘上全表拉取时瘦身可显著减少传输量。

        返回:
        List[dict]: 检测记录列表

        All detections ordered by filename/bird_index. Set
        include_polygon=False to skip the bulky mask_polygon column.
        """
        cols = "*" if include_polygon else (
            "id, filename, bird_index, is_selected, "
            "bbox_x, bbox_y, bbox_w, bbox_h, "
            "area_ratio, yolo_conf, crop_sharpness, "
            "species_cn, species_en, scientific_name, "
            "species_confidence, class_id, gbif_rarity_100, "
            "notable, notable_reason, edited, deleted, "
            "created_at, updated_at")
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT {cols} FROM bird_detections "
                f"ORDER BY filename, bird_index"
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_notable_species_map(self) -> dict:
        """
        V5.2 返回召回照片的「待确认鸟种」映射。

        返回:
        Dict[str, List[str]]: {filename: [召回鸟种中文名, ...]}

        Return the buried-species map for recalled photos.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT filename, GROUP_CONCAT(DISTINCT species_cn) AS sp "
                "FROM bird_detections "
                "WHERE notable = 1 AND deleted = 0 "
                "AND species_cn IS NOT NULL "
                "GROUP BY filename")
            return {row[0]: [s for s in (row[1] or "").split(",") if s]
                    for row in cursor.fetchall()}

    def apply_recall_marks(self, photo_flags: List[str],
                           detection_marks: List[dict]) -> int:
        """
        V5.2 物种召回：批量写入照片级与逐鸟级召回标记（单事务）。

        参数:
        photo_flags (List[str]): 要标记 notable=1 的照片前缀列表
        detection_marks (List[dict]): 每项 {filename, bird_index,
            notable_reason}，写入 bird_detections.notable/reason

        返回:
        int: 标记的照片数

        Bulk-apply species-recall flags to photos + detections in one
        transaction.
        """
        now = _now_iso()
        with self._lock:
            with self._conn:
                # 幂等重标：先清全部旧召回标记（仅原标记行，updated_at
                # 只在标记变化时才被动到，避免全量 sidecar 重导出）
                self._conn.execute(
                    "UPDATE photos SET notable = 0, updated_at = ? "
                    "WHERE notable = 1", (now,))
                self._conn.execute(
                    "UPDATE bird_detections SET notable = 0, "
                    "notable_reason = NULL, updated_at = ? "
                    "WHERE notable = 1", (now,))
                for filename in photo_flags:
                    self._conn.execute(
                        "UPDATE photos SET notable = 1, updated_at = ? "
                        "WHERE filename = ?", (now, filename))
                for m in detection_marks:
                    self._conn.execute(
                        "UPDATE bird_detections SET notable = 1, "
                        "notable_reason = ?, updated_at = ? "
                        "WHERE filename = ? AND bird_index = ?",
                        (m.get("notable_reason"), now,
                         m.get("filename"), m.get("bird_index")))
            self._safe_commit()
        return len(photo_flags)

    def update_detection_species(
        self,
        filename: str,
        bird_index: int,
        species_cn: Optional[str],
        species_en: Optional[str],
        scientific_name: Optional[str] = None,
        class_id: Optional[int] = None,
    ) -> bool:
        """
        人工修改某一只鸟的物种（二期编辑功能入口，本期预留）。

        参数:
        filename (str): 照片前缀
        bird_index (int): 鸟序号
        species_cn / species_en / scientific_name (Optional[str]): 新物种名
        class_id (Optional[int]): 新模型类别 ID（可反查时填）

        返回:
        bool: 是否命中并更新（无该行返回 False）

        Manually overwrite the species of one detected bird.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE bird_detections SET species_cn = ?, species_en = ?, "
                "scientific_name = ?, class_id = ?, edited = 1, updated_at = ? "
                "WHERE filename = ? AND bird_index = ?",
                (species_cn, species_en, scientific_name, class_id,
                 _now_iso(), filename, bird_index)
            )
            updated = cursor.rowcount > 0
            self._safe_commit()
            return updated

    def update_detection_selection(self, filename: str,
                                   bird_indexes: List[int]) -> int:
        """
        人工设置一张照片的主鸟（可多只，V5.4 编辑器保存入口）。

        bird_indexes 内的行 is_selected=1，其余行清 0；只更新发生变化的
        行（updated_at 不动无变化行，避免误触发 sidecar 重导出）。

        参数:
        filename (str): 照片前缀
        bird_indexes (List[int]): 主鸟的鸟序号列表（≤3 由调用方约束）

        返回:
        int: 发生变化的行数

        Human-select main birds (multiple allowed). Only changed rows
        are touched so unchanged photos never re-export.
        """
        wanted = [int(i) for i in dict.fromkeys(bird_indexes)]
        now = _now_iso()
        changed = 0
        with self._lock:
            with self._conn:
                for row in self._conn.execute(
                        "SELECT bird_index, is_selected, deleted "
                        "FROM bird_detections WHERE filename = ?",
                        (filename,)).fetchall():
                    idx, cur, deleted = row[0], row[1], row[2]
                    target = 1 if (idx in wanted and not deleted) else 0
                    if cur != target:
                        self._conn.execute(
                            "UPDATE bird_detections SET is_selected = ?, "
                            "updated_at = ? "
                            "WHERE filename = ? AND bird_index = ?",
                            (target, now, filename, idx))
                        changed += 1
            self._safe_commit()
        return changed

    def soft_delete_detections(self, filename: str,
                               bird_indexes: List[int]) -> int:
        """
        软删除一张照片的若干检测框（人工删框，只隐藏不物理删行）。

        软删行同时清除主鸟/召回标记（is_selected=0, notable=0），筛选、
        召回、编辑器、sidecar 导出均跳过 deleted=1 的行。

        参数:
        filename (str): 照片前缀
        bird_indexes (List[int]): 要删除的鸟序号列表

        返回:
        int: 实际标记删除的行数（幂等，重复调用返回 0）

        Soft-delete detection boxes (hidden, never physically removed).
        """
        if not bird_indexes:
            return 0
        placeholders = ", ".join(["?"] * len(bird_indexes))
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE bird_detections SET deleted = 1, is_selected = 0, "
                "notable = 0, notable_reason = NULL, updated_at = ? "
                f"WHERE filename = ? AND bird_index IN ({placeholders}) "
                "AND deleted = 0",
                (_now_iso(), filename, *bird_indexes))
            updated = cursor.rowcount
            if updated:
                with self._conn:
                    # 照片级召回标记同步修正：本照片的待确认种被删光时
                    # 立即归位（不依赖后台召回重算，编辑器保存后视图
                    # 即刻正确）
                    # Fix photo-level notable right away so the editor
                    # save path shows correct state without waiting for
                    # the background recall rebuild.
                    self._conn.execute(
                        "UPDATE photos SET notable = "
                        "CASE WHEN EXISTS (SELECT 1 FROM bird_detections d "
                        "WHERE d.filename = photos.filename "
                        "AND d.notable = 1 AND d.deleted = 0) "
                        "THEN 1 ELSE 0 END, updated_at = ? "
                        "WHERE filename = ? AND notable = 1",
                        (_now_iso(), filename))
                self._safe_commit()
            return updated

    def soft_delete_species_detections(
        self, species_cn: Optional[str] = None,
        species_en: Optional[str] = None,
        scientific_name: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        """
        全目录软删除某鸟种的全部检测框（批量清理 AI 误识别）。

        只删 is_selected=0 的行（该鸟种既然进了「待确认」列表，就从未
        当过主鸟）；已删行跳过。名字匹配中文名/英文名/学名任一相等。

        参数:
        species_cn / species_en / scientific_name (Optional[str]): 鸟种名

        返回:
        Tuple[List[str], int]: (受影响照片前缀列表, 删除的检测行数)

        Batch soft-delete every detection of one species in this
        directory (AI-misidentify cleanup). Returns affected filenames
        and the number of deleted rows.
        """
        conds, params = [], []
        for col, val in (("species_cn", species_cn),
                         ("species_en", species_en),
                         ("scientific_name", scientific_name)):
            if isinstance(val, str) and val.strip():
                conds.append(f"{col} = ?")
                params.append(val.strip())
        if not conds:
            return [], 0
        where = ("WHERE deleted = 0 AND is_selected = 0 AND ("
                 + " OR ".join(conds) + ")")
        now = _now_iso()
        with self._lock:
            rows = self._conn.execute(
                "SELECT filename, bird_index FROM bird_detections "
                f"{where}", params).fetchall()
            if not rows:
                return [], 0
            with self._conn:
                for row in rows:
                    self._conn.execute(
                        "UPDATE bird_detections SET deleted = 1, "
                        "is_selected = 0, notable = 0, notable_reason = NULL, "
                        "updated_at = ? "
                        "WHERE filename = ? AND bird_index = ?",
                        (now, row[0], row[1]))
                # photos.notable 同步修正：某照片的待确认种被删光时，
                # 照片级标记立即归位（不依赖后续召回重算）
                # Fix photo-level notable flags right away for photos
                # whose only pending species was just deleted.
                for filename in sorted({r[0] for r in rows}):
                    self._conn.execute(
                        "UPDATE photos SET notable = "
                        "CASE WHEN EXISTS (SELECT 1 FROM bird_detections d "
                        "WHERE d.filename = photos.filename "
                        "AND d.notable = 1 AND d.deleted = 0) "
                        "THEN 1 ELSE 0 END, updated_at = ? "
                        "WHERE filename = ? AND notable = 1",
                        (now, filename))
            self._safe_commit()
            filenames = sorted({row[0] for r in rows})
            return filenames, len(rows)

    def soft_delete_species_everywhere(
        self, species_cn: Optional[str] = None,
        species_en: Optional[str] = None,
        scientific_name: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        """
        全目录软删除某鸟种的**全部**识别（含主鸟框，并清 photos 主鸟种）。

        与 soft_delete_species_detections 的区别：后者只删「待确认」框
        （is_selected=0，召回清单里的鸟种从未当过主鸟）；本方法面向
        「整批照片的主鸟种识别错了」的场景——
          1. bird_detections 该鸟种全部未删行软删（含 is_selected=1 的
             主鸟框：deleted=1, is_selected=0, notable=0）；
          2. photos 表主鸟种（bird_species_cn/en）命中该名的行清空为
             NULL，照片回到「无鸟种」状态（星级不动，需要时另跑 restar）；
          3. photos.notable 同步修正（照抄待确认删除的归位逻辑）。
        名字匹配中文名/英文名/学名任一相等；软删可恢复，原照片不动。

        参数:
        species_cn / species_en / scientific_name (Optional[str]): 鸟种名

        返回:
        Tuple[List[str], int]: (受影响照片前缀列表, 删除的检测行数)

        Batch soft-delete EVERY identification of one species, including
        main-bird rows, and clear the matching photos.main-species fields.
        For the "the whole batch was misidentified as species X" case.
        Soft-deleted rows stay recoverable; original photos untouched.
        """
        conds, params = [], []
        for col, val in (("species_cn", species_cn),
                         ("species_en", species_en),
                         ("scientific_name", scientific_name)):
            if isinstance(val, str) and val.strip():
                conds.append(f"{col} = ?")
                params.append(val.strip())
        if not conds:
            return [], 0
        det_where = ("WHERE deleted = 0 AND (" + " OR ".join(conds) + ")")
        now = _now_iso()
        with self._lock:
            rows = self._conn.execute(
                "SELECT filename, bird_index FROM bird_detections "
                f"{det_where}", params).fetchall()
            with self._conn:
                for row in rows:
                    self._conn.execute(
                        "UPDATE bird_detections SET deleted = 1, "
                        "is_selected = 0, notable = 0, notable_reason = NULL, "
                        "updated_at = ? "
                        "WHERE filename = ? AND bird_index = ?",
                        (now, row[0], row[1]))
                # 主鸟种清空（cn/en 任一命中该名字）
                # Clear photos.main-species wherever the name matches.
                photo_conds, photo_args = [], []
                for col_pair_val in (species_cn, species_en):
                    if isinstance(col_pair_val, str) and col_pair_val.strip():
                        photo_conds.append(
                            "(bird_species_cn = ? OR bird_species_en = ?)")
                        photo_args.extend([col_pair_val.strip(),
                                           col_pair_val.strip()])
                if photo_conds:
                    self._conn.execute(
                        "UPDATE photos SET bird_species_cn = NULL, "
                        "bird_species_en = NULL, updated_at = ? "
                        "WHERE " + " OR ".join(photo_conds),
                        [now] + photo_args)
                # photos.notable 归位：待确认种被删光的照片立即摘标记
                for filename in sorted({r[0] for r in rows}):
                    self._conn.execute(
                        "UPDATE photos SET notable = "
                        "CASE WHEN EXISTS (SELECT 1 FROM bird_detections d "
                        "WHERE d.filename = photos.filename "
                        "AND d.notable = 1 AND d.deleted = 0) "
                        "THEN 1 ELSE 0 END, updated_at = ? "
                        "WHERE filename = ? AND notable = 1",
                        (now, filename))
            self._safe_commit()
            filenames = sorted({row[0] for r in rows})
            return filenames, len(rows)

    def rename_species_everywhere(
        self, old_cn: Optional[str] = None, old_en: Optional[str] = None,
        old_sci: Optional[str] = None,
        new_cn: Optional[str] = None, new_en: Optional[str] = None,
        new_sci: Optional[str] = None,
    ) -> Tuple[List[str], int]:
        """
        全目录批量把某鸟种**全部**识别改为另一个鸟种（含主鸟）。

        与 soft_delete_species_everywhere 同族，面向「整批照片的鸟种
        识别错了、且知道正确答案」的场景——
          1. bird_detections 该鸟种全部未删行（含 is_selected=1 主鸟框）
             的 species_cn/en/scientific_name 改为新名；
          2. photos 主鸟种（bird_species_cn/en）命中旧名的行改写为新名；
          3. notable 等召回标记保持，交由后续召回重算按新名重新评估。
        旧名匹配中文名/英文名/学名任一相等；原照片不动。

        参数:
        old_cn / old_en / old_sci (Optional[str]): 旧鸟种名（任一非空）
        new_cn / new_en / new_sci (Optional[str]): 新鸟种名（未提供的
            维度写 NULL——调用方应至少给中文名或英文名）

        返回:
        Tuple[List[str], int]: (受影响照片前缀列表, 改写的检测行数)

        Batch-rename every identification of one species (main-bird rows
        included) to another species directory-wide, for the
        "whole batch misidentified, correct answer known" case.
        """
        conds, params = [], []
        for col, val in (("species_cn", old_cn),
                         ("species_en", old_en),
                         ("scientific_name", old_sci)):
            if isinstance(val, str) and val.strip():
                conds.append(f"{col} = ?")
                params.append(val.strip())
        if not conds or not any(
                isinstance(v, str) and v.strip()
                for v in (new_cn, new_en, new_sci)):
            return [], 0
        det_where = ("WHERE deleted = 0 AND (" + " OR ".join(conds) + ")")
        now = _now_iso()
        with self._lock:
            rows = self._conn.execute(
                "SELECT filename, bird_index FROM bird_detections "
                f"{det_where}", params).fetchall()
            with self._conn:
                for row in rows:
                    self._conn.execute(
                        "UPDATE bird_detections SET species_cn = ?, "
                        "species_en = ?, scientific_name = ?, updated_at = ? "
                        "WHERE filename = ? AND bird_index = ?",
                        (new_cn or None, new_en or None, new_sci or None,
                         now, row[0], row[1]))
                # photos 主鸟种改写（cn/en 任一命中旧名）
                # Rewrite photos.main-species wherever the old name matches.
                photo_conds, photo_args = [], []
                for v in (old_cn, old_en):
                    if isinstance(v, str) and v.strip():
                        photo_conds.append(
                            "(bird_species_cn = ? OR bird_species_en = ?)")
                        photo_args.extend([v.strip(), v.strip()])
                if photo_conds:
                    self._conn.execute(
                        "UPDATE photos SET bird_species_cn = ?, "
                        "bird_species_en = ?, updated_at = ? "
                        "WHERE " + " OR ".join(photo_conds),
                        [new_cn or None, new_en or None, now] + photo_args)
            self._safe_commit()
            filenames = sorted({row[0] for r in rows})
            return filenames, len(rows)

    def get_notable_species_counts(self) -> List[dict]:
        """
        汇总召回鸟种统计（V5.4 召回筛选下方的待确认鸟种列表数据）。

        返回:
        List[dict]: 每鸟种一项，按照片数降序:
            {cn, en, scientific, photos, detections}

        Aggregate recall-species stats for the pending-species list.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT species_cn, species_en, scientific_name, "
                "COUNT(DISTINCT filename) AS photos, COUNT(*) AS dets "
                "FROM bird_detections "
                "WHERE notable = 1 AND deleted = 0 "
                "AND (species_cn IS NOT NULL OR species_en IS NOT NULL) "
                "GROUP BY species_cn, species_en, scientific_name "
                "ORDER BY photos DESC, species_cn")
            return [{"cn": r[0] or "", "en": r[1] or "",
                     "scientific": r[2] or "",
                     "photos": r[3], "detections": r[4]}
                    for r in cursor.fetchall()]

    def get_export_stamps(self) -> Dict[str, str]:
        """
        读 sidecar 导出戳缓存 {filename: stamp}（表惰性创建）。

        V5.4 性能优化：导出器据此跳过"DB 内容未变化"的照片，不再逐个
        打开 NAS 上的 JSON 比对戳（1240 个文件 × ~30ms 是全量导出的
        主要成本）。缓存只影响"是否重写文件"，不影响导出内容本身；
        缓存缺失/不匹配时导出器自动回退到读文件比对并回填缓存。

        返回:
        Dict[str, str]: {照片前缀: 导出戳}

        Read the export-stamp cache (lazily created table) so the
        exporter can skip unchanged photos without opening each JSON.
        """
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS export_stamps ("
                "filename TEXT PRIMARY KEY, stamp TEXT)")
            rows = self._conn.execute(
                "SELECT filename, stamp FROM export_stamps").fetchall()
            return {r[0]: r[1] for r in rows}

    def upsert_export_stamps(self, stamps: Dict[str, str]) -> int:
        """
        批量写导出戳缓存（单事务）。

        参数:
        stamps (Dict[str, str]): {照片前缀: 导出戳}

        返回:
        int: 写入行数

        Bulk-upsert the export-stamp cache in one transaction.
        """
        if not stamps:
            return 0
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS export_stamps ("
                    "filename TEXT PRIMARY KEY, stamp TEXT)")
                self._conn.executemany(
                    "INSERT OR REPLACE INTO export_stamps "
                    "(filename, stamp) VALUES (?, ?)",
                    list(stamps.items()))
            self._safe_commit()
        return len(stamps)

    def get_main_species_map(self) -> dict:
        """
        每张照片的主鸟种列表（V5.4 多主鸟缩略图标题用）。

        返回:
        Dict[str, List[Tuple[str, str]]]: {filename: [(中文名, 英文名),
        ...]}，按 bird_index 升序；无主鸟检测的照片不在映射里。

        Per-photo main-species list (ordered by bird_index) for the
        multi-main thumbnail title.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT filename, species_cn, species_en "
                "FROM bird_detections "
                "WHERE is_selected = 1 AND deleted = 0 "
                "ORDER BY filename, bird_index")
            result: dict = {}
            for row in cursor.fetchall():
                if row[1] or row[2]:
                    result.setdefault(row[0], []).append((row[1] or "",
                                                          row[2] or ""))
            return result

    def get_photos_by_species(
        self,
        cn: Optional[str] = None,
        en: Optional[str] = None,
        exclude_filename: Optional[str] = None,
    ) -> List[dict]:
        """
        查当前项目内同鸟种、可当正样本的照片（has_bird=1 且 rating!=-1）。

        cn 优先；cn 为空时用 en。exclude_filename 排除被改正图自身。
        """
        col = None
        val = None
        if cn and cn.strip():
            col, val = "bird_species_cn", cn.strip()
        elif en and en.strip():
            col, val = "bird_species_en", en.strip()
        if col is None:
            return []
        assert col in {"bird_species_cn", "bird_species_en"}
        sql = (
            f"SELECT * FROM photos WHERE {col} = ? AND has_bird = 1 "
            "AND rating != -1"
        )
        params: List[Any] = [val]
        if exclude_filename:
            sql += " AND filename != ?"
            params.append(exclude_filename)
        sql += " ORDER BY filename"
        with self._lock:
            cursor = self._conn.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_statistics(self) -> dict:
        """
        获取评分统计信息。

        Returns:
            包含统计数据的字典，如:
            {
                "total": 217,
                "has_bird": 180,
                "flying": 15,
                "by_rating": {0: 50, 1: 60, 2: 45, 3: 25}
            }
        """
        stats = {}

        with self._lock:
            # 总数
            row = self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()
            stats["total"] = row[0]

            # 有鸟数
            row = self._conn.execute(
                "SELECT COUNT(*) FROM photos WHERE has_bird = 1"
            ).fetchone()
            stats["has_bird"] = row[0]

            # 飞行数
            row = self._conn.execute(
                "SELECT COUNT(*) FROM photos WHERE is_flying = 1"
            ).fetchone()
            stats["flying"] = row[0]

            # 按评分统计
            cursor = self._conn.execute(
                "SELECT rating, COUNT(*) as cnt FROM photos GROUP BY rating ORDER BY rating"
            )
            stats["by_rating"] = {row[0]: row[1] for row in cursor.fetchall()}

        return stats

    def count(self) -> int:
        """返回总记录数。"""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()
            return row[0]

    def exists(self) -> bool:
        """数据库文件是否存在。"""
        return os.path.exists(self.db_path)

    # ==========================================================================
    #  更新操作
    # ==========================================================================

    def update_photo(self, filename: str, data: dict) -> bool:
        """
        按 filename 更新指定字段。

        Args:
            filename: 照片文件名
            data: 要更新的字段字典（仅包含需要更新的字段）

        Returns:
            是否成功更新
        """
        cleaned = self._clean_data(data)
        cleaned["updated_at"] = _now_iso()

        # 仅保留合法列，排除 filename 和 id
        columns = [k for k in cleaned if k in COLUMN_NAMES and k not in ("filename", "id")]
        if not columns:
            return False

        values = [cleaned[k] for k in columns]
        set_clause = ", ".join(f"{c} = ?" for c in columns)

        sql = f"UPDATE photos SET {set_clause} WHERE filename = ?"
        values.append(filename)

        with self._lock:
            cursor = self._conn.execute(sql, values)
            self._safe_commit()
            return cursor.rowcount > 0

    def update_burst_ids(self, burst_map: dict) -> int:
        """
        批量更新照片的 burst_id 和 burst_position。
        
        Args:
            burst_map: 字典，格式为
                {filename: (burst_id, burst_position)} 或
                {(source_dir, filename): (burst_id, burst_position)}
            
        Returns:
            成功更新的记录数
        """
        if not burst_map:
            return 0
            
        updates = []
        now = _now_iso()
        for photo_key, (bid, pos) in burst_map.items():
            if isinstance(photo_key, tuple):
                filename = photo_key[-1]
            else:
                filename = photo_key
            if not filename:
                continue
            updates.append((bid, pos, now, filename))
            
        sql = """
        UPDATE photos 
        SET burst_id = ?, burst_position = ?, updated_at = ? 
        WHERE filename = ?
        """
        
        with self._lock:
            cursor = self._conn.executemany(sql, updates)
            self._safe_commit()
            return cursor.rowcount

    def clear_burst_ids(self) -> int:
        """清空全部连拍分组字段。"""
        sql = """
        UPDATE photos
        SET burst_id = NULL, burst_position = NULL, updated_at = ?
        WHERE burst_id IS NOT NULL OR burst_position IS NOT NULL
        """
        with self._lock:
            cursor = self._conn.execute(sql, [_now_iso()])
            self._safe_commit()
            return cursor.rowcount

    def delete_photo(self, filename: str) -> bool:
        """从 photos 表中删除指定文件名的记录。

        Args:
            filename: 照片文件名

        Returns:
            是否成功删除
        """
        sql = "DELETE FROM photos WHERE filename = ?"
        with self._lock:
            cursor = self._conn.execute(sql, [filename])
            self._safe_commit()
            return cursor.rowcount > 0

    def update_ratings_batch(self, updates: List[dict]) -> int:
        """
        批量更新评分及相关数据。

        用于重新评星场景（PostAdjustmentEngine）。

        Args:
            updates: 更新数据列表，每个字典必须包含 "filename" 键，
                     以及要更新的字段（如 rating, adj_sharpness, adj_topiq）

        Returns:
            成功更新的记录数
        """
        if not updates:
            return 0

        now = _now_iso()
        count = 0

        with self._lock:
            with self._conn:
                for upd in updates:
                    filename = upd.get("filename")
                    if not filename:
                        continue

                    cleaned = self._clean_data(upd)
                    cleaned["updated_at"] = now

                    columns = [k for k in cleaned if k in COLUMN_NAMES and k not in ("filename", "id")]
                    if not columns:
                        continue

                    values = [cleaned[k] for k in columns]
                    set_clause = ", ".join(f"{c} = ?" for c in columns)

                    sql = f"UPDATE photos SET {set_clause} WHERE filename = ?"
                    values.append(filename)

                    cursor = self._conn.execute(sql, values)
                    if cursor.rowcount > 0:
                        count += 1

        return count

    def clear_cache_paths(self) -> int:
        """清空缓存相关路径字段（临时 JPG、调试裁切、YOLO 调试图）。"""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE photos SET debug_crop_path = NULL, temp_jpeg_path = NULL, yolo_debug_path = NULL"
            )
            self._safe_commit()
            return cursor.rowcount

    # ==========================================================================
    #  元数据操作
    # ==========================================================================

    def get_meta(self, key: str) -> Optional[str]:
        """获取元数据值。"""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """设置元数据值。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value)
            )
            self._safe_commit()

    # ==========================================================================
    #  同步预留
    # ==========================================================================

    def get_updated_since(self, since: str) -> List[dict]:
        """
        获取指定时间之后更新的记录（增量同步用）。

        Args:
            since: ISO 8601 时间字符串

        Returns:
            更新记录列表
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM photos WHERE updated_at > ? ORDER BY updated_at",
                (since,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==========================================================================
    #  连接管理
    # ==========================================================================

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ==========================================================================
    #  内部方法
    # ==========================================================================

    def _safe_commit(self) -> None:
        """仅在存在活动事务时提交，兼容 autocommit 场景。"""
        if not self._conn:
            return
        try:
            if self._conn.in_transaction:
                self._conn.commit()
        except sqlite3.OperationalError as e:
            # 某些运行时在 autocommit 下会抛 "no transaction is active"
            if "no transaction is active" in str(e).lower():
                return
            raise

    @staticmethod
    def _clean_data(data: dict) -> dict:
        """
        清洗输入数据，处理 CSV 兼容格式转换。

        转换规则：
        - "yes"/"no" → 1/0（仅对 has_bird, is_flying 字段）
        - "-" 或空字符串 → None
        - 数值字符串 → 对应的 float/int
        """
        cleaned = {}
        for key, value in data.items():
            # 跳过非法列名
            if key not in COLUMN_NAMES:
                continue

            # 布尔/yes-no 字段（优先处理，"-"/None/空 → 0）
            if key in ("has_bird", "is_flying"):
                if value is None or value == "-" or value == "":
                    cleaned[key] = 0
                elif isinstance(value, str):
                    cleaned[key] = 1 if value.lower() in ("yes", "1", "true") else 0
                else:
                    cleaned[key] = 1 if value else 0
                continue

            # 处理 None 和占位符
            if value is None or value == "-" or value == "":
                cleaned[key] = None
                continue

            # 数值字段
            if key in ("confidence", "head_sharp", "left_eye", "right_eye",
                        "beak", "nima_score", "flight_conf", "focus_x",
                        "focus_y", "adj_sharpness", "adj_topiq",
                        # V2: 新增数值字段
                        "focal_length", "gps_latitude", "gps_longitude",
                        "gps_altitude", "birdid_confidence"):
                try:
                    cleaned[key] = float(value)
                except (ValueError, TypeError):
                    cleaned[key] = None
                continue

            # 整数字段
            if key in ("rating", "iso", "focal_length_35mm"):
                try:
                    cleaned[key] = int(float(value))
                except (ValueError, TypeError):
                    cleaned[key] = 0 if key == "rating" else None
                continue

            # 文本字段直接使用（包括 V2 新增的文本字段）
            # shutter_speed, aperture, camera_model, lens_model,
            # title, caption, city, state_province, country,
            # date_time_original, bird_species_cn, bird_species_en, exposure_status
            cleaned[key] = value

        return cleaned


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
