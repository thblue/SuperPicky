#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_rename_species — 鸟种修正 CLI（BirdIndex 网站站内改种的写后端）

场景：在 BirdIndex 网站上浏览时发现某张/某批照片的鸟种识别错了（或
根本不是鸟），需要不开 SuperPicky 界面、可被网站后台以子进程委托执行
的修正工具。本工具对单个照片库（含 .superpicky/report.db 的目录）执行
四类操作，覆盖「单张 / 整种」×「改种 / 误检清除」：

  photo   <库> <照片名> --to-cn/--to-en/--to-sci   单张改主鸟种
  photo   <库> <照片名> --wipe                     单张清除识别（不是鸟）
  species <库> --old-cn/--old-en/--old-sci
          --to-cn/--to-en/--to-sci                 整库整种改为其他鸟种
  species <库> --old-cn/... --wipe                 整库整种识别删除（误检）

写入链路与结果浏览器「整批改种/整批删除」（V5.4）完全一致：
  report.db（photos 主鸟种 + bird_detections 检测框软删/改写）→
  sidecar JSON 同步（.superpicky/meta/，edits 留痕 actor=human）→
  物种召回重算 + 增量重导出（无鸟照片的 JSON 幂等删除，BirdIndex
  数据源自动收敛）。原照片文件永不移动或删除；仅清理
  .superpicky/cache/ 下的生成预览（temp_jpeg_path 可能指向原片，
  防误删硬闸门见 _remove_previews）。

「误检清除」的归一规则：照片的所有检测框软删后若再无任何存活框，
photos 行归一无鸟态（has_bird=0、rating=-1、主鸟种/置信度/召回标记/
精选/飞鸟/稀有度字段清空）；仍存活其他鸟种的照片只删该鸟种，保持
有鸟态。软删行保留可追溯（deleted=1），DB 层可恢复。

默认 dry-run 只统计将要发生的变化，--apply 才落盘。--json 时人类
可读日志转 stderr、结果 JSON 走 stdout，供 BirdIndex 的 /api/fix
子进程解析（BirdIndex 侧负责把 photo_rel 的扩展名剥掉或原样传入，
本工具用 os.path.splitext 兼容两种形态）。

用法:
    python spb_rename_species.py photo "//NAS/库" DSC_0001.ARW \
        --to-cn 白鹭 --to-en "Little Egret" --apply
    python spb_rename_species.py photo "//NAS/库" DSC_0001 --wipe --apply
    python spb_rename_species.py species "//NAS/库" --old-cn 乌鸦 \
        --to-cn 小嘴乌鸦 --to-en "Carrion Crow" --apply
    python spb_rename_species.py species "//NAS/库" --old-cn 乌鸦 --wipe --apply

Species-fix CLI: the write backend delegated to by the BirdIndex web
UI (single-photo rename / single-photo wipe / whole-species rename /
whole-species wipe), mirroring the results browser's bulk-edit chain.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
from typing import List, Optional, Tuple

# 允许直接 `python spb_rename_species.py` 运行（仓库根即 CWD）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.report_db import ReportDB

# 无鸟归一：photos 行清回无鸟态的字段集（照抄 report_db.
# wipe_all_identifications 的写法，评级归 -1=无鸟）
_NO_BIRD_FIELDS = {
    "has_bird": 0,
    "rating": -1,
    "bird_species_cn": None,
    "bird_species_en": None,
    "birdid_confidence": None,
    "notable": 0,
    "picked": 0,
    "is_flying": 0,
    "flight_conf": None,
    "rarity_index": None,
    "iucn_category": None,
    "gbif_rarity_100": None,
}


def _log(message: str) -> None:
    """
    人类可读日志走 stderr（--json 模式下 stdout 只留给结果 JSON）。

    参数:
    message (str): 日志行
    """
    print(message, file=sys.stderr)


@functools.lru_cache(maxsize=1)
def _lookup_class_id(sci: Optional[str], en: Optional[str]) -> \
        Optional[int]:
    """
    反查新鸟种的模型类别 ID（birdid 名录库，学名优先、英文名回退）。

    人工改种后检测框的 class_id 仍指向旧种的模型类别，会污染训练数据
    关联；多鸟编辑器保存时同样反查回填。名录库缺失或未命中返回 None
    （等价于「无已知类别关联」）。BirdDatabaseManager 初始化会 print，
    重定向到 stderr 保证 --json 的 stdout 纯净。

    参数:
    sci / en (Optional[str]): 新鸟种学名 / 英文名

    返回:
    Optional[int]: 模型类别 ID；未命中返回 None
    """
    try:
        from birdid.bird_database_manager import BirdDatabaseManager
        with contextlib.redirect_stdout(sys.stderr):
            mgr = BirdDatabaseManager()
        return mgr.get_class_id_by_scientific_name(
            sci or "", english_name=en or "")
    except Exception:  # noqa: BLE001（名录库不可用时静默降级）
        return None


def _strip_ext(name: str) -> str:
    """
    照片名归一为 report.db 的 filename 前缀（无扩展名）。

    兼容 BirdIndex 传来的 photo_rel 文件名（带扩展名）与裸前缀两种
    形态，统一剥掉最后一个扩展名段。

    参数:
    name (str): 照片文件名或前缀

    返回:
    str: 去扩展名前缀
    """
    return os.path.splitext(os.path.basename(name.strip()))[0]


def _open_db(root: str) -> Optional[ReportDB]:
    """
    打开照片库的 report.db；库不存在时返回 None。

    参数:
    root (str): 照片库目录（含 .superpicky/report.db）

    返回:
    Optional[ReportDB]: 已打开的库；目录非法返回 None
    """
    if not os.path.isfile(os.path.join(root, '.superpicky', 'report.db')):
        _log(f"⚠️ 跳过（无 report.db）: {root}")
        return None
    return ReportDB(root)


def _species_match_conditions(cn: Optional[str], en: Optional[str],
                              sci: Optional[str]) -> Tuple[List[str], List]:
    """
    构造鸟种名任一命中（中/英/学名 OR）的 SQL 条件（与 report_db 的
    rename/soft_delete_species_everywhere 完全同款匹配语义）。

    参数:
    cn / en / sci (Optional[str]): 鸟种名（至少一个非空）

    返回:
    Tuple[List[str], List]: (条件片段列表, 参数列表)；全空时均为空
    """
    conds, params = [], []
    for col, val in (("species_cn", cn), ("species_en", en),
                     ("scientific_name", sci)):
        if isinstance(val, str) and val.strip():
            conds.append(f"{col} = ?")
            params.append(val.strip())
    return conds, params


def _preview_species_scope(db: ReportDB, cn: Optional[str],
                           en: Optional[str], sci: Optional[str]) -> dict:
    """
    dry-run 预览：统计整种操作将影响的检测框、照片与将归一无鸟的照片。

    参数:
    db (ReportDB): 已打开的库
    cn / en / sci (Optional[str]): 旧鸟种名

    返回:
    dict: {'detections', 'photos', 'to_no_bird'} 计数
    """
    conds, params = _species_match_conditions(cn, en, sci)
    if not conds:
        return {"detections": 0, "photos": 0, "to_no_bird": 0}
    where = "WHERE deleted = 0 AND (" + " OR ".join(conds) + ")"
    rows = db._conn.execute(
        f"SELECT DISTINCT filename FROM bird_detections {where}",
        params).fetchall()
    filenames = [r[0] for r in rows]
    det_count = db._conn.execute(
        f"SELECT COUNT(*) FROM bird_detections {where}", params).fetchone()[0]
    no_bird = 0
    for name in filenames:
        live = db._conn.execute(
            "SELECT COUNT(*) FROM bird_detections "
            "WHERE filename = ? AND deleted = 0", (name,)).fetchone()[0]
        species_live = db._conn.execute(
            f"SELECT COUNT(*) FROM bird_detections "
            f"WHERE filename = ? AND deleted = 0 AND ({' OR '.join(conds)})",
            [name] + params).fetchone()[0]
        if live == species_live:
            no_bird += 1
    return {"detections": det_count, "photos": len(filenames),
            "to_no_bird": no_bird}


def _sweep_main_species(directory: str, old_cn: Optional[str],
                        old_en: Optional[str], old_sci: Optional[str],
                        new_cn: Optional[str] = None,
                        new_en: Optional[str] = None,
                        new_sci: Optional[str] = None) -> int:
    """
    全库扫尾：把 sidecar main_species 数组里命中旧名的条目改写/移除。

    整种操作的受影响文件列表按 report.db 检测框匹配生成，而多鸟编辑器
    设的主鸟可能只存在于 main_species 数组、DB 里没有同名检测框（人工
    选的主鸟种与 AI 检测框物种不一致）——这类照片不在列表里，逐文件
    sidecar 同步碰不到它，旧鸟名会残留在 main_species（BirdIndex 取种
    优先级最高），表现为「改种不生效」。本函数遍历 meta 下全部 JSON，
    字符串/对象两种条目形态统一处理：rename 改写为新名（对象条目保留
    bird_index 等辅助键），new_cn/en/sci 全空（wipe 语义）则移除条目。
    幂等可重复执行。

    参数:
    directory (str): 照片库目录（含 .superpicky/meta/）
    old_cn / old_en / old_sci (Optional[str]): 旧鸟种名（任一非空）
    new_cn / new_en / new_sci (Optional[str]): 新鸟种名；全空=移除条目

    返回:
    int: 改写/移除条目涉及的照片数
    """
    from core.sidecar_export import (_atomic_write_json,
                                     _main_entry_matches)

    names = {v.strip() for v in (old_cn, old_en, old_sci)
             if isinstance(v, str) and v.strip()}
    if not names:
        return 0
    wiping = not any(isinstance(v, str) and v.strip()
                     for v in (new_cn, new_en, new_sci))
    meta_dir = os.path.join(directory, ".superpicky", "meta")
    if not os.path.isdir(meta_dir):
        return 0
    import datetime
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    touched = 0
    for json_name in sorted(os.listdir(meta_dir)):
        if not json_name.endswith(".json"):
            continue
        path = os.path.join(meta_dir, json_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        main = payload.get("main_species") if isinstance(payload, dict) else None
        if not isinstance(main, list) or not any(
                _main_entry_matches(m, names) for m in main):
            continue
        new_main, changed = [], False
        for entry in main:
            if not _main_entry_matches(entry, names):
                new_main.append(entry)
                continue
            changed = True
            if wiping:
                continue  # wipe：直接移除条目
            if isinstance(entry, dict):
                fresh = {k: v for k, v in entry.items()
                         if k not in ("cn", "en", "scientific")}
                fresh.update({"cn": new_cn, "en": new_en,
                              "scientific": new_sci})
                new_main.append(fresh)
            else:
                new_main.append(new_cn or new_en)
        if not changed:
            continue
        payload["main_species"] = new_main
        payload.setdefault("edits", []).append({
            "timestamp": stamp, "actor": "human",
            "action": "species_renamed" if not wiping else "main_species_removed",
            "bird_index": None,
            "old": {"cn": old_cn, "en": old_en, "scientific": old_sci},
            "new": None if wiping else {"cn": new_cn, "en": new_en,
                                        "scientific": new_sci},
        })
        try:
            _atomic_write_json(path, payload)
            touched += 1
        except OSError as e:
            _log(f"  ⚠️ main_species 扫尾写入失败 {json_name}: {e}")
    return touched


def _remove_previews(directory: str, rel_paths: List[Optional[str]]) -> int:
    """
    删除被归一无鸟照片的生成预览（仅限 .superpicky/cache/ 之下）。

    temp_jpeg_path 对纯 JPG / RAW+JPG 配对照片指向原片或伴随 JPG，绝对
    不能删——这是防误删原片的硬闸门（与 spb_wipe_ident 同款）。

    参数:
    directory (str): 照片库目录
    rel_paths (List[Optional[str]]): photos.temp_jpeg_path 相对路径列表

    返回:
    int: 实际删除的文件数
    """
    cache_root = os.path.normpath(os.path.join('.superpicky', 'cache'))
    removed = 0
    for rel in rel_paths:
        if not rel:
            continue
        norm = os.path.normpath(rel)
        if not norm.startswith(cache_root + os.sep):
            continue
        abs_path = os.path.join(directory, norm)
        try:
            if os.path.exists(abs_path):
                os.remove(abs_path)
                removed += 1
        except OSError as e:
            _log(f"  ⚠️ 预览删除失败 {rel}: {e}")
    return removed


def _recall_threshold() -> float:
    """
    读取召回置信度阈值（高级配置）；配置不可用时回退默认 35.0。

    返回:
    float: 召回触发的分类置信度阈值（百分比）
    """
    try:
        from advanced_config import get_advanced_config
        return float(get_advanced_config().recall_species_threshold)
    except Exception:  # noqa: BLE001（独立运行环境无 GUI 配置时兜底）
        return 35.0


def _finalize(db: ReportDB, directory: str) -> int:
    """
    收尾：重算物种召回 + 增量重导出 sidecar（与浏览器后台链路同款）。

    无鸟照片的 JSON 由此幂等删除（BirdIndex 数据源自动收敛）；导出戳
    缓存保证只重写内容变化的照片。

    参数:
    db (ReportDB): 已打开的库
    directory (str): 照片库目录

    返回:
    int: 本次写入的 sidecar JSON 数
    """
    from core.species_recall import run_species_recall
    from core.sidecar_export import export_directory_sidecars
    run_species_recall(db, species_threshold=_recall_threshold(),
                       log=_log)
    return export_directory_sidecars(db, directory, log=lambda *_: None)


def _sidecars_written(result: dict, count: int) -> dict:
    """把导出计数合入结果字典（链路末端的统一记录点）。"""
    result["sidecars_written"] = count
    return result


def cmd_photo_rename(db: ReportDB, root: str, filename: str,
                     new_cn: Optional[str], new_en: Optional[str],
                     new_sci: Optional[str], apply: bool) -> dict:
    """
    单张照片改主鸟种：photos 主鸟种 + 匹配检测框 + sidecar 同步改写。

    匹配规则：照片当前主鸟种名（bird_species_cn/en）作为旧名；该照片
    未删框中物种名命中旧名的框全部改写，若无任何框命中则改写主鸟框
    （is_selected=1，兼容主鸟种只存 photos 表的旧数据）。

    参数:
    db (ReportDB): 已打开的库
    root (str): 照片库目录
    filename (str): 照片前缀（无扩展名）
    new_cn / new_en / new_sci (Optional[str]): 新鸟种名（cn/en 至少一项）
    apply (bool): True 落盘，False 仅 dry-run

    返回:
    dict: 结果（status/changed_detections/sidecars_written 等）
    """
    from core.sidecar_export import rename_species_in_sidecar

    photo = db.get_photo(filename)
    if not photo:
        return {"status": "error", "reason": f"照片不存在: {filename}"}
    if not photo.get("has_bird"):
        return {"status": "error",
                "reason": f"该照片无鸟识别（has_bird=0）: {filename}"}

    old_cn = photo.get("bird_species_cn") or ""
    old_en = photo.get("bird_species_en") or ""
    detections = [d for d in db.get_detections(filename)
                  if not d.get("deleted")]
    targets = [d for d in detections
               if (old_cn and d.get("species_cn") == old_cn)
               or (old_en and d.get("species_en") == old_en)]
    if not targets:
        targets = [d for d in detections if d.get("is_selected")]

    result = {"status": "ok", "op": "photo-rename", "root": root,
              "filename": filename,
              "old": {"cn": old_cn, "en": old_en},
              "new": {"cn": new_cn, "en": new_en, "sci": new_sci},
              "changed_detections": len(targets)}
    if not apply:
        return result

    db.update_photo(filename, {
        "bird_species_cn": new_cn or None,
        "bird_species_en": new_en or None,
    })
    class_id = _lookup_class_id(new_sci, new_en)
    for det in targets:
        db.update_detection_species(filename, det["bird_index"],
                                    new_cn, new_en, new_sci,
                                    class_id=class_id)
    rename_species_in_sidecar(
        root, filename,
        new_cn=new_cn, new_en=new_en, new_sci=new_sci,
        old_cn=old_cn or None, old_en=old_en or None,
        class_id=class_id)
    return _sidecars_written(result, _finalize(db, root))


def cmd_photo_wipe(db: ReportDB, root: str, filename: str,
                   apply: bool) -> dict:
    """
    单张照片清除识别（误检）：全部检测框软删 + 照片归一无鸟态。

    参数:
    db (ReportDB): 已打开的库
    root (str): 照片库目录
    filename (str): 照片前缀（无扩展名）
    apply (bool): True 落盘，False 仅 dry-run

    返回:
    dict: 结果（status/soft_deleted/previews_removed 等）
    """
    photo = db.get_photo(filename)
    if not photo:
        return {"status": "error", "reason": f"照片不存在: {filename}"}

    live = [d for d in db.get_detections(filename) if not d.get("deleted")]
    result = {"status": "ok", "op": "photo-wipe", "root": root,
              "filename": filename, "has_bird": photo.get("has_bird"),
              "soft_deleted": len(live)}
    if not apply:
        return result

    if live:
        db.soft_delete_detections(filename, [d["bird_index"] for d in live])
    if photo.get("has_bird"):
        db.update_photo(filename, dict(_NO_BIRD_FIELDS,
                                       temp_jpeg_path=None))
    previews = _remove_previews(root, [photo.get("temp_jpeg_path")])
    result["previews_removed"] = previews
    # 无鸟照片的 sidecar JSON 由导出幂等删除
    return _sidecars_written(result, _finalize(db, root))


def cmd_species_rename(db: ReportDB, root: str, old_cn: Optional[str],
                       old_en: Optional[str], old_sci: Optional[str],
                       new_cn: Optional[str], new_en: Optional[str],
                       new_sci: Optional[str], apply: bool) -> dict:
    """
    整库整种改名为其他鸟种（复用 rename_species_everywhere 全链路）。

    参数:
    db (ReportDB): 已打开的库
    root (str): 照片库目录
    old_cn / old_en / old_sci (Optional[str]): 旧鸟种名（任一非空）
    new_cn / new_en / new_sci (Optional[str]): 新鸟种名（cn/en 至少一项）
    apply (bool): True 落盘，False 仅 dry-run

    返回:
    dict: 结果（status/photos/changed_detections/sidecars_written）
    """
    from core.sidecar_export import rename_species_in_sidecar

    result = {"status": "ok", "op": "species-rename", "root": root,
              "old": {"cn": old_cn, "en": old_en, "sci": old_sci},
              "new": {"cn": new_cn, "en": new_en, "sci": new_sci}}
    if not apply:
        scope = _preview_species_scope(db, old_cn, old_en, old_sci)
        result.update({"detections": scope["detections"],
                       "photos": scope["photos"]})
        return result

    files, det_count = db.rename_species_everywhere(
        old_cn, old_en, old_sci, new_cn, new_en, new_sci)
    for name in files:
        rename_species_in_sidecar(
            root, name,
            new_cn=new_cn, new_en=new_en, new_sci=new_sci,
            old_cn=old_cn, old_en=old_en, old_sci=old_sci)
    # 全库扫尾：main_species 里命中旧名但 DB 无同名检测框的照片（人工
    # 主鸟与 AI 框物种不一致）不在 files 里，逐文件同步碰不到
    swept = _sweep_main_species(root, old_cn, old_en, old_sci,
                                new_cn, new_en, new_sci)
    result.update({"photos": len(files), "changed_detections": det_count,
                   "mainspecies_swept": swept})
    return _sidecars_written(result, _finalize(db, root))


def cmd_species_wipe(db: ReportDB, root: str, old_cn: Optional[str],
                     old_en: Optional[str], old_sci: Optional[str],
                     apply: bool) -> dict:
    """
    整库整种识别删除（误检）：该鸟种检测框全软删，余下无存活框的
    照片归一无鸟态（仍是鸟的照片只删该鸟种，保持有鸟）。

    参数:
    db (ReportDB): 已打开的库
    root (str): 照片库目录
    old_cn / old_en / old_sci (Optional[str]): 旧鸟种名（任一非空）
    apply (bool): True 落盘，False 仅 dry-run

    返回:
    dict: 结果（status/photos/soft_deleted/to_no_bird 等）
    """
    from core.sidecar_export import mark_species_deleted_in_sidecar

    result = {"status": "ok", "op": "species-wipe", "root": root,
              "old": {"cn": old_cn, "en": old_en, "sci": old_sci}}
    if not apply:
        result.update(_preview_species_scope(db, old_cn, old_en, old_sci))
        return result

    files, det_count = db.soft_delete_species_everywhere(
        old_cn, old_en, old_sci)
    # 无余鸟照片归一：该照片存活框已被删光且仍有鸟标记 → 无鸟态
    conds, params = _species_match_conditions(old_cn, old_en, old_sci)
    live_cond = ("SELECT 1 FROM bird_detections d WHERE d.filename = "
                 "photos.filename AND d.deleted = 0")
    no_bird_rows = db._conn.execute(
        f"SELECT filename, temp_jpeg_path FROM photos WHERE has_bird = 1 "
        f"AND filename IN (SELECT DISTINCT filename FROM bird_detections "
        f"WHERE deleted = 0 AND ({' OR '.join(conds)})) "
        f"AND NOT EXISTS ({live_cond})", params).fetchall()
    for row in no_bird_rows:
        db.update_photo(row[0], dict(_NO_BIRD_FIELDS, temp_jpeg_path=None))
    previews = _remove_previews(root, [r[1] for r in no_bird_rows])
    for name in files:
        mark_species_deleted_in_sidecar(
            root, name, species_cn=old_cn, species_en=old_en,
            scientific_name=old_sci)
    # 全库扫尾：main_species 里命中旧名但 DB 无同名检测框的照片，移除
    # 其孤悬条目（同 rename 场景，见 _sweep_main_species）
    swept = _sweep_main_species(root, old_cn, old_en, old_sci)
    result.update({"photos": len(files), "soft_deleted": det_count,
                   "to_no_bird": len(no_bird_rows),
                   "previews_removed": previews,
                   "mainspecies_swept": swept})
    return _sidecars_written(result, _finalize(db, root))


def _require_db(root: str) -> Tuple[Optional[ReportDB], Optional[dict]]:
    """
    打开库并把失败包装为结果字典（main 里统一走 JSON/退出码）。

    返回:
    Tuple[Optional[ReportDB], Optional[dict]]: (库, 错误结果)；成功时
    错误侧为 None
    """
    db = _open_db(root)
    if db is None:
        return None, {"status": "error",
                      "reason": f"目录无 .superpicky/report.db: {root}"}
    return db, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="鸟种修正 CLI（默认 dry-run，--apply 执行；"
                    "--json 输出机器可读结果）")
    # 全局开关走 parent parser，子命令前后都能识别
    # Global flags via a parent parser so they parse in any position.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--apply", action="store_true",
                        help="真正执行（默认 dry-run 只统计）")
    common.add_argument("--json", action="store_true",
                        help="结果 JSON 走 stdout（日志转 stderr）")
    parser.add_argument("--apply", action="store_true",
                        help="真正执行（默认 dry-run 只统计）")
    parser.add_argument("--json", action="store_true",
                        help="结果 JSON 走 stdout（日志转 stderr）")
    sub = parser.add_subparsers(dest="op", required=True)

    p_photo = sub.add_parser("photo", parents=[common],
                             help="单张照片改种 / 清除识别")
    p_photo.add_argument("root", help="照片库目录（含 .superpicky/report.db）")
    p_photo.add_argument("filename", help="照片文件名或前缀（自动剥扩展名）")
    p_photo.add_argument("--to-cn", default=None, help="新鸟种中文名")
    p_photo.add_argument("--to-en", default=None, help="新鸟种英文名")
    p_photo.add_argument("--to-sci", default=None, help="新鸟种学名")
    p_photo.add_argument("--wipe", action="store_true",
                         help="清除该照片全部识别（照片不是鸟）")

    p_species = sub.add_parser("species", parents=[common],
                               help="整库整种改种 / 识别删除")
    p_species.add_argument("root", help="照片库目录")
    p_species.add_argument("--old-cn", default=None, help="旧鸟种中文名")
    p_species.add_argument("--old-en", default=None, help="旧鸟种英文名")
    p_species.add_argument("--old-sci", default=None, help="旧鸟种学名")
    p_species.add_argument("--to-cn", default=None, help="新鸟种中文名")
    p_species.add_argument("--to-en", default=None, help="新鸟种英文名")
    p_species.add_argument("--to-sci", default=None, help="新鸟种学名")
    p_species.add_argument("--wipe", action="store_true",
                           help="删除该鸟种全部识别（整种误检）")

    args = parser.parse_args()

    result: dict
    if args.op == "photo":
        filename = _strip_ext(args.filename)
        db, err = _require_db(args.root)
        if err:
            result = err
        elif args.wipe:
            result = cmd_photo_wipe(db, args.root, filename, args.apply)
        elif args.to_cn or args.to_en:
            result = cmd_photo_rename(
                db, args.root, filename, args.to_cn, args.to_en,
                args.to_sci, args.apply)
        else:
            result = {"status": "error",
                      "reason": "需要 --to-cn/--to-en 或 --wipe"}
        if db:
            db.close()
    else:
        old_names = [args.old_cn, args.old_en, args.old_sci]
        if not any(isinstance(v, str) and v.strip() for v in old_names):
            result = {"status": "error",
                      "reason": "需要 --old-cn/--old-en/--old-sci 之一"}
        else:
            db, err = _require_db(args.root)
            if err:
                result = err
            elif args.wipe:
                result = cmd_species_wipe(
                    db, args.root, args.old_cn, args.old_en, args.old_sci,
                    args.apply)
            elif args.to_cn or args.to_en:
                result = cmd_species_rename(
                    db, args.root, args.old_cn, args.old_en, args.old_sci,
                    args.to_cn, args.to_en, args.to_sci, args.apply)
            else:
                result = {"status": "error",
                          "reason": "需要 --to-cn/--to-en 或 --wipe"}
            if db:
                db.close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        _log(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == '__main__':
    sys.exit(main())
