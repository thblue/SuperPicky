# -*- coding: utf-8 -*-
"""
sidecar JSON 契约快照测试 / Sidecar JSON Contract Snapshot Test

守护 dev-docs/INTERFACE_CONTRACTS.md §1：在临时照片库上执行一次真实的
report.db → sidecar 导出，对 JSON 的关键结构与定位锚点字段做快照断言。
BirdIndex 的 indexer/contract.py 与 scan.py 依赖这些字段；任何偏离都会
在本测试先行暴露，而不是等网站扫描静默出错。

Guards §1 of the frozen-interface list: runs a real report.db → sidecar
export on a throwaway library and snapshots the key structure and the
V5.3 location-anchor fields that BirdIndex depends on.
"""
from __future__ import annotations

import json

import pytest

from core.sidecar_export import export_directory_sidecars

pytestmark = pytest.mark.contract

# 冻结的顶层结构（_export_stamp 等实现细节字段允许存在）/ Frozen top-level
# sections; implementation-detail keys like _export_stamp are allowed on top.
EXPECTED_TOP_KEYS = {"schema_version", "photo", "detections", "edits", "processing"}
EXPECTED_SCHEMA_VERSION = 1


def _read_sidecar(library, filename: str) -> dict:
    sidecar = library / ".superpicky" / "meta" / f"{filename}.json"
    assert sidecar.exists(), f"sidecar 未导出: {sidecar}"
    with open(sidecar, "r", encoding="utf-8") as f:
        return json.load(f)


def test_export_writes_only_bird_photos(photo_library):
    """V5.5 契约：无鸟照片不导出 / No-bird photos are not exported."""
    db, library = photo_library
    written = export_directory_sidecars(db, str(library), log=lambda *_: None)
    assert written == 1, "只有 has_bird=1 的照片应产生 sidecar"
    stale = library / ".superpicky" / "meta" / "IMG_0002.CR3.json"
    assert not stale.exists(), "无鸟照片不得留下 sidecar（V5.5 幂等删除）"


def test_sidecar_top_level_snapshot(photo_library):
    """顶层结构与 schema_version 快照 / Top-level structure snapshot."""
    db, library = photo_library
    export_directory_sidecars(db, str(library), log=lambda *_: None)
    payload = _read_sidecar(library, "IMG_0001.CR3")
    missing = EXPECTED_TOP_KEYS - set(payload)
    assert not missing, f"sidecar 顶层缺少冻结字段: {missing}"
    assert payload["schema_version"] == EXPECTED_SCHEMA_VERSION


def test_v53_location_anchors_frozen(photo_library):
    """
    V5.3 定位锚点冻结：library_path 为库内相对路径，filename 带扩展名。

    V5.3 anchors are frozen: library_path is the library-relative current
    location; filename keeps its extension. BirdIndex 定位照片全靠这两个字段。
    """
    db, library = photo_library
    export_directory_sidecars(db, str(library), log=lambda *_: None)
    photo = _read_sidecar(library, "IMG_0001.CR3")["photo"]
    assert photo["filename"] == "IMG_0001.CR3", "photo.filename 必须带扩展名"
    assert photo["library_path"] == "IMG_0001.CR3", (
        "photo.library_path 必须是相对库根的当前真实位置（V5.3 契约锚点）"
    )
    # relative_path（存档语义）与 preview_path（可显示 JPEG）字段必须存在，
    # 值允许为 None（未移动/无预览缓存时）。
    assert "relative_path" in photo
    assert "preview_path" in photo


def test_human_edits_survive_reexport(photo_library):
    """
    人工 edits 原样保留契约：手改 JSON 后重导出不丢编辑。

    Manual `edits` must survive re-export (the round-trip guarantee the
    website's fix workflow and spb_sync_edits rely on).
    """
    db, library = photo_library
    export_directory_sidecars(db, str(library), log=lambda *_: None)
    sidecar = library / ".superpicky" / "meta" / "IMG_0001.CR3.json"
    with open(sidecar, "r", encoding="utf-8") as f:
        payload = json.load(f)
    edits = payload.get("edits") or []
    edits.append({"actor": "human", "action": "rename", "note": "契约测试"})
    payload["edits"] = edits
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    export_directory_sidecars(db, str(library), log=lambda *_: None)
    with open(sidecar, "r", encoding="utf-8") as f:
        after = json.load(f)
    assert after.get("edits"), "重导出不得丢弃人工 edits（对外契约 §1）"
