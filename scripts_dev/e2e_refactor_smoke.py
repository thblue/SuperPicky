# -*- coding: utf-8 -*-
"""
重构端到端冒烟 / Post-refactor end-to-end smoke (throwaway sandbox).

链路 / Chain under test:
  SuperPicky process → sidecar JSON → BirdIndex scan → Web 页面/API
  → POST /api/fix → spb_rename_species.py 子进程 --apply
  → report.db + sidecar 更新 → BirdIndex 定向重扫

全部产物写入 scripts_dev/_backfill_sandbox/sp_refactor_smoke（gitignored），
不触碰任何真实照片库。
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

BI_ROOT = Path(r"G:/code/BirdIndex")
SP_ROOT = Path(r"G:/code/SuperPicky")
SB = SP_ROOT / "scripts_dev/_backfill_sandbox/sp_refactor_smoke"

sys.path.insert(0, str(BI_ROOT))

from indexer.config import load_config          # noqa: E402
from indexer.service import IndexerService      # noqa: E402
from web.app import create_app                  # noqa: E402
from fastapi.testclient import TestClient       # noqa: E402


def main() -> int:
    cfg = {
        "photo_roots": [
            {"id": "smoke", "name": "smoke", "path": str(SB / "library")}
        ],
        "index_db": str(SB / "birdindex.db"),
        "thumbnail_dir": str(SB / "thumbs"),
        "superpicky_dir": str(SP_ROOT),
    }
    cfg_path = SB / "bi_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    config = load_config(cfg_path)

    # 1) 增量扫描
    svc = IndexerService(config)
    svc.run_scan()
    conn = sqlite3.connect(config["index_db"])
    n = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    conn.close()
    assert n == 2, f"应索引 2 张照片，实际 {n}"
    print(f"[1/4] BirdIndex scan OK, photos={n}")

    app = create_app(config)
    with TestClient(app) as client:
        # 2) 页面与只读 API
        assert client.get("/").status_code == 200
        assert client.get("/species").status_code == 200
        species = client.get("/api/species").json()
        assert species, "/api/species 不应为空"
        print(f"[2/4] Web pages + /api/species OK ({len(species)} species)")

        r = client.get("/api/thumb/1/grid")
        assert r.status_code == 200, f"thumb 失败: {r.status_code}"
        print("[3/4] /api/thumb OK")

        # 3) 站内改种（子进程委托 → SuperPicky --apply → report.db/sidecar）
        photo_id = 1
        new_cn = "测试改种-黑雁"
        r = client.post("/api/fix", json={
            "op": "photo-rename", "photo_id": photo_id,
            "to": {"cn": new_cn, "en": "Barnacle Goose"},
        })
        assert r.status_code == 200, f"/api/fix 失败: {r.status_code} {r.text}"
        for _ in range(120):
            st = client.get("/api/fix/status").json()
            if not st.get("running"):
                break
            time.sleep(1)
        print(f"    fix status: {json.dumps(st, ensure_ascii=False)[:300]}")
        assert not st.get("running"), "fix 超时未完成"

        # 4) 写回验证：report.db 主鸟种已改 + 定向重扫后索引一致
        report_db = SB / "library/.superpicky/report.db"
        conn = sqlite3.connect(str(report_db))
        row = conn.execute(
            "SELECT bird_species_cn FROM photos WHERE filename='IMG_9001'"
        ).fetchone()
        conn.close()
        assert row and row[0] == new_cn, f"report.db 未改种: {row}"
        conn = sqlite3.connect(config["index_db"])
        hit = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE species_cn=?", (new_cn,)
        ).fetchone()[0]
        conn.close()
        assert hit == 1, f"重扫后索引未同步: {hit}"
        print("[4/4] /api/fix → spb_rename_species --apply → 重扫 同步 OK")

    print("E2E SMOKE: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
