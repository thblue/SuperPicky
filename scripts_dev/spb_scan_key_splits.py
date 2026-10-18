#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spb_scan_key_splits — 键分裂巡检：中英名与学名打架的行全库扫描。

站点（BirdIndex）按 (中文名, 学名) 分组展示鸟种。当 photos 行中英名与
主检测行（is_selected=1）的 scientific_name 属于不同鸟种时——历史管线
写偏、或旧版 spb_rename_species photo 改种只同步同名检测行所致——同一
鸟种会被劈成两个条目。本脚本以 bird_reference.sqlite 为权威映射，把每
行选中检测框的学名回查参考库中文名，与 photos 行中文名比对，列出全部
不一致的组合（只读，不改任何数据）。

修复方式：
- photos 行错、检测行对 → spb_rename_species photo 按检测行物种改写
  （2026-10 起工具会连主框一并同步）；
- photos 行对、检测行错 → 旧版工具同步不到该行，用直接 SQL 更新选中
  检测行三段命名后 wal_checkpoint(TRUNCATE)，再 reexport_sidecars。

用法:
    python scripts_dev/spb_scan_key_splits.py "<照片库目录>"

参数:
    directory: 照片库目录（其下应有 .superpicky/report.db）

输出:
    每个 (photos中文名, 英文名, 学名, 参考库中文名, 张数) 不一致组合
    一行；退出码 0=无分裂，1=发现分裂（可作 CI/巡检判据）。
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
REF_DB = REPO / 'birdid' / 'data' / 'bird_reference.sqlite'


def load_reference_map(ref_db: Path) -> Dict[str, str]:
    """
    从鸟种参考库构建 学名 → 参考库中文名 映射。

    参数:
    ref_db (Path): bird_reference.sqlite 路径

    返回:
    Dict[str, str]: 学名（去空白）→ 中文简体名
    """
    conn = sqlite3.connect(str(ref_db))
    cur = conn.cursor()
    mapping: Dict[str, str] = {}
    for cn, sci in cur.execute(
            'SELECT chinese_simplified, scientific_name FROM BirdCountInfo'):
        if sci:
            mapping[sci.strip()] = cn
    conn.close()
    return mapping


def scan(directory: Path, name2cn: Dict[str, str]) -> List[Tuple[str, str, str, str, int]]:
    """
    扫描一个照片库：列出中英名与学名打架的 (组合, 张数)。

    参数:
    directory (Path): 照片库目录
    name2cn (Dict[str, str]): 学名 → 参考库中文名 映射

    返回:
    List[Tuple]: (photos中文名, 英文名, 学名, 参考库中文名, 张数)，
    只含有分歧的组合；空列表 = 无分裂。
    """
    db = directory / '.superpicky' / 'report.db'
    conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT p.bird_species_cn, p.bird_species_en,
               d.scientific_name, COUNT(*)
        FROM photos p
        JOIN bird_detections d ON d.filename = p.filename AND d.is_selected = 1
        WHERE p.has_bird = 1
          AND p.bird_species_cn IS NOT NULL AND p.bird_species_cn != ''
          AND d.scientific_name IS NOT NULL AND d.scientific_name != ''
        GROUP BY p.bird_species_cn, p.bird_species_en, d.scientific_name
        ORDER BY p.bird_species_cn""").fetchall()
    conn.close()
    bad = []
    for cn, en, sci, n in rows:
        ref_cn = name2cn.get(sci.strip())
        if ref_cn is not None and ref_cn != cn:
            bad.append((cn, en, sci, ref_cn, n))
    return bad


def main() -> int:
    """
    命令行入口：扫描指定照片库的键分裂残留。

    返回:
    int: 0=无分裂；1=发现分裂；2=目录/参考库缺失
    """
    ap = argparse.ArgumentParser(
        description='键分裂巡检：中英名与学名打架的行全库扫描（只读）')
    ap.add_argument('directory', help='照片库目录（含 .superpicky/report.db）')
    args = ap.parse_args()

    directory = Path(args.directory)
    if not (directory / '.superpicky' / 'report.db').is_file():
        print(f'❌ 未找到 report.db: {directory / ".superpicky" / "report.db"}')
        return 2
    if not REF_DB.is_file():
        print(f'❌ 未找到鸟种参考库: {REF_DB}')
        return 2

    name2cn = load_reference_map(REF_DB)
    bad = scan(directory, name2cn)
    if not bad:
        print('✅ 无键分裂残留（中英名与学名全部一致）')
        return 0
    print(f'⚠️ 发现 {len(bad)} 组键分裂（中英名与学名打架）:')
    for cn, en, sci, ref_cn, n in bad:
        print(f'  photos「{cn}」({en}) + 检测行学名「{sci}」'
              f'(=参考库「{ref_cn}」) × {n}张')
    print('修复: photos 行错→按检测行物种 photo 改种；'
          '检测行错→直接 SQL 更新选中行三段命名 + checkpoint + reexport。')
    return 1


if __name__ == '__main__':
    sys.exit(main())
