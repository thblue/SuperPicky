#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可疑鸟种探针：分布存疑 + 单张稀有（batch-identify skill 第 6 节的自动清单生成器）。

把 skill 第 6 节「可疑鸟种目视复核」里原本靠人工判断的两个必检规则机械化：

- 规则 A「分布存疑」：照片有 GPS，但其鸟种的 class_id 在 GBIF 分布库
  （birdid/data/geo_distribution.db）的本格 / 3x3 邻域 / 国家三层**全部无
  记录**——即识别时跌穿全部分布层、靠无过滤兜底认出的情形（博物馆标本、
  图册/明信片翻拍、跨半球误判的重灾区）。无 GPS 照片不做此判定。
- 规则 B「单张稀有」：该库内某鸟种照片数 ≤ --max-species-photos（默认 2，
  与 6.4「≤2 张有星种全量目检」同口径）且至少一张有星（rating ≥ 1）。

两类命中都会用 spb_review.py 生成带框标注的复核图（bbox 双坐标系由其内部
处理），逐张交人工/agent 目视定案——本工具只产清单与图，不改任何数据
（report.db / sidecar / 照片文件零写入，除 spb_review 的复核图输出目录）。

用法:
    python scripts_dev/spb_check_suspects.py <库根目录>... [--max-species-photos 2]
        [--no-crops] [--out <目录>] [--report <md路径>]

清单生成后按 skill 6.4 的定案手法处理：能确认的保留 / 改种
（spb_rename_species photo），明显无鸟或标本图册的
`photo --wipe --apply` 抹标，拿不准的保留原判进汇报。

Suspect-species probe for the batch-identify skill: mechanizes the two
mandatory review rules (species absent from ALL GBIF distribution tiers
despite GPS; species with <=2 photos in the library). Generates an
annotated review image per suspect via spb_review.py. Read-only against
report.db / sidecars / photo files.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts_dev.backfill_net_image_gate import find_library_dirs  # noqa: E402
from tools.report_db import ReportDB  # noqa: E402

# 复核图默认输出根（沙盒内，gitignored）
# Default output root for review crops (inside the sandbox, gitignored).
_DEFAULT_OUT = os.path.join('scripts_dev', '_backfill_sandbox', 'suspect_check')

_GEO_DB = os.path.join('birdid', 'data', 'geo_distribution.db')
_REF_DB = os.path.join('birdid', 'data', 'bird_reference.sqlite')


def load_class_ids(species_names: Set[str]) -> Dict[str, List[int]]:
    """
    鸟种中文名 → 模型 class_id 列表（识别管线的同源映射，BirdCountInfo 表）。

    参数:
    species_names (Set[str]): 去重的鸟种中文名集合

    返回:
    Dict[str, List[int]]: 种名 → class_id 列表（含近缘同名字段的全匹配优先）

    Map species CN names to model class ids via the same reference table
    the identification pipeline uses.
    """
    import sqlite3
    mapping: Dict[str, List[int]] = {}
    conn = sqlite3.connect(_REF_DB)
    try:
        cur = conn.cursor()
        for name in species_names:
            cur.execute(
                "SELECT model_class_id FROM BirdCountInfo "
                "WHERE chinese_simplified = ? AND model_class_id IS NOT NULL",
                (name,))
            ids = [r[0] for r in cur.fetchall()]
            if not ids:
                cur.execute(
                    "SELECT model_class_id FROM BirdCountInfo "
                    "WHERE chinese_simplified LIKE ? AND model_class_id IS NOT NULL "
                    "LIMIT 3",
                    (f'%{name}%',))
                ids = [r[0] for r in cur.fetchall()]
            if ids:
                mapping[name] = ids
    finally:
        conn.close()
    return mapping


def geo_presence(class_ids: List[int], lat: float, lon: float,
                 country: Optional[str]) -> Dict[str, bool]:
    """
    查一个鸟种在照片坐标处的三层分布命中情况（本格 / 3x3 邻域 / 国家）。

    参数:
    class_ids (List[int]): 该种名下全部候选 class_id（近缘同名取并集）
    lat (float): 纬度
    lon (float): 经度
    country (Optional[str]): ISO 国家代码（reverse_geocoder 反解）

    返回:
    Dict[str, bool]: {'cell':..., 'neighborhood':..., 'country':...}，True=有记录

    Check a species' presence in the cell / 3x3 neighborhood / country
    tiers of the GBIF distribution DB at the photo's coordinates.
    """
    import sqlite3
    lat_bin, lon_bin = int(lat // 1), int(lon // 1)
    cell = (max(-90, min(89, lat_bin)) + 90) * 360 + (max(-180, min(179, lon_bin)) + 180)
    neigh = [(la + 90) * 360 + ((lo + 180) % 360)
             for la in (lat_bin - 1, lat_bin, lat_bin + 1)
             for lo in (lon_bin - 1, lon_bin, lon_bin + 1)]
    conn = sqlite3.connect(_GEO_DB)
    try:
        cur = conn.cursor()
        hit_cell = hit_nb = hit_cty = False
        ph = ','.join('?' * len(neigh))
        for cid in class_ids:
            if not hit_cell and cur.execute(
                    "SELECT 1 FROM cell_species WHERE cell_id=? AND class_id=?",
                    (cell, cid)).fetchone():
                hit_cell = True
            if not hit_nb and cur.execute(
                    f"SELECT 1 FROM cell_species WHERE class_id=? AND cell_id IN ({ph})",
                    [cid] + neigh).fetchone():
                hit_nb = True
            if country and not hit_cty and cur.execute(
                    "SELECT 1 FROM country_species WHERE country=? AND class_id=?",
                    (country, cid)).fetchone():
                hit_cty = True
            if hit_cell and hit_nb and (hit_cty or not country):
                break
    finally:
        conn.close()
    return {'cell': hit_cell, 'neighborhood': hit_nb, 'country': hit_cty}


def collect_rows(photo_dir: str) -> List[Dict]:
    """
    拉一个库里全部有鸟且有种的 photos 行（含 video 封面行）。

    参数:
    photo_dir (str): 照片库目录

    返回:
    List[Dict]: photos 行列表

    Fetch all identified-bird photo rows of one library.
    """
    db = ReportDB(photo_dir)
    try:
        rows = [dict(r) for r in db.get_all_photos()
                if r.get('has_bird') and (r.get('bird_species_cn')
                                          or r.get('bird_species_en'))]
        # V2 盲区修复：种名只在 bird_detections（低置信未采纳行）的
        # 照片对 BirdIndex 物种页同样可见，规则 B 必须覆盖——用检测行
        # 最高置信种名回填 photos 行的显示种名。
        have = {r['filename'] for r in rows}
        dets = db.get_all_detections(include_polygon=False)
        best: dict = {}
        for d in dets:
            fn = d.get('filename')
            sp = d.get('species_cn')
            if not fn or fn in have or not sp or d.get('deleted'):
                continue
            conf = d.get('species_confidence') or 0.0
            if fn not in best or conf > best[fn][0]:
                best[fn] = (conf, sp)
        for fn, (conf, sp) in best.items():
            for r in rows_by_fn(db, fn):
                r['bird_species_cn'] = sp
                rows.append(r)
        return rows
    finally:
        db.close()


def rows_by_fn(db, filename):
    row = db.get_photo(filename)
    return [dict(row)] if row else []


def geo_presence_country(class_ids: List[int], country: str) -> bool:
    """
    查一个鸟种是否在某国家层有 GBIF 记录（无 GPS 照片的规则 A2 用）。

    参数:
    class_ids (List[int]): 该种名下全部候选 class_id
    country (str): ISO 国家代码

    返回:
    bool: True=国家层有记录（不放行探针）；False=国家层全无记录

    Check whether a species has any GBIF record in a country tier
    (rule A2 for GPS-less photos).
    """
    import sqlite3
    conn = sqlite3.connect(_GEO_DB)
    try:
        cur = conn.cursor()
        ph = ','.join('?' * len(class_ids))
        row = cur.execute(
            f"SELECT 1 FROM country_species WHERE country=? "
            f"AND class_id IN ({ph}) LIMIT 1",
            [country] + list(class_ids)).fetchone()
        return row is not None
    finally:
        conn.close()


def find_suspects(photo_dir: str, max_species_photos: int,
                  country_resolver,
                  no_gps_country: Optional[str] = 'CN') -> Tuple[List[Dict], List[Dict]]:
    """
    对一个库执行两条探针规则，返回 (分布存疑, 单张稀有) 两组命中行。

    参数:
    photo_dir (str): 照片库目录
    max_species_photos (int): 规则 B 的种照片数上限
    country_resolver: GPS → ISO 国家代码函数（可返回 None）
    no_gps_country (Optional[str]): 无 GPS 照片规则 A2 的国家码
        （默认 'CN'，与识别管线 DEFAULT_COUNTRY_NO_GPS 同源；None=跳过 A2）

    返回:
    Tuple[List[Dict], List[Dict]]: 各行附加 _rules/_geo/_reasons 字段

    Run both probe rules on one library.
    """
    rows = collect_rows(photo_dir)
    species_names = {r['bird_species_cn'] for r in rows if r.get('bird_species_cn')}
    class_map = load_class_ids(species_names) if species_names else {}

    # 规则 A1：有 GPS 且三层全无记录；规则 A2：无 GPS 按国家码只查国家层
    #（与识别管线 DEFAULT_COUNTRY_NO_GPS 同源默认 CN，旅外库用
    # --no-gps-country 覆盖；只查国家层是因为无格心坐标算不了本格/邻域）
    geo_suspects: List[Dict] = []
    for r in rows:
        lat, lon = r.get('gps_latitude'), r.get('gps_longitude')
        ids = class_map.get(r.get('bird_species_cn') or '', [])
        if not ids:
            continue
        if lat is not None and lon is not None:
            country = country_resolver(lat, lon)
            presence = geo_presence(ids, lat, lon, country)
            if not (presence['cell'] or presence['neighborhood']
                    or presence['country']):
                r['_rules'] = ['geo']
                r['_geo'] = presence
                r['_country'] = country
                r['_geo_basis'] = 'gps'
                geo_suspects.append(r)
        elif no_gps_country:
            if not geo_presence_country(ids, no_gps_country):
                r['_rules'] = ['geo']
                r['_geo'] = {'cell': None, 'neighborhood': None,
                             'country': False}
                r['_country'] = no_gps_country
                r['_geo_basis'] = 'no-gps'
                geo_suspects.append(r)

    # 规则 B：种照片数 ≤ 上限且至少一张有星（rating ≥ 1）
    by_species: Dict[str, List[Dict]] = {}
    for r in rows:
        by_species.setdefault(r['bird_species_cn'], []).append(r)
    rare_suspects: List[Dict] = []
    for name, group in by_species.items():
        if len(group) > max_species_photos:
            continue
        if not any((r.get('rating') or 0) >= 1 for r in group):
            continue
        for r in group:
            rules = r.get('_rules') or []
            r['_rules'] = rules + (['rare'] if 'geo' not in rules else [])
            if 'geo' not in rules:
                rare_suspects.append(r)
            else:
                r['_rules'] = rules + ['rare']
    return geo_suspects, rare_suspects


def generate_crops(suspects: List[Dict], photo_dir: str, out_dir: str,
                   python_exe: str) -> List[str]:
    """
    用 spb_review.py 给命中行生成带框复核图（bbox 双坐标系由其处理）。

    参数:
    suspects (List[Dict]): 命中行
    photo_dir (str): 照片库目录
    out_dir (str): 复核图输出目录
    python_exe (str): 当前解释器路径（子进程用同环境）

    返回:
    List[str]: 成功生成的复核图路径

    Generate annotated review images for suspects via spb_review.py.
    """
    # 定位待检文件本体（current_path / original_path 皆可）
    paths: List[str] = []
    for r in suspects:
        for key in ('current_path', 'original_path'):
            rel = r.get(key)
            if not rel:
                continue
            p = os.path.normpath(os.path.join(photo_dir, rel))
            if os.path.exists(p):
                paths.append(p)
                break
    if not paths:
        return []
    os.makedirs(out_dir, exist_ok=True)
    review_script = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'spb_review.py')
    try:
        proc = subprocess.run(
            [python_exe, review_script, *paths, '--out', out_dir],
            capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"    ⚠️ spb_review 退出码 {proc.returncode}: "
                  f"{(proc.stderr or '').strip()[-300:]}")
    except subprocess.TimeoutExpired:
        print("    ⚠️ spb_review 超时（600s），本库裁剪中止")
        return []
    crops = []
    for name in os.listdir(out_dir):
        if name.endswith('_review.jpg'):
            crops.append(os.path.join(out_dir, name))
    return sorted(crops)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='可疑鸟种探针：分布存疑（GBIF 三层全无）+ 单张稀有（种照片≤N）')
    parser.add_argument('roots', nargs='+',
                        help='照片库根目录（递归查找 .superpicky/report.db）')
    parser.add_argument('--max-species-photos', type=int, default=2,
                        help='规则 B：种照片数上限（默认 2，与 skill 6.4 同口径）')
    parser.add_argument('--no-crops', action='store_true',
                        help='只出清单，不生成复核图')
    parser.add_argument('--no-gps-country', default='CN',
                        help='无 GPS 照片规则 A2 的国家码（默认 CN，'
                             '与识别管线 DEFAULT_COUNTRY_NO_GPS 同源；'
                             '旅外无 GPS 库按实际国家传，如 --no-gps-country JP；'
                             '传 none 跳过 A2）')
    parser.add_argument('--rules', default='geo,rare',
                        help='只输出/裁剪指定规则命中（逗号分隔：geo,rare；'
                             '默认全部）')
    parser.add_argument('--out', default=None,
                        help='复核图输出根目录（默认 scripts_dev/_backfill_sandbox/suspect_check）')
    args = parser.parse_args()

    libs = find_library_dirs(args.roots)
    if not libs:
        print("❌ 未找到任何含 .superpicky/report.db 的照片库")
        return 1

    # GPS 反解国家：复用识别管线同一个 reverse_geocoder 封装
    from birdid.bird_identifier import _resolve_country_code_from_gps

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_root = args.out or _DEFAULT_OUT
    no_gps_country = (args.no_gps_country or '').strip().upper() or None
    report_lines: List[str] = [
        f"# 可疑鸟种清单 {stamp}",
        "规则 A=分布存疑（有 GPS：GBIF 本格/邻域/国家三层全无；"
        f"无 GPS：国家层全无，国家码={no_gps_country or '跳过'}）",
        f"规则 B=单张稀有（种照片数≤{args.max_species_photos} 且有星）",
        "",
    ]
    total_geo = total_rare = 0
    python_exe = sys.executable

    for lib in libs:
        geo, rare = find_suspects(lib, args.max_species_photos,
                                  _resolve_country_code_from_gps,
                                  no_gps_country)
        want = {x.strip() for x in args.rules.split(',') if x.strip()}
        if 'geo' not in want:
            geo = []
        if 'rare' not in want:
            rare = []
        if not geo and not rare:
            continue
        lib_name = os.path.basename(lib.rstrip('\\/')) or lib
        print(f"📋 {lib}: 分布存疑 {len(geo)}，单张稀有 {len(rare)}")
        report_lines.append(f"## {lib}")
        crops_dir = os.path.join(out_root, stamp, lib_name)
        crops = [] if args.no_crops else generate_crops(
            geo + rare, lib, crops_dir, python_exe)
        crop_by_prefix = {
            os.path.basename(c).replace('_review.jpg', ''): c for c in crops}
        for r in geo + rare:
            rules = '+'.join(r.get('_rules') or [])
            pres = r.get('_geo') or {}
            if 'geo' in (r.get('_rules') or []):
                if r.get('_geo_basis') == 'no-gps':
                    geo_txt = (f"国家层全无({r.get('_country') or '?'},无GPS)")
                else:
                    geo_txt = f"三层全无({r.get('_country') or '?'})"
            else:
                geo_txt = ""
            if args.no_crops:
                crop = '（未生成）'
            else:
                crop = crop_by_prefix.get(r['filename'], '（裁剪生成失败）')
            line = (f"- [{rules}] {r['filename']} | "
                    f"{r.get('bird_species_cn')} "
                    f"(conf {r.get('birdid_confidence') or 0:.0f}%, "
                    f"{r.get('rating')}★) | {geo_txt} | {crop}")
            print("   " + line)
            report_lines.append(line)
            if pres and 'geo' in (r.get('_rules') or []):
                if r.get('_geo_basis') == 'no-gps':
                    report_lines.append(
                        f"  - 分布详情: 无GPS按国家码 {r.get('_country')} 查，"
                        f"国家层无记录")
                else:
                    report_lines.append(
                        f"  - 分布详情: 本格={pres['cell']} 邻域={pres['neighborhood']} "
                        f"国家={pres['country']}")
        report_lines.append("")
        total_geo += len(geo)
        total_rare += len(rare)

    report_path = os.path.join(out_root, stamp, 'report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"\n🏁 命中：分布存疑 {total_geo}，单张稀有 {total_rare}")
    print(f"📝 清单: {report_path}")
    if not args.no_crops:
        print("下一步：逐张目视复核裁剪图 → 按 skill 6.4 定案（保留/改种/photo --wipe）")
    return 0


if __name__ == '__main__':
    sys.exit(main())
