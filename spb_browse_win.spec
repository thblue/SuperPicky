# -*- coding: utf-8 -*-
"""
spb_browse_win.spec — SPBBrowse.exe（结果浏览器独立封装）打包配置

与主程序 SuperPicky_win64.spec 的区别 / differences from the main spec:
  - 入口 spb_browse.py（结果浏览器，不跑 AI 处理流水线）
  - 不打包 models / ultralytics / torch / exiftool 二进制，
    体积从数 GB 级降到约两百 MB 级
  - 打星写 XMP 遵守全局 metadata_write_mode（none 时跳过，见
    exiftool_manager.set_rating_and_pick）；本 exe 面向「结果全部落在
    report.db」的工作流，不带 exiftools_win，模块本身仍打包保证 import 完整
  - 保留 ioc（鸟种搜索 birdname.db）

datas 全部落在 onedir 的 _internal/ 下，与 config.get_install_scoped_resource_path /
sys._MEIPASS 的 frozen 定位逻辑（config.py）匹配。

Build spec for SPBBrowse.exe — the standalone results browser. Slices the
browser's dependency chain only: no torch/ultralytics/models, but keeps
exiftools_win (XMP rating writes) and ioc (birdname search database).
"""

import os
import sys

sys.path.append(os.path.abspath('.'))

base_path = os.path.abspath('.')

# rawpy 数据文件（CR3/NEF 等 RAW 解码兜底，多鸟编辑器加载纯 RAW 原片用）
# rawpy data files for the torch-free RAW decode fallback.
from PyInstaller.utils.hooks import collect_data_files

rawpy_datas = collect_data_files('rawpy')

# 浏览器实际需要的资源目录（落入 _internal/）
# Resources the browser actually needs (landed under _internal/).
all_datas = [
    # 图片/图标资源（icon_utils 按需加载）
    (os.path.join(base_path, 'img'), 'img'),
    # 国际化语言包（tools.i18n 从 _MEIPASS/locales 读取）
    (os.path.join(base_path, 'locales'), 'locales'),
    # 鸟种搜索数据库（birdname_search_widget / bird_species_edit_dialog）
    (os.path.join(base_path, 'ioc'), 'ioc'),
]
all_datas.extend(rawpy_datas)

a = Analysis(
    ['spb_browse.py'],
    pathex=[base_path],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        # ── 第三方 / third-party ──
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # 多鸟编辑器 / 裁剪工作室顶层使用 cv2+numpy
        # cv2 + numpy used at top level by the multibird editor / crop studio
        'cv2',
        'numpy',
        # RAW 直读兜底（_read_image 第三级，绕开 bird_identifier 的 torch）
        # torch-free RAW decode fallback (tier 3 of _read_image)
        'rawpy',
        # ── 入口脚本函数体内延迟导入（PyInstaller 静态分析发现不了）──
        # lazily imported inside spb_browse.main()
        'ui.browse_launcher_dialog',
        'ui.results_browser_window',
        # ── 结果浏览器依赖闭包中的延迟导入模块 ──
        # lazily imported modules across the browser dependency closure
        'tools.merged_report_db',      # open_directory 合并多目录模式
        'tools.report_db',             # merged_report_db 函数内导入
        'tools.exiftool_manager',      # 打星写 XMP（mode=none 时内部跳过；不打包 exef 二进制）
        'tools.utils',
        'ui.multibird_editor_dialog',  # 右键「多鸟编辑」
        'ui.bird_species_edit_dialog', # 右键「修改鸟种」
        'ui.birdname_search_widget',   # 鸟种搜索（birdname.db）
        'ui.crop_studio',              # 裁剪建议
        'ui.custom_dialogs',
        'ui.submission_review_dialog',
        'birdid.bird_database_manager',# 鸟种数据库管理器
        'core.correction_tracker',     # 鸟种纠错记录
        'core.rating_mover',           # 打星后按星级移动文件
        'core.sidecar_export',         # sidecar 导出
        'core.species_recall',         # 稀有鸟召回重算
        'spb_review',                  # multibird_editor_dialog 函数内导入
        'core.folder_layout',          # advanced_config 函数内导入
        # ── 浏览器顶层依赖（自动可发现，显式列出保险）──
        # top-level deps, listed explicitly for safety
        'tools.i18n',
        'tools.file_utils',
        'core.rarity_tier',
        'core.recursive_scanner',
        'advanced_config',
        'config',
        'constants',
        'ui.filter_panel',
        'ui.thumbnail_grid',
        'ui.detail_panel',
        'ui.fullscreen_viewer',
        'ui.comparison_viewer',
        'ui.icon_utils',
        'ui.styles',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_cv2.py'] if os.path.exists('pyi_rth_cv2.py') else [],
    # 浏览器链路不使用的重型依赖全部排除，显著缩减体积
    # Exclude heavy deps unused by the browser to shrink the bundle.
    # 注意：rawpy 必须保留（多鸟编辑器读 RAW 原片；bird_identifier 因 torch
    # 被排除而不可用，rawpy 是轻量环境唯一的 RAW 解码路径）
    # Note: rawpy must stay — it is the only RAW decoder left once
    # bird_identifier (torch-gated) is excluded.
    excludes=[
        'torch', 'torchvision', 'ultralytics', 'timm',
        'matplotlib',
        'flask', 'cryptography',
        'imageio', 'pillow_heif', 'pi_heif',
        'imagehash', 'pywt',
        'PyQt5', 'PyQt6', 'tkinter',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Windows 使用高精度 icon.ico（与主程序同源）
_icon_ico = os.path.join(base_path, 'img', 'icon.ico')
_exe_icon = _icon_ico if (sys.platform == 'win32' and os.path.exists(_icon_ico)) else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SPBBrowse',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # 二进制对 UPX 压缩敏感（cv2/PySide6），保持关闭以稳为主
    # Keep UPX off for binary stability (cv2/PySide6).
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SPBBrowse',
)
