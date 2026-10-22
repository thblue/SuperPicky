#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky 常量定义
统一管理全局常量，避免重复定义
"""

import os

# 应用版本号
APP_VERSION = "4.5.0"


# 评分对应的文件夹名称映射（向后兼容，默认中文）
RATING_FOLDER_NAMES = {
    3: "3星_优选",
    2: "2星_良好",
    1: "1星_普通",
    0: "0星_放弃",
    -1: "0星_放弃"  # 无鸟照片也放入0星目录
}

# 英文文件夹名称
RATING_FOLDER_NAMES_EN = {
    3: "3star_excellent",
    2: "2star_good",
    1: "1star_average",
    0: "0star_reject",
    -1: "0star_reject"
}

def get_rating_folder_names():
    """
    获取当前语言的评分文件夹名称映射

    Returns:
        dict: {评分: 文件夹名称}
    """
    try:
        from tools.i18n import get_i18n
        i18n = get_i18n()
        if i18n.current_lang.startswith('en'):
            return RATING_FOLDER_NAMES_EN.copy()
    except Exception:
        pass
    return RATING_FOLDER_NAMES.copy()

def get_rating_folder_name(rating: int) -> str:
    """
    获取指定评分的文件夹名称（根据当前语言）

    Args:
        rating: 评分 (-1 to 3)

    Returns:
        str: 文件夹名称
    """
    folders = get_rating_folder_names()
    return folders.get(rating, folders.get(0, "0star_reject"))

# 支持的 RAW 文件扩展名（小写）
RAW_EXTENSIONS = ['.nef', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2', '.pef', '.dng', '.3fr', '.iiq']

# 元数据强制写 XMP 侧车（不重写 RAW 本体）的专有 RAW 格式。
# DNG 除外：Lightroom/C1 惯例是 DNG 读写内嵌 XMP、不认侧车；专有 RAW 则
# 侧车优先。实测嵌入式重写 RAW 本体 ~190ms/张 vs 侧车 ~10ms/张（A/B 快 33%），
# 且不动本体更安全（曾有 ExFAT overwrite_original 丢原图事故）。
# Proprietary RAW formats whose metadata is force-written to XMP sidecars
# (never rewriting the RAW body). DNG is excluded: LR/C1 convention reads
# embedded XMP in DNG and ignores sidecars, while proprietary RAW prefers
# sidecars. Measured: embedded body rewrite ~190ms/photo vs sidecar ~10ms
# (33% faster end-to-end), and leaving the body untouched is safer.
SIDECAR_RAW_EXTENSIONS = [ext for ext in RAW_EXTENSIONS if ext != '.dng']

# 支持的 HEIF 文件扩展名（小写）- Sony HIF / Apple HEIC 等
HEIF_EXTENSIONS = ['.hif', '.heif', '.heic']

# 支持的 JPG 文件扩展名（小写）
JPG_EXTENSIONS = ['.jpg', '.jpeg']

# 所有支持的图片扩展名（用于文件查找，包含大小写）
IMAGE_EXTENSIONS = (
    [ext.lower() for ext in RAW_EXTENSIONS] +
    [ext.upper() for ext in RAW_EXTENSIONS] +
    [ext.lower() for ext in HEIF_EXTENSIONS] +
    [ext.upper() for ext in HEIF_EXTENSIONS] +
    [ext.lower() for ext in JPG_EXTENSIONS] +
    [ext.upper() for ext in JPG_EXTENSIONS]
)

# V4.3 Phase 1: 视频分析支持的扩展名
# macOS 优先：OpenCV 原生支持 + AVFoundation 后端解码
# Supported video extensions for video analysis (Phase 1, macOS first)
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.m4v']
VIDEO_EXTENSIONS_ALL = (
    [ext.lower() for ext in VIDEO_EXTENSIONS] +
    [ext.upper() for ext in VIDEO_EXTENSIONS]
)

# V-cover（视频封面）保留文件名后缀：
# 每个视频的代表性帧会落成一张真实 JPEG（<视频名>_vcover.jpg），作为一条普通
# photos 记录进入 report.db，视频文件作为它的伴生文件跟随移动。
# 该后缀是全仓库保留名——PhotoProcessor._scan_files 必须跳过 *_vcover.jpg，
# 防止封面被照片管线当作纯 JPEG 二次处理（flat 布局/未整理时封面留在顶层）。
# Reserved filename suffix for V-cover (video cover) JPEGs. The representative
# frame of each video becomes a real JPEG named <stem>_vcover.jpg, stored as a
# regular photos row; the video file follows it as a companion. Photo scanning
# must skip *_vcover.jpg so covers are never re-processed as plain photos.
VIDEO_COVER_SUFFIX = '_vcover'
VIDEO_COVER_EXT = '.jpg'

# 传输/网图文件名前缀（大小写不敏感）：手机截屏与社区下载图不是相机原图，
# 不参与鸟种识别（2026-10-20 用户定则：只保留相机本身拍摄的照片/视频，
# 小红书等下载图一律排除）。
# 注意：`2021-05-01 13.11.45.jpg` 这类日期时间命名**不在**排除之列——它是
# Windows 照片导入/手机助手给相机原图导入 PC 时的标准命名，常带完整 EXIF
# （2026-10-20 复盘教训：曾误当微信传输格式批量误伤 184 张相机原图，已恢复）。
# Filename prefixes of transferred/net images (case-insensitive): phone
# screenshots and community downloads are not camera originals and never
# enter bird identification (user rule 2026-10-20). NOTE: date-time naming
# like 2021-05-01 13.11.45.jpg is NOT excluded — it is the standard naming
# of Windows photo import / phone-assistant tools for camera originals
# (2026-10-20 lesson: mis-banishing it wiped 184 camera originals, restored).
TRANSFER_NET_IMAGE_PREFIXES = [
    'screenshot',     # 手机截屏 Screenshot_xxx / Screenshot xxx
    'screen shot',    # 截屏空格变体
    'camera_xhs',     # 小红书下载
    'camera_1040',    # 社区/电商图片下载 ID
    'mmexport',       # 微信图片导出（mmexport+时间戳）
]


def is_transfer_or_net_image(filename: str) -> bool:
    """
    判断文件名是否为截屏/社区下载图（跳过识别；文件本体不受影响）。

    参数:
    filename (str): 文件名（可含路径，按 basename 小写后匹配）

    返回:
    bool: True=跳过识别；False=正常处理

    Whether the filename is a screenshot/community download (skipped from
    identification; the file itself is never touched).
    """
    low = os.path.basename(filename).lower()
    return any(low.startswith(p) for p in TRANSFER_NET_IMAGE_PREFIXES)


# 「非相机原图」EXIF 判定所需的字段（与 PhotoProcessor._read_all_exif_metadata
# 返回字典的键保持一致）。相机型号/拍摄时间/GPS 三项在一切相机原图
# （单反 RAW/JPG、手机 JPEG/HEIC、Windows 导入改名图）上必然至少有一项；
# QQ/网页/聊天工具转存下载图通常三项全空。
# Keys consulted by is_non_camera_original_exif(); must match the dict
# returned by PhotoProcessor._read_all_exif_metadata(). Every camera
# original (DSLR RAW/JPG, phone JPEG/HEIC, import-renamed copies) carries
# at least one of camera model / capture time / GPS; transferred net
# images typically have none.
NON_CAMERA_ORIGINAL_EXIF_KEYS = ('camera_model', 'date_time_original',
                                 'gps_latitude')


def is_non_camera_original_exif(exif_data: dict) -> bool:
    """
    判断 EXIF 元数据是否「三全空」＝无法证实为相机原图（按网图处理）。

    2026-10-21 用户定则的元数据层延伸：文件名前缀黑名单
    （TRANSFER_NET_IMAGE_PREFIXES）管不住伪装成相机命名的下载图
    （如 IMG_8475.JPG 煤山雀高清网图），这类图唯一稳定特征是
    相机型号/拍摄时间/GPS 三项全空。判定规则：

    - 三项全空 → True（非相机原图，按无鸟处理）；
    - 任一项存在 → False（相机原图；微信「原图」等保留 EXIF 的
      转发图同样放行，与 2026-10-20 复盘定则一致）；
    - exif_data 为 None/空字典（EXIF 预取被禁用或读取失败）→ False
      （保守放过，不因元数据缺失误杀无法验证的照片）。

    参数:
    exif_data (dict): _read_all_exif_metadata 风格的元数据字典
                      （键见 NON_CAMERA_ORIGINAL_EXIF_KEYS）

    返回:
    bool: True=非相机原图（按网图处理）；False=相机原图或无法判定

    Whether the EXIF metadata proves the file is NOT a camera original
    (camera model, capture time and GPS all empty — the signature of
    transferred net images masquerading as camera files, e.g. the
    IMG_8475.JPG Coal Tit). Any field present → camera original (WeChat
    "original image" forwards keep EXIF and pass, per the 2026-10-20
    rule). None/empty input → conservatively False (never reject on
    unverifiable metadata).
    """
    if not exif_data:
        return False
    return all(exif_data.get(key) is None
               for key in NON_CAMERA_ORIGINAL_EXIF_KEYS)
