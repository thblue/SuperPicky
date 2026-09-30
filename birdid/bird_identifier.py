#!/usr/bin/env python3
"""
鸟类识别核心模块。
Core bird-identification module.

从 SuperBirdID 移植，负责鸟类检测、分类与离线资源路径兼容。
Ported from SuperBirdID and responsible for bird detection, classification,
and compatibility with offline resource paths.
"""

__version__ = "1.0.0"

import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import io
import json
import os
import sys
from typing import Any, Callable, Optional, List, Dict, Tuple, Set, cast
from tools.i18n import t as _t
from birdid.geo_filter import TIER_NONE, get_geo_filter
from config import (
    get_best_device,
    get_lazy_registry,
    get_app_config_dir,
    get_install_scoped_resource_path,
    get_packaged_model_relative_path,
    get_runtime_meipass,
)

CLASSIFIER_DEVICE = torch.device(str(get_best_device()))

RESAMPLING_LANCZOS = Image.Resampling.LANCZOS

try:
    import rawpy
    import imageio

    RAW_SUPPORT = True
except ImportError:
    rawpy = cast(Any, None)
    imageio = cast(Any, None)
    RAW_SUPPORT = False

# V4.2.7: reverse_geocoder lazy 单例 — 首次用时加载 ~70MB cKDTree 数据，
# 之后所有线程共享同一个只读索引。
# V4.2.7: reverse_geocoder lazy singleton — first use loads ~70MB cKDTree,
# subsequent calls share the read-only index across threads.
import threading

_RG_LOCK = threading.Lock()
_RG_INSTANCE: Any = None  # 标记是否已初始化（None 表示未尝试）
_RG_AVAILABLE = True

# 分类器推理锁：批处理 BirdID executor 与补救扫描确认可能跨线程并发调用
# forward，MPS/CUDA 下并发安全性有限，统一串行化。
# Classifier inference lock: the batch BirdID executor and the rescue-scan
# confirmation may call forward concurrently from different threads; MPS/CUDA
# concurrency safety is limited, so all forwards are serialized here.
_CLASSIFIER_INFER_LOCK = threading.Lock()


def _resolve_country_code_from_gps(lat: float, lon: float) -> Optional[str]:
    """
    用 reverse_geocoder 把 GPS 坐标反查成 ISO 3166-1 alpha-2 国家代码。

    Convert a GPS coordinate to ISO 3166-1 alpha-2 country code via
    reverse_geocoder (offline, cKDTree-backed). Returns None when the
    library is unavailable or the lookup fails.
    """
    global _RG_INSTANCE, _RG_AVAILABLE
    if not _RG_AVAILABLE:
        return None
    try:
        if _RG_INSTANCE is None:
            with _RG_LOCK:
                if _RG_INSTANCE is None:
                    import reverse_geocoder as rg
                    _RG_INSTANCE = rg
        result = _RG_INSTANCE.search([(lat, lon)], mode=1, verbose=False)
        if result and result[0].get("cc"):
            return str(result[0]["cc"]).upper()
    except Exception:
        _RG_AVAILABLE = False  # 永久禁用，避免反复 import 失败
    return None

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
    # ultralytics 导入时会全局 cv2.setNumThreads(0)，立即恢复线程池
    # ultralytics globally disables the cv2 thread pool at import; restore it
    from config import ensure_cv2_thread_pool

    ensure_cv2_thread_pool()
except ImportError:
    YOLO = cast(Any, None)
    YOLO_AVAILABLE = False

BIRDID_DIR = os.path.dirname(os.path.abspath(__file__))


# 项目根目录（code_updates overlay 场景下 __file__ 指向 code_updates/birdid/，需通过 sys.path 找真实根）
# Project root lookup: in code_updates overlay scenarios __file__ points to
# code_updates/birdid/, so fall back to sys.path entries to locate the real root.
def _find_project_root() -> str:
    candidate = os.path.dirname(BIRDID_DIR)
    if os.path.exists(os.path.join(candidate, "models", "model20240824.pth")):
        return candidate
    for p in sys.path:
        if (
            p
            and os.path.isdir(p)
            and os.path.exists(os.path.join(p, "models", "model20240824.pth"))
        ):
            return p
    return candidate


def _find_birdid_dir() -> str:
    if os.path.exists(os.path.join(BIRDID_DIR, "data", "bird_reference.sqlite")):
        return BIRDID_DIR
    for p in sys.path:
        if p and os.path.isdir(p):
            candidate = os.path.join(p, "birdid")
            if os.path.exists(os.path.join(candidate, "data", "bird_reference.sqlite")):
                return candidate
    return BIRDID_DIR


PROJECT_ROOT = _find_project_root()
BIRDID_DIR = _find_birdid_dir()


def get_birdid_path(relative_path: str) -> str:
    """
    返回 `birdid/` 目录下的资源路径。
    Return a resource path under the `birdid/` directory.

    Windows Lite 构建需要从安装目录 `_internal` 读取资源，其余冻结环境仍跟随
    PyInstaller bundle 目录；源码环境则回退到仓库内的 `birdid/` 目录。
    Windows Lite builds read from the install-scoped `_internal` tree, other
    frozen builds still follow the PyInstaller bundle, and source runs fall back
    to the repository `birdid/` directory.
    """
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        return str(
            get_install_scoped_resource_path(os.path.join("birdid", relative_path))
        )
    if getattr(sys, "frozen", False):
        meipass = get_runtime_meipass()
        if meipass is not None:
            return os.path.join(meipass, "birdid", relative_path)
    return os.path.join(BIRDID_DIR, relative_path)


def get_project_path(relative_path: str) -> str:
    """
    返回项目级资源路径。
    Return a project-level resource path.

    这里统一兼容 Windows Lite 安装目录、普通 PyInstaller bundle 与源码目录，
    避免各调用方再自行拼接 `_MEIPASS` 路径。
    This helper centralizes path selection for Windows Lite installs, regular
    PyInstaller bundles, and source checkouts so callers do not rebuild
    `_MEIPASS`-based paths themselves.
    """
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        packaged_relative_path = None
        if relative_path.startswith("models/"):
            packaged_relative_path = get_packaged_model_relative_path(relative_path)
        return str(
            get_install_scoped_resource_path(
                relative_path, packaged_relative_path=packaged_relative_path
            )
        )
    if getattr(sys, "frozen", False):
        meipass = get_runtime_meipass()
        if meipass is not None:
            return os.path.join(meipass, relative_path)
    return os.path.join(PROJECT_ROOT, relative_path)


def get_user_data_dir() -> str:
    user_data_dir = str(get_app_config_dir())
    os.makedirs(user_data_dir, exist_ok=True)
    return user_data_dir


MODEL_PATH = get_project_path("models/model20240824.pth")
MODEL_PATH_LEGACY = get_birdid_path("models/birdid2024.pt")
MODEL_PATH_ENC = get_birdid_path("models/birdid2024.pt.enc")
OSEA_NUM_CLASSES = 11000
DATABASE_PATH = get_birdid_path("data/bird_reference.sqlite")
YOLO_MODEL_PATH = get_project_path("models/yolo11l-seg.pt")


def decrypt_model(encrypted_path: str, password: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()

    salt = encrypted_data[:16]
    iv = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode())

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext_padded = decryptor.update(ciphertext) + decryptor.finalize()

    padding_length = plaintext_padded[-1]
    return plaintext_padded[:-padding_length]


def _load_torchscript_from_bytes(model_data: bytes):
    buffer = io.BytesIO(model_data)
    return torch.jit.load(buffer, map_location="cpu")


def get_classifier():
    registry = get_lazy_registry()

    def _factory():
        import torchvision.models as models

        if os.path.exists(MODEL_PATH):
            model = models.resnet34(num_classes=OSEA_NUM_CLASSES)
            state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device=CLASSIFIER_DEVICE)
            model.eval()
            return model

        SECRET_PASSWORD = "SuperBirdID_2024_AI_Model_Encryption_Key_v1"
        if os.path.exists(MODEL_PATH_ENC):
            model_data = decrypt_model(MODEL_PATH_ENC, SECRET_PASSWORD)
            model = _load_torchscript_from_bytes(model_data)
        elif os.path.exists(MODEL_PATH_LEGACY):
            try:
                model = torch.jit.load(MODEL_PATH_LEGACY, map_location="cpu")
            except RuntimeError as e:
                if "open file failed" not in str(e) or "fopen" not in str(e):
                    raise
                with open(MODEL_PATH_LEGACY, "rb") as f:
                    model_data = f.read()
                model = _load_torchscript_from_bytes(model_data)
        else:
            raise RuntimeError(f"未找到分类模型: {MODEL_PATH} 或 {MODEL_PATH_LEGACY}")

        model = model.to(CLASSIFIER_DEVICE)
        model.eval()  # noqa: model.eval() is a PyTorch API call, not Python eval()
        print(_t("logs.birdid_fallback_model"))
        return model

    return registry.get_or_create("birdid.classifier", _factory)


def get_bird_model():
    return get_classifier()


def get_database_manager():
    registry = get_lazy_registry()

    def _factory():
        try:
            from birdid.bird_database_manager import BirdDatabaseManager

            if os.path.exists(DATABASE_PATH):
                return BirdDatabaseManager(DATABASE_PATH)
            print(f"[BirdID] 数据库文件不存在，罕见度/IUCN/AviList 命名将不可用: {DATABASE_PATH}")
        except Exception as e:
            # V4.4: 这里以前完全静默——调用方后续都用 `if db_manager:` 跳过相关功能，
            # 用户只会看到"罕见度/IUCN/AviList 名称全部消失"，却无从判断是数据库损坏、
            # 权限问题还是别的原因。这个 registry 是进程级单例缓存，只会失败一次就
            # 定型，所以这条日志只会打印一次，不会刷屏。
            # V4.4: This used to fail completely silently — callers all guard with
            # `if db_manager:` and skip the related features, so the user only sees
            # "rarity/IUCN/AviList names all vanished" with no way to tell whether
            # it's a corrupt DB, a permissions issue, or something else. The result
            # is cached for the process lifetime by the lazy registry, so this log
            # line fires at most once, not on every call.
            print(f"[BirdID] 数据库管理器初始化失败 / database manager init failed: {e}")
        return False

    result = registry.get_or_create("birdid.database_manager", _factory)
    return result if result is not False else None


def get_yolo_detector():
    if not YOLO_AVAILABLE:
        return None
    registry = get_lazy_registry()
    return registry.get_or_create(
        "birdid.yolo_detector",
        lambda: (
            YOLOBirdDetector(YOLO_MODEL_PATH)
            if os.path.exists(YOLO_MODEL_PATH)
            else None
        ),
    )


class YOLOBirdDetector:
    def __init__(self, model_path: Optional[str] = None):
        if not YOLO_AVAILABLE:
            self.model = None
            return

        if model_path is None:
            model_path = YOLO_MODEL_PATH

        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            self.model = None
            return

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            self.model = None

    def detect_and_crop_bird(
        self,
        image_input,
        confidence_threshold: float = 0.25,
        padding_ratio: float = 0.15,
        fill_color: Tuple[int, int, int] = (0, 0, 0),
        focus_point: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        用 YOLO 检测鸟主体并方形裁剪。

        参数:
            image_input: 文件路径 或 PIL Image
            confidence_threshold (float): YOLO 置信度阈值
            padding_ratio (float): 裁剪框 padding 比例
            fill_color: 补边颜色
            focus_point (Optional[Tuple[float,float]]): 相机对焦点归一化坐标 (x,y)∈[0,1]。
                多目标时优先选「bbox 包含对焦点」的框（与选鸟模式 ai_model.detect_and_draw_birds
                对齐），都不含则回退最高置信度。仅 RAW 通常带此信息。

        返回:
            (裁剪后的 PIL RGB Image 或 None, 信息字符串)

        Detect the bird subject with YOLO and square-crop it.
        When focus_point is given and multiple birds are detected, prefer the bbox
        containing the focus point (matching the main picking pipeline); otherwise
        fall back to the highest-confidence detection.
        """
        if self.model is None:
            return None, "YOLO模型未可用"

        try:
            if isinstance(image_input, str):
                image = load_image(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                return None, "不支持的图像输入类型"

            # ultralytics 约定 numpy 输入为 BGR；PIL 是 RGB，必须转换，
            # 否则通道反转会让杂背景下的小鸟/弱对比目标漏检（与选鸟模式 cv2.imread 对齐）。
            # ultralytics expects BGR numpy input; PIL is RGB, so convert — otherwise
            # the channel swap makes small/low-contrast birds in clutter undetectable.
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # imgsz=1024 与选鸟模式 (preprocess_image → TARGET_IMAGE_SIZE=1024) 对齐：
            # 默认 640 会把高像素原图直接降采样到 640，杂背景里的远距小鸟被抹掉而漏检。
            # imgsz=1024 matches the picking pipeline; the default 640 downsamples a
            # high-res frame too aggressively and drops small distant birds.
            # V4.4: 显式指定推理设备，与项目统一的 get_best_device() 策略对齐
            # （Intel Mac 强制 CPU 等规则），避免这里悄悄走 ultralytics 自己的
            # 默认设备选择、与主选片流程的设备行为不一致。
            # V4.4: Explicitly pin the inference device to the project-wide
            # get_best_device() policy (e.g. Intel Mac forced to CPU) instead of
            # silently falling back to ultralytics' own default device selection,
            # which could diverge from the main picking pipeline.
            results = self.model(img_array, conf=confidence_threshold, imgsz=1024, device=CLASSIFIER_DEVICE.type)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        if class_id == 14:
                            detections.append(
                                {
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(confidence),
                                }
                            )

            if not detections:
                return None, "未检测到鸟类"

            img_width, img_height = image.size

            # —— 多目标选框：对焦点优先，回退最高置信度 ——
            # —— Subject selection: focus-point first, fall back to top confidence ——
            best = None
            if len(detections) > 1 and focus_point is not None:
                fx, fy = focus_point
                fx_px, fy_px = int(fx * img_width), int(fy * img_height)
                for det in detections:
                    bx1, by1, bx2, by2 = det["bbox"]
                    if bx1 <= fx_px <= bx2 and by1 <= fy_px <= by2:
                        best = det
                        break
            if best is None:
                best = max(detections, key=lambda x: x["confidence"])

            x1, y1, x2, y2 = best["bbox"]
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            max_side = max(bbox_width, bbox_height)
            target_side = int(max_side * (1 + padding_ratio))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half = target_side // 2

            sq_x1 = cx - half
            sq_y1 = cy - half
            sq_x2 = cx + half
            sq_y2 = cy + half

            crop_x1 = max(0, sq_x1)
            crop_y1 = max(0, sq_y1)
            crop_x2 = min(img_width, sq_x2)
            crop_y2 = min(img_height, sq_y2)

            cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_w, crop_h = cropped.size

            if crop_w != crop_h:
                sq_size = max(crop_w, crop_h)
                square = Image.new("RGB", (sq_size, sq_size), fill_color)
                paste_x = (sq_size - crop_w) // 2
                paste_y = (sq_size - crop_h) // 2
                square.paste(cropped, (paste_x, paste_y))
                cropped = square

            info = f"conf={best['confidence']:.3f}, size={cropped.size}"

            return cropped, info

        except Exception as e:
            return None, f"检测失败: {e}"


def _auto_orient(img: Image.Image, raw_flip: int = 0) -> Image.Image:
    """
    将图片旋转到正确朝向，再交给识别流程。
    优先用图自带的 EXIF Orientation（相机 thumb/JpgFromRaw/JPEG 通常带 274 标签）；
    rawpy postprocess / BITMAP 等无 EXIF 的情形回退 libraw 的 flip 值。
    必须在 convert("RGB") 之前调用——convert 会丢弃 EXIF。

    Rotate an image upright before identification: prefer the embedded EXIF
    Orientation, fall back to libraw's flip for EXIF-less RAW bitmaps.
    Must run before convert("RGB"), which drops EXIF.

    背景：竖拍 RAW（如 Orientation=Rotate 270 CW）的 thumb/传感器数据是横向的，
    若不旋转，鸟会"横躺"送进 YOLO+分类器，导致识别错误且置信度极低。
    """
    from PIL import ImageOps
    try:
        if img.getexif().get(274, 1) != 1:
            return ImageOps.exif_transpose(img)
    except Exception:
        pass
    # libraw flip → PIL transpose（把传感器原始方向转正）
    flip_map = {3: Image.ROTATE_180, 5: Image.ROTATE_90, 6: Image.ROTATE_270}
    transpose = flip_map.get(raw_flip)
    if transpose is not None:
        try:
            return img.transpose(transpose)
        except Exception:
            pass
    return img


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()

    raw_extensions = [
        ".cr2",
        ".cr3",
        ".nef",
        ".nrw",
        ".arw",
        ".srf",
        ".dng",
        ".raf",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
        ".raw",
        ".rwl",
        ".3fr",
        ".fff",
        ".erf",
        ".mef",
        ".mos",
        ".mrw",
        ".x3f",
        ".hif",
        ".heif",
        ".heic",
    ]

    heif_extensions = {".hif", ".heif", ".heic"}

    if ext in raw_extensions:
        if ext in heif_extensions:
            return _load_heif(image_path)
        if RAW_SUPPORT:
            thumb_format_enum = getattr(rawpy, "ThumbFormat", None)
            jpeg_thumb_format = getattr(thumb_format_enum, "JPEG", None)
            bitmap_thumb_format = getattr(thumb_format_enum, "BITMAP", None)
            rawpy_internal = getattr(rawpy, "_rawpy", None)
            unsupported_error = getattr(
                rawpy_internal, "LibRawFileUnsupportedError", None
            )
            try:
                with rawpy.imread(image_path) as raw:
                    raw_flip = int(getattr(raw.sizes, "flip", 0) or 0)
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == jpeg_thumb_format:
                            from io import BytesIO

                            # V4.3.0: thumb JPEG 自带 EXIF Orientation，先按方向旋转再 convert，
                            # 否则竖拍照片被当横图送入识别，鸟"横躺"导致识别错（见 _auto_orient）。
                            img = Image.open(BytesIO(thumb.data))
                            return _auto_orient(img, raw_flip).convert("RGB")
                        elif thumb.format == bitmap_thumb_format:
                            img = _auto_orient(Image.fromarray(thumb.data), raw_flip)
                            return img.convert("RGB")
                    except Exception as e:
                        pass

                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=8,
                        no_auto_bright=False,
                        auto_bright_thr=0.01,
                        half_size=True,
                    )
                    # postprocess 输出传感器原始方向（无 EXIF），按 libraw flip 旋转
                    img = _auto_orient(Image.fromarray(rgb), raw_flip)
                    return img
            except Exception as e:
                if unsupported_error is not None and isinstance(e, unsupported_error):
                    return _load_raw_via_exiftool(image_path)
                raise Exception(f"RAW处理失败: {e}")
        else:
            raise ImportError("需要安装 rawpy 来处理 RAW 格式")
    else:
        # V4.3.0: JPEG 等自带 Orientation，按方向旋转后再转 RGB
        return _auto_orient(Image.open(image_path)).convert("RGB")


def _load_raw_via_exiftool(image_path: str) -> Image.Image:
    """
    使用 ExifTool 从 RAW 文件提取可解码预览图。
    Extract a decodable preview image from a RAW file via ExifTool.

    复用 tools.exiftool_manager 的常驻进程，而非自行拼路径 + 裸 subprocess：
    旧实现硬编码 mac-only 路径且从未在 Windows 上传 creationflags=
    CREATE_NO_WINDOW，导致 Windows 用户处理时弹出一闪而过的控制台窗口。

    Reuse tools.exiftool_manager's persistent process instead of rolling our
    own path lookup + bare subprocess: the old code hardcoded macOS-only paths
    and never passed creationflags=CREATE_NO_WINDOW on Windows, flashing a
    console window during processing.
    """
    from io import BytesIO
    from tools.exiftool_manager import get_exiftool_manager

    manager = get_exiftool_manager()
    for tag in ["-JpgFromRaw", "-PreviewImage", "-ThumbnailImage"]:
        try:
            data = manager.extract_binary(image_path, tag)
            if data and len(data) > 1000:
                # V4.3.0: JpgFromRaw/Preview 自带 Orientation，按方向旋转再转 RGB
                img = _auto_orient(Image.open(BytesIO(data))).convert("RGB")
                return img
        except Exception:
            continue

    raise Exception(
        f"\u6682\u4e0d\u652f\u6301\u6b64 RAW \u683c\u5f0f\uff08{os.path.basename(image_path)}\uff09\u3002"
        "Sony A7M5 \u7b49\u76f8\u673a\u7684 NeXt/Compressed RAW 2 \u683c\u5f0f\u76ee\u524d\u7b2c\u4e09\u65b9\u5e93\u5c1a\u672a\u5b8c\u6574\u652f\u6301\uff0c"
        "\u5c06\u0627\u5728\u540e\u7eed\u7248\u672c\u4e2d\u4fee\u590d\u3002\u5efa\u8bae\u4e34\u65f6\u4f7f\u7528\u65e0\u538b\u7f29 RAW \u6216 JPEG \u683c\u5f0f\u62cd\u6444\u3002"
    )


def _load_heif(image_path: str) -> Image.Image:
    try:
        import pillow_heif

        heif_file = pillow_heif.read_heif(image_path)
        if heif_file.data is None:
            raise ValueError("HEIF 解码结果缺少像素数据")
        img = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        ).convert("RGB")
        return img
    except ImportError:
        raise Exception(
            "请安装 pillow-heif 来支持 HIF/HEIC 格式： pip install pillow-heif"
        )
    except Exception as e:
        raise Exception(f"HEIF 解码失败 ({os.path.basename(image_path)}): {e}")


def _gps_coords_present(lat: Optional[float], lon: Optional[float]) -> bool:
    """
    判断 GPS 坐标是否存在（两者均非 None）。

    0.0 是合法坐标——赤道（lat=0.0）与本初子午线（lon=0.0）——所以这里
    必须用 `is not None` 而非真值判断；identify_bird 曾因 `if lat and lon:`
    把这类照片当作无 GPS，导致拍摄国家解析与按国家归一化 rarity 静默失效。

    参数:
    lat (Optional[float]): 纬度，无 GPS 时为 None
    lon (Optional[float]): 经度，无 GPS 时为 None

    返回:
    bool: 两者均非 None 时为 True

    Check whether GPS coordinates are present (both non-None).

    0.0 is a legal coordinate — the equator (lat=0.0) and the prime meridian
    (lon=0.0) — so this must use `is not None`, never truthiness;
    identify_bird once used `if lat and lon:` and silently dropped such
    photos' country resolution and country-aware rarity normalization.

    Parameters:
    lat (Optional[float]): Latitude, None when GPS is absent
    lon (Optional[float]): Longitude, None when GPS is absent

    Return:
    bool: True when both are non-None
    """
    return lat is not None and lon is not None


def extract_gps_from_exif(
    image_path: str,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    从照片提取 GPS 坐标：优先走 ExifTool，取不到再退回 PIL EXIF。

    复用 tools.exiftool_manager 的常驻进程，而非自行拼路径 + 裸 subprocess：
    旧实现硬编码 mac-only 路径且从未在 Windows 上传 creationflags=
    CREATE_NO_WINDOW；这个函数在 identify_bird() 里每张照片都会调用一次，
    导致 Windows 用户开启识鸟处理文件夹时每张照片都弹一次一闪而过的控制台
    窗口（关闭识鸟就不会走到这条路径，现象完全对应）。

    Extract GPS coordinates from a photo: try ExifTool first, then fall back
    to PIL EXIF.

    Reuse tools.exiftool_manager's persistent process instead of rolling our
    own path lookup + bare subprocess: the old code hardcoded macOS-only paths
    and never passed creationflags=CREATE_NO_WINDOW on Windows. This function
    runs once per photo inside identify_bird(), so Windows users saw a console
    window flash for every photo while Bird ID was on (and never otherwise —
    matching the exact symptom reported).
    """
    try:
        from tools.exiftool_manager import get_exiftool_manager

        gps_data = get_exiftool_manager().read_metadata(
            image_path,
            extra_args=[
                "-GPSLatitude",
                "-GPSLongitude",
                "-GPSLatitudeRef",
                "-GPSLongitudeRef",
            ],
        )

        if gps_data:
            lat_str = gps_data.get("GPSLatitude", "")
            lon_str = gps_data.get("GPSLongitude", "")
            lat_ref = gps_data.get("GPSLatitudeRef", "N")
            lon_ref = gps_data.get("GPSLongitudeRef", "E")

            if lat_str and lon_str:

                def parse_dms(dms_str):
                    import re

                    match = re.search(
                        r'(\d+)\s*deg\s*(\d+)\'\s*([\d.]+)"?', str(dms_str)
                    )
                    if match:
                        d, m, s = (
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        )
                        return d + m / 60 + s / 3600
                    try:
                        return float(dms_str)
                    except:
                        return None

                lat = parse_dms(lat_str)
                lon = parse_dms(lon_str)

                if lat is not None and lon is not None:
                    if lat_ref and lat_ref.upper().startswith("S"):
                        lat = -lat
                    if lon_ref and lon_ref.upper().startswith("W"):
                        lon = -lon
                    return lat, lon, f"GPS: {lat:.6f}, {lon:.6f}"
    except Exception:
        pass

    try:
        image = Image.open(image_path)
        exif_data = image.getexif()

        if not exif_data:
            return None, None, "无EXIF数据"

        gps_info = {}
        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                for gps_tag in value:
                    gps_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_decoded] = value[gps_tag]
                break

        if not gps_info:
            return None, None, "无GPS数据"

        def convert_to_degrees(coord, ref):
            d, m, s = coord
            decimal = d + (m / 60.0) + (s / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal

        lat = None
        lon = None

        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = convert_to_degrees(
                gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"]
            )

        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = convert_to_degrees(
                gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"]
            )

        if lat is not None and lon is not None:
            return lat, lon, f"GPS: {lat:.6f}, {lon:.6f}"

        return None, None, "GPS坐标不完整"

    except Exception as e:
        return None, None, f"GPS解析失败: {e}"


def smart_resize(image: Image.Image, target_size: int = 224) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)

    if max_dim < 1000:
        return image.resize((target_size, target_size), RESAMPLING_LANCZOS)

    resized = image.resize((256, 256), RESAMPLING_LANCZOS)
    left = (256 - target_size) // 2
    top = (256 - target_size) // 2
    return resized.crop((left, top, left + target_size, top + target_size))


def apply_enhancement(image: Image.Image, method: str = "unsharp_mask") -> Image.Image:
    if method == "unsharp_mask":
        return image.filter(ImageFilter.UnsharpMask())
    elif method == "edge_enhance_more":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif method == "contrast_edge":
        enhanced = ImageEnhance.Brightness(image).enhance(1.2)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
        return enhanced.filter(ImageFilter.EDGE_ENHANCE)
    elif method == "desaturate":
        return ImageEnhance.Color(image).enhance(0.5)
    return image


# V4.5: transform 与温度收敛到 birdid/osea_preprocess.py 单一事实源，
# 与 osea_classifier.py 共享同一份定义，杜绝双份复制漂移。
# V4.5: Transforms and temperature now come from the shared SSOT module
# birdid/osea_preprocess.py, shared with osea_classifier.py — no more
# duplicated definitions drifting apart.
from birdid.osea_preprocess import (
    OSEA_TEMPERATURE,
    OSEA_TRANSFORM,
    OSEA_TRANSFORM_DIRECT,
)


def predict_bird(
    image: Image.Image,
    top_k: int = 5,
    species_class_ids: Optional[Set[int]] = None,
    is_yolo_cropped: bool = False,
    name_format: Optional[str] = None,
    photo_country_code: Optional[str] = None,
) -> List[Dict]:
    model = get_classifier()
    db_manager = get_database_manager()

    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = OSEA_TRANSFORM_DIRECT if is_yolo_cropped else OSEA_TRANSFORM
    transformed_tensor = cast(torch.Tensor, transform(image))
    input_tensor = transformed_tensor.unsqueeze(0)

    with _CLASSIFIER_INFER_LOCK:
        input_tensor = input_tensor.to(CLASSIFIER_DEVICE)
        with torch.no_grad():
            output = model(input_tensor)[0]

    num_classes = min(10964, output.shape[0])
    output = output[:num_classes]

    best_probs = torch.nn.functional.softmax(output / OSEA_TEMPERATURE, dim=0)

    k = min(100 if species_class_ids else top_k, len(best_probs))
    top_probs, top_indices = torch.topk(best_probs, k)

    results = []
    for i in range(len(top_indices)):
        class_id = top_indices[i].item()
        confidence = top_probs[i].item() * 100
        min_confidence = 0.3 if species_class_ids else 1.0
        if confidence < min_confidence:
            continue

        cn_name = None
        en_name = None
        scientific_name = None
        ebird_code = None
        description = None

        if db_manager:
            info = db_manager.get_bird_by_class_id(class_id)
            if info:
                cn_name = info.get("chinese_simplified")
                en_name = info.get("english_name")
                scientific_name = info.get("scientific_name")
                ebird_code = info.get("ebird_code")
                description = info.get("short_description_zh")

        if not cn_name:
            cn_name = f"Unknown (ID: {class_id})"
            en_name = f"Unknown (ID: {class_id})"

        if name_format and name_format != "default" and db_manager:
            avilist_info = db_manager.get_avilist_names_by_class_id(class_id)
            if avilist_info and avilist_info.get("match_type") != "no_match":
                if name_format == "scientific":
                    en_name = (
                        avilist_info.get("scientific_name_avilist")
                        or scientific_name
                        or en_name
                    )
                else:
                    col = f"en_name_{name_format}"
                    alt_name = avilist_info.get(col)
                    if alt_name:
                        en_name = alt_name
                    elif name_format != "avilist" and avilist_info.get(
                        "en_name_avilist"
                    ):
                        en_name = avilist_info["en_name_avilist"]

        region_match = False
        if species_class_ids:
            if class_id in species_class_ids:
                region_match = True
            else:
                continue

        iucn_category = (
            db_manager.get_iucn_by_class_id(class_id) if db_manager else None
        )
        # V4.2.7: GBIF 罕见度（按拍摄地国家优先，未命中回退到全球）
        # V4.2.7: GBIF rarity — country-aware (per photo country) with global fallback
        gbif_rarity_100 = (
            db_manager.get_gbif_rarity_by_class_id(class_id, photo_country_code)
            if db_manager
            else None
        )
        # iRateBird 鸟种美学(颜值)分（0–100，与照片无关的物种级指标）
        # iRateBird species aesthetic score (0–100, species-level, photo-agnostic)
        aesthetic_index = (
            db_manager.get_aesthetic_by_class_id(class_id) if db_manager else None
        )
        # 国家重点保护野生动物等级（1=一级/2=二级，物种级属性，与拍摄地无关）
        # China national protection level (1 or 2; species-level attribute,
        # independent of where the photo was taken)
        china_protection_level = (
            db_manager.get_china_protection_by_class_id(class_id)
            if db_manager
            else None
        )

        results.append(
            {
                "class_id": class_id,
                "cn_name": cn_name,
                "en_name": en_name,
                "scientific_name": scientific_name,
                "iucn_category": iucn_category,
                "gbif_rarity_100": gbif_rarity_100,
                "china_protection_level": china_protection_level,
                "aesthetic_index": aesthetic_index,
                "confidence": confidence,
                "ebird_code": ebird_code,
                "region_match": region_match,
                "description": description or "",
            }
        )

        if len(results) >= top_k:
            break

    return results


# RAW 扩展名（带相机对焦点元数据）；JPEG/HEIF 通常无对焦点。
# RAW extensions that carry camera autofocus-point metadata.
_FOCUS_RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".dng", ".raf",
    ".orf", ".rw2", ".pef", ".srw", ".raw", ".rwl", ".3fr", ".fff",
    ".erf", ".mef", ".mos", ".mrw", ".x3f",
}


def _read_focus_point_for_path(
    image_path: Optional[str],
) -> Optional[Tuple[float, float]]:
    """
    从 RAW 文件读取相机自动对焦点（归一化坐标 x,y ∈ [0,1]）。

    供识鸟面板 / CLI 在多目标场景下选「对焦的鸟」，与选鸟模式
    (core.photo_processor → ai_model.detect_and_draw_birds) 的对焦点选框对齐。

    参数:
        image_path (Optional[str]): 图片路径；非 RAW 或读取失败返回 None。

    返回:
        Optional[Tuple[float, float]]: 归一化对焦点 (x, y)，无有效对焦点时 None。

    Read the camera autofocus point (normalized x,y) from a RAW file so the
    BirdID dock/CLI can pick the focused bird in multi-subject scenes, matching
    the main picking pipeline. Returns None for non-RAW or on any failure.
    """
    if not image_path:
        return None
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in _FOCUS_RAW_EXTENSIONS:
        return None
    try:
        # 延迟导入，避免 birdid 包对 core 的硬依赖（仅 RAW 路径才触发）。
        # Lazy import to avoid a hard birdid→core dependency; only RAW hits this.
        from core.focus_point_detector import get_focus_detector

        focus = get_focus_detector().detect(image_path)
        if focus is not None and getattr(focus, "is_valid", False):
            return (float(focus.x), float(focus.y))
    except Exception:
        pass
    return None


def _identify_with_tiers(
    image,
    top_k: int,
    lat: Optional[float],
    lon: Optional[float],
    country_code: Optional[str],
    is_yolo_cropped: bool,
    name_format: Optional[str],
    photo_country_code: Optional[str],
) -> Tuple[List[Dict], str, Optional[int]]:
    """
    遍历地理候选层，命中即停 / Walk the geo candidate tiers, stopping at the first hit.

    旧实现一次性取候选集，过窄时直接崩到无过滤（冰岛网格仅 54 类即触发该路径，
    产出小企鹅、蓝脚鲣鸟等跨半球错误）。改为逐层放宽后，稀疏网格会平滑降到
    邻域或国家级，不再出现「候选集塌陷 → 完全放弃过滤」。

    The old implementation took a single candidate set and collapsed straight to
    unfiltered when it was too narrow (Iceland's 54-class cell triggered exactly
    that, yielding cross-hemisphere errors). Widening tier by tier lets sparse
    cells degrade smoothly to the neighbourhood or country level.

    参数 / Parameters:
        image: 待识别图像 / Image to identify.
        top_k (int): 返回结果数 / Number of results.
        lat (Optional[float]): 纬度 / Latitude.
        lon (Optional[float]): 经度 / Longitude.
        country_code (Optional[str]): 国家/地区代码 / Country or region code.
        is_yolo_cropped (bool): 是否已由 YOLO 裁剪 / Whether YOLO already cropped.
        name_format (Optional[str]): 鸟名格式 / Bird name format.
        photo_country_code (Optional[str]): 拍摄国家，供 GBIF 罕见度使用 /
            Shooting country, used for GBIF rarity.

    返回 / Returns:
        tuple: (结果列表, 命中的层标签, 该层候选数或 None) /
            (results, tier label, candidate count or None).
    """
    geo = get_geo_filter()
    if geo is None:
        results = predict_bird(
            image,
            top_k=top_k,
            species_class_ids=None,
            is_yolo_cropped=is_yolo_cropped,
            name_format=name_format,
            photo_country_code=photo_country_code,
        )
        return results, TIER_NONE, None

    for candidates, tier in geo.iter_candidates(lat, lon, country_code):
        results = predict_bird(
            image,
            top_k=top_k,
            species_class_ids=candidates,
            is_yolo_cropped=is_yolo_cropped,
            name_format=name_format,
            photo_country_code=photo_country_code,
        )
        if results:
            return results, tier, (len(candidates) if candidates else None)
    return [], TIER_NONE, None


# ---------------------------------------------------------------------------
# 人工定种白名单 / Manual species whitelist
# ---------------------------------------------------------------------------

_WHITELIST_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "map": {}}


def _whitelist_path() -> str:
    """
    白名单文件路径（应用配置目录）/ Whitelist path in the app config dir.

    文件与 advanced_config.json 同目录（Windows:
    ~/AppData/Local/SuperPicky/），随应用更新保留、由用户手工维护。
    config 不可导入时（独立 birdid 包场景）返回空串，功能关闭。

    The file lives beside advanced_config.json and is user-maintained. An
    empty string is returned when config is not importable (standalone
    birdid package), disabling the feature.

    返回 / Returns:
        str: JSON 路径；不可用时空串 / The JSON path, or "" when unavailable.
    """
    try:
        from config import get_app_config_dir

        return os.path.join(str(get_app_config_dir()), "species_whitelist.json")
    except Exception:
        return ""


def load_species_whitelist(
    path: Optional[str] = None, force: bool = False
) -> Dict[int, Dict]:
    """
    载入白名单对：demote class_id → keep 物种信息 / Load demote→keep pairs.

    文件格式（UTF-8 JSON，可手工增删对；demote 支持单字符串或数组）::

        {"pairs": [
            {"keep_scientific": "Pica serica",
             "demote_scientific": "Pica pica"},
            {"keep_scientific": "Saxicola maurus",
             "demote_scientific": ["Saxicola stejnegeri",
                                   "Saxicola rubicola"]}
        ]}

    keep/demote 学名经 BirdCountInfo 反查 class_id；任一侧解析失败或两侧
    相同则该 demote 跳过。国家相关的 GBIF 稀有度留空，替换时按拍摄国现查。
    结果按 (path, mtime) 进程级缓存，文件改动自动重载。

    The scientific names resolve to class ids via BirdCountInfo; a demote
    entry is skipped when either side fails to resolve. Country-scoped GBIF
    rarity is left empty and queried at substitution time. Results are
    cached per (path, mtime) so manual edits reload automatically.

    参数 / Parameters:
        path (Optional[str]): 显式路径（测试用）；None 时用 _whitelist_path /
            Explicit path (tests); defaults to _whitelist_path.
        force (bool): 忽略缓存强读 / Ignore the cache.

    返回 / Returns:
        dict[int, dict]: {demote_class_id: keep 物种条目}，无文件/解析
            失败时为空 dict / {demote id: keep entry}, empty on failure.
    """
    p = path or _whitelist_path()
    if not p or not os.path.exists(p):
        return {}
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return {}
    if (
        not force
        and _WHITELIST_CACHE["path"] == p
        and _WHITELIST_CACHE["mtime"] == mtime
    ):
        return _WHITELIST_CACHE["map"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}

    db = get_database_manager()
    out: Dict[int, Dict] = {}
    for pair in data.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        keep_sci = str(pair.get("keep_scientific") or "").strip()
        raw_demote = pair.get("demote_scientific")
        if isinstance(raw_demote, str):
            demote_list = [raw_demote.strip()]
        elif isinstance(raw_demote, list):
            demote_list = [str(x).strip() for x in raw_demote if x]
        else:
            demote_list = []
        if not keep_sci or not demote_list or db is None:
            continue
        try:
            keep_cid = db.get_class_id_by_scientific_name(keep_sci)
        except Exception:
            continue
        if keep_cid is None:
            continue
        info = db.get_bird_by_class_id(int(keep_cid)) or {}
        keep_entry = {
            "class_id": int(keep_cid),
            "cn_name": info.get("chinese_simplified"),
            "en_name": info.get("english_name"),
            "scientific_name": info.get("scientific_name") or keep_sci,
            "ebird_code": info.get("ebird_code"),
            "description": info.get("short_description_zh") or "",
            "iucn_category": db.get_iucn_by_class_id(int(keep_cid)),
            "china_protection_level": db.get_china_protection_by_class_id(
                int(keep_cid)
            ),
            "aesthetic_index": db.get_aesthetic_by_class_id(int(keep_cid)),
            # 国家相关，替换时按拍摄国现查 / Country-scoped, queried at
            # substitution time.
            "gbif_rarity_100": None,
        }
        for demote_sci in demote_list:
            try:
                demote_cid = db.get_class_id_by_scientific_name(demote_sci)
            except Exception:
                continue
            if demote_cid is None or demote_cid == keep_cid:
                continue
            out[int(demote_cid)] = keep_entry
    _WHITELIST_CACHE.update(path=p, mtime=mtime, map=out)
    return out


def _apply_species_whitelist(
    results: List[Dict],
    whitelist: Dict[int, Dict],
    photo_country_code: Optional[str] = None,
    rarity_fn: Optional[Callable[[int, Optional[str]], Optional[float]]] = None,
) -> List[Dict]:
    """
    把结果中的 demote 物种就地替换为 keep 物种 / Substitute demote→keep.

    场景：模型图像先验偏爱训练图更多的姊妹种（欧洲欧亚喜鹊 vs 东亚喜鹊），
    地理过滤无法拆开（两者都是中国合法分布种）。用户核定的白名单对在
    identify_bird 出口统一替换，等于把人工改种自动化——主鸟采纳、多鸟行、
    视频封面、补种脚本全部一致生效。

    规则：demote 条目在原位置换成 keep 条目（保留该位置的置信度与
    region_match——置信度此时代表模型对这对姊妹复合体的把握）；keep 物种
    若原本也上榜则删除其低置信旧条目避免重复；keep 已自然排在前面的场合
    只删除后面的 demote 条目。

    Substitution happens at the identify_bird exit so every consumer (main
    adoption, per-bird rows, video covers, backfills) stays consistent. The
    demoted entry is replaced in place by the keep entry, carrying over its
    confidence and region_match; a naturally-ranked keep entry keeps its own
    original position, with later demote entries simply dropped.

    参数 / Parameters:
        results (list): predict_bird 的输出 / predict_bird output.
        whitelist (dict): load_species_whitelist() 的映射 / The loaded map.
        photo_country_code (Optional[str]): 拍摄国，供稀有度现查 /
            Shooting country for the rarity lookup.
        rarity_fn (Optional[Callable]): (class_id, country) → gbif_rarity_100，
            缺省不查 / Rarity lookup callback, omitted when None.

    返回 / Returns:
        list[dict]: 替换后的结果（原列表不被修改）/ The new list.
    """
    if not results or not whitelist:
        return results
    keep_cids = {info["class_id"] for info in whitelist.values()}
    out: List[Dict] = []
    for r in results:
        cid = r.get("class_id")
        info = whitelist.get(cid) if cid is not None else None
        if info is not None:
            if any(e.get("class_id") == info["class_id"] for e in out):
                continue  # keep 已上榜/已替换 → 丢弃重复 demote
            entry = dict(info)
            entry["confidence"] = r.get("confidence")
            entry["region_match"] = r.get("region_match", True)
            entry["whitelist_substituted"] = True
            if rarity_fn is not None:
                try:
                    entry["gbif_rarity_100"] = rarity_fn(
                        info["class_id"], photo_country_code
                    )
                except Exception:
                    pass
            print(
                f"  🔄 白名单定种 / manual whitelist: "
                f"{r.get('cn_name') or r.get('en_name')} → "
                f"{entry.get('cn_name') or entry.get('en_name')}"
            )
            out.append(entry)
        elif cid in keep_cids and any(e.get("class_id") == cid for e in out):
            continue  # keep 已由替换占据，去掉其低置信重复条目
        else:
            out.append(r)
    return out


def identify_bird(
    image_path: str,
    use_yolo: bool = True,
    use_gps: bool = True,
    use_geo_filter: bool = True,
    country_code: Optional[str] = None,
    region_code: Optional[str] = None,
    top_k: int = 5,
    name_format: Optional[str] = None,
    preloaded_crop: Optional[Image.Image] = None,
    focus_point: Optional[Tuple[float, float]] = None,
    dark_retry_conf: Optional[float] = None,
    retry_crop: Optional[Image.Image] = None,
    retry_factory=None,
) -> Dict:
    """
    识别单张照片中的鸟种，支持暗框提亮重识别。

    参数:
    image_path (str): 照片路径（GPS/EXIF 读取用）
    use_yolo (bool): 无预载裁剪时是否先 YOLO 检测裁剪
    use_gps (bool): 是否读取 GPS 做地理过滤/国家解析
    use_geo_filter (bool): 是否启用分层地理候选集
    country_code (Optional[str]): 手选国家码（无 GPS 时用）
    region_code (Optional[str]): 手选地区码
    top_k (int): 返回候选数
    name_format (Optional[str]): 鸟名格式
    preloaded_crop (Optional[Image.Image]): 上游已裁剪的鸟框图（跳过 YOLO）
    focus_point (Optional[Tuple[float, float]]): 对焦点（多目标选框）
    dark_retry_conf (Optional[float]): 备选渲染重试的置信度线（0-100，
        与 results[].confidence 同口径）。None=关闭重试。启用后的重试面
        （V5.9.5）：提供 retry_crop 时暗版对照**不限采纳线**（总是对照、
        净胜裕度内翻盘——提亮版可能把错误种推过线）；未提供时该线门控
        惰性工厂与内存伽马重试（结果缺失或 top1 低于线才触发）。动机：
        逆光/林荫猛禽常「框检出但太黑定不了种」，与 DPP 伽马批次的补种
        经验一致——提亮/暗版对照看得更清。
    retry_crop (Optional[Image.Image]): 暗片原始渲染的同位框图（V5.9.2
        双渲染对比）。提供时与亮版**总是对照一次**（V5.9.5 起不限采纳
        线），净胜裕度内翻盘——暗图直判实测本就有 65-72% 正确率的样本；
        未提供时回退「压死证据门控 + 内存伽马提亮重试」。框图坐标由
        调用方按两版分辨率换算（共享实现：
        tools.find_bird_util.build_dark_retry_crop，主鸟与多鸟次要鸟
        同源）。
    retry_factory (Optional[Callable]): 惰性重试图工厂（V5.9.3 平坦雾片
        救援）。首判低于采纳线且 retry_crop 未提供时调用一次，返回
        (PIL框图, retry_kind) 或 None——开发 RAW（~2s）只在真正需要
        重试时发生，不在批处理主循环白白付费。retry_kind 用于上层
        落复核图命名（如 raw_stretch）。

    返回:
    Dict: success/results/yolo_info/gps_info/geo_info/error；重试且净胜
        时额外带 brightened_retry（orig_conf/bright_conf/retry_kind=
        dark_original|raw_stretch|gamma 等）与 brightened_crop（胜出的
        PIL 框图，供上层保存复核缩略图）。

    Identify the bird species in one photo, with a brightened-crop retry
    for dark frames.

    Parameters:
    image_path (str): photo path (for GPS/EXIF reads)
    use_yolo (bool): run YOLO detection when no preloaded crop
    use_gps (bool): read GPS for geo filter / country resolution
    use_geo_filter (bool): layered geographic candidate sets
    country_code (Optional[str]): user-selected country (no-GPS case)
    region_code (Optional[str]): user-selected region
    top_k (int): number of candidates to return
    name_format (Optional[str]): name format
    preloaded_crop (Optional[Image.Image]): upstream crop (skips YOLO)
    focus_point (Optional[Tuple[float, float]]): focus point for subject selection
    dark_retry_conf (Optional[float]): confidence line (0-100, same scale
        as results[].confidence) gating the alternate-rendition retry;
        None disables. With retry_crop the dark compare runs regardless
        of the line (V5.9.5 — the brightened pass can push a wrong
        species just past it); without it the line gates the lazy
        factory and the in-memory gamma retry (missing results or top1
        under the line). Rationale: backlit/shaded raptors often get
        "box detected, species rejected", and the DPP gamma batches
        proved the classifier reads brightened crops better.
    retry_crop (Optional[Image.Image]): co-registered crop from the
        original dark rendition (V5.9.2 dual-rendition compare). When
        provided it is ALWAYS classified against the brightened pass
        (V5.9.5 lifted the below-line gate) and replaces it only on a
        net win by the margin — the dark direct pass alone measures
        65-72% correct on real samples; without it the caller-side
        crush-evidence + in-memory gamma retry applies. Crops are built
        by the shared tools.find_bird_util.build_dark_retry_crop for
        main and secondary birds alike.
    retry_factory (Optional[Callable]): lazy retry-image factory
        (V5.9.3 flat-fog rescue), called once below the line when no
        retry_crop was given; returns (PIL crop, retry_kind) or None —
        the ~2s RAW develop is paid only when actually needed.

    Return:
    Dict: success/results/yolo_info/gps_info/geo_info/error; when the
        brightened retry wins, the dict also carries brightened_retry
        (old/new confidence etc.) and brightened_crop (the brightened PIL
        crop, for callers persisting a brightened thumbnail).
    """
    result = {
        "success": False,
        "image_path": image_path,
        "results": [],
        "yolo_info": None,
        "gps_info": None,
        "geo_info": None,
        "error": None,
    }

    try:
        is_yolo_cropped = False
        if preloaded_crop is not None:
            image = preloaded_crop
            is_yolo_cropped = True
            result["yolo_info"] = {"preloaded": True}
        else:
            image = load_image(image_path)

        if preloaded_crop is None and use_yolo and YOLO_AVAILABLE:
            # 未显式传对焦点时，若输入是 RAW 则自动读取，用于多目标选框
            # （与选鸟模式对齐）。JPEG 通常无对焦点，focus_point 保持 None。
            # Auto-read the focus point from RAW when not explicitly provided, so
            # multi-subject selection matches the main picking pipeline.
            if focus_point is None:
                focus_point = _read_focus_point_for_path(image_path)
            width, height = image.size
            if max(width, height) > 640:
                detector = get_yolo_detector()
                if detector:
                    cropped, info = detector.detect_and_crop_bird(
                        image, focus_point=focus_point
                    )
                    if cropped:
                        image = cropped
                        result["yolo_info"] = info
                        result["cropped_image"] = cropped
                        is_yolo_cropped = True
                    else:
                        result["success"] = True
                        result["results"] = []
                        result["yolo_info"] = {"bird_count": 0}
                        return result

        lat = lon = None
        photo_country_code: Optional[str] = None

        # V4.2.7: 提前提取 GPS（无论是否启用 ebird 过滤），用于反查拍摄国家
        # → 为 GBIF 按国家归一化 rarity 提供输入
        # V4.2.7: Extract GPS upfront (regardless of ebird filter) so we can
        # reverse-geocode the shooting country for country-aware GBIF rarity.
        if use_gps:
            try:
                lat, lon, _gps_msg = extract_gps_from_exif(image_path)
                # 0.0 是合法坐标（赤道/本初子午线），必须用 is not None 语义判断
                # 0.0 is a legal coordinate (equator/prime meridian) — presence
                # must be checked with is-not-None semantics, not truthiness.
                if _gps_coords_present(lat, lon):
                    result["gps_info"] = {
                        "latitude": lat,
                        "longitude": lon,
                        "info": _gps_msg,
                    }
                    photo_country_code = _resolve_country_code_from_gps(lat, lon)
                    if photo_country_code:
                        result["gps_info"]["country_code"] = photo_country_code
            except Exception:
                pass

        # 无 GPS 时地理过滤的默认国家：本应用面向中国鸟友，识别候选集对
        # 无 GPS 照片默认从 CN 起步（分层放宽可自愈）。注意：稀有度写入
        # 不使用这个默认——无证据时猜国家会持久化错误分数（伦敦/新加坡的
        # 照片会被误标中国口径）。
        # Default country for the GEO FILTER when no GPS: the app targets
        # Chinese birders and candidate-set guessing self-corrects via tier
        # widening. Rarity writes deliberately do NOT use this default —
        # guessing a country would persist wrong scores (a London/Singapore
        # photo would be mislabeled as CN-scoped).
        DEFAULT_COUNTRY_NO_GPS = "CN"

        # V4.4: 稀有度国家解析链：GPS 反解 → 手选国家 → 放弃（全球分）。
        # 此前稀有度严格依赖 GPS 反解，无 GPS 的照片永远拿全球分，中国
        # 口径对无 GPS 工作流完全不生效。刻意不做「默认中国」兜底：稀有度
        # 会写入数据库，猜测的代价是持久错误数据；让用户在设置里把手选
        # 国家设为 CN 即可让国内无 GPS 照片吃到中国分。
        # result["gps_info"]["country_code"] 仍保留纯 GPS 反解真值（原样
        # 展示/导出），这里的解析只影响 gbif_rarity_by_country 的查询国家。
        # V4.4: Rarity country chain: GPS-derived → user-selected → give up
        # (global score). No silent CN default on purpose: rarity is
        # persisted, and a guessed country would mislabel overseas photos.
        # Users shooting GPS-less in China should set the country to CN in
        # settings. gps_info keeps the raw GPS-derived value; this chain
        # only affects which country row the rarity lookup uses.
        photo_country_code = photo_country_code or country_code

        # 地理过滤：分层候选集逐层放宽，替代旧的「一次性候选 + 三级断裂兜底」。
        # 无 GPS 时用用户手选的地区/国家从 L4 起步；两者都缺则默认中国（CN），
        # 兜底仍为完全无过滤。
        # Geo filter: layered candidates widened tier by tier, replacing the old
        # single candidate set with three disconnected fallbacks. Without GPS we
        # start at L4 using the user's chosen region/country; lacking both, we
        # default to China (CN). The final fallback is still unfiltered.
        effective_region = (
            region_code
            or country_code
            or photo_country_code
            or DEFAULT_COUNTRY_NO_GPS
        )
        def _classify_once(img: Image.Image) -> Tuple[List[Dict], Dict]:
            """
            用同一套地理/命名参数对给定图跑一次分类。

            Run one classification pass with the same geo/naming setup.

            参数 / Parameters:
                img (Image.Image): 待分类图（原图或提亮框图）/ image to
                                  classify (original or brightened crop).

            返回 / Returns:
                Tuple[List[Dict], Dict]: (候选列表, geo_info 字典)
                                        (candidates, geo_info dict).
            """
            if use_geo_filter:
                _results, tier, count = _identify_with_tiers(
                    img,
                    top_k=top_k,
                    lat=lat if use_gps else None,
                    lon=lon if use_gps else None,
                    country_code=effective_region,
                    is_yolo_cropped=is_yolo_cropped,
                    name_format=name_format,
                    photo_country_code=photo_country_code,
                )
                _geo = {
                    "enabled": tier != TIER_NONE,
                    "tier": tier,
                    "species_count": count,
                    "country_code": effective_region,
                }
            else:
                _results = predict_bird(
                    img,
                    top_k=top_k,
                    species_class_ids=None,
                    is_yolo_cropped=is_yolo_cropped,
                    name_format=name_format,
                    photo_country_code=photo_country_code,
                )
                _geo = {
                    "enabled": False,
                    "tier": TIER_NONE,
                    "species_count": None,
                    "country_code": None,
                }
            return _results, _geo

        results, geo_info = _classify_once(image)
        result["geo_info"] = geo_info

        # V5.9.2/V5.9.3/V5.9.5: 备选渲染重试。暗片管线（raw_to_jpeg RAW 级
        # 提亮）下 preloaded_crop 是提亮版框图；原始暗渲染被另存为
        # <前缀>_dark.jpg 且由调用方裁成同位框图传入 retry_crop（主鸟与
        # 多鸟次要鸟同享共享裁剪实现）。平坦雾片（全图灰阶跨度 <45）由
        # 调用方传 retry_factory 惰性开发 RAW + 框内拉伸重判。实测
        # （2026-09-30）：提亮版/内嵌预览分类可能崩（7%/16%），备选渲染
        # 本就正确率更高；提亮版也可能把错误种推过采纳线（41% 错判无
        # 对照）——故预载暗版存在时对照**不限采纳线**（V5.9.5），净胜
        # 裕度内择优；惰性工厂与内存伽马维持「低于采纳线才触发」的成本/
        # 风险口径。无备选（背光护栏豁免/纯 JPEG）时回退 V5.9.1 的
        # 「压死证据 + 内存伽马」重试。
        # V5.9.2/V5.9.3/V5.9.5: alternate-rendition retry. With the
        # RAW-grade pipeline the preloaded_crop is the BRIGHTENED crop;
        # the original dark rendition is preserved as <prefix>_dark.jpg
        # and cropped co-registered by the caller into retry_crop (main
        # and secondary birds share the same builder). Flat fog frames
        # (gray span <45) get a retry_factory that lazily develops the
        # RAW with an in-box contrast stretch. Measured: the
        # brightened/embedded pass can collapse (7%/16%) while alternates
        # ID better — and it can also push a wrong species just past the
        # adoption line, so an eager dark crop compares regardless of
        # the line (V5.9.5) and wins only by the margin, while the lazy
        # factory and the gamma stay below-line-only. Without alternates
        # (highlight-guard frames, pure JPEGs) fall back to the V5.9.1
        # crush-evidence + in-memory gamma retry.
        if dark_retry_conf is not None:
            from tools.tone_curve import (
                BRIGHTEN_WIN_MARGIN, DEFAULT_CRUSH_P95, DEFAULT_DARK_MEAN,
                DEFAULT_MIN_GAMMA, compute_brighten_gamma,
            )
            _top_conf = (float(results[0].get("confidence") or 0.0)
                         if results else None)
            # V5.9.5: 双渲染对照扩展到采纳线之上——提亮版可能把错误种推过
            # 采纳线（41% 错判此前无对照机会）。预载暗版（retry_crop）存在时
            # 总是对照一次，净胜裕度内才翻盘（此时挑战者 ≥ 原判定+3 ≥ 采纳
            # 线+3，天然是可采纳候选）；惰性工厂（~2s RAW 开发）与内存伽马
            # 维持「低于采纳线才触发」的成本/风险口径不变。
            # V5.9.5: the dual-rendition compare now also runs ABOVE the
            # adoption line — the brightened pass can push a wrong species
            # just past it. With an eager dark crop the compare always
            # runs and only wins by the margin (the challenger is then
            # ≥ original+3 ≥ line+3, a valid candidate); the lazy factory
            # (~2s RAW develop) and the in-memory gamma stay below-line.
            _below_line = (_top_conf is None
                           or _top_conf < float(dark_retry_conf))
            _needs_retry = True if retry_crop is not None else _below_line
            if _needs_retry:
                _o_conf = (_top_conf if _top_conf is not None else -1.0)
                # 备选渲染解析：预载暗版优先，其次惰性工厂（平坦雾片的
                # RAW 开发 ~2s，只在首判失败时才付费），最后回退伽马。
                # Alternate rendition resolution: eager dark crop first,
                # then the lazy factory (the ~2s RAW develop for flat fog
                # frames is paid only when the first pass fails), then
                # the gamma fallback.
                _alt_crop, _alt_kind = retry_crop, "dark_original"
                if _alt_crop is None and retry_factory is not None:
                    try:
                        _made = retry_factory()
                        if _made is not None:
                            _alt_crop, _alt_kind = _made
                    except Exception:
                        _alt_crop = None
                if _alt_crop is not None:
                    # 备选渲染直判重试（无色调操作，坐标同位由调用方保证）
                    # Alternate-rendition retry (no tone op; the caller
                    # keeps the crop co-registered).
                    _r_results, _r_geo = _classify_once(_alt_crop)
                    _r_conf = (float(_r_results[0].get("confidence") or 0.0)
                               if _r_results else -1.0)
                    if _r_conf >= _o_conf + BRIGHTEN_WIN_MARGIN:
                        results = _r_results
                        result["brightened_retry"] = {
                            "orig_conf": _top_conf,
                            "bright_conf": _r_conf,
                            "retry_kind": _alt_kind,
                        }
                        result["brightened_crop"] = _alt_crop
                else:
                    # V5.9: 内存伽马重识别（回退路径）。门控为「压死证据」
                    # 双条件——均值暗 **且** p95 低（高光饥荒）。实测校准：
                    # 正确曝光的黑鸟（乌鸫/乌鸦）框均值也低，但羽轴/眼点
                    # 留有高光拖尾（p95≈173-203），提亮无益且可能翻错种；
                    # 压死暗片 p95≈87-91 才是适用面。伽马在压死片上可能
                    # 把 66% 正确判定崩到 16%，故须净胜才翻盘。
                    # V5.9: in-memory gamma retry (fallback path), gated by
                    # two-signal "crush evidence" — dark mean AND low p95
                    # (highlight starvation). Calibrated on real library
                    # samples: correctly-exposed black birds keep a plumage
                    # highlight tail (p95 ~173-203) and must be left alone;
                    # crushed frames (p95 ~87-91) are the target. Gamma was
                    # measured collapsing a correct 66% ID to 16% on a
                    # crushed frame, so it must win by the margin.
                    _gray = np.asarray(image.convert("L"), dtype=np.float32)
                    _crop_mean = float(_gray.mean())
                    _crop_p95 = float(np.percentile(_gray, 95))
                    if 0 <= _crop_mean < DEFAULT_DARK_MEAN \
                            and _crop_p95 < DEFAULT_CRUSH_P95:
                        _gamma = compute_brighten_gamma(
                            _crop_mean, min_gamma=DEFAULT_MIN_GAMMA)
                        if _gamma < 1.0:
                            # PIL point() 对多通道图要求「每通道各 256 项」
                            # 的表，同一 LUT 平铺三份即各通道同表提亮，不改
                            # 变色相平衡，与预览提亮同一套数学。
                            # PIL point() needs 256 entries PER BAND for
                            # multi-band images; tiling the same LUT three
                            # times brightens every channel identically —
                            # hue balance preserved, same math as the
                            # preview brightening.
                            lut = np.array(
                                [((i / 255.0) ** _gamma) * 255.0
                                 for i in range(256)],
                                dtype=np.uint8,
                            )
                            # point() 的 LUT 长度按通道数校验，先归一到
                            # RGB，防御 RGBA/L 等非常规模式（正常链路均为
                            # RGB 框图）。
                            # point() validates the LUT length against the
                            # band count; normalize to RGB first to be safe
                            # against exotic modes (the normal path always
                            # sees RGB crops).
                            _retry_src = (image if image.mode == "RGB"
                                          else image.convert("RGB"))
                            _bright = _retry_src.point(
                                np.tile(lut, 3).tolist())
                            _b_results, _b_geo = _classify_once(_bright)
                            _b_conf = (float(_b_results[0]
                                             .get("confidence") or 0.0)
                                       if _b_results else -1.0)
                            # 净胜裕度：有原判定时必须明显更好才翻盘（防
                            # +1 噪声翻转）。
                            # Win margin: must clearly beat the original
                            # (guards against +1 noise flips).
                            if _b_conf >= _o_conf + BRIGHTEN_WIN_MARGIN:
                                results = _b_results
                                result["brightened_retry"] = {
                                    "orig_conf": (_top_conf
                                                  if _top_conf is not None
                                                  else None),
                                    "bright_conf": _b_conf,
                                    "retry_kind": "gamma",
                                    "crop_mean": _crop_mean,
                                    "crop_p95": _crop_p95,
                                    "gamma": _gamma,
                                }
                                result["brightened_crop"] = _bright

        # 人工定种白名单：用户核定的易混种对（欧亚喜鹊→喜鹊等）在此出口
        # 统一替换。放最后一层是为了让主鸟采纳、多鸟行、视频封面、补种
        # 等所有消费方一致生效；替换失败只跳过，不影响识别结果本身。
        # Manual species whitelist: user-curated confusable pairs are
        # substituted at this single exit so every consumer stays consistent.
        # A substitution failure is skipped, never fatal to the result.
        try:
            _wl = load_species_whitelist()
            if _wl:
                _db = get_database_manager()
                results = _apply_species_whitelist(
                    results,
                    _wl,
                    photo_country_code=photo_country_code,
                    rarity_fn=(
                        lambda cid, cc: _db.get_gbif_rarity_by_class_id(cid, cc)
                        if _db
                        else None
                    ),
                )
        except Exception:
            pass

        result["success"] = True
        result["results"] = results

    except Exception as e:
        result["error"] = str(e)

    return result


def quick_identify(image_path: str, top_k: int = 3) -> List[Dict]:
    result = identify_bird(image_path, top_k=top_k)
    return result.get("results", [])


if __name__ == "__main__":
    print("BirdIdentifier 模块测试")
    print(f"YOLO 可用: {YOLO_AVAILABLE}")
    print(f"RAW 支持: {RAW_SUPPORT}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据库路径: {DATABASE_PATH}")
