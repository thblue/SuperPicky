import os

# 禁用 Ultralytics 逐次推理日志。必须在 import ultralytics 之前设置——
# ultralytics 在自身 import 时读取该环境变量,之后再设无效。
# Disable Ultralytics per-inference logging. Must be set BEFORE importing
# ultralytics — it reads this env var at its own import time.
os.environ.setdefault('YOLO_VERBOSE', 'False')

import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple
from tools.utils import log_message
from config import config, get_lazy_registry, ensure_cv2_thread_pool

# ultralytics 导入时会全局 cv2.setNumThreads(0)，立即恢复线程池
# ultralytics globally disables the cv2 thread pool at import; restore it
ensure_cv2_thread_pool()
# V3.2: 移除未使用的 sharpness 计算器导入
from core.iqa_scorer import get_iqa_scorer
from advanced_config import get_advanced_config
# V4.2.1
from tools.i18n import get_i18n


def load_yolo_model(log_callback=None):
    """加载 YOLO 模型（使用最佳计算设备）"""
    model_path = os.path.abspath(config.ai.get_model_path())
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")
    model = YOLO(str(model_path))

    # 使用统一的设备检测逻辑
    try:
        from config import get_best_device
        device = get_best_device()
        i18n = get_i18n()
        
        # 使用 i18n 翻译设备类型消息
        if device.type == 'mps':
            msg = i18n.t("ai.using_mps")
        elif device.type == 'cuda':
            msg = i18n.t("ai.using_cuda")
        else:
            msg = i18n.t("ai.using_cpu")
        
        # 使用日志回调或直接打印
        if log_callback:
            log_callback(msg, "info")
        else:
            print(msg)
    except Exception as e:
        i18n = get_i18n()
        error_msg = i18n.t("ai.device_detection_failed", error=str(e))
        if log_callback:
            log_callback(error_msg, "warning")
        else:
            print(error_msg)

    return model


def read_image_bgr(image_path: str) -> Optional[np.ndarray]:
    """以 BGR 格式读取图像，兼容 Windows 中文路径。"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def read_image_dims(image_path: str) -> Optional[Tuple[int, int]]:
    """
    仅解析图像头部获取宽高 (w, h)，不解码像素数据。

    用于 bbox 坐标换算等只需要尺寸的场景：45MP JPEG 整图解码约
    200-400ms，而 PIL 惰性打开只读头部（约 1ms），且不占用解码
    内存。HEIF 需 pillow-heif 已注册才能识别，未注册时抛异常由
    调用方回退。

    参数:
    image_path (str): 图像文件路径

    返回:
    Optional[Tuple[int, int]]: (宽, 高)；头部无法解析时 None

    Read only the image header for (width, height) without decoding
    pixels — for size-only consumers such as bbox coordinate scaling.
    A 45MP JPEG full decode costs 200-400ms while the header probe is
    ~1ms. Returns None when the header cannot be parsed.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None


def preprocess_image(
    image_path: str,
    target_size: Optional[int] = None,
    source_image: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """预处理图像，允许复用上游已解码的 BGR 图像。"""
    if target_size is None:
        target_size = config.ai.TARGET_IMAGE_SIZE

    img = source_image if source_image is not None else read_image_bgr(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    if new_size == (w, h):
        return img.copy()
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


# V3.2: 移除 _get_sharpness_calculator（锐度现在由 keypoint_detector 计算）

def _get_iqa_scorer():
    """获取 IQA 评分器单例"""
    from config import get_best_device
    registry = get_lazy_registry()
    key = f"ai_model.iqa_scorer::{get_best_device().type}"
    return registry.get_or_create(key, lambda: get_iqa_scorer(device=get_best_device().type))


def _get_rescue_birdid():
    """
    获取补救确认用的 BirdID 适配器单例（懒加载，经 lazy registry 管理）。

    返回:
    BirdIDAdapter: 适配器实例（底层模型与批处理识鸟共享，无重复显存）

    Lazily get the BirdID adapter singleton for rescue confirmation via the
    lazy registry; the underlying model is shared with batch bird-ID.
    """
    registry = get_lazy_registry()

    def _factory():
        from core.birdid_adapter import BirdIDAdapter
        return BirdIDAdapter()

    return registry.get_or_create("ai_model.rescue_birdid_adapter", _factory)


def _load_fullres_for_confirm(
    image: np.ndarray, image_path: Optional[str], box,
) -> tuple:
    """
    为识鸟守门准备全分辨率裁剪源。

    1024 预处理图上的极小目标（占画面 0.1% 量级）候选框只有几十像素，
    分类器无法辨认，是「检出但救不回」漏检的主因（2026-09 天坛/海淀
    实测：同一目标 1024 图裁剪 9% vs 原图裁剪 82%）。本函数把候选框按
    预处理图→原图的比例缩放，返回原图与其上的框。

    参数:
    image (np.ndarray): 已预处理的 BGR 图（长边 1024）
    image_path (Optional[str]): 预览图文件路径（内嵌全分辨率 JPEG）
    box: 候选框 (x1, y1, x2, y2)，image 坐标系

    返回:
    tuple: (full_image, scaled_box)；无更高分辨率/读取失败时返回
           (None, None)，调用方回退旧的 1024 图裁剪路径

    Prepare a full-resolution crop source for the rescue BirdID gate.
    Returns (full_image, scaled_box), or (None, None) to fall back to
    the legacy 1024-crop path when no higher resolution is available.
    """
    if not image_path:
        return None, None
    try:
        full = read_image_bgr(image_path)
        if full is None:
            return None, None
        fh, fw = full.shape[:2]
        ih, iw = image.shape[:2]
        if fw <= iw or fh <= ih:
            return None, None
        sx, sy = fw / iw, fh / ih
        x1, y1, x2, y2 = box
        scaled = (
            max(0, min(fw - 1, int(round(x1 * sx)))),
            max(0, min(fh - 1, int(round(y1 * sy)))),
            max(0, min(fw, int(round(x2 * sx)))),
            max(0, min(fh, int(round(y2 * sy)))),
        )
        if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
            return None, None
        return full, scaled
    except Exception:
        return None, None


def _birdid_confirm(image: np.ndarray, xyxy,
                    full_image: Optional[np.ndarray] = None,
                    xyxy_full=None) -> tuple:
    """
    把候选框裁下来交给 BirdID 分类器确认是否为鸟。

    参数:
    image (np.ndarray): BGR 整图（长边 1024 预处理后）
    xyxy: 候选框 (x1, y1, x2, y2)，位于 image 坐标系
    full_image (Optional[np.ndarray]): 全分辨率 BGR 图；与 xyxy_full 成对
        提供时优先从原图裁剪（极小目标在 1024 图上只剩几十像素，分类器
        无法辨认——2026-09 漏检根因修复）
    xyxy_full: 候选框在 full_image 坐标系中的位置

    返回:
    tuple[str, float]: (top1 鸟种名, 置信度百分比 0-100)；失败返回 ("", 0.0)

    Crop the candidate box and ask the BirdID classifier whether it is a
    bird. When a full-resolution source is provided (with the box scaled
    into its coordinates) the crop comes from there — tiny targets on the
    1024 image leave the classifier only a few dozen pixels. Returns
    (top1 species name, confidence percent 0-100); ("", 0.0) on any
    failure (model missing, load error) so the caller degrades gracefully.
    """
    try:
        adapter = _get_rescue_birdid()
        if full_image is not None and xyxy_full is not None:
            res = adapter.identify(full_image, top_k=1,
                                   bbox=tuple(int(v) for v in xyxy_full))
        else:
            res = adapter.identify(image, top_k=1,
                                   bbox=tuple(int(v) for v in xyxy))
    except Exception:
        return "", 0.0
    if not res:
        return "", 0.0
    top = res[0]
    return (top.name_zh or top.name_en or ""), top.confidence * 100.0


def _rescue_scan(model, image: np.ndarray, accept_conf: float,
                 birdid_gate: int, dir, i18n,
                 image_path: Optional[str] = None) -> Optional[dict]:
    """
    无鸟补救扫描：1024px 低阈值重扫 + BirdID 分类器守门（V4.6）。

    第一遍 640 检测低于 UI 阈值时调用。规则：
    1. 重扫最佳 bird 置信度 >= accept_conf → 直接救回；
    2. 否则取最佳候选框（弱 bird，或 airplane/kite 混淆类）交 BirdID 确认，
       top1 置信度 >= birdid_gate(%) → 救回；
    3. 都不满足 → None，维持原拒绝结果。

    原图瓦片检测已独立为每张照片必跑的常规通道 `_tile_detect_pass`
    （V5.8，含三重门槛），不再挂在补救路径下——补救只管 1024 整图上
    的低置信候选，瓦片管 1024 分辨率看不见的原生小目标。

    参数:
    model: 共享的 YOLO 模型实例（调用方已持有 yolo_infer_lock）
    image (np.ndarray): 已预处理 BGR 图（长边 1024）
    accept_conf (float): UI「AI 置信度」阈值 (0-1)
    birdid_gate (int): 弱候选识鸟确认门槛（百分比 0-100）
    dir: 日志目录
    i18n: I18n 实例（可为 None）
    image_path (Optional[str]): 预览图文件路径（内嵌全分辨率 JPEG）；
        提供时规则 2 的守门裁剪优先从原图取框（1024 图上极小目标的
        裁剪像素过少，分类器必然低分——2026-09 漏检根因修复）

    返回:
    Optional[dict]: 救回时含 xyxy/conf/mask/source/species/species_conf
                    （坐标均为预处理图坐标系），否则 None

    No-bird rescue scan (V4.6): low-threshold 1024 rescan with the BirdID
    classifier as gatekeeper. The tiled full-res detection now lives in
    the universal `_tile_detect_pass` (V5.8) and runs for every photo.
    Returns the rescued candidate dict (coordinates in the preprocessed
    frame) or None.
    """
    t = i18n.t if i18n else get_i18n().t

    def _evaluate(xyxy, confs, clss, masks_np,
                  log_prefix: str) -> Optional[dict]:
        """
        对一轮候选数组执行规则 1/2（直接采纳 / 识鸟守门），返回救回字典。

        阶段 1（1024 整图）与阶段 2（瓦片重扫）共用同一套裁决逻辑，
        log_prefix 区分日志键（rescue / rescue_tile）。

        Evaluate one round of candidate arrays with rescue rules 1/2;
        shared by both stages, log_prefix selects the i18n log keys.
        """

        def _mask_of(i: int):
            if masks_np is not None and i < len(masks_np):
                return masks_np[i]
            return None

        def _result(i: int, source: str, species: str = "",
                    species_conf: float = 0.0) -> dict:
            # V5.0(multibird): 救回时带回重扫的全部鸟框（含救回候选），让
            # 调用方重建完整的 all_birds——鸟群照第一遍 640 分辨率常整体
            # 漏检，补救扫描是它们唯一的检测来源，只带回最佳一只会让逐鸟
            # 分类拿到 bird_count=1 永远不触发。带回列表按
            # RESCUE_MULTIBIRD_MIN_CONF (0.2) 过滤碎小误检框；混淆类候选
            # （airplane/kite）不在鸟类索引中，追加在末尾，由调用方统一按
            # 鸟类处理（已过识鸟守门）。
            # V5.0: carry every rescanned bird box back (conf >= 0.2 floor
            # against junk fragments) so the caller can rebuild the full
            # all_birds list; distant flocks are often only detected by
            # this rescan. A confusable-class rescue candidate is appended
            # (it already passed the BirdID gate).
            keep = [int(j) for j in bird_ix
                    if confs[j] >= config.ai.RESCUE_MULTIBIRD_MIN_CONF]
            if i not in keep:
                keep.append(i)
            return {
                "xyxy": xyxy[i], "conf": float(confs[i]),
                "mask": _mask_of(i),
                "source": source, "species": species,
                "species_conf": species_conf,
                "detections": xyxy[keep],
                "detection_confs": confs[keep],
                "detection_masks": (masks_np[keep]
                                    if masks_np is not None else None),
            }

        # 规则 1：重扫 bird 直接过 UI 阈值
        # Rule 1: rescanned bird clears the UI threshold
        cand_i, source = None, ""
        bird_ix = np.flatnonzero(clss == config.ai.BIRD_CLASS_ID)
        if bird_ix.size:
            j = int(bird_ix[confs[bird_ix].argmax()])
            if confs[j] >= accept_conf:
                log_message(t(f"logs.{log_prefix}_direct",
                              conf=f"{confs[j]:.2f}"), dir)
                return _result(j, "bird")
            cand_i, source = j, "bird"

        # 规则 2：弱 bird 或 airplane/kite 混淆候选，识鸟守门
        # Rule 2: weak bird or airplane/kite confusable candidate,
        # gated by the BirdID classifier
        if cand_i is None:
            conf_ix = np.flatnonzero(
                np.isin(clss, list(config.ai.RESCUE_CONFUSABLE_CLASS_IDS)))
            if conf_ix.size:
                j = int(conf_ix[confs[conf_ix].argmax()])
                cand_i = j
                source = config.ai.RESCUE_CONFUSABLE_CLASS_IDS[int(clss[j])]
        if cand_i is None:
            return None

        full_image, xyxy_full = _load_fullres_for_confirm(
            image, image_path, xyxy[cand_i])
        if full_image is not None and xyxy_full is not None:
            species, species_conf = _birdid_confirm(image, xyxy[cand_i],
                                                    full_image, xyxy_full)
        else:
            species, species_conf = _birdid_confirm(image, xyxy[cand_i])
        if species_conf >= birdid_gate:
            log_message(t(f"logs.{log_prefix}_confirmed", source=source,
                          species=species, conf=f"{species_conf:.0f}"), dir)
            return _result(cand_i, source, species, species_conf)
        return None

    # 阶段 1：1024 整图低阈值重扫（V4.6 原逻辑）
    # Stage 1: low-threshold rescan of the whole 1024 image (V4.6)
    try:
        from config import get_best_device
        device = get_best_device()
        results = model(image, imgsz=config.ai.RESCUE_IMGSZ,
                        conf=config.ai.RESCUE_CONF, device=device.type,
                        verbose=False)
    except Exception:
        return None

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        masks_np = None
        if getattr(results[0], "masks", None) is not None:
            masks_np = results[0].masks.data.cpu().numpy()
        del results
        rescued = _evaluate(xyxy, confs, clss, masks_np, "rescue")
        if rescued is not None:
            return rescued
    else:
        del results
    return None


def _rescue_tile_scan(model, image_path: str,
                      proc_hw: Tuple[int, int]) -> Optional[tuple]:
    """
    瓦片检测原语（V5.7 引入，V5.8 起由 _tile_detect_pass 调用）。

    把原图按 RESCUE_TILE_SIZE 边长、RESCUE_TILE_OVERLAP 重叠的瓦片网格
    逐块推理（imgsz=RESCUE_TILE_IMGSZ，与瓦片边长一致 → 长边 1:1 不缩放，
    小目标保持原生像素量级），框坐标偏移回全图后做跨瓦贪心 IoU 去重
    （重叠区里同一只鸟只留一框），最后按预处理图/原图比例映射回预处理图
    坐标系。只做检测与坐标归一，不过滤、不识别——门槛在 _tile_detect_pass。

    资源约束：瓦片数超 RESCUE_TILE_MAX_TILES 时自动放大步长（减小重叠）
    压缩到上限内；掩码不跨瓦合并（各瓦掩码网格彼此独立），统一返回
    None——救回框多边形可由调用方确定性重算，不影响入库。

    参数:
    model: YOLO 模型实例（调用方已持有 yolo_infer_lock）
    image_path (str): 预览图文件路径（内嵌全分辨率 JPEG）
    proc_hw (Tuple[int, int]): 预处理图 (height, width)，返回坐标目标系

    返回:
    Optional[tuple]: (xyxy(N,4), confs(N,), clss(N,), None)；无候选或
                     小图（长边 <= 瓦片边长，阶段 1 已覆盖）返回 None

    Stage 2: tiled full-resolution scan. Runs YOLO per overlapping tile
    at native resolution, dedupes boxes across tiles with greedy IoU NMS,
    then maps coordinates back into the preprocessed frame. Mask data is
    not merged across tiles (None).
    """
    full = read_image_bgr(image_path)
    if full is None:
        return None
    fh, fw = full.shape[:2]
    tile = max(256, int(config.ai.RESCUE_TILE_SIZE))
    if max(fh, fw) <= tile:
        return None
    overlap = min(0.5, max(0.0, float(config.ai.RESCUE_TILE_OVERLAP)))
    step = max(32, int(tile * (1.0 - overlap)))

    def _starts(total: int, cur_step: int) -> list:
        # 覆盖整轴的瓦片起点：0, step, 2*step...，末尾对齐右/下边界
        # Tile start offsets covering the axis, end-aligned to the border.
        if total <= tile:
            return [0]
        starts = list(range(0, total - tile + 1, cur_step))
        if starts[-1] != total - tile:
            starts.append(total - tile)
        return starts

    xs, ys = _starts(fw, step), _starts(fh, step)
    max_tiles = max(1, int(config.ai.RESCUE_TILE_MAX_TILES))
    while len(xs) * len(ys) > max_tiles and step < max(fh, fw):
        step = int(step * 1.5) + 1
        xs, ys = _starts(fw, step), _starts(fh, step)

    from config import get_best_device
    device = get_best_device()
    xyxy_parts, conf_parts, cls_parts = [], [], []
    for y0 in ys:
        for x0 in xs:
            th, tw = min(tile, fh - y0), min(tile, fw - x0)
            tile_img = full[y0:y0 + th, x0:x0 + tw]
            try:
                results = model(tile_img,
                                imgsz=config.ai.RESCUE_TILE_IMGSZ,
                                conf=config.ai.RESCUE_CONF,
                                device=device.type, verbose=False)
            except Exception:
                continue
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                del results
                continue
            xyxy_parts.append(
                boxes.xyxy.cpu().numpy()
                + np.array([x0, y0, x0, y0], dtype=np.float32))
            conf_parts.append(boxes.conf.cpu().numpy())
            cls_parts.append(boxes.cls.cpu().numpy().astype(int))
            del results
    if not xyxy_parts:
        return None
    detections = np.concatenate(xyxy_parts).astype(np.float64)
    confidences = np.concatenate(conf_parts)
    class_ids = np.concatenate(cls_parts)
    # 跨瓦贪心 IoU 去重：重叠瓦片里同一只鸟只保留置信度最高的一框
    # Greedy IoU dedupe so a bird straddling two tiles keeps one box.
    detections, confidences, class_ids, _ = _dedupe_bird_boxes(
        detections, confidences, class_ids, None)
    if len(detections) == 0:
        return None
    ih, iw = int(proc_hw[0]), int(proc_hw[1])
    detections[:, [0, 2]] *= (iw / fw)
    detections[:, [1, 3]] *= (ih / fh)
    return detections, confidences, class_ids, None


def _tile_detect_pass(model, image_path: str, proc_image: np.ndarray,
                      existing_xyxy, birdid_gate: int, dir, i18n) -> list:
    """
    常规瓦片检测（V5.8）：每张照片都在原图瓦片上补找 1024 整图看不见的
    隐蔽小鸟，三重门槛过滤后返回新框。

    背景（2026-09-27 乐活中堤）：栗耳鹀/褐柳莺/红喉歌鸲等深度伪装小目标
    在 1024 整图上仅剩 ~30px，低于可检下限，整组漏检；瓦片把有效分辨率
    拉回原生量级后全部可检出。与主检测互补——大鸟由整图层检出（瓦片会
    把大鸟切碎、且跨瓦掩码不连续），瓦片专管小目标。

    三重门槛（枯叶/杂物防线）：
    a. 与已有框（主检测/补救结果）IoU > 0.55、或被已有框包含
       （inter/area(新框) > 0.55）→ 视为同一只鸟丢弃。被包含规则同时
       拦下大鸟被瓦片切出的"半鸟框"；
    b. YOLO conf < RESCUE_TILE_MIN_CONF → 直接丢弃，不送识别
       （实测该区间无真鸟）；
    c. 从原图裁剪过 BirdID，top1 < birdid_gate → 丢弃。gate=10 时代
       曾放进枯花误救，2026-09-12 定档 25。

    参数:
    model: YOLO 模型实例（与主检测同一线程上下文）
    image_path (str): 预览图文件路径（内嵌全分辨率 JPEG）
    proc_image (np.ndarray): 已预处理 BGR 图（长边 1024，坐标目标系 +
        守门裁剪的 1024 回退源）
    existing_xyxy: 已有框数组 (N,4)（proc 坐标，含全部类别），可为空
    birdid_gate (int): BirdID 守门门槛（百分比 0-100）
    dir: 日志目录
    i18n: I18n 实例（可为 None）

    返回:
    list: 通过门槛的新框字典列表（按置信度降序），每项含
          xyxy/conf/species/species_conf（species 为守门分类结果，仅供
          日志与工具脚本参考；批量链路的逐鸟分类会以更优裁剪重新识别）

    Universal tiled detection pass (V5.8): runs on every photo to find
    tiny camouflaged birds invisible at 1024, filtered by three gates
    (existing-box dedupe / conf floor / BirdID gate). Returns the kept
    boxes as dicts sorted by confidence.
    """
    if not image_path or not config.ai.RESCUE_TILE_ENABLED:
        return []
    arrays = _rescue_tile_scan(model, image_path, proc_image.shape[:2])
    if arrays is None:
        return []
    cand_xyxy, cand_confs, cand_clss, _ = arrays
    bird_ix = np.flatnonzero(cand_clss == config.ai.BIRD_CLASS_ID)
    if bird_ix.size == 0:
        return []
    t = i18n.t if i18n else get_i18n().t
    has_existing = existing_xyxy is not None and len(existing_xyxy) > 0
    full_image = None
    xyxy_full = None
    kept = []
    rejected = 0
    # 置信度降序逐个过门槛 / evaluate candidates, highest conf first
    for j in bird_ix[np.argsort(-cand_confs[bird_ix])]:
        conf = float(cand_confs[j])
        box = cand_xyxy[j]
        # 门槛 a：与已有框高度重叠或被包含 → 同一只鸟/半鸟框，丢弃
        # Gate a: duplicate or fragment of an existing detection
        if has_existing:
            dup = False
            bx1, by1, bx2, by2 = box
            b_area = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
            for e in existing_xyxy:
                if _iou_xyxy(box, e) > 0.55:
                    dup = True
                    break
                ix1 = max(bx1, e[0])
                iy1 = max(by1, e[1])
                ix2 = min(bx2, e[2])
                iy2 = min(by2, e[3])
                iw_ = max(0.0, float(ix2 - ix1))
                ih_ = max(0.0, float(iy2 - iy1))
                if b_area > 0 and (iw_ * ih_) / b_area > 0.55:
                    dup = True
                    break
            if dup:
                continue
        # 门槛 b：置信度地板，低于此值不送识别
        # Gate b: conf floor — no classifier call for junk
        if conf < config.ai.RESCUE_TILE_MIN_CONF:
            rejected += 1
            continue
        # 门槛 c：全分辨率裁剪过 BirdID 守门
        # Gate c: BirdID confirmation on the full-res crop
        if full_image is None:
            full_image, xyxy_full = _load_fullres_for_confirm(
                proc_image, image_path, box)
        if full_image is not None and xyxy_full is not None:
            species, species_conf = _birdid_confirm(proc_image, box,
                                                    full_image, xyxy_full)
        else:
            species, species_conf = _birdid_confirm(proc_image, box)
        if species_conf < birdid_gate:
            rejected += 1
            continue
        kept.append({"xyxy": box, "conf": conf,
                     "species": species, "species_conf": species_conf})
    if kept or rejected:
        detail = " · ".join(f"{b['species']} {b['species_conf']:.0f}%"
                            for b in kept) or "—"
        log_message(t("logs.tile_pass", added=len(kept), rejected=rejected,
                      detail=detail), dir)
    return kept


def _mask_to_polygon(masks, idx: int, width: int, height: int,
                     max_points: int = 32) -> Optional[list]:
    """
    把 YOLO 分割掩码简化成轮廓多边形点阵（处理图坐标）。

    掩码先按需缩放到 (width, height)（与主鸟 bird_mask 同一逻辑），
    findContours 取最大轮廓，approxPolyDP 逐步放宽 epsilon 直到
    顶点数 ≤ max_points。像素级掩码可随时由 YOLO 确定性重算，
    这里只存轻量轮廓供可视化/网站叠加。

    参数:
    masks (np.ndarray): YOLO 全部掩码 (N, h, w)，可为 None
    idx (int): 检测序号
    width / height (int): 处理图尺寸（多边形输出坐标系）
    max_points (int): 轮廓最大顶点数

    返回:
    Optional[list]: [[x, y], ...]；无掩码/空轮廓返回 None

    Simplify a YOLO segmentation mask into a contour polygon (processed
    frame coordinates). Returns None when no mask is available.
    """
    if masks is None or idx >= len(masks):
        return None
    try:
        raw = masks[idx]
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw, (width, height),
                             interpolation=cv2.INTER_NEAREST)
        binary = (raw > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        epsilon = max(1.0, perimeter * 0.005)
        for _ in range(8):
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) <= max_points:
                break
            epsilon *= 1.6
        return [[int(p[0][0]), int(p[0][1])] for p in approx]
    except Exception:
        return None


def _iou_xyxy(a, b) -> float:
    """
    计算两个 xyxy 框的 IoU（交并比）。

    参数:
    a / b: (x1, y1, x2, y2) 框坐标（np 行或序列均可）

    返回:
    float: IoU，0~1；任一框面积为 0 时返回 0

    IoU of two xyxy boxes.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, float(ix2 - ix1)), max(0.0, float(iy2 - iy1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    area_b = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe_bird_boxes(detections, confidences, class_ids, masks,
                       iou_thresh: float = 0.55):
    """
    对鸟类框做贪心 NMS 去重（同一只鸟多个框时只保留置信度最高的）。

    补救扫描用 RESCUE_CONF=0.05 的低置信度地板，YOLO 内建 NMS（IoU 0.7）
    对密集小鸟群太宽松，同一只鸟常留下两个大小相近的框。这里按置信度
    降序贪心保留，抑制与已保留框 IoU > iou_thresh 的鸟类框；非鸟类框
    原样保留。四个数组同步过滤，索引保持一致。

    参数:
    detections / confidences / class_ids / masks: 解析后的 YOLO 数组
    iou_thresh (float): 去重 IoU 阈值（相邻不重叠的两只鸟 IoU 通常 <0.4）

    返回:
    tuple: 过滤后的 (detections, confidences, class_ids, masks)

    Greedy NMS over bird-class boxes so one bird keeps one box.
    All four arrays are filtered consistently.
    """
    bird_ix = [i for i, c in enumerate(class_ids)
               if int(c) == config.ai.BIRD_CLASS_ID]
    keep_mask = np.ones(len(detections), dtype=bool)
    for a in sorted(bird_ix, key=lambda i: -float(confidences[i])):
        if not keep_mask[a]:
            continue
        for b in bird_ix:
            if b == a or not keep_mask[b]:
                continue
            if _iou_xyxy(detections[a], detections[b]) > iou_thresh:
                keep_mask[b] = False
    if bool(keep_mask.all()):
        return detections, confidences, class_ids, masks
    kept_masks = masks[keep_mask] if masks is not None else None
    return (detections[keep_mask], confidences[keep_mask],
            class_ids[keep_mask], kept_masks)


def _select_main_bird(all_birds, focus_point, width, height):
    """
    主鸟选择（三级规则，V5.1）。

    1. 单鸟 → 唯一即主鸟；
    2. 多鸟 + 对焦点命中：先精确（落在某只鸟的分割多边形内 → 该鸟），
       再宽泛（落在 bbox 内 → 该鸟）；
    3. 多鸟但对焦点未命中/无对焦点（对焦非人工控制）→ 暂取置信度最高，
       reason='fallback'，下游逐鸟分类完成后按「稀有优先 / 大而清晰」
       综合重选（core/multi_bird.select_main_bird）。

    参数:
    all_birds (List[dict]): 检测列表（含 bbox/mask_polygon/conf）
    focus_point (Optional[Tuple[float, float]]): 归一化对焦点 0-1
    width / height (int): 处理图尺寸（对焦点换算用）

    返回:
    Tuple[int, str]: (选中的 bird idx, 'single'|'focus'|'fallback')

    Main-bird selection: single → focus-point hit (polygon first,
    then bbox) → max-confidence fallback (re-selected downstream by
    the comprehensive scorer when focus was not decisive).
    """
    if not all_birds:
        return -1, 'fallback'
    if len(all_birds) == 1:
        return all_birds[0]['idx'], 'single'
    if focus_point is not None:
        fx, fy = focus_point
        pt = (float(int(fx * width)), float(int(fy * height)))
        # 规则 2a：对焦点落在分割多边形内（比 bbox 精确）
        for bird in all_birds:
            poly = bird.get('mask_polygon')
            if poly and len(poly) >= 3:
                contour = np.array(poly, dtype=np.int32)
                try:
                    if cv2.pointPolygonTest(contour, pt, False) >= 0:
                        return bird['idx'], 'focus'
                except cv2.error:
                    pass
        # 规则 2b：对焦点落在 bbox 内。仅对「没有多边形」的鸟生效——
        # 有多边形的鸟以轮廓为准（bbox 内、轮廓外 = 检测框里的背景，
        # 不能当作命中），避免大框吃掉旁边小框的焦点。
        fx_px, fy_px = int(pt[0]), int(pt[1])
        for bird in all_birds:
            poly = bird.get('mask_polygon')
            if poly and len(poly) >= 3:
                continue
            x1, y1, x2, y2 = bird['bbox']
            if x1 <= fx_px <= x2 and y1 <= fy_px <= y2:
                return bird['idx'], 'focus'
    # 规则 3：对焦未命中（对焦非人工控制）→ 暂取置信度最高
    return max(all_birds, key=lambda b: b['conf'])['idx'], 'fallback'


def detect_and_draw_birds(
    image_path,
    model,
    output_path,
    dir,
    ui_settings,
    i18n=None,
    skip_nima=False,
    focus_point=None,
    report_db=None,
    decoded_image: Optional[np.ndarray] = None,
):
    """
    检测并标记鸟类（V4.2 - 支持多鸟对焦点选择）

    Args:
        image_path: 图片路径
        model: YOLO模型
        output_path: 输出路径（带框图片）
        dir: 工作目录
        ui_settings: [ai_confidence, sharpness_threshold, nima_threshold, save_crop, normalization_mode]
        i18n: I18n instance for internationalization (optional)
        skip_nima: 如果为True，跳过NIMA计算（用于双眼不可见的情况）
        focus_point: 对焦点坐标 (x, y)，归一化 0-1，用于多鸟时选择对焦的鸟
        decoded_image: 复用上游已解码的 BGR 图像，减少重复 JPEG 解码
    
    Returns:
        11-tuple (found_bird, bird_result, confidence, sharpness, nima_score, bird_bbox, img_dims, bird_mask, bird_count, rescued, all_birds)
        bird_count: 检测到的鸟的数量（V4.2 新增）
        rescued: 是否经补救扫描救回（V4.6 新增）/ whether rescued by the rescue scan (V4.6 new)
        all_birds: 全部鸟检测项（V5.0 multibird 新增），每项含
            idx/conf/bbox/area_ratio/mask_polygon（处理图坐标），空列表表示无鸟
    """
    # V3.1: 从 ui_settings 获取参数
    ai_confidence = ui_settings[0] / 100  # AI置信度：50-100 -> 0.5-1.0（仅用于过滤）
    sharpness_threshold = ui_settings[1]  # 锐度阈值：6000-9000
    nima_threshold = ui_settings[2]       # NIMA美学阈值：5.0-6.0
    save_crop = ui_settings[3]            # 是否保存裁切（V4.1: 恢复支持）

    # V3.2: 移除未使用的 normalization_mode 和 sharpness_calculator
    # 锐度现在由 photo_processor 中的 keypoint_detector 计算

    found_bird = False
    bird_sharp = False
    bird_result = False
    nima_score = None  # 美学评分
    # V3.2: 移除 BRISQUE（不再使用）

    # 使用配置检查文件类型
    if not config.is_jpg_file(image_path):
        log_message("ERROR: not a jpg file", dir)
        return None

    if not os.path.exists(image_path):
        log_message(f"ERROR: in detect_and_draw_birds, {image_path} not found", dir)
        return None

    # 记录总处理开始时间
    total_start = time.time()

    # Step 1: 图像预处理
    step_start = time.time()
    image = preprocess_image(image_path, source_image=decoded_image)
    if image is None:
        log_message(f"ERROR: cannot decode image {image_path}", dir)
        return None
    height, width, _ = image.shape
    preprocess_time = (time.time() - step_start) * 1000
    # V3.3: 简化日志，移除步骤详情
    # log_message(f"  ⏱️  [1/4] 图像预处理: {preprocess_time:.1f}ms", dir)

    # Step 2: YOLO推理
    step_start = time.time()
    # 使用最佳设备进行推理
    try:
        from config import get_best_device
        device = get_best_device()

        # 使用最佳设备进行推理
        results = model(image, device=device.type, verbose=False)
    except Exception as device_error:
        # 设备推理失败，清理 GPU 显存后降级到 CPU
        t = i18n.t if i18n else get_i18n().t
        log_message(t("ai.device_inference_failed", error=device_error), dir)
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            results = model(image, device='cpu', verbose=False)
        except Exception as cpu_error:
            log_message(t("ai.ai_inference_failed", error=cpu_error), dir)
            # 返回"无鸟"结果（V3.1）
            # V3.3: 使用英文列名
            data = {
                "filename": os.path.splitext(os.path.basename(image_path))[0],
                "has_bird": "no",
                "confidence": 0.0,
                "head_sharp": "-",
                "left_eye": "-",
                "right_eye": "-",
                "beak": "-",
                "nima_score": "-",
                "rating": -1
            }
            if report_db:
                report_db.insert_photo(data)
            return found_bird, bird_result, 0.0, 0.0, None, None, None, None, 0, False, []  # V5.0: 11 values with all_birds

    yolo_time = (time.time() - step_start) * 1000
    # V3.3: 简化日志，移除步骤详情
    # if i18n:
    #     log_message(i18n.t("logs.yolo_inference", time=yolo_time), dir)
    # else:
    #     log_message(f"  ⏱️  [2/4] YOLO推理: {yolo_time:.1f}ms", dir)

    # Step 3: 解析检测结果（立即提取为 numpy，释放 GPU tensor）
    step_start = time.time()
    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # 获取掩码数据（如果是分割模型）
    masks = None
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()

    # 数据已转为 numpy，立即释放 YOLO results（含 GPU tensor），避免长批次显存堆积
    del results

    # V5.0(multibird): 鸟类框 NMS 去重（同一只鸟可能留下多个框）
    detections, confidences, class_ids, masks = _dedupe_bird_boxes(
        detections, confidences, class_ids, masks)

    # V4.2: 收集所有检测到的鸟
    # V5.0(multibird): 每项附 area_ratio 与 mask_polygon（处理图坐标），
    # 供逐鸟分类/入库使用；masks 此时可用，轮廓压缩在此完成。
    # V4.2: Collect all detected birds; each entry also carries
    # area_ratio and the simplified mask polygon for per-bird use.
    all_birds = []
    for idx, (detection, conf, class_id) in enumerate(zip(detections, confidences, class_ids)):
        if int(class_id) == config.ai.BIRD_CLASS_ID:
            x1, y1, x2, y2 = detection
            box_w = max(0, int(x2) - int(x1))
            box_h = max(0, int(y2) - int(y1))
            all_birds.append({
                'idx': idx,
                'conf': float(conf),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'area_ratio': (box_w * box_h) / float(width * height)
                              if width > 0 and height > 0 else 0.0,
                'mask_polygon': _mask_to_polygon(masks, idx, width, height),
            })
    
    bird_count = len(all_birds)

    # V4.6: 无鸟补救扫描——第一遍低于 UI 阈值时触发 1024px 重扫 + 识鸟守门
    # V4.6: No-bird rescue scan — when pass-1 falls below the UI threshold,
    # rescan at 1024px with the BirdID classifier as gatekeeper.
    rescued = False
    _best_pass1 = max((b['conf'] for b in all_birds), default=0.0)
    if _best_pass1 < ai_confidence:
        _adv = get_advanced_config()
        if _adv.rescue_scan_enabled:
            _rescue = _rescue_scan(model, image, ai_confidence,
                                   _adv.rescue_birdid_gate, dir, i18n,
                                   image_path=image_path)
            if _rescue is not None:
                # V5.0(multibird): 用重扫的全部鸟框重建检测结果（不再只
                # 覆盖单只救回候选）——数组与 all_birds 索引天然对齐，下方
                # 主鸟选择策略照常运行（对焦点优先/最高置信度），逐鸟分类
                # 拿到真实 bird_count。混淆类候选已过识鸟守门，统一按鸟类。
                # V5.0: rebuild the full detection arrays from the rescan
                # so indices align with all_birds; the standard selection
                # strategy below runs unchanged and per-bird classification
                # sees the real bird_count.
                det_arr = _rescue.get("detections")
                if det_arr is not None and len(det_arr):
                    detections = np.asarray(det_arr, dtype=np.float64)
                    confidences = np.asarray(_rescue["detection_confs"],
                                             dtype=np.float64)
                    class_ids = np.full(len(confidences),
                                        float(config.ai.BIRD_CLASS_ID))
                    masks = _rescue.get("detection_masks")
                    # V5.0(multibird): 重扫数组同样去重后再重建 all_birds
                    detections, confidences, class_ids, masks = (
                        _dedupe_bird_boxes(detections, confidences,
                                           class_ids, masks))
                    all_birds = []
                    for pos in range(len(detections)):
                        dx1, dy1, dx2, dy2 = [int(v) for v in detections[pos]]
                        box_w = max(0, dx2 - dx1)
                        box_h = max(0, dy2 - dy1)
                        all_birds.append({
                            'idx': pos,
                            'conf': float(confidences[pos]),
                            'bbox': (dx1, dy1, dx2, dy2),
                            'area_ratio': (box_w * box_h)
                                          / float(width * height)
                                          if width > 0 and height > 0 else 0.0,
                            'mask_polygon': _mask_to_polygon(
                                masks, pos, width, height),
                        })
                    bird_count = len(all_birds)
                else:
                    # 兜底：重扫未带回数组时维持旧版单候选行为
                    # Fallback: legacy single-candidate overwrite.
                    detections = np.array([_rescue["xyxy"]], dtype=np.float64)
                    confidences = np.array([_rescue["conf"]], dtype=np.float64)
                    class_ids = np.array([float(config.ai.BIRD_CLASS_ID)])
                    masks = (_rescue["mask"][None, ...]
                             if _rescue["mask"] is not None else None)
                    rx1, ry1, rx2, ry2 = [int(v) for v in _rescue["xyxy"]]
                    all_birds = [{
                        'idx': 0,
                        'conf': _rescue["conf"],
                        'bbox': (rx1, ry1, rx2, ry2),
                        'area_ratio': ((rx2 - rx1) * (ry2 - ry1))
                                      / float(width * height)
                                      if width > 0 and height > 0 else 0.0,
                        'mask_polygon': _mask_to_polygon(masks, 0, width, height),
                    }]
                    bird_count = 1
                rescued = True

    # V5.8: 原图瓦片检测——每张照片必跑（不止补救场景）。补找 1024 整图
    # 分辨率下不可见的隐蔽小鸟（2026-09-27 乐活中堤实测：栗耳鹀/褐柳莺/
    # 红喉歌鸲深度伪装目标整组漏检）；三重门槛（已有框去重/置信地板/
    # BirdID 守门）保证枯叶杂物不混入。新框同步追加进 detections 数组，
    # idx 与 all_birds 严格对齐，主鸟选择与逐鸟分类照常运行。
    # V5.8: universal tiled detection pass on EVERY photo (not just
    # rescues) — finds tiny camouflaged birds invisible at 1024; three
    # gates keep leaves/junk out. New boxes are appended to the
    # detection arrays with idx aligned to all_birds.
    if config.ai.RESCUE_TILE_ENABLED:
        _had_bird_before_tile = bird_count > 0
        _adv_tile = get_advanced_config()
        _tile_new = _tile_detect_pass(
            model, image_path, image, detections,
            _adv_tile.rescue_birdid_gate, dir, i18n)
        if _tile_new:
            _n_new = len(_tile_new)
            _new_det = np.array([b["xyxy"] for b in _tile_new],
                                dtype=np.float64)
            _new_conf = np.array([b["conf"] for b in _tile_new],
                                 dtype=np.float64)
            _base_idx = len(detections)
            detections = (np.vstack([detections, _new_det])
                          if len(detections) else _new_det)
            confidences = (np.concatenate([confidences, _new_conf])
                           if len(confidences) else _new_conf)
            class_ids = np.concatenate(
                [class_ids, np.full(_n_new, float(config.ai.BIRD_CLASS_ID))])
            # masks 不补行（瓦片掩码不跨瓦合并）；_mask_to_polygon 对越界
            # idx 返回 None，多边形缺失走对焦 bbox 回退，链路已兼容
            for _k, _b in enumerate(_tile_new):
                _x1, _y1, _x2, _y2 = [int(v) for v in _b["xyxy"]]
                _bw = max(0, _x2 - _x1)
                _bh = max(0, _y2 - _y1)
                all_birds.append({
                    'idx': _base_idx + _k,
                    'conf': _b["conf"],
                    'bbox': (_x1, _y1, _x2, _y2),
                    'area_ratio': (_bw * _bh) / float(width * height)
                                  if width > 0 and height > 0 else 0.0,
                    'mask_polygon': None,
                })
            bird_count = len(all_birds)
            # 救回语义：原本判无鸟、靠瓦片检出的照片视同补救
            # Mark as rescued when the pass turned a no-bird photo around.
            if not _had_bird_before_tile:
                rescued = True

    # V4.2/V5.1: 鸟选择策略（单鸟 → 对焦点命中[多边形优先,bbox回退] →
    # 置信度兜底）。fallback 时下游逐鸟分类后会按稀有/画质综合重选。
    bird_idx, _selection_reason = _select_main_bird(
        all_birds, focus_point, width, height)

    # V5.0(multibird): 给选中项打标 + 记录选择原因（V5.1），下游
    # 逐鸟分类据此识别主鸟、fallback 时触发综合重选
    for bird in all_birds:
        bird['is_selected'] = (bird['idx'] == bird_idx)
        if bird['idx'] == bird_idx:
            bird['selection_reason'] = _selection_reason

    parse_time = (time.time() - step_start) * 1000
    # V3.3: 简化日志，移除步骤详情
    # if i18n:
    #     log_message(i18n.t("logs.result_parsing", time=parse_time), dir)
    # else:
    #     log_message(f"  ⏱️  [3/4] 结果解析: {parse_time:.1f}ms", dir)

    # 如果没有找到鸟，记录到CSV并返回（V3.1）
    if bird_idx == -1:
        # 诊断日志：记录 YOLO 实际返回的最高置信度，便于排查跨设备差异
        if len(confidences) == 0:
            log_message(f"DEBUG YOLO no_bird: {os.path.basename(image_path)} → 0 detections (all below YOLO conf_thresh=0.25)", dir)
        else:
            best_conf = float(confidences.max())
            best_cls = int(class_ids[confidences.argmax()])
            log_message(f"DEBUG YOLO no_bird: {os.path.basename(image_path)} → {len(confidences)} detections, best_conf={best_conf:.3f} cls={best_cls} (bird_class={config.ai.BIRD_CLASS_ID})", dir)
        # V3.3: 使用英文列名
        data = {
            "filename": os.path.splitext(os.path.basename(image_path))[0],
            "has_bird": "no",
            "confidence": 0.0,
            "head_sharp": "-",
            "left_eye": "-",
            "right_eye": "-",
            "beak": "-",
            "nima_score": "-",
            "rating": -1
        }
        if report_db:
            report_db.insert_photo(data)
        return found_bird, bird_result, 0.0, 0.0, None, None, None, None, 0, False, []  # V5.0: 11 values with all_birds
    # V3.2: 移除 NIMA 计算（现在由 photo_processor 在裁剪区域上计算）
    # nima_score 设为 None，photo_processor 会重新计算
    nima_score = None
    
    # V3.9.3: 提前声明默认值，避免 continue 后变量未定义
    sharpness = 0.0
    x, y, w, h = 0, 0, 0, 0

    # 只处理面积最大的那只鸟
    for idx, (detection, conf, class_id) in enumerate(zip(detections, confidences, class_ids)):
        # 跳过非鸟类或非最大面积的鸟
        if idx != bird_idx:
            continue
        x1, y1, x2, y2 = detection

        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)
        class_id = int(class_id)

        # 使用配置中的鸟类类别 ID
        if class_id == config.ai.BIRD_CLASS_ID:
            found_bird = True
            area_ratio = (w * h) / (width * height)
            filename = os.path.basename(image_path)

            # V3.1: 不再保存Crop图片
            crop_path = None

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                log_message(f"ERROR: Invalid crop region for {image_path}", dir)
                continue

            crop_img = image[y:y + h, x:x + w]

            if crop_img is None or crop_img.size == 0:
                log_message(f"ERROR: Crop image is empty for {image_path}", dir)
                continue

            # V3.2: 移除 Step 5 锐度计算（现在由 photo_processor 中的 keypoint_detector 计算 head_sharpness）
            # 设置占位值以保持 CSV 兼容性
            real_sharpness = 0.0
            sharpness = 0.0
            effective_pixels = 0

            # V3.2: 移除 BRISQUE 评估（不再使用）

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # V3.1: 新的评分逻辑
            # 计算中心坐标（仅用于日志输出）
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height

            # V3.3: 简化日志，移除AI详情输出
            # log_message(f" AI: {conf:.2f} - Class: {class_id} "
            #             f"- Area:{area_ratio * 100:.2f}% - Pixels:{effective_pixels:,d}"
            #             f" - Center_x:{center_x:.2f} - Center_y:{center_y:.2f}", dir)

            # V3.2: 移除评分逻辑（现在由 photo_processor 的 RatingEngine 计算）
            # rating_value 设为占位值，photo_processor 会重新计算
            rating_value = 0

            # V3.3: 使用英文列名
            # V4.1: 添加路径信息
            try:
                rel_current_path = os.path.relpath(image_path, dir)
            except ValueError:
                rel_current_path = image_path # Fallback to absolute if different drive
                
            rel_debug_path = None
            
            # V4.1: 如果启用了保存裁切 (save_crop) 且没有指定 output_path，自动保存到 cache/debug
            if save_crop and not output_path:
                from tools.file_utils import ensure_hidden_directory
                
                superpicky_dir = os.path.join(dir, ".superpicky")
                cache_dir = os.path.join(superpicky_dir, "cache")
                # V4.2: Rename to yolo_debug for clarity
                debug_dir = os.path.join(cache_dir, "yolo_debug")
                
                try:
                    ensure_hidden_directory(superpicky_dir)
                    ensure_hidden_directory(debug_dir)
                    
                    filename = os.path.basename(image_path)
                    prefix, ext = os.path.splitext(filename)
                    output_path = os.path.join(debug_dir, f"{prefix}.jpg")
                except Exception:
                    pass


            
            data = {
                "filename": os.path.splitext(os.path.basename(image_path))[0],
                "has_bird": "yes" if found_bird else "no",
                "confidence": float(f"{conf:.2f}"),
                "head_sharp": "-",        # 将由 photo_processor 填充
                "left_eye": "-",          # 将由 photo_processor 填充
                "right_eye": "-",         # 将由 photo_processor 填充
                "beak": "-",              # 将由 photo_processor 填充
                "nima_score": float(f"{nima_score:.2f}") if nima_score is not None else "-",
                "rating": rating_value,
                # V4.1 Paths
                "current_path": rel_current_path,
                "debug_crop_path": None, # Will be filled by photo_processor
                "yolo_debug_path": None  # Will fill below
            }
            
            # Update yolo_debug_path if we generated it
            if found_bird and save_crop and output_path:
                try:
                    data["yolo_debug_path"] = os.path.relpath(output_path, dir)
                except ValueError:
                    data["yolo_debug_path"] = output_path

            # Step 5: CSV写入
            step_start = time.time()
            if report_db:
                report_db.insert_photo(data)
            csv_time = (time.time() - step_start) * 1000
            # V3.3: 简化日志
            # log_message(f"  ⏱️  [4/4] CSV写入: {csv_time:.1f}ms", dir)


    # V5.0(multibird): 调试图画出全部鸟——主鸟红粗框（上方循环已画），
    # 其余灰细框+序号，多鸟场景的完整检测清单不再丢失。
    # V5.0: draw every bird on the debug image — the main bird keeps its
    # red box; others get a thin gray box plus an index label.
    if found_bird and output_path and len(all_birds) > 1:
        for bird in all_birds:
            if bird['idx'] == bird_idx:
                continue
            bx1, by1, bx2, by2 = bird['bbox']
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (160, 160, 160), 1)
            cv2.putText(image, str(bird['idx']), (bx1, max(12, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # 只有在 found_bird 为 True 且 output_path 有效时，才保存带框的图片
    if found_bird and output_path:
        cv2.imwrite(output_path, image)
    # --- 修改结束 ---

    # 计算总处理时间 (V3.3: 移除此处日志, 由 photo_processor 输出真正总耗时)
    total_time = (time.time() - total_start) * 1000
    # log_message(f"  ⏱️  ========== 总耗时: {total_time:.1f}ms ==========", dir)

    # 返回 found_bird, bird_result, AI置信度, 归一化锐度, NIMA分数, bbox, 图像尺寸, 分割掩码
    bird_confidence = float(confidences[bird_idx]) if bird_idx != -1 else 0.0
    bird_sharpness = sharpness if bird_idx != -1 else 0.0
    # bbox 格式: (x, y, w, h) - 在缩放后的图像上
    # img_dims 格式: (width, height) - 缩放后图像的尺寸，用于计算缩放比例
    bird_bbox = (x, y, w, h) if found_bird else None
    img_dims = (width, height) if found_bird else None
    
    # 获取对应鸟的掩码
    bird_mask = None
    if found_bird and masks is not None:
        # masks shape: (N, H, W) where N is number of detections
        # YOLO masks are usually same size as input image (or smaller and upscaled)
        # Ultralytics results.masks.data is usually (N, H, W) 
        # But we need to be careful about resizing if it's smaller
        # results.masks.data contains masks for all detections
        # We need the one corresponding to bird_idx
        try:
            # Mask is already resized to image size by ultralytics by default in modern versions
            # But let's verify if we need to resize
            raw_mask = masks[bird_idx]
            
            # Ensure mask is same size as processed image (width, height)
            if raw_mask.shape != (height, width):
                raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary uint8 mask (0 or 255)
            # YOLO masks are float [0,1], threshold at 0.5
            bird_mask = (raw_mask > 0.5).astype(np.uint8) * 255
        except Exception as e:
            # Mask processing failed, ignore
            pass

    return found_bird, bird_result, bird_confidence, bird_sharpness, nima_score, bird_bbox, img_dims, bird_mask, bird_count, rescued, all_birds
