# -*- coding: utf-8 -*-
"""
将旧版本 SuperPicky 处理过的照片目录复位为初始平铺状态。

功能：
1. 把所有子目录（鸟种/星级/无鸟等）中的照片与视频文件移动回目录根部；
2. 删除旧处理产物：.xmp 旁车文件（含旧评分，避免污染重跑）、.superpicky 缓存、
   .superpicky_manifest.json / .superpicky_video_manifest.json、superpicky.log；
3. 删除清空后的子目录。

用法：python reset_photo_dir.py <目录1> [目录2 ...]
"""
import shutil
import sys
from pathlib import Path

# 照片/视频扩展名（复位时需要保留并移回根目录）
MEDIA_EXTS = {".cr3", ".cr2", ".jpg", ".jpeg", ".mp4", ".mov", ".avi", ".xmp"}
# 需要整体删除的旧处理产物（目录或文件）
PURGE_NAMES = {".superpicky", ".superpicky_manifest.json",
               ".superpicky_video_manifest.json", "superpicky.log"}


def reset_dir(root: Path) -> None:
    """复位单个照片目录：媒体文件平铺到根部，删除旧处理产物与空目录。

    参数:
    root (Path): 要复位的目录

    异常:
    OSError: 文件移动或删除失败时抛出
    """
    moved, deleted_xmp = 0, 0
    for name in PURGE_NAMES:
        target = root / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    for path in root.rglob("*"):
        if not path.is_file() or ".superpicky" in path.parts:
            continue
        ext = path.suffix.lower()
        dest = root / path.name
        if path.parent == root:
            if ext == ".xmp":
                path.unlink()  # 根部旧评分旁车，直接删除
                deleted_xmp += 1
            continue
        if ext == ".xmp":
            path.unlink()  # 旧评分旁车，删除避免污染重跑
            deleted_xmp += 1
        elif ext in MEDIA_EXTS:
            if dest.exists():
                print(f"[冲突] 目标已存在，跳过: {dest}")
                continue
            shutil.move(str(path), str(dest))
            moved += 1
        else:
            print(f"[跳过] 未知类型文件: {path}")

    # 清理空目录（自底向上）
    for dirpath in sorted((d for d in root.rglob("*") if d.is_dir()),
                          key=lambda p: len(p.parts), reverse=True):
        try:
            dirpath.rmdir()
        except OSError:
            pass
    print(f"[{root}] 移回 {moved} 个文件，删除 {deleted_xmp} 个旧 xmp")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        reset_dir(Path(arg))
