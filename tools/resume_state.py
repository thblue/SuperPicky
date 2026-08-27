#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import tempfile
import time
from typing import Dict, List, Optional

from .file_utils import ensure_hidden_directory


class ResumeStateManager:
    """Persist lightweight resume state outside the report database."""

    FILENAME = "resume_state.json"

    # SMB/NAS 上状态文件可能被索引服务或杀软短暂锁住（WinError 5/32）：重试后仍失败
    # 则放弃本次断点写入并静默降级，绝不向上抛异常——否则会把正在处理的照片连带记成
    # 「处理异常跳过」，污染统计（历史 8/24 掉 42 张问题的根因）。
    # On SMB/NAS shares the state file can be briefly locked by the indexer or an
    # antivirus scanner (WinError 5/32): retry, then degrade quietly instead of
    # propagating into the photo pipeline (root cause of past skipped-photo losses).
    WRITE_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 0.15

    def __init__(self, directory: str):
        self.directory = directory
        self.state_dir = os.path.join(directory, ".superpicky")
        self.state_path = os.path.join(self.state_dir, self.FILENAME)
        # 瞬时写入失败只告警一次，避免长跑刷屏。
        # Warn only once for transient write failures to avoid log spam.
        self._write_failure_warned = False

    def exists(self) -> bool:
        return os.path.exists(self.state_path)

    def load(self) -> Optional[Dict]:
        if not self.exists():
            return None
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def start(self, ordered_prefixes: List[str]) -> None:
        payload = {
            "version": 1,
            "status": "running",
            "total_files": len(ordered_prefixes),
            "next_index": 1,
            "pending_prefixes": list(ordered_prefixes),
        }
        self._write(payload)

    def get_resume_plan(self, available_prefixes: List[str]) -> Optional[Dict]:
        state = self.load()
        if not state or state.get("status") != "running":
            return None
        available_set = set(available_prefixes)
        pending = [prefix for prefix in (state.get("pending_prefixes") or []) if prefix in available_set]
        if not pending:
            return None
        total_files = int(state.get("total_files") or len(available_prefixes))
        completed = max(0, total_files - len(pending))
        return {
            "total_files": total_files,
            "next_index": completed + 1,
            "pending_prefixes": pending,
        }

    def mark_completed(self, prefix: str) -> None:
        state = self.load()
        if not state:
            return
        pending = [item for item in (state.get("pending_prefixes") or []) if item != prefix]
        state["pending_prefixes"] = pending
        total_files = int(state.get("total_files") or 0)
        state["next_index"] = min(total_files + 1, total_files - len(pending) + 1) if total_files > 0 else 1
        if not pending:
            self.clear()
            return
        self._write(state)

    def clear(self) -> None:
        """删除断点状态文件；幂等清理，删除失败仅告警不抛出。

        Remove the resume state file idempotently; deletion failures only warn.
        """
        try:
            if self.exists():
                os.remove(self.state_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._warn_write_failure(exc)

    def _write(self, payload: Dict) -> None:
        """原子写入断点状态；NAS 瞬时锁失败退避重试后放弃，不抛出。

        Atomically write the resume state. Transient SMB sharing violations are
        retried with backoff, then swallowed — bookkeeping must never fail a photo.

        参数 / Parameters:
            payload (Dict): 待写入的状态字典 / state dict to persist
        """
        ensure_hidden_directory(self.state_dir)
        last_error: Optional[Exception] = None
        for attempt in range(self.WRITE_RETRIES):
            fd, temp_path = tempfile.mkstemp(prefix="resume_", suffix=".json", dir=self.state_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, self.state_path)
                return
            except OSError as exc:
                # WinError 5/32 等占用类错误按瞬时态处理，线性退避后重试。
                # Treat sharing-violation style errors as transient and back off.
                last_error = exc
                time.sleep(self.RETRY_BACKOFF_SECONDS * (attempt + 1))
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
        self._warn_write_failure(last_error)

    def _warn_write_failure(self, exc: Optional[Exception]) -> None:
        """断点状态操作最终失败时打印一次性告警（CLI 日志可见），不再抛出。

        Print a one-time warning when resume-state bookkeeping finally fails,
        without raising into the caller.
        """
        if not self._write_failure_warned:
            print(f"⚠️ 断点状态读写失败（不影响照片处理结果，仅影响中断后续跑）: {exc}")
            self._write_failure_warned = True
