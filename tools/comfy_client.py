# -*- coding: utf-8 -*-
"""
ComfyUI HTTP 客户端 / ComfyUI HTTP client.

供 spb_pixel_art 等后处理脚本与本地 ComfyUI 服务交互：上传输入图、
提交 API 格式工作流、轮询执行结果、下载生成图。端点与官方示例脚本
（script_examples/websockets_api_example.py）一致：

    POST /upload/image          multipart 上传，返回 {"name","subfolder","type"}
    POST /prompt                {"prompt": 工作流, "client_id": uuid} → prompt_id
    GET  /history/{prompt_id}   执行完成前返回 {}；完成后含 outputs/status
    GET  /view?filename=...     下载生成文件（二进制）
    GET  /system_stats          探活

本模块只做 HTTP 客户端，不负责启动/停止 ComfyUI 进程（调用方自启）。
使用 httpx（项目既有依赖），同步阻塞式，适合 CLI 逐张串行场景。

Thin HTTP client for a locally running ComfyUI server: upload input
images, queue API-format workflows, poll history, download outputs.
Never launches or manages the ComfyUI process itself.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import httpx


class ComfyUIError(Exception):
    """
    ComfyUI 交互失败（服务不可达 / 上传失败 / 排队被拒 / 执行出错 / 超时）。

    ComfyUI interaction failure (unreachable / upload failed / rejected
    prompt / execution error / timeout).
    """


class ComfyClient:
    """
    单 ComfyUI 服务的同步客户端，建议以 with 语句使用以确定性关闭连接。

    同步 HTTP client for one ComfyUI server; prefer a `with` block so the
    underlying connection pool is closed deterministically.

    参数 / Parameters:
        host (str): "host:port" 形式，默认本机 8188 / "host:port", default local 8188
        request_timeout (float): 单次 HTTP 请求超时秒数 / per-request timeout seconds
    """

    def __init__(self, host: str = "127.0.0.1:8188", request_timeout: float = 30.0) -> None:
        self.base_url: str = f"http://{host}"
        self.client_id: str = str(uuid.uuid4())
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout, connect=5.0),
        )

    # ── 生命周期 / lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        """关闭底层连接池 / Close the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> "ComfyClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── 各端点 / endpoints ──────────────────────────────────────────────

    def is_alive(self) -> bool:
        """
        探活：GET /system_stats 是否可达。

        返回:
        bool: 服务可达返回 True；任何异常（拒绝连接/超时/非 200）返回 False。

        Liveness probe via GET /system_stats; True iff reachable.
        """
        try:
            resp = self._http.get("/system_stats")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def upload_image(self, name: str, data: bytes,
                     content_type: str = "image/jpeg") -> Dict[str, str]:
        """
        上传输入图到 ComfyUI input 目录（overwrite=true，幂等）。

        参数:
        name (str): 目标文件名（建议含唯一前缀，防多任务串图）。
        data (bytes): 图片二进制内容。
        content_type (str): multipart MIME 类型，默认 image/jpeg。

        返回:
        Dict[str, str]: {"name": 服务端文件名, "subfolder": 子目录, "type": "input"}，
                        其中 name 需回填进工作流的 LoadImage 节点。

        异常:
        ComfyUIError: 上传失败或响应异常。

        Upload an image into ComfyUI's input folder (overwrite=true).
        The returned "name" must be injected into the LoadImage node.
        """
        try:
            resp = self._http.post(
                "/upload/image",
                files={"image": (name, data, content_type)},
                data={"overwrite": "true", "type": "input"},
            )
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"上传失败 / upload failed: {exc}") from exc
        if resp.status_code != 200:
            raise ComfyUIError(
                f"上传失败 HTTP {resp.status_code}: {resp.text[:300]}")
        info = resp.json()
        if not isinstance(info, dict) or "name" not in info:
            raise ComfyUIError(f"上传响应异常 / bad upload response: {info!r}")
        return {"name": str(info["name"]),
                "subfolder": str(info.get("subfolder", "")),
                "type": str(info.get("type", "input"))}

    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        提交 API 格式工作流（"Save (API Format)" 导出的那种 dict-of-nodes）。

        参数:
        workflow (Dict[str, Any]): {node_id: {"class_type": ..., "inputs": {...}}}。

        返回:
        str: prompt_id，用于后续 wait_result 轮询。

        异常:
        ComfyUIError: 提交失败，或服务端校验出 node_errors（含节点级错误详情）。

        Queue an API-format workflow; returns prompt_id. Raises on
        transport failure or server-side node_errors.
        """
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            resp = self._http.post("/prompt", json=payload)
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"提交失败 / queue failed: {exc}") from exc
        if resp.status_code != 200:
            raise ComfyUIError(
                f"提交失败 HTTP {resp.status_code}: {resp.text[:500]}")
        info = resp.json()
        node_errors = info.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError(
                f"工作流校验失败 / node_errors: {str(node_errors)[:800]}")
        prompt_id = info.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"提交响应缺少 prompt_id: {info!r}")
        return str(prompt_id)

    def wait_result(self, prompt_id: str, timeout: float = 300.0,
                    poll_interval: float = 1.0) -> Dict[str, Any]:
        """
        轮询 GET /history/{prompt_id} 直到执行结束（ComfyUI 执行期间该端点
        返回 {}，结束后才出现条目，因此「出现条目」即「已结束」）。

        参数:
        prompt_id (str): queue_prompt 返回的 ID。
        timeout (float): 最长等待秒数，默认 300。
        poll_interval (float): 轮询间隔秒数，默认 1。

        返回:
        Dict[str, Any]: history 条目，含 "outputs"（按节点分组的结果）与
                        "status"（status_str/completed/messages）。

        异常:
        ComfyUIError: 超时未完成、执行报错（status_str == "error"）或轮询请求失败。

        Poll history until the prompt finishes ({} means still running).
        Returns the history entry; raises on timeout or execution error.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                resp = self._http.get(f"/history/{prompt_id}")
            except httpx.HTTPError as exc:
                raise ComfyUIError(f"轮询失败 / history poll failed: {exc}") from exc
            if resp.status_code != 200:
                raise ComfyUIError(
                    f"轮询失败 HTTP {resp.status_code}: {resp.text[:300]}")
            history = resp.json() or {}
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    messages = status.get("messages") or []
                    raise ComfyUIError(
                        f"ComfyUI 执行出错 / execution error: "
                        f"{str(messages)[:800]}")
                outputs = entry.get("outputs") or {}
                if not outputs:
                    raise ComfyUIError(
                        "ComfyUI 执行结束但无任何输出节点结果 / "
                        "finished with empty outputs")
                return entry
            if time.monotonic() >= deadline:
                raise ComfyUIError(
                    f"等待超时（{timeout:.0f}s）/ timeout waiting for "
                    f"prompt {prompt_id}")
            time.sleep(poll_interval)

    def fetch_output(self, filename: str, subfolder: str = "",
                     file_type: str = "output") -> bytes:
        """
        下载生成文件：GET /view?filename=&subfolder=&type=。

        参数:
        filename (str): history outputs 里给出的文件名（含扩展名）。
        subfolder (str): 子目录，通常为空。
            file_type (str): "output" / "input" / "temp"，默认 "output"。

        返回:
        bytes: 文件二进制内容，调用方直接落盘。

        异常:
        ComfyUIError: 下载失败或非 200。

        Download a generated file via GET /view; returns raw bytes.
        """
        try:
            resp = self._http.get(
                "/view",
                params={"filename": filename, "subfolder": subfolder,
                        "type": file_type},
            )
        except httpx.HTTPError as exc:
            raise ComfyUIError(f"下载失败 / view failed: {exc}") from exc
        if resp.status_code != 200:
            raise ComfyUIError(
                f"下载失败 HTTP {resp.status_code}: {filename}")
        return resp.content


def collect_output_images(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从 history 条目里收集所有生成图片的下载描述。

    遍历 entry["outputs"] 各节点的 "images" 列表（跳过动画 gif 等非图片
    输出），返回可直接交给 ComfyClient.fetch_output 的参数字典序列，
    保持节点顺序稳定，便于确定「主输出 = 第一张」。

    参数:
    entry (Dict[str, Any]): wait_result 返回的 history 条目。

    返回:
    List[Dict[str, str]]: [{"filename","subfolder","type"}, ...]，可能为空。

    Collect all generated images from a history entry into a list of
    {"filename","subfolder","type"} dicts (node order preserved).
    """
    images: List[Dict[str, str]] = []
    outputs = entry.get("outputs") or {}
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for img in node_out.get("images") or []:
            if not isinstance(img, dict) or "filename" not in img:
                continue
            images.append({
                "filename": str(img["filename"]),
                "subfolder": str(img.get("subfolder", "")),
                "type": str(img.get("type", "output")),
            })
    return images
