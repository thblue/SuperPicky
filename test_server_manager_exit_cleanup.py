# -*- coding: utf-8 -*-
"""server_manager 确定性退出清理回归测试 / Deterministic exit-cleanup tests.

背景 / Background:
本进程启动的 BirdID 服务子进程此前完全依赖调用方记得调 stop_server()，
异常/信号等退出路径会遗留孤儿进程。V4.x 起改为「谁启动、谁兜底」：
启动路径登记所有权，atexit + 信号钩子在进程退出时兜底停止；
CLI `server_manager.py start`（Lightroom 插件守护用法）显式豁免。

这组测试锁定三条语义：
1. auto_cleanup=True 的进程退出后，其启动的服务子进程被兜底停止；
2. auto_cleanup=False（CLI 守护）的进程退出后，服务独立存活；
3. 复用他人已启动的健康服务时不取得所有权，退出不得误杀。

The BirdID server subprocess used to rely solely on the caller invoking
stop_server(); abnormal exits leaked orphans. Now "own-then-cleanup":
start paths register ownership and an atexit + signal backstop stops the
server on process exit, while the standalone CLI `start` opts out. These
tests pin: (1) owned servers die with their owner; (2) opt-out daemons
survive the starter; (3) reusing a healthy foreign server never takes
ownership.
"""
import os
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_PORT = 5199  # 独立测试端口，避开默认 5156 与 BirdIndex 8300 / dedicated test port

# 子进程脚本模板：启动服务后自然退出，退出路径由 atexit 兜底
# Child script template: start the server, let the process exit naturally.
_HELPER_START = (
    "from server_manager import start_server_daemon; "
    "start_server_daemon(port={port}, auto_cleanup={auto})"
)


def _spawn_helper(auto_cleanup: bool) -> subprocess.Popen:
    """启动一个「start 后即退出」的子进程 / Spawn a start-then-exit child."""
    code = _HELPER_START.format(port=TEST_PORT, auto=auto_cleanup)
    return subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_port(state: bool, timeout: float = 45.0) -> bool:
    """轮询测试端口直到达到期望状态 / Poll test port until it reaches state."""
    from server_manager import is_port_in_use

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_in_use(TEST_PORT) == state:
            return True
        time.sleep(0.5)
    return is_port_in_use(TEST_PORT) == state


def _pid_alive(pid) -> bool:
    """检查 PID 进程是否存活 / Whether the pid is still running."""
    from server_manager import is_process_running

    return is_process_running(pid)


@pytest.fixture(autouse=True)
def _cleanup_server():
    """每个用例结束后确定性清场：停服务、清 PID 文件、杀残留端口占用。"""
    yield
    from server_manager import stop_server

    stop_server()
    # 兜底：仍有测试端口占用则按端口找不到 PID 时只能报错，由断言暴露
    if _wait_port(False, timeout=10.0) is False:
        pytest.fail(f"测试端口 {TEST_PORT} 在清场后仍被占用 / port still in use")


def test_owned_server_stops_on_owner_exit():
    """auto_cleanup=True：启动进程退出 → 服务子进程被 atexit 兜底停止。"""
    child = _spawn_helper(auto_cleanup=True)
    assert child.wait(timeout=60) == 0
    # 子进程已退出；其启动的服务要么已被兜底停止，要么极短时间内停止
    assert _wait_port(False, timeout=15), (
        "所有者进程退出后测试端口仍被占用 / owned server leaked after owner exit"
    )


def test_optout_daemon_survives_starter_exit():
    """auto_cleanup=False（CLI 守护）：启动器退出后服务独立存活。"""
    child = _spawn_helper(auto_cleanup=False)
    assert child.wait(timeout=60) == 0
    assert _wait_port(True, timeout=45), (
        "豁免清理的服务未能在启动器退出后保持存活 / opt-out daemon died"
    )
    from server_manager import get_server_status

    status = get_server_status(TEST_PORT)
    assert status["running"], "豁免守护应仍在运行 / opt-out daemon should be running"


def test_reuse_healthy_server_takes_no_ownership():
    """复用他人启动的健康服务：复用方退出不得停掉服务。"""
    # 第一步：豁免方式拉起守护（模拟 LR 插件 CLI 启动）
    starter = _spawn_helper(auto_cleanup=False)
    assert starter.wait(timeout=60) == 0
    assert _wait_port(True, timeout=45), "守护未能就绪 / daemon not up"

    # 第二步：另一个进程以默认 auto_cleanup=True 复用它后退出
    reuser = _spawn_helper(auto_cleanup=True)
    assert reuser.wait(timeout=60) == 0

    # 第三步：服务必须仍然存活（复用不取得所有权）
    from server_manager import check_server_health

    assert check_server_health(TEST_PORT), (
        "复用方退出后健康服务被误停 / healthy foreign server killed by reuser exit"
    )
