#!/usr/bin/env python3
"""
SuperPicky BirdID 服务器管理器
管理 API 服务器的生命周期：启动、停止、状态检查
支持守护进程模式，使服务器可以独立于 GUI 运行

V4.0.0 修复：打包模式下使用线程方式启动，避免重复启动整个应用

确定性退出清理 / Deterministic exit cleanup:
    本进程启动的服务器（子进程或线程模式）自动登记「所有权」，并通过
    atexit + 信号钩子（SIGINT/SIGTERM/SIGBREAK）在进程退出时兜底停止，
    不再依赖调用方记得调 stop_server()。豁免场景：`server_manager.py start`
    命令行启动（Lightroom 插件的守护用法，启动器进程随即退出，服务必须
    独立存活）传入 auto_cleanup=False 显式退出登记。硬杀（taskkill /F、
    断电、os._exit）不经过 Python，atexit 无法覆盖——由下次启动时已有的
    「僵尸端口占用 → stop_server」恢复路径兜底。
"""

import atexit
import os
import sys
import signal
import socket
import subprocess
import time
import json
import threading

# V4.2.1: I18n support
from tools.i18n import get_i18n
from config import config, get_app_config_dir, get_lazy_registry

def get_t():
    """Get translator function"""
    try:
        # Try to get language from config file if possible, or default
        # For server manager, we might just default to system locale or english if config not loaded
        # But get_i18n handles defaults.
        return get_i18n().t
    except Exception:
        # Fallback if core module not found (e.g. running check script standalone without path)
        return lambda k, **kw: k

# PID 文件位置
def get_pid_file_path():
    """获取 PID 文件路径"""
    pid_dir = get_app_config_dir()
    pid_dir.mkdir(parents=True, exist_ok=True)
    return str(pid_dir / 'birdid_server.pid')


def get_server_script_path():
    """获取服务器脚本路径"""
    # 支持开发模式和打包模式
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'birdid_server.py')


def is_port_in_use(port, host='127.0.0.1'):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def check_server_health(port=None, host=None, timeout=None):
    """检查服务器健康状态"""
    port = config.server.PORT if port is None else port
    host = config.server.HOST if host is None else host
    timeout = config.server.HEALTH_TIMEOUT_SECONDS if timeout is None else timeout
    try:
        import urllib.request
        import ssl
        
        url = f'http://{host}:{port}/health'
        req = urllib.request.Request(url, method='GET')
        
        # macOS SSL证书问题修复
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('status') == 'ok'
    except Exception:
        pass
    return False


def read_pid():
    """读取 PID 文件"""
    pid_file = get_pid_file_path()
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            pass
    return None


def write_pid(pid):
    """写入 PID 文件"""
    pid_file = get_pid_file_path()
    with open(pid_file, 'w') as f:
        f.write(str(pid))


def remove_pid():
    """删除 PID 文件"""
    pid_file = get_pid_file_path()
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
        except OSError:
            pass


def is_process_running(pid):
    """
    检查进程是否存在（跨平台可靠实现）。

    参数:
    pid (int): 进程 PID

    返回:
    bool: 进程是否存在。

    说明: 旧实现用 os.kill(pid, 0)，在 Windows 上对无控制台进程
    （CREATE_NO_WINDOW 启动的服务器）抛 WinError 87 被误判为已退出，
    导致 stop_server 永远走「未运行」分支、服务泄漏。psutil 的
    pid_exists 基于 OpenProcess 查询，无此问题；无 psutil 时退回信号 0。

    Check whether a process exists. The old os.kill(pid, 0) probe raised
    WinError 87 for console-less Windows processes (our CREATE_NO_WINDOW
    server), reporting live servers as dead so stop_server() leaked them.
    psutil.pid_exists queries via OpenProcess and is reliable; fall back
    to the signal-0 probe when psutil is unavailable.
    """
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:
        try:
            os.kill(pid, 0)  # 发送信号 0 检查进程是否存在
            return True
        except (OSError, ProcessLookupError):
            return False


def find_listener_pid(port, host='127.0.0.1'):
    """
    找到正在监听指定端口的进程 PID。

    参数:
    port (int): 目标端口
    host (str): 监听地址（当前仅用于语义提示，匹配按端口执行）

    返回:
    int or None: 监听进程 PID，找不到返回 None。

    背景: venv 垫脚启动器（.venv/Scripts/python.exe）会把真实解释器
    作为子进程拉起，Popen 返回的 PID 只是垫脚进程；按端口解析才能
    拿到真正持有 socket 的服务进程，stop_server 才能停得掉。

    Resolve the pid actually LISTENING on the port. Venv launcher
    shims spawn the real interpreter as a child, so the Popen pid is
    just the shim; only port-based resolution yields the process that
    holds the socket.
    """
    try:
        import psutil
        for conn in psutil.net_connections(kind='inet'):
            if (conn.status == psutil.CONN_LISTEN
                    and conn.laddr is not None
                    and conn.laddr.port == port
                    and conn.pid):
                return conn.pid
    except Exception:
        pass
    return None


def get_server_status(port=None):
    port = config.server.PORT if port is None else port
    """
    获取服务器状态
    
    Returns:
        dict: {
            'running': bool,
            'pid': int or None,
            'healthy': bool,
            'port': int
        }
    """
    pid = read_pid()
    process_running = is_process_running(pid)
    port_in_use = is_port_in_use(port)
    healthy = check_server_health(port)
    
    return {
        'running': process_running or port_in_use,
        'pid': pid if process_running else None,
        'healthy': healthy,
        'port': port
    }


# 全局变量迁移到统一懒加载注册器


def _get_server_thread():
    return get_lazy_registry().get('server_manager.thread')


def _set_server_thread(value):
    get_lazy_registry().set('server_manager.thread', value)


def _get_server_instance():
    return get_lazy_registry().get('server_manager.instance')


def _set_server_instance(value):
    get_lazy_registry().set('server_manager.instance', value)


# ---------------------------------------------------------------------------
# 确定性退出清理 / Deterministic exit cleanup
# 「谁启动、谁兜底」：本进程启动的服务器登记所有权，进程退出时（正常退出、
# 异常退出、SIGINT/SIGTERM/SIGBREAK）由 atexit + 信号钩子兜底停止。
# CLI `server_manager.py start`（LR 插件守护用法）显式豁免。
# Own-then-cleanup: servers started by THIS process are registered and
# stopped on process exit (normal, exception, or signal) via atexit +
# signal hooks. The standalone CLI `start` opts out explicitly.
# ---------------------------------------------------------------------------

def _get_owned_server_pids():
    """获取本进程拥有的服务器子进程 PID 列表 / Owned server subprocess pids."""
    return get_lazy_registry().get('server_manager.owned_pids') or []


def _add_owned_server_pid(pid):
    """登记一个本进程启动的服务器子进程 / Register an owned server subprocess."""
    pids = list(_get_owned_server_pids())
    if pid not in pids:
        pids.append(pid)
    get_lazy_registry().set('server_manager.owned_pids', pids)


def _discard_owned_server_pid(pid):
    """移除所有权登记（已停止/非本进程启动）/ Drop ownership record."""
    pids = [p for p in _get_owned_server_pids() if p != pid]
    get_lazy_registry().set('server_manager.owned_pids', pids)


def _set_thread_server_owned(owned):
    """标记线程模式服务是否由本进程启动（退出时需 shutdown）/ Mark thread-mode ownership."""
    get_lazy_registry().set('server_manager.thread_owned', bool(owned))


def _is_thread_server_owned():
    """线程模式服务是否由本进程启动 / Whether thread-mode server is owned."""
    return bool(get_lazy_registry().get('server_manager.thread_owned'))


def _cleanup_owned_servers(reason="exit"):
    """
    退出兜底：停止本进程启动的全部服务器（幂等，可安全重复调用）。

    参数:
    reason (str): 触发来源（atexit / signal），仅用于日志。

    Returns:
    int: 实际停止的服务数量。

    Exit backstop: stop every server started by this process.
    Idempotent and safe to call repeatedly.
    """
    stopped = 0
    try:
        # 线程模式：调用 werkzeug shutdown() 结束 serve_forever 循环
        # （必须从非服务线程调用；atexit/信号处理运行在主线程，满足条件）
        # Thread mode: call werkzeug shutdown() to end the serve_forever
        # loop (must be called off the serving thread; atexit/signal
        # handlers run on the main thread, which qualifies).
        if _is_thread_server_owned():
            _set_thread_server_owned(False)
            instance = _get_server_instance()
            if instance is not None:
                try:
                    instance.shutdown()
                    stopped += 1
                except Exception as e:
                    print(f"[server_manager] thread-server shutdown failed ({reason}): {e}")
        # 子进程模式：逐个停掉仍存活的本进程所有服务器
        # Subprocess mode: stop each owned server still alive.
        for pid in list(_get_owned_server_pids()):
            _discard_owned_server_pid(pid)
            if is_process_running(pid):
                try:
                    stop_server()
                    stopped += 1
                except Exception as e:
                    print(f"[server_manager] owned server {pid} stop failed ({reason}): {e}")
    except Exception as e:
        # 兜底路径绝不能抛异常阻断退出 / The backstop must never raise.
        print(f"[server_manager] exit cleanup error ({reason}): {e}")
    return stopped


def _make_signal_cleanup_handler(prev_handler):
    """
    构造信号处理器：先清理自有服务器，再接续原有处理链。

    参数:
    prev_handler: 注册前 signal.getsignal() 返回的原处理器。

    Returns:
    新的信号处理函数。

    Build a signal handler: clean up owned servers first, then chain to
    whatever handler was installed before.
    """

    def _handler(signum, frame):
        _cleanup_owned_servers(f"signal {signum}")
        if callable(prev_handler):
            prev_handler(signum, frame)
            return
        # 恢复默认处置并重新触发，保持退出码语义
        # Restore default disposition and re-raise for exit-code semantics.
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, ValueError):
            pass
        try:
            os.kill(os.getpid(), signum)
        except (OSError, ValueError):
            sys.exit(128 + int(signum))

    return _handler


def ensure_exit_cleanup_registered():
    """
    注册退出兜底（幂等）：atexit + SIGINT/SIGTERM/SIGBREAK 信号钩子。

    首次登记所有权时由启动路径自动调用；重复调用无副作用。

    Register the exit backstop (idempotent): atexit + signal hooks.
    Called automatically by the start paths on first ownership.
    """
    if get_lazy_registry().get('server_manager.cleanup_registered'):
        return
    get_lazy_registry().set('server_manager.cleanup_registered', True)
    atexit.register(_cleanup_owned_servers, "atexit")
    # SIGHUP 仅 POSIX 存在；SIGBREAK 仅 Windows 存在
    # SIGHUP exists only on POSIX; SIGBREAK only on Windows.
    handled = [sig for sig in (
        signal.SIGINT, signal.SIGTERM,
        getattr(signal, 'SIGHUP', None),
        getattr(signal, 'SIGBREAK', None),
    ) if sig is not None]
    for sig in handled:
        try:
            prev = signal.getsignal(sig)
            # 已被他人安装的可调用处理器保留并接续，默认/忽略处置则接管
            # Keep chaining to existing callable handlers; take over defaults.
            signal.signal(sig, _make_signal_cleanup_handler(prev))
        except (OSError, ValueError):
            # 个别信号在当前平台不允许安装，跳过即可
            # Some signals cannot be installed on this platform; skip.
            continue


def start_server_thread(port=None, log_callback=None, auto_cleanup=True):
    port = config.server.PORT if port is None else port
    """
    在线程中启动服务器（用于打包模式）

    Args:
        port: 监听端口
        log_callback: 日志回调函数
        auto_cleanup: 本进程退出时兜底停止（CLI 守护场景传 False）

    Returns:
        tuple: (success: bool, message: str, thread: Thread or None)
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    # 检查是否已经运行
    t = get_t()
    if check_server_health(port):
        log(t("server.server_already_running", port=port))
        return True, "Server already running", _get_server_thread()

    try:
        # 导入服务器模块
        from birdid_server import app, ensure_models_loaded
        from werkzeug.serving import make_server

        log(t("server.packaged_mode_thread"))

        # 线程模式登记所有权：进程退出时 shutdown() 兜底
        # Thread mode ownership: shutdown() backstop on process exit.
        _set_thread_server_owned(auto_cleanup)
        if auto_cleanup:
            ensure_exit_cleanup_registered()
        
        def run_server():
            try:
                # 异步预加载模型（不阻塞服务器启动）
                def load_models_async():
                    try:
                        log(t("server.loading_models"))
                        ensure_models_loaded()
                        log(t("server.models_loaded"))
                    except Exception as e:
                        log(t("server.model_load_error", error=e))
                
                # 在后台线程中加载模型
                model_thread = threading.Thread(target=load_models_async, daemon=True)
                model_thread.start()
                
                # 立即启动服务器，不等待模型加载完成
                _set_server_instance(make_server(config.server.HOST, port, app, threaded=True))
                log(t("server.server_started", port=port))
                _get_server_instance().serve_forever()
            except Exception as e:
                log(t("server.server_thread_error", error=e))
        
        # 创建并启动守护线程
        server_thread = threading.Thread(target=run_server, daemon=True, name="BirdID-API-Server")
        _set_server_thread(server_thread)
        server_thread.start()
        
        wait_seconds = max(1.0, float(config.server.STARTUP_WAIT_SECONDS))
        poll_interval = max(0.1, float(config.server.POLL_INTERVAL_SECONDS))
        max_attempts = max(1, int(wait_seconds / poll_interval))
        for i in range(max_attempts):
            time.sleep(poll_interval)
            if check_server_health(port):
                log(t("server.server_health_ok", port=port))
                return True, "Server start success", _get_server_thread()
        
        log(t("server.server_timeout"))
        return True, "Server starting", _get_server_thread()
        
    except Exception as e:
        log(t("server.thread_start_failed", error=e))
        import traceback
        traceback.print_exc()
        return False, str(e), None


def start_server_daemon(port=None, log_callback=None, auto_cleanup=True):
    port = config.server.PORT if port is None else port
    """
    启动服务器

    打包模式下使用线程方式启动（避免重复启动整个应用）
    开发模式下使用子进程方式启动

    Args:
        port: 监听端口
        log_callback: 日志回调函数
        auto_cleanup: 本进程退出时兜底停止自有服务器
                      （`server_manager.py start` 的 LR 插件守护场景须传 False）

    Returns:
        tuple: (success: bool, message: str, pid: int or None)
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    # 检查是否已经运行
    t = get_t()
    status = get_server_status(port)
    if status['healthy']:
        log(t("server.server_already_running", port=port))
        # 复用他人启动的健康服务器：不登记所有权，退出时不动它
        # Reusing someone else's healthy server: no ownership, don't touch.
        return True, "Server already running", status['pid']

    # 如果端口被占用但不健康，可能是僵尸进程
    if status['running'] and not status['healthy']:
        log(t("server.zombie_process"))
        stop_server()
        time.sleep(1)

    # 检测运行模式
    is_frozen = getattr(sys, 'frozen', False)

    if is_frozen:
        # 打包模式：使用线程方式启动
        log(t("server.packaged_mode_detected"))
        success, message, thread = start_server_thread(port, log_callback,
                                                       auto_cleanup=auto_cleanup)
        # 线程模式没有独立 PID，返回主进程 PID
        return success, message, os.getpid() if success else None
    else:
        # 开发模式：使用子进程方式启动
        log(t("server.dev_mode_subprocess"))
        return _start_server_subprocess(port, log_callback,
                                        auto_cleanup=auto_cleanup)


def _start_server_subprocess(port=None, log_callback=None, auto_cleanup=True):
    port = config.server.PORT if port is None else port
    """
    以子进程方式启动服务器（仅开发模式使用）

    Args:
        port: 监听端口
        log_callback: 日志回调函数
        auto_cleanup: 登记所有权并在本进程退出时兜底停止
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    
    python_exe = sys.executable
    server_script = get_server_script_path()
    
    t = get_t()
    
    if not os.path.exists(server_script):
        return False, f"Server script not found: {server_script}", None
    
    cmd = [python_exe, server_script, '--port', str(port)]
    log(t("server.starting_daemon", cmd=' '.join(cmd)))
    
    try:
        # 以守护进程方式启动（分离子进程）
        if sys.platform == 'darwin':
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True
            )
        elif sys.platform == 'win32':
            # Windows: 使用 CREATE_NO_WINDOW 标志避免显示控制台窗口
            # 注意：CREATE_NO_WINDOW 在 Python 3.7+ 中可用
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    start_new_session=False
                )
            except AttributeError:
                # 如果 CREATE_NO_WINDOW 不可用，使用 CREATE_NEW_CONSOLE 和 DETACHED_PROCESS
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                    start_new_session=False
                )
        else:
            # Linux/Unix
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True
            )
        
        write_pid(process.pid)
        # 立即登记所有权：即使后续健康检查超时，退出兜底也能停掉它
        # Register ownership immediately: the exit backstop covers the
        # child even if the health check below times out.
        if auto_cleanup:
            _add_owned_server_pid(process.pid)
            ensure_exit_cleanup_registered()
        log(t("server.server_pid", pid=process.pid))

        # 等待服务器启动
        for i in range(10):
            time.sleep(0.5)
            if check_server_health(port):
                log(t("server.server_health_ok", port=port))
                # 健康后按端口解析真实服务进程：Popen 拿到的可能只是 venv
                # 垫脚启动器，PID 文件与所有权都要指向真正持有端口的进程，
                # 否则 stop_server 停不掉（垫脚进程等待子进程、不持有端口）。
                # Once healthy, resolve the real server pid by port: the
                # Popen pid may be a venv launcher shim that neither holds
                # the port nor dies with the server. Both the pid file and
                # the ownership record must target the true listener.
                real_pid = find_listener_pid(port) or process.pid
                if real_pid != process.pid:
                    write_pid(real_pid)
                    _discard_owned_server_pid(process.pid)
                    _add_owned_server_pid(real_pid)
                    log(f"[server_manager] pid 修正: {process.pid} → {real_pid} (venv launcher shim)")
                return True, "Server start success", real_pid

        if is_process_running(process.pid):
            log(t("server.server_started_health_fail"))
            return True, "Server starting", process.pid
        else:
            log(t("server.server_process_exited"))
            _discard_owned_server_pid(process.pid)
            remove_pid()
            return False, "服务器启动失败", None
            
    except Exception as e:
        log(t("server.start_failed", error=e))
        return False, str(e), None


def stop_server(log_callback=None):
    """
    停止服务器
    
    Returns:
        tuple: (success: bool, message: str)
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    
    t = get_t()
    pid = read_pid()

    def _send_signal(sig):
        if sys.platform == 'win32':
            return
        try:
            os.killpg(pid, sig)
        except Exception:
            os.kill(pid, sig)
    
    if pid and is_process_running(pid):
        log(t("server.stop_server", pid=pid))
        try:
            # Windows 平台使用不同的信号处理
            if sys.platform == 'win32':
                # Windows 没有 SIGTERM/SIGKILL，使用 terminate() 方法
                import subprocess
                try:
                    # 尝试使用 taskkill 命令（/T 连同子进程一起结束：
                    # venv 垫脚启动器下真实服务是 recorded pid 的子进程）
                    # taskkill with /T takes the whole tree: under the venv
                    # launcher shim the real server is a child of the pid.
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                                  capture_output=True, timeout=5)
                except Exception:
                    # 如果 taskkill 失败，尝试其他方法
                    pass
            else:
                # Unix/Linux/macOS 平台使用信号
                _send_signal(signal.SIGTERM)
            
            # 等待进程退出
            for i in range(24):
                time.sleep(0.25)
                if not is_process_running(pid):
                    break
            
            # 如果还没退出，强制终止（仅限非Windows平台）
            if is_process_running(pid) and sys.platform != 'win32':
                log(t("server.force_kill"))
                _send_signal(signal.SIGKILL)
                time.sleep(0.5)
            
            remove_pid()
            _discard_owned_server_pid(pid)
            log(t("server.server_stopped"))
            return True, "Server stopped"

        except Exception as e:
            log(t("server.stop_failed", error=e))
            remove_pid()
            _discard_owned_server_pid(pid)
            return False, str(e)
    else:
        # 清理可能的僵尸 PID 文件
        remove_pid()
        log(t("server.server_not_running"))
        return True, "Server not running"


def restart_server(port=None, log_callback=None):
    """重启服务器"""
    port = config.server.PORT if port is None else port
    stop_server(log_callback)
    time.sleep(1)
    return start_server_daemon(port, log_callback)


# 命令行入口
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BirdID 服务器管理器')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'],
                        help='操作: start/stop/restart/status')
    parser.add_argument('--port', type=int, default=config.server.PORT, help='端口号')
    
    args = parser.parse_args()
    
    if args.action == 'start':
        # CLI 启动 = 独立守护（Lightroom 插件等外部调用方），启动器进程随即
        # 退出，服务必须存活：显式豁免退出兜底。
        # CLI start = standalone daemon (e.g. Lightroom plugin caller); the
        # starter exits right away, so opt out of the exit backstop.
        success, msg, pid = start_server_daemon(args.port, auto_cleanup=False)
        print(msg)
        sys.exit(0 if success else 1)
        
    elif args.action == 'stop':
        success, msg = stop_server()
        print(msg)
        sys.exit(0 if success else 1)
        
    elif args.action == 'restart':
        success, msg, pid = restart_server(args.port)
        print(msg)
        sys.exit(0 if success else 1)
        
    elif args.action == 'status':
        status = get_server_status(args.port)
        print(f"运行状态: {'运行中' if status['running'] else '未运行'}")
        print(f"健康状态: {'正常' if status['healthy'] else '异常'}")
        print(f"PID: {status['pid'] or 'N/A'}")
        print(f"端口: {status['port']}")
        sys.exit(0 if status['healthy'] else 1)
