"""轻量内存监控：后台轮询 /proc/meminfo，内存紧张时打印警告。"""

import threading
import time

# 可用内存低于该阈值（MB）时打印警告
MEM_WARN_AVAILABLE_MB = 1024
MEM_POLL_INTERVAL_SEC = 2


def read_meminfo_mb():
    info = {}
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            key, value = line.split(':')
            info[key] = int(value.strip().split()[0]) // 1024  # kB -> MB
    return {
        'total': info.get('MemTotal', 0),
        'available': info.get('MemAvailable', 0),
        'swap_total': info.get('SwapTotal', 0),
        'swap_free': info.get('SwapFree', 0),
    }


def _watchdog(stop_event, warn_mb, interval_sec):
    warned = False
    while not stop_event.wait(interval_sec):
        mem = read_meminfo_mb()
        swap_used = mem['swap_total'] - mem['swap_free']
        if mem['available'] < warn_mb:
            print(
                f'\n[MEM WARNING] 可用内存仅剩 {mem["available"]}MB / {mem["total"]}MB，'
                f'swap 已用 {swap_used}MB！很可能会因内存不足中断，请留意进程是否被杀。\n',
                flush=True,
            )
            warned = True
        elif warned and mem['available'] >= warn_mb * 2:
            print(
                f'\n[MEM INFO] 可用内存回升至 {mem["available"]}MB，暂时缓解。\n',
                flush=True,
            )
            warned = False


def start_mem_watchdog(warn_mb=MEM_WARN_AVAILABLE_MB, interval_sec=MEM_POLL_INTERVAL_SEC):
    """启动后台内存监控，返回 (stop_event, thread)。结束时调用 stop_event.set()。"""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_watchdog,
        args=(stop_event, warn_mb, interval_sec),
        daemon=True,
    )
    thread.start()
    mem = read_meminfo_mb()
    print(
        f'[MEM] 已启动内存监控：可用 < {warn_mb}MB 时告警 '
        f'(当前可用 {mem["available"]}MB)',
        flush=True,
    )
    return stop_event, thread


def stop_mem_watchdog(stop_event, thread, timeout=3):
    """停止后台内存监控。"""
    stop_event.set()
    thread.join(timeout=timeout)
