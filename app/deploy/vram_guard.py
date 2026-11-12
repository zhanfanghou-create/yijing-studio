"""显存/CPU 硬保护：宁可这一次任务报错，也不许拖死整机。

## 为什么需要这个文件

2026-11-10 本机（RTX 4070 12 GB / 31.8 GB RAM）跑 Qwen-Image 1024² 时：

    [gen] 1112.1s @ 1024x1024 / 4 steps
    {"peak_alloc_gb": 21.0, "peak_reserved_gb": 21.32}

21 GB 峰值里有 9 GB 不在显卡上 —— Windows 的 **CUDA 系统内存回退**
（sysmem fallback）把放不下的显存分配落到普通内存，靠 PCIe 反复换页。
结果不是 OOM，而是**整台机器卡到鼠标都不动**，用户操作不了。

所以低显存档必须做两件事：

1. **画一条硬线**：`set_per_process_memory_fraction`。超线立刻
   `torch.cuda.OutOfMemoryError`，报错可以重试，卡死只能重启。
2. **不要把 CPU 抢满**：`torch.set_num_threads` 限住。torch 默认按逻辑核数
   开满，24 线程跑 safetensors 解压会把系统调度饿死，界面同样卡。

预算来源（优先级从高到低）：
    YJ_VRAM_BUDGET_GB   → 绝对 GB 数
    YJ_VRAM_FRACTION    → 占总显存的百分比（默认 0.88）
    都没设              → 0.88

留 12% 是给显示输出、浏览器、Electron 合成器用的。4070 12 GB 上是
`10.5 GB` 可用 —— 足够放 Q4 量化的 transformer，但放不下 bf16 全量，
这正是我们要的行为：**大模型档在该卡上就应该明确失败**。
"""

from __future__ import annotations

import os


def _f(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, "") or default)
    except ValueError:
        return default


def cap_vram(log=print, tag: str = "") -> float | None:
    """按预算限制本进程可用显存，返回生效的预算 GB（不可用时返回 None）。

    必须在**任何**张量分配到 cuda 之前调用。
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None

    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
        total = total_b / 2**30
        free = free_b / 2**30
        budget = _f("YJ_VRAM_BUDGET_GB", 0.0)
        if budget <= 0:
            budget = total * min(1.0, max(0.3, _f("YJ_VRAM_FRACTION", 0.88)))
        budget = min(budget, total)
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
        prefix = f"[{tag}] " if tag else ""
        log(
            f"{prefix}显存预算 {budget:.1f} / {total:.1f} GB"
            f"（当前空闲 {free:.1f}），超线即报错不回落系统内存"
        )
        return round(budget, 2)
    except Exception as e:  # noqa: BLE001
        log(f"显存预算设置失败（不致命，但可能回落系统内存）：{e!r}")
        return None


def cap_threads(log=print, tag: str = "") -> int:
    """限住 torch 的计算线程，别把系统调度饿死。"""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return 0
    want = int(_f("YJ_TORCH_THREADS", 0.0))
    if want <= 0:
        cores = os.cpu_count() or 4
        want = max(1, min(8, cores - 4))
    try:
        torch.set_num_threads(want)
        try:
            torch.set_num_interop_threads(max(1, want // 2))
        except Exception:
            pass  # 进程内已经调过就不允许再改
        prefix = f"[{tag}] " if tag else ""
        log(f"{prefix}torch 线程 {want}（共 {os.cpu_count()} 逻辑核，留余量给界面）")
        return want
    except Exception as e:  # noqa: BLE001
        log(f"线程数设置失败：{e!r}")
        return 0


def setup(log=print, tag: str = "") -> dict:
    """一次装好显存与线程两条线，返回生效值用于写进状态文件。"""
    th = cap_threads(log, tag)
    vr = cap_vram(log, tag)
    return {"vram_budget_gb": vr, "torch_threads": th}
