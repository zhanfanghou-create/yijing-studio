#!/usr/bin/env python
"""H3-Base 在 4090D 48G 上到底能不能跑：只做「加载」这一步的实验。

结论要的是数字，不是印象：
  - 权重有多少字节（bf16 = 参数量 × 2）
  - 峰值内存 / 显存到哪
  - 从 from_pretrained 到「可以调用」要多久

不生成视频。生成放在确认能加载之后，否则一跑半小时才知道 OOM。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

MODEL = os.environ.get("YJ_VIDEO_MODEL", "/root/cloud/models/MiniMax-H3/FL2VA")


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def gpu_gb() -> tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return (0.0, 0.0)
        return (
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)


def log(msg: str) -> None:
    a, r = gpu_gb()
    print(f"[{time.strftime('%H:%M:%S')}] {msg} | RSS {rss_gb():.1f}G | VRAM {a:.1f}/{r:.1f}G", flush=True)


def main() -> int:
    log("开始")
    import torch
    # model_index.json 里写的是 MiniMaxH3Pipeline，但 diffusers 里真正的类名是
    # MiniMaxH3ModularPipeline（模块化管线），所以不能走通用的 ModularPipeline.from_pretrained
    # ——它按 _class_name 去 diffusers 命名空间里找类，找不到就 AttributeError。
    from diffusers import MiniMaxH3ModularPipeline as Pipe
    import inspect

    log(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")
    log(f"load_components 签名 {inspect.signature(Pipe.load_components)}")
    if not os.path.isdir(MODEL):
        print(f"模型目录不存在: {MODEL}")
        return 2

    t0 = time.time()
    log("from_pretrained（读 model_index + 组件类）…")
    try:
        pipe = Pipe.from_pretrained(MODEL)
    except Exception as e:  # noqa: BLE001
        log(f"from_pretrained 失败: {type(e).__name__}: {e}")
        return 3
    log(f"from_pretrained 完成 {time.time() - t0:.1f}s")

    # 组件清单 —— 先看清要加载多少个、多大
    try:
        blocks = pipe.blocks if hasattr(pipe, "blocks") else None
        names = [getattr(b, "name", type(b).__name__) for b in blocks] if blocks else []
        log(f"blocks({len(names)}): {names[:12]}")
    except Exception as e:  # noqa: BLE001
        log(f"读 blocks 失败: {e}")

    log("load_components(bf16) … 这一步最可能 OOM")
    t1 = time.time()
    try:
        pipe.load_components(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    except TypeError as e:
        log(f"带 low_cpu_mem_usage 不支持({e})，退回 load_components(torch_dtype=…)")
        try:
            pipe.load_components(torch_dtype=torch.bfloat16)
        except Exception as e2:  # noqa: BLE001
            log(f"load_components 失败: {type(e2).__name__}: {e2}")
            return 4
    except Exception as e:  # noqa: BLE001
        log(f"load_components 失败: {type(e).__name__}: {e}")
        return 4

    log(f"load_components 完成，耗时 {time.time() - t1:.1f}s（总 {time.time() - t0:.1f}s）")

    # 能不能只加载单个组件？决定「分段加载」这条路走不走得通
    try:
        log(f"load_components 支持的参数 = {inspect.signature(pipe.load_components)}")
    except Exception:  # noqa: BLE001
        pass
    try:
        log(f"offload 方法 = {[n for n in dir(pipe) if 'offload' in n]}")
    except Exception:  # noqa: BLE001
        pass

    gc.collect()
    res = {
        "ok": True,
        "rss_gb": round(rss_gb(), 2),
        "vram_alloc_gb": round(gpu_gb()[0], 2),
        "load_seconds": round(time.time() - t0, 1),
        "components": sorted(set(dir(pipe)) & {
            "transformer", "text_encoder", "video_vae", "audio_vae", "tokenizer", "processor",
        }),
        "has_blocks": hasattr(pipe, "blocks"),
    }
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
