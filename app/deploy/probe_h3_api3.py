#!/usr/bin/env python
"""H3 加载：绕开 from_pretrained 的类名解析，直接用构造函数塞 blocks。

model_index.json 是老格式（_class_name: MiniMaxH3Pipeline），diffusers 0.41 的
modular 加载器按 _blocks_class_name 找 block，老格式里没有这个字段，所以
from_pretrained 一定报 "No pipeline blocks could be resolved"。
既然 MiniMaxH3Blocks 就在 diffusers 命名空间里，那就自己把它传进去。
"""
from __future__ import annotations

import time

import torch
from diffusers import MiniMaxH3Blocks, ModularPipeline

MODEL = "/root/cloud/models/MiniMax-H3/FL2VA"


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m} | RSS {rss_gb():.1f}G", flush=True)


t0 = time.time()
log("构造 blocks")
blocks = MiniMaxH3Blocks()
log(f"blocks 就绪，子块 {len(getattr(blocks, 'block_classes', {}))} 个: {list(getattr(blocks, 'block_classes', {}))[:10]}")

log("构造 pipeline（含读 model_index 组件类）")
try:
    pipe = ModularPipeline(pretrained_model_name_or_path=MODEL, blocks=blocks)
    log(f"pipeline 就绪 {time.time() - t0:.1f}s")
except Exception as e:  # noqa: BLE001
    log(f"构造失败 {type(e).__name__}: {e}")
    raise SystemExit(3)

log("load_components(bf16) —— 62G DiT + 63G 文本编码器，这里会决定成不成")
t1 = time.time()
try:
    pipe.load_components(torch_dtype=torch.bfloat16)
    log(f"load_components OK，{time.time() - t1:.1f}s")
    print("RESULT ok rss=%.1f" % rss_gb(), flush=True)
except Exception as e:  # noqa: BLE001
    log(f"load_components 失败 {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    raise SystemExit(4)
