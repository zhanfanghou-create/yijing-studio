#!/usr/bin/env python
"""MiniMaxH3ModularPipeline 的入口组合到底哪种能过：print 每种尝试的结果。"""
import inspect

from diffusers import MiniMaxH3Blocks, MiniMaxH3ModularPipeline, ModularPipeline

MODEL = "/root/cloud/models/MiniMax-H3/FL2VA"

print("MiniMaxH3ModularPipeline.__init__ =", inspect.signature(MiniMaxH3ModularPipeline.__init__))
print("MiniMaxH3Blocks.__init__ =", inspect.signature(MiniMaxH3Blocks.__init__))
print("Pipe.default_blocks_name =", getattr(MiniMaxH3ModularPipeline, "default_blocks_name", "<无>"))
print("Blocks.default_blocks_name =", getattr(MiniMaxH3Blocks, "default_blocks_name", "<无>"))
print("Blocks 有 default_blocks_name 类属性吗 =", "default_blocks_name" in vars(MiniMaxH3Blocks))

from diffusers.modular_pipelines import modular_pipeline as mpm  # noqa: E402

print("\nmodular_pipeline 模块 =", [n for n in dir(mpm) if not n.startswith("_")])

tries = [
    ("blocks=MiniMaxH3Blocks()", lambda: MiniMaxH3ModularPipeline.from_pretrained(MODEL, blocks=MiniMaxH3Blocks())),
    ("blocks=MiniMaxH3Blocks", lambda: ModularPipeline.from_pretrained(MODEL, blocks=MiniMaxH3Blocks)),
    ("Pipe(blocks=...) + load", None),
]
for name, fn in tries:
    if fn is None:
        continue
    try:
        p = fn()
        print(f"[OK] {name} -> {type(p).__name__}")
    except Exception as e:  # noqa: BLE001
        print(f"[NO] {name} -> {type(e).__name__}: {str(e)[:160]}")
