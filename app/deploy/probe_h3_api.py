#!/usr/bin/env python
"""先摸清楚 diffusers 里 H3 的入口到底叫什么，再谈加载。"""
import inspect

import diffusers

print("diffusers", diffusers.__version__)

print("\n== 顶层有 MiniMax 的属性 ==")
print([n for n in dir(diffusers) if "inimax" in n or "H3" in n])

try:
    import diffusers.modular_pipelines as mp

    print("\n== modular_pipelines 属性 ==")
    print([n for n in dir(mp) if not n.startswith("_")][:40])
    print("\n== minimax 相关 ==")
    print([n for n in dir(mp) if "inimax" in n.lower() or "h3" in n.lower()])
except Exception as e:  # noqa: BLE001
    print("import modular_pipelines 失败:", e)

try:
    from diffusers import ModularPipeline

    print("\nModularPipeline.from_pretrained 签名:")
    print(inspect.signature(ModularPipeline.from_pretrained))
except Exception as e:  # noqa: BLE001
    print("ModularPipeline:", e)

# 模块路径探测
for path in (
    "diffusers.modular_pipelines.minimax_h3",
    "diffusers.modular_pipelines.minimax",
):
    try:
        m = __import__(path, fromlist=["*"])
        print(f"\n{path} ->", [n for n in dir(m) if not n.startswith("_")])
    except Exception as e:  # noqa: BLE001
        print(f"\n{path} 不可用: {type(e).__name__}: {e}")
