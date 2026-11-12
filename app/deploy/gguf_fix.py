"""修 diffusers 的 GGUF 加载 bug：一维 BF16 张量按「字节数」建参数。

复现（2026-11 diffusers 0.42.1 + unsloth 的 Qwen-Image-2.1-Turbo Q4_K_M）：

    model.diffusion_model.txt_in.text_norm.weight  →  GGUF 里 4096 元素 / 8192 字节
    加载后 param.shape = (8192,)                    →  第一次前向就炸：
    RuntimeError: The size of tensor a (4096) must match the size of tensor b (8192)

二维 BF16 张量（img_in / txt_in.in_layer / out_layer）加载是对的，
错的只有一维那一批。所以要修的就是「GGUF 元素数 ≠ 参数元素数」的 BF16 张量：
按原始字节重新解成 bf16，覆盖回去。

为什么不做成「跳过这个文件」：Qwen-Image 的 12 GB 路线只有 GGUF 这一条，
而 Qwen-Image 2.1 的归一化层权重本来就是 bf16 存的一维张量 —— 不修就没图可出。
"""

from __future__ import annotations

from pathlib import Path


def _strip_prefix(name: str) -> str:
    for pre in ("model.diffusion_model.", "model."):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def split_fused_mlp(model, gguf_path: str | Path, log=print) -> int:
    """把 GGUF 里融合的 `img_mlp.gate_up` 拆成模型要的 `gate_layer` + `proj`。

    社区量化（unsloth / city96 / ComfyUI 口径）把 SwiGLU 的 gate 与 up 拼成一个
    张量存，键名 `img_mlp.gate_up.weight`；而 diffusers 的
    `QwenImage21Transformer2DModel` 是两个独立 Linear：`gate_layer` / `proj`。

    后果是 `from_single_file` 按名字对不上 —— 每个 block 的这两个权重**静默保持随机
    初始化**（没有 missing keys 报错，`strict=False` 就是这么安静），于是模型能跑完、
    不报错、吐出一张纯噪声图。实测就是这样浪费了 20 分钟。

    为什么按字节切而不是反量化后再切：Q4_K 是 256 元素一块，gate_up 的
    out 维是 24576 = 2×12288，正好整块对齐，直接对半切字节得到两个合法的
    Q4_K 张量，保留 GGUFLinear 在 CUDA 上的融合算子（比每层反量化快得多）。

    前半是 gate、后半是 up —— 对齐 HF `LlamaMLP` 的 `gate_proj`/`up_proj` 拼接顺序。
    """
    try:
        import numpy as np
        import torch
        from gguf import GGUFReader
    except Exception as e:  # noqa: BLE001
        log(f"跳过融合 MLP 拆分（缺依赖）：{e!r}")
        return 0

    try:
        from diffusers.quantizers.gguf.utils import (  # noqa: PLC0415
            GGUFLinear,
            GGUFParameter,
        )
    except Exception as e:  # noqa: BLE001
        log(f"跳过融合 MLP 拆分（diffusers 没有 GGUF 工具）：{e!r}")
        return 0

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as e:  # noqa: BLE001
        log(f"跳过融合 MLP 拆分（读不了文件）：{e!r}")
        return 0

    done = 0
    for t in reader.tensors:
        key = _strip_prefix(t.name)
        if not key.endswith("img_mlp.gate_up.weight"):
            continue
        blk = key.rsplit(".img_mlp.gate_up.weight", 1)[0]
        try:
            mlp = model.get_submodule(f"{blk}.img_mlp")
        except Exception:
            continue
        try:
            raw = torch.from_numpy(np.ascontiguousarray(t.data)).view(torch.uint8)
            if raw.shape[0] % 2:
                raise RuntimeError(f"out 维 {raw.shape[0]} 不是偶数，没法对半切")
            half = raw.shape[0] // 2
            w_gate = GGUFParameter(raw[:half].clone(), quant_type=t.tensor_type)
            w_up = GGUFParameter(raw[half:].clone(), quant_type=t.tensor_type)
            out_f, in_f = w_gate.quant_shape  # GGUF 的 quant_shape 是 (out, in)
            for attr, w in (("gate_layer", w_gate), ("proj", w_up)):
                lin = GGUFLinear(
                    in_features=int(in_f),
                    out_features=int(out_f),
                    bias=False,
                    compute_dtype=torch.bfloat16,
                )
                lin.weight = w
                setattr(mlp, attr, lin)
            done += 1
        except Exception as e:  # noqa: BLE001
            log(f"拆分 {blk} 失败：{e!r}")

    if done:
        log(f"融合 MLP 已拆开：{done} 个 block 的 gate_up → gate_layer + proj")
    else:
        log("没有找到 img_mlp.gate_up（权重可能是标准 diffusers 命名，不用拆）")
    return done


def verify_keys(model, gguf_path: str | Path, log=print) -> dict:
    """逐键对账：GGUF 里有多少张量真的落到了模型参数上。

    这是「不吐噪声图」的守门人。`from_single_file` 对不上的键是静默跳过的，
    唯一可靠的判据是自己比一遍：GGUF 张量数 vs 模型参数数 vs 对上的数。
    """
    try:
        import torch
        from gguf import GGUFReader
    except Exception:  # noqa: BLE001
        return {}
    try:
        reader = GGUFReader(str(gguf_path))
    except Exception:  # noqa: BLE001
        return {}

    have = {k for k, _ in model.named_parameters()}
    gg = [_strip_prefix(t.name) for t in reader.tensors]
    # gate_up 会变成两个键，其余一对一
    expanded, missing = [], []
    for n in gg:
        if n.endswith("img_mlp.gate_up.weight"):
            base = n.rsplit(".gate_up.weight", 1)[0]
            expanded += [f"{base}.gate_layer.weight", f"{base}.proj.weight"]
        else:
            expanded.append(n)
    for n in expanded:
        if n not in have:
            missing.append(n)
    unmatched_model = [k for k in have if k not in set(expanded)]
    info = {
        "ggufTensors": len(gg),
        "modelParams": len(have),
        "matched": len(expanded) - len(missing),
        "missingInModel": missing[:8],
        "missingCount": len(missing),
        "unfilledModelParams": unmatched_model[:8],
        "unfilledCount": len(unmatched_model),
    }
    if missing or unmatched_model:
        log(
            f"⚠️ GGUF 对账不齐：缺 {len(missing)} 项（{missing[:4]}），"
            f"模型多出 {len(unmatched_model)} 项（{unmatched_model[:4]}）"
        )
    else:
        log(f"GGUF 对账通过：{len(expanded)} 项权重全部落到模型参数上")
    return info


def repair_bf16_1d(model, gguf_path: str | Path, log=print) -> int:
    """把被放大的 BF16 一维参数改回正确形状，返回修了几个。"""
    try:
        import torch
        from gguf import GGMLQuantizationType, GGUFReader
    except Exception as e:  # noqa: BLE001
        log(f"跳过 GGUF BF16 修补（缺依赖）：{e!r}")
        return 0

    try:
        reader = GGUFReader(str(gguf_path))
    except Exception as e:  # noqa: BLE001
        log(f"跳过 GGUF BF16 修补（读不了文件）：{e!r}")
        return 0

    fixed = 0
    for t in reader.tensors:
        if t.tensor_type != GGMLQuantizationType.BF16 or len(t.shape) != 1:
            continue
        key = _strip_prefix(t.name)
        mod_path, _, leaf = key.rpartition(".")
        try:
            parent = model.get_submodule(mod_path)
        except Exception:
            continue
        cur = getattr(parent, leaf, None)
        want = int(t.n_elements)
        if cur is None or cur.numel() == want:
            continue
        try:
            raw = bytes(t.data.tobytes())
            buf = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
            if buf.numel() != want:
                raise RuntimeError(f"字节数对不上：{buf.numel()} vs {want}")
            device = cur.device
            dtype = cur.dtype
            setattr(
                parent,
                leaf,
                torch.nn.Parameter(buf.to(device=device, dtype=dtype).reshape(want), requires_grad=False),
            )
            fixed += 1
            log(f"修补 {key}：{tuple(cur.shape)} → ({want},)")
        except Exception as e:  # noqa: BLE001
            log(f"修补 {key} 失败：{e!r}")
    return fixed
