#!/usr/bin/env python3
"""
Wan2.2-TI2V-5B worker（低显存档 · 12 GB 消费卡）

和 `video_worker.py` 是**同一份契约**：命令行参数、状态 JSON、产物落盘位置全一样，
由 model-server 按后端选择器二选一。这样换后端不动 API 层与前端。

为什么不复用 video_worker.py：那条路径绑死了 H3（DiffSynth 的 NF4 单文件格式、
`17n+5` 帧规则、原生音频合流）。Wan 是纯视频、帧规则是 `4n+1`、权重是 GGUF，
硬塞进一个函数会变成一堆 if，两边都难维护。

低显存的三条关键配置（12 GB 卡能跑的原因）：
  1. **权重 GGUF Q4**（约 3.4 GB）—— 从 20 GB bf16 压到 3.4 GB；
  2. **`enable_model_cpu_offload()`** —— umT5-XXL（11.4 GB）编码完就挪回内存，
     不和采样抢显存；
  3. **VAE tiling** —— 解码 49–121 帧时激活值不一次性铺满。

用法（model-server 会这么调）：
    python wan_worker.py --job xx --status xx.json --out DIR --port 3911 \
        --model <dir> --prompt "..." --size 832x480 --duration 5 \
        [--image first.png] [--last-frame last.png] [--seed 42] [--steps 8]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import traceback
from pathlib import Path


def write_status(path, **kw):
    """原子写：半截 JSON 会让轮询端读到脏数据。"""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), "utf-8")
    tmp.replace(p)


def norm_size(size: str, default=(832, 480)):
    try:
        w, h = [int(float(x)) for x in str(size).lower().replace("*", "x").split("x")]
    except Exception:
        return default
    # Wan 的 patch 是 2×16×16 + VAE 16 倍空间压缩：长宽都要是 32 的倍数。
    # 舍入显式写 floor(v/32 + 0.5)：Python 的 round 是银行家舍入，
    # 和上游 `_video_size()` / `src/shared/size.ts` 的算法在 .5 处会错开一档。
    align = lambda v: max(256, int(v / 32.0 + 0.5) * 32)  # noqa: E731
    return align(w), align(h)


def frames_for(duration: float, fps: int = 24, cap: int = 121) -> int:
    """`4n+1` 帧（VAE 时间压缩 4），且不超过模型上限 121 帧。"""
    want = max(5, int(round(duration * fps)))
    n = max(1, int(round((want - 1) / 4.0)))
    return min(cap, 4 * n + 1)


def pick_gguf(model_dir: Path, explicit: str | None) -> Path:
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    env = os.environ.get("YJ_VIDEO_GGUF", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    cands = sorted(model_dir.glob("*.gguf"))
    if not cands:
        raise RuntimeError(
            f"{model_dir} 里没有 .gguf 权重。低显存档需要 GGUF（Q4_K_M 约 3.4 GB）"
        )
    # 优先 Turbo（4 步蒸馏）：同样是 Q4_K_M，只因权重被蒸馏过，
    # 就能把 30+ 步的采样压到 4 步 —— 12 GB 卡上这是最大的那个倍数。
    for prefer in ("Turbo", "turbo"):
        for c in cands:
            if prefer in c.name and "Q4_K_M" in c.name:
                return c
    # 其次 Q4_K_M：12 GB 卡的性价比档
    for prefer in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"):
        for c in cands:
            if prefer in c.name:
                return c
    return cands[0]


def is_turbo(gguf_path: Path) -> bool:
    """蒸馏权重（Turbo / Lightning / Distill）：4 步、CFG 1.0、不吃负向提示词。

    判文件名而不是判配置：这些仓库都是「同一个架构、不同权重」，
    目录里的 config.json 完全一样，能区分的只有权重名。
    """
    n = gguf_path.name.lower()
    return any(k in n for k in ("turbo", "lightning", "distill", "4step", "4-step"))


def load_transformer(gguf_path: Path, torch, model_dir: Path):
    from diffusers import GGUFQuantizationConfig  # noqa: PLC0415
    from diffusers.models.transformers.transformer_wan import (  # noqa: PLC0415
        WanTransformer3DModel,
    )

    # **必须显式给 config**：不给的话 diffusers 会去 Hub 上按默认仓库名
    # （Wan-AI/Wan2.1-I2V-14B-480P-Diffusers）拉 config.json —— 离线机器上
    # 报的是「does not appear to have a file named config.json」，
    # 看起来像权重坏了，其实只是它在找一份本地已有的配置。
    cfg_dir = model_dir / "transformer"
    kwargs = {}
    if (cfg_dir / "config.json").is_file():
        kwargs["config"] = str(cfg_dir)

    quant = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    model = WanTransformer3DModel.from_single_file(
        str(gguf_path),
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        **kwargs,
    )
    # 同一个 diffusers 缺陷：一维 BF16 张量会被按字节数建参数（见 deploy/gguf_fix.py）
    try:
        from gguf_fix import repair_bf16_1d  # noqa: PLC0415

        repair_bf16_1d(model, gguf_path)
    except Exception:
        pass
    return model


def build_pipe(model_dir: Path, transformer, with_image: bool):
    """按有无首帧选管线：TI2V 的 5B 是同一个 transformer 干两件事。"""
    import diffusers  # noqa: PLC0415

    names = (
        ("WanImageToVideoPipeline", "WanPipeline")
        if with_image
        else ("WanPipeline", "WanImageToVideoPipeline")
    )
    cls = None
    for name in names:
        cls = getattr(diffusers, name, None)
        if cls is not None:
            break
    if cls is None:
        raise RuntimeError(
            "当前 diffusers 没有 Wan 管线，请 pip install -U diffusers（需 ≥ 0.35）"
        )
    return cls.from_pretrained(str(model_dir), transformer=transformer, torch_dtype=None)


def call_kwargs(pipe, base: dict) -> dict:
    """只传管线真的认的形参（不同 diffusers 版本差异很大）。"""
    import inspect  # noqa: PLC0415

    params = set(inspect.signature(pipe.__call__).parameters)
    out = {}
    for k, v in base.items():
        if k in params:
            out[k] = v
        elif k == "image" and "image" not in params and "start_image" in params:
            out["start_image"] = v
        elif k == "last_image" and "end_image" in params:
            out["end_image"] = v
    return out


def force_fast_attention(torch) -> str:
    """把 SDPA 后端钉在 flash / mem-efficient 上，返回实际启用的名字。

    为什么必须显式设：PyTorch 的自动挑选在部分 head_dim / dtype 组合下会退回
    `math` 后端，而 math 会把**整张 attention 矩阵实体化**（O(n²)）。
    97 帧 480×832 是 156k token，math 后端光这一项就要十几 GB ——
    报出来的却是「Tried to allocate 80.00 MiB」，看着像只差一点点，
    实际是把显存全铺在了一张根本不需要落地的矩阵上。

    4070 是 sm89，flash 与 mem-efficient 都支持；两个都开，让 PyTorch 自己挑
    更快的那个。不禁用 math：万一某个头维度两个后端都吃不下，
    退回 math 虽然慢，但至少能出片，比直接抛错好。
    """
    picked = []
    for name, fn in (
        ("flash", torch.backends.cuda.enable_flash_sdp),
        ("mem_efficient", torch.backends.cuda.enable_mem_efficient_sdp),
    ):
        try:
            fn(True)
            picked.append(name)
        except Exception:  # noqa: BLE001, S110
            pass
    return "+".join(picked) or "默认"


def pick_offload_mode(torch) -> str:
    """没显式配 `YJ_WAN_OFFLOAD` 时，按显存自动选档。

    为什么要自动：`model`（组件级）会把整个 5B 搬上卡，fp16 就是 10 GB ——
    12 GB 卡上加上 activations 必然撞线，而报错只写「差 80 MiB」，
    用户和我们都看不出是策略选错了。12 GB 卡必须走逐参数 offload，
    显存换时间；20 GB 以上才值得全量上卡。
    """
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:  # noqa: BLE001
        return "model"
    if total_gb >= 20:
        return "none"
    if total_gb >= 14:
        return "model"
    return "group"


def apply_offload(pipe, torch) -> None:
    """显存换时间的三档策略，由 `YJ_WAN_OFFLOAD` 选，默认 model。

    · `model`（默认）—— `enable_model_cpu_offload()`：整组件在 CPU/GPU 之间搬。
      umT5 编码器（11 GB）只在编码那一步上卡，采样期间显存全部留给 DiT。
      12 GB 卡实测峰值约 8 GB。
    · `group` —— 再细一层，按 block 过：显存峰值降到 3–4 GB，
      8 GB 甚至 6 GB 卡也能跑，代价是每个 block 都要过一次 PCIe，慢 1.5–2 倍。
    · `none` —— 全放卡上，24 GB 以上才用。
    """
    mode = (os.environ.get("YJ_WAN_OFFLOAD") or "").strip().lower()
    if not mode:
        mode = pick_offload_mode(torch)
    print(f"[offload] 模式 {mode}", flush=True)
    if mode == "none":
        pipe.to("cuda")
        return
    if mode == "sequential":
        # 逐子模块 offload：不依赖「管线结构能不能被切成 group」，
        # 靠 accelerate 的 hook 一层一层搬，峰值只剩一层权重。
        #
        # 为什么需要这一档：`group` 表面上配了 block_level/每 group 1 块，
        # 但 Wan 的 transformer 结构不被识别时，整块 5B（fp16 约 10 GB）会被
        # 当成**一个** group 搬上卡 —— 峰值和全量上卡没区别，
        # 报错却是「差 80 MiB」，查起来像差一点，其实是策略没生效。
        # 代价是慢（每层过一次 PCIe），但 12 GB 卡上能把事办成。
        try:
            pipe.enable_sequential_cpu_offload()
            return
        except Exception as e:  # noqa: BLE001
            # 不能静默 pass：这条一旦失败就会落到组件级 offload，
            # 也就是「整个 5B 上卡」，而两条路在报错上都是「差几十 MiB」，
            # 不打印真实异常根本分不出是策略没生效还是显存真不够。
            print(f"[offload] sequential 不可用：{e!r}", flush=True)
    if mode == "group":
        try:
            from diffusers.hooks import apply_group_offloading  # noqa: PLC0415

            # 用 leaf_level 而**不是** block_level。
            #
            # block_level 靠 `_no_split_modules` 找模块边界；Wan 的 DiT 没有这个
            # 声明，于是整块 5B（fp16 约 10 GB）被当成**一个** group 搬上卡 ——
            # 峰值和「全量上卡」一样，报错只是「差 80 MiB」，看着像差一点。
            # 实测三个不同分辨率 / 帧数配置，占用恒为 10.51 GiB，就是这个原因。
            # leaf_level 逐个参数搬，不依赖结构声明，一定能压下来。
            for mod, name in ((pipe.transformer, "transformer"), (pipe.text_encoder, "text_encoder")):
                try:
                    apply_group_offloading(
                        mod,
                        onload_device=torch.device("cuda"),
                        offload_device=torch.device("cpu"),
                        offload_type="leaf_level",
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[offload] {name} leaf_level 失败：{e!r}", flush=True)
            # VAE 反而要**整体上卡**。
            #
            # 它只有 ~250 MB（127M 参数 fp16），逐参数搬一次的代价却摊在每一层卷积上：
            # 实测 97 帧解码被它拖到 250 秒，占整条链路的一半以上。
            # 真正吃显存的是 transformer（5B），那个已经按 leaf_level 压住了，
            # 采样时实测只占 5.0 GB，留出的余量足够 VAE 整块解码。
            try:
                pipe.vae.to("cuda")
            except Exception as e:  # noqa: BLE001
                print(f"[offload] vae 上卡失败：{e!r}", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[offload] group 不可用：{e!r}", flush=True)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe.to("cuda")


def save_video(frames, dest: Path, fps: int) -> None:
    """写 mp4。优先 imageio（自带 ffmpeg），没有再退回 opencv。

    不用 `diffusers.utils.export_to_video` 直接写：它两个后端都没装时抛的是
    ImportError，而这条链路里「生成成功了但写不出文件」是最冤的失败。
    """
    try:
        import imageio  # noqa: PLC0415

        w = imageio.get_writer(str(dest), fps=fps, codec="libx264", quality=8,
                               macro_block_size=None)
        try:
            for f in frames:
                w.append_data(f)
        finally:
            w.close()
        return
    except Exception:
        pass
    from diffusers.utils import export_to_video  # noqa: PLC0415

    export_to_video(frames, str(dest), fps=fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=3911)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--duration", type=float, default=5)
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--backend", default="wan")
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--last-frame", default=None)
    ap.add_argument("--gguf", default=None)
    args = ap.parse_args()

    status = args.status
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"video_{args.job}.mp4"
    write_status(status, status="processing", progress=0.02, message="准备加载 Wan2.2-TI2V-5B")

    try:
        import torch  # noqa: PLC0415

        # 视频 worker 是独立进程，必须自己画线：子进程不继承父进程的显存预算。
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from vram_guard import setup as _vram_setup  # noqa: PLC0415

            write_status(
                status, status="processing", progress=0.03,
                message="限制显存预算（超线报错，不回落系统内存）",
            )
            info = _vram_setup(lambda m: write_status(
                status, status="processing", progress=0.03, message=m
            ), "wan")
            write_status(
                status, status="processing", progress=0.04,
                message=f"显存预算 {info.get('vram_budget_gb')} GB / 线程 {info.get('torch_threads')}",
            )
        except Exception as e:  # noqa: BLE001
            write_status(
                status, status="processing", progress=0.04,
                message=f"显存预算未启用（{e!r}），仍会尝试生成",
            )

        # attention 后端必须在建管线之前定好：processor 是在 pipeline 构造时
        # 绑上去的，之后再改环境变量对已经建好的模块无效。
        _attn = force_fast_attention(torch)
        write_status(
            status, status="processing", progress=0.04,
            message=f"attention 后端 {_attn}",
        )

        model_dir = Path(args.model)
        if not (model_dir / "model_index.json").is_file():
            raise RuntimeError(f"{model_dir} 不是 diffusers 模型目录（缺 model_index.json）")

        gguf = pick_gguf(model_dir, args.gguf)
        turbo = is_turbo(gguf)
        w, h = norm_size(args.size)
        fps = 24
        frames = frames_for(args.duration, fps)
        # 蒸馏权重只认 4 步 / CFG 1.0：跑 28 步不但慢 7 倍，还会因为超出训练分布
        # 把画面跑糊（这是少步数模型最常见的误用）。
        #
        # 非蒸馏的 Q4_K_M 反过来 —— **8 步根本不够**。实测 97 帧 / 8 步：
        # 第 49 帧人物就拖成油彩，末帧结构全毁（店铺变成抽象色块）。
        # 同一输入换 24 步，末帧人物、路面反光、店面结构全部正常。
        # 官方口径是 50 步；24 步是 12 GB 卡上「画质能交付」和「等得起」的折中。
        steps = args.steps or int(
            os.environ.get("YJ_VIDEO_STEPS", "4" if turbo else "24")
        )
        cfg_scale = float(
            os.environ.get("YJ_VIDEO_CFG", "1.0" if turbo else "5.0")
        )

        write_status(
            status, status="processing", progress=0.05,
            message=(
                f"加载 {gguf.name}（{gguf.stat().st_size/1e9:.2f} GB"
                f"{'，4 步蒸馏' if turbo else ''}）"
            ),
        )
        transformer = load_transformer(gguf, torch, model_dir)

        first = Path(args.image[0]) if args.image else None
        write_status(status, status="processing", progress=0.12,
                     message=f"加载管线（{'图生视频' if first else '文生视频'}）—— 首次约 3 分钟")
        pipe = build_pipe(model_dir, transformer, bool(first))
        apply_offload(pipe, torch)

        # 诊断：offload 到底有没有把权重搬走。
        #
        # 只看「配置了 group offload」不够 —— 参数如果还留在 cuda 上，hook 每次
        # onload 都是空操作，显存照样占满，而报错只说「差 80 MiB」，看起来像差一点。
        # 实测就是这样：改分辨率后占用一个字节都不变，说明占的就是权重本身。
        def _dev(mod):
            try:
                return str(next(mod.parameters()).device)
            except Exception:  # noqa: BLE001
                return "-"

        def _gb(fn):
            try:
                return f"{fn() / 1e9:.2f}GB"
            except Exception:  # noqa: BLE001
                return "?"

        print(
            "[offload] device  "
            f"transformer={_dev(pipe.transformer)} "
            f"text_encoder={_dev(pipe.text_encoder)} "
            f"vae={_dev(pipe.vae)}  "
            f"allocated={_gb(torch.cuda.memory_allocated)} "
            f"reserved={_gb(torch.cuda.memory_reserved)} "
            f"budget={_gb(torch.cuda.get_device_properties(0).total_memory)}",
            flush=True,
        )
        for attr, fn in (("vae", "enable_tiling"), ("vae", "enable_slicing")):
            try:
                getattr(getattr(pipe, attr), fn)()
            except Exception:
                pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        from PIL import Image  # noqa: PLC0415

        prompt = args.prompt.strip() or "自然流畅的镜头运动，真实摄影质感"

        def on_step(p, i, t, kw):
            # 0.15 → 0.88 的采样段
            try:
                total = len(getattr(p, "_num_timesteps", None) or range(steps))
            except Exception:
                total = steps
            write_status(status, status="processing",
                         progress=0.15 + 0.73 * (i + 1) / max(1, total),
                         message=f"采样 {i + 1}/{max(1, total)}")
            return kw

        base = {
            "prompt": prompt,
            "height": h,
            "width": w,
            "num_frames": frames,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "callback_on_step_end": on_step,
        }
        # 蒸馏权重没有 CFG：给负向提示词只会多跑一遍编码，结果一模一样
        if cfg_scale > 1.0:
            base["negative_prompt"] = "低质量，模糊，畸变，静止画面"

        if args.seed not in (None, ""):
            try:
                base["generator"] = torch.Generator("cpu").manual_seed(int(args.seed))
            except Exception:
                pass
        if first:
            base["image"] = Image.open(first).convert("RGB")
        if args.last_frame:
            base["last_image"] = Image.open(args.last_frame).convert("RGB")

        kw = call_kwargs(pipe, base)
        write_status(status, status="processing", progress=0.14,
                     message=f"生成 {w}×{h} / {frames} 帧 / {frames/fps:.1f}s")
        t0 = time.time()
        result = pipe(**kw)
        video = result.frames[0] if hasattr(result, "frames") else result[0]

        write_status(status, status="processing", progress=0.92, message="解码 / 写盘")
        save_video(video, dest, fps)
        if not dest.exists():
            raise RuntimeError("生成结束但没有产物文件")

        size_mb = round(dest.stat().st_size / 1048576, 2)
        write_status(
            status, status="succeeded", progress=1.0,
            url=f"http://127.0.0.1:{args.port}/files/{dest.name}",
            message=f"完成（{size_mb} MB / {w}x{h} / {frames/fps:.1f}s / {time.time()-t0:.0f}s / {gguf.name}）",
        )
    except Exception as e:  # noqa: BLE001
        print(traceback.format_exc()[-1800:], file=sys.stderr, flush=True)
        write_status(status, status="failed", progress=0.0,
                     message=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
