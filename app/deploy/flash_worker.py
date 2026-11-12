#!/usr/bin/env python3
"""
HyperFlow H3 Flash worker（云端/大显存档 · 音画同出）

和 `video_worker.py`（H3-NF4 / DiffSynth）、`wan_worker.py`（Wan2.2 低显存档）
是**同一份契约**：命令行参数、状态 JSON、产物落盘位置全一样，由 model-server
按后端选择器三选一。换后端不动 API 层与前端。

为什么单独一个 worker
---------------------
HyperFlow H3 Flash 是 MiniMax-H3 的**压缩少步版**（13.7B / 官方约 31B）：
  · 主干只有原版一半不到，INT8 convrot 16 GB（GGUF 另有一路，走 ComfyUI）；
  · 4 / 6 / 8 步、无 CFG —— 默认 6 步，OOM 风险与耗时都按步数线性下降；
  · 一次前向出「视频 + 32 kHz 立体声」，不需要再跑 TTS 或音效；
  · 文本编码器、视频 VAE、音频 VAE 都复用 MiniMax-H3 上游仓库，只有主干是本仓。
三条路线（DiffSynth NF4 / Wan2.2 GGUF / HyperFlow）的帧规则都不一样：
  H3-NF4     17n+5 之外还受 DiffSynth 的 latent 切片约束
  Wan2.2     4n+1，上限 121 帧
  HyperFlow  17n+5，124–362 帧（5.2–15 s @24 fps）
所以硬塞进一个 worker 只会变成 if 森林，两边都难维护。

用法（model-server 会这么调）：
    python flash_worker.py --job xx --status xx.json --out DIR --port 3911 \
        --model /root/models/hyperflow-h3-flash \
        --base-model /root/models/MiniMax-H3 \
        --prompt "..." --size 1344x768 --duration 5.2 \
        [--image first.png] [--last-frame last.png] [--seed 42] [--steps 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

FPS = 24
# 帧规则：17n + 5。124 帧 = 5.17 s，362 帧 = 15.08 s（上游 `admit_max_frames` 认这个上限）。
MIN_FRAMES, MAX_FRAMES = 124, 362
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344


def write_status(path, **kw):
    """原子写：半截 JSON 会让轮询端读到脏数据。"""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), "utf-8")
    tmp.replace(p)


def align_frames(n: int) -> int:
    """向上取到最近的 `17n + 5`。"""
    n = max(MIN_FRAMES, min(MAX_FRAMES, int(n)))
    return n + (5 - n) % 17


def frames_for(duration: float) -> int:
    """时长 → 帧数。少于 5.2 s 一律给 124 帧：这是这代的训练下限。"""
    want = int(round(max(0.5, float(duration)) * FPS))
    return align_frames(want)


def canvas_for(size: str, default=(16, 9)) -> tuple[int, int]:
    """按请求的宽高比，落到 HyperFlow 的画布约束上（短边 768 / 32 对齐 / 上限 768×1344）。

    上游 `resolve_canvas()` 只吃宽高比；这里多一层：客户端传的是 `1344x768`
    这种具体像素，直接当画布用会撞上「不是 32 的倍数」或「超像素上限」。
    返回 `(height, width)` —— 和 diffusers 的入参顺序一致，别写反。
    """
    try:
        w, h = [float(x) for x in str(size).lower().replace("*", "x").split("x")]
        if w <= 0 or h <= 0:
            raise ValueError
        ratio = w / h
    except Exception:
        ratio = default[0] / default[1]
    short, max_px = CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS
    if ratio >= 1:
        width, height = short * ratio, short
    else:
        width, height = short, short / ratio
    if width * height > max_px:
        k = (max_px / (width * height)) ** 0.5
        width, height = width * k, height * k
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def pick_weights(model_dir: Path, explicit: str | None) -> tuple[Path, str]:
    """返回 (权重根目录, 分区子目录)。分区名带 `fl2va` 的优先。"""
    root = Path(explicit) if explicit else model_dir
    if (root / "config.json").is_file():
        return root, ""
    for sub in ("fl2va_int8_convrot", "fl2va_int8_convrot_comfyui", "fl2va_gguf"):
        if (root / sub / "config.json").is_file():
            return root, sub
    # 有的转存把 config.json 放在根，但权重在子目录里
    for sub in ("fl2va_int8_convrot", "fl2va_int8_convrot_comfyui"):
        if (root / sub).is_dir():
            return root, sub
    raise RuntimeError(
        f"{root} 里没有找到 fl2va 分区（要 fl2va_int8_convrot/config.json）。"
        "HyperFlow H3 Flash 仓库只含压缩后的主干，其余组件在 MiniMax-H3 里"
    )


def pick_base(base: str | None) -> str:
    """MiniMax-H3 组件目录：只要文本编码器 / VAE / tokenizer / processor 在就算齐。"""
    cand = Path(base) if base else None
    if cand and (cand / "modular_model_index.json").is_file():
        return str(cand)
    for name in ("MiniMax-H3", "MiniMax-H3-NF4"):
        for root in (os.environ.get("YJ_LOCAL_MODELS", "/root/models"),
                     os.environ.get("YJ_CLOUD_MODELS", "/root/cloud/models")):
            d = Path(root) / name
            if (d / "modular_model_index.json").is_file():
                return str(d)
    if cand and cand.is_dir():
        return str(cand)
    raise RuntimeError(
        "找不到 MiniMax-H3 组件目录（需要 modular_model_index.json + text_encoder / vae / audio_vae）"
    )


def plan_offload(want: str, vram_gb: float) -> str:
    """offload 档位。

    `auto` 交给上游会按显存门槛判定（none≥90GiB / model≥60GiB / stream 其余）。
    本机 48 GB 卡会落到 stream —— 能用但慢。这里把 `auto` 显式化并在日志里写明，
    因为「为什么这么慢」和「为什么没吃满显存」是两件事，得能一眼看到原因。
    """
    want = (want or "auto").strip().lower()
    if want in ("none", "model", "stream"):
        return want
    if vram_gb >= 90:
        return "none"
    if vram_gb >= 60:
        return "model"
    return "stream"


class Ticker(threading.Thread):
    """上游管线没有稳定的 step 回调（模块化管线每版签名都在动），

    所以进度用「时间轴估算 + 单调递增」来报：至少让前端知道还在动、
    以及大概卡在哪个阶段。真正的阶段切换点由主流程打点覆盖。
    """

    def __init__(self, status, lo: float, hi: float, seconds: float, label: str):
        super().__init__(daemon=True)
        self.status, self.lo, self.hi = status, lo, hi
        self.seconds, self.label = max(1.0, seconds), label
        self._stop = threading.Event()
        self._at = lo

    def run(self):
        t0 = time.time()
        while not self._stop.wait(2.0):
            frac = min(0.97, (time.time() - t0) / self.seconds)
            self._at = self.lo + (self.hi - self.lo) * frac
            write_status(self.status, status="processing", progress=round(self._at, 3),
                         message=self.label)

    def stop(self, progress: float | None = None):
        self._stop.set()
        return self._at if progress is None else progress


def save_video(result, dest: Path) -> str:
    """写 mp4。优先带音轨（HyperFlow 原生出 32 kHz 立体声），失败再退回纯画面。"""
    from diffusers.utils.export_utils import encode_video

    frames = result["videos"][0]
    audio = result.get("audio")
    rate = result.get("sampling_rate")
    if audio is not None and rate:
        try:
            encode_video(frames, fps=FPS, output_path=str(dest),
                         audio=audio[0], audio_sample_rate=rate)
            return "video+audio"
        except Exception as e:  # noqa: BLE001
            print(f"[flash] 带音轨封装失败（{e!r}），退回纯画面", file=sys.stderr, flush=True)
    encode_video(frames, fps=FPS, output_path=str(dest))
    return "video"


def patch_triton_scope() -> str:
    """让上游的 Triton kernel 在本机 triton 版本上能编译。

    上游把 `@triton.jit` kernel 定义在 `_kernels()` **函数内部**，`tl` 是那一层的
    局部变量。triton 3.2（torch 2.6 自带的那版）在 `ast_to_ttir` 里只查函数的
    `__globals__`，闭包变量不在里面 —— 于是首次采样就炸：

        triton.compiler.errors.CompilationError: NameError('tl is not defined')

    这条不是「模型装错了」，也不好排查（栈里全是 triton 内部帧）。修法是把
    `tl` / `triton` 注入 int8 模块的 globals：kernel 的 `__globals__` 就是模块的
    `__dict__`，注入后就能解析到。**不改上游源码**，升级包也不会被覆盖掉。

    返回版本串，写进状态文件，方便出问题时一眼看到跑的是哪版 triton。
    """
    try:
        import triton  # noqa: PLC0415
        import triton.language as tl  # noqa: PLC0415
        from hyperflow_h3_flash import int8 as _i8  # noqa: PLC0415

        if getattr(_i8, "tl", None) is None:
            _i8.tl = tl  # type: ignore[attr-defined]
        if getattr(_i8, "triton", None) is None:
            _i8.triton = triton  # type: ignore[attr-defined]
        return f"triton {triton.__version__}"
    except Exception as e:  # noqa: BLE001
        return f"triton shim 跳过（{e!r}）"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=3911)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--duration", type=float, default=5.2)
    ap.add_argument("--size", default="1344x768")
    ap.add_argument("--backend", default="hyperflow")
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--last-frame", default=None)
    ap.add_argument("--offload", default="auto")
    ap.add_argument("--text-encoder", default="bf16")
    args = ap.parse_args()

    status = args.status
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"video_{args.job}.mp4"
    write_status(status, status="processing", progress=0.02,
                 message="准备加载 HyperFlow H3 Flash")

    tick = None
    try:
        import torch  # noqa: PLC0415

        # 只限线程数，**不设显存硬上限**。
        #
        # `vram_guard.cap_vram()` 是给「一次把所有权重铺上卡」的档位用的
        # （`set_per_process_memory_fraction` 直接砍 allocator 池）。HyperFlow 的
        # `stream` offload 恰恰相反 —— 它按 block 换进换出，某一刻的"池"峰值远大于
        # 稳态占用；那条线会把本来能跑的任务判成 OOM。真要限，用
        # `YJ_VRAM_BUDGET_GB` 显式给，别让默认值替用户做决定。
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from vram_guard import cap_threads  # noqa: PLC0415

            n = cap_threads(lambda m: write_status(status, status="processing",
                                                   progress=0.03, message=m), "flash")
            if n:
                write_status(status, status="processing", progress=0.03,
                             message=f"torch 线程 {n}（显存交给 stream offload 自己换页）")
        except Exception as e:  # noqa: BLE001
            write_status(status, status="processing", progress=0.03,
                         message=f"线程设置未生效（{e!r}）")

        weights, sub = pick_weights(Path(args.model), None)
        base = pick_base(args.base_model)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        offload = plan_offload(args.offload, vram_gb)
        steps = int(args.steps or os.environ.get("YJ_FLASH_STEPS", "6"))
        if steps not in (4, 6, 8):
            # 上游只在 4/6/8 三档上训过（两段条件的时间表按步数定），
            # 传 12 步不会更清晰，只会把训练分布外推成噪点。
            steps = min((4, 6, 8), key=lambda s: abs(s - steps))
        height, width = canvas_for(args.size)
        frames = frames_for(args.duration)
        # 必须在建管线之前：kernel 的 globals 要在第一次编译前就有 tl。
        _tri = patch_triton_scope()

        write_status(
            status, status="processing", progress=0.04,
            message=(f"加载主干 {sub or weights.name} / offload={offload} / "
                     f"{width}×{height} / {frames} 帧 / {steps} 步 / {_tri}"),
        )

        from hyperflow_h3_flash import load_pipeline  # noqa: PLC0415

        workflow = "fl2va" if args.image else "t2va"
        t0 = time.time()
        tick = Ticker(status, 0.05, 0.55, 240,
                      "加载管线（首次要读 60 GB 级文本编码器，约 3–6 分钟）")
        tick.start()
        pipe = load_pipeline(
            workflow,
            weights=str(weights),
            subfolder=sub or None,
            base_model=base,
            device="cuda",
            offload=offload,
            text_encoder_precision=args.text_encoder,
        )
        tick.stop()
        loaded = time.time() - t0

        from diffusers.utils import load_image  # noqa: PLC0415

        inputs = {
            "prompt": args.prompt.strip() or "自然流畅的镜头运动，真实摄影质感",
            "num_frames": frames,
            "num_inference_steps": steps,
            "height": height,
            "width": width,
        }
        if args.seed not in (None, ""):
            inputs["generator"] = torch.Generator("cpu").manual_seed(int(args.seed))
        if args.image:
            inputs["image"] = load_image(str(args.image[0]))
            if args.last_frame:
                inputs["last_image"] = load_image(str(args.last_frame))

        # 采样耗时参考：4090 + INT8 6 步 / 5.2 s 片段约 157 s（官方 GGUF 数据），
        # stream offload 会再慢一截，这里给 360 s 的估算窗，只影响进度条不影响结果。
        tick = Ticker(status, 0.56, 0.9, 360, f"采样 {steps} 步 / {frames} 帧 / {width}×{height}")
        tick.start()
        out = pipe(**inputs, output=["videos", "audio", "sampling_rate"])
        tick.stop()

        write_status(status, status="processing", progress=0.92, message="封装 mp4（h264 + aac）")
        kind = save_video(out, dest)
        if not dest.exists():
            raise RuntimeError("生成结束但没有产物文件")

        size_mb = round(dest.stat().st_size / 1048576, 2)
        peak = torch.cuda.max_memory_allocated() / 2**30
        write_status(
            status, status="succeeded", progress=1.0,
            url=f"http://127.0.0.1:{args.port}/files/{dest.name}",
            message=(f"完成（{size_mb} MB / {width}x{height} / {frames / FPS:.2f}s / "
                     f"{kind} / {steps} 步 / 加载 {loaded:.0f}s / 峰值 {peak:.1f} GiB）"),
        )
    except Exception as e:  # noqa: BLE001
        if tick is not None:
            try:
                tick.stop()
            except Exception:
                pass
        print(traceback.format_exc()[-1800:], file=sys.stderr, flush=True)
        write_status(status, status="failed", progress=0.0,
                     message=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
