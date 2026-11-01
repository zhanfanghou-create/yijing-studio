#!/usr/bin/env python3
"""
视频生成 worker（由 model-server 起的子进程，跑完就退）

为什么单独一个进程（这一条从一开始就没变过）：
  - 一条 H3 视频 2-8 分钟，崩了不能把 API 服务带走；
  - 进程退出时显存 100% 还给驱动 —— 比在主进程里 del + empty_cache 干净得多；
  - 想换模型只换这个脚本，API 层不用动。

本版新增：**MiniMax H3-Base**（768p / 24fps / 原生立体声，音画一起出）。
它和 Wan 的区别不只是慢，是「一个模型干两件事」：
H3 自己出音频轨，所以出片后要拿 ffmpeg 把音视频合流；
Wan 是纯视频，配音得另外走 TTS。

四种后端（H3 的生态还在动，所以不把任何一条写死）：

  diffusers  diffusers 里有 MiniMaxH3 的 pipeline 就直接用（最省事）
  diffsynth  ModelScope 官方推荐的 NF4 方案，纯 Python，不需要 ComfyUI
  sglang     官方推荐的高性能路线，走它自己的 HTTP 服务
  cmd        兜底：给一条命令模板，占位符替换后执行（生态再变也不会完全没路）

进度只写一个 JSON 文件，不走管道 —— 管道一断全丢，文件不会。

    { "status": "queued|processing|succeeded|failed",
      "progress": 0.0~1.0,
      "url": "http://127.0.0.1:3911/files/xxx.mp4",
      "message": "..." }
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def write_status(path, **kw):
    """原子写：半截 JSON 会让轮询端读到脏数据，宁愿读旧值也不要读半截。"""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), "utf-8")
    tmp.replace(p)


def is_h3(model: str) -> bool:
    m = model.lower()
    return "h3" in m or "minimax" in m


def norm_size(size: str, default=(1344, 768)):
    if not size:
        return default
    try:
        w, h = [int(float(x)) for x in size.lower().replace("*", "x").split("x")]
    except Exception:
        return default
    # H3 的 patch 是 16×16；Wan 也是 16 的倍数。统一向下取整，别让最后一层卷积报错。
    return (max(256, w - w % 16), max(256, h - h % 16))


# ---------------------------------------------------------------- H3

def h3_diffusers(args, w, h, frames, status, tick):
    """
    diffusers 路线。

    实测（2026-10，diffusers 0.41.0 + transformers 5.20）：H3 **不是**传统的
    `MiniMaxH3Pipeline` 类，而是新的 **modular pipeline** ——
    `diffusers.modular_pipelines.minimax_h3` 里是 `MiniMaxH3ModularPipeline`，
    模型仓库给的 model_index.json 里 `_class_name` 写的还是老名字，直接
    `DiffusionPipeline.from_pretrained` 会报「找不到类」。

    正确用法三步：from_pretrained 读配置 → load_components 加载权重 → 调用。
    """
    import torch
    from diffusers import ModularPipeline  # noqa: PLC0415

    write_status(status, status="processing", progress=0.05, message="加载 H3 配置")
    pipe = ModularPipeline.from_pretrained(args.model)

    write_status(status, status="processing", progress=0.10,
                 message="加载 H3 权重（首次 3-8 分钟，之后走缓存）")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        pipe.load_components(torch_dtype=dtype)
    except TypeError:
        # 老一点的签名不认 torch_dtype，退回位置参数
        pipe.load_components(dtype)

    if os.environ.get("YJ_VIDEO_OFFLOAD", "1") != "0":
        # 48G 卡也建议开：主干 + 文本编码器全量常驻会顶到边
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            try:
                pipe.to("cuda")
            except Exception:
                pass
    else:
        pipe.to("cuda")

    kw = dict(
        prompt=args.prompt,
        num_frames=frames,
        width=w,
        height=h,
        num_inference_steps=args.steps or 50,
    )
    if args.image:
        kw["image"] = args.image[0]
    # output 决定回包里带哪些产物；H3 的卖点是音画一起出，两个都要
    kw["output"] = ["videos", "audios"]

    write_status(status, status="processing", progress=0.15,
                 message=f"开始采样（{frames} 帧 {w}x{h}，{kw['num_inference_steps']} 步）")

    try:
        out = pipe(**kw)
    except TypeError:
        # 不同分支对 output 的取值名不一样，去掉再试一次
        kw.pop("output", None)
        out = pipe(**kw)

    frames_out = getattr(out, "videos", None) or getattr(out, "frames", None)
    audio = getattr(out, "audios", None) or getattr(out, "audio", None)
    if frames_out is None and isinstance(out, dict):
        frames_out = out.get("videos") or out.get("frames")
        audio = audio or out.get("audios")
    return (frames_out[0] if isinstance(frames_out, (list, tuple)) else frames_out), audio


# NF4 包里真正要用到的四件。顺序照官方 `MiniMax-H3-NF4-FL2VA.py` 示例，
# 少一件都加载不起来，所以缺哪个就在报错里点名。
DS_NF4_FILES = (
    "minimax-h3-fl2va-nf4.safetensors",
    "minimax-h3-text-encoder-nf4.safetensors",
    "video_vae_nf4.safetensors",
    "audio_vae_nf4.safetensors",
)


def h3_diffsynth(args, w, h, status, tick):
    """
    DiffSynth 路线（ModelScope 官方的 NF4 配方）。

    **为什么不能走 diffusers**：NF4 包是 DiffSynth 的单文件格式 —— 四个
    `*_nf4.safetensors` 加一个 `configuration.json`，**没有 `model_index.json`**，
    也没有按组件分目录。`DiffusionPipeline.from_pretrained` 认的是那套目录结构，
    所以「34 GB 下完了、进度 100%、加载还是失败」的根因不在权重，在运行时。

    API 取 2026-11 上游 main 分支（`diffsynth.pipelines.minimax_h3_audio_video`）：
    PyPI 上的 2.2.0 里没有这个模块，必须 `pip install --no-deps -e <源码目录>`。

    首尾帧在这里是真支持的：官方 fl2va 示例就是
    `keyframes=[first, last], keyframe_indices=[0, -1]`。
    """
    try:
        from diffsynth.pipelines.minimax_h3_audio_video import (  # noqa: PLC0415
            MiniMaxH3Pipeline,
            ModelConfig,
        )
        from diffsynth.utils.data.audio_video import write_video_audio  # noqa: PLC0415
    except Exception as e:
        raise RuntimeError(
            f"没有 DiffSynth 的 H3 管线（{e}）。PyPI 的 2.2.0 缺这个模块，装法："
            " curl -sL https://codeload.github.com/modelscope/DiffSynth-Studio/tar.gz/refs/heads/main"
            " | tar xz -C /root/ds-src --strip-components=1 && "
            ".venv/bin/pip install --no-deps -e /root/ds-src"
        )

    import torch
    from PIL import Image

    root = Path(args.model)
    missing = [n for n in DS_NF4_FILES if not (root / n).is_file()]
    if missing:
        raise RuntimeError(
            "NF4 权重不齐，缺：" + "、".join(missing)
            + "（在「模型配置」里把视频模型下完，或 bash deploy/bootstrap.sh models video）"
        )

    # 低显存档位：权重放磁盘、用时再搬。48 G 的卡也建议这么配 ——
    # 主干 + 文本编码器全量常驻会顶到边，和语音抢显存时直接 OOM。
    vram = {
        "offload_dtype": "disk", "offload_device": "disk",
        "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    }
    configs = [ModelConfig(path=str(root / n), skip_download=True, **vram) for n in DS_NF4_FILES]

    # processor（tokenizer + 视频预处理）在包的 `processor/` 子目录里；
    # 没有的话回落到仓库 id —— 但它只有几 MB，下载器早就一起拉下来了。
    proc_dir = root / "processor"
    processor = (
        ModelConfig(path=str(proc_dir), skip_download=True)
        if (proc_dir / "preprocessor_config.json").is_file()
        else ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/")
    )

    write_status(status, status="processing", progress=0.08,
                 message="加载 H3-NF4（首次 4-10 分钟，之后常驻）")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=configs,
        processor_config=processor,
        vram_limit=max(6.0, torch.cuda.mem_get_info("cuda")[1] / 1024 ** 3 - 4),
    )

    # H3 的帧数不是任意值：画布几何按 `17n + 5` 排，取最近的一档，
    # 否则会得到「时长对了但最后一帧被截」这种看不出来的偏差。
    want = max(24, int(round(args.duration * 24)))
    n = max(1, int(round((want - 5) / 17)))
    frames = 17 * n + 5

    def _bar(it):
        """替掉 tqdm：把步数写进任务状态里，前端才有真实进度条。

        `MiniMaxH3Pipeline.__call__` 没有 `progress_callback` 这个形参，
        只认 `progress_bar_cmd` —— 传错名字会直接 TypeError，
        所以这里做成「迭代器包装」而不是回调。
        """
        items = list(it)
        total = max(1, len(items))
        for i, x in enumerate(items):
            write_status(status, status="processing",
                         progress=0.15 + 0.7 * (i + 1) / total,
                         message=f"采样 {i + 1}/{total}")
            yield x

    kw = dict(
        prompt=args.prompt,
        height=h,
        width=w,
        num_frames=frames,
        num_inference_steps=args.steps or 50,
        cfg_scale=float(os.environ.get("YJ_H3_CFG", "1.0")),
        progress_bar_cmd=_bar,
        seed=int(args.seed) if args.seed not in (None, "") else 0,
    )

    # 关键帧：有尾帧 → fl2va（首尾过渡）；只有首帧 → 首帧条件；都没有 → t2va
    keyframes = [Image.open(args.image[0]).convert("RGB")] if args.image else []
    if args.last_frame:
        keyframes.append(Image.open(args.last_frame).convert("RGB"))
        kw["keyframes"] = keyframes
        kw["keyframe_indices"] = [0, -1]
        write_status(status, status="processing", progress=0.12,
                     message=f"首尾帧模式（{frames} 帧 {w}x{h}）")
    elif keyframes:
        kw["keyframes"] = keyframes
        kw["keyframe_indices"] = [0]
        write_status(status, status="processing", progress=0.12,
                     message=f"首帧模式（{frames} 帧 {w}x{h}）")
    else:
        write_status(status, status="processing", progress=0.12,
                     message=f"文生视频（{frames} 帧 {w}x{h}）")

    video, audio = pipe(**kw)

    # 写成 raw_ 前缀：main 那边会再拷成 tmp_ 并落到 `vid_<job>.mp4`，
    # 直接写最终名会让它自己拷自己（SameFileError）。
    dest = Path(args.out) / f"raw_{args.job}.mp4"
    write_video_audio(video=video, audio=audio, output_path=str(dest), fps=24,
                      audio_sample_rate=32000)
    # 已经合流好了：返回路径，让 main 走「拷贝到 dest」那条分支，不再二次处理
    return dest, None


def h3_sglang(args, w, h, status):
    """
    SGLang 路线：官方 cookbook 里的配方最快，但它自己管显存，
    所以要求先 `sglang serve --model-path MiniMaxAI/MiniMax-H3` 起好。
    接口路径随版本变，用 YJ_H3_PATH 覆盖（默认 /v1/videos/generations）。
    """
    import urllib.request

    base = os.environ.get("YJ_H3_ENDPOINT", "http://127.0.0.1:30010").rstrip("/")
    path = os.environ.get("YJ_H3_PATH", "/v1/videos/generations")
    body = {
        "prompt": args.prompt, "width": w, "height": h,
        "num_frames": max(24, int(args.duration * 24)),
        "num_inference_steps": args.steps or 50,
    }
    if args.image:
        import base64

        body["image"] = "data:image/png;base64," + base64.b64encode(
            Path(args.image[0]).read_bytes()
        ).decode()

    write_status(status, status="processing", progress=0.15, message="SGLang 生成中")
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=int(os.environ.get("YJ_H3_TIMEOUT", "7200"))) as r:
        info = json.loads(r.read())
    url = info.get("url") or (info.get("data") or [{}])[0].get("url")
    if not url:
        raise RuntimeError(f"SGLang 没给结果地址：{str(info)[:300]}")
    return url, None


def h3_cmd(args, w, h, out_dir, status):
    """
    cmd 兜底：生态再变也有路。
    YJ_H3_CMD 里用 {prompt} {image} {out} {width} {height} {frames} {steps} 占位。
    """
    tpl = os.environ.get("YJ_H3_CMD", "").strip()
    if not tpl:
        raise RuntimeError(
            "YJ_VIDEO_BACKEND=cmd 但没有配 YJ_H3_CMD。示例：\n"
            "  export YJ_H3_CMD='python generate.py --prompt {prompt} --width {width} "
            "--height {height} --frames {frames} --output {out}'"
        )
    out = out_dir / f"vid_cmd_{args.job}.mp4"
    cmd = tpl.format(
        prompt=args.prompt.replace('"', "'"),
        image=(args.image[0] if args.image else ""),
        out=str(out), width=w, height=h,
        frames=max(24, int(args.duration * 24)), steps=args.steps or 50,
    )
    write_status(status, status="processing", progress=0.15, message="执行自定义命令")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"命令失败（exit={proc.returncode}）：{(proc.stderr or '')[-400:]}")
    return out


# ---------------------------------------------------------------- 合流

def mux_audio(video_path: Path, audio, out_path: Path):
    """
    H3 的原生音频是它自己出的 —— 这是它相对 Wan 最大的价值（不用另外配音）。
    但它给的是 numpy/张量，得先落成 wav 再和视频合流，否则画布那边拿到的是无声片。
    """
    if audio is None:
        return video_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError("有音频轨但没有 ffmpeg，装一下：apt-get install -y ffmpeg")

    wav = out_path.with_suffix(".wav")
    try:
        import numpy as np
        import soundfile as sf  # noqa: PLC0415

        a = audio
        if hasattr(a, "cpu"):
            a = a.cpu().numpy()
        a = np.asarray(a, dtype="float32")
        if a.ndim == 2 and a.shape[0] in (1, 2) and a.shape[0] < a.shape[1]:
            a = a.T  # (2, N) → (N, 2)
        sf.write(str(wav), a, int(os.environ.get("YJ_H3_SR", "24000")))
    except Exception as e:
        raise RuntimeError(f"音频轨写盘失败（{type(e).__name__}: {e}）") from e

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path), "-i", str(wav),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 合流失败：{proc.stderr[-300:]}")
    wav.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", default="3911")
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--size", default="")
    ap.add_argument("--image", action="append", default=None)
    # 尾帧只收不用：H3 的两条本机路线（diffusers / DiffSynth）都只接一张条件图。
    # 收下参数是为了「收到但用不上」能被记录下来，而不是当场 Exit 2 让人以为是参数写错了。
    ap.add_argument("--last-frame", default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = args.status

    try:
        h3 = is_h3(args.model)
        w, h = norm_size(args.size, (1344, 768) if h3 else (1280, 704))
        # H3 是 24fps 原生；Wan 是 16fps。帧数算错会得到「时长不对」，不会报错。
        fps = 24 if h3 else 16
        frames = max(24 if h3 else 17, int(round(args.duration * fps / 4)) * 4 + 1)

        write_status(status, status="processing", progress=0.02,
                     message=f"{Path(args.model).name} · {w}x{h} · {args.duration:g}s"
                             f"{' · 尾帧未生效' if args.last_frame else ''}")

        def tick(pipeline=None, step=0, timestep=None, kwargs=None):
            total = args.steps or (50 if h3 else 30)
            write_status(status, status="processing",
                         progress=0.1 + 0.85 * (step + 1) / max(1, total),
                         message=f"采样 {step + 1}/{total}")
            return kwargs if kwargs is not None else {}

        if args.seed not in (None, ""):
            try:
                import torch

                torch.manual_seed(int(args.seed))
            except Exception:
                pass

        backend = args.backend
        if backend == "auto":
            backend = "diffusers"

        dest = out_dir / f"vid_{args.job}.mp4"

        if not h3:
            # 这里以前会退到 Wan 出片（小显存机器的降级档）。产品上明确只要 H3，
            # 所以不再降级：「悄悄换个模型给你出片」比「直说这个模型不支持」有害得多 ——
            # 用户会以为自己用的是 H3，实际拿到的是 5B 的效果。
            raise RuntimeError(
                f"视频只支持 MiniMax H3，收到 {args.model!r}。其他视频后端已从这条链路移除。"
            )
        elif backend == "sglang":
            url, _audio = h3_sglang(args, w, h, status)
            import urllib.request

            write_status(status, status="processing", progress=0.9, message="下载结果")
            urllib.request.urlretrieve(url, dest)
        elif backend == "cmd":
            dest = h3_cmd(args, w, h, out_dir, status)
        else:
            # 两条路的取舍：diffusers 的 API 最稳，DiffSynth 的 NF4 最省显存。
            # 先试 diffusers，不行再退 DiffSynth；两个错都留着一起报，
            # 否则用户要来回改配置试三轮才知道到底缺什么。
            if backend == "diffsynth":
                tries = [("diffsynth", lambda: h3_diffsynth(args, w, h, status, tick))]
            else:
                # 目录名带 nf4 就是 DiffSynth 的量化包（扁平结构，没有 model_index.json）。
                # 那种目录 diffusers 的 from_pretrained 注定读不了，先试它只是白等一轮报错，
                # 所以按「这份权重是什么形态」决定顺序，而不是固定先 diffusers。
                _d = ("diffsynth", lambda: h3_diffsynth(args, w, h, status, tick))
                _f = ("diffusers", lambda: h3_diffusers(args, w, h, frames, status, tick))
                tries = [_d, _f] if "nf4" in str(args.model).lower() else [_f, _d]

            video_audio, errors = None, []
            for name, run in tries:
                try:
                    video_audio = run()
                    break
                except Exception as e:
                    errors.append(f"  · {name}: {type(e).__name__}: {e}")
            if video_audio is None:
                raise RuntimeError("H3 后端都不可用：\n" + "\n".join(errors))

            video, audio = video_audio
            write_status(status, status="processing", progress=0.92, message="写盘 / 合流")
            tmp_v = out_dir / f"tmp_{args.job}.mp4"
            if hasattr(video, "save"):
                try:
                    video.save(str(tmp_v))
                except Exception:
                    from diffusers.utils import export_to_video  # noqa: PLC0415

                    export_to_video(video, str(tmp_v), fps=fps)
            elif isinstance(video, (str, Path)):
                shutil.copy(str(video), str(tmp_v))
            else:
                from diffusers.utils import export_to_video  # noqa: PLC0415

                export_to_video(video, str(tmp_v), fps=fps)

            if audio is not None:
                mux_audio(tmp_v, audio, dest)
                tmp_v.unlink(missing_ok=True)
            else:
                tmp_v.replace(dest)

        if not dest.exists():
            raise RuntimeError("生成结束但没有产物文件")

        size_mb = round(dest.stat().st_size / 1048576, 2)
        write_status(status, status="succeeded", progress=1.0,
                     url=f"http://127.0.0.1:{args.port}/files/{dest.name}",
                     message=f"完成（{size_mb} MB / {w}x{h} / {args.duration:g}s"
                             f"{' / 原生音频' if h3 else ''}）")

    except Exception as e:
        tb = traceback.format_exc()[-1500:]
        print(tb, file=sys.stderr, flush=True)
        write_status(status, status="failed", progress=0.0,
                     message=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
