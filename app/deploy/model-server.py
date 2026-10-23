#!/usr/bin/env python3
"""
艺镜 Studio · 本机模型服务（model-server）· 2026-10 版

把「文本 / 图片 / 视频 / 语音」四模态包成一组 OpenAI 兼容的 HTTP 接口，
让画布的通道层（electron/ipc/ai.ts）不需要知道自己连的是云厂商还是自己的卡。

本版相较 FLUX + Wan 那版的三个变化：

  1. **图片换成 Qwen-Image-2.1**（7B 单流 DiT，文生图 + 编辑一体，原生 RGBA 透明，
     最多 10 张参考图，原生 2K）。它比 FLUX 多出来的正好是画布需要的：
     透明素材、多图参考、局部编辑。代价是文本编码器很重（Qwen3-VL-8B，17.5 GB），
     所以必须开 cpu offload —— 编码完就把它挪回内存，这是 48G 卡上能同时留住语音的关键。
  2. **视频换成 MiniMax H3-Base**（768p / 24fps / 原生立体声，音画一起出）。
     它有 33B transformer + 32B 文本编码器，是这台机器上唯一「必须独占」的模态。
  3. **显存从「谁用谁清」升级为「租约」**。旧版是 evict（要装新的就把旧的扔掉），
     在 H3 这种 33 GB 的家伙面前不够用：它需要语音主动让位。所以分成两档：
       · 共享档 image   与语音共存（靠 offload 把峰值压到 ~22 GB）
       · 独占档 video   提交前调语音的 /unload，跑完语音按需重载

接口：
    GET  /healthz                  各模态状态 / 显存 / 当前租约 / 队列
    GET  /v1/models                给界面列模型
    POST /v1/images/generations    文生图 / 图生图 / 多图编辑（Qwen-Image-2.1）
    POST /v1/videos                提交视频任务 → {id}
    GET  /v1/videos/{id}           查进度；成功后给 url
    POST /v1/chat/completions      透传给 vLLM / llama.cpp（它们是 OpenAI 协议）
    POST /v1/audio/speech          透传给语音引擎（vendor/voice）
    GET  /files/<name>             产物直链

环境变量：
    YJ_PORT            默认 3911
    YJ_MODEL_DATA      产物与状态目录，默认 ./model-data
    YJ_TEXT_UPSTREAM   文本后端，默认 http://127.0.0.1:8001/v1（vLLM）
    YJ_VOICE_BASE      语音引擎，默认 http://127.0.0.1:3910
    YJ_IMAGE_MODEL     默认 Qwen/Qwen-Image-2.1
    YJ_IMAGE_STEPS     默认 28（官方标准 40 步，28 步在 2K 下质量已很难分辨）
    YJ_IMAGE_MAX_SIDE  默认 1536；设 2752 才出原生 2K（慢 3-4 倍）
    YJ_IMAGE_OFFLOAD   默认 1（CPU offload 换显存，必须开，否则和语音抢不过来）
    YJ_VIDEO_MODEL     默认 MiniMaxAI/MiniMax-H3
    YJ_VIDEO_FAST_MODEL 默认 Wan-AI/Wan2.2-TI2V-5B（请求里 model=fast 时用）
    YJ_VIDEO_BACKEND   auto | diffusers | diffsynth | sglang | cmd
    YJ_MODE            full | image | text（只开某几个模态，省显存）
"""

import base64
import gc
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("YJ_PORT", "3911"))
DATA = Path(os.environ.get("YJ_MODEL_DATA", ROOT / "model-data")).resolve()
OUT = DATA / "out"
STATUS = DATA / "jobs"
TEXT_UPSTREAM = os.environ.get("YJ_TEXT_UPSTREAM", "http://127.0.0.1:8001/v1").rstrip("/")
VOICE_BASE = os.environ.get("YJ_VOICE_BASE", "http://127.0.0.1:3910").rstrip("/")
IMAGE_MODEL = os.environ.get("YJ_IMAGE_MODEL", "Qwen/Qwen-Image-2.1")
VIDEO_MODEL = os.environ.get("YJ_VIDEO_MODEL", "MiniMaxAI/MiniMax-H3")
VIDEO_FAST = os.environ.get("YJ_VIDEO_FAST_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
# 「没指定 model 时用哪个」和「H3 是哪个」是两件事，必须能分开配。
# 实测：H3-Base 光权重就 ~125 GB bf16（DiT 62 GB + Qwen3-VL 文本编码器 63 GB），
# 4090D 的 48 GB 显存必须全量 CPU offload，120 GB 内存也刚好卡在边缘；
# 而且它的 model_index 是老格式，还要求 transformers 里有 MiniMaxH3Qwen3VLHFEncoder。
# 所以小机器上把默认档指向能跑的那个（Wan 5B），H3 用 model=h3 显式点名。
VIDEO_DEFAULT = os.environ.get("YJ_VIDEO_DEFAULT", VIDEO_MODEL)
VIDEO_BACKEND = os.environ.get("YJ_VIDEO_BACKEND", "auto")
IMAGE_STEPS = int(os.environ.get("YJ_IMAGE_STEPS", "28"))
IMAGE_MAX_SIDE = int(os.environ.get("YJ_IMAGE_MAX_SIDE", "1536"))
IMAGE_OFFLOAD = os.environ.get("YJ_IMAGE_OFFLOAD", "1") != "0"
MODE = os.environ.get("YJ_MODE", "full")

for d in (OUT, STATUS):
    d.mkdir(parents=True, exist_ok=True)

log_lock = threading.Lock()


def log(*args):
    with log_lock:
        print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


# ============================================================== 显存租约

class Lease:
    """
    单卡上最贵的错误不是选错模型，是**两个模态同时以为自己有地方住**。

    分两档：

      image  —— 共享档。Qwen-Image-2.1 开 cpu offload 后峰值 ~22 GB，
                和语音（1.7B 三件套按需单载，~8 GB）+ 文本（~7 GB）能一起活着。
      video  —— 独占档。H3 的 33B 主干 + 32B 文本编码器，量化后仍要 30 GB 以上。
                提交前必须让语音把权重还回去（POST /unload），跑完再让它按需重载。

    为什么不做成「自动 LRU」：因为卸载语音要跨进程，猜错一次就是静默 OOM。
    显式租约的代价只是「生成视频时不能同时配音」——这本来就是单卡的物理事实。
    """

    def __init__(self):
        self.holder = None          # None | "image" | "video"
        self.lock = threading.Lock()
        self.image_pipe = {"key": None, "pipe": None, "kind": None}
        self.voice_evicted = False

    # ---- torch 工具

    @staticmethod
    def torch():
        try:
            import torch  # noqa: PLC0415

            return torch
        except Exception:
            return None

    def free(self):
        torch = self.torch()
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def stats(self):
        torch = self.torch()
        if torch is None or not torch.cuda.is_available():
            return {"gpu": None, "holder": self.holder}
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "freeMB": round(free / 1048576),
            "totalMB": round(total / 1048576),
            "usedMB": round((total - free) / 1048576),
            "holder": self.holder,
        }

    # ---- 语音让位（跨进程）

    def _voice_unload(self):
        try:
            req = urllib.request.Request(
                f"{VOICE_BASE}/unload", data=b"{}",
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                info = json.loads(r.read() or b"{}")
            if info.get("unloaded"):
                log(f"语音让出显存（{info['unloaded']}）")
            return True
        except urllib.error.HTTPError as e:
            # 老版本引擎没有这个路由：不算失败，只是拿不到那块显存
            if e.code == 404:
                return False
            log("语音 unload 失败", e.code)
            return False
        except Exception as e:
            log("语音 unload 连不上", repr(e))
            return False

    # ---- 租约

    def acquire(self, kind):
        with self.lock:
            if kind == "video":
                if self.holder == "image":
                    log("卸载图片（给视频腾显存）")
                    self.image_pipe = {"key": None, "pipe": None, "kind": None}
                self.voice_evicted = self._voice_unload()
                self.free()
            elif kind == "image":
                if self.holder == "video":
                    raise RuntimeError("视频任务正在占用显存，等它跑完再出图（单卡只能串行）")
            self.holder = kind

    def release(self, kind):
        with self.lock:
            if self.holder == kind:
                self.holder = None
        self.free()


LEASE = Lease()


# ============================================================== 图片：Qwen-Image-2.1

# Qwen-Image-2.1 官方比例档位（长边上限 2752 = 原生 2K 的 16:9）。
# 全部是 16 的倍数：single-stream DiT 的 patch 是 2×2 + VAE 16 倍压缩，
# 不是 16 的倍数会在最后一层 reconstruct 时崩，而且报错信息完全不提尺寸。
RATIOS = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "21:9": (21, 9),
}


def _round16(v):
    return max(256, int(round(v / 16.0)) * 16)


def _size_for(ratio, long_side):
    """按比例和长边算出 16 的倍数的宽高。"""
    if ratio in RATIOS:
        a, b = RATIOS[ratio]
        if a >= b:
            w, h = long_side, long_side * b / a
        else:
            w, h = long_side * a / b, long_side
        return _round16(w), _round16(h)
    return _round16(long_side), _round16(long_side)


def _parse_size(size, ratio, long_side):
    """优先用显式 size，其次 ratio，最后正方。"""
    if size:
        try:
            w, h = [int(float(x)) for x in str(size).lower().replace("*", "x").split("x")]
            return _round16(w), _round16(h)
        except Exception:
            pass
    return _size_for(ratio, long_side)


def _decode_data_url(data_url):
    """'data:image/png;base64,xxx' → PIL.Image（保持 RGBA，透明参考图不能丢 alpha）"""
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    raw = data_url.split(",", 1)[-1]
    img = Image.open(BytesIO(base64.b64decode(raw)))
    return img if img.mode in ("RGB", "RGBA") else img.convert("RGB")


def _pick_image_class():
    """
    找到 Qwen-Image 的 pipeline 类。

    为什么要探测而不是写死：2.1 的 pipeline 在 diffusers 里改过名
    （QwenImagePipeline → QwenImage21Pipeline 之类），写死一个名字会在
    升级 diffusers 的那天突然全挂，而且报的是 ImportError，看不出是命名问题。
    """
    import diffusers  # noqa: PLC0415

    for name in (
        "QwenImage21Pipeline",
        "QwenImagePipeline",
        "QwenImageEditPlusPipeline",
        "AutoPipelineForText2Image",
    ):
        cls = getattr(diffusers, name, None)
        if cls is not None:
            return name, cls
    raise RuntimeError("当前 diffusers 里没有 Qwen-Image 的 pipeline，请 pip install -U diffusers")


def _load_image_pipe(kind, model_id):
    """kind: edit（有参考图）| text（纯文生图）。2.1 是同一个权重做两件事。"""
    key = f"{kind}:{model_id}"
    if LEASE.image_pipe["key"] == key and LEASE.image_pipe["pipe"] is not None:
        return LEASE.image_pipe["pipe"]

    import torch  # noqa: PLC0415

    name, cls = _pick_image_class()
    LEASE.acquire("image")
    log(f"加载 {model_id}（{name}, {kind}）—— 首次 1-3 分钟，之后常驻")

    t0 = time.time()
    kwargs = {"torch_dtype": torch.bfloat16}
    try:
        pipe = cls.from_pretrained(model_id, **kwargs)
    except Exception:
        # 老版本 diffusers 不认 torch_dtype，退回 fp16 的写法
        kwargs["torch_dtype"] = torch.float16
        pipe = cls.from_pretrained(model_id, **kwargs)

    if IMAGE_OFFLOAD:
        # 关键：Qwen3-VL-8B 文本编码器占 17.5 GB，编码完前几步就不再需要。
        # offload 把它挪回内存，峰值从 ~40 GB 降到 ~22 GB —— 这是能和语音共存的原因。
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    log(f"图片模型就绪，{time.time() - t0:.1f}s")
    LEASE.image_pipe.update({"key": key, "pipe": pipe, "kind": kind})
    return pipe


def _img_kwargs(pipe, body, width, height, steps):
    """
    不同代 Qwen-Image 的 cfg 参数名不一样：
      1.x 用 true_cfg_scale（不是 guidance_scale），2.1 社区样例 guidance 5.0。
    签名里有什么就用什么，避免「参数不被接受」这种低级失败。
    """
    import inspect  # noqa: PLC0415

    params = set(inspect.signature(pipe.__call__).parameters)
    kw = {"prompt": body["prompt"], "width": width, "height": height, "num_inference_steps": steps}
    cfg = float(body.get("guidance") or body.get("cfg") or 5.0)
    if "true_cfg_scale" in params:
        kw["true_cfg_scale"] = cfg
    elif "guidance_scale" in params:
        kw["guidance_scale"] = cfg
    if body.get("negative") and "negative_prompt" in params:
        kw["negative_prompt"] = str(body["negative"])
    if "max_sequence_length" in params:
        kw["max_sequence_length"] = 512
    return kw


def _resolve_repo(name, default, alias=None):
    """把前端传来的模型名收敛成一个真实的 HF 仓库名。

    画布上的模型下拉里是给人看的名（`qwen-image-2.1`、`minimax-h3`），
    直接拿它去 `from_pretrained` 会去找一个不存在的仓库然后报 401/404 ——
    看起来像「鉴权失败」，实际是名字没映射。所以这里做一次收敛：
    带 `/` 的当仓库名照用（高级用户想换权重就写全名），
    不带 `/` 的先查别名表，查不到就回落到默认权重（不报错，先出图）。
    """
    raw = (name or "").strip()
    if not raw:
        return default
    if "/" in raw:
        return raw
    key = raw.lower()
    if alias and key in alias:
        return alias[key]
    if key in ("default", "auto", "local"):
        return default
    return default


def generate_image(body):
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("没有提示词")

    model_id = _resolve_repo(body.get("model"), IMAGE_MODEL)
    width, height = _parse_size(body.get("size"), body.get("ratio") or "1:1", IMAGE_MAX_SIDE)
    steps = int(body.get("steps") or IMAGE_STEPS)

    refs = body.get("image")
    if isinstance(refs, str):
        refs = [refs]
    refs = [r for r in (refs or []) if isinstance(r, str) and r.startswith("data:")]
    # 2.1 单次最多吃 10 张参考图；多传的部分直接截断，让请求成功而不是报错
    refs = refs[:10]

    if refs:
        pipe = _load_image_pipe("edit", model_id)
        images = [_decode_data_url(r) for r in refs]
        kw = _img_kwargs(pipe, body, width, height, steps)
        try:
            out = pipe(image=images if len(images) > 1 else images[0], **kw).images[0]
        except TypeError:
            # 某些版本只认单图直传，退一步只喂第一张
            out = pipe(image=images[0], **kw).images[0]
    else:
        pipe = _load_image_pipe("text", model_id)
        out = pipe(**_img_kwargs(pipe, body, width, height, steps)).images[0]

    # 原生透明：2.1 的 RGBA VAE 直接出 alpha 通道，保存成 PNG 就带透明底，
    # 不需要再跑一遍抠图 —— 画布里的贴纸/素材节点要的正是这个。
    if out.mode not in ("RGB", "RGBA"):
        out = out.convert("RGBA" if body.get("transparent") else "RGB")

    name = f"img_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    dest = OUT / name
    out.save(dest)
    LEASE.release("image")
    return {
        "created": int(time.time()),
        "data": [{
            "url": f"http://127.0.0.1:{PORT}/files/{name}",
            "size": f"{out.width}x{out.height}",
            "has_alpha": out.mode == "RGBA" and _has_alpha(out),
        }],
    }


def _has_alpha(img):
    try:
        return img.getchannel("A").getextrema()[0] < 255
    except Exception:
        return False


# ============================================================== 视频：MiniMax H3

def submit_video(body):
    job_id = uuid.uuid4().hex[:12]
    status_file = STATUS / f"{job_id}.json"
    status_file.write_text(
        json.dumps({"status": "queued", "progress": 0, "created": time.time()}), "utf-8"
    )

    alias = {
        "fast": VIDEO_FAST, "wan": VIDEO_FAST, "quick": VIDEO_FAST, "turbo": VIDEO_FAST,
        "wan2.2": VIDEO_FAST, "wan2.2-ti2v-5b": VIDEO_FAST, "video-fast": VIDEO_FAST,
        "wan-fast": VIDEO_FAST, "wan_fast": VIDEO_FAST,
        "h3": VIDEO_MODEL, "minimax": VIDEO_MODEL, "minimax-h3": VIDEO_MODEL,
        "minimax-h3-base": VIDEO_MODEL, "h3-base": VIDEO_MODEL, "video": VIDEO_MODEL,
    }
    model = _resolve_repo(body.get("model"), VIDEO_DEFAULT, alias)
    fast = model == VIDEO_FAST

    args = [
        sys.executable, str(ROOT / "video_worker.py"),
        "--job", job_id, "--status", str(status_file), "--out", str(OUT),
        "--port", str(PORT), "--model", model,
        "--prompt", body.get("prompt") or "",
        "--duration", str(body.get("duration") or body.get("seconds") or 5),
        "--size", str(body.get("size") or body.get("resolution") or ""),
        "--backend", str(body.get("backend") or VIDEO_BACKEND),
        "--variant", str(body.get("variant") or "auto"),
        "--steps", str(body.get("steps") or 0),
    ]
    if body.get("seed") not in (None, ""):
        args += ["--seed", str(body["seed"])]

    imgs = body.get("image")
    if isinstance(imgs, str):
        imgs = [imgs]
    imgs = [r for r in (imgs or []) if isinstance(r, str) and r.startswith("data:")][:3]
    for i, one in enumerate(imgs):
        p = OUT / f"ref_{job_id}_{i}.png"
        _decode_data_url(one).save(p)
        args += ["--image", str(p)]

    # 视频独占显存：先把语音请出去。子进程一退出显存就彻底还回去，
    # 所以这里不需要「跑完再恢复」的补偿逻辑 —— 语音下次要出声会自己重载。
    LEASE.acquire("video")

    logf = open(DATA / "video.log", "a", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=logf, stderr=logf, start_new_session=True)

    # 后台看门狗：进程结束就释放租约，别让一次崩溃把整台机器卡在「视频占用中」
    def reap():
        proc.wait()
        LEASE.release("video")
        log(f"视频任务 {job_id} 进程结束（exit={proc.returncode}）")

    threading.Thread(target=reap, daemon=True).start()
    return {"id": job_id, "status": "queued", "model": model, "created": int(time.time())}


def video_status(job_id):
    f = STATUS / f"{os.path.basename(job_id)}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text("utf-8"))
    except Exception:
        return {"status": "processing", "progress": 0}


# ============================================================== 透传

def proxy_json(path, body, timeout=600):
    req = urllib.request.Request(
        f"{TEXT_UPSTREAM}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def proxy_stream(handler, path, body):
    """把上游的 SSE 原样泼回去。不要自己解包再拼 —— 中间加一层就会丢 tool_calls 增量。"""
    req = urllib.request.Request(
        f"{TEXT_UPSTREAM}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            while True:
                chunk = r.read(4096)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except Exception as e:
        payload = json.dumps({"error": str(e)}).encode()
        handler.wfile.write(b"data: " + payload + b"\n\n")
        handler.wfile.flush()


def proxy_voice(handler, body):
    req = urllib.request.Request(
        f"{VOICE_BASE}/v1/audio/speech",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        data = r.read()
        ctype = r.headers.get("Content-Type", "audio/wav")
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


# ============================================================== HTTP

def _probe(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return "ready" if r.status < 400 else f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return "ready" if e.code < 500 else f"HTTP {e.code}"
    except Exception:
        return "down"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "yj-model-server/2.0"

    def log_message(self, fmt, *args):
        return

    def _json(self, code, obj):
        buf = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise ValueError("请求体不是合法 JSON")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/healthz":
            caps = {
                "text": _probe(TEXT_UPSTREAM + "/models"),
                "voice": _probe(VOICE_BASE + "/healthz"),
                "image": "ready",
                "video": "ready" if (ROOT / "video_worker.py").exists() else "missing",
            }
            return self._json(200, {
                "ok": True, "mode": MODE, "caps": caps, "vram": LEASE.stats(),
                "models": {
                    "image": IMAGE_MODEL, "video": VIDEO_MODEL,
                    "video_fast": VIDEO_FAST, "video_default": VIDEO_DEFAULT,
                    "image_steps": IMAGE_STEPS,
                    "image_max_side": IMAGE_MAX_SIDE, "image_offload": IMAGE_OFFLOAD,
                },
            })

        if path == "/v1/models":
            return self._json(200, {
                "object": "list",
                "data": [
                    {"id": IMAGE_MODEL, "object": "model", "owned_by": "local", "kind": "image"},
                    {"id": VIDEO_MODEL, "object": "model", "owned_by": "local", "kind": "video"},
                    {"id": VIDEO_FAST, "object": "model", "owned_by": "local", "kind": "video"},
                    {"id": os.environ.get("YJ_TEXT_MODEL", "Qwen3-8B-Instruct"),
                     "object": "model", "owned_by": "local", "kind": "chat"},
                ],
            })

        if path.startswith("/v1/videos/"):
            job = video_status(path.rsplit("/", 1)[-1])
            if job is None:
                return self._json(404, {"error": {"message": "没有这个任务"}})
            return self._json(200, job)

        if path.startswith("/files/"):
            name = os.path.basename(path)
            f = OUT / name
            if not f.exists():
                return self._json(404, {"error": {"message": "文件不存在"}})
            # 视频要给 Range：浏览器和 <video> 都靠它做拖动预览，
            # 一次性吐整个 mp4 会让播放在大文件上「点了没反应」。
            return self._send_file(f)

        return self._json(404, {"error": {"message": f"未知路径 {path}"}})

    def _send_file(self, f: Path):
        size = f.stat().st_size
        ctype = {
            ".mp4": "video/mp4", ".wav": "audio/wav", ".mp3": "audio/mpeg",
            ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
        }.get(f.suffix.lower(), "application/octet-stream")

        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else 0
                end = int(b) if b else size - 1
                end = min(end, size - 1)
                if start > end:
                    raise ValueError
            except Exception:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with f.open("rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    chunk = fh.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            return None

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with f.open("rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
        return None

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
        except ValueError as e:
            return self._json(400, {"error": {"message": str(e)}})

        try:
            if path == "/v1/images/generations":
                if MODE == "text":
                    return self._json(503, {"error": {"message": "当前 YJ_MODE=text，图片未启用"}})
                return self._json(200, generate_image(body))

            if path == "/v1/videos":
                if MODE == "text":
                    return self._json(503, {"error": {"message": "当前 YJ_MODE=text，视频未启用"}})
                return self._json(200, submit_video(body))

            if path == "/v1/chat/completions":
                if body.get("stream"):
                    return proxy_stream(self, "/chat/completions", body)
                code, raw = proxy_json("/chat/completions", body)
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return None

            if path == "/v1/audio/speech":
                return proxy_voice(self, body)

            return self._json(404, {"error": {"message": f"未知路径 {path}"}})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:400]
            return self._json(502, {"error": {"message": f"上游 {e.code}：{detail}"}})
        except urllib.error.URLError as e:
            return self._json(502, {"error": {"message": f"连不上上游：{e.reason}"}})
        except Exception as e:
            log("生成失败", repr(e))
            return self._json(500, {"error": {"message": str(e)}})


def main():
    log(f"model-server 起在 http://0.0.0.0:{PORT}（mode={MODE}）")
    log(f"产物目录 {OUT}")
    log(f"文本 {TEXT_UPSTREAM} / 语音 {VOICE_BASE}")
    log(f"图片 {IMAGE_MODEL}（{IMAGE_STEPS} 步 / 长边 {IMAGE_MAX_SIDE}px / offload={IMAGE_OFFLOAD}）")
    log(f"视频 {VIDEO_MODEL}（快档 {VIDEO_FAST}，backend={VIDEO_BACKEND}）")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("收到中断，退出")
        srv.shutdown()


if __name__ == "__main__":
    main()
