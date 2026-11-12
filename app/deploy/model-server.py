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
    YJ_VIDEO_MODEL     默认 MiniMaxAI/MiniMax-H3（**视频只有这一个后端**）
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
# 空字符串 = 没配。`os.environ.get(k, default)` 在「变量存在但为空」时会返回 ''，
# 那样 VIDEO_MODEL 变成空路径，`_has_weights('')` 会去 stat 当前目录 —— 一个假的绿。
_VIDEO_ENV = os.environ.get("YJ_VIDEO_MODEL", "").strip()
VIDEO_MODEL = _VIDEO_ENV or "MiniMaxAI/MiniMax-H3"
# 视频后端按显存分两档，**不静默降级**：
#
#   H3      —— 全量档。实测 bf16 权重 125 GB（DiT 62 + Qwen3-VL 编码器 63），
#              官方 NF4 版约 42 GB 但仍需 30–36 GB 显存。单卡 24 GB 跑不了，
#              48 GB 是起步线。
#   Wan2.2  —— 低显存档（GGUF Q4_K_M 3.4 GB + 蒸馏 4 步），12 GB 卡可跑。
#              走 `deploy/wan_worker.py` 独立子进程，退出即释放显存。
#
# 选哪个由盘上装了哪份权重决定（`video_model()`），选了装不上的会明确失败 ——
# 「悄悄换了个模型出片」比「直接告诉我不够」有害得多。
VIDEO_DEFAULT = os.environ.get("YJ_VIDEO_DEFAULT", "").strip() or VIDEO_MODEL
VIDEO_BACKEND = os.environ.get("YJ_VIDEO_BACKEND", "auto")

# ---- HyperFlow H3 Flash：云端默认视频档 ----
#
# 上游 MiniMax-H3 主干 31B / bf16 125 GB，48 GB 卡要开重 offload 才勉强跑，
# 一条 5 秒片十分钟起步。HyperFlow H3 Flash 是它的**压缩少步版**（13.7B）：
#   · 主干 INT8 convrot 16 GB（另有 GGUF 7.2–15.2 GB 一路，那条走 ComfyUI）；
#   · 4 / 6 / 8 步、无 CFG（默认 6 步），4000 系卡上 5.2 s 片段 150 s 量级；
#   · 一次前向同时出视频 + 32 kHz 立体声，省掉单独的音效 / 配音一趟；
#   · 文本编码器、视频 VAE、音频 VAE 都复用 MiniMax-H3 仓库里的那几件。
#
# **权重分两处放**，缺一不可（这也是它和以前那两档最不一样的地方）：
#   WEIGHTS  /root/models/hyperflow-h3-flash/fl2va_int8_convrot/   16 GB 主干
#   BASE     /root/models/MiniMax-H3/                              63 GB 编码器 + VAE
# 所以判据不能只看一个目录的体积 —— 见 `flash_ready()`。
FLASH_DIR = Path(
    os.environ.get("YJ_FLASH_DIR")
    or (Path(os.environ.get("YJ_LOCAL_MODELS") or "/root/models") / "hyperflow-h3-flash")
)
# 组件目录（文本编码器 / 视频 VAE / 音频 VAE / tokenizer / processor）。
#
# 两个位置都认，顺序即优先级：
#   ① `<FLASH_DIR>/MiniMax-H3/` —— 客户端下载器就是这么放的（一个包目录装全套，
#      免得「主模型一个目录、组件另一个目录」在桌面端各处拷贝时散架）；
#   ② `<models>/MiniMax-H3/`   —— 命令行 / 平台 UI 手工下的那份（也是 HyperFlow
#      官方 CLI 的默认行为）。
# 写死其中一条都会在另一种装法上判成「组件缺失」，而现象是「权重明明在、就是不出片」。
FLASH_BASE = Path(
    os.environ.get("YJ_FLASH_BASE")
    or (Path(os.environ.get("YJ_LOCAL_MODELS") or "/root/models") / "MiniMax-H3")
)
# 主干 INT8 是 15.9 GB（3 个分片，十进制）。门槛取 15.0 GB：
# 半截分片不能被判成 ready；也别按标称 16.2 卡死 —— 上游重新导出会挪动几十 MB。
#
# 单位是**十进制 GB**（`* 1e9`），和仓库文件清单里的 GB 同口径。
# 用 `1024**3` 会把 15.9 GB 算成 14.8「GB」，门槛 15.0 反而判不过 ——
# 这正是第一次接进来时 flash 报 ready=false 的原因，排查了半轮。
FLASH_MIN_BYTES = int(float(os.environ.get("YJ_FLASH_MIN_GB", "15.0")) * 1e9)
# 帧规则与上游一致：17n+5，124（5.17 s）到 362（15.08 s）。
FLASH_MIN_FRAMES, FLASH_MAX_FRAMES = 124, 362

IMAGE_STEPS = int(os.environ.get("YJ_IMAGE_STEPS", "28"))
IMAGE_MAX_SIDE = int(os.environ.get("YJ_IMAGE_MAX_SIDE", "1536"))
# 低显存档的出图长边。1024² 在 12 GB 卡上实测峰值 21 GB（激活值 + 注意力），
# 超出部分会被 Windows 的 CUDA 系统内存回退吞掉 → 整机分页 → 鼠标都不动。
# 768² 面积只有 1024² 的 56%，激活值同比降，能稳稳落在预算内。
IMAGE_SMALL_SIDE = int(os.environ.get("YJ_IMAGE_SMALL_SIDE", "768"))
IMAGE_OFFLOAD = os.environ.get("YJ_IMAGE_OFFLOAD", "1") != "0"
# 低显存档的 offload 口径与全量档不同：编码器 17 GB 整块搬不上 12 GB 卡，
# 默认按 block 分组过（见 `_group_offload_text_encoder`）。设 1 可强制走整组件搬运，
# 只在大显存机器上才有意义（例如 48 GB，整块搬反而更快）。
IMAGE_OFFLOAD_WHOLE = os.environ.get("YJ_IMAGE_OFFLOAD_WHOLE", "0") == "1"
# 空闲多久把权重彻底还给系统（秒）。0 = 永不自动卸载。
# 分级释放，不是一个数：
#   每次生成结束 → 立刻把 DiT/VAE 挪回内存，显存归零（秒级，下次只花几秒搬回来）
#   空闲超过这个值 → 连内存一起还（管线对象整个丢掉，下次要重新加载 3-5 分钟）
# 默认 180 秒：一次创作里连续出图不会踩到重新加载，放下不管了就把机器还回去。
IDLE_UNLOAD_S = int(os.environ.get("YJ_IDLE_UNLOAD_S", "180"))
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
        # 图片请求的串行闸：不是「租约」，是「排队」——单卡只有一份管线，
        # 并发调用会把 CPU offload 的 hook 搅乱（见 generate_image 注释）。
        self.image_lock = threading.Lock()
        self.image_pipe = {"key": None, "pipe": None, "kind": None}
        self.voice_evicted = False
        # 分级释放的记账：用于「用完就把显存还回去」和空闲超时深度卸载
        self.image_used_at = 0.0
        self.image_offloaded = False
        self.unloads = 0
        self._watcher = None

    # ---- 空闲守护

    def start_idle_watcher(self):
        """后台线程：空闲超 IDLE_UNLOAD_S 就把权重彻底还给系统。

        为什么要有这个：本机 31.8 GB 内存，图片管线常驻是
        （Qwen3-VL-8B 编码器 17.5 GB + DiT 4.2 GB + 杂项）≈ 22 GB。
        用户「用完就不管了」是常态，让它一直占着 22 GB 内存 + 4.5 GB 显存，
        整台机器就别想干别的了 —— 实测会让操作系统开始分页，鼠标都卡。
        """
        if IDLE_UNLOAD_S <= 0 or self._watcher is not None:
            return None

        def loop():
            while True:
                time.sleep(min(10, max(2, IDLE_UNLOAD_S // 6)))
                try:
                    if not self.image_pipe["pipe"]:
                        continue
                    idle = time.time() - self.image_used_at
                    if idle < IDLE_UNLOAD_S:
                        continue
                    if self.holder is not None:
                        continue  # 正在用，跳过
                    if not self.image_lock.acquire(blocking=False):
                        continue  # 有请求在排队，别抢锁
                    try:
                        log(f"空闲 {idle:.0f}s ≥ {IDLE_UNLOAD_S}s，释放图片模型（显存 + 内存）")
                        self.unload_image(why=f"空闲 {idle:.0f}s")
                    finally:
                        self.image_lock.release()
                except Exception as e:  # noqa: BLE001
                    log("空闲卸载线程异常（继续跑）", repr(e))

        self._watcher = threading.Thread(target=loop, name="idle-unload", daemon=True)
        self._watcher.start()
        log(f"空闲自动卸载已启用：{IDLE_UNLOAD_S}s 无人使用就释放权重")
        return self._watcher

    def touch(self, kind="image"):
        self.image_used_at = time.time()
        if kind == "image":
            self.image_offloaded = False

    # ---- 分级释放

    @staticmethod
    def _image_modules(pipe):
        """管线里所有**真正占显存**的子模块。

        不能写死 `("transformer", "vae", "text_encoder")`：Qwen-Image 的骨干叫
        `transformer`，SDXL 的叫 `unet`，编码器也分 `text_encoder` / `text_encoder_2`。
        漏掉的那个会一直趴在显存里 —— 表现为「点了释放、nvidia-smi 还占 5 GB」。
        """
        out = []
        for attr in (
            "transformer",
            "unet",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
        ):
            mod = getattr(pipe, attr, None)
            if mod is not None:
                out.append(mod)
        return out

    def offload_image(self, why=""):
        """把图片管线搬回内存：显存立刻归零，内存还留着（下次几秒就能用）。

        注意不能只丢引用：`enable_model_cpu_offload` 会给模块挂 accelerate hook，
        丢引用而模块还被 hook 抓着的话，显存不会真的回来。
        """
        pipe = self.image_pipe.get("pipe")
        if pipe is None:
            return False
        try:
            for mod in self._image_modules(pipe):
                try:
                    mod.to("cpu")
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            log("搬回内存时出错（继续释放）", repr(e))
        self.free()
        self.image_offloaded = True
        if why:
            log(f"图片权重已退出显存（{why}），内存保留")
        return True

    def unload_image(self, why=""):
        """彻底卸载图片管线：显存和内存一起还。下次请求要重新加载。"""
        pipe = self.image_pipe.get("pipe")
        if pipe is None:
            return False
        # 先摘 hook 再丢引用，否则 accelerate 的 hook 会吊着模块不放
        try:
            for mod in self._image_modules(pipe):
                try:
                    from accelerate.hooks import remove_hook_from_module  # noqa: PLC0415

                    remove_hook_from_module(mod, recurse=True)
                except Exception:
                    pass
                try:
                    mod.to("cpu")
                except Exception:
                    pass
        except Exception:
            pass
        self.image_pipe = {"key": None, "pipe": None, "kind": None}
        self.image_offloaded = False
        self.unloads += 1
        try:
            del pipe
        except Exception:
            pass
        self.free()
        torch = self.torch()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if why:
            log(f"图片模型已卸载（{why}）")
        return True

    def image_state(self):
        loaded = self.image_pipe.get("pipe") is not None
        idle = round(time.time() - self.image_used_at, 1) if self.image_used_at else None
        return {
            "loaded": loaded,
            "offloaded": bool(self.image_offloaded),
            "key": self.image_pipe.get("key"),
            "idleSeconds": idle,
            "idleUnloadSeconds": IDLE_UNLOAD_S,
            "unloads": self.unloads,
        }

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
            return {"gpu": None, "holder": self.holder, "image": self.image_state()}
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu": torch.cuda.get_device_name(0),
            "freeMB": round(free / 1048576),
            "totalMB": round(total / 1048576),
            "usedMB": round((total - free) / 1048576),
            "holder": self.holder,
            "image": self.image_state(),
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
                # 无条件走 `unload_image()`，不看 holder。
                #
                # 以前这里写的是 `if self.holder == "image": self.image_pipe = {...}`
                # 两条都错：
                #   1. 出图结束后 `release("image")` 已经把 holder 置成 None，
                #      所以这个分支**从来不执行** —— SDXL 的 6.5 GB 原地不动；
                #   2. 就算执行，只丢引用不摘 accelerate hook，显存和内存都不会真的
                #      还回来（offload_image 的注释里就写了这一条）。
                # 后果：视频 worker 加载 Wan（约 17 GB）时系统内存只剩 2 GB，
                # 卡在「Loading pipeline components 4/5」十几分钟不动，
                # 看起来像死锁，其实是内存不够在换页。
                self.unload_image(why="让出内存给视频")
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
    # 显式 floor(v/16 + 0.5)：Python 的 round 是银行家舍入，和 JS 的 Math.round 在
    # 恰好 .5 时差一档（42.5 → 42 vs 43）。图像这条链两端都会算这个数，别让它分叉。
    return max(256, int(v / 16.0 + 0.5) * 16)


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


def _load_gguf_transformer(gguf_path: Path, torch, model_dir: Path | None = None):
    """把 GGUF 量化权重读成一个 transformer 对象。

    这是低显存档的核心：Q4_K_M 只有 bf16 的 1/3 体积（4.2 GB vs 14.2 GB），
    在 12 GB 卡上能整块放进显存，剩下只留给激活值和 VAE。

    **必须传 `config=`**：不传的话 diffusers 会拿默认仓库名去 Hub 上找 config.json，
    离线机器上会报「does not appear to have a file named config.json」——
    看着像权重坏了，其实是它在找一份本地早就有的配置。同理要先读本地配置里的
    `_class_name` 再决定用哪个类（2.1 是 `QwenImage21Transformer2DModel`，
    用 1.x 的类去装会静默按错误的层数构造，报错点离真正原因很远）。
    """
    from diffusers import GGUFQuantizationConfig  # noqa: PLC0415

    import diffusers  # noqa: PLC0415

    cfg_dir = Path(model_dir) / "transformer" if model_dir else None
    want = ""
    if cfg_dir and (cfg_dir / "config.json").is_file():
        try:
            want = str(json.loads((cfg_dir / "config.json").read_text("utf-8")).get("_class_name") or "")
        except Exception:
            want = ""

    names = []
    if want:
        names.append(want)
    names += ["QwenImage21Transformer2DModel", "QwenImageTransformer2DModel"]
    cands = []
    for name in names:
        c = getattr(diffusers, name, None)
        if c is not None and c not in cands:
            cands.append(c)
    if not cands:
        raise RuntimeError("当前 diffusers 没有 Qwen-Image 的 transformer 类，无法加载 GGUF")

    err: Exception | None = None
    for cls in cands:
        try:
            kw = {}
            if cfg_dir and (cfg_dir / "config.json").is_file():
                kw["config"] = str(cfg_dir)
            model = cls.from_single_file(
                str(gguf_path),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                **kw,
            )
            # diffusers 会把一维 BF16 张量按字节数建参数（4096 → 8192），
            # 不修就在第一次前向炸在 text_norm 上。详见 deploy/gguf_fix.py
            try:
                from gguf_fix import repair_bf16_1d  # noqa: PLC0415

                repair_bf16_1d(model, gguf_path, log=log)
            except Exception as e:  # noqa: BLE001
                log("GGUF BF16 修补没跑起来", repr(e))
            # 社区 GGUF 把 SwiGLU 的 gate/up 融合成一键，diffusers 要两个 Linear。
            # 不拆的话每个 block 的 MLP 保持随机初始化 —— 不报错，只吐噪声图。
            try:
                from gguf_fix import split_fused_mlp, verify_keys  # noqa: PLC0415

                split_fused_mlp(model, gguf_path, log=log)
                verify_keys(model, gguf_path, log=log)
            except Exception as e:  # noqa: BLE001
                log("GGUF 融合 MLP 拆分没跑起来", repr(e))
            return model
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"GGUF 加载失败（{gguf_path.name}）：{err}")


def _load_image_pipe(kind, model_id):
    """kind: edit（有参考图）| text（纯文生图）。2.1 是同一个权重做两件事。"""
    key = f"{kind}:{model_id}"
    if LEASE.image_pipe["key"] == key and LEASE.image_pipe["pipe"] is not None:
        return LEASE.image_pipe["pipe"]

    import torch  # noqa: PLC0415

    name, cls = _pick_image_class()
    LEASE.acquire("image")

    # 换形态（text↔edit）时先把旧管线丢掉再加载：
    # 两个管线各自的 `enable_model_cpu_offload()` 会往同一批模块上挂 hook，
    # 旧的留在内存里就会被下一轮 offload 覆盖，报出来的是一句看不懂的
    # `'Qwen3VLForConditionalGeneration' object has no attribute '_hf_hook'`。
    if LEASE.image_pipe["pipe"] is not None:
        LEASE.image_pipe = {"key": None, "pipe": None, "kind": None}
        gc.collect()
        torch.cuda.empty_cache()

    # 低显存档：目录里有 .gguf 就走量化加载（权重只占 4.2 GB）
    target = str(model_id)
    gguf = None
    if Path(target).is_dir():
        gguf = _gguf_in(Path(target))
    if gguf is None:
        small = small_image_dir()
        if small:
            target = small
            gguf = _gguf_in(Path(small))

    log(f"加载 {target}（{name}, {kind}{', GGUF ' + gguf.name if gguf else ''}）—— 首次 1-3 分钟，之后常驻")

    t0 = time.time()
    if gguf is not None:
        transformer = _load_gguf_transformer(gguf, torch, Path(target))
        pipe = cls.from_pretrained(target, transformer=transformer, torch_dtype=torch.bfloat16)
    else:
        kwargs = {"torch_dtype": torch.bfloat16}
        try:
            pipe = cls.from_pretrained(target, **kwargs)
        except Exception:
            # 老版本 diffusers 不认 torch_dtype，退回 fp16 的写法
            kwargs["torch_dtype"] = torch.float16
            pipe = cls.from_pretrained(target, **kwargs)

    if gguf is not None and not IMAGE_OFFLOAD_WHOLE:
        # 低显存档**不能**用 `enable_model_cpu_offload()`：
        # 它是整组件搬运，而 Qwen3-VL-8B 编码器 bf16 就有 17.5 GB ——
        # 比整张 12 GB 卡还大，搬上去必 OOM。
        # 只对编码器做 block 分组，DiT / VAE **先留在内存**：
        # 上卡时机交给 `_generate_small` 的两段式（编码时 DiT 不在卡上，
        # 编完卸编码器、再让 DiT 上卡）。在这里提前上卡会立刻顶穿预算。
        _group_offload_text_encoder(pipe, torch)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    elif IMAGE_OFFLOAD:
        # 全量权重档：Qwen3-VL-8B 编码器编码完就不再需要，offload 把它挪回内存。
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    log(f"图片模型就绪，{time.time() - t0:.1f}s")
    LEASE.image_pipe.update({"key": key, "pipe": pipe, "kind": kind})
    LEASE.touch("image")
    return pipe


def _group_offload_text_encoder(pipe, torch) -> None:
    """低显存档的文本编码器摆放策略。

    实测结论（12 GB 卡 + Qwen3-VL-8B 17.5 GB bf16 编码器）：

      block_level 分组  → 编码阶段就 OOM（10.42/10.55 GiB）。
        非分组部分（embed_tokens 1.2 GB + visual 塔 + 各 block 的 norm）
        由根 hook 一起 onload，光常驻就吃满预算，DiT 还没上卡就没地方了。
      leaf_level        → 峰值能压住，但每个算子一次 PCIe 往返，
        750 个张量编码一次要几分钟，产品上不能用。
      **CPU 编码**       → 显存零占用，代价是一次 prompt 几十秒。
        17.5 GB 权重放进 31.8 GB 内存是够的，且编码只做一次。

    所以默认走 CPU 编码（`YJ_TE_DEVICE=cpu`），大显存机器可以设 `gpu` 换速度。
    """
    mode = os.environ.get("YJ_TE_DEVICE", "cpu").lower()
    if mode == "gpu":
        try:
            from diffusers.hooks import apply_group_offloading  # noqa: PLC0415

            apply_group_offloading(
                pipe.text_encoder,
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                block_modules=["model.language_model.layers", "model.visual.blocks"],
            )
            log("文本编码器按 block 分组 offload（需大显存，12 GB 卡会 OOM）")
        except Exception as e:  # noqa: BLE001
            log("分组 offload 不可用，改用 CPU 编码", repr(e))
            mode = "cpu"
    if mode != "gpu":
        try:
            pipe.text_encoder.to("cpu")
        except Exception:
            pass
        log("文本编码器常驻内存（CPU 编码，显存零占用）")


def _park_text_encoder(pipe, torch, log_fn=None) -> None:
    """编码完就把文本编码器整个退出显存 —— 两段式的关键一步。

    编码器（17 GB）和 DiT（4.2 GB）只有 12 GB 卡，两者不能同时占着。
    摘掉 hook 再 to("cpu")，否则 accelerate 的 hook 会吊着模块不放，
    `empty_cache()` 也拿不回那 5.9 GB。
    """
    te = getattr(pipe, "text_encoder", None)
    if te is None:
        return
    try:
        from accelerate.hooks import remove_hook_from_module  # noqa: PLC0415

        remove_hook_from_module(te, recurse=True)
    except Exception:
        pass
    try:
        te.to("cpu")
    except Exception:
        pass
    torch.cuda.empty_cache()
    if log_fn:
        free, total = torch.cuda.mem_get_info()
        log_fn(f"编码器已退出显存（空闲 {free / 2**30:.1f}/{total / 2**30:.1f} GB）")


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
    """
    图片请求**串行**执行。

    单卡上只有一份图片管线，而 `enable_model_cpu_offload()` 是把 hook 挂在模块上、
    按需在 CPU/GPU 之间搬权重的 —— 两个请求同时进来就会互相把对方搬一半的模块
    再搬走，报出来的是 `object has no attribute '_hf_hook'` 这种不像人话的 500。
    客户端上一秒点了「生成」、下一秒又点了别的卡，正好撞进这个窗口。

    排队比并发快：一个请求 20–90 秒，并发的代价是两边都失败还要重新加载 31 GB 权重。
    """
    with LEASE.image_lock:
        return _generate_image_locked(body)


def _generate_small(pipe, kw, images):
    """低显存档的两段式出图：先编码（DiT 不在卡上），再卸编码器、让 DiT 上卡。

    为什么必须分两段：12 GB 卡装不下「编码器常驻 ~5.9 GB + DiT 4.2 GB +
    去噪激活」。一次性 `pipe(...)` 的结果是实测 OOM（10.37/10.55 GiB 已分配）。
    拆开后每一段的峰值都落在 7 GB 以内，代价是多一次 `torch.cuda.empty_cache()`。
    """
    import torch  # noqa: PLC0415

    dev = torch.device("cuda")
    # 编码器在 CPU（见 `_group_offload_text_encoder`）：输入的 device 也要给 cpu，
    # 否则 transformer 的 embedding 在 cpu 上收到 cuda 张量会直接报设备不一致。
    enc_dev = pipe.text_encoder.device if hasattr(pipe.text_encoder, "device") else torch.device("cpu")
    prompt = kw.pop("prompt")
    # CFG=1（蒸馏权重的唯一正确用法）时不走无分类器引导，负面提示词本来就被忽略，
    # 传了反而让 encode_prompt 多编一遍，白烧几十秒。
    kw.pop("negative_prompt", None)

    t0 = time.time()
    embeds, embeds_mask, image_pad_mask = pipe.encode_prompt(
        prompt, image=images, device=enc_dev, num_images_per_prompt=1
    )
    log(f"提示词编码完成 {time.time() - t0:.1f}s（{tuple(embeds.shape)}，设备 {enc_dev}）")
    # 编完立刻搬到显卡，并让编码器彻底退出显存（CPU 编码时本来就不占，这步是兜底）
    embeds = embeds.to(dev)
    if embeds_mask is not None:
        embeds_mask = embeds_mask.to(dev)
    image_pad_mask = image_pad_mask.to(dev)

    _park_text_encoder(pipe, torch, log_fn=log)

    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    kw["prompt_embeds"] = embeds
    if embeds_mask is not None:
        kw["prompt_embeds_mask"] = embeds_mask
    kw["image_pad_mask"] = image_pad_mask
    return pipe(**kw).images[0]


def _load_sdxl_pipe(kind):
    """SDXL-Lightning 管线。kind: text（文生图）| edit（图生图）。

    为什么要单开一条而不是塞进 `_load_image_pipe`：两条链没有任何可复用部件 ——
    单 vs 双文本编码器、MMDiT vs UNet、指令式编码 vs CLIP 池化、CFG 语义也不同。
    混在一起只会得到「参数被忽略但不报错」这种最难查的问题。
    """
    key = f"sdxl:{kind}"
    if LEASE.image_pipe["key"] == key and LEASE.image_pipe["pipe"] is not None:
        pipe = LEASE.image_pipe["pipe"]
        if not IMAGE_OFFLOAD and LEASE.image_offloaded:
            # 上次出图后被 `Lease.offload_image()` 搬回内存了。全放卡模式没有
            # accelerate hook 帮忙搬回来，不补这一步的话下一次会在 CPU 上采样
            # —— 表现是「点了生成，十分钟没反应」。
            pipe.to("cuda")
            LEASE.image_offloaded = False
            log("SDXL 权重搬回显存（上次出图后已释放）")
        return pipe

    import torch  # noqa: PLC0415

    LEASE.acquire("image")
    if LEASE.image_pipe["pipe"] is not None:
        LEASE.image_pipe = {"key": None, "pipe": None, "kind": None}
        gc.collect()
        torch.cuda.empty_cache()

    d = Path(sdxl_dir() or SDXL_DIR)
    ck = _sdxl_ckpt(d)
    if ck is None:
        raise RuntimeError(
            f"{d} 里没有完整的 SDXL checkpoint（需要 ≥6 GB 的 .safetensors）。"
            "低显存图像档靠它，缺了就只能回落到 Qwen-Image（要 22 GB 内存）。"
        )

    from diffusers import (  # noqa: PLC0415
        EulerDiscreteScheduler,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    cls = StableDiffusionXLPipeline if kind == "text" else StableDiffusionXLImg2ImgPipeline
    log(f"加载 SDXL-Lightning {ck.name}（{ck.stat().st_size / 2**30:.2f} GiB, {kind}）")

    t0 = time.time()
    # config 指向本地目录（从 SDXL base 抄来的各组 config.json）。
    # 不给的话 diffusers 会去 HF 拉配置，离线部署直接失败。
    pipe = cls.from_single_file(
        str(ck),
        config=str(d),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # 蒸馏权重的正确采样器：欧拉离散 + trailing 时间步。
    # 用默认的 DPM 多步 + leading 会让 4 步出图偏糊、偏灰 —— 这不是调参问题，
    # 是蒸馏时用的就是这个 schedule，错了等于换了采样轨迹。
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    if IMAGE_OFFLOAD:
        # 整组件 offload（不是 block 级）：SDXL 最大的组件是 UNet 5 GB，
        # 12 GB 卡装得下，只是不想让它长期空占。每次请求搬一次，1-2 秒。
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    for attr, fn in (("vae", "enable_tiling"), ("vae", "enable_slicing")):
        try:
            getattr(getattr(pipe, attr), fn)()
        except Exception:
            pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    log(f"SDXL 就绪，{time.time() - t0:.1f}s")
    LEASE.image_pipe.update({"key": key, "pipe": pipe, "kind": kind})
    LEASE.touch("image")
    return pipe


def _warm_translator():
    """启动后在后台把中译英模型载入内存。

    为什么预热：MarianMT 首次加载 30–40 秒（258 个张量 + tokenizer），
    以前这段时间全压在用户第一次出图的等待里 —— 现象是「点了生成，
    一分半没动静」，而显卡一直空着，看起来像卡死。
    放到服务启动后跑，用户就感觉不到这一层。

    失败不影响服务：`_localize_prompt` 里会再试一次，再失败就按原文提交。
    """
    if not (TRANS_DIR / "config.json").is_file():
        return

    def run():
        try:
            from prompt_l10n import preload  # noqa: PLC0415

            t0 = time.time()
            if preload(str(TRANS_DIR)):
                log(f"中译英模型已预热（{time.time() - t0:.1f}s）")
        except Exception as e:  # noqa: BLE001
            log("中译英预热失败（不影响出图，首次翻译会现加载）", repr(e))

    threading.Thread(target=run, name="warm-translator", daemon=True).start()


def _finish_image(out, body):
    """存盘 + 释放显存。两条图像链路共用的收尾。"""
    # 原生透明：2.1 的 RGBA VAE 直接出 alpha，保存成 PNG 就带透明底，
    # 不需要再跑一遍抠图 —— 画布里的贴纸/素材节点要的正是这个。
    # SDXL 没有 alpha VAE，这里不会误伤（mode 已是 RGB）。
    if out.mode not in ("RGB", "RGBA"):
        out = out.convert("RGBA" if body.get("transparent") else "RGB")

    name = f"img_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
    dest = OUT / name
    out.save(dest)
    LEASE.touch("image")
    LEASE.release("image")
    # 出图完成就把模型退出显存：一次创作里连续出图时，下次只花几秒搬回来；
    # 但显存不再被「暂时没人用」的模型占着。内存留着，超时后由守护线程回收。
    try:
        LEASE.offload_image(why="出图完成")
    except Exception as e:  # noqa: BLE001
        log("出图后释放显存失败（不影响本次结果）", repr(e))
    return {
        "created": int(time.time()),
        "data": [{
            "url": f"http://127.0.0.1:{PORT}/files/{name}",
            "size": f"{out.width}x{out.height}",
            "has_alpha": out.mode == "RGBA" and _has_alpha(out),
        }],
    }


def _localize_prompt(prompt):
    """中文提示词转英文。返回 (实际喂给模型的提示词, 是否翻译过)。

    SDXL 的双 CLIP 只认英文：实测同一 pipeline 同一种子，英文 prompt 出「橘猫坐木地板」，
    中文 prompt 出抽象色块。所以在 SDXL 这条路上必须翻 —— 不能指望用户自己写英文。
    Qwen-Image 走 Qwen3-VL 编码器，本身懂中文，不经过这里。
    """
    try:
        from prompt_l10n import has_cjk, localize  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        log("提示词本地化模块不可用", repr(e))
        return prompt, False

    if not has_cjk(prompt):
        return prompt, False

    d = TRANS_DIR
    try:
        ok = (d / "config.json").is_file()
    except OSError:
        ok = False
    if not ok:
        log(
            f"提示词是中文，但没装翻译模型（{d}）。SDXL 只认英文，"
            "这一张的构图会基本不听提示词 —— 建议补装 opus-mt-zh-en（298 MB）"
        )
        return prompt, False

    out = localize(prompt, d)
    if out == prompt:
        log("中文提示词翻译失败，已按原文提交（画面可能不听提示词）")
        return prompt, False
    log(f"提示词已本地化：{prompt[:24]}… → {out[:60]}")
    return out, True


def _sdxl_size(width, height):
    """SDXL 尺寸归一化：64 对齐 + 上下限。

    SDXL 的 UNet 下采样 64 倍，尺寸必须是 **64 的倍数**（16 的倍数在中间层
    会对不齐，轻则 shape mismatch，重则出雪花）。`_parse_size` 只对到 16，
    这里补到 64 —— 720 → 768 这种轻微变形比直接报错好。

    抽成独立函数是为了能单测：这条链上的尺寸问题（「选了 2K 出 1K」）
    只能靠对账查，不能靠看代码猜。
    """
    def one(v):
        return max(512, min(SDXL_MAX_SIDE, int(round(float(v) / 64.0)) * 64))
    return one(width), one(height)


def _generate_image_sdxl(body, prompt):
    """SDXL-Lightning 出图。四步蒸馏，CFG 关掉。"""
    steps = int(body.get("steps") or SDXL_STEPS)
    # 蒸馏权重超过 8 步只会越来越糊（训练分布就 4 步）。
    steps = max(1, min(8, steps))
    width, height = _parse_size(body.get("size"), body.get("ratio") or "1:1", SDXL_MAX_SIDE)
    raw_w, raw_h = width, height
    width, height = _sdxl_size(width, height)
    # 把「收到什么、实际用什么」原样记下来。用户报「参数没生效」时，
    # 这一行就能区分「没提交上来」还是「提交了但被后端按硬件收窄」。
    log(
        f"SDXL 请求参数 size={body.get('size')!r} ratio={body.get('ratio')!r} "
        f"resolution={body.get('resolution')!r} steps={body.get('steps')!r} "
        f"→ {width}x{height} / {steps} 步"
        + ("" if (raw_w, raw_h) == (width, height) else f"（原 {raw_w}x{raw_h} 已按 64 对齐/上限收窄）")
    )

    en_prompt, translated = _localize_prompt(prompt)

    refs = body.get("image")
    if isinstance(refs, str):
        refs = [refs]
    refs = [r for r in (refs or []) if isinstance(r, str) and r.startswith("data:")]
    refs = refs[:4]

    kw = {
        "prompt": en_prompt,
        "num_inference_steps": steps,
        # Lightning 官方口径是 0：不做无分类器引导。传 1.0 也接近，但 0 才是
        # 它蒸馏时的条件 —— 而且 guidance>1 会让每步多算一次 UNet，慢一倍。
        "guidance_scale": 0.0,
    }

    if refs:
        pipe = _load_sdxl_pipe("edit")
        from PIL import Image  # noqa: PLC0415

        init = _decode_data_url(refs[0]).convert("RGB")
        if init.size != (width, height):
            init = init.resize((width, height), Image.LANCZOS)
        kw["image"] = init
        # 参考图相似度：0.6 是「听提示词但仍保留构图」的常用值。
        kw["strength"] = max(0.05, min(0.95, float(body.get("strength") or 0.6)))
    else:
        pipe = _load_sdxl_pipe("text")
        kw["width"], kw["height"] = width, height

    t0 = time.time()
    out = pipe(**kw).images[0]
    log(f"SDXL 出图 {out.width}x{out.height} / {steps} 步 / {time.time() - t0:.1f}s")
    res = _finish_image(out, body)
    # 把「实际喂了什么」写回响应：用户看到画不对时，第一件要确认的就是
    # 提示词有没有被翻译、翻成了什么。
    res["prompt_used"] = en_prompt
    res["prompt_translated"] = translated
    return res


def _generate_image_locked(body):
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("没有提示词")

    # 低显存档分流。SDXL 与 Qwen-Image 没有任何可复用的部件，必须走各自的链。
    # 判定口径：
    #   显式要 sdxl → SDXL（没装就明确报错，不静默换模型）
    #   没指定      → 装了 SDXL 就用 SDXL（这是本机的默认档）
    #   指定别的    → 走原来的 Qwen 链
    raw_model = str(body.get("model") or "").strip().lower()
    if raw_model in SDXL_ALIASES and not sdxl_dir():
        raise RuntimeError(
            "选了 SDXL-Lightning，但本地没有完整权重。"
            f"把 fp16 checkpoint 放到 {SDXL_DIR}（≥6 GB），或改用 qwen-image-2.1。"
        )
    if raw_model in SDXL_ALIASES or (not raw_model and sdxl_dir()):
        return _generate_image_sdxl(body, prompt)

    model_id = _resolve_repo(body.get("model"), IMAGE_MODEL)
    # 低显存档长边压到 768（见 IMAGE_SMALL_SIDE 注释）：1024² 会顶到 21 GB 峰值
    max_side = IMAGE_SMALL_SIDE if _is_small(model_id) else IMAGE_MAX_SIDE
    width, height = _parse_size(body.get("size"), body.get("ratio") or "1:1", max_side)

    # 低显存档是 4–8 步蒸馏权重（Turbo / Lightning）：按 28 步跑不但慢 4 倍，
    # 还会因为「步数远超训练分布」把画面跑糊。cfg 同理，蒸馏权重只认 1.0。
    small_img = bool(small_image_dir()) and (
        not Path(str(model_id)).is_dir() or _is_small(model_id)
    )
    if small_img and body.get("guidance") in (None, ""):
        body = {**body, "guidance": 1.0}
    steps = int(body.get("steps") or (8 if small_img else IMAGE_STEPS))

    refs = body.get("image")
    if isinstance(refs, str):
        refs = [refs]
    refs = [r for r in (refs or []) if isinstance(r, str) and r.startswith("data:")]
    # 2.1 单次最多吃 10 张参考图；多传的部分直接截断，让请求成功而不是报错
    refs = refs[:10]

    if refs:
        pipe = _load_image_pipe("edit", model_id)
        images = [_decode_data_url(r) for r in refs]
    else:
        pipe = _load_image_pipe("text", model_id)
        images = None

    kw = _img_kwargs(pipe, body, width, height, steps)
    if small_img:
        out = _generate_small(pipe, kw, images)
    elif images:
        try:
            out = pipe(image=images if len(images) > 1 else images[0], **kw).images[0]
        except TypeError:
            # 某些版本只认单图直传，退一步只喂第一张
            out = pipe(image=images[0], **kw).images[0]
    else:
        out = pipe(**kw).images[0]

    return _finish_image(out, body)


def _has_alpha(img):
    try:
        return img.getchannel("A").getextrema()[0] < 255
    except Exception:
        return False


# ============================================================== 视频：MiniMax H3

def _worker_path():
    """视频 worker 的入口文件。

    `init.sh` 的部署流程会把 `.py` 编译成 `.pyc` 再删掉源码（源码保护），
    所以这里**不能写死 `video_worker.py`** —— 线上那个文件根本不存在，
    `Popen` 会直接 `No such file`，表现为「提交成功、任务永远 queued」。
    `.pyc` 可以直接交给 python 执行（CPython 认这个入口）。
    """
    for name in ("video_worker.py", "video_worker.pyc"):
        p = ROOT / name
        if p.is_file():
            return str(p)
    return str(ROOT / "video_worker.py")


def _wan_worker_path():
    """低显存档的 worker（Wan2.2 + GGUF）。判据同 `_worker_path`：源码或 .pyc 都认。"""
    for name in ("wan_worker.py", "wan_worker.pyc"):
        p = ROOT / name
        if p.is_file():
            return str(p)
    return str(ROOT / "wan_worker.py")


def _flash_worker_path():
    """HyperFlow H3 Flash 的 worker。同上：`.py` 优先，`.pyc` 兜底。"""
    for name in ("flash_worker.py", "flash_worker.pyc"):
        p = ROOT / name
        if p.is_file():
            return str(p)
    return str(ROOT / "flash_worker.py")


VIDEO_LONG_SIDE = {"480p": 832, "720p": 1344, "1080p": 1920, "1440p": 2560}

# 尾帧能不能吃，取决于 worker 走哪条路线：
# DiffSynth（NF4，主路线）走 `keyframes=[first, last]`，真支持；
# diffusers 的模块化管线也有 `last_image`，但我们这份 NF4 权重它加载不了。
# 所以「能」——worker 拿到 `--last-frame` 就会用。
VIDEO_LAST_FRAME = True


def _align32(v):
    """视频边长对齐到 32 的倍数。

    Wan 的 patch 是 `2×16×16`，再叠 VAE 16 倍空间压缩 ⇒ 长宽都必须落 32；
    H3 只要 16。取 32 同时满足两者。

    舍入显式写 `int(v/32 + 0.5)`：Python 的 `round()` 是银行家舍入
    （14.5 → 14），JS 的 `Math.round` 是向上取半（14.5 → 15）。以前两端各用各的，
    在 .5 上会错开一档，`832×464` 与 `832×448` 就是这么分叉的。
    """
    return max(256, int(v / 32.0 + 0.5) * 32)


def _video_size(body):
    """提交尺寸：优先客户端给的 `WxH`，其次按 `aspect_ratio` + `resolution` 现算。

    以前这里把 `resolution`（"720p"）当 size 直接交给 worker，
    `norm_size` 解析不了就退回 1344×768 —— 于是「9:16 竖版」出来的是一张横片，
    而且全程不报错。网页端此前只发 aspect_ratio，正好踩在这条路上。

    两条路径都过 `_align32`：客户端已经对齐过一次，这里再对一次是为了
    兜住直接打 HTTP 的调用方（以及手工 curl 的 `size=832x464`）——
    **日志里印的数必须就是模型跑的数**，否则「参数没生效」永远查不清。
    """
    raw = str(body.get("size") or "").strip().lower().replace("*", "x").replace("×", "x")
    parts = raw.split("x")
    if len(parts) == 2:
        try:
            w, h = int(float(parts[0])), int(float(parts[1]))
            if w > 0 and h > 0:
                return f"{_align32(w)}x{_align32(h)}"
        except ValueError:
            pass

    ratio = str(body.get("aspect_ratio") or body.get("ratio") or "16:9").replace("：", ":")
    try:
        rw, rh = (float(x) for x in ratio.split(":"))
    except Exception:
        rw, rh = 16.0, 9.0
    if rw <= 0 or rh <= 0:
        rw, rh = 16.0, 9.0

    long = VIDEO_LONG_SIDE.get(str(body.get("resolution") or "").lower(), 1344)
    f = rw / rh
    w, h = (long, long / f) if f >= 1 else (long * f, long)
    return f"{_align32(w)}x{_align32(h)}"


def submit_video(body):
    """
    提交视频任务。

    和图片生成共用一把串行闸：提交动作本身很快（真正干活的是子进程），
    但它会**卸载图片管线**给视频腾显存 —— 如果此刻正有一张图在采样，
    卸载就等于把它的权重从脚底下抽走。排队 0–90 秒换一个「两边都不失败」，值得。
    """
    with LEASE.image_lock:
        return _submit_video_locked(body)


def _submit_video_locked(body):
    job_id = uuid.uuid4().hex[:12]
    status_file = STATUS / f"{job_id}.json"
    status_file.write_text(
        json.dumps({"status": "queued", "progress": 0, "created": time.time()}), "utf-8"
    )

    # 别名表也现取：`h3` 这类短名要指向**当前可加载的那份**，
    # 而不是启动那一刻 env 写的那个（可能还不存在）。
    _v = video_model()
    if not _video_ready(_v):
        # 直说「权重没齐」比让 diffusers 抛一个认不出目录的错好查得多。
        # 配置检测里那行「下载中 x%」才是用户该看的东西。
        got = _dir_bytes(_v) / 1024 ** 3 if os.path.isdir(_v) else 0
        need = VIDEO_MIN_BYTES / 1024 ** 3
        raise RuntimeError(
            f"视频权重未就绪：{_v or '未找到'} 现有 {got:.1f} GB，需要 {need:.1f} GB"
            "（在「模型配置」里把视频模型下完，或 bash deploy/bootstrap.sh models video）"
        )
    alias = {
        "h3": _v, "minimax": _v, "minimax-h3": _v,
        "minimax-h3-base": _v, "h3-base": _v, "video": _v,
    }
    # HyperFlow 档的别名。画布下拉里写的是 `hyperflow-h3-flash`，
    # 这个名字必须能落到本地那份权重上，否则选了它只会报「权重未就绪」。
    flash_home = flash_dir()
    if flash_home:
        for k in ("hyperflow", "hyperflow-h3-flash", "h3-flash", "h3flash", "flash"):
            alias[k] = flash_home
    # 小模型档的别名：画布下拉里写的是 `wan2.2-ti2v-5b`，
    # 这个名字必须能落到本地那份 GGUF 目录上，否则选了它只会报「权重未就绪」。
    if small_video_dir():
        for k in ("wan2.2-ti2v-5b", "wan2.2", "wan", "wan-ti2v", "ti2v"):
            alias[k] = small_video_dir()
    model = _resolve_repo(body.get("model"), _v or VIDEO_DEFAULT, alias)

    # 走哪条 worker：三条路线的帧规则、加载方式和产物封装都不一样。
    #   flash  —— HyperFlow H3 Flash（INT8 主干 + MiniMax-H3 组件，音画同出）
    #   small  —— Wan2.2-TI2V-5B + GGUF（12 GB 消费卡）
    #   其余   —— H3-NF4 / DiffSynth
    small = _is_small(model)
    flash = _is_flash(model)
    if flash:
        worker = _flash_worker_path()
    elif small:
        worker = _wan_worker_path()
    else:
        worker = _worker_path()
    args = [
        sys.executable, worker,
        "--job", job_id, "--status", str(status_file), "--out", str(OUT),
        "--port", str(PORT), "--model", model,
        "--prompt", body.get("prompt") or "",
        "--duration", str(body.get("duration") or body.get("seconds") or 5),
        "--size", _video_size(body),
        "--backend", str(body.get("backend") or VIDEO_BACKEND),
        "--variant", str(body.get("variant") or "auto"),
        "--steps", str(body.get("steps") or 0),
    ]
    if flash:
        # 组件目录必须显式给：worker 自己扫盘时认的是 env 与 /root/models 下的固定名，
        # 而 www 面板可能把它下到了别处（YJ_FLASH_BASE）。服务端知道就直说。
        args += ["--base-model", flash_ready()[1]]
        if body.get("offload"):
            args += ["--offload", str(body["offload"])]
    if small:
        # 显式给 GGUF 路径：目录里可能同时存在多个量化档，选哪个由服务端决定，
        # 不让 worker 自己猜（猜错就是「加载了 Q2、画面糊成一片」这种无声降质）。
        g = _gguf_in(Path(small_video_dir()))
        if g is not None:
            args += ["--gguf", str(g)]
    if body.get("seed") not in (None, ""):
        args += ["--seed", str(body["seed"])]
    # 同样把收到的规格参数记一行：视频这条链上「时长/分辨率没生效」的投诉，
    # 一半是上游压根没传，另一半是被后端按 121 帧 / 显存上限收窄了。
    log(
        f"视频提交 model={model} size={_video_size(body)} "
        f"resolution={body.get('resolution')!r} ratio={body.get('ratio')!r} "
        f"duration={body.get('duration')!r} steps={body.get('steps')!r} "
        f"backend={body.get('backend')!r}"
    )

    unsupported = []
    tail = body.get("last_frame") or body.get("tail_image")
    if isinstance(tail, str) and tail.startswith("data:"):
        tp = OUT / f"tail_{job_id}.png"
        _decode_data_url(tail).save(tp)
        args += ["--last-frame", str(tp)]
        # Wan2.2-TI2V-5B 这条管线只吃首帧；H3 的 DiffSynth 路线才真支持首尾帧
        if not small and not VIDEO_LAST_FRAME:
            # 收下这张图是为了「日志里能证明客户端确实发过来了」，
            # 但这条链路用不上它 —— 回执里说清楚，客户端会把它挂成一条可见提示。
            unsupported.append("last_frame")
        if small:
            unsupported.append("last_frame")

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

    # 参数被后端硬性收窄时必须回话。
    # Wan 这条链路上限 121 帧（24 fps ≈ 5.04 秒），而画布的时间轨是 1–30 秒连续可选 ——
    # 用户拖到 10 秒、出片只有 5 秒，如果回执不说，现象就是「时长参数没生效」。
    clamped = {}
    if small:
        want_dur = float(body.get("duration") or body.get("seconds") or 5)
        max_dur = 121 / 24.0
        if want_dur > max_dur + 0.01:
            clamped["durationSec"] = {"want": want_dur, "max": round(max_dur, 2)}
    elif flash:
        # HyperFlow 的帧规则是 `17n + 5`，能落在 124（5.17 s）到 362（15.08 s）之间。
        # 请求短于下限会被**抬**到 124 帧（训练下限，少给帧会直接崩结构），
        # 长于上限会被压到 362。两个方向都得回话 —— 用户拖了 3 秒拿回 5.2 秒，
        # 不说清楚就是「时长参数没生效」。
        want_dur = float(body.get("duration") or body.get("seconds") or 5)
        lo, hi = FLASH_MIN_FRAMES / 24.0, FLASH_MAX_FRAMES / 24.0
        if want_dur < lo - 0.01:
            clamped["durationSec"] = {"want": want_dur, "min": round(lo, 2)}
        elif want_dur > hi + 0.01:
            clamped["durationSec"] = {"want": want_dur, "max": round(hi, 2)}
    size_want = str(body.get("size") or "")
    if small and size_want:
        try:
            w, h = [int(float(x)) for x in size_want.lower().replace("*", "x").split("x")]
            long_side = max(w, h)
            # 12 GB 卡上长边超过 1216 就要顶穿显存预算（vram_guard 会直接 OOM）
            if long_side > 1216:
                clamped["resolution"] = {"want": size_want, "max": "1216"}
        except Exception:
            pass
    if flash and size_want:
        try:
            w, h = [int(float(x)) for x in size_want.lower().replace("*", "x").split("x")]
            # 画布是「短边 768 + 32 对齐 + 上限 768×1344」；超出这个盒子的请求
            # 会被 worker 按比例收进来（不是拉伸），所以这里按盒子报回去。
            if min(w, h) > 768 or max(w, h) > 1344:
                clamped["resolution"] = {"want": size_want, "max": "1344x768 (短边 768)"}
        except Exception:
            pass

    return {
        "id": job_id, "status": "queued", "model": model, "created": int(time.time()),
        # 客户端拿它决定「要不要在卡片上留一条提示」：请求里带了、后端用不上的部分
        "unsupported": unsupported,
        # 请求里带了、但被后端按硬件上限收窄的部分
        "clamped": clamped,
    }


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
    """把上游的 SSE 原样泼回去。不要自己解包再拼 —— 中间加一层就会丢 tool_calls 增量。

    这里有三件事必须自己管，缺一件前端就会「说一半卡住一直等」：

    1. **应答体要有结束标记**。SSE 没有 Content-Length，也不用 chunked，
       HTTP/1.1 下客户端只能靠连接关闭判断结束；BaseHTTPRequestHandler 默认
       keep-alive，do_POST 返回后连接不关 ⇒ 客户端收完 [DONE] 还在等下一个字节。
    2. **别先发 200 再连上游**。上游没起来时先发 200 等于把错误藏起来，
       前端只能拿到一个空流。
    3. **读的时候用 read1**。`HTTPResponse.read(n)` 会一直攒够 n 字节才返回，
       SSE 分片只有一两百字节，攒满 4 KB 才出头一次 ⇒ 首字白白慢好几秒。
    """
    req = urllib.request.Request(
        f"{TEXT_UPSTREAM}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        upstream = urllib.request.urlopen(req, timeout=900)
    except Exception as e:  # noqa: BLE001
        # 上游连不上：还能好好回一个 HTTP 错误码，比「200 + 空流」好排障得多
        payload = json.dumps({"error": f"文本服务连不上：{e}"}, ensure_ascii=False).encode("utf-8")
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(payload)
        handler.close_connection = True
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True

    read_some = getattr(upstream, "read1", None) or upstream.read
    try:
        with upstream:
            while True:
                chunk = read_some(4096)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
    except Exception as e:  # noqa: BLE001
        # 连接已经半死，除了补一条 SSE 错误帧也没别的可做
        try:
            payload = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            handler.wfile.write(b"data: " + payload + b"\n\n")
            handler.wfile.flush()
        except Exception:  # noqa: BLE001
            pass


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


def _has_weights(path_str):
    """权重目录能不能被加载器认出来。

    判据取「入口清单文件」，命中任意一个即可：
      · `model_index.json`  —— diffusers 管线（图片全部、H3 官方全量）
      · `configuration.json` —— DiffSynth 的 NF4 量化版（H3 默认下的就是这份，扁平结构）

    只认死 `model_index.json` 的话，H3-NF4 装齐了也会被报成 missing —— 这和以前
    「脚本文件存在性」那类错是同一个毛病：**拿实现细节当能力判据**。

    不在 healthz 里 stat 全部分片 —— 这个端点被 run.sh 每 0.5 秒轮询一次，
    在 webdav 上逐个 stat 几十 G 的分片会把存活探针拖死。
    """
    try:
        p = Path(str(path_str))
        if not p.is_dir():
            return False
        if (p / "model_index.json").is_file() or (p / "configuration.json").is_file():
            # 排除「只有组件、没有主干」的目录。
            #
            # HyperFlow 的组件库就是 MiniMax-H3 原仓（text_encoder / vae / audio_vae …），
            # 我们**没有**下它那个 66 GB 的 transformer。这种目录 73 GB、入口文件也在，
            # 按体积判会一路通过，于是 video 报 ready —— 但交给 DiffSynth 只会
            # `找不到 transformer`。判据要落在「有没有主干」上。
            if (p / "modular_model_index.json").is_file() and not (p / "transformer").is_dir():
                return False
            return True
        # H3 官方 diffusers 仓库：入口在 FL2VA 分区里
        return (p / "FL2VA" / "model_index.json").is_file()
    except Exception:
        return False


# 视频权重可以**在服务跑着的时候**出现在盘上（网页「模型配置」里的下载按钮
# 就是由宿主进程下的，34 GB 落盘要一两个小时）。所以「用哪个目录」不能是
# 启动时算一次就定死的常量：
#   · 以前 VIDEO_MODEL 直接取自 env，而下权重之前根本不存在那个目录；
#   · 更要命的是 run.sh 曾在没装时兜底 export 一条**不存在**的路径，
#     于是下完 34 GB、重启 also 白搭 —— env 会一直把它按回复位到那条空路径。
# 现在每次要用的时候现问一遍：env 指的目录**真能加载**就用它，否则按 run.sh
# 的同一套判据扫盘（系统盘优先，再云盘）。扫到就立刻生效，不用重启服务。
_VIDEO_SCAN = {"at": 0.0, "path": ""}
_VIDEO_SCAN_TTL = 10.0

# 视频权重的**体积门槛**（GB）。
#
# `_has_weights()` 只看入口文件在不在 —— 对「配置级」判断够了，对视频不够：
# H3-NF4 的 `configuration.json` 只有 34 字节，下载器最先落盘的就是它，
# 34 GB 的分片要下一两个小时。这中间按入口文件判，`/healthz` 会一路报
# `caps.video = ready`，用户点「生成视频」才炸 —— 又是一次假绿。
#
# 34.4 GB × 0.85 = 29.2 GB，和宿主 `packStatus()` 的 85% 兜底判据取同一个数，
# 两边才不会出现「面板说齐了、引擎说没有」。
VIDEO_MIN_BYTES = int(float(os.environ.get("YJ_VIDEO_MIN_GB", "29.2")) * 1024 ** 3)


def _dir_bytes(p: str, limit: int = 0) -> int:
    """目录总字节。给了 `limit` 就够数即停，别为了一个健康检查把 34 GB 全走一遍。"""
    total = 0
    for root, _dirs, files in os.walk(p):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
            if limit and total >= limit:
                return total
    return total


def _video_ready(p: str) -> bool:
    """能真的加载：入口文件在，**并且**权重大体齐了。

    低显存小模型档例外：它是 GGUF 量化权重，本来就只有 3–4 GB，
    按 H3 的 29.2 GB 体积门槛判会把「装好了」判成「没装」——
    又是一次「拿别人的规格当自己的判据」。这一档只看组件齐不齐。
    HyperFlow 档同理（主干 16 GB + 组件 63 GB，且分散在两个目录），
    它的判据由 `flash_ready()` 单独负责。
    """
    small = small_video_dir()
    if small and os.path.abspath(str(p)) == os.path.abspath(small):
        return True
    if _is_flash(p):
        return True
    return _has_weights(p) and _dir_bytes(str(p), VIDEO_MIN_BYTES) >= VIDEO_MIN_BYTES


def _vram_gb() -> float:
    """本机最大一张卡的显存（GB）。取不到算 0 —— 判档位时才用，不影响可用性。"""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0.0
        return max(
            torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())
        ) / 1024 ** 3
    except Exception:
        return 0.0


# ============================================================== 低显存小模型档
#
# 消费卡（12 GB）跑不动 H3-NF4（34.4 GB 权重 / 需 30–36 GB 显存），
# 也跑不动 Qwen-Image 的 bf16 全量（33 GB 权重 + 约 22 GB 峰值）。
# 这一档不换模型家族，只换**加载精度**：
#
#   图片  Qwen-Image-2.1 + GGUF Q4 transformer（Q4_K_M 约 4.2 GB）
#   视频  Wan2.2-TI2V-5B + GGUF Q4 transformer（Q4_K_M 约 3.4 GB，文生/图生同一个模型）
#
# 判据仍然是「能不能加载」：目录里要有 diffusers 的 `model_index.json`、
# 各组件的 config，以及至少一个 `.gguf`。**只放一个 gguf 文件不算**。

SMALL_ROOT = Path(os.environ.get("YJ_SMALL_ROOT") or (ROOT.parent / "models"))
IMAGE_SMALL = Path(os.environ.get("YJ_IMAGE_SMALL_DIR") or (SMALL_ROOT / "Qwen-Image-2.1"))
VIDEO_SMALL = Path(os.environ.get("YJ_VIDEO_SMALL_DIR") or (SMALL_ROOT / "Wan2.2-TI2V-5B"))

# ---- 低显存图像档 B：SDXL-Lightning ----
#
# Qwen-Image-2.1 在「12 GB 显存 + 32 GB 内存」这类机器上**功能没坏，但产品上不能用**：
# Qwen3-VL-8B 编码器 bf16 就 17.5 GB，加 DiT 4.2 GB，一加载常驻 22 GB 内存、
# 冷启动 250 秒 —— 实测两次把用户的机器拖到鼠标都动不了。
#
# SDXL-Lightning 是 SDXL base 的 4 步蒸馏版，一个完整 checkpoint 6.6 GB：
#   加载  20 秒级（vs 250 秒）
#   常驻  内存 ~7 GB / 显存 ~6 GB（vs 22 GB / 21 GB）
#   出图  1024² 四步 2-3 秒
# 画质不如 2.1，但这是这类机器上唯一「打开就能用」的档。2.1 留给 64 GB 内存以上的机器。
SDXL_DIR = Path(os.environ.get("YJ_SDXL_DIR") or (SMALL_ROOT / "SDXL-Lightning"))
SDXL_STEPS = int(os.environ.get("YJ_SDXL_STEPS", "4"))
# Lightning 从 SDXL base 蒸馏，原生 1024。往下压到 768 不会更快多少，
# 反而偏离 base 的分桶训练分布（构图塌）。
SDXL_MAX_SIDE = int(os.environ.get("YJ_SDXL_MAX_SIDE", "1024"))
# fp16 完整 checkpoint 6.6 GB。门槛取 6.0 GB：半截文件（3.x GB）不能被判成 ready。
SDXL_MIN_BYTES = int(6.0 * 1024 ** 3)

# 别名 → SDXL 档。前端下拉里传的是给人看的名。
SDXL_ALIASES = ("sdxl-lightning", "sdxl_lightning", "sdxl", "lightning", "sd-xl")

# 中文提示词 → 英文的本地翻译（MarianMT，298 MB）。
# SDXL 的双 CLIP 是纯英文语料训的，中文提示词会退化成噪声（实测：出抽象色块）。
# 装上它，中文提示词在低显存档上才能正常工作。
TRANS_DIR = Path(os.environ.get("YJ_TRANS_DIR") or (SMALL_ROOT / "opus-mt-zh-en"))


def _sdxl_ckpt(d: Path) -> Path | None:
    """挑完整的 SDXL checkpoint —— 跳过 LoRA 那种几百 MB 的小文件。"""
    try:
        cands = sorted(Path(d).glob("*.safetensors"))
    except OSError:
        return None
    for c in cands:
        try:
            if c.stat().st_size >= SDXL_MIN_BYTES:
                return c
        except OSError:
            continue
    return None


def sdxl_dir() -> str:
    """SDXL-Lightning 可加载目录；不可用返回空串（真看文件，不看标记）。"""
    d = SDXL_DIR
    try:
        ok = (d / "model_index.json").is_file() and _sdxl_ckpt(d) is not None
    except OSError:
        ok = False
    return str(d) if ok else ""


def _gguf_in(d: Path) -> Path | None:
    """找一个能用的 GGUF 分片。

    排序不是随便定的：**同目录里先找 Turbo（4 步蒸馏）**，再按量化档。
    蒸馏权重只改采样步数不改架构，Q4_K_M 也能在 12 GB 卡上 4 步出片 ——
    这是低显存档能把「一条 5 秒片」从半小时压到几分钟的关键那一项。
    """
    try:
        cands = sorted(Path(d).glob("*.gguf"))
    except OSError:
        return None
    if not cands:
        return None
    for prefer in ("turbo", "Turbo", "Lightning", "lightning"):
        for c in cands:
            if prefer in c.name and "Q4_K_M" in c.name:
                return c
    for prefer in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"):
        for c in cands:
            if prefer in c.name:
                return c
    return cands[0]


def small_video_dir() -> str:
    d = VIDEO_SMALL
    try:
        ok = (
            (d / "model_index.json").is_file()
            and (d / "transformer" / "config.json").is_file()
            and _gguf_in(d) is not None
        )
    except OSError:
        ok = False
    return str(d) if ok else ""


def small_image_dir() -> str:
    d = IMAGE_SMALL
    try:
        ok = (d / "model_index.json").is_file() and _gguf_in(d) is not None
    except OSError:
        ok = False
    return str(d) if ok else ""


def _is_small(p) -> bool:
    """这个路径是不是小模型档的目录（用来分叉 worker / 参数默认值）。"""
    if not p:
        return False
    ap = os.path.abspath(str(p))
    return any(ap == os.path.abspath(x) for x in (small_video_dir(), small_image_dir()) if x)


# ---------------------------------------------------------------- HyperFlow H3 Flash
#
# 判据是**两处都在**，不是「某个目录够大」：
#   ① 主干：`<FLASH_DIR>/fl2va_int8_convrot/config.json` + 三个分片合计 ≥ 15 GB
#   ② 组件：`<FLASH_BASE>/modular_model_index.json` + text_encoder / vae / audio_vae
# 只看①会在「编码器没下完」时报 ready（上游首次运行才去拉 63 GB），
# 用户点生成、等十分钟、拿到一个 `did not load` —— 又是一次假绿。
def _flash_partition(d: Path) -> Path | None:
    """主干分区：带 config.json 的那一层。允许直接把分区目录当权重根传进来。"""
    try:
        if (d / "config.json").is_file():
            return d
        for sub in ("fl2va_int8_convrot", "fl2va_int8_convrot_comfyui"):
            if (d / sub / "config.json").is_file():
                return d / sub
    except OSError:
        pass
    return None


def _flash_base_ok(d: Path) -> tuple[bool, str]:
    """MiniMax-H3 侧组件齐不齐。返回 (ok, 缺的那件)。"""
    try:
        if not (d / "modular_model_index.json").is_file():
            return False, "modular_model_index.json"
        for part in ("text_encoder", "vae", "audio_vae"):
            if not (d / part).is_dir():
                return False, part
    except OSError as e:
        return False, repr(e)
    return True, ""


def _flash_base_candidates() -> list[Path]:
    """组件目录候选，顺序即优先级（env → 包内 → 同级）。"""
    out: list[Path] = []
    env = os.environ.get("YJ_FLASH_BASE", "").strip()
    if env:
        out.append(Path(env))
    out.append(FLASH_DIR / "MiniMax-H3")
    out.append(FLASH_BASE)
    return out


def flash_ready() -> tuple[str, str]:
    """(主干权重根, 组件目录)。任一不齐就返回空串 —— 真看文件，不看标记。"""
    part = _flash_partition(FLASH_DIR)
    if part is None:
        return "", ""
    if _dir_bytes(str(part), FLASH_MIN_BYTES) < FLASH_MIN_BYTES:
        return "", ""
    for cand in _flash_base_candidates():
        ok, _missing = _flash_base_ok(cand)
        if ok:
            return str(FLASH_DIR), str(cand)
    return "", ""


def flash_dir() -> str:
    return flash_ready()[0]


def _is_flash(p) -> bool:
    """这个路径是不是 HyperFlow 档（用来分叉 worker / 帧规则 / offload 口径）。"""
    if not p:
        return False
    root = flash_dir()
    if not root:
        return False
    ap, ar = os.path.abspath(str(p)), os.path.abspath(root)
    return ap == ar or ap.startswith(ar + os.sep)


def _scan_video_dirs():
    """按 `run.sh pick_video` 的判据扫盘，返回第一个**能加载**的视频权重目录。"""
    for base in (
        os.environ.get("YJ_LOCAL_MODELS", "/root/models"),
        os.environ.get("YJ_CLOUD_MODELS", "/root/cloud/models"),
    ):
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for name in names:
            if not name.startswith("MiniMax-H3"):
                continue
            d = os.path.join(base, name)
            sub = os.path.join(d, "FL2VA")
            if _video_ready(sub):
                return sub
            if _video_ready(d):
                return d
    return ""


def video_model():
    """当前该用的视频权重。

    优先级（每一档都先验「真能加载」）：
      1. env `YJ_VIDEO_MODEL` 指的目录 —— 用户显式指定就照办；
      2. **HyperFlow H3 Flash**（本机显存 ≥ 24 GB 且有这份权重时作默认）；
      3. 低显存小档（Wan2.2 GGUF）；
      4. 现扫一遍盘上的 MiniMax-H3；
      5. 都没有 —— 返回仓库 id，让上层明确报「未就绪」。

    第 2 档压在第 3 档前面是因为「装了哪份就用哪份」在这两档上会打架：
    48 GB 的机器上两份都装得下，此时用户要的是画质和音画同出，不是省显存。
    反过来 12 GB 的机器上 flash 那份本来就装不下也不会被判 ready，
    所以门槛用显存兜一道，而不是靠目录名猜。
    """
    if _video_ready(VIDEO_MODEL):
        return VIDEO_MODEL
    flash = flash_dir()
    if flash and _vram_gb() >= 24:
        return flash
    small = small_video_dir()
    if small:
        return small
    now = time.time()
    if now - _VIDEO_SCAN["at"] > _VIDEO_SCAN_TTL:
        _VIDEO_SCAN["at"] = now
        _VIDEO_SCAN["path"] = _scan_video_dirs()
    return _VIDEO_SCAN["path"] or VIDEO_MODEL


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
                # 图片 / 视频按**默认档的权重目录**判，不看脚本文件在不在。
                #
                # 以前这里写的是 (ROOT / "video_worker.py").exists()，而 init.sh 的
                # 部署流程会把 .py 编译成 .pyc 再删源码 —— 于是线上永远报
                # video: missing，明明 H3 权重齐着、也能出片。用「脚本文件存在性」
                # 判能力本身就是脆的：文件在也不代表权重在。
                # image 以前直接写死 "ready"，同理改成真看目录。
                # 低显存档的目录名不叫 `Qwen/Qwen-Image-2.1`（那是仓库名，
                # 本地落在 models/Qwen-Image-2.1），只看仓库名会永远报 missing ——
                # 明明盘上装着 23 GB 权重。所以两个来源都要判。
                "image": "ready" if (_has_weights(IMAGE_MODEL) or small_image_dir() or sdxl_dir()) else "missing",
                "video": "ready" if _video_ready(video_model()) else "missing",
            }
            return self._json(200, {
                "ok": True, "mode": MODE, "caps": caps, "vram": LEASE.stats(),
                "models": {
                    # 报「真正会跑的那个」：装了 SDXL 就不再声称默认是 2.1。
                    "image": "sdxl-lightning" if sdxl_dir() else IMAGE_MODEL,
                    "image_steps": SDXL_STEPS if sdxl_dir() else IMAGE_STEPS,
                    "image_max_side": SDXL_MAX_SIDE if sdxl_dir() else IMAGE_MAX_SIDE,
                    "image_offload": IMAGE_OFFLOAD,
                    "video": video_model(),
                    "video_default": video_model() or VIDEO_DEFAULT,
                    # 三档视频各自以为准的状态：面板据此显示「装了哪几份 / 缺哪件」，
                    # 不用去猜目录名（以前面板显示的名字和引擎实际读的不是一个）。
                    "video_tiers": {
                        "flash": {
                            "ready": bool(flash_dir()),
                            "weights": flash_dir(),
                            "base": flash_ready()[1],
                            "frames": [FLASH_MIN_FRAMES, FLASH_MAX_FRAMES],
                        },
                        "wan": {"ready": bool(small_video_dir()), "dir": small_video_dir()},
                        "h3": {"ready": _has_weights(VIDEO_MODEL), "dir": VIDEO_MODEL},
                    },
                },
            })

        if path == "/v1/models":
            # 报**画布认的短名**，不报目录路径：路径是这台机器的实现细节，
            # 而 `wan2.2-ti2v-5b` 这种短名才是画布下拉里的选项。
            # 客户端据此决定「这台机器上到底有哪些模型可选」——
            # 列一堆装不上的名字，用户选中后必然失败（以前就是这样）。
            smallv = _is_small(video_model())
            flashv = _is_flash(video_model())
            img_models = []
            # 低显存档排在前面 = 默认选中它。它装了就一定比 2.1 更适合这台机器：
            # 2.1 要 22 GB 内存，装得下不代表跑得动。
            if sdxl_dir():
                img_models.append(
                    {"id": "sdxl-lightning", "object": "model",
                     "owned_by": "local", "kind": "image"}
                )
            if small_image_dir() or _has_weights(IMAGE_MODEL):
                # 这里只暴露**别名**，不要把权重路径当模型名。
                # 之前全量版直接给 IMAGE_MODEL（`/root/models/Qwen-Image-2.1`），
                # 于是画布的模型下拉和生成回执里挂着一串文件系统路径 ——
                # 用户看不懂，换台机器装到别的目录又会变成另一个名字。
                # `_resolve_repo()` 对不带 `/` 的名字会收敛到默认权重，别名照样能出图。
                img_models.append(
                    {"id": "qwen-image-2.1", "object": "model",
                     "owned_by": "local", "kind": "image"}
                )
            return self._json(200, {
                "object": "list",
                "data": [
                    *img_models,
                    {"id": ("hyperflow-h3-flash" if flashv
                            else "wan2.2-ti2v-5b" if smallv
                            else "minimax-h3"),
                     "object": "model", "owned_by": "local", "kind": "video"},
                    {"id": os.environ.get("YJ_TEXT_NAME", "qwen3-8b"),
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

            if path in ("/v1/unload", "/unload"):
                # 手动把权重还给系统。三档由 body 决定：
                #   {"target":"image"}  只退图片（显存 + 内存）
                #   {"target":"gpu"}    图片权重退出显存、内存保留（下次几秒可用）
                #   {"target":"all"}    图片 + 语音一起退
                target = str(body.get("target") or "image").lower()
                did = []
                if target in ("image", "all"):
                    if LEASE.unload_image(why="手动 /v1/unload"):
                        did.append("image")
                if target in ("gpu",):
                    if LEASE.offload_image(why="手动 /v1/unload"):
                        did.append("image:gpu")
                if target == "all":
                    if LEASE._voice_unload():
                        did.append("voice")
                return self._json(200, {
                    "unloaded": ",".join(did) or None,
                    "image": LEASE.image_state(),
                    "vram": LEASE.stats(),
                })

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
    # 先画显存与线程两条硬线，再让任何 torch 张量落地。
    # 不设的话 1024² 出图峰值 21 GB / 卡上 12 GB，Windows 会回落到系统内存，
    # 把整机拖到鼠标都不动 —— 这是必须挡住的事故形态。
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vram_guard import setup as _vram_setup  # noqa: PLC0415

        _VRAM = _vram_setup(log, "model-server")
    except Exception as _e:  # noqa: BLE001
        _VRAM = {}
        log(f"显存保护未启用：{_e!r}")
    log(f"model-server 起在 http://0.0.0.0:{PORT}（mode={MODE}）")
    log(f"产物目录 {OUT}")
    log(f"文本 {TEXT_UPSTREAM} / 语音 {VOICE_BASE}")
    # 报**实际生效**的档位，不是全量档的常量：低显存档走 768 / 8 步 / GGUF，
    # 照常量印 1536 / 28 步会让人以为配错了（面板与日志口径必须与真跑的一致）。
    if sdxl_dir():
        _ck = _sdxl_ckpt(SDXL_DIR)
        log(
            f"图片 SDXL-Lightning（低显存档 {SDXL_STEPS} 步 / 长边 {SDXL_MAX_SIDE}px / "
            f"{_ck.name if _ck else '?'} / offload={IMAGE_OFFLOAD}）"
        )
        if small_image_dir() or _has_weights(IMAGE_MODEL):
            log(
                f"     Qwen-Image 2.1 也可用（{IMAGE_STEPS} 步 / 长边 {IMAGE_MAX_SIDE}px），"
                "但它常驻 ~22 GB 内存，只有显式选它才走"
            )
    else:
        _small = _is_small(IMAGE_MODEL) or bool(small_image_dir())
        _side = IMAGE_SMALL_SIDE if _small else IMAGE_MAX_SIDE
        log(
            f"图片 {IMAGE_MODEL}"
            f"（{'低显存档 ' if _small else ''}{'8' if _small else IMAGE_STEPS} 步 / "
            f"长边 {_side}px / offload={IMAGE_OFFLOAD}）"
        )
    log(f"视频 {video_model() or '未安装（权重装了会自动生效，不用重启）'}（backend={VIDEO_BACKEND}）")
    _fdir = flash_dir()
    if _fdir:
        log(f"     HyperFlow H3 Flash 可用（默认档 / 6 步 / 音画同出 / 124–362 帧 / 组件 {flash_ready()[1]}）")
    _sdir = small_video_dir()
    if _sdir and not _is_flash(video_model()):
        log(f"     Wan2.2-TI2V-5B 低显存档也可用（{_sdir}）")
    LEASE.start_idle_watcher()
    _warm_translator()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("收到中断，退出")
        srv.shutdown()


if __name__ == "__main__":
    main()
