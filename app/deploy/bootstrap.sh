#!/usr/bin/env bash
# ============================================================================
# 艺镜 Studio · 仙宫云一键部署
#
# 目标：开机（或第一次部署后）跑一次，浏览器打开平台给的端口地址就能用画布，
#       不需要客户端、不需要 ComfyUI。
#
# 用法：
#   bash deploy/bootstrap.sh install            # 装环境（只跑一次，之后保存镜像）
#   bash deploy/bootstrap.sh install --with-models   # 连 torch/diffusers 一起装（大，约 8 GB）
#   bash deploy/bootstrap.sh start|stop|status  # 启停 / 看状态
#
# 设计原则：
#   - **幂等**：重复跑不会装第二遍，也不会覆盖已有的 .env。
#   - **不假设路径**：仙宫云不同镜像的持久盘位置不一样（/data、/workspace、/root/xgpu-tmp
#     都见过），这里按可用空间自动挑，挑完写进 .env，后续都用同一个。
#   - **缺什么说什么**：没有 systemd 就退回 nohup + pid 文件；没有 GPU 就只起界面，
#     生成能力在 /healthz 里如实标 down，而不是假装能用。
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/deploy/.env"
RUN_DIR="$ROOT/.run"
LOG_DIR="$ROOT/.logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_END=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✓%s %s\n' "$C_OK" "$C_END" "$*"; }
warn() { printf '%s!%s %s\n' "$C_WARN" "$C_END" "$*"; }
die()  { printf '%s✗%s %s\n' "$C_ERR" "$C_END" "$*" >&2; exit 1; }
dim()  { printf '%s%s%s\n' "$C_DIM" "$*" "$C_END"; }

# ---------------------------------------------------------------- 环境探测

detect_data_dir() {
  # 已经配过就直接用，不要每次开机换目录（换了等于项目全丢）
  if [ -f "$ENV_FILE" ]; then
    local cfg
    cfg="$(grep -E '^YJ_DATA=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)"
    if [ -n "${cfg:-}" ]; then printf '%s' "$cfg"; return; fi
  fi
  # 候选按「持久盘优先级」排：容器重建后这些通常还活着。
  # /root/cloud 是仙宫云官方的「云储存」目录——它不进镜像、重开实例还在，
  # 所以模型放这儿最稳（20 GB 的系统镜像不用被模型撑爆）。
  # 其余是各家平台的惯例挂载点，留着不吃亏。
  for d in /root/cloud /root/xgpu-tmp /data /workspace /root/autodl-tmp /mnt/data; do
    if [ -d "$d" ] && [ -w "$d" ]; then
      printf '%s/yjai' "$d"; return
    fi
  done
  printf '%s' "$HOME/yjai"
}

detect_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'none|0|0'; return
  fi
  local line
  line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [ -z "$line" ]; then printf 'none|0|0'; return; fi
  local name mem
  name="$(printf '%s' "$line" | cut -d, -f1 | sed 's/^ *//;s/ *$//')"
  mem="$(printf '%s' "$line" | cut -d, -f2 | tr -dc '0-9')"
  printf '%s|%s|%s' "$name" "${mem:-0}" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
}

need_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_node() {
  if need_cmd node; then
    local v; v="$(node -v)"
    if [ "${v#v}" \< "18" ]; then warn "node $v 太旧，装 20"; else ok "node $v"; return; fi
  fi
  say "安装 Node 20（宿主需要，约 40 MB）..."
  local arch; arch="$(uname -m)"
  case "$arch" in
    x86_64) arch="x64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) die "不认识的架构：$arch" ;;
  esac
  local ver="v20.19.0"
  local url="https://nodejs.org/dist/$ver/node-$ver-linux-$arch.tar.xz"
  mkdir -p /opt/node
  curl -fsSL "$url" -o /tmp/node.tar.xz || die "下载 Node 失败（容器没网？先开联网功能）"
  tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1
  ln -sf /opt/node/bin/node /usr/local/bin/node
  ln -sf /opt/node/bin/npm /usr/local/bin/npm
  ok "node $(node -v)"
}

ensure_ffmpeg() {
  if need_cmd ffmpeg; then ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"; return; fi
  say "安装 ffmpeg（抽首帧 / 拼音频 / 导出都要它）..."
  if need_cmd apt-get; then
    (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1 || die "ffmpeg 装失败"
    ok "ffmpeg 已装"
  else
    die "没有 apt，请手动装 ffmpeg 后重试"
  fi
}

ensure_python() {
  if need_cmd python3; then ok "python $(python3 -V 2>&1 | awk '{print $2}')"; else die "没有 python3"; fi
  # 镜像选得对不对，这一条最直接：自带的 torch 能不能看见卡。
  # 不能也不一定失败（我们会在 .venv 里自己装），但说明这个镜像的 CUDA 栈有问题，
  # 早点知道比装完 8 GB 依赖后才发现要划算。
  local tv
  tv="$(python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || true)"
  if [ -n "$tv" ]; then
    case "$tv" in
      *True)
        ok "镜像自带 torch $tv"
        # 能看见卡就复用它：仙宫云的 pytorch 2.5.1 镜像已经把 cu124 的 torch 装好了，
        # 再下一遍是 3 GB 白花。--system-site-packages 让 .venv 能看见它，
        # 而 venv 里 pip 装的新包优先级更高，diffusers / transformers 照样能升级。
        YJ_REUSE_TORCH=1
        ;;
      *)     warn "镜像自带 torch 是 CPU 版或看不见 GPU（$tv）—— 会装独立 .venv 的 cu124 版，
       但如果是 CUDA 版本不匹配，建议换一个 PyTorch 基础镜像重开" ;;
    esac
  else
    dim "  镜像里没有 torch，没关系：.venv 会自己装（多花 3–5 分钟）"
  fi
  if [ ! -d "$ROOT/.venv" ]; then
    say "建虚拟环境 .venv ..."
    if [ "${YJ_REUSE_TORCH:-0}" = "1" ] && [ "${YJ_FRESH_TORCH:-0}" != "1" ]; then
      python3 -m venv --system-site-packages "$ROOT/.venv"
      dim "  复用了镜像里的 torch（想强制自己装一遍：YJ_FRESH_TORCH=1 bash ...）"
    else
      python3 -m venv "$ROOT/.venv"
    fi
  fi
  # 用国内源：仙宫云出口带宽一般，pypi 官方源装 torch 会慢到以为卡死
  "$ROOT/.venv/bin/pip" install -q --upgrade pip >/dev/null
  ok "虚拟环境 $ROOT/.venv"
}

# ---------------------------------------------------------------- 安装

cmd_install() {
  local with_models=0 with_voice=0
  for a in "$@"; do
    case "$a" in
      --with-models) with_models=1 ;;
      --with-voice) with_voice=1 ;;
    esac
  done

  say ""
  say "════ 艺镜 Studio · 环境安装 ════"
  local gpu; gpu="$(detect_gpu)"
  IFS='|' read -r gname gmem gdrv <<<"$gpu"
  if [ "$gname" = "none" ]; then
    warn "没有探到 GPU（无卡模式？）—— 界面能开，生成能力会是 down"
  else
    ok "GPU $gname / ${gmem} MiB / 驱动 $gdrv"
    if [ "${gmem:-0}" -lt 40000 ] 2>/dev/null; then
      warn "显存 ${gmem} MiB 偏小：Qwen-Image-2.1 开 offload 能跑，但 MiniMax H3 要 40 GB 以上，"
      warn "  视频会 OOM。先用 YJ_MODE=image 把图片链路跑通，或换 48G 卡。"
    fi
  fi

  ensure_node
  ensure_ffmpeg
  ensure_python

  local data; data="$(detect_data_dir)"
  mkdir -p "$data/projects" "$data/assets" "$data/thumbs" "$data/model"
  ok "数据目录 $data"

  # ---- .env：只在第一次生成，之后不覆盖（用户可能手调过模型名）
  if [ ! -f "$ENV_FILE" ]; then
    cat >"$ENV_FILE" <<EOF
# 艺镜 Studio · 部署配置（bootstrap 生成，可手改）
# 数据目录：项目、资产、模型都放这里。容器重建后要保证这个路径还在。
YJ_DATA=$data

# 界面端口：仙宫云的「自定义服务 / 端口映射」填这个
YJ_PORT=6006

# 模型服务端口（只在本机，不对外）
YJ_MODEL_PORT=3911

# 只开图片（显存紧时），可选 full / image / text
YJ_MODE=full

# 文本后端：vLLM 默认 8001，llama.cpp server 一般 8080
YJ_TEXT_UPSTREAM=http://127.0.0.1:8001/v1
YJ_TEXT_MODEL=Qwen3-8B-Instruct

# ---- 图片：Qwen-Image-2.1（7B 单流 DiT；文生图 + 编辑一体，原生 RGBA 透明，最多 10 图参考）
# 完整权重 33 GB（其中文本编码器 Qwen3-VL-8B 占 17.5 GB）——这也是必须开 offload 的原因
YJ_IMAGE_MODEL=Qwen/Qwen-Image-2.1
# 官方标准 40 步；28 步在 2K 下肉眼难分，快 30%
YJ_IMAGE_STEPS=28
# 长边上限；改 2752 才是原生 2K（慢 3-4 倍，显存多 4 倍）
YJ_IMAGE_MAX_SIDE=1536
# 必须为 1：把文本编码器用完挪回内存，峰值 40G→22G
YJ_IMAGE_OFFLOAD=1

# ---- 视频：MiniMax H3-Base（768p / 24fps / 原生立体声，音画一起出）
# 产品上只支持这一个视频后端，不再有第二档快模型
YJ_VIDEO_MODEL=MiniMaxAI/MiniMax-H3
# auto | diffusers | diffsynth | sglang | cmd
YJ_VIDEO_BACKEND=auto
# 48G 卡也建议开：H3 主干 + 32B 文本编码器全量放不下
YJ_VIDEO_OFFLOAD=1
# 单条 H3 可能 8-15 分钟，旧的 25 分钟上限已不够用
YJ_VIDEO_TIMEOUT_MIN=120

# 语音引擎（vendor/voice，端口 3910）：Qwen3-TTS-1.7B 三件套
# custom=预设音色+风格指令 / base=3 秒克隆 / design=文字造音色
YJ_VOICE_BASE=http://127.0.0.1:3910

# 可选：给 /api 和 /files 加一道 Bearer。平台端口本身已带 token，可不设。
# YJ_TOKEN=
EOF
    ok "写了 $ENV_FILE"
  else
    dim "  $ENV_FILE 已存在，保留不动"
  fi

  # ---- 前端产物：优先用仓库里已经构建好的
  if [ ! -f "$ROOT/dist-web/index.html" ]; then
    say "构建网页版（dist-web）..."
    if [ -d "$ROOT/node_modules" ]; then
      node "$ROOT/node_modules/vite/bin/vite.js" build --config vite.web.config.ts >/dev/null
      ok "构建完成"
    else
      warn "没有 node_modules，跳过构建。请在本地跑 npm run build:web 后把 dist-web 一起上传"
    fi
  else
    ok "网页产物已就绪（dist-web）"
  fi

  # ---- 模型依赖
  if [ "$with_models" = "1" ]; then
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    # torch 只在需要时装：镜像自带可用的 cu124 torch 就跳过，省 3 GB 下载与 5 分钟。
    # 注意这里要「看得见 GPU」才算可用 —— CPU 版复用了等于白干。
    if "$ROOT/.venv/bin/python" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
      ok "跳过 torch：.venv 已能 import 到可用的 $( "$ROOT/.venv/bin/python" -c 'import torch;print(torch.__version__)' )"
    else
      say "安装 torch（cu124，约 3 GB）..."
      "$ROOT/.venv/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
    fi
    # diffusers 必须够新：Qwen-Image-2.1 的 pipeline 是 2026-09 才合进去的，
    # 老版本 from_pretrained 一个 7B 单流 DiT 会直接报「不认识的架构」。
    say "安装 diffusers / transformers ..."
    "$ROOT/.venv/bin/pip" install -q -U "diffusers>=0.36" "transformers>=4.60" \
      accelerate safetensors pillow imageio imageio-ffmpeg ftfy sentencepiece protobuf \
      soundfile numpy "huggingface_hub[cli,hf_transfer]" scipy
    # NF4 / INT8 量化是可选的：只有显存吃紧或想再快一点时才需要
    "$ROOT/.venv/bin/pip" install -q bitsandbytes torchao 2>/dev/null || \
      warn "bitsandbytes / torchao 没装上（不影响 bf16 主路径）"
    ok "推理依赖就绪（diffusers $("$ROOT/.venv/bin/python" -c 'import diffusers;print(diffusers.__version__)' 2>/dev/null || echo '?')）"

    # 语音引擎：qwen-tts 与 diffusers 对 transformers 的版本要求是**互相矛盾**的
    # （diffusers 0.41 要 5.x，qwen-tts 要 4.x，细节见 voice-server/server.py 顶部）。
    #
    # 所以 qwen-tts 一律用 --no-deps 装，另把 4.x 那份 transformers 放到
    # pylibs/tf4 由语音进程自己 sys.path.insert(0, ...)。
    # 不加 --no-deps 的后果是实测过的：pip 会把主环境的 transformers 降到 4.x，
    # 图片和视频立刻报 cannot import name 'resolve_revision'；
    # 反过来把主环境修好，语音又报 check_model_inputs() missing an argument。
    if [ ! -d "$ROOT/pylibs/tf4" ]; then
      say "安装语音依赖（transformers 4.x 隔离副本，约 240 MB）..."
      "$ROOT/.venv/bin/pip" install -q --target "$ROOT/pylibs/tf4" --no-deps \
        transformers==4.57.3 huggingface_hub==0.36.2 tokenizers==0.22.2 \
        safetensors regex requests filelock tqdm packaging pyyaml numpy 2>/dev/null || \
        warn "隔离副本没装上：语音引擎会不可用（界面会显示 voice: down）"
    fi
    "$ROOT/.venv/bin/pip" install -q --no-deps qwen-tts 2>/dev/null || \
      warn "qwen-tts 没装上（语音引擎会不可用，界面会如实显示 voice: down）"
    # 逐个点名而不是让 pip 解 qwen-tts 的依赖树：那条树里有 gradio，
    # 它会要求 huggingface-hub<2.0，正好把 diffusers 需要的新版踢掉。
    "$ROOT/.venv/bin/pip" install -q accelerate einops librosa onnxruntime \
      torchaudio sherpa-onnx fastapi uvicorn python-multipart 2>/dev/null || \
      warn "语音运行时依赖没装全（合成或转录可能失败）"

    # DiffSynth：H3 的 NF4 路线（ModelScope 官方推荐），装不上就退回 diffusers/SGLang
    if [ -d "$ROOT/vendor/DiffSynth-Studio" ]; then
      "$ROOT/.venv/bin/pip" install -q -e "$ROOT/vendor/DiffSynth-Studio[quant]" 2>/dev/null && ok "DiffSynth 已装"
    else
      dim "  想要 H3 的 NF4 路由：git clone https://github.com/modelscope/DiffSynth-Studio vendor/DiffSynth-Studio"
    fi
  else
    warn "跳过模型依赖（加 --with-models 才装）。只装界面：生成会报「没装 torch / diffusers」"
  fi

  # ---- 语音引擎
  if [ "$with_voice" = "1" ]; then
    if [ -f "$ROOT/vendor/voice/app/requirements.txt" ]; then
      say "安装语音引擎依赖（Qwen3-TTS + SenseVoice）..."
      "$ROOT/.venv/bin/pip" install -q -r "$ROOT/vendor/voice/app/requirements.txt"
      ok "语音依赖就绪（模型仍需按 README 放 vendor/voice/data/models）"
    else
      warn "找不到 vendor/voice/app/requirements.txt"
    fi
  fi

  cmd_status || true
  say ""
  ok "安装完成。下一步：bash deploy/bootstrap.sh start"
}

# ---------------------------------------------------------------- 模型下载

cmd_models() {
  load_env
  local want="${1:-auto}"
  # auto = 按这台机器**实际显存**决定下哪些（deploy/vram-plan.sh）。
  # 视频只有 H3 一个后端，默认那份是 NF4 量化版（只收 FL2VA 四件 = 34.4 GB，
  # 官方口径最低 8 G 显存就能跑）。bf16 全量要 125 G 显存，`all` 里绝不该带上它。
  if [ "$want" = "auto" ]; then
    if [ -x "$ROOT/deploy/vram-plan.sh" ] || [ -f "$ROOT/deploy/vram-plan.sh" ]; then
      want="$(bash "$ROOT/deploy/vram-plan.sh" --want 2>/dev/null | tr -d '\r')"
    fi
    [ -n "${want:-}" ] || want="all"
    say "  按显存自动选择模型：$want"
  fi
  local py="$ROOT/.venv/bin/python"
  [ -x "$py" ] || die "还没有 .venv，先跑：bash deploy/bootstrap.sh install --with-models"

  # 国内直连 huggingface.co 基本下不动；镜像 + hf_transfer 是两个数量级的差别
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  local dir="$YJ_DATA/models"
  mkdir -p "$dir"

  say ""
  say "════ 下载模型到 $dir（HF 镜像 $HF_ENDPOINT）════"
  say "  图片  Qwen-Image-2.1                ~33 GB"
  say "  视频  MiniMax-H3 NF4（默认）        ~34 GB（FL2VA 四件；8 G 显存起；唯一视频后端）"
  say "        官方 bf16 全量要显式点名：models video-full   ~171 GB / 125 GB 显存"
  say "  文本  Qwen3-8B-Instruct            ~16 GB"
  say "  语音  Qwen3-TTS-1.7B × 3           ~13 GB"
  say "  ─────────────────────────────────────────────"
  say "  该装哪些由显存决定（deploy/vram-plan.sh）；视频只有 H3，NF4 是默认那份。"

  YJ_WANT="$want" YJ_MODEL_DIR="$dir" YJ_VOICE_DIR="$ROOT/vendor/voice/data/models" \
  "$py" - <<'PY'
import os
import sys

want = os.environ.get("YJ_WANT", "all")
# want 允许三种形态：all / 单个 key / 空格或逗号分隔的多个 key。
# 多个 key 是给「按显存自动选择」用的（见 deploy/vram-plan.sh）。
_wants = None if want.strip() == "all" else set(want.replace(",", " ").split())

def take(key):
    # `video-full`（官方 bf16 全量 171 GB）**只能点名装**：
    # 它要 125 GB 显存，`all` 里带上它等于让每台机器白下 171 GB。
    if key == "video-full":
        return _wants is not None and key in _wants
    return _wants is None or key in _wants
root = os.environ["YJ_MODEL_DIR"]
vodir = os.environ["YJ_VOICE_DIR"]

# (仓库, 目标子目录, 说明, 只下这些路径)。最后一列为 None 表示整仓都下。
#
# 视频默认下 **DiffSynth 的 NF4 量化版**，不是官方 bf16 全量（171 GB / 125 GB 显存）。
# NF4 仓库里其实躺着 5 个大分片（fl2va / ref2va / 各自的 pruned 版），
# 我们只要 FL2VA 这一路的 4 个 —— 全收是 72.5 GB，收白名单是 34.4 GB。
H3_NF4_FILES = [
    "configuration.json",
    "minimax-h3-fl2va-nf4.safetensors",
    "minimax-h3-text-encoder-nf4.safetensors",
    "video_vae_nf4.safetensors",
    "audio_vae_nf4.safetensors",
]
PLAN = [
    ("image",      "Qwen/Qwen-Image-2.1",              None,               "图片 · Qwen-Image-2.1", None),
    ("video",      "DiffSynth-Studio/MiniMax-H3-NF4",  "MiniMax-H3-NF4",   "视频 · MiniMax H3 NF4（唯一视频后端）", H3_NF4_FILES),
    # 分词器不在 NF4 仓里，只有 11 MB，但少了它 pipeline 拿不到 prompt 的 token
    ("video",      "MiniMaxAI/MiniMax-H3",             "MiniMax-H3-NF4",   "视频 · H3 分词器（11 MB）", ["FL2VA/processor/*"]),
    ("video-full", "MiniMaxAI/MiniMax-H3",             None,               "视频 · MiniMax H3 官方 bf16 全量（要 125 GB 显存）", None),
    ("text",       "Qwen/Qwen3-8B-Instruct",           None,               "文本 · Qwen3-8B", None),
]

VOICE = [
    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen-tts-custom", "语音 · 预设音色 + 风格指令"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",        "qwen-tts-base",   "语音 · 3 秒克隆"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "qwen-tts-design", "语音 · 文字造音色"),
    ("Qwen/Qwen3-TTS-Tokenizer-12Hz",        "qwen-tts-tokenizer", "语音 · 编解码器（必需）"),
]

def download(repo, dest, label, only=None):
    from huggingface_hub import snapshot_download
    print(f"\n── {label}\n   {repo} → {dest}", flush=True)
    try:
        p = snapshot_download(
            repo_id=repo,
            local_dir=dest,
            allow_patterns=only,      # None = 整仓
        )
        print(f"   ✓ 就绪 {p}", flush=True)
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"   ✗ 失败 {msg[:300]}", flush=True)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            print("     → 这个仓库要在网页上先同意许可，再 `hf auth login`（填 Access Token）", flush=True)
        return False

ok = fail = 0
for key, repo, sub, label, only in PLAN:
    if not take(key):
        continue
    dest = os.path.join(root, sub or repo.split("/")[-1])
    if download(repo, dest, label, only):
        ok += 1
    else:
        fail += 1

# 分词器是按仓库内的 FL2VA/processor/* 取下来的，落盘会带上那层目录；
# DiffSynth 认的是包目录下的 processor/，所以搬平一层，顺手清掉空壳。
for base in (os.path.join(root, "MiniMax-H3-NF4"),):
    src = os.path.join(base, "FL2VA", "processor")
    dst = os.path.join(base, "processor")
    if os.path.isdir(src) and not os.path.isdir(dst):
        os.rename(src, dst)
        try:
            os.rmdir(os.path.join(base, "FL2VA"))
        except OSError:
            pass
        print(f"   ✓ 分词器已归位 {dst}", flush=True)

if take("voice"):
    for repo, sub, label in VOICE:
        if download(repo, os.path.join(vodir, sub), label):
            ok += 1
        else:
            fail += 1

print(f"\n完成：{ok} 个成功 / {fail} 个失败")
sys.exit(1 if fail else 0)
PY
  local rc=$?
  if [ "$rc" = "0" ]; then
    ok "模型下载完成。Vram 账本：图片 22G（共享）/ 视频 H3-NF4 34G（独占）"
  else
    warn "有模型没下全。失败原因看上面；下完重跑：bash deploy/bootstrap.sh models $want"
  fi
  return 0
}

# ---------------------------------------------------------------- 进程管理

pid_of() { local f="$RUN_DIR/$1.pid"; [ -f "$f" ] && cat "$f" 2>/dev/null || true; }
alive()  { local p; p="$(pid_of "$1")"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

start_one() {
  local name="$1"; shift
  if alive "$name"; then ok "$name 已在跑（pid $(pid_of "$name")）"; return; fi
  nohup "$@" >>"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$RUN_DIR/$name.pid"
  sleep 1
  if alive "$name"; then ok "$name 启动（pid $(pid_of "$name")）"; else
    warn "$name 启动失败，看 $LOG_DIR/$name.log"
    tail -n 8 "$LOG_DIR/$name.log" || true
  fi
}

stop_one() {
  local name="$1" p
  p="$(pid_of "$name")"
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
    kill "$p" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$p" 2>/dev/null || break; sleep 0.4; done
    kill -9 "$p" 2>/dev/null || true
    ok "$name 已停"
  else
    dim "  $name 没在跑"
  fi
  rm -f "$RUN_DIR/$name.pid"
}

load_env() {
  [ -f "$ENV_FILE" ] || die "还没有配置，先跑：bash deploy/bootstrap.sh install"
  set -a; . "$ENV_FILE"; set +a
  : "${YJ_PORT:=6006}"; : "${YJ_MODEL_PORT:=3911}"; : "${YJ_DATA:=$HOME/yjai}"
}

cmd_start() {
  load_env
  say ""
  say "════ 启动 ════"

  # token 落一个本地文件，方便本机 curl 自检时带上
  local token="${YJ_TOKEN:-}"

  # 1) 语音引擎（先起它：它是唯一「不给就能用」的模态，冷启动也要 4-8 秒）
  local py="$ROOT/.venv/bin/python"
  [ -x "$py" ] || { warn "还没有 .venv，退回系统 python（先跑 install 更稳）"; py="$(command -v python3)"; }
  if [ -f "$ROOT/vendor/voice/app/server.py" ]; then
    YJ_VOICE_PORT="$(printf '%s' "${YJ_VOICE_BASE:-http://127.0.0.1:3910}" | sed 's|.*:||;s|/.*||')" \
    YJ_VOICE_DATA="$ROOT/vendor/voice/data" \
    start_one voice "$py" "$ROOT/vendor/voice/app/server.py"
  else
    warn "没有 vendor/voice/app/server.py，语音能力会是 down"
  fi

  # 2) 模型服务
  YJ_PORT="$YJ_MODEL_PORT" YJ_MODEL_DATA="$YJ_DATA/model" \
  YJ_TEXT_UPSTREAM="${YJ_TEXT_UPSTREAM:-http://127.0.0.1:8001/v1}" \
  YJ_TEXT_MODEL="${YJ_TEXT_MODEL:-Qwen3-8B-Instruct}" \
  YJ_VOICE_BASE="${YJ_VOICE_BASE:-http://127.0.0.1:3910}" \
  YJ_IMAGE_MODEL="${YJ_IMAGE_MODEL:-Qwen/Qwen-Image-2.1}" \
  YJ_IMAGE_STEPS="${YJ_IMAGE_STEPS:-28}" \
  YJ_IMAGE_MAX_SIDE="${YJ_IMAGE_MAX_SIDE:-1536}" \
  YJ_IMAGE_OFFLOAD="${YJ_IMAGE_OFFLOAD:-1}" \
  YJ_VIDEO_MODEL="${YJ_VIDEO_MODEL:-MiniMaxAI/MiniMax-H3}" \
  YJ_VIDEO_BACKEND="${YJ_VIDEO_BACKEND:-auto}" \
  YJ_VIDEO_OFFLOAD="${YJ_VIDEO_OFFLOAD:-1}" \
  YJ_MODE="${YJ_MODE:-full}" \
  start_one model "$py" "$ROOT/deploy/model-server.py"

  # 3) 画布宿主
  PORT="$YJ_PORT" YJ_DATA="$YJ_DATA" YJ_WEB="$ROOT/dist-web" \
  YJ_MODEL_BASE="http://127.0.0.1:$YJ_MODEL_PORT/v1" \
  YJ_VOICE_BASE="${YJ_VOICE_BASE:-http://127.0.0.1:3910}" \
  YJ_TOKEN="$token" YJ_VIDEO_TIMEOUT_MIN="${YJ_VIDEO_TIMEOUT_MIN:-90}" \
  start_one host node "$ROOT/web/host.mjs"

  sleep 1
  cmd_status
}

cmd_stop() {
  load_env
  stop_one host
  stop_one model
  stop_one voice
}

cmd_status() {
  load_env 2>/dev/null || true
  local p="${YJ_PORT:-6006}" mp="${YJ_MODEL_PORT:-3911}"
  say ""
  say "── 状态 ──"
  if alive host; then ok "画布宿主  http://127.0.0.1:$p  (pid $(pid_of host))"; else warn "画布宿主 未运行"; fi
  if alive model; then ok "模型服务  http://127.0.0.1:$mp/v1  (pid $(pid_of model))"; else warn "模型服务 未运行"; fi
  if alive voice; then ok "语音引擎  ${YJ_VOICE_BASE:-http://127.0.0.1:3910}  (pid $(pid_of voice))"; else warn "语音引擎 未运行"; fi

  # 显存账本：直接问驱动，比任何估算都准
  if command -v nvidia-smi >/dev/null 2>&1; then
    local used total
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    [ -n "${used:-}" ] && dim "  显存 ${used} / ${total} MiB 已用"
  fi

  local hz
  hz="$(curl -fsS -m 4 "http://127.0.0.1:$p/healthz" 2>/dev/null || true)"
  if [ -n "$hz" ]; then
    say ""
    dim "  /healthz → ${hz:0:400}"
  fi

  # 给最直观的一条：平台端口映射该填什么
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  say ""
  say "  平台端口映射 → $p"
  dim "  本机自测：curl http://127.0.0.1:$p/healthz"
  [ -n "$ip" ] && dim "  内网地址：http://$ip:$p"
}

# ---------------------------------------------------------------- 开机自启

cmd_autostart() {
  load_env
  local unit="/etc/systemd/system/yjai-studio.service"
  if ! need_cmd systemctl || [ ! -d /run/systemd/system ]; then
    warn "这个容器没有 systemd。两种做法："
    dim "  1) 仙宫云支持「启动脚本」的话，填：bash $ROOT/deploy/bootstrap.sh start"
    dim "  2) 或者把环境保存成自定义镜像，开机后手动跑一次上面这行"
    return
  fi
  cat >"$unit" <<EOF
[Unit]
Description=YiJing Studio (canvas + local models)
After=network-online.target

[Service]
Type=forking
WorkingDirectory=$ROOT
ExecStart=/bin/bash $ROOT/deploy/bootstrap.sh start
ExecStop=/bin/bash $ROOT/deploy/bootstrap.sh stop
RemainAfterExit=yes
Restart=no

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable yjai-studio.service >/dev/null 2>&1 || true
  ok "已注册开机自启（systemctl start yjai-studio 可手动启）"
}

# ---------------------------------------------------------------- 入口

case "${1:-help}" in
  install) shift; cmd_install "$@" ;;
  models)  shift; cmd_models "$@" ;;
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  autostart) cmd_autostart ;;
  *)
    say "艺镜 Studio · 部署脚本"
    say ""
    say "  install [--with-models] [--with-voice]   装环境"
    say "  models [all|image|video|text|voice]   预下载模型（按显存自动取舍，别用 all）"
    say "  models video-full                    只在 ≥125 GB 显存的卡上用（官方 bf16 全量，171 GB）"
    say "  start | stop | restart | status          启停 / 看状态"
    say "  autostart                                注册 systemd 开机自启（有 systemd 才有效）"
    ;;
esac
