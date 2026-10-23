#!/usr/bin/env bash
# 仙宫云一体机：单独启停模型服务与画布宿主（不走 bootstrap 的完整流程）。
#
#   bash deploy/run.sh start|stop|restart|status|log [model|host|chat|voice]
#
# 四个服务各自独立，一个挂了不影响另一个：
#   host  6006  静态页面 + /api/rpc + /files + SSE
#   model 3911  四个模态的 OpenAI 兼容接口（懒加载，第一次请求才占显存）
#   chat  8001  文本（Qwen3-8B），被 model 的 /v1/chat/completions 反代
#   voice 3910  语音（Qwen3-TTS），被 model 的 /v1/audio/speech 反代
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${YJ_DATA:-/root/cloud/yjai}"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
NODE="${YJ_NODE:-/opt/node/bin/node}"
LOG="$DATA/logs"

mkdir -p "$DATA" "$LOG" "$DATA/projects" "$DATA/assets" "$DATA/thumbs"

# 模型实际目录。两处都留着，用哪个以「哪边有」为准：
#   /root/models        系统盘。实测读 67 MB/s，加载权重是几十秒级 —— 常用模型放这。
#                       放 /root 下面不是随手挑的：仙宫云的镜像说明明确要求
#                       「应用及模型请部署安装在 /root/ 目录下」，放 /opt 保存镜像可能丢。
#   /root/cloud/models  webdav 云储存。不进镜像、重开实例还在，但**冷读只有 7–15 MB/s**，
#                       32 GB 的 Wan 冷加载要半小时以上。只放不定期用的大件（H3 168 GB）。
pick_model() {
  local name="$1" local_root="${YJ_LOCAL_MODELS:-/root/models}"
  if [ -d "$local_root/$name" ]; then echo "$local_root/$name"; else echo "/root/cloud/models/$name"; fi
}

YJ_MODEL_DIR="${YJ_MODEL_DIR:-/root/cloud/models}"
export YJ_PORT="${YJ_MODEL_PORT:-3911}"
export YJ_MODEL_DATA="${YJ_MODEL_DATA:-/root/cloud/model-data}"
export YJ_IMAGE_MODEL="${YJ_IMAGE_MODEL:-$(pick_model Qwen-Image-2.1)}"
# H3 的仓库根下只有 FL2VA / Ref2VA 两个分区，model_index.json 在分区里，
# 所以必须指到 FL2VA 这一层，指到 MiniMax-H3 根目录 diffusers 认不出来
export YJ_VIDEO_MODEL="${YJ_VIDEO_MODEL:-$YJ_MODEL_DIR/MiniMax-H3/FL2VA}"
# 默认档走 Wan 5B：H3-Base 只有几十 G 显存的机器扛不住（权重 ~125 GB bf16），
# 强行默认成 H3 会让「点生视频」变成等十分钟然后 OOM。H3 仍然留在别名表里，
# 请求里写 model=h3 才会用。等换到 2×80G 或多卡，把 YJ_VIDEO_DEFAULT 改回去即可。
# 注意仓库名带 -Diffusers 后缀：Wan 官方主仓库是原生权重（diffusion_pytorch_model + .pth），
# DiffusionPipeline.from_pretrained 读不了，必须用转过格式的那一份（有 model_index.json）。
export YJ_VIDEO_FAST_MODEL="${YJ_VIDEO_FAST_MODEL:-$(pick_model Wan2.2-TI2V-5B-Diffusers)}"
export YJ_VIDEO_DEFAULT="${YJ_VIDEO_DEFAULT:-$YJ_VIDEO_FAST_MODEL}"
# 文本：chat-server 监听 8001，正好是 model-server 默认的 TEXT_UPSTREAM
export YJ_TEXT_PORT="${YJ_TEXT_PORT:-8001}"
export YJ_TEXT_MODEL="${YJ_TEXT_MODEL:-$(pick_model Qwen3-8B)}"
export YJ_TEXT_UPSTREAM="${YJ_TEXT_UPSTREAM:-http://127.0.0.1:$YJ_TEXT_PORT/v1}"
# 语音：voice-server/server.py，默认端口 3910 —— 正好是 model-server 的 VOICE_BASE
# 和 host.mjs 的 VOICE_BASE 默认值，所以两边都不用再传地址。
export YJ_VOICE_PORT="${YJ_VOICE_PORT:-3910}"
export YJ_VOICE_DATA="${YJ_VOICE_DATA:-$DATA/voice}"
# 权重用环境变量直指，不走软链：$DATA 在 webdav 上，而 webdav 不支持 symlink
export YJ_TTS_CUSTOM="${YJ_TTS_CUSTOM:-$(pick_model Qwen3-TTS-CustomVoice)}"
export YJ_TTS_BASE="${YJ_TTS_BASE:-$(pick_model Qwen3-TTS-Base)}"
export YJ_TTS_DESIGN="${YJ_TTS_DESIGN:-$(pick_model Qwen3-TTS-Design)}"
# 语音识别：SenseVoice int8（230 MB）。它不只是「转录节点」的后端——
# 克隆音色时参考音频的转写也靠它，没有它克隆会退到 x-vector 模式（音色贴合度下降）。
export YJ_TTS_ASR="${YJ_TTS_ASR:-$(pick_model SenseVoice)}"
export YJ_VOICE_DEVICE="${YJ_VOICE_DEVICE:-cuda}"
export YJ_IMAGE_OFFLOAD="${YJ_IMAGE_OFFLOAD:-1}"
export YJ_IMAGE_MAX_SIDE="${YJ_IMAGE_MAX_SIDE:-1536}"
export YJ_IMAGE_STEPS="${YJ_IMAGE_STEPS:-28}"
export YJ_MODE="${YJ_MODE:-full}"

alive() {
  local p="$1"
  [ -n "$(ss -ltn 2>/dev/null | awk -v p=":$p" '$4 ~ p {print $4}')" ]
}

# 端口释放是异步的：kill 之后内核还要收尾，立刻 start 会看到「还在监听」而跳过启动，
# 结果旧进程死了、新进程没起，服务直接空窗。restart 必须等端口真的空出来。
wait_free() {
  local p="$1" i=0
  while [ $i -lt 40 ]; do
    alive "$p" || return 0
    sleep 0.25
    i=$((i + 1))
  done
  return 1
}

# 端口起来 ≠ 可用：模型服务要等 Python 把 socket 挂好才算真的在。
# 只判端口会在启动失败时误报成功，所以这里按 HTTP 探。
wait_http() {
  local url="$1" i=0
  while [ $i -lt 60 ]; do
    curl -sf -m 2 "$url" >/dev/null 2>&1 && return 0
    sleep 0.5
    i=$((i + 1))
  done
  return 1
}

start_model() {
  if curl -sf -m 3 "http://127.0.0.1:${YJ_PORT}/healthz" >/dev/null 2>&1; then
    echo "  model 已在 ${YJ_PORT}"; return
  fi
  # 端口被占但健康检查不通 = 僵尸残留，先清掉再起
  if alive "$YJ_PORT"; then
    stop_one "$YJ_PORT" "model(残留)"
    wait_free "$YJ_PORT" || echo "  ! 端口 ${YJ_PORT} 仍被占，仍尝试启动"
  fi
  cd "$ROOT" || return 1
  setsid nohup "$PY" "$ROOT/deploy/model-server.py" >"$LOG/model.log" 2>&1 &
  if wait_http "http://127.0.0.1:${YJ_PORT}/healthz"; then
    echo "  ✓ model  http://127.0.0.1:${YJ_PORT}/v1  (pid $(pgrep -f 'deploy/model-server.py' | head -1))"
  else
    echo "  ✗ model 启动失败："; tail -12 "$LOG/model.log"
  fi
}

start_chat() {
  if curl -sf -m 3 "http://127.0.0.1:${YJ_TEXT_PORT}/v1/models" >/dev/null 2>&1; then
    echo "  chat 已在 ${YJ_TEXT_PORT}"; return
  fi
  if alive "$YJ_TEXT_PORT"; then
    stop_one "$YJ_TEXT_PORT" "chat(残留)"
    wait_free "$YJ_TEXT_PORT" || echo "  ! 端口 ${YJ_TEXT_PORT} 仍被占，仍尝试启动"
  fi
  if [ ! -f "$ROOT/deploy/chat-server.py" ]; then echo "  chat 跳过（没有 chat-server.py）"; return; fi
  if [ ! -d "${YJ_TEXT_MODEL:-}" ]; then echo "  chat 跳过（权重目录缺：${YJ_TEXT_MODEL:-未设置}）"; return; fi
  cd "$ROOT" || return 1
  setsid nohup "$PY" "$ROOT/deploy/chat-server.py" >"$LOG/chat.log" 2>&1 &
  # 权重是懒加载的，这里只等 HTTP 起来，不等模型进显存
  if wait_http "http://127.0.0.1:${YJ_TEXT_PORT}/v1/models"; then
    echo "  ✓ chat   http://127.0.0.1:${YJ_TEXT_PORT}/v1  (懒加载)"
  else
    echo "  ✗ chat 启动失败："; tail -12 "$LOG/chat.log"
  fi
}

start_host() {
  if curl -sf -m 3 "http://127.0.0.1:6006/healthz" >/dev/null 2>&1; then
    echo "  host 已在 6006"; return
  fi
  if alive 6006; then
    stop_one 6006 "host(残留)"
    wait_free 6006 || echo "  ! 端口 6006 仍被占，仍尝试启动"
  fi
  cd "$ROOT" || return 1
  PORT=6006 YJ_DATA="$DATA" YJ_WEB="$ROOT/dist-web" \
  YJ_MODEL_BASE="http://127.0.0.1:${YJ_PORT}/v1" \
  YJ_VIDEO_TIMEOUT_MIN="${YJ_VIDEO_TIMEOUT_MIN:-120}" \
  setsid nohup "$NODE" "$ROOT/web/host.mjs" >"$LOG/host.log" 2>&1 &
  if wait_http "http://127.0.0.1:6006/healthz"; then
    echo "  ✓ host   http://127.0.0.1:6006"
  else
    echo "  ✗ host 启动失败："; tail -12 "$LOG/host.log"
  fi
}

start_voice() {
  if curl -sf -m 3 "http://127.0.0.1:${YJ_VOICE_PORT}/healthz" >/dev/null 2>&1; then
    echo "  voice 已在 ${YJ_VOICE_PORT}"; return
  fi
  if alive "$YJ_VOICE_PORT"; then
    stop_one "$YJ_VOICE_PORT" "voice(残留)"
    wait_free "$YJ_VOICE_PORT" || echo "  ! 端口 ${YJ_VOICE_PORT} 仍被占，仍尝试启动"
  fi
  if [ ! -f "$ROOT/voice-server/server.py" ]; then echo "  voice 跳过（没有 voice-server/server.py）"; return; fi
  if [ ! -d "${YJ_TTS_CUSTOM:-}" ] && [ ! -d "${YJ_TTS_BASE:-}" ]; then
    echo "  voice 跳过（预设与克隆权重都缺）"; return
  fi
  cd "$ROOT" || return 1
  setsid nohup "$PY" "$ROOT/voice-server/server.py" >"$LOG/voice.log" 2>&1 &
  # 模型是启动时预热的（17 秒），但这里只等 HTTP——不然 start 要卡半分钟。
  # 前端探的是 /healthz，等它起来才说明 uvicorn 真的接上了。
  if wait_http "http://127.0.0.1:${YJ_VOICE_PORT}/healthz"; then
    echo "  ✓ voice  http://127.0.0.1:${YJ_VOICE_PORT}  (预热中)"
  else
    echo "  ✗ voice 启动失败："; tail -12 "$LOG/voice.log"
  fi
}

stop_one() {
  local p="$1" name="$2"
  local pids
  pids="$(ss -ltnp 2>/dev/null | awk -v p=":$p" '$4 ~ p' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)"
  if [ -z "$pids" ]; then echo "  $name 没在跑"; return; fi
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null
  sleep 1
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null
  echo "  ✓ 已停 $name ($pids)"
}

cmd_status() {
  echo "── 服务 ──"
  if alive 3911; then
    echo "  model  3911  up   $(curl -s -m 4 http://127.0.0.1:3911/healthz | head -c 200)"
  else echo "  model  3911  down"; fi
  if alive 6006; then
    echo "  host   6006  up   $(curl -s -m 4 http://127.0.0.1:6006/healthz | head -c 160)"
  else echo "  host   6006  down"; fi
  if alive "$YJ_TEXT_PORT"; then
    echo "  chat   $YJ_TEXT_PORT  up   $(curl -s -m 4 http://127.0.0.1:$YJ_TEXT_PORT/healthz | head -c 160)"
  else echo "  chat   $YJ_TEXT_PORT  down"; fi
  if alive "$YJ_VOICE_PORT"; then
    echo "  voice  $YJ_VOICE_PORT  up   $(curl -s -m 4 http://127.0.0.1:$YJ_VOICE_PORT/healthz | head -c 160)"
  else echo "  voice  $YJ_VOICE_PORT  down"; fi
  echo "── 显存 ──"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (无 nvidia-smi)"
  echo "── 磁盘 ──"
  du -sh "${YJ_LOCAL_MODELS:-/root/models}"/* 2>/dev/null || true
  df -h /root/cloud / 2>/dev/null | tail -2
}

case "${1:-status}" in
  start)   start_chat; start_voice; start_model; start_host ;;
  stop)    stop_one "$YJ_TEXT_PORT" chat; stop_one "$YJ_VOICE_PORT" voice; stop_one 3911 model; stop_one 6006 host ;;
  restart)
    stop_one "$YJ_VOICE_PORT" voice; stop_one "$YJ_TEXT_PORT" chat; stop_one 3911 model; stop_one 6006 host
    wait_free "$YJ_VOICE_PORT"; wait_free "$YJ_TEXT_PORT"; wait_free 3911; wait_free 6006
    start_chat; start_voice; start_model; start_host ;;
  status)  cmd_status ;;
  log)
    case "${2:-model}" in
      host)  tail -40 "$LOG/host.log" ;;
      chat)  tail -40 "$LOG/chat.log" ;;
      voice) tail -40 "$LOG/voice.log" ;;
      *)     tail -40 "$LOG/model.log" ;;
    esac ;;
  *) echo "用法: bash deploy/run.sh start|stop|restart|status|log [model|host|chat|voice]"; exit 2 ;;
esac
