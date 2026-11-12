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
# 本机覆盖：一行一个 KEY=VALUE，用来放「这台机器和别人不一样」的设置，不进仓库。
# 典型用途：YJ_INSTALL_ROOT=/root/models
#   —— 云盘挂载不稳、或者冷读太慢时，把权重装到系统盘（67 MB/s vs 7–15 MB/s）。
# `set -a` 让 source 进来的变量自动 export，下游 python / node 才看得到。
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi
DATA="${YJ_DATA:-/root/cloud/yjai}"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
# Node 路径不写死。宿主机上可能同时存在系统自带的老 node（有的镜像自带 v12，
# 跑 host.mjs 会直接语法报错）和 init.sh 装到 /opt/node20 的新 node，
# 而安装位置各家镜像并不统一。所以按「显式指定 → /opt/node20 → PATH」的顺序挑，
# **挑的时候真的执行一次问版本**，低于 18 的跳过，不要凭路径猜。
pick_node() {
  local cand major
  for cand in "${YJ_NODE:-}" /opt/node20/bin/node /opt/node/bin/node \
              /usr/local/bin/node /usr/bin/node "$(command -v node 2>/dev/null)"; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    major="$("$cand" -p 'process.versions.node.split(".")[0]' 2>/dev/null)"
    if [ "${major:-0}" -ge 18 ] 2>/dev/null; then printf '%s' "$cand"; return 0; fi
  done
}
NODE="$(pick_node)"
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

YJ_MODEL_DIR="${YJ_MODEL_DIR:-/root/models}"
# 配置检测里点「下载」时的**落盘根**。默认系统盘（2026-11-01 改口径）：
#   · 云储存 ¥0.00036/GB/小时（免费 25 GB），镜像 ¥0.00003/GB/小时（免费 30 GB），差 12 倍；
#   · 官方明确「存镜像时云储存里的数据不进镜像」—— 权重留在云盘，镜像永远是空壳，
#     云储存费也永远省不掉，删了实例还在扣。
# 所以权重一律落系统盘、由镜像带走；要改回云盘就在 .env 里写 YJ_INSTALL_ROOT=/root/cloud/models。
# 宿主与检测**必须用同一个值**，否则又是「下到 A、查 B」。
export YJ_INSTALL_ROOT="${YJ_INSTALL_ROOT:-$YJ_MODEL_DIR}"
export YJ_PORT="${YJ_MODEL_PORT:-3911}"
export YJ_MODEL_DATA="${YJ_MODEL_DATA:-/root/cloud/model-data}"
# 显存/线程两条硬线（deploy/vram_guard.py）与空闲自动释放：
#   超预算时宁可这次任务报错，也不许 Windows 的 CUDA 系统内存回退把整机拖死
#   （实测 1024² 峰值 21 GB / 卡上 12 GB，机器卡到鼠标不动）。
#   空闲 180 秒没人用就把权重还给系统，别一直占着显存和内存。
export YJ_VRAM_FRACTION="${YJ_VRAM_FRACTION:-0.88}"
export YJ_IDLE_UNLOAD_S="${YJ_IDLE_UNLOAD_S:-180}"
export YJ_IMAGE_MODEL="${YJ_IMAGE_MODEL:-$(pick_model Qwen-Image-2.1)}"
# 视频有两档后端：H3（全量，默认）与 Wan2.2-TI2V-5B（低显存 GGUF，12 G 卡可跑）。
# 由 `pick_video` 按盘上实际装了哪份决定，两者目录结构与采样参数完全不同 ——
#
#   A. 官方 diffusers 仓库 `MiniMaxAI/MiniMax-H3`：根下只有 FL2VA / Ref2VA 两个分区，
#      `model_index.json` 在分区里 → 必须指到 `<root>/FL2VA` 这一层。
#   B. DiffSynth 的 NF4 量化版 `DiffSynth-Studio/MiniMax-H3-NF4`：扁平结构，
#      根下就是 `configuration.json` + 几个 `*_nf4.safetensors` → 指到目录根。
#
# 按「哪个真存在」决定，不要按名字猜：这两种目录名都含 `MiniMax-H3`，
# 拼死一个 `/FL2VA` 后缀在 B 上就是一条不存在的路径。
# 走 pick_model 而不是拼 $YJ_MODEL_DIR：/root/models 上的权重会被优先选中
# （系统盘 67 MB/s vs webdav 7–15 MB/s，加载一个几十 G 的模型差十几分钟）。
pick_video() {
  local base
  for base in "${YJ_LOCAL_MODELS:-/root/models}" /root/cloud/models; do
    # A：官方分区
    if [ -f "$base/MiniMax-H3/FL2VA/model_index.json" ]; then
      echo "$base/MiniMax-H3/FL2VA"; return
    fi
    # A'：目录名带后缀的官方分区（例如从平台 UI 下的 MiniMax-H3-Diffusers）
    local d
    for d in "$base"/MiniMax-H3*/; do
      [ -d "$d" ] || continue
      if [ -f "${d}FL2VA/model_index.json" ]; then echo "${d%/}/FL2VA"; return; fi
      # B：NF4 扁平包
      if [ -f "${d}configuration.json" ] && ls "${d}"*_nf4.safetensors >/dev/null 2>&1; then
        echo "${d%/}"; return
      fi
    done
  done
  # 都没有：**什么都不输出**。
  #
  # 以前这里 echo "$(pick_model MiniMax-H3)/FL2VA"，好让上层「照常报权重缺」。
  # 代价是 YJ_VIDEO_MODEL 被 export 成一条**不存在**的路径，而宿主进程的
  # `resolveModelDir()` 见到 env 就直接用 —— 于是配置检测永远盯着那条空路径：
  # 面板里下完 34 GB、进度 100%、清单也写了，状态栏还是「未下载」。
  # 空字符串的含义是「没配」，让 model-server 用它自己的默认（仓库 id）、
  # 让宿主按目录名去扫 —— 两边都能在权重出现后**自动看见**，不必重启。
}

_video_dir="$(pick_video)"
if [ -n "$_video_dir" ]; then
  export YJ_VIDEO_MODEL="${YJ_VIDEO_MODEL:-$_video_dir}"
  export YJ_VIDEO_DEFAULT="${YJ_VIDEO_DEFAULT:-$YJ_VIDEO_MODEL}"
fi
# H3 的显存门槛取决于**盘上那份是哪一版**：
#   · bf16 全量约 125 G（DiT 62 + Qwen3-VL 编码器 63）→ 全球也没几张消费卡能跑；
#   · NF4 量化包官方口径「最低 8 G 显存即可运行」（DiffSynth 会自动做 disk offload），
#     24 G 的 4090 是能出片的，只是慢。
# 所以这里按目录形态判门槛，不能在 24 G 卡上无条件报「会失败」——
# 那会让人以为买了张废卡，实际能跑。
# 仍然**不在启动时降级到别的模型**（那是以前的做法），但要把话说在前面。
_vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
  | sort -rn | head -1 | awk '{ printf "%d", ($1 + 512) / 1024 }')
_vram_gb="${_vram_gb:-0}"
case "${YJ_VIDEO_MODEL:-}" in
  *nf4*|*NF4*) _vram_need=8 ;;
  *)           _vram_need=125 ;;
esac
if [ -z "${YJ_VIDEO_MODEL:-}" ] || [ ! -d "$YJ_VIDEO_MODEL" ]; then
  # 30 G 镜像里放不下 34 G 的 NF4 权重，所以「盘上没有」是常见状态，不是故障。
  # 说清「没装」比甩一串显存门槛有用。两条补装路径都写上：
  # 网页面板那条是现在的主路径（宿主自己下、能续传），bootstrap 那条留给纯命令行。
  echo "  · 视频：未安装权重（${YJ_VIDEO_MODEL:-未配置}）"
  echo "    装：网页「模型配置」里点视频模型下载（落盘 ${YJ_INSTALL_ROOT}/MiniMax-H3-NF4，约 34 GB）"
  echo "        或 bash deploy/bootstrap.sh models video"
elif [ "$_vram_gb" -gt 0 ] && [ "$_vram_gb" -lt "$_vram_need" ]; then
  echo "  ! 视频：当前权重（${YJ_VIDEO_MODEL}）需要 ${_vram_need} G 以上显存，"
  echo "    这台只有 ${_vram_gb} G —— 生视频会显存不足。换 NF4 量化版可降到 8 G 起。"
fi
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
  if [ -z "$NODE" ]; then
    echo "  ✗ host 启动失败：找不到 Node 18+（装到 /opt/node20 或设 YJ_NODE=<路径>）"
    return 1
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

# 自定义服务入口（仙宫云「艺镜AI 官网」那一条，8899）。
# 它只做一件事：把浏览器跳到官网。不碰模型、不占显存、不读权重，
# 所以优先用系统 python3 —— venv 还没建好时也能起，这是用户最先想点的入口。
start_link() {
  local port="${YJ_LINK_PORT:-8899}" py
  if curl -sf -m 2 "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
    echo "  link  已在 ${port}"; return
  fi
  if alive "$port"; then
    stop_one "$port" "link(残留)"
    wait_free "$port" || echo "  ! 端口 ${port} 仍被占，仍尝试启动"
  fi
  for py in "$PY" "$(command -v python3 2>/dev/null)" /usr/bin/python3 /usr/local/bin/python3; do
    [ -n "$py" ] && [ -x "$py" ] && break
    py=""
  done
  if [ -z "$py" ]; then echo "  link 跳过（找不到 python3）"; return; fi
  cd "$ROOT" || return 1
  YJ_LINK_PORT="$port" setsid nohup "$py" "$ROOT/deploy/home-link.py" >"$LOG/link.log" 2>&1 &
  if wait_http "http://127.0.0.1:${port}/healthz"; then
    echo "  ✓ link   http://0.0.0.0:${port}/ → ${YJ_LINK_TARGET:-https://www.yjai.top/}"
  else
    echo "  ✗ link 启动失败："; tail -12 "$LOG/link.log"
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
  # 官网入口只做跳转，健康检查回了就能点，不必看别的东西
  if curl -sf -m 2 "http://127.0.0.1:${YJ_LINK_PORT:-8899}/healthz" >/dev/null 2>&1; then
    echo "  link   ${YJ_LINK_PORT:-8899}  up   → ${YJ_LINK_TARGET:-https://www.yjai.top/}"
  else echo "  link   ${YJ_LINK_PORT:-8899}  down"; fi
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

# 单个服务的名字 → 起点 / 端口。`start model` 这种用法是必需的：
# 只改了一个 python 服务时去 `restart` 全家桶，会把正在跑的 34 GB 下载
# （宿主进程里的后台任务，只在内存里）一起干掉。少一次全量重启就少一次事故。
svc_port() {
  case "$1" in
    link)  echo "${YJ_LINK_PORT:-8899}" ;;
    chat)  echo "$YJ_TEXT_PORT" ;;
    voice) echo "$YJ_VOICE_PORT" ;;
    model) echo 3911 ;;
    host)  echo 6006 ;;
    *)     echo "" ;;
  esac
}
svc_start() {
  case "$1" in
    link)  start_link ;;
    chat)  start_chat ;;
    voice) start_voice ;;
    model) start_model ;;
    host)  start_host ;;
  esac
}
# 名字 → 给人看的标签，只用于停服务时的日志
svc_stop() {
  local port; port="$(svc_port "$1")"
  [ -n "$port" ] && stop_one "$port" "$1"
}

ALL_SVCS="link chat voice model host"

case "${1:-status}" in
  # link 放最前：它零成本秒起，别让「等 host 探活 30 秒」拖住那个入口
  start)
    if [ -n "${2:-}" ]; then svc_start "$2"; else for s in $ALL_SVCS; do svc_start "$s"; done; fi ;;
  stop)
    if [ -n "${2:-}" ]; then svc_stop "$2"; else for s in $ALL_SVCS; do svc_stop "$s"; done; fi ;;
  restart)
    if [ -n "${2:-}" ]; then
      svc_stop "$2"; wait_free "$(svc_port "$2")"; svc_start "$2"
    else
      # 顺序沿用原样：先停 link/voice/chat，再 model/host
      for s in link voice chat model host; do svc_stop "$s"; done
      for s in $ALL_SVCS; do wait_free "$(svc_port "$s")"; done
      for s in $ALL_SVCS; do svc_start "$s"; done
    fi ;;
  status)  cmd_status ;;
  log)
    case "${2:-model}" in
      host)  tail -40 "$LOG/host.log" ;;
      chat)  tail -40 "$LOG/chat.log" ;;
      voice) tail -40 "$LOG/voice.log" ;;
      link)  tail -40 "$LOG/link.log" ;;
      *)     tail -40 "$LOG/model.log" ;;
    esac ;;
  *) echo "用法: bash deploy/run.sh start|stop|restart|status|log [model|host|chat|voice|link]"; exit 2 ;;
esac
