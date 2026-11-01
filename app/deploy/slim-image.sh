#!/usr/bin/env bash
# 存镜像前的瘦身：把权重从系统盘清掉，让镜像只留代码 + 环境。
#
# 为什么需要这一步（仙宫云 2026-11-01 核实）：
#   · 「储存镜像是将**系统盘里所有数据**都储存到镜像里」
#   · 只有 `/root/cloud`（云储存）和预付费实例的数据盘**不参与**镜像打包
#   · 镜像免费用 30 GB，超出 ¥0.00003/GB/小时
# 所以只要 `/root/models` 里躺着 90 G 权重，存出来的镜像就是 100 G 起，
# 这笔钱是按月一直扣的。想「镜像里没有模型」，只能在存镜像前把它们删掉，
# 之后靠配置检测面板按需下载（宿主自己会下，`web/host.mjs` 的 `dlInstall`）。
#
# 用法（在 /root/yjai-studio 下）：
#   bash deploy/slim-image.sh                 只看会删什么、能瘦到多少
#   bash deploy/slim-image.sh --run           真的删
#   bash deploy/slim-image.sh --keep image    额外保留某个包（可重复，逗号分隔）
#   bash deploy/slim-image.sh --with-trash    连 /.xgcos/trash 一起清（默认也清，见下）
#
# 默认保留 SenseVoice：它只能从 HuggingFace 镜像下，而这台机器的宿主**连不上**
# hf-mirror.com / huggingface.co（实测 `fetch failed`），删了就再也装不回来。
# 它只有 0.23 GB，留在镜像里对费用没有影响。

set -uo pipefail

DST="${YJ_LOCAL_MODELS:-/root/models}"
CLOUD="${YJ_CLOUD_MODELS:-/root/cloud/models}"
TRASH="/.xgcos/trash"

# 会被删的（顺序即占用从大到小）。SenseVoice 不在其中 —— 见文件头。
PACKS=(MiniMax-H3-NF4 Qwen-Image-2.1 Qwen3-8B Qwen3-TTS-CustomVoice Qwen3-TTS-Base Qwen3-TTS-Design)
KEEP=",SenseVoice,"

RUN=0
PURGE_TRASH=1
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    --no-trash) PURGE_TRASH=0 ;;
    --keep) echo "用法：--keep=包名[,包名]（注意是等号）" >&2; exit 2 ;;
    --keep=*) KEEP=",${a#--keep=},,SenseVoice," ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "不认识的参数：$a" >&2; exit 2 ;;
  esac
done

sz() { du -sb "$1" 2>/dev/null | cut -f1; }
gb() { awk -v b="${1:-0}" 'BEGIN { printf "%.1f", b / 1073741824 }'; }
# 目录存在但没有一个文件（下到一半就断了/建了个空壳）——这种也要删
is_empty_dir() { [ -d "$1" ] && [ "$(find "$1" -type f 2>/dev/null | head -1)" = "" ]; }

plan() {
  local n p total=0 freed=0
  echo "== 系统盘现状 =="
  df -h / | sed 's/  */ /g'
  echo
  echo "== $DST 里有什么 =="
  for p in "$DST"/*; do
    [ -e "$p" ] || continue
    n="$(basename "$p")"
    local b; b="$(sz "$p")"
    total=$(( total + b ))
    case "$KEEP" in
      *",$n,"*) printf '  %-26s %8sG  保留（%s）\n' "$n" "$(gb "$b")" "$n" ;;
      *)        printf '  %-26s %8sG  将删除\n' "$n" "$(gb "$b")"; freed=$(( freed + b )) ;;
    esac
  done
  echo "  合计 $(gb "$total")G，其中将释放 $(gb "$freed")G"
  echo
  echo "== 云储存（**不进镜像**，这里只看要不要单独清）=="
  du -sh /root/cloud 2>/dev/null
  ls -1 "$CLOUD" 2>/dev/null | sed 's/^/  /' | head -12
  echo
  echo "== 回收站缓存（进镜像，必须清）=="
  local tb=0
  tb="$(sz "$TRASH")"
  echo "  $TRASH $(gb "$tb")G"
  if [ "$PURGE_TRASH" = 1 ]; then freed=$(( freed + tb )); fi
  echo
  echo "== 预计 =="
  echo "  删除后系统盘约减少 $(gb "$freed")G"
  echo "  镜像体积 ≈ 现在的「已用」 - $(gb "$freed")G（下面这行是删完后的预估）"
  local used; used=$(( $(df -Pk / | awk 'NR==2 {print $3}') * 1024 ))
  echo "  当前已用 $(gb "$used")G → 预计 $(gb "$(( used > freed ? used - freed : 0 ))")G"
  echo
  if [ "$RUN" = 0 ]; then
    echo "确认无误后执行： bash deploy/slim-image.sh --run"
    echo "（删完再去控制台点「储存镜像」；新实例开机后在配置检测里点「一键修复」把模型下回来）"
  fi
}

do_run() {
  local n p
  echo "=== $(date '+%F %T') 开始瘦身 ==="
  for p in "$DST"/*; do
    [ -e "$p" ] || continue
    n="$(basename "$p")"
    case "$KEEP" in
      *",$n,"*) echo "[keep] $n"; continue ;;
    esac
    echo "[rm]   $n $(du -sh "$p" 2>/dev/null | cut -f1)"
    rm -rf -- "$p"
    [ -e "$p" ] && echo "  ! 没删掉：$p"
  done
  # 空壳目录（比如下到一半失败的）单独扫一遍，它们会让配置检测显示「未完成」
  for p in "$DST"/*; do
    [ -e "$p" ] || continue
    n="$(basename "$p")"
    case "$KEEP" in *",$n,"*) continue ;; esac
    is_empty_dir "$p" && { echo "[rm]   空目录 $n"; rm -rf -- "$p"; }
  done
  if [ "$PURGE_TRASH" = 1 ] && [ -d "$TRASH" ]; then
    echo "[rm]   $TRASH $(du -sh "$TRASH" 2>/dev/null | cut -f1)（云盘客户端的删除缓存，进镜像等于白占）"
    find "$TRASH" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
  # 这些也是纯缓存，进镜像没意义
  rm -rf /root/.cache/pip /root/.cache/huggingface/hub 2>/dev/null
  sync
  echo
  echo "=== 完成 ==="
  df -h / | sed 's/  */ /g'
  echo "现在去控制台点「储存镜像」。"
  echo "新实例开机后：配置检测 → 一键修复，模型会按需下回来。"
}

if [ "$RUN" = 1 ]; then plan; echo; do_run; else plan; fi
