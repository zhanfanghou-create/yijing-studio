#!/usr/bin/env bash
# 按**实际显存**决定这台机器该装哪些模型，而不是一律下全套。
#
#   bash deploy/vram-plan.sh           人类可读的清单与理由
#   bash deploy/vram-plan.sh --json    给脚本消费
#   bash deploy/vram-plan.sh --want   只输出 bootstrap 要的 key（空格分隔）
#
# 为什么需要它：视频只有 MiniMax-H3 一个后端（Wan 已移除），而 H3 有两套完全不同的形态 ——
# 官方 bf16 全量 171 GB 权重 / 125 GB 显存（除了 8 卡机器谁都别想）；
# DiffSynth 的 NF4 量化包只收 FL2VA 那一路时 **34.4 GB 权重**，
# 官方 README 的口径是「最低 8 G 显存即可运行」（靠 disk offload）。
# 所以 24 GB 的 4090 是**能出片的**，但 bf16 全量下到盘上纯属白占一百多 G。
# 判据取**单卡最大显存**，不取求和：这套推理是按「一个管线放一张卡」写的，
# 不会把 H3 的层切到两张卡上，所以 2×24 GB 的机器也不等于能跑 48 GB 的模型。
set -uo pipefail

# 各模型的实际胃口（含管线里必须同驻的文本编码器 / VAE）。
# 照抄数字是危险的，这里每个数都对应一次真机实测或权重表算出来的体积，
# 改动时请一并改注释里的依据。
read -r -d '' PLAN <<'JSON' || true
[
  {"key":"text","label":"Qwen3-8B 对话","disk":16,"vram":18,
   "note":"bf16 权重 16 G；24 G 卡要限制上下文长度才留得下 KV cache"},
  {"key":"image","label":"Qwen-Image-2.1 生图","disk":31,"vram":24,
   "note":"transformer 约 20 G + 文本编码器；24 G 卡必须开 offload，出图慢一档"},
  {"key":"video","label":"MiniMax-H3 视频（唯一后端）","disk":34,"vram":12,
   "note":"NF4 只收 FL2VA 四件 = 34.4 G 权重（官方 README：最低 8 G 显存，靠 disk offload；24 G 以上才有实用速度）。官方 bf16 全量 171 G / 125 G 显存，别下那份"},
  {"key":"voice","label":"Qwen3-TTS-1.7B 三件套","disk":13,"vram":6,
   "note":"同一时刻只驻留一个，6 G 足够；三件套 12 G 磁盘"}
]
JSON

json_field() { # $1=key $2=field
  printf '%s' "$PLAN" | tr -d '\n' | grep -o "\"key\":\"$1\"[^}]*" | grep -o "\"$2\":[0-9]*" | cut -d: -f2
}

# 单卡最大显存（MiB）。没有 nvidia-smi 说明是 CPU 机器。
# YJ_VRAM_MB 是给「替别的机器算账」和写文档时用的假值入口，正常部署不设。
vram_mb() {
  local v
  if [ -n "${YJ_VRAM_MB:-}" ]; then printf '%s' "$YJ_VRAM_MB"; return; fi
  v="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)"
  printf '%s' "${v:-0}"
}

VRAM_MB="$(vram_mb)"
# 24564 MiB / 1024 = 23.98，整除会变 23 —— 一张 24 G 的卡被判进 16 G 档。
# 这类「差一个整数就跨档」的错最难看出来，所以四舍五入。
VRAM_GB=$(( (VRAM_MB + 512) / 1024 ))
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
GPU_NAME="${YJ_GPU_NAME:-${GPU_NAME:-无 GPU}}"

# 档位只决定「哪些进必装清单」，不改变单个模型的判断：
# 真正能不能跑由 vram 字段比大小决定，档位只是给个人话说明。
if   [ "$VRAM_GB" -ge 128 ]; then TIER="full";        TIER_DESC="大显存单卡 · H3 可跑 bf16"
elif [ "$VRAM_GB" -ge 96  ]; then TIER="xl";          TIER_DESC="96 G 级 · H3 可跑"
elif [ "$VRAM_GB" -ge 48  ]; then TIER="large";       TIER_DESC="48 G 级 · H3 上 NF4 版"
elif [ "$VRAM_GB" -ge 24  ]; then TIER="standard";    TIER_DESC="24 G 级 · 生图要 offload，视频走 NF4"
elif [ "$VRAM_GB" -ge 16  ]; then TIER="compact";     TIER_DESC="16 G 级 · 生图吃力，视频靠 disk offload"
else                              TIER="minimal";     TIER_DESC="显存不足 16 G · 只跑对话与语音"
fi

want=""
skip=""
detail=""
for key in text image video voice; do
  disk="$(json_field "$key" disk)"
  need="$(json_field "$key" vram)"
  label="$(printf '%s' "$PLAN" | tr -d '\n' | grep -o "\"key\":\"$key\"[^}]*" | grep -o '"label":"[^"]*"' | cut -d'"' -f4)"
  note="$(printf '%s' "$PLAN" | tr -d '\n' | grep -o "\"key\":\"$key\"[^}]*" | grep -o '"note":"[^"]*"' | cut -d'"' -f4)"

  # 留 2 GB 余量给 CUDA context / 显示输出；显存恰好等于需求时不算能跑。
  if [ "$VRAM_GB" -ge $(( need + 2 )) ] || { [ "$need" -le 0 ] && [ "$VRAM_GB" -ge 8 ]; }; then
    want="$want $key"
    detail="${detail}${key}|ok|${disk}|${need}|${label}|${note}"$'\n'
  elif [ "$VRAM_GB" -ge $(( need - 6 )) ]; then
    # 差一点点：能跑但要 offload，出片慢一档。归到必装，但标注降速。
    want="$want $key"
    detail="${detail}${key}|tight|${disk}|${need}|${label}|${note}"$'\n'
  else
    skip="$skip $key"
    detail="${detail}${key}|no|${disk}|${need}|${label}|${note}"$'\n'
  fi
done

want="$(echo "$want" | xargs)"
skip="$(echo "$skip" | xargs)"

case "${1:-}" in
  --json)
    # 手写 JSON，免得为了一个脚本引入 jq 依赖
    tojson() {
      printf '%s' "$detail" | while IFS='|' read -r k st disk need label note; do
        [ -n "${k:-}" ] || continue
        printf '{"key":"%s","state":"%s","diskGB":%s,"vramGB":%s,"label":"%s","note":"%s"},' \
          "$k" "$st" "$disk" "$need" "$label" "$note"
      done | sed 's/,$//'
    }
    printf '{"gpu":"%s","vramGB":%s,"tier":"%s","want":"%s","skip":"%s","models":[%s]}\n' \
      "$GPU_NAME" "$VRAM_GB" "$TIER" "$want" "$skip" "$(tojson)"
    exit 0
    ;;
  --want)
    printf '%s\n' "$want"
    exit 0
    ;;
esac

echo ""
echo "  显卡    $GPU_NAME"
echo "  显存    ${VRAM_GB} GB（单卡最大；按单卡判，不按多卡求和）"
echo "  档位    $TIER — $TIER_DESC"
echo ""
printf '  %-6s %-30s %8s %8s  %s\n' "状态" "模型" "磁盘" "需显存" "说明"
printf '  %-6s %-30s %8s %8s  %s\n' "────" "──────────────────────────" "────" "──────" "────"
printf '%s' "$detail" | while IFS='|' read -r k st disk need label note; do
  [ -n "${k:-}" ] || continue
  case "$st" in
    ok)    mark="可跑" ;;
    tight) mark="勉强" ;;
    *)     mark="放不下" ;;
  esac
  printf '  %-6s %-30s %7sG %7sG  %s\n' "$mark" "$label" "$disk" "$need" "$note"
done
echo ""
total=0
for k in $want; do total=$(( total + $(json_field "$k" disk) )); done
all=0
for k in text image video voice; do all=$(( all + $(json_field "$k" disk) )); done
echo "  该装    $want"
[ -n "$skip" ] && echo "  该跳过  $skip"
if [ -n "$skip" ]; then
  echo "  磁盘    约 ${total} GB（全套 ${all} GB，省下 $(( all - total )) GB）"
else
  # 四项全能装时写「省下 0 GB」是废话，直接给总数
  echo "  磁盘    约 ${total} GB（这套机器全能装）"
fi
echo ""
if [ -n "$skip" ]; then
  echo "  下载：bash deploy/bootstrap.sh models all 会把跳过的也拉下来；"
  echo "        只装该装的，按下面逐个跑（或用 --want 喂给部署脚本）："
  for k in $want; do echo "          bash deploy/bootstrap.sh models $k"; done
fi
echo ""
