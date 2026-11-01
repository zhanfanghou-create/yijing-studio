#!/usr/bin/env bash
# 一体机存储整理：把云储存（webdav `/root/cloud`）上的模型搬进系统盘，然后清掉云盘副本。
#
# 为什么这么做（2026-11-01 改口径，与 2026-10-24 相反）：
#   · 云储存 ¥0.00036 /GB/小时（免费额度 25 GB）
#   · 镜像储存 ¥0.00003 /GB/小时（免费额度 30 GB），便宜 12 倍
#   · 官方明确：**存镜像时云储存里的数据不会被打进镜像**
# 所以只要权重还躺在 /root/cloud，镜像就是空壳，云储存费永远省不掉，
# 而且删掉实例后这笔钱还在扣。搬进 /root/models 后：云储存归零，
# 镜像自带权重，新实例开机即用（不必再等 28 GB 下载）。
#
# 用法（全部在 /root/yjai-studio 下执行，统一用 `bash` 前缀，不依赖可执行位）：
#   bash deploy/migrate-to-image.sh              只看计划，不动任何东西
#   bash deploy/migrate-to-image.sh --run        后台拷贝 → 逐包校验 → 通过才清源 → 重启服务
#   bash deploy/migrate-to-image.sh --log        看进度
#   bash deploy/migrate-to-image.sh --verify     只校验（不删东西）
#   bash deploy/migrate-to-image.sh --trash      看云盘回收站缓存 /.xgcos/trash 里有什么
#   bash deploy/migrate-to-image.sh --trash-yes  清空它（腾系统盘，进镜像前必做）
#
# 设计约束：
#   1. **先搬、验通过、最后才删**。每搬完一个包就当场校验，不通过立刻中止，
#      云盘副本原样保留 —— 96 G 权重删错要重下好几个小时。
#   2. 拷贝落在 `<目标>.part`，校验通过才改名。中途断掉不会留下「看着像完整」的目录。
#   3. 目标已存在的包**不覆盖**，只跳过 —— 宁可少搬一个，也不拿半份覆盖完整的一份。
#
# 2026-11-01 修的坑（线上真事故，务必保留这段注释）：
#   旧版 `bytes_of` 用 `awk '{s+=$1} END {printf "%d", s}'`。服务器的 awk 是 **mawk**，
#   `%d` 把 double 往 32 位 int 上套，**任何超过 2 GiB 的目录都恒定返回 2147483647**。
#   于是「字节数相等就删云盘」这条闸对 16 G 和 4.3 G 的包一律判等 —— 全是假绿。
#   现在改成两层判据：
#     · `printf "%.0f"` 求和（double 原样输出，不再溢出）；
#     · 真正的闸是**逐文件清单比对**（相对路径 + 字节数，排序后整串相等），
#       它连「少一个文件」「同名但大小不同」都能抓到，且完全不做整数运算。

set -uo pipefail

SRC="${YJ_CLOUD_MODELS:-/root/cloud/models}"
DST="${YJ_LOCAL_MODELS:-/root/models}"
LOG="${YJ_MIGRATE_LOG:-/root/migrate-to-image.log}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# 云储存上的模型包（2026-11-01 线上 model.status 实测的目录名）。
# 图像与视频两个大件早就在 /root/models 上，这里只列还在云盘的那批。
NAMES=(Qwen3-8B Qwen3-TTS-CustomVoice Qwen3-TTS-Base Qwen3-TTS-Design SenseVoice)

# 文件字节总和。**不要改回 %d**（see 注释顶部）。
bytes_of() {
  [ -d "$1" ] || { echo 0; return; }
  find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{ s += $1 } END { printf "%.0f", s + 0 }'
}
count_of() {
  [ -d "$1" ] || { echo 0; return; }
  find "$1" -type f \( ! -name '.yj-*' \) ! -name '*.part' 2>/dev/null | wc -l | tr -d ' '
}
# 逐文件清单：`字节<TAB>相对路径`，按路径排序。唯一的真判据。
#
# **只列权重文件**，排除两类本地生成物：
#   · `.yj-manifest.json` —— 下载器（host.mjs 的 dlInstall）装完后写的安装清单，
#     几百字节的元数据，配置检测靠它逐文件核对。它不是权重，却是「这份是下载器装的」
#     与「这份是手工 cp 进来的」唯一差别。
#   · `*.part` —— 没写完的半截文件。
# 不排除它们会出现这种情况（2026-11-01 线上原样）：云盘 16 文件 vs 系统盘 17 文件，
# 差的只是 `806 .yj-manifest.json`，于是判「不一致」→ 中止 → 云盘永远清不掉。
# 排除后判据仍然是有效的：真残缺一定表现为「少权重文件」或「同名文件字节不同」，都抓得到。
manifest_of() {
  [ -d "$1" ] || return 0
  ( cd "$1" && find . -type f \( ! -name '.yj-*' \) ! -name '*.part' -printf '%s\t%P\n' 2>/dev/null | sort -t "$(printf '\t')" -k2 )
}
# 被排除的本地元数据，只用于把话说清楚（「忽略了什么」要让人看到，不能静默）
ignored_of() {
  [ -d "$1" ] || return 0
  ( cd "$1" && find . -type f \( -name '.yj-*' -o -name '*.part' \) -printf '%P\n' 2>/dev/null | sort | tr '\n' ' ' )
}
gb() { awk -v b="${1:-0}" 'BEGIN { printf "%.2f", b / 1073741824 }'; }

plan() {
  local n s d sb
  echo "== 磁盘 =="
  df -h / "$SRC" 2>/dev/null | sed 's/  */ /g'
  echo
  echo "== 云储存挂载 =="
  echo "  /root/cloud -> $(readlink -f /root/cloud 2>/dev/null)"
  mount 2>/dev/null | grep -Ei 'cloud|xgcos|webdav|davfs' | head -5
  echo
  echo "== 本机覆盖配置 =="
  if [ -f "$REPO/.env" ]; then grep -Ev '^[[:space:]]*(#|$)' "$REPO/.env" | head -10; else echo "  （无 .env）"; fi
  echo
  echo "== 待搬的包（$SRC → $DST）=="
  local need=0
  for n in "${NAMES[@]}"; do
    s="$SRC/$n"; d="$DST/$n"
    sb="$(bytes_of "$s")"
    if [ -d "$d" ]; then
      printf '  %-24s 云盘 %7sG  目标已存在 %7sG → 跳过\n' "$n" "$(gb "$sb")" "$(gb "$(bytes_of "$d")")"
    elif [ -d "$s" ]; then
      need=$(( need + sb ))
      printf '  %-24s 云盘 %7sG  (%s 字节, %s 个文件) → 待搬\n' "$n" "$(gb "$sb")" "$sb" "$(count_of "$s")"
    else
      printf '  %-24s （云盘没有，跳过）\n' "$n"
    fi
  done
  echo "  合计待搬 $(gb "$need")G，系统盘可用 $(( $(df -Pk "$DST" | awk 'NR==2 {print $4}') / 1048576 ))G"
  echo
  echo "== 云储存明细 =="
  du -sh /root/cloud 2>/dev/null
  du -sh "$SRC"/* 2>/dev/null | sort -h | tail -8
  du -sh /root/cloud/yjai 2>/dev/null
  echo
  echo "== 系统盘明细 =="
  du -sh "$DST"/* 2>/dev/null | sort -h | tail -8
  echo
  echo "== 云盘回收站缓存（进镜像前应清空，否则等于把已删的东西打进镜像）=="
  du -sh /.xgcos/trash 2>/dev/null
  ls -la /.xgcos/trash 2>/dev/null | head -10
  echo
  echo "确认无误后执行： bash deploy/migrate-to-image.sh --run"
}

# 单包校验。返回：0=一致；1=不一致（危险，云盘副本必须保留）；2=还没搬（正常中间态）
verify_one() {
  local n="$1" s="$SRC/$1" d="$DST/$1" ms md sk
  if [ ! -d "$s" ]; then echo "  · $n 云盘已无此包（当作已清理）"; return 0; fi
  if [ ! -d "$d" ]; then echo "  · $n 系统盘还没有（待搬）"; return 2; fi
  ms="$(manifest_of "$s")"; md="$(manifest_of "$d")"
  sk="$(ignored_of "$d")"
  if [ "$ms" = "$md" ]; then
    if [ -n "$sk" ]; then
      echo "  ✓ $n $(gb "$(bytes_of "$s")")G / $(count_of "$s") 个权重文件（清单逐行一致；已忽略本地元数据：$sk）"
    else
      echo "  ✓ $n $(gb "$(bytes_of "$s")")G / $(count_of "$s") 个权重文件（清单逐行一致）"
    fi
    return 0
  fi
  echo "  ✗ $n 清单不一致 —— 云盘 $(gb "$(bytes_of "$s")")G/$(count_of "$s") 文件 vs 系统盘 $(gb "$(bytes_of "$d")")G/$(count_of "$d") 文件"
  echo "    差异（前 6 行，左=云盘 右=系统盘）："
  diff <(printf '%s\n' "$ms") <(printf '%s\n' "$md") 2>/dev/null | head -6 | sed 's/^/      /'
  return 1
}

verify() {
  local n rc bad=0 todo=0
  echo "== 逐包校验：权重文件清单必须逐行相等 =="
  for n in "${NAMES[@]}"; do
    verify_one "$n"; rc=$?
    [ "$rc" -eq 1 ] && bad=$(( bad + 1 ))
    [ "$rc" -eq 2 ] && todo=$(( todo + 1 ))
  done
  if [ "$bad" -gt 0 ]; then
    echo "  => $bad 个包内容不一致，**不会删除云盘副本**"
    return 1
  fi
  if [ "$todo" -gt 0 ]; then
    echo "  => 还有 $todo 个包没搬，云盘副本保留；执行 --run 继续搬"
    return 2
  fi
  echo "  => 全部一致"
  return 0
}

ticker() {
  local d="$1"
  while [ -d "$d" ]; do
    echo "    …已写 $(du -sh "$d" 2>/dev/null | cut -f1)  $(date +%T)"
    sleep 60
  done
}

worker() {
  local n s d rc tp
  echo "=== $(date '+%F %T') 迁移开始 ==="
  mkdir -p "$DST"

  # 目标必须在系统盘上，不能是指进云盘的软链（webdav 也不支持软链）
  if [ "$(readlink -f "$DST")" != "$DST" ]; then
    echo "[abort] $DST 解析到了 $(readlink -f "$DST")，不是系统盘上的实体目录"
    return 1
  fi

  # 先看空间够不够，不够就别开始 —— 搬到一半满盘比不搬更糟
  local need=0 avail
  for n in "${NAMES[@]}"; do
    [ -d "$SRC/$n" ] && [ ! -d "$DST/$n" ] && need=$(( need + $(bytes_of "$SRC/$n") ))
  done
  avail=$(( $(df -Pk "$DST" | awk 'NR==2 { print $4 }') * 1024 ))
  echo "[plan] 需要 $(gb "$need")G（$need 字节），可用 $(gb "$avail")G"
  if [ "$need" -gt "$avail" ]; then
    echo "[abort] 系统盘空间不足，未拷贝任何文件"
    return 1
  fi

  for n in "${NAMES[@]}"; do
    s="$SRC/$n"; d="$DST/$n"
    if [ ! -d "$s" ]; then echo "[skip] $n 云盘不存在"; continue; fi
    if [ -d "$d" ]; then
      echo "[skip] $n 系统盘已存在，先校验再决定"
      verify_one "$n"; rc=$?
      if [ "$rc" -eq 1 ]; then
        echo "[abort] 已存在的 $n 与云盘内容不一致，请人工确认后再继续（云盘副本未动）"
        return 1
      fi
      echo "[skip] $n 校验通过，无需重搬"
      continue
    fi
    if [ -d "$d.part" ]; then echo "[abort] $n 有上次没搬完的 $d.part，先人工确认（确认是残件后 rm -rf 它）"; return 1; fi
    echo "[copy] $n $(gb "$(bytes_of "$s")")G 开始 $(date +%T)"
    ticker "$d.part" & tp=$!
    cp -r "$s" "$d.part" 2>&1
    rc=$?
    kill "$tp" 2>/dev/null
    wait "$tp" 2>/dev/null
    if [ "$rc" -ne 0 ]; then
      echo "[fail] $n 拷贝失败 rc=$rc，中止；云盘副本未动，残件在 $d.part"
      return 1
    fi
    mv "$d.part" "$d" || { echo "[fail] $n 改名失败，保留 $d.part 供人工检查"; return 1; }
    # 搬完立刻验这一个包，不等全部搬完 —— 早失败早停
    if ! verify_one "$n"; then
      echo "[abort] $n 校验未通过，云盘副本保留、后续包不再搬"
      return 1
    fi
    echo "[done] $n $(du -sh "$d" 2>/dev/null | cut -f1)  $(date +%T)"
  done

  if ! verify; then
    echo "[abort] 全量校验未通过，云盘副本全部保留。请把上面这段贴出来看"
    return 1
  fi

  echo "[sweep] 删除云盘副本"
  for n in "${NAMES[@]}"; do
    s="$SRC/$n"
    if [ -d "$s" ]; then
      if rm -rf -- "$s"; then echo "  已删 $s"; else echo "  ! 删不掉 $s（webdav 只读？）"; fi
    fi
  done
  echo "[sweep] 云储存现在 $(du -sh /root/cloud 2>/dev/null | cut -f1)"

  echo "[restart] 重启服务，让新路径生效"
  ( cd "$REPO" && bash deploy/run.sh restart ) 2>&1 | tail -24
  sleep 3
  echo "[health] $(curl -s -m 30 http://127.0.0.1:6006/healthz | head -c 400)"
  echo "=== $(date '+%F %T') 完成 ==="
}

trash_list() {
  echo "== /.xgcos/trash（云盘客户端把删除/新写入的目录缓存到系统盘）=="
  du -sh /.xgcos/trash 2>/dev/null || echo "  （不存在）"
  ls -la /.xgcos/trash 2>/dev/null | head -20
  echo
  echo "确认这些是可以丢的东西后执行： bash deploy/migrate-to-image.sh --trash-yes"
}

trash_purge() {
  local p="/.xgcos/trash"
  [ -d "$p" ] || { echo "$p 不存在，无需清理"; return 0; }
  # 先确认五个包在系统盘都有完整副本 —— 回收站里那 33 G 全是权重残骸，
  # 一旦本地没搬全就清空，等于把唯一一份也删了。
  local n miss=0
  for n in "${NAMES[@]}"; do
    [ -d "$DST/$n" ] || { echo "  ! 系统盘缺 $DST/$n"; miss=1; }
  done
  if [ "$miss" -ne 0 ]; then
    echo "先跑完迁移（bash deploy/migrate-to-image.sh --run）再清回收站，什么都没删。"
    return 1
  fi
  local before; before="$(du -sh "$p" 2>/dev/null | cut -f1)"
  echo "清理前：$p = $before"
  find "$p" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  sync
  echo "已清空 $p（原 $before），系统盘现在："
  df -h / | tail -1
}

case "${1:---plan}" in
  --run)
    mkdir -p "$(dirname "$LOG")"
    setsid nohup bash "$0" --worker < /dev/null >> "$LOG" 2>&1 &
    echo "已在后台开始（logout 不影响），日志：$LOG"
    sleep 2
    echo "--- 当前日志 ---"
    tail -8 "$LOG" 2>/dev/null
    echo "--- 看进度： bash deploy/migrate-to-image.sh --log ---"
    ;;
  --worker)    worker ;;
  --verify)    verify ;;
  --log)       tail -40 "$LOG" 2>/dev/null || echo "还没有日志" ;;
  --trash)     trash_list ;;
  --trash-yes) trash_purge ;;
  --plan|"")   plan ;;
  *)
    echo "用法: bash $0 [--run|--worker|--log|--verify|--trash|--trash-yes]" >&2
    exit 2 ;;
esac
