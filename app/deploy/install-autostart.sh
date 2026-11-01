#!/bin/bash
# 把 deploy/autostart.sh 注册成开机自启。
#
# 为什么需要注册三种机制
# ----------------------
# 仙宫云容器的 PID 1 是 `docker-init -- /scripts/start.sh`，**既没有 systemd、
# 也没有 sysvinit**。以前这里只装了 systemd 服务，后果是双重失效：
#   · `systemctl daemon-reload` 在这台机器上返回非 0
#   · init.sh 开头是 `set -e`，于是脚本在自启那一步**直接中断**，
#     后面「启动服务」一步根本没执行 → 部署「完成」了，6006 一直是空的
#   · 后来补写 /etc/rc.local 也无效：没有 init 会去读它
#
# 所以现在按宿主机的实际情况注册，命中哪个用哪个：
#   1. /scripts/start.d/   仙宫云容器官方扩展点（平台入口 start.sh 会遍历执行）
#   2. systemd             真有 systemd 的云主机（判据取 /run/systemd/system，
#                          不能用 `command -v systemctl` —— 容器里二进制在，服务没跑）
#   3. /etc/rc.local       有 sysvinit 的镜像兜底
#
# 三者都只转调 deploy/autostart.sh，所以「启动逻辑只有一份」。
# 本脚本幂等，可以在 init.sh / bootstrap.sh 里反复调。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/deploy/autostart.sh"
LOG="${YJ_AUTOSTART_LOG:-/var/log/yjai-autostart.log}"

echo "── 开机自启"
echo "   项目目录 $ROOT"

if [ ! -f "$RUNNER" ]; then
  echo "   ✗ 找不到 $RUNNER，跳过"
  exit 1
fi
chmod +x "$RUNNER" 2>/dev/null
echo "   ✓ $RUNNER"

INSTALLED=""

# ── 1. 仙宫云容器：/scripts/start.d/ ─────────────────────────────────
# 平台入口 /scripts/start.sh 末尾会 `for script in /scripts/start.d/*` 逐个执行，
# 镜像自带的 jupyter 就是这么自启的。这是首选钩子：容器里只有它一定被读到。
HOOK_DIR=/scripts/start.d
HOOK="$HOOK_DIR/50-yjai.sh"
if mkdir -p "$HOOK_DIR" 2>/dev/null && [ -w "$HOOK_DIR" ]; then
  cat > "$HOOK" <<EOS
#!/bin/bash
# 艺镜 AI · 容器开机自启
# 由 $ROOT/deploy/install-autostart.sh 生成，改这里会被下次安装覆盖。
# 实际逻辑在 $RUNNER（三种机制共用一份）。
[ -x "$RUNNER" ] && exec "$RUNNER"
EOS
  chmod +x "$HOOK"
  echo "   ✓ $HOOK"
  INSTALLED="$INSTALLED start.d"
else
  echo "   · 没有可写的 $HOOK_DIR（不是仙宫云容器，跳过）"
fi

# ── 2. systemd：只在真的用 systemd 当 init 时装 ────────────────────
# 判据必须是 /run/systemd/system —— 它只在 systemd 作为 PID 1 时存在。
if [ -d /run/systemd/system ]; then
  UNIT=/etc/systemd/system/yjai-studio.service
  cat > "$UNIT" <<EOS
[Unit]
Description=Yijing Studio (canvas host + model/voice/chat services)
After=network-online.target
Wants=network-online.target

[Service]
# oneshot + RemainAfterExit：run.sh start 会把四个服务 fork 成后台进程后返回，
# 本身不是一个长驻进程，用 Type=forking 反而会被判成「主进程已退出」。
Type=oneshot
RemainAfterExit=yes
ExecStart=$RUNNER
ExecStop=$ROOT/deploy/run.sh stop
User=root

[Install]
WantedBy=multi-user.target
EOS
  if systemctl daemon-reload 2>/dev/null && systemctl enable yjai-studio.service >/dev/null 2>&1; then
    echo "   ✓ $UNIT（已 enable）"
    INSTALLED="$INSTALLED systemd"
  else
    echo "   ! $UNIT 写好了但 enable 失败，看 systemctl status"
  fi
else
  echo "   · 没有运行中的 systemd（PID 1 不是 systemd），跳过 systemd 服务"
fi

# ── 3. rc.local：sysvinit 兜底 ─────────────────────────────────────
if [ -d /etc ]; then
  cat > /etc/rc.local <<EOS
#!/bin/bash
# 兜底自启。真正干活的是 $RUNNER。
[ -x "$RUNNER" ] && "$RUNNER" &
exit 0
EOS
  chmod +x /etc/rc.local
  echo "   ✓ /etc/rc.local"
  INSTALLED="$INSTALLED rc.local"
fi

echo "   已注册：${INSTALLED:- 无}"
echo "   日志：$LOG"

# ── 当场跑一次，不用等下次开机 ─────────────────────────────────────
# autostart.sh 内部调 run.sh start，而 run.sh 幂等，已经在跑的服务不会被重起。
if [ "${YJ_NOW:-1}" = "1" ]; then
  echo "   · 立刻执行一次（已在跑的服务不受影响）"
  "$RUNNER"
  echo "   · 完成，最近日志："
  tail -8 "$LOG" 2>/dev/null | sed 's/^/     /'
fi

echo "   完成。下次开机不用手动 start。"
