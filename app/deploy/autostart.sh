#!/bin/bash
# 开机自启真正干活的那个脚本。三种注册方式（start.d / systemd / rc.local）
# 全部转调这一个文件，所以逻辑只有一份，改这里不用重新装钩子。
#
# 它做三件事，顺序不能换：
#   1. 等 webdav 挂上（/root/cloud）—— 权重在那儿，run.sh 是按目录存在性选路径的
#   2. 调 run.sh start（幂等，已在跑的服务不会被重起）
#   3. 探一次 6006，把结果写进日志
#
# 单独跑也安全：bash deploy/autostart.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${YJ_AUTOSTART_LOG:-/var/log/yjai-autostart.log}"

[ -d "$ROOT/deploy" ] || exit 0

# 日志别无限长：超过 256 KB 就只留后半段（容器可能连开很多天）
if [ -f "$LOG" ]; then
  _sz=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "$_sz" -gt 262144 ]; then
    tail -c 131072 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
  fi
fi

exec >>"$LOG" 2>&1
echo "===== $(date '+%F %T') boot autostart ====="

# 云盘挂载通常比容器启动慢几秒到几十秒。服务本身是懒加载、不依赖权重，
# 但 run.sh 的 pick_model 是**按目录存不存在**来定 YJ_*_MODEL 的：
# 云盘没挂上就会指到一条不存在的路径，之后即使挂上了也认不回来，
# 面板会一直显示「未下载」，只有下次重启才自愈。所以先等挂载点。
for _i in $(seq 1 60); do
  grep -q " /root/cloud " /proc/mounts 2>/dev/null && break
  sleep 2
done

cd "$ROOT" || exit 0

# run.sh 的每个 start_* 都是「先探端口 / HTTP，通了就跳过」，重复执行安全。
# 它内部用 setsid nohup 起服务，所以本脚本退出后服务继续跑。
bash deploy/run.sh start

_code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 5 "http://127.0.0.1:6006/" 2>/dev/null)
# 官网入口单独探一次：它跟模型无关，可能这一项好了、画布还没好
_link=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' -m 3 "http://127.0.0.1:${YJ_LINK_PORT:-8899}/healthz" 2>/dev/null)
echo "===== $(date '+%F %T') host=${_code:-000} link=${_link:-000} ====="
