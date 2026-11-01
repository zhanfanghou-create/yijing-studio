#!/usr/bin/env python3
"""自定义服务入口：点开就跳转到艺镜AI 官网。

背景：仙宫云的自定义服务 = 「容器内监听某个端口」+「平台反代成
https://<实例id>-<端口>.container.x-gpu.com」。平台只认端口，不认 URL，
所以想「点一下直接到官网」，就必须有个进程占着那个端口。

为什么用 200 + HTML 跳转，不用 302：
    平台的反代有可能自己跟随 3xx（把目标页面抓回来再吐给浏览器）。那样地址栏
    仍然是容器域名，页面里的相对链接全坏。HTML 跳转是浏览器执行的，反代无从干预。
    为了让「连 JS 都不跑」的极端情况也能跳，页面里同时放了 meta refresh 和一个可点的链接。

端口：默认 8899。已被占用的四个端口别再动 ——
    6006 画布宿主 / 3911 模型服务 / 8001 对话 / 3910 语音。

单独跑：
    python3 deploy/home-link.py
    YJ_LINK_PORT=8900 python3 deploy/home-link.py
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET = os.environ.get("YJ_LINK_TARGET", "https://www.yjai.top/")
PORT = int(os.environ.get("YJ_LINK_PORT", "8899"))
HOST = os.environ.get("YJ_LINK_HOST", "0.0.0.0")

# 目标地址进 HTML 前统一走 JSON 转义，避免配置里带引号把页面写坏
_TARGET_HTML = (
    TARGET.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
)
_TARGET_JS = json.dumps(TARGET)

PAGE = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="0;url={_TARGET_HTML}">
<title>正在打开 艺镜AI</title>
<style>
  html,body{{height:100%;margin:0}}
  body{{display:flex;align-items:center;justify-content:center;
       background:#0e0e12;color:#d8d8e0;
       font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
  a{{color:#8b7bff;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  .box{{text-align:center}}
  .dot{{display:inline-block;width:6px;height:6px;margin-right:8px;border-radius:50%;
       background:#8b7bff;vertical-align:middle}}
</style>
</head>
<body>
  <div class="box">
    <p><span class="dot"></span>正在打开 <a href="{_TARGET_HTML}">艺镜AI 官网</a>…</p>
    <p style="font-size:12px;color:#7a7a88">没有自动跳转？点上面的链接</p>
  </div>
  <script>location.replace({_TARGET_JS});</script>
</body>
</html>
""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "yjai-link/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def version_string(self) -> str:
        # 基类会把 server_version 与 sys_version 用空格拼起来，后者留空就多一个尾空格
        return self.server_version

    def _send(self, code: int, ctype: str, body: bytes, head_only: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 跳转页不能进缓存，否则换域名后用户点开还是老的
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler 的约定命名)
        # 探活用，任意路径都能跳，只有 /healthz 例外
        if self.path.split("?")[0] == "/healthz":
            self._send(200, "text/plain; charset=utf-8", b"ok\n")
            return
        self._send(200, "text/html; charset=utf-8", PAGE)

    def do_HEAD(self) -> None:  # noqa: N802
        self._send(200, "text/html; charset=utf-8", PAGE, head_only=True)

    def log_message(self, fmt: str, *args) -> None:
        # 默认往 stderr 打个不停，容器连开几十天会把日志撑满。只在出错时留一行。
        if len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("home-link: %s\n" % (fmt % args))


def main() -> int:
    try:
        srv = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        # 端口被占是唯一需要人看一眼的失败：说明已经有一个在跑，或撞了别的服务
        sys.stderr.write(f"home-link: 绑定 {HOST}:{PORT} 失败 —— {e}\n")
        return 1
    srv.daemon_threads = True
    print(f"home-link: http://{HOST}:{PORT}/ -> {TARGET}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
