#!/usr/bin/env python
"""本机文本服务：OpenAI 兼容的 /v1/chat/completions（含流式）。

为什么不用 vLLM：
  这台机器的 torch 是 2.14.1+cu130（很新），vLLM 对 torch 版本卡得死，装下去大概率
  把已经跑通的 torchvision / diffusers 一起搞坏。画布对文本的需求是「能对话、能流式」，
  8B 模型用 transformers 直接 generate 完全够，代价只是并发差一点 —— 单机单用户，够用。

接口只做三件事：/v1/models、/v1/chat/completions（stream 与非 stream）、/healthz。
"""
from __future__ import annotations

import faulthandler
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("YJ_TEXT_MODEL", "/root/cloud/models/Qwen3-8B")
MODEL_NAME = os.environ.get("YJ_TEXT_NAME", "qwen3-8b")
PORT = int(os.environ.get("YJ_TEXT_PORT", "8001"))
# 默认上限压住，避免画布一次性塞几万字把显存顶爆
MAX_NEW = int(os.environ.get("YJ_TEXT_MAX_NEW", "2048"))
# 流式总时长上限。单卡上视频任务在跑时文本会被挤慢，给太短会误杀；但不设上限
# 一旦生成线程卡死，连接就会永久挂着（客户端既不超时也不报错，最难查）
STREAM_TIMEOUT = float(os.environ.get("YJ_TEXT_STREAM_TIMEOUT", "900"))
# 两个分片之间最多能安静多久。正常生成是几十毫秒一片；45 秒还没有动静就是真的停了
STREAM_IDLE = float(os.environ.get("YJ_TEXT_STREAM_IDLE", "45"))


def _iter_stream(streamer):
    """
    按分片迭代，但**不给无限等待的机会**。

    为什么不用 `for piece in streamer`：生成线程停住时（或某个 token 之后再也不 put），
    那个写法会永久阻塞 —— 客户端看到的就是「回复说到一半、光标一直闪」：
    既不报错也不超时，用户只能刷新。改成带超时的取队列后，`STREAM_IDLE` 秒没有
    新分片就当作生成已结束，把已经吐出来的内容正常收尾，并在日志里留一行记录。
    """
    q = getattr(streamer, "text_queue", None)
    if q is None:  # 老版本没有这个属性：退回原写法，行为不变
        for piece in streamer:
            yield piece
        return
    while True:
        try:
            piece = q.get(timeout=STREAM_IDLE)
        except queue.Empty:
            log(f"流式停顿 {STREAM_IDLE:.0f}s 没有新分片，主动收尾（生成线程可能已停）")
            return
        if piece is None:
            return
        yield piece

_lock = threading.Lock()
_state: dict = {"model": None, "tok": None, "loading": False, "error": None}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_model():
    """懒加载：第一次请求才占显存，和服务里其他模态的租约模型一致。"""
    with _lock:
        if _state["model"] is not None:
            return _state
        if _state["error"]:
            raise RuntimeError(_state["error"])
        _state["loading"] = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.utils import logging as hf_logging

            # 默认的 "Loading weights: 65%|███" 会刷几千行，把 journalctl / boot.log 淹掉
            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()

            log(f"加载 {MODEL_DIR}")
            tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            # transformers 5.x 把 torch_dtype 改名成 dtype（旧的会打 deprecation 警告），
            # 两个版本都可能在，用 TypeError 兜一层，别把版本差异写成死分支
            common = dict(device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, dtype=torch.bfloat16, **common
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, torch_dtype=torch.bfloat16, **common
                )
            model.eval()
            _state.update(model=model, tok=tok, loading=False)
            log("就绪")
        except Exception as e:  # noqa: BLE001
            _state.update(loading=False, error=f"{type(e).__name__}: {e}")
            log(f"加载失败 {_state['error']}")
            raise
        return _state


def build_prompt(tok, messages):
    """优先用模型自带模板（Qwen3 会处理 think 开关）；模板坏了就退化成朴素拼接。"""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages) + "\nassistant:"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # noqa: A003
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        p = self.path.split("?")[0]
        if p in ("/healthz", "/v1/healthz"):
            # ok 表示「服务能接活」，不表示「权重已进显存」——权重是懒加载的，
            # 拿 loaded 当健康判据会让刚启动的服务看起来是坏的
            return self._json(200, {
                "ok": True,
                "model": MODEL_NAME,
                "loaded": _state["model"] is not None,
                "loading": _state["loading"],
                "error": _state["error"],
            })
        if p == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "local", "kind": "text"},
            ]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            return self._json(400, {"error": "bad json"})

        messages = body.get("messages") or []
        stream = bool(body.get("stream"))
        max_new = min(int(body.get("max_tokens") or MAX_NEW), MAX_NEW)

        try:
            st = ensure_model()
        except Exception as e:  # noqa: BLE001
            return self._json(503, {"error": {"message": str(e), "type": "model_load_failed"}})

        import torch

        model, tok = st["model"], st["tok"]
        prompt = build_prompt(tok, messages)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        created = int(time.time())
        rid = f"chatcmpl-{created}"

        if not stream:
            with torch.inference_mode():
                out = model.generate(
                    **ids, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                    top_p=0.8, repetition_penalty=1.05,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            return self._json(200, {
                "id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": int(ids["input_ids"].shape[1]),
                          "completion_tokens": int(out.shape[1] - ids["input_ids"].shape[1]),
                          "total_tokens": int(out.shape[1])},
            })

        # 流式：用 TextIteratorStreamer 边生成边吐，别等整段生成完
        from transformers import TextIteratorStreamer

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # 这里**必须**让应答体有明确的结束标记。
        #
        # 流式 SSE 既没有 Content-Length 也不用 chunked（每个分片直接 write+flush），
        # HTTP/1.1 下客户端只能靠「连接被关闭」判断结束。而 BaseHTTPRequestHandler 在
        # HTTP/1.1 请求里默认 keep-alive，do_POST 返回后连接不关 —— 客户端于是
        # 收得到文字、收得到 [DONE]，然后永远等下一个字节。
        # 前端表现就是「对话说一半卡住一直转圈」，服务端日志一片干净，最难查。
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def chunk(delta: dict, finish=None) -> None:
            payload = {
                "id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(
            **ids, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.8,
            repetition_penalty=1.05, pad_token_id=tok.eos_token_id, streamer=streamer,
        )

        def _run() -> None:
            """
            生成线程**必须自己兜异常**。

            `TextIteratorStreamer` 只有在 generate 正常结束时才 put 一个 None 收尾。
            生成线程抛异常（显存不够、dtype 不匹配、max_new 算错……）时那个 None
            永远不会来，主线程的 `for piece in streamer` 就永久阻塞 ——
            客户端看到的正是最难排查的那种现象：**连接建立成功、响应头 200、一个字都不来**，
            既不超时也不报错。这里补上 finally：异常照原样打到日志，并主动 end() 收尾。
            """
            try:
                with torch.inference_mode():
                    model.generate(**kw)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001
                    pass

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        chunk({"role": "assistant", "content": ""})

        # 总时长兜底：单卡上视频任务在跑时，文本能被挤到几十秒才出一个字，
        # 所以这里不给「空闲」超时（会误杀正常但很慢的生成），只给一个说得过去的上限。
        # 到点调 `streamer.end()`（公开 API，等价于 put None），主循环就能正常收尾。
        watchdog = threading.Timer(STREAM_TIMEOUT, streamer.end)
        watchdog.daemon = True
        watchdog.start()
        sent = 0
        try:
            for piece in _iter_stream(streamer):
                if piece:
                    chunk({"content": piece})
                    sent += 1
            th.join(timeout=1)
            chunk({}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 前端取消，正常
        except Exception as e:  # noqa: BLE001
            try:
                self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
            except Exception:  # noqa: BLE001
                pass
        finally:
            watchdog.cancel()


def main() -> int:
    log(f"文本服务 {MODEL_NAME} @ {PORT}，权重 {MODEL_DIR}")
    # 生成线程卡住时要能看栈。容器里没有 CAP_SYS_PTRACE（py-spy 用不了），
    # 但进程自己 dump 不需要权限：kill -USR1 <pid> 就把所有线程栈写进 chat.log
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("YJ_TEXT_PRELOAD") == "1":
        try:
            ensure_model()
        except Exception:  # noqa: BLE001
            pass
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
