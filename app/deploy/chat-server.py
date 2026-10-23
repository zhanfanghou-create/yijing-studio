#!/usr/bin/env python
"""本机文本服务：OpenAI 兼容的 /v1/chat/completions（含流式）。

为什么不用 vLLM：
  这台机器的 torch 是 2.14.1+cu130（很新），vLLM 对 torch 版本卡得死，装下去大概率
  把已经跑通的 torchvision / diffusers 一起搞坏。画布对文本的需求是「能对话、能流式」，
  8B 模型用 transformers 直接 generate 完全够，代价只是并发差一点 —— 单机单用户，够用。

接口只做三件事：/v1/models、/v1/chat/completions（stream 与非 stream）、/healthz。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("YJ_TEXT_MODEL", "/root/cloud/models/Qwen3-8B")
MODEL_NAME = os.environ.get("YJ_TEXT_NAME", "qwen3-8b")
PORT = int(os.environ.get("YJ_TEXT_PORT", "8001"))
# 默认上限压住，避免画布一次性塞几万字把显存顶爆
MAX_NEW = int(os.environ.get("YJ_TEXT_MAX_NEW", "2048"))

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
        self.send_header("Connection", "keep-alive")
        self.end_headers()

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
        th = threading.Thread(target=lambda: model.generate(**kw), daemon=True)
        th.start()
        chunk({"role": "assistant", "content": ""})
        try:
            for piece in streamer:
                if piece:
                    chunk({"content": piece})
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


def main() -> int:
    log(f"文本服务 {MODEL_NAME} @ {PORT}，权重 {MODEL_DIR}")
    if os.environ.get("YJ_TEXT_PRELOAD") == "1":
        try:
            ensure_model()
        except Exception:  # noqa: BLE001
            pass
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
