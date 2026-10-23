"""两条依赖链是否还都活着 —— 装 qwen-tts 时降级过 transformers，必须逐条确认。

不写 `try: import x` 的批量循环，是因为这里要的是「具体哪个符号找不到」，
一次性 import 全部只会得到第一个报错，后面的问题还得再跑一轮。
"""

import sys


def probe(label, fn):
    try:
        v = fn()
        print(f"OK   {label}  {v}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {label}  {type(e).__name__}: {e}")
        return False


ok = True
ok &= probe("torch", lambda: __import__("torch").__version__)
ok &= probe("transformers", lambda: __import__("transformers").__version__)
ok &= probe("huggingface_hub", lambda: __import__("huggingface_hub").__version__)
ok &= probe("hub.resolve_revision", lambda: bool(__import__("huggingface_hub").resolve_revision))
ok &= probe("diffusers", lambda: __import__("diffusers").__version__)
ok &= probe("diffusers.QwenImage21Pipeline",
            lambda: bool(__import__("diffusers").QwenImage21Pipeline))
ok &= probe("diffusers.WanImageToVideoPipeline", lambda: bool(
    getattr(__import__("diffusers"), "WanImageToVideoPipeline", None)) or "（无此类，回退 WanPipeline）")
ok &= probe("qwen_tts", lambda: bool(__import__("qwen_tts").Qwen3TTSModel))
ok &= probe("sherpa_onnx", lambda: bool(__import__("sherpa_onnx")))
ok &= probe("soundfile", lambda: bool(__import__("soundfile")))

print("RESULT", "ALL_OK" if ok else "BROKEN")
sys.exit(0 if ok else 1)
