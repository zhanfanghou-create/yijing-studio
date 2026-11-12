"""SDXL 的提示词本地化：中文 → 英文。

**为什么必须有这一层**（实测，不是推测）：

同一个 SDXL-Lightning pipeline、同一个种子、同样 4 步 1024²：

    prompt = "A ginger cat sitting on a sunlit wooden floor, cinematic,
              shallow depth of field"
    → 一张橘猫坐在木地板上的写实照片，光影正确

    prompt = "一只橘猫坐在洒满阳光的木地板上，电影感，浅景深"
    → 一张绿蓝相间的抽象色块，完全没有猫

原因不在模型坏了：SDXL 的两个文本塔（CLIP-L + OpenCLIP-bigG）是纯英文语料训的，
中文被切成字节级 token，条件信息约等于噪声，UNet 只能按先验乱画。
Qwen-Image 用 Qwen3-VL 编码器，原生中英双语，所以**只有低显存档需要翻译**。

模型选 `Helsinki-NLP/opus-mt-zh-en`（MarianMT，298 MB）：CPU 单句 < 1 秒，
不占显存，不依赖别的服务。比「让 8B 对话模型翻译」便宜两个数量级。
"""

import re
import threading

# CJK 统一表意文字（含扩展 A、兼容区）+ 日文假名。前者覆盖简体/繁体，
# 后者是因为中文提示词里常混日系风格词（"赛博朋克" 之类不会，但 "漫画风" 会带假名）。
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")

# 术语表：**先用它替换，再翻译剩下的中文**。
#
# 为什么不能纯靠机器翻译：opus-mt 是新闻语料训的通用翻译模型，不懂图像提示词的
# 行话，「浅景深」被译成 "shallow view" 之后，SDXL 真的会画出「浅的视野」——
# 实测在一张橘猫图上凭空生成了一圈蓝色条块。这类错误不会让图崩，但会污染构图，
# 比完全失败更难发现。把行话钉死成 SDXL 认的词，剩下的自然语言再交给翻译。
#
# 顺序有意义：长词必须排在短词前面（"浅景深" 在 "景深" 之前），否则被截断。
_GLOSSARY = [
    ("电影感", "cinematic"),
    ("电影质感", "cinematic"),
    ("电影级", "cinematic"),
    ("浅景深", "shallow depth of field"),
    ("大景深", "deep depth of field"),
    ("景深", "depth of field"),
    ("背景虚化", "bokeh background"),
    ("虚化", "bokeh"),
    ("广角镜头", "wide angle shot"),
    ("广角", "wide angle"),
    ("长焦", "telephoto lens"),
    ("微距", "macro photography"),
    ("特写", "close-up"),
    ("近景", "medium close-up"),
    ("中景", "medium shot"),
    ("远景", "long shot"),
    ("全景", "panorama"),
    ("俯拍", "top-down view"),
    ("俯视", "high angle"),
    ("仰拍", "low angle shot"),
    ("仰视", "low angle"),
    ("平视", "eye level"),
    ("逆光", "backlight"),
    ("侧光", "side lighting"),
    ("顶光", "top lighting"),
    ("柔光", "soft light"),
    ("硬光", "hard light"),
    ("自然光", "natural lighting"),
    ("黄金时刻", "golden hour"),
    ("蓝调时刻", "blue hour"),
    ("赛博朋克", "cyberpunk"),
    ("蒸汽朋克", "steampunk"),
    ("光伏玻璃", "photovoltaic glass"),
    ("留白构图", "negative space composition"),
    ("留白", "negative space"),
    ("对称构图", "symmetrical composition"),
    ("三分构图", "rule of thirds"),
    ("水墨风格", "ink wash painting style"),
    ("水墨", "ink wash painting"),
    ("国风", "chinese traditional style"),
    ("写实", "photorealistic"),
    ("超写实", "hyperrealistic"),
    ("写实照片", "photorealistic photo"),
    ("插画", "illustration"),
    ("水彩", "watercolor"),
    ("油画", "oil painting"),
    ("素描", "pencil sketch"),
    ("像素风", "pixel art"),
    ("低多边形", "low poly"),
    ("三渲二", "cel shading"),
    ("扁平化", "flat design"),
    ("高细节", "highly detailed"),
    ("超清", "ultra detailed"),
    ("高清", "high resolution"),
    ("8K", "8k"),
    ("电影宽银幕", "cinematic widescreen"),
    ("宽银幕", "widescreen"),
    ("竖构图", "vertical composition"),
    ("横构图", "horizontal composition"),
    ("极简主义", "minimalism"),
    ("极简", "minimalist"),
]
# 长的先替换，避免 "景深" 把 "浅景深" 拆散
_GLOSSARY.sort(key=lambda kv: -len(kv[0]))

_lock = threading.Lock()
_loaded = None  # (tokenizer, model) | False（加载失败，别反复试）
_cache = {}


def has_cjk(text: str) -> bool:
    """这段提示词里有没有中文（含日文假名）。"""
    return bool(_CJK.search(text or ""))


def _clean(text: str) -> str:
    """收尾：去音乐符号、并标点、压空白。

    `♪` 是 opus-mt 的一个真实毛病：把中文短语猜成歌词，输出带音符。
    那几个字符进了 CLIP 只会添乱。
    """
    t = re.sub(r"[♪♫♬♩]+", " ", text)
    t = re.sub(r"\.\s*,", ",", t)  # "sunshine., cinematic" → "sunshine, cinematic"
    t = re.sub(r"\s*([,;])\s*", r"\1 ", t)
    t = re.sub(r"[，、]+", ", ", t)
    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" ,;")


def _split_terms(seg: str):
    """把一段文字按术语切成 [(是否英文, 文本)]。

    为什么必须切开：`_apply_glossary` 那种「整段replace」会造出中英混合的句子，
    而 MarianMT 遇到混合输入会把英文部分直接丢掉 —— 实测「水墨风格的远山与孤舟」
    替换后送进去，出来的是 "♪ In the mountains ♪"，"ink wash painting style"
    整个没了。切干净之后，英文段原样保留，中文段单独翻译。
    """
    hits = []
    for zh, en in _GLOSSARY:
        start = 0
        while True:
            i = seg.find(zh, start)
            if i < 0:
                break
            hits.append((i, i + len(zh), en))
            start = i + len(zh)
    if not hits:
        return [("zh", seg)]
    # **起点相同的优先取长的**：「广角镜头」与「广角」都从 0 开始，
    # 按元组默认排序（短的在前）会让「广角」先占位，把「镜头」甩出来单独翻译成
    # "Camera" —— 实测就是这么出现 "wide angle, Camera" 的。
    hits.sort(key=lambda h: (h[0], h[0] - h[1]))
    picked = []
    last = -1
    for a, b, en in hits:
        if a >= last:  # 跳过与已选区间重叠的（长词优先已由排序保证）
            picked.append((a, b, en))
            last = b
    out = []
    pos = 0
    for a, b, en in picked:
        if a > pos:
            out.append(("zh", seg[pos:a]))
        out.append(("en", en))
        pos = b
    if pos < len(seg):
        out.append(("zh", seg[pos:]))
    return out


def localize(text: str, model_dir) -> str:
    """完整本地化：切术语 → 只翻纯中文片段 → 清洗 → 缓存。

    分段而不是整句丢给翻译模型，因为短句译文明显更准；术语独立成英文段之后，
    也不会再被「二次翻译」毁掉（"cyberpunk" 曾被译成 "Saberpenk"）。
    """
    src = (text or "").strip()
    if not src:
        return src
    if src in _cache:
        return _cache[src]
    if not has_cjk(src):
        return src

    pieces = []
    for seg in re.split(r"[,，、;；。!！?？\n]+", src):
        seg = seg.strip()
        if not seg:
            continue
        for kind, chunk in _split_terms(seg):
            chunk = chunk.strip()
            if not chunk:
                continue
            if kind == "en":
                pieces.append(chunk)
            elif not has_cjk(chunk):
                pieces.append(chunk)
            else:
                pieces.append(translate(chunk, model_dir))

    res = _clean(", ".join(p for p in pieces if p))
    if not res:
        return src
    # 缓存上限：一次创作里提示词去重率很高，但别无限涨
    if len(_cache) > 512:
        _cache.clear()
    _cache[src] = res
    return res


def _load(model_dir):
    """懒加载 MarianMT。失败只试一次 —— 缺 sentencepiece 之类的问题重试也没用。"""
    global _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        try:
            from transformers import MarianMTModel, MarianTokenizer  # noqa: PLC0415

            tok = MarianTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            mdl = MarianMTModel.from_pretrained(str(model_dir), local_files_only=True)
            mdl.eval()
            _loaded = (tok, mdl)
        except Exception:
            _loaded = False
        return _loaded


def preload(model_dir) -> bool:
    """把模型提前载进内存；成功返回 True。

    给 model-server 启动后在后台线程调一次：首次加载要 30–40 秒，
    以前这段时间全部压在用户第一次点「生成」的等待里 ——
    表现是「点了没反应，显卡还空着」。预热带走它，用户感觉不到。
    """
    try:
        return _load(model_dir) is not False
    except Exception:  # noqa: BLE001
        return False


def translate(text: str, model_dir) -> str:
    """中译英。任何一步失败都**原样返回**（由调用方从返回值判断有没有翻成）。

    不抛异常是有意的：翻译退回原文时最坏结果是「图不对」，而抛异常会让
    整个出图失败 —— 用户宁愿拿到一张不听话的图，也不想拿到 500。
    """
    src = (text or "").strip()
    if not src:
        return src

    pair = _load(model_dir)
    if not pair:
        return src

    tok, mdl = pair
    try:
        import torch  # noqa: PLC0415

        # MarianMT 的输入上限 512 token，提示词一般远小于此；超了截断即可
        batch = tok([src], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            # num_beams=1（贪心）而不是 4：
            # beam search 要跑 4 条候选路径，实测把首次出图里 90 秒的翻译段
            # 压到 20 秒上下。提示词是「一句话描述画面」，不是要推敲的译文，
            # 贪心出来的词序对 SDXL 的双 CLIP 完全够用 —— 它本来也只取前 77 token。
            ids = mdl.generate(**batch, max_new_tokens=160, num_beams=1, no_repeat_ngram_size=3)
        out = tok.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return src

    return out or src
