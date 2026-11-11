"""THÔNG TIN POSTER GIẤU TRONG FILE PNG: đủ để sửa poster sau này mà máy chủ không phải lưu gì (11/11).

  png = save_png(img, info)       # info = {run_id, parent, depth, aspect, mode, product, texts, prompt_en}
  info = read(raw_bytes)          # None: không phải PNG / không có khối / khối hỏng

Một khối iTXt nén (khoá "textfix", JSON). Chatbot lưu ảnh lên kho rồi gửi lại nguyên byte khi người dùng muốn sửa: còn khối
này = poster của mình, có sẵn danh sách chữ đúng. Ảnh bị nén lại / đổi định dạng thì mất khối -> máy chủ coi là ảnh lạ.
"""

from __future__ import annotations

import io
import json

import numpy as np

KEY = "textfix"
VERSION = 1
FIELDS = ("run_id", "parent", "depth", "aspect", "mode", "product", "texts", "prompt_en")
TEXT_FIELDS = ("text", "role", "source", "ordered", "shown")   # shown: số chỗ câu xuất hiện trên poster, khi > 1 (shown_counts)


def shown_counts(ops: list[dict], texts: list[dict]) -> dict[int, int]:
    """Số chỗ mỗi câu được viết trên poster (plan["ops"]): đơn vị mang câu (client T#) + đơn vị viết lặp đúng chữ câu đó (FLUX vẽ
    lặp, bám nháp). Sửa poster cần biết (11/11 edit1 e03): prompt sửa trích mỗi câu một lần -> FLUX vẽ nút thứ hai TRỐNG chữ."""
    import html
    import re
    flat = lambda s: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip().casefold()   # noqa: E731
    by_text = {flat(t["text"]): k for k, t in enumerate(texts)}
    n: dict[int, int] = {}
    for o in ops:
        if o.get("kind") != "text":
            continue
        ks = [int(c[1:]) for c in o.get("client") or [] if c[1:].isdigit()]
        if not ks and flat(o.get("html")) in by_text:
            ks = [by_text[flat(o.get("html"))]]
        for k in ks:
            n[k] = n.get(k, 0) + 1
    return n


def set_shown(meta: dict, ops: list[dict]) -> dict:
    """Ghi số chỗ xuất hiện (> 1) vào danh sách chữ của khối thông tin, sau khi dựng xong."""
    n = shown_counts(ops, meta.get("texts") or [])
    for k, t in enumerate(meta.get("texts") or []):
        t.pop("shown", None)
        if n.get(k, 0) > 1:
            t["shown"] = n[k]
    return meta


def info(run_id: str, B: dict, aspect: str, product: bool, parent: dict | None = None) -> dict:
    """Thông tin của một poster vừa tạo / vừa sửa (B: brief hoặc bản sửa). parent: thông tin của poster được sửa."""
    texts = [{k: t[k] for k in TEXT_FIELDS if k in t} for t in B.get("texts") or []]
    return {"run_id": run_id, "parent": (parent or {}).get("run_id"), "depth": int((parent or {}).get("depth") or 0) + bool(parent),
            "aspect": aspect, "mode": B.get("mode") or "poster", "product": bool(product), "texts": texts,
            "prompt_en": B.get("prompt_en") or ""}


def save_png(img: np.ndarray, meta: dict | None = None, compress_level: int = 3) -> bytes:
    """PNG (nén mức 3: cùng điểm ảnh, mã hoá nhanh gấp đôi mức 6), kèm khối thông tin khi có."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    kw = {}
    if meta is not None:
        p = PngInfo()
        p.add_itxt(KEY, json.dumps({"v": VERSION, **{k: meta.get(k) for k in FIELDS}}, ensure_ascii=False), zip=True)
        kw["pnginfo"] = p
    b = io.BytesIO()
    Image.fromarray(np.asarray(img)).save(b, "PNG", compress_level=compress_level, **kw)
    return b.getvalue()


def read(raw: bytes) -> dict | None:
    """Khối thông tin của một file ảnh, hoặc None. Không bao giờ ném lỗi."""
    if not raw or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            s = im.info.get(KEY)
        d = json.loads(str(s)) if s else None
    except Exception:
        return None
    if not isinstance(d, dict) or d.get("v") != VERSION or not isinstance(d.get("texts"), list):
        return None
    d["texts"] = [t for t in d["texts"] if isinstance(t, dict) and str(t.get("text") or "").strip()]
    d["depth"] = int(d.get("depth") or 0) if str(d.get("depth") or 0).isdigit() else 0
    return d
