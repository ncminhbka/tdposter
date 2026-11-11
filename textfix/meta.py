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
TEXT_FIELDS = ("text", "role", "source", "ordered")


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
