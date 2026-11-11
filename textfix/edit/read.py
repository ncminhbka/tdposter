"""ĐỌC ẢNH LẠ (không do hệ thống tạo, không có khối thông tin): chữ trên ảnh -> danh sách chữ, để đi chung luồng sửa (11/11).

    R = read_image(img, ink, vlm)     # ink = hisam.strokes(img)  (trạm VẼ)
    R = {texts: [{text, role, source: "user"}], prompt_en: <mô tả ảnh>, product: bool, lines: n}

Dùng lại nửa đầu dây chuyền: dòng (OCR + nét chữ, lines.build) -> MỘT lời gọi VLM, khuôn JSON bắt buộc mỗi L# một mục: chép
NGUYÊN VĂN chữ của dòng, vai, nhóm (các dòng của cùng một câu), và loại (chữ poster / chữ trên vật / rác). Chữ trên vật
(nhãn, logo sản phẩm) không vào danh sách: bật product để TextFix giữ nguyên điểm ảnh của nó.
"""

from __future__ import annotations

import os

import numpy as np

from ..brief.instructions import ROLES
from ..compose import vlm as cvlm
from ..compose.call import _png
from ..lines.view import marked_image

SYSTEM = """You read the lettering of an image a user uploaded (usually a poster, banner or flyer; sometimes a photo). Code found \
every text line and marked it with a red L# on SLOTS; the list after the images gives each line's machine reading, size and color. \
For EVERY L# give:
- "kind": "text" for the poster's own lettering (titles, offers, prices, dates, contacts, buttons, badges, list rows, captions); \
"keep" for lettering that belongs to a depicted product or object (label or brand on a package, logo on the product, a screen); \
"skip" when the mark holds no real lettering (texture, pattern, an icon).
- "text": the words of that line copied EXACTLY as written, character by character: every Vietnamese diacritic, the same upper / \
lower case, every digit, symbol and punctuation mark. The machine reading helps for numbers but often drops diacritics: trust \
the image. Never correct, translate, complete or summarize. A "skip" line gets "".
- "role": what the line is on the poster (one of the listed roles).
- "group": one sentence broken over several stacked lines shares one group number; every other line gets its own number. \
Groups are numbered in reading order.
Also "description_en": one English sentence of 30-60 words describing the image -- subject, scene, colors, style and layout -- \
without quoting any of its text. Answer only the JSON object."""


def read_side() -> int:
    """Cạnh dài ảnh gửi VLM để ĐỌC chữ (to hơn lúc ghép câu: phải đọc từng dấu): TEXTFIX_VLM_READ_SIDE, mặc định 1024."""
    return int(os.environ.get("TEXTFIX_VLM_READ_SIDE", "1024"))


def schema(M: dict) -> dict:
    ids = [l["id"] for l in M["L"]]
    line = {"type": "object", "additionalProperties": False, "required": ["kind", "text", "role", "group"],
            "properties": {"kind": {"type": "string", "enum": ["text", "keep", "skip"]}, "text": {"type": "string"},
                           "role": {"type": "string", "enum": ROLES}, "group": {"type": "integer"}}}
    return {"type": "object", "additionalProperties": False, "required": ["lines", "description_en"],
            "properties": {"lines": {"type": "object", "additionalProperties": False, "required": ids,
                                     "properties": {i: line for i in ids}},
                           "description_en": {"type": "string"}}}


def read_image(img: np.ndarray, ink: np.ndarray, vlm) -> dict:
    """vlm(system, parts, schema=...) -> dict (llm.json_call)."""
    from .. import lines
    M = cvlm.view(lines.build(img, ink))
    if not M["L"]:   # ảnh không chữ (ảnh chụp): không gọi VLM, sửa thì chỉ thêm chữ / sửa hình
        return {"texts": [], "prompt_en": "", "product": False, "lines": 0}
    H, W = img.shape[:2]
    side = read_side()
    k = min(1.0, side / max(W, H))
    mk = marked_image(img, M, label=max(12, round(14 / k)), width=max(2, round(2 / k)))
    user = f"Image size: {W}x{H}px.\nRoles: {', '.join(ROLES)}.\n\n{cvlm.describe(M)}"
    raw = vlm(SYSTEM, [user, "IMAGE:", _png(img, side), "SLOTS (marked):", _png(mk, side)], schema=schema(M))
    return {**collect(raw, M), "lines": len(M["L"])}


def collect(raw, M: dict) -> dict:
    """Câu trả lời VLM -> danh sách chữ: dòng "text" gộp theo nhóm (thứ tự nhóm, trong nhóm theo thứ tự dòng)."""
    got = raw.get("lines") if isinstance(raw, dict) and isinstance(raw.get("lines"), dict) else {}
    groups: dict[int, list[dict]] = {}
    product = False
    for n, l in enumerate(M["L"]):
        e = got.get(l["id"])
        if not isinstance(e, dict):
            continue
        if e.get("kind") == "keep":
            product = True
            continue
        text = " ".join(str(e.get("text") or "").split())
        if e.get("kind") != "text" or not text:
            continue
        g = e.get("group") if isinstance(e.get("group"), int) else 10 ** 6 + n
        groups.setdefault(g, []).append({"text": text, "role": e.get("role") if e.get("role") in ROLES else "body"})
    texts = [{"text": " ".join(x["text"] for x in v), "role": v[0]["role"], "source": "user"}
             for _, v in sorted(groups.items())]
    desc = str(raw.get("description_en") or "").strip() if isinstance(raw, dict) else ""
    return {"texts": texts, "prompt_en": desc, "product": product}
