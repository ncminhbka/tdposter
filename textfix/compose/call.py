"""Một lời gọi VLM: lời dặn + PROMPT của FLUX + câu khách + bảng dòng + ba ảnh (nháp, nháp đánh dấu L#, nền đã gỡ chữ).
"""

from __future__ import annotations

import io
import json
import os

import numpy as np

from ..lines.view import marked_image


def max_side() -> int:
    """Cạnh dài ảnh gửi VLM (TEXTFIX_VLM_MAX_SIDE, mặc định 768 -- 12/11; trước: cỡ nháp ~1 MP). VLM chỉ cần nhận ra dòng nào là
    câu nào (có chữ OCR + khung L# kèm theo), không cần đọc từng nét: ảnh nhỏ hơn = ít token đọc vào, máy mã hoá ảnh nhẹ hơn."""
    return int(os.environ.get("TEXTFIX_VLM_MAX_SIDE", "768"))


def _png(im, side: int | None = None) -> bytes:
    from PIL import Image
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    if side and max(im.size) > side:
        k = side / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", compress_level=3)   # cùng điểm ảnh, mã hoá nhanh gấp đôi mức 6
    return b.getvalue()


def _system(base: str) -> str:
    """Lời dặn VLM + danh mục font (khoá, họ font, lớp, bề rộng chữ). Không còn danh mục icon: TextFix chỉ sửa chữ."""
    from ..render.fonts import CATALOG, WIDTH
    fonts = "; ".join(f"{k}: {v[0]} ({v[1]}" + (f", {WIDTH[k][0]:.2f} / caps {WIDTH[k][1]:.2f}em" if k in WIDTH else "") + ")"
                      for k, v in CATALOG.items())
    return base.replace("{fonts}", fonts)


def _obj(x) -> dict:
    """Đầu ra VLM về dict: đôi khi trả JSON lồng trong chuỗi, hoặc mảng."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except ValueError:
            return {}
    if isinstance(x, list):
        x = {"ops": x}
    return x if isinstance(x, dict) else {}


def _texts(texts: list[dict]) -> str:
    # câu "ai" = chữ phụ trợ bước brief tự viết (tiêu đề / khẩu hiệu / nút / nhãn khi khách chỉ điền sự thật): không phải lời khách
    return "\n".join(f"T{i} ({t.get('role', '')}{', OURS' if t.get('source') == 'ai' else ''}): {t['text']}"
                     for i, t in enumerate(texts))


PRODUCT_NOTE = ("The client uploaded a photo of their product and the image model copied it into the draft. Text, logos and "
                "graphics printed on that product (label, package, screen) are part of the photo and MUST stay exactly as they "
                "are: give every slot on the product kind \"keep\" -- never rewrite, correct, translate, redraw or skip it, "
                "and never write a client text there, even when the product's name is one of the client texts.")


def plan(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, system: str, desc: str,
         product: bool = False, meta: dict | None = None, schema: dict | None = None) -> dict:
    """Một lời gọi VLM: lời dặn (system), PROMPT của FLUX, câu khách, bảng mô tả dòng (desc), hai ảnh DRAFT / SLOTS.
    Không gửi nền đã gỡ chữ (20/10): VLM chỉ đọc và trỏ id dòng; nền đó gỡ cả chữ trên sản phẩm khách tải, nhìn như nhãn trống
    để viết vào (dev2 d13).
    meta: nhận số token / thời gian (llm.json_call); schema: khuôn JSON của lời gọi. -> câu trả lời JSON (compose.parse đọc)."""
    H, W = draft.shape[:2]
    side = max_side()
    k = min(1.0, side / max(W, H))
    # nhãn L# và khung vẽ ở cỡ gốc rồi thu cùng ảnh: phóng to theo tỉ lệ thu để sau khi thu vẫn ~14 px, nét ~2 px
    mk = marked_image(draft, M, label=max(12, round(14 / k)), width=max(2, round(2 / k)))
    imgs = ["DRAFT:", _png(draft, side), "SLOTS (marked):", _png(mk, side)]
    shown = "" if k >= 1 else f" (images shown at {round(W * k)}x{round(H * k)}; every size in px below is at full size)"
    user = (f"PROMPT given to the image model:\n{prompt}\n\nClient texts (exact words):\n{_texts(texts)}\n\n"
            f"Poster size: {W}x{H}px{shown}.\n\n"
            + (PRODUCT_NOTE + "\n\n" if product else "") + desc)
    kw = {"schema": schema} if schema else {}
    return _obj(call(system, [user, *imgs], meta, **kw) if meta is not None else call(system, [user, *imgs], **kw))


def _tid(c) -> int | None:
    c = str(c or "")
    return int(c[1:]) if c[:1] in "Tt" and c[1:].isdigit() else int(c) if c.isdigit() else None
