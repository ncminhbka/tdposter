"""Lời dặn LLM SỬA, khuôn JSON trả lời và thông điệp gửi đi.

Đầu vào: danh sách chữ của poster cũ (T0..), mô tả poster cũ (prompt FLUX đã vẽ nó, hoặc mô tả VLM viết cho ảnh lạ), lời sửa
của người dùng. Đầu ra cùng khuôn chữ với brief, để phần sau của dây chuyền chạy y như tạo mới.
"""

from __future__ import annotations

from ..brief.instructions import ORIENT, ROLES

RATIOS = ["1:1", "9:16", "16:9", "3:4", "4:3"]   # = server/contract.py RATIOS

SYSTEM = """You edit an existing Vietnamese poster for a poster generator built on FLUX.2 by Black Forest Labs.
Input: the texts the current poster shows (T0, T1, ...), a description of the current poster, and ONE chat message in which the user asks for a change. Users are ordinary shop owners: the message is often messy -- lower case, typos, missing diacritics, no quotation marks. The image model receives the current poster as reference image 1 and redraws it from your prompt; code then wipes the drawn letters and sets every text of your list itself, word for word. Output ONE JSON object with:

1. texts -- the COMPLETE list of texts of the edited poster, in reading order. Each: text, role, source, old.
   - Every current text the user does not ask to change: copy it EXACTLY (same words, accents, case, punctuation), with its role, its source and "old" = its number (T3 -> 3). Never rewrite, shorten, translate, re-case or "improve" a text the user did not mention.
   - A text the user changes ("đổi giá thành 199k", "sửa tên quán thành ..."): "old" = its number, the new words in "text", source "user". Change only what the user asked: "đổi giá thành 199k" on "Chỉ 250k/tháng" gives "Chỉ 199k/tháng".
   - A text the user removes ("bỏ dòng ngày", "xoá số điện thoại"): leave it out.
   - A new text the user asks for ("thêm dòng hotline 0909 123 456"): "old" = -1, source "user". Light clean-up only: drop wrapper words ("thêm dòng", "ghi là", "đổi thành"), add missing Vietnamese diacritics, fix capitalization. NEVER change, round or reformat a number, price, percentage, date, time, phone, address, link or proper name; never translate. A quoted string is copied exactly.
   - Support copy (source "ai") only when the user asks you to write it ("thêm slogan", "viết thêm câu kêu gọi"): at most 2 short Vietnamese texts in sentence case, with no number, price, date, contact, name or promise.
   - A request about the picture only (colors, background, objects, people, products, mood) or about the look of the lettering ("chữ to hơn", "đổi font", "chữ màu đỏ") keeps every text unchanged.
   role: one of """ + ", ".join(ROLES) + """.

2. prompt_en -- an English instruction for FLUX.2 that edits reference image 1, ONE paragraph of 40-110 words:
   - Start with "The poster from image 1". State each change concretely (what changes, into what, where it goes if the user said so), then say that everything else stays the same: layout, colors, lettering style, background and products.
   - A new text is an ADDED line (a new row of a list or menu, a new line under the others): say it is added in addition to every current line, that no current line is replaced or removed, and that the layout makes room for it.
   - Only a removed text is gone: when the user removes a text, say that its line disappears; every other current text stays on the poster in its place.
   - Put EVERY text of "texts" in quotation marks exactly once, inside the description of the poster part that carries it (the headline, the price tag, a new line under the shop name, the button). Never put any other words in quotes.
   - A current text marked "shown N times" is written in N places of the current poster (two identical buttons, a name at the top and again at the bottom): when it stays, still quote it ONCE but say in that sentence that every one of those places carries it ('both buttons read "Đặt lịch ngay"'), so no place is left empty.
   - A new product photo attached as image 2 (see the hints): "replace the product with the product from image 2".
   - Never name a font or typeface, never give a text size in numbers, never write aspect ratios or resolutions.

3. aspect -- null, unless the user asks for another canvas shape ("khổ dọc", "khổ ngang", "hình vuông", "story", "banner ngang"): then one of """ + ", ".join(RATIOS) + """, and prompt_en describes the new shape with words (vertical, horizontal, square) and says the layout is rearranged to fit it.
Answer only the JSON object."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["texts", "prompt_en", "aspect"],
          "properties": {"texts": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["text", "role", "source", "old"],
              "properties": {"text": {"type": "string"}, "role": {"type": "string", "enum": ROLES},
                             "source": {"type": "string", "enum": ["user", "ai"]}, "old": {"type": "integer"}}}},
                         "prompt_en": {"type": "string"},
                         "aspect": {"type": ["string", "null"], "enum": [*RATIOS, None]}}}


def request_message(request: str, old: dict, new_product: bool) -> str:
    """Thông điệp gửi LLM: lời sửa + chữ cũ + mô tả cũ + gợi ý kỹ thuật."""
    texts = "\n".join(f"T{i} ({t.get('role') or 'body'}, {t.get('source') or 'user'}"
                      + (f", shown {t['shown']} times" if (t.get("shown") or 1) > 1 else "") + f"): {t['text']}"
                      for i, t in enumerate(old.get("texts") or [])) or "(the current image shows no text)"
    desc = (old.get("prompt_en") or "").strip() or "(no description)"
    hints = []
    if old.get("aspect") in ORIENT:
        hints.append(f"current canvas: {ORIENT[old['aspect']]}")
    if new_product:
        hints.append("the user attached a new photo of their product as reference image 2")
    if old.get("product"):
        hints.append("the current poster shows the user's product photo: its own printed label and logo stay as they are")
    parts = [f"Edit request:\n{request}", f"Current texts:\n{texts}", f"Current poster description:\n{desc}"]
    if hints:
        parts.append("Design hints (not texts, never draw them as words):\n- " + "\n- ".join(hints))
    return "\n\n".join(parts)
