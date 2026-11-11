"""BƯỚC 4b: VLM ĐỌC nháp -- dòng nào là câu khách nào, dòng nào giữ / xoá, mỗi vai trông thế nào.

  A = ask(draft, M, prompt, texts, call)   # MỘT lời gọi cho mọi dòng, khuôn JSON bắt buộc mỗi L# một mục
VLM không đưa toạ độ, chỉ trỏ id dòng; mọi thứ ngoài dòng chữ (vỏ, icon, hoạ tiết) là điểm ảnh của nháp. Số phận cuối cùng của
mỗi dòng: compose/fate.py. Lời dặn có ba số phận: text (chữ poster), keep (chữ trên vật trong ảnh), skip (rác đứng riêng).
"""

from __future__ import annotations

import numpy as np

from .call import _system, plan
from .parse import parse


def view(M: dict) -> dict:
    """Ô để VLM và bộ dựng làm việc: chỉ dòng chữ (plan.json cũ có thể còn S / I / shell: bỏ)."""
    L = [{k: v for k, v in l.items() if k != "shell"} for l in M["L"]]
    return {"L": L, "size": M["size"], "by": {l["id"]: l for l in L}, "geo_log": M.get("geo_log") or []}


def describe(M: dict) -> str:
    note = lambda m: (_curve_note(m) if m.get("path") else f", drawn at {m['angle']:.0f} degrees" if abs(m.get("angle") or 0) >= 2   # noqa: E731
                      else "") + (", found by the stroke detector (big decorated lettering)" if m.get("missed") and not m.get("orphan") else "") \
        + (", STRAY LETTERING (letter shapes that may be junk, a logo or decoration)" if m.get("orphan") else "") \
        + (f", EMPTY PLACE: the previous version of this poster showed {m['restored']} here and the image model left it without "
           f"letters -- give it {m['restored']} unless {m['restored']} already has another line" if m.get("restored") else "")
    # CHỮ OCR GỬI LẠI làm NEO (người dùng 14/10 tối: "đưa OCR hint quay lại nếu thấy performance giảm"; mix1 v03: thiếu chữ OCR,
    # VLM ghép ngày vào mẩu 21px và xoá dòng OCR đọc đúng "01/12 - 07/12"). Chỉ để VLM biết L# nào là dòng nào -- code không
    # dùng chữ này để viết.
    li = "\n".join(f"{m['id']}: machine reading {(m.get('ocr') or '').strip()!r}, ~{m['size']:.0f}px, color {m['color']}{note(m)}"
                   f"{_look(m)}" for m in M["L"]) or "(none)"
    return f"Text lines:\n{li}"


SYSTEM = """You are a senior graphic designer finishing a poster. An image model drew the DRAFT from the PROMPT below. Everything in it \
is good except its lettering, which is garbled and misspelt. Code measured every TEXT LINE of the draft (red L# on SLOTS), \
will wipe the old letters and set the CLIENT TEXTS there itself, word for word, at the size, place and alignment the \
draft shows. YOUR JOB IS TO READ, NOT TO WRITE: say which client text each drawn line was meant to be, and how each kind of \
text looks in the draft.

1. EVERY LINE HAS ITS OWN ENTRY. Look at the DRAFT and SLOTS and read what each marked line was meant to say (the drawn \
letters are garbled: "Hỏ mây ầm" is "Hồ nước ấm"). "lines" holds one entry for every L#. One drawn line holding two client \
texts (a dish name and its price) gets both ("client": ["T5","T6"]), in reading order. Never give one client text to two \
lines, except as parts (rule 2).
2. ONE TEXT OVER SEVERAL LINES (the model broke it over stacked lines, drew a small "THANH TOÁN" over a huge "NHANH CHÓNG", or \
"NGÀY" and "HỘI" in two boxes): each of those lines gets that client text and in "part" the words of it that line shows -- the \
parts in reading order joined must give the client text exactly. A line that shows a whole text has "part": null.
3. EXTRA LINES (the model drew more lines than the client has texts, or repeated one): every drawn line of poster lettering gets words. (a) A repeated row of a list or table gets a client text OF THE SAME ROLE that has no line yet -- never move a text of another role into it; (b) none left: "client": [] with "text": GUESS WHAT THE MODEL MEANT TO WRITE THERE and write it -- read the garbled letters and the machine reading (numbers, prices and short words usually come out right), fit it to the lines around it (another benefit beside benefits, another dish and its price beside dishes) and to that business, in the language and length of the lettering around it; unreadable: a short phrase of the same kind. Give a filled line the "role" it plays (the role of the row it repeats). A filled line never repeats a client text that already has a line, nor a piece of one, and never carries a number, price, time or date that no client text has (code wipes such a line).
4. EVERY LINE GETS ONE KIND. "text": poster lettering -- titles, offers, buttons, badges, stickers, list rows, captions, also \
an extra line on a pill or beside its peers, and ANY line that says one of the client texts wherever it is drawn (on a sign a \
character holds, a board, a banner, a ribbon, a badge): it gets a client text, a filler or nothing as above. "keep": only the \
own lettering of a depicted product or object -- brand and label on a package, logo, screen interface, seal -- even when \
garbled: its pixels stay as drawn; never the poster's message. "skip": only junk standing alone, neither poster lettering nor \
on an object: it is wiped. Unsure between "keep" and "skip": "keep".
5. Client texts the model drew no line for go in "missing". Never cram them into another line.
6. Bullets, ticks, icons, numbers and pictures before a line stay as drawn: never mention them.

STYLES -- one entry per client-text ROLE (the role is written after each T#): how that kind of text looks in the DRAFT. Every \
text of the role is set exactly alike, so describe what the role looks like, not one line: "font" (a key from the catalog \
whose family matches the draft's letterforms -- condensed, rounded, geometric, serif, script, display; a condensed or \
narrow family only when the draft letters are clearly tall and narrow, ordinary lettering gets an ordinary-width family: the \
code narrows a tight line by itself), "weight" (100-900, as heavy as the draft: a heavy headline stays 800-900), \
"italic", "case" ("upper" when the draft shows capitals, else "none"), "tracking" (em, only when clearly spaced) and "effect" \
(CSS only for outlines, shadows, glows or gradient / metallic letters, as strong as the draft: text-shadow, -webkit-text-stroke, \
background-clip:text with a linear-gradient) and "color" (#hex fill of the letters as drawn -- white letters with a black \
outline are "#ffffff" with the outline in "effect").
ACCENT (optional): a word or number the draft draws in another color inside one client text ("50%" in "Giảm 50% hôm nay"): \
{"client": "T#", "text": "<exact substring>", "color": "#hex"}. Sizes come from the draft: never give one.

IMAGES: DRAFT = what the model drew. SLOTS = the draft with every text line marked (red L#). The list \
after the images gives each line's machine reading of the garbled letters (often wrong, but numbers, \
dates and short words usually come out right -- use it to tell which L# is which line), measured size, color and look.

OUTPUT: one JSON object, "lines" with an entry for EVERY L#
{"lines": {"L1": {"kind": "text" | "keep" | "skip", "client": ["T0"] | ["T5","T6"] | [], "part": "..." | null,
                  "text": null | "<words of a filled line>", "role": null | "<role of a filled line>"}, "L2": {...}},
 "styles": {"<role>": {"font": "<key>", "weight": 700, "italic": false, "case": "none", "tracking": null, "color": "#hex", "effect": ""}},
 "accent": [], "missing": ["T5"]}
"text" and "role" are ONLY for a filled line ("client": [], rule 3b): a line with a client text has "text": null and \
"role": null -- the code copies the client text itself, never repeat it.

FONT CATALOG (key: family, class): {fonts}"""


def ask(draft: np.ndarray, M: dict, prompt: str, texts: list[dict], call, product: bool = False,
        meta: dict | None = None) -> dict:
    """MỘT lời gọi cho mọi dòng, ép theo khuôn của poster này (schema: mỗi L# một mục bắt buộc -- VLM không bỏ sót dòng được,
    không cần hỏi lại; người dùng chốt 16/10) -> câu trả lời đã đọc (parse)."""
    raw = plan(draft, M, prompt, texts, call, _system(SYSTEM), describe(M), product=product, meta=meta,
               schema=schema(M, texts))
    return parse(raw, M)


def schema(M: dict, texts: list[dict]) -> dict:
    """Khuôn JSON của câu trả lời cho MỘT poster: "lines" có đúng các khoá L# của poster, mục nào cũng bắt buộc; câu khách chỉ
    trỏ được T# có thật; kiểu theo đúng các vai của câu khách, font trong danh mục."""
    from ..render.fonts import CATALOG
    tids = [f"T{i}" for i in range(len(texts))]
    tid = {"type": "string", "enum": tids} if tids else {"type": "string", "maxLength": 0}
    S = lambda t: {"type": [t, "null"]}   # noqa: E731
    # 12/11: bỏ "why" / "notes" (~20% token sinh ra, code không dùng -- chỉ để đọc debug)
    line = {"type": "object", "additionalProperties": False, "required": ["kind", "client", "part", "text", "role"],
            "properties": {"kind": {"type": "string", "enum": ["text", "keep", "skip"]},
                           "client": {"type": "array", "items": tid, **({} if tids else {"maxItems": 0})},
                           "part": S("string"), "text": S("string"), "role": S("string")}}
    style = {"type": "object", "additionalProperties": False,
             "required": ["font", "weight", "italic", "case", "tracking", "color", "effect"],
             "properties": {"font": {"type": "string", "enum": list(CATALOG)}, "weight": {"type": "integer"},
                            "italic": {"type": "boolean"}, "case": {"type": "string", "enum": ["upper", "none"]},
                            "tracking": S("number"), "color": {"type": "string"}, "effect": {"type": "string"}}}
    roles = list(dict.fromkeys(str(t.get("role") or "") for t in texts if t.get("role")))
    ids = [l["id"] for l in M["L"]]
    return {"type": "object", "additionalProperties": False, "required": ["lines", "styles", "accent", "missing"],
            "properties": {
                "lines": {"type": "object", "additionalProperties": False, "required": ids, "properties": {i: line for i in ids}},
                "styles": {"type": "object", "additionalProperties": False, "required": roles, "properties": {r: style for r in roles}},
                "accent": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["client", "text", "color"],
                                                      "properties": {"client": tid, "text": {"type": "string"}, "color": {"type": "string"}}}},
                "missing": {"type": "array", "items": tid}}}


def _look(m: dict) -> str:
    """Kiểu chữ đo trên nháp: độ đậm (độ dày nét / cỡ chữ), nét dày-mảnh (chữ có chân / viết tay), viết hoa, bóng / viền."""
    out = []
    if m.get("stroke") and m.get("size"):
        r = m["stroke"] / m["size"]   # độ dày nét / em
        out.append("weight " + ("light" if r < 0.07 else "regular" if r < 0.11 else "bold" if r < 0.16 else "black")
                   + f" (stroke ~{m['stroke']:.0f}px)")
    if (m.get("stroke_contrast") or 0) >= 2.2:
        out.append("thick-thin strokes (serif / script)")
    if m.get("caps"):
        out.append("all caps")
    e = m.get("effect")
    if e:
        if abs(e.get("dx", 0)) + abs(e.get("dy", 0)) >= 1:
            out.append(f"drop shadow {e.get('color')} offset ({e['dx']:.0f},{e['dy']:.0f})px")
        else:
            out.append(f"outline / glow {e.get('color')}")
    return (", " + ", ".join(out)) if out else ""


def _curve_note(m: dict) -> str:
    """Ô chữ chạy theo đường cong: vòm lên hay võng xuống, dài bao nhiêu (để VLM viết vừa một dòng)."""
    pt = np.asarray(m["path"], np.float64)
    ln = float(np.sum(np.hypot(*np.diff(pt, axis=0).T)))
    mid, chord = pt[len(pt) // 2], (pt[0] + pt[-1]) / 2
    n = max(2, int(round(ln / max(1.0, 0.55 * (m.get("size") or 1.0)))))
    return (f", CURVED LINE: the letters run along an {'arch (bulging up)' if mid[1] < chord[1] else 'arc bulging down (a smile)'}, "
            f"room for about {n} characters")
