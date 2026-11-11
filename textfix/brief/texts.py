"""Chốt danh sách chữ của poster từ đề xuất của LLM.

Mỗi câu LLM đề xuất được xếp vào một trong ba loại, theo bằng chứng chứ không theo lời LLM tự khai:
  user  mang chữ / sự thật của khách -> bắt buộc lên poster, các bước sau không được bỏ hay viết lại;
  ai    chữ phụ trợ LLM tự viết -> chỉ giữ khi không mang sự thật hay lời hứa nào; các bước sau được rút gọn / bỏ;
  bỏ    có số / liên kết khách không viết, hoặc thêm lời hứa vào câu khách.
Sau đó các bảo đảm không phụ thuộc LLM: chuỗi khách đặt trong ngoặc kép luôn có mặt, số sao là hình chứ không phải chữ, câu
phụ trợ viết hoa kiểu câu (không Viết Hoa Từng Chữ).
Câu nào đổi cách viết thì ghi một cặp thay thế cho prompt_en (`rewrites`), để prompt vẽ đúng các câu đã chốt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .evidence import Evidence, fold
from .instructions import ROLES

USER_COVERAGE = 0.5   # câu "user" phải có >= nửa số chữ trong lời khách; ít hơn là LLM tự viết
PRESENT = 0.8         # chuỗi khách coi là đã có khi >= 80% chữ của nó và mọi số của nó nằm trong các câu đã chốt
SNAP = 0.85           # câu "user" giống >= 85% một đoạn khách cùng số chữ, khác chữ cái = LLM chép sai -> lấy đoạn khách

# chữ phụ trợ (source "ai"): số lượng, độ dài, và những gì không được mang
AI_MAX = 6
AI_MAX_CHARS = 56
AI_BAN = re.compile(r"\d|[%@₫$€£]|https?:|www\.|\.(?:vn|com|net|org)\b", re.I)
PROMISE_WORDS = ("giảm", "tặng", "miễn phí", "khuyến mãi", "khuyến mại", "bảo hành", "cam kết", "hoàn tiền", "trả góp", "đổi trả",
                 "free", "sale", "voucher", "số một", "tốt nhất", "rẻ nhất", "hàng đầu", "chính hãng", "uy tín nhất")

# "Bước 1: ", "1. ", "2) " đầu câu -- không phải giờ / số / khoảng ("6:00", "1.500.000đ", "3 - 5 ngày", final3 h08: "6:00" thành "00")
ORDINAL = re.compile(r"^\s*(?:(?:bước|step)\s*)?\d{1,2}\s*[.):\-–]\s*+(?!\d)", re.I)
STAR_TEXT = re.compile(r"^\d+(?:[.,]\d+)?\s*(?:sao|stars?)$", re.I)   # câu chỉ ghi số sao ("5 sao")
USER_QUOTES = re.compile(r'"([^"]+)"|“([^”]+)”')
QUOTED = re.compile(r'"([^"]+)"')


class Notes(dict):
    """Nhật ký các quyết định (ghi vào brief.json để soi lại): khoá -> danh sách."""

    def add(self, key: str, item) -> None:
        self.setdefault(key, []).append(item)


@dataclass
class TextPlan:
    texts: list[dict] = field(default_factory=list)                 # [{text, role, source, ordered?}]
    rewrites: list[tuple[str, str]] = field(default_factory=list)   # (đoạn trong prompt_en, đoạn thay)
    notes: Notes = field(default_factory=Notes)

    def has(self, text: str) -> bool:
        return any(t["text"] == text for t in self.texts)


def norm(s: str) -> str:
    return " ".join(str(s).split())


def plan_texts(proposed: list, *, prompt: str | None, prompt_en: str, intent: str) -> TextPlan:
    """proposed: texts LLM trả. prompt_en: prompt chính đã thống nhất ngoặc (để nhận chuỗi khách LLM dùng làm từ tả)."""
    prompt = norm(prompt or "")
    passages = [prompt]
    plan = TextPlan()
    vet(proposed, Evidence(passages), passages, plan)
    user_quotes = list(dict.fromkeys(next(g for g in m.groups() if g) for m in USER_QUOTES.finditer(prompt)))
    stars_to_pictures(plan, user_quotes)
    restore_user_quotes(plan, user_quotes, prompt_en)
    if intent == "image" and not any(t["source"] == "user" for t in plan.texts):
        plan.texts = []   # ảnh thường: không tự thành poster vì vài câu phụ trợ
    finalize(plan)
    return plan


def vet(proposed: list, ev: Evidence, passages: list[str], plan: TextPlan) -> None:
    """Xếp từng câu LLM đề xuất: user / ai / bỏ (xem đầu tệp)."""
    n_ai = 0
    for t in proposed or []:
        if not isinstance(t, dict):
            continue
        text, role = norm(t.get("text") or ""), t.get("role") if t.get("role") in ROLES else "body"
        if role in ("step", "list_item") and ORDINAL.match(text) and ORDINAL.sub("", text):
            # đánh số là việc của code (các bước theo thứ tự khách): bỏ số LLM tự viết, cả trong prompt
            plain = ORDINAL.sub("", text)
            plan.rewrites.append((f'"{text}"', f'"{plain}"'))
            plan.notes.add("ordinal_removed", text)
            text = plain
        claimed_user = t.get("source") != "ai"
        if not text or plan.has(text):
            continue
        bad = ev.unknown_facts(text)
        if bad:
            plan.notes.add("dropped", text)
            plan.notes.add("facts_not_from_user", f"{text} ({', '.join(bad)})")
            continue
        cov, extra = ev.coverage(text)
        if claimed_user and extra:
            fixed = snap_to_user(text, passages)
            if fixed:
                plan.notes.add("snapped", f"{text} -> {fixed}")
                plan.rewrites.append((f'"{text}"', f'"{fixed}"'))
                text = fixed
                cov, extra = ev.coverage(text)
        promise = promise_word(" ".join(extra))
        if claimed_user and cov >= USER_COVERAGE:
            if promise:
                plan.notes.add("dropped", text)
                plan.notes.add("ai_rejected", f"{text} (thêm '{promise}')")
                continue
            if extra:
                plan.notes.add("user_cleaned", text)
            source = "user"
        elif ev.facts(text) and not promise:
            plan.notes.add("ai_wraps_user", text)   # câu LLM viết quanh một sự thật của khách ("Ưu đãi 20%")
            source = "user"
        else:
            why = reject_support_copy(text, n_ai)
            if why:
                plan.notes.add("dropped", text)
                plan.notes.add("ai_rejected", f"{text} ({why})")
                continue
            n_ai += 1
            plan.notes.add("ai_added", text)
            source = "ai"
        plan.texts.append({"text": text, "role": role, "source": source})


def snap_to_user(text: str, passages: list[str]) -> str | None:
    """Đoạn khách cùng số chữ giống câu nhất, nếu giống >= SNAP và khác ở CHỮ CÁI (khác dấu / hoa thường là LLM chuẩn hoá, giữ).
    Đánh đổi: lỗi gõ chữ cái của chính khách cũng được giữ, vì không phân biệt được với LLM chép sai."""
    n, best = len(text.split()), (0.0, None)
    for s in passages:
        w = re.sub(r"[\"“”]", " ", s).split()
        for i in range(len(w) - n + 1):
            c = " ".join(w[i:i + n]).strip(" ,.;:!?\"'“”")
            r = SequenceMatcher(None, text.casefold(), c.casefold()).ratio()
            if r > best[0]:
                best = (r, c)
    r, c = best
    if not c or r < SNAP or fold(c) == fold(text):
        return None
    return cap_first(c) if text[:1].isupper() else c


def promise_word(s: str) -> str | None:
    low = " " + re.sub(r"[^\w\s]", " ", s.casefold()) + " "
    return next((w for w in PROMISE_WORDS if f" {w} " in low), None)


def reject_support_copy(text: str, n_have: int) -> str | None:
    """Lý do không nhận một câu phụ trợ, hoặc None."""
    if n_have >= AI_MAX:
        return f"quá {AI_MAX} câu phụ trợ"
    if len(text) > AI_MAX_CHARS:
        return "dài"
    if AI_BAN.search(text):
        return "có số / tiền / liên kết"
    w = promise_word(text)
    return f"có chữ cam kết / ưu đãi '{w}'" if w else None


def stars_to_pictures(plan: TextPlan, user_quotes: list[str]) -> None:
    """"5 sao" là hàng sao vẽ, không phải chữ -- trừ khi khách viết đúng chuỗi đó trong ngoặc kép."""
    literal = {fold(x) for x in user_quotes}
    for t in [t for t in plan.texts if STAR_TEXT.match(t["text"]) and fold(t["text"]) not in literal]:
        plan.texts.remove(t)
        n = re.match(r"\d+", t["text"]).group(0)
        plan.rewrites.append((f'"{t["text"]}"', f"a row of {n} gold rating stars"))
        plan.notes.add("stars_not_text", t["text"])


def restore_user_quotes(plan: TextPlan, user_quotes: list[str], prompt_en: str) -> None:
    """Chuỗi khách đặt trong ngoặc kép luôn lên poster -- trừ khi LLM dùng nó làm TỪ TẢ trong prompt_en (khách đặt ngoặc cho
    phong cách: "Y2K", "vintage")."""
    outside = QUOTED.sub(" ", prompt_en).casefold()
    for q in map(norm, user_quotes):
        if Evidence([t["text"] for t in plan.texts]).holds(q, PRESENT):
            continue
        if re.search(r"(?<!\w)" + re.escape(q.casefold()) + r"(?!\w)", outside):
            plan.notes.add("quoted_as_style", q)
            continue
        plan.texts.append({"text": q, "role": "body", "source": "user"})
        plan.notes.add("added", q)


def finalize(plan: TextPlan) -> None:
    """Vai "step" = mục danh sách có thứ tự (các bước theo thứ tự khách); mục danh sách viết hoa chữ đầu; câu phụ trợ Viết Hoa
    Từng Chữ ("Quán Cà Phê") -> viết hoa kiểu câu ("Quán cà phê": câu phụ trợ không mang tên riêng nào, instructions.py)."""
    for t in plan.texts:
        if t["role"] == "step":
            t["role"], t["ordered"] = "list_item", True
        c = t["text"]
        if t["role"] == "list_item":
            c = cap_first(c)
        if t["source"] == "ai":
            c = sentence_case(c)
        if c != t["text"]:
            plan.rewrites.append((f'"{t["text"]}"', f'"{c}"'))
            plan.notes.add("recased", f'{t["text"]} -> {c}')
            t["text"] = c


def sentence_case(s: str) -> str:
    """"Thưởng Thức Hương Vị" -> "Thưởng thức hương vị": chỉ khi MỌI từ (>= 2 từ) viết hoa chữ đầu mà không phải chữ in hết."""
    w = s.split()
    if len(w) < 2 or s.isupper() or not all(x[:1].isupper() for x in w if x[:1].isalpha()):
        return s
    return " ".join([w[0]] + [x[:1].lower() + x[1:] if x[1:].islower() or len(x) == 1 else x for x in w[1:]])


def cap_first(s: str) -> str:
    """Viết hoa chữ cái đầu (bỏ qua ký hiệu đứng đầu: "- ý"); câu bắt đầu bằng số giữ nguyên ("3 bước")."""
    for i, ch in enumerate(s):
        if ch.isdigit():
            return s
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s
