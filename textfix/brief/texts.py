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
    complete_cut_texts(plan, prompt)
    restore_dropped_facts(plan, prompt)
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
        if role in ("step", "list_item") and ORDINAL.match(text) and ORDINAL.sub("", text) and not own_numbering(text, passages):
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


def own_numbering(text: str, passages: list[str]) -> bool:
    """Khách TỰ đánh số mục ("1. Tắm tráng trước khi xuống hồ. 2. ...") -> số là chữ của khách, giữ. (11/11 hard1 h13: LLM để vai
    list_item cho 5 điều -> code bỏ số, điều 6 vai khác giữ "6." -> danh sách lệch.)"""
    m = ORDINAL.match(text)
    head = fold(text[:m.end()].strip() + " " + " ".join(text[m.end():].split()[:2]))
    return any(head in fold(" ".join(p.split())) for p in passages)


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


# câu khách bị LLM cắt cụt ở chữ nối (11/11 edit1: "... part-time cho quán cơm" -> "... part-time cho"): dạng không dấu
CONNECTORS = {"cho", "cua", "va", "voi", "tai", "o", "tu", "den", "gom", "trong", "tren", "duoi", "bang", "cung", "of", "for",
              "and", "with", "at", "to", "in", "from"}
CLAUSE_END = re.compile(r"[,.;:!?)\]]$")
CUT_MAX = 6   # nối thêm tối đa 6 chữ (tới hết vế câu)


def _core(tok: str) -> str:
    return fold(re.sub(r"^[^\w%]+|[^\w%]+$", "", tok))


def complete_cut_texts(plan: TextPlan, prompt: str) -> None:
    """Câu "user" khớp LIỀN một đoạn lời khách (bỏ dấu) mà dừng ở CHỮ NỐI ("cho", "của", "và", "tại"...) trong khi vế câu của
    khách còn tiếp trước dấu , . ; -> nối nốt tới hết vế, bằng chữ của khách. Chỉ đuôi câu: chữ rơi giữa câu / đảo thứ tự không
    phân biệt được với chữ đệm khách được bỏ ("làm poster cho ...") nên không đoán (nhiệt độ thấp lo phần đó)."""
    raw = prompt.split()
    core = [_core(t) for t in raw]
    others = [fold(t["text"]) for t in plan.texts]
    for t in plan.texts:
        words = [w for w in (_core(x) for x in t["text"].split()) if w]
        if t["source"] != "user" or len(words) < 2 or words[-1] not in CONNECTORS:
            continue
        n = len(words)
        at = next((i for i in range(len(core) - n + 1) if core[i:i + n] == words), None)
        if at is None or CLAUSE_END.search(raw[at + n - 1]):
            continue
        tail = []
        for tok in raw[at + n:at + n + CUT_MAX]:
            tail.append(tok)
            if CLAUSE_END.search(tok):
                break
        else:
            if at + n + CUT_MAX < len(raw):   # vế câu dài hơn CUT_MAX chữ: không nối bừa
                continue
        add = " ".join(tail).rstrip(",.;:!?")
        seg = " ".join(raw[at:at + n])
        if not add or any(fold(add) in o for o in others if o != fold(t["text"])):
            continue   # đoạn còn lại đã là câu khác
        if fold(seg) == seg and fold(t["text"]) != t["text"]:
            continue   # khách gõ không dấu, LLM đã thêm dấu: nối chữ không dấu vào thì hỏng câu
        new = f"{t['text']} {add}"
        plan.rewrites.append((f'"{t["text"]}"', f'"{new}"'))
        plan.notes.add("completed", f"{t['text']} -> {new}")
        t["text"] = new


# số khách đưa mà LLM bỏ sót (11/11 sme1: s01 "ban banh mi thit 15k" -> không có 15k; s08 mất giá phòng / điện / nước)
CLAUSE = re.compile(r"[,;](?=\s|$)|\n|(?<!\b\d)(?<!\b\d\d)\.(?=\s|$)|:(?=\s)|\s[-–]\s")   # "6,8%", "1. Tắm" không cắt
WRAPPER = {"lam", "poster", "banner", "anh", "tao", "ve", "thiet", "ke", "giup", "minh", "toi", "cho", "voi", "ban", "mot", "cai",
           "di", "nha", "nhe", "hinh", "quang", "cao", "voi", "la", "ghi", "dong", "chu"}
DESIGN = re.compile(r"\d\s*(?:px|pt)\b|\bfont\b|\b(?:1:1|9:16|16:9|3:4|4:3)\b")   # cụm lời dặn thiết kế: cỡ chữ, khung
LABEL_MAX = 4   # chữ nhãn trước số
AFTER_MAX = 6   # chữ sau số tới hết vế câu ("Thứ 6 Tay", "Giảm 30% tất cả dịch vụ sơn gel"); vế dài hơn: chỉ cụm số
MONEY = re.compile(r"\d\s*(?:k|d|tr|trieu|nghin|vnd)(?![a-z])")   # dạng không dấu: "15k", "3tr5", "25.000d"


def restore_dropped_facts(plan: TextPlan, prompt: str) -> None:
    """Mọi số khách đưa (giá, điện thoại, ngày, giờ, kích thước) phải có trên poster: số không câu nào mang -> thêm một câu
    "user" = chữ của khách trong cùng vế câu: nhãn ngay trước số (<= LABEL_MAX chữ, dừng ở chữ đệm "làm poster ...", bỏ phần
    đã là một câu) + nguyên cụm số + vài chữ sau ("Giá 3tr5/tháng", "Thứ 6 Tay", "15k"). Không đụng số thứ tự đầu mục, số sao,
    lời dặn thiết kế."""
    have = Evidence([t["text"] for t in plan.texts])
    for clause in CLAUSE.split(prompt):
        words = clause.split()
        if DESIGN.search(fold(clause)):
            continue
        for i, w in enumerate(words):
            if not re.search(r"\d", w) or ORDINAL.match(w + " x") or STAR_TEXT.match(" ".join(words[i:i + 2]).strip(".,;:!?")):
                continue
            if not have.unknown_facts(w):
                continue
            j = i
            while j + 1 < len(words) and re.fullmatch(r"\d[\d.,]*", words[j + 1]):
                j += 1   # "0909 123 456": cả dãy số
            fact = " ".join(words[i:j + 1]).strip(".,;:!?)(\"'“”")
            lo = i   # nhãn: lùi tới chữ đệm, chữ mang số khác, hoặc LABEL_MAX chữ
            while lo > 0 and i - lo < LABEL_MAX and _core(words[lo - 1]) not in WRAPPER and not re.search(r"\d", words[lo - 1]):
                lo -= 1
            hi = j + 1   # chữ sau số tới hết vế câu ("Thứ 6 Tay") hoặc số kế tiếp
            while hi < len(words) and hi - j - 1 < AFTER_MAX + 1 and not re.search(r"\d", words[hi]):
                hi += 1
            if hi - j - 1 > AFTER_MAX:
                hi = j + 1
            for t in plan.texts:   # ngay trước số là một câu đã có ("banh mi thit 15k": "Bánh Mì Thịt") -> chỉ thêm số
                n = len(t["text"].split())
                if 0 < n <= i and fold(" ".join(words[i - n:i])) == fold(t["text"]):
                    lo = max(lo, i)
                    break
            head = words[lo:i]
            text = cap_first(" ".join(head + [fact] + words[j + 1:hi]).strip(" ,.;:-–"))
            if not text or plan.has(text):
                continue
            plan.texts.append({"text": text, "role": "price" if MONEY.search(fold(fact)) else "info", "source": "user"})
            plan.notes.add("fact_restored", text)
            have = Evidence([t["text"] for t in plan.texts])


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
