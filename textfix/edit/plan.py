"""LLM SỬA: poster cũ (chữ + mô tả) + lời sửa -> bản sửa cùng khuôn với brief, để FLUX sửa ảnh cũ và TextFix chạy y như tạo mới.

    B = make_edit(llm, request, old, new_product)
    B = {mode, prompt_en, texts, kho: [], log, edit: {parent, depth, aspect, old_texts, changes}}

Code không tin lời LLM mà kiểm lại (vet):
  - câu cũ giữ chỉ số mà đổi chữ: chữ mới phải mang chữ / số của lời sửa hoặc của câu cũ, không thì trả lại câu cũ;
  - câu mới: số / liên kết phải có trong lời sửa (hoặc chữ cũ), không thì bỏ; câu phụ trợ theo luật của brief;
  - câu cũ biến mất mà lời sửa không có ý bỏ / thay -> trả lại đúng chỗ (LLM chép sót);
rồi prompt.compose: chữ trong ngoặc kép của prompt đúng bằng danh sách đã chốt.
"""

from __future__ import annotations

import re

from ..brief.evidence import Evidence, fold
from ..brief.instructions import ROLES
from ..brief.prompt import IMAGE_REF, compose, unify_quotes
from ..brief.texts import USER_COVERAGE, TextPlan, finalize, norm, reject_support_copy
from .instructions import RATIOS, SYSTEM, request_message, schema_for
from .style import carry

NO_TEXT = " No text, letters, words or logos anywhere in the image."
AI_NEW_MAX = 2
# lời sửa có ý BỎ một câu (đối chiếu trên dạng không dấu). "đổi / thay / sửa" không tính: "đổi màu nền" không bỏ câu nào,
# còn câu bị thay mà LLM đánh là câu mới thì nhận ra nhờ câu mới cùng vai (_restore)
REMOVE_CUE = re.compile(r"\b(?:bo|xoa|bot|go|loai bo|an|khong can|khong hien|khong ghi|remove|delete)\b")


# lời sửa đòi đổi CỠ chữ -> không giữ cỡ lần dựng trước (compose/layout.SIZE_GROW) cho các vai được gọi tên (hoặc mọi vai)
SIZE_CUE = re.compile(r"\b(to hon|lon hon|nho hon|be hon|co chu|chu to|chu nho|phong to|thu nho|to len|nho lai|to ra)\b")


def free_sizes(texts: list[dict], request: str) -> None:
    from .style import named_roles
    f = fold(request or "")
    if not SIZE_CUE.search(f):
        return
    roles = named_roles(f)
    for t in texts:
        if roles is None or t.get("role") in roles:
            t.pop("fs", None)


def make_edit(llm, request: str, old: dict, new_product: bool = False) -> dict:
    """llm(system, [message], schema=...) -> dict (llm.json_call). old: {texts, prompt_en, aspect, product, run_id, depth}."""
    request = norm(request or "")
    raw = llm(SYSTEM, [request_message(request, old, new_product)], schema=schema_for(old))
    if not isinstance(raw, dict):
        raise ValueError(f"LLM trả lời không phải JSON object: {type(raw).__name__}")
    old_texts = [dict(t) for t in old.get("texts") or []]
    plan = vet(raw.get("texts"), old_texts, request, old.get("panels"))
    free_sizes(plan.texts, request)
    finalize(plan)
    pe = unify_quotes(str(raw.get("prompt_en") or ""))
    if not IMAGE_REF.search(pe):   # FLUX phải biết đây là sửa ảnh tham chiếu 1, không phải vẽ mới
        pe = "The poster from image 1, edited: " + pe
    pe, notes = compose(pe, plan.texts, plan.rewrites)
    log = dict(plan.notes)
    log.update(notes)
    mode = "poster" if plan.texts else "image"
    if mode == "image":
        pe += NO_TEXT
    aspect = raw.get("aspect") if raw.get("aspect") in RATIOS else None
    return {"mode": mode, "prompt_en": pe, "texts": plan.texts, "kho": [], "log": log,
            "edit": {"parent": old.get("run_id"), "depth": int(old.get("depth") or 0) + 1, "aspect": aspect,
                     "keep_style": carry(old.get("styles"), request),
                     "old_texts": old_texts, "changes": plan.notes.get("changes", []),
                     # câu cũ không còn nguyên văn trong danh sách mới (bỏ / đổi): FLUX vẽ lại thì không viết (assign._drop_gone)
                     "gone": [o["text"] for o in old_texts if o["text"] not in {t["text"] for t in plan.texts}]}}


def vet(proposed, old_texts: list[dict], request: str, panels: dict | None = None) -> TextPlan:
    """Danh sách chữ LLM trả -> danh sách đã kiểm (xem đầu tệp). panels: ảnh chia ô (textfix/panels.py) -- câu cũ giữ ô cũ, câu
    mới lấy ô LLM ghi (ngoài 0..n -> 0 = chung)."""
    plan = TextPlan()
    ev_new = Evidence([request] + [t["text"] for t in old_texts])
    used: set[int] = set()
    n_ai = 0
    proposed = [p for p in proposed if isinstance(p, dict)] if isinstance(proposed, list) else []
    olds = _match_old(proposed, old_texts)
    for p, k in zip(proposed, olds):
        text = norm(p.get("text") or "")
        role = p.get("role") if p.get("role") in ROLES else "body"
        if not text:
            continue
        if 0 <= k < len(old_texts) and k not in used:
            used.add(k)
            o = old_texts[k]
            if text == o["text"]:
                plan.texts.append({**o, "_old": k})
                continue
            ev = Evidence([request, o["text"]])
            bad = ev.unknown_facts(text)
            cov, extra = ev.coverage(text)
            if bad or cov < USER_COVERAGE:
                plan.texts.append({**o, "_old": k})
                plan.notes.add("reverted", f"T{k}: {text!r} ({'số lạ ' + str(bad) if bad else 'chữ lạ ' + str(extra)})")
                continue
            t = {**o, "text": text, "source": "user", "_old": k}
            if p.get("role") in ROLES:
                t["role"] = role
            plan.texts.append(t)
            plan.notes.add("changes", f"T{k}: {o['text']!r} -> {text!r}")
            continue
        src = p.get("source") if p.get("source") in ("user", "ai") else "user"
        if src == "ai":
            why = "quá số câu phụ trợ" if n_ai >= AI_NEW_MAX else reject_support_copy(text, 0)
            if why:
                plan.notes.add("dropped", f"{text!r} ({why})")
                continue
            n_ai += 1
        else:
            bad = ev_new.unknown_facts(text)
            if bad:
                plan.notes.add("dropped", f"{text!r} (số lạ {bad})")
                continue
            if Evidence([request]).coverage(text)[0] < USER_COVERAGE and not ev_new.holds(text, USER_COVERAGE):
                plan.notes.add("dropped", f"{text!r} (không có trong lời sửa)")
                continue
        panel = {}
        if panels:
            q = p.get("panel")
            panel = {"panel": q if isinstance(q, int) and not isinstance(q, bool) and 0 <= q <= panels["n"] else 0}
        plan.texts.append({"text": text, "role": role, "source": src, **panel})
        plan.notes.add("changes", f"+ {text!r}")
    _retarget(plan, old_texts, request, panels)
    _restore(plan, old_texts, used, request)
    for t in plan.texts:
        t.pop("_old", None)
    return plan


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", fold(NUM.sub(" ", s))))


def _match_old(proposed: list[dict], old_texts: list[dict]) -> list[int]:
    """"old" của từng câu LLM trả -- không tin chỉ số LLM ghi (17/11 panels1 p13: ảnh 4 ô, LLM ghi old = số ô 1..4 thay cho T0..T3,
    mọi câu lệch một): câu trùng nguyên văn một câu cũ -> câu đó; câu đổi -> câu cũ chưa dùng chung nhiều chữ nhất (bỏ số), khi
    chung nhiều hơn câu LLM chỉ; còn lại giữ chỉ số LLM."""
    n = len(old_texts)
    ks = [p.get("old") if isinstance(p.get("old"), int) and not isinstance(p.get("old"), bool) and 0 <= p["old"] < n else -1
          for p in proposed]
    texts = [norm(p.get("text") or "") for p in proposed]
    out = [-1] * len(proposed)
    taken: set[int] = set()
    for i, t in enumerate(texts):   # trùng nguyên văn: ưu tiên chỉ số LLM ghi nếu khớp, không thì câu cũ đầu tiên cùng chữ
        same = [j for j, o in enumerate(old_texts) if o["text"] == t and j not in taken]
        if same:
            out[i] = ks[i] if ks[i] in same else same[0]
            taken.add(out[i])
    for i, t in enumerate(texts):
        if out[i] >= 0 or ks[i] < 0:
            continue
        w = _words(t)
        free = [j for j in range(n) if j not in taken]
        best = max(free, key=lambda j: (len(w & _words(old_texts[j]["text"])), j == ks[i]), default=None)
        mine = len(w & _words(old_texts[ks[i]]["text"])) if ks[i] not in taken else -1
        j = best if best is not None and len(w & _words(old_texts[best]["text"])) > mine else (ks[i] if ks[i] not in taken else -1)
        out[i] = j
        if j >= 0:
            taken.add(j)
    return out


NUM = re.compile(r"\d[\d.,]*\s*(?:k|d|đ|vnd|tr|trieu|%)?(?![\w])", re.I)
# chữ không dùng để nhận ra câu nào được nhắc (động từ sửa, từ chỉ ảnh)
FILLER = {"sua", "doi", "thay", "them", "bo", "xoa", "gia", "thanh", "tren", "anh", "hinh", "nay", "cho", "lai", "la", "nhe", "di",
          "voi", "va", "cua", "o", "trong", "vao", "len"}


def _named(text: str, req: list[str]) -> tuple[int, int]:
    """Lời sửa gọi tên câu này rõ đến đâu: (số cặp từ liền nhau của câu có trong lời sửa, số từ) -- so trên chữ bỏ dấu."""
    w = [x for x in re.findall(r"[a-z]+", fold(NUM.sub(" ", text))) if x not in FILLER]
    pairs = {(a, b) for a, b in zip(req, req[1:])}
    return sum((a, b) in pairs for a, b in zip(w, w[1:])), sum(x in req for x in w)


def _retarget(plan: TextPlan, old_texts: list[dict], request: str, panels: dict | None = None) -> None:
    """ĐỔI SỐ NHẦM CÂU (12/11 quick1 q17: "sua gia tra dao ... thanh 32k" -> LLM đổi "Trà sữa trân châu 25k"): câu bị sửa chỉ khác
    câu cũ ở MỘT con số, mà lời sửa gọi tên một câu khác (chưa sửa, có đúng một con số) rõ hơn -> trả câu kia về cũ, đổi số ở câu
    được gọi tên. Ảnh chia ô: lời sửa gọi ô nào ("ô bên phải", "ô 2") thì chỉ đổi câu trong ô đó -- câu bị sửa nằm ngoài các ô
    được gọi -> đổi sang câu cùng khuôn trong ô được gọi, cả khi tên món như nhau (mỗi ô một "Trà sữa Mây")."""
    from ..panels import named as named_panels
    req = [x for x in re.findall(r"[a-z]+", fold(NUM.sub(" ", request))) if x not in FILLER]
    pn = named_panels(fold(request), panels) if panels else None
    for t in plan.texts:
        k = t.get("_old")
        if k is None or t["text"] == old_texts[k]["text"]:
            continue
        o = old_texts[k]["text"]
        a, b = NUM.findall(o), NUM.findall(t["text"])
        if NUM.sub("#", o) != NUM.sub("#", t["text"]) or len(a) != len(b) or sum(x != y for x, y in zip(a, b)) != 1:
            continue
        new = next(y for x, y in zip(a, b) if x != y)
        mine = _named(o, req)
        outside = bool(pn) and t.get("panel") not in pn
        best = max((u for u in plan.texts if u.get("_old") not in (None, k) and u["text"] == old_texts[u["_old"]]["text"]
                    and len(NUM.findall(u["text"])) == 1 and (not pn or u.get("panel") in pn)),
                   key=lambda u: _named(u["text"], req), default=None)
        if best is None or (not outside and _named(best["text"], req) <= mine):
            continue
        old_num = NUM.findall(best["text"])[0]
        t["text"] = o
        best["text"] = best["text"].replace(old_num, new.strip() + (" " if old_num.endswith(" ") else ""), 1)
        plan.notes.add("retarget", f"lời sửa gọi tên {best['text']!r}, không phải {o!r}: đổi số ở câu được gọi tên")


def _restore(plan: TextPlan, old_texts: list[dict], used: set[int], request: str) -> None:
    """Câu cũ LLM bỏ: lời sửa có ý bỏ, hoặc có câu mới cùng vai (LLM thay câu mà đánh là câu mới) -> ghi là đã bỏ / thay;
    không -> LLM chép sót: trả lại ngay sau câu cũ đứng trước nó."""
    cue = bool(REMOVE_CUE.search(fold(request)))
    new_roles = [t["role"] for t in plan.texts if "_old" not in t]
    for k, o in enumerate(old_texts):
        if k in used:
            continue
        if o.get("role") in new_roles:
            new_roles.remove(o["role"])
            plan.notes.add("changes", f"~ T{k}: {o['text']!r} (thay bằng câu mới cùng vai)")
            continue
        if cue:
            plan.notes.add("changes", f"- T{k}: {o['text']!r}")
            continue
        at = max((i + 1 for i, t in enumerate(plan.texts) if t.get("_old", k) < k), default=0)
        plan.texts.insert(at, {**o, "_old": k})
        plan.notes.add("restored", f"T{k}: {o['text']!r}")
