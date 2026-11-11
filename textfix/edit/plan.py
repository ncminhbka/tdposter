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
from .instructions import RATIOS, SCHEMA, SYSTEM, request_message

NO_TEXT = " No text, letters, words or logos anywhere in the image."
AI_NEW_MAX = 2
# lời sửa có ý BỎ một câu (đối chiếu trên dạng không dấu). "đổi / thay / sửa" không tính: "đổi màu nền" không bỏ câu nào,
# còn câu bị thay mà LLM đánh là câu mới thì nhận ra nhờ câu mới cùng vai (_restore)
REMOVE_CUE = re.compile(r"\b(?:bo|xoa|bot|go|loai bo|an|khong can|khong hien|khong ghi|remove|delete)\b")


def make_edit(llm, request: str, old: dict, new_product: bool = False) -> dict:
    """llm(system, [message], schema=...) -> dict (llm.json_call). old: {texts, prompt_en, aspect, product, run_id, depth}."""
    request = norm(request or "")
    raw = llm(SYSTEM, [request_message(request, old, new_product)], schema=SCHEMA)
    if not isinstance(raw, dict):
        raise ValueError(f"LLM trả lời không phải JSON object: {type(raw).__name__}")
    old_texts = [dict(t) for t in old.get("texts") or []]
    plan = vet(raw.get("texts"), old_texts, request)
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
                     "old_texts": old_texts, "changes": plan.notes.get("changes", [])}}


def vet(proposed, old_texts: list[dict], request: str) -> TextPlan:
    """Danh sách chữ LLM trả -> danh sách đã kiểm (xem đầu tệp)."""
    plan = TextPlan()
    ev_new = Evidence([request] + [t["text"] for t in old_texts])
    used: set[int] = set()
    n_ai = 0
    for p in proposed if isinstance(proposed, list) else []:
        if not isinstance(p, dict):
            continue
        text = norm(p.get("text") or "")
        role = p.get("role") if p.get("role") in ROLES else "body"
        k = p.get("old") if isinstance(p.get("old"), int) else -1
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
        plan.texts.append({"text": text, "role": role, "source": src})
        plan.notes.add("changes", f"+ {text!r}")
    _restore(plan, old_texts, used, request)
    for t in plan.texts:
        t.pop("_old", None)
    return plan


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
