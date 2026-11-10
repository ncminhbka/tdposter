"""LLM ĐẦU LUỒNG: yêu cầu người dùng (form và / hoặc prompt tự do) -> brief cho cả dây chuyền.

    B = make_brief(llm, prompt, form, design, product_image, n_variants, aspect)
    B = {mode: "poster" | "image", prompt_en, variants: [{name, prompt_en}], texts: [{text, role, source, ordered?}], kho, log}
    kho: id ảnh kho tham chiếu DÙNG THẬT -- engine.draw đưa kèm cho model vẽ sau ảnh sản phẩm. Code tìm ứng viên và chấm độ
         chắc (kho.match_detail); ứng viên không chắc chỉ dùng khi LLM nhắc "image N" (_pick_refs)

Một lời gọi LLM (lời dặn: instructions.py) trả về ý định, danh sách chữ, prompt FLUX và các hướng thiết kế. Code không tin lời
LLM tự khai mà kiểm lại theo bằng chứng:
  texts.py     chốt danh sách chữ: câu nào của khách, câu nào LLM viết thêm, câu nào bịa sự thật (evidence.py đối chiếu);
  prompt.py    sửa mọi prompt cho chữ trong ngoặc kép đúng bằng danh sách đã chốt và vừa giới hạn token của FLUX.
"""

from __future__ import annotations

from .. import kho
from .instructions import ROLES, SCHEMA, SYSTEM, TECH_KEYS, clean_form, request_message, system
from .prompt import cited, compose, count_tokens, renumber, unify_quotes
from .texts import plan_texts

__all__ = ["make_brief", "SCHEMA", "SYSTEM", "ROLES", "count_tokens"]

NO_TEXT = " No text, letters, words or logos anywhere in the image."


def make_brief(llm, prompt: str | None = None, form: dict | None = None, design: dict | None = None,
               product_image: bool = False, n_variants: int = 0, aspect: str | None = None) -> dict:
    """llm(system, [message]) -> dict, thường là llm.json_call(..., schema=SCHEMA).
    n_variants: số hướng thiết kế khác (ảnh 1 = prompt_en, ảnh 2.. = các hướng; cùng danh sách chữ)."""
    raw_form = form or {}
    ordered_lists = raw_form.get("category") == "guide" or bool(raw_form.get("guide_steps"))
    form = clean_form(raw_form)
    hits = {h["id"]: h for h in kho.match_detail(_user_words(prompt, raw_form))}
    cands = [{"n": n, "id": e["id"], "name": kho.name_of(e), "sure": hits[e["id"]]["sure"], "word": hits[e["id"]]["word"]}
             for n, e in kho.numbered(list(hits), product_image)]
    raw = llm(system(bool(form), n_variants), [request_message(prompt, form, design, product_image, n_variants, aspect, cands)])
    if not isinstance(raw, dict):
        raise ValueError(f"LLM trả lời không phải JSON object: {type(raw).__name__}")
    intent = raw.get("intent") if raw.get("intent") in ("poster", "image") else "poster"
    main = unify_quotes(str(raw.get("prompt_en") or ""))
    variants = [{"name": " ".join(str(v.get("name") or "").split())[:60], "prompt_en": unify_quotes(str(v.get("prompt_en") or ""))}
                for v in (raw.get("variants") or []) if isinstance(v, dict) and str(v.get("prompt_en") or "").strip()]
    variants = variants[:max(0, int(n_variants))]
    main, variants, kho_ids, refs, kho_log = _pick_refs(main, variants, cands)

    plan = plan_texts(raw.get("texts"), prompt=prompt, form=form, overridden=raw.get("overridden") or [], prompt_en=main,
                      intent=intent, ordered_lists=ordered_lists)
    log = dict(plan.notes)
    main, notes = compose(main, plan.texts, plan.rewrites, refs)
    log.update(notes)
    for v in variants:
        v["prompt_en"], vn = compose(v["prompt_en"], plan.texts, plan.rewrites, refs)
        if vn:
            log.setdefault("variants", []).append({"name": v["name"], **vn})

    # không câu nào -> chỉ FLUX vẽ (bỏ gỡ chữ / VLM / dựng); LLM tưởng là poster thì dặn thêm "không chữ" để FLUX khỏi vẽ giun
    mode = "poster" if plan.texts else "image"
    log["intent"] = intent
    log.update(kho_log)
    if mode == "image" and intent == "poster":
        main += NO_TEXT
        for v in variants:
            v["prompt_en"] += NO_TEXT
    return {"mode": mode, "prompt_en": main, "texts": plan.texts, "variants": variants, "kho": kho_ids, "log": log}


def _pick_refs(main: str, variants: list[dict], cands: list[dict]):
    """ẢNH KHO DÙNG THẬT (20/10, kết hợp code + LLM): ứng viên CHẮC luôn dùng (LLM quên nhắc thì compose chèn câu nhắc); ứng
    viên KHÔNG CHẮC chỉ dùng khi prompt chính hoặc một hướng thiết kế nhắc "image N" (LLM hiểu "trà sữa cốm" là trà sữa, "com
    tam" là cơm tấm). Bỏ ảnh thì đánh số lại các ảnh sau trong mọi prompt (ảnh sản phẩm image 1 giữ nguyên).
    -> (main, variants, id kho dùng, refs (số mới, tên) của ảnh chắc cho compose, log)."""
    if not cands:
        return main, variants, [], [], {}
    said = cited(main).union(*(cited(v["prompt_en"]) for v in variants))
    keep = [c for c in cands if c["sure"] or c["n"] in said]
    first = cands[0]["n"]
    mapping = {c["n"]: first + k for k, c in enumerate(keep)}
    mapping.update({c["n"]: c["n"] for c in cands if c not in keep})   # số của ảnh bỏ: không còn câu nào nhắc, giữ nguyên
    if any(old != new for old, new in mapping.items()):
        main = renumber(main, mapping)
        for v in variants:
            v["prompt_en"] = renumber(v["prompt_en"], mapping)
    log = {}
    if keep:
        log["kho"] = [f"{c['name']} = image {mapping[c['n']]}" for c in keep]
    sure = [c for c in cands if c["sure"]]
    if sure:
        log["kho_sure"] = [c["name"] for c in sure]
    for key, group in (("kho_llm_kept", [c for c in keep if not c["sure"]]),
                       ("kho_llm_dropped", [c for c in cands if c not in keep])):
        if group:
            log[key] = [f"{c['name']} ('{c['word']}')" for c in group]
    return main, variants, [c["id"] for c in keep], [(mapping[c["n"]], c["name"]) for c in sure], log


def _user_words(prompt: str | None, form: dict) -> str:
    """Lời khách để khớp kho: prompt + giá trị các ô nội dung của form (bỏ trường kỹ thuật như ảnh base64)."""
    vals = [v for k, v in form.items() if k not in TECH_KEYS]
    flat = [str(x) for v in vals for x in (v if isinstance(v, (list, tuple)) else [v]) if isinstance(x, (str, int, float))]
    return " | ".join([prompt or "", *flat])
