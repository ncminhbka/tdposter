"""LLM ĐẦU LUỒNG: lời chat của người dùng -> brief cho cả dây chuyền.

    B = make_brief(llm, prompt, product_image, aspect)
    B = {mode: "poster" | "image", prompt_en, texts: [{text, role, source, ordered?}], kho, log}
    kho: id ảnh kho tham chiếu DÙNG THẬT -- engine.draw đưa kèm cho model vẽ sau ảnh sản phẩm. Code tìm ứng viên và chấm độ
         chắc (kho.match_detail); ứng viên không chắc chỉ dùng khi LLM nhắc "image N" (_pick_refs)

Một lời gọi LLM (lời dặn: instructions.py) trả về ý định, danh sách chữ và prompt FLUX. Code không tin lời LLM tự khai mà kiểm
lại theo bằng chứng:
  texts.py     chốt danh sách chữ: câu nào của khách, câu nào LLM viết thêm, câu nào bịa sự thật (evidence.py đối chiếu);
  prompt.py    sửa prompt cho chữ trong ngoặc kép đúng bằng danh sách đã chốt và vừa giới hạn token của FLUX.
"""

from __future__ import annotations

from .. import kho
from .instructions import ROLES, SCHEMA, SYSTEM, request_message
from .prompt import cited, compose, count_tokens, renumber, unify_quotes
from .texts import plan_texts

__all__ = ["make_brief", "SCHEMA", "SYSTEM", "ROLES", "count_tokens"]

NO_TEXT = " No text, letters, words or logos anywhere in the image."


def make_brief(llm, prompt: str, product_image: bool = False, aspect: str | None = None) -> dict:
    """llm(system, [message]) -> dict, thường là llm.json_call(..., schema=SCHEMA)."""
    prompt = (prompt or "").strip()
    hits = {h["id"]: h for h in kho.match_detail(prompt)}
    cands = [{"n": n, "id": e["id"], "name": kho.name_of(e), "sure": hits[e["id"]]["sure"], "word": hits[e["id"]]["word"]}
             for n, e in kho.numbered(list(hits), product_image)]
    raw = llm(SYSTEM, [request_message(prompt, product_image, aspect, cands)])
    if not isinstance(raw, dict):
        raise ValueError(f"LLM trả lời không phải JSON object: {type(raw).__name__}")
    intent = raw.get("intent") if raw.get("intent") in ("poster", "image") else "poster"
    main = unify_quotes(str(raw.get("prompt_en") or ""))
    main, kho_ids, refs, kho_log = _pick_refs(main, cands)

    plan = plan_texts(raw.get("texts"), prompt=prompt, prompt_en=main, intent=intent)
    log = dict(plan.notes)
    main, notes = compose(main, plan.texts, plan.rewrites, refs)
    log.update(notes)

    # không câu nào -> chỉ FLUX vẽ (bỏ gỡ chữ / VLM / dựng); LLM tưởng là poster thì dặn thêm "không chữ" để FLUX khỏi vẽ giun
    mode = "poster" if plan.texts else "image"
    log["intent"] = intent
    log.update(kho_log)
    if mode == "image" and intent == "poster":
        main += NO_TEXT
    return {"mode": mode, "prompt_en": main, "texts": plan.texts, "kho": kho_ids, "log": log}


def _pick_refs(main: str, cands: list[dict]):
    """ẢNH KHO DÙNG THẬT (20/10, kết hợp code + LLM): ứng viên CHẮC luôn dùng (LLM quên nhắc thì compose chèn câu nhắc); ứng
    viên KHÔNG CHẮC chỉ dùng khi prompt nhắc "image N" (LLM hiểu "trà sữa cốm" là trà sữa, "com
    tam" là cơm tấm). Bỏ ảnh thì đánh số lại các ảnh sau trong prompt (ảnh sản phẩm image 1 giữ nguyên).
    -> (main, id kho dùng, refs (số mới, tên) của ảnh chắc cho compose, log)."""
    if not cands:
        return main, [], [], {}
    said = cited(main)
    keep = [c for c in cands if c["sure"] or c["n"] in said]
    first = cands[0]["n"]
    mapping = {c["n"]: first + k for k, c in enumerate(keep)}
    mapping.update({c["n"]: c["n"] for c in cands if c not in keep})   # số của ảnh bỏ: không còn câu nào nhắc, giữ nguyên
    if any(old != new for old, new in mapping.items()):
        main = renumber(main, mapping)
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
    return main, [c["id"] for c in keep], [(mapping[c["n"]], c["name"]) for c in sure], log
