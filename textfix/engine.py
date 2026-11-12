"""ENGINE end to end: lời chat -> poster. TextFix CHỈ SỬA CHỮ: poster = nháp với chữ viết lại cho rõ.

  with Engine(flux) as E:                                    # flux = flux.Flux("distill").load() (None: chỉ fix() nháp có sẵn)
      B = E.brief(prompt=..., aspect="1:1")                  # LLM: prompt FLUX + câu khách
      r = E.make(B, W, H, seed)                              # FLUX vẽ nháp -> fix()
      r = E.fix(draft, B, seed, ink=None)                    # nét chữ (Hi-SAM) -> dòng -> gỡ chữ (LaMa) -> VLM điền -> dựng -> kiểm
  r: {draft, ink (mặt nạ nét chữ), clean (nền gỡ mọi dòng), base (nền dựng), mask (mặt nạ gỡ chữ), poster, poster_first,
      plan (JSON: dòng, lệnh, lỗi, nhật ký, câu thiếu), timing, M, kept} + ảnh debug (debug_images: steps 5 cột, ink_image, hisam_image)
  B["mode"] == "image" (không chữ khách): make() chỉ FLUX vẽ -> r = {draft, poster (= nháp), plan {mode}, timing}

Hai trạm dùng riêng được (máy chủ chạy dây chuyền nhiều phiên bản song song, server/app.py):
  d = draw(flux, B, W, H, seed, product)          # GPU: nháp -> nét chữ
  r = design(d, B, vlm, page, product=...)        # dòng -> gỡ chữ -> VLM -> rà lệnh -> dựng -> kiểm

14/10: không còn bản xoá FLUX và bước dọn chữ sót (đường đó: nhánh git `legacy-flux-erase`). Dây chuyền: docs/ROADMAP.md.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, kho


class Engine:
    def __init__(self, flux=None, progress=None, llm=None, vlm=None):
        """llm / vlm: hàm gọi tự truyền (thử nghiệm, VLM giả); None: lấy từ biến môi trường (config.models)."""
        self.flux = flux
        self.progress = progress or (lambda msg: None)
        self.llm, self.vlm = llm, vlm

    def __enter__(self) -> "Engine":
        from .brief import SCHEMA
        from .llm import brief_temperature, json_call
        from .render.browser import Browser
        config.load_env()
        self.models = config.models()
        m = self.models
        if self.llm is None:
            self.llm = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA,
                                 temperature=brief_temperature())
        if self.vlm is None:
            self.vlm = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
        self.browser = Browser().__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self.browser.__exit__(*exc)

    def brief(self, prompt: str, product_image: bool = False, aspect: str | None = None) -> dict:
        from .brief import make_brief
        return make_brief(self.llm, prompt, product_image, aspect=aspect)

    def make(self, B: dict, W: int, H: int, seed: int, product: np.ndarray | None = None) -> dict:
        d = draw(self.flux, B, W, H, seed, product, progress=self.progress)
        if B.get("mode") == "image":
            return {"draft": d["draft"], "poster": d["draft"], "plan": {"mode": "image"},
                    "timing": {k: round(v, 1) for k, v in d["timing"].items()}}
        return design(d, B, self.vlm, self.browser.page, product=product is not None, progress=self.progress)

    def fix(self, draft: np.ndarray, B: dict, seed: int = 0, ink: np.ndarray | None = None, product: bool = False) -> dict:
        """product: người dùng tải ảnh sản phẩm (chữ / logo in trên sản phẩm trong nháp -> designer "keep")."""
        d = draw(self.flux, B, draft.shape[1], draft.shape[0], seed, draft=draft, ink=ink, progress=self.progress)
        return design(d, B, self.vlm, self.browser.page, product=product, progress=self.progress)


# ---------------------------------------------------------------------------------------------------- hai trạm
# Máy chủ chạy song song nhiều phiên bản theo DÂY CHUYỀN (server/app.py): trạm VẼ trên luồng của từng GPU (mỗi card một FLUX +
# một Hi-SAM), trạm THIẾT KẾ trên nhóm luồng riêng (OCR, đo dòng, LaMa, VLM ở máy khác, Chromium mỗi luồng một trình duyệt).
def _noop(msg: str) -> None:
    pass


def draw(flux, B: dict, W: int, H: int, seed: int, product: np.ndarray | None = None, draft: np.ndarray | None = None,
         ink: np.ndarray | None = None, progress=_noop, refs: list[np.ndarray] | None = None) -> dict:
    """TRẠM VẼ (GPU): FLUX vẽ nháp -> Hi-SAM tìm nét chữ. draft / ink có sẵn thì bỏ bước tương ứng.
    refs: ảnh tham chiếu dựng sẵn (sửa poster: [poster cũ, ảnh sản phẩm mới?]) thay cho ảnh sản phẩm + ảnh kho.
    -> {draft, ink (bool cỡ nháp), timing}."""
    from .draw import hisam
    T = {}
    if draft is None:
        progress("FLUX vẽ nháp")
        t0 = time.time()
        draft = flux.generate(B["prompt_en"], W, H, seed, refs=(refs if refs is not None else kho.refs_for_draw(B, product)) or None)   # ảnh khách là image 1, rồi ảnh kho
        T["ve_nhap"] = time.time() - t0
    if B.get("mode") == "image":   # ảnh thường (không chữ khách): FLUX vẽ là xong
        return {"draft": draft, "timing": T}
    if ink is None:
        progress("Hi-SAM tìm nét chữ")
        t0 = time.time()
        ink = hisam.strokes(draft, getattr(flux, "dev_dit", None))   # cùng card với bản FLUX của luồng vẽ này
        T["net_chu"] = time.time() - t0
    out = {"draft": draft, "ink": np.asarray(ink, bool), "timing": T,
           "device": getattr(flux, "dev_dit", None)}   # card của luồng vẽ: LaMa ở trạm thiết kế chạy cùng card (chia tải hai card)
    return out


def design(d: dict, B: dict, vlm, page, product: bool = False, progress=_noop, debug: bool = True) -> dict:
    """TRẠM THIẾT KẾ (chặng B, 14/10): dòng (OCR + nét chữ) -> gỡ mọi dòng (LaMa, MỘT lần) -> VLM GHÉP câu khách <-> dòng + kiểu
    theo vai -> code chép nguyên văn, cỡ / lề / tập theo nháp (compose.layout) -> trả điểm ảnh nháp cho dòng giữ -> dựng -> kiểm -> sửa
    bằng code. d = draw(...). debug=False: bỏ ảnh debug (máy chủ dựng sau bằng debug_images, khi đã trả poster)."""
    from . import erase, lines, render
    from .compose import fate
    from .compose import vlm as cvlm
    from .compose.assign import resolve
    from .compose.layout import layout, public
    draft, ink, T = d["draft"], d["ink"], dict(d["timing"])
    texts = B["texts"]
    t0 = time.time()
    progress("đo dòng (OCR + nét chữ)")
    M0 = lines.build(draft, ink, product=product)
    geo_log = list(M0.get("geo_log") or [])
    if (B.get("edit") or {}).get("keep_frame"):   # sửa poster, khung không đổi: chỗ câu cũ nháp bỏ trống -> trả dòng về (edit/slots.py)
        from .edit.slots import restore
        geo_log += restore(M0, texts, draft)
    M = cvlm.view(M0)
    T["dong"] = time.time() - t0
    # GỠ CHỮ (LaMa, GPU) CHẠY SONG SONG VỚI VLM (máy khác, chờ mạng): VLM chỉ nhìn nháp + dòng, không nhìn nền đã gỡ (20/10);
    # nền sạch chỉ cần từ bước bố cục trở đi (12/11: bớt ~1,5 s mỗi poster)
    elog, plates = [], {}

    def lama():
        t1 = time.time()
        out = erase.erase(draft, ink, M, device=d.get("device"), log=elog, plates=plates)
        T["go_chu"] = time.time() - t1
        return out

    with ThreadPoolExecutor(1, thread_name_prefix="lama") as ex:
        f_clean = ex.submit(lama)
        t0 = time.time()
        progress("VLM ghép câu")
        # MỘT lời gọi, khuôn JSON bắt buộc mỗi dòng một mục (compose.vlm.schema): không hỏi lại dòng bỏ sót / dòng không còn chữ /
        # ghép rỗng (người dùng chốt 16/10). Ghép rỗng -> lỗi nặng "empty_plan".
        try:
            A = cvlm.ask(draft, M, B["prompt_en"], texts, vlm, product=product)
        finally:
            T["vlm"] = time.time() - t0
        t1 = time.time()
        progress("gỡ chữ (LaMa)")
        clean, mask0 = f_clean.result()
        elog0 = list(elog)   # nhật ký mặt nạ của lượt gỡ (settle chép lại khi dùng lại mặt nạ)
        T["cho_lama"] = time.time() - t1

    gone = {"mask": None, "clean": None}   # nút / thẻ không còn chữ: xoá cả hình (erase.empty_plates), một lượt LaMa nhỏ

    def base_of(U_, log_=None):
        base_, mask_ = erase.settle(draft, clean, ink, M, {"ops": U_}, log_, done=(mask0, elog0))
        if gone["mask"] is not None:
            base_ = np.where(gone["mask"][..., None], gone["clean"], base_)
            mask_ = mask_ | gone["mask"]
        return base_, mask_

    t0 = time.time()
    wrote = any(u["kind"] == "text" and (u["clients"] or u["text_vlm"]) for u in A.get("assign") or [])
    empty = bool(M["L"] and texts) and not wrote
    llog = []
    if A.get("dropped"):
        llog.append(f"bỏ mục ghép sai khuôn: {A['dropped']}")
    from .edit.style import apply as keep_styles
    from .edit.style import used as styles_used
    slog = keep_styles(A["styles"], (B.get("edit") or {}).get("keep_style") or {})   # sửa poster: giữ kiểu chữ cũ (edit/style.py)
    U, rlog0, miss_i = resolve(A, M, texts, gone=(B.get("edit") or {}).get("gone") or ())
    if plates:
        used = {m for u in U if u["kind"] in ("text", "keep") for m in u["marks"]}
        pm, plog = erase.empty_plates(plates, M, used)
        llog += plog
        if pm is not None:
            t1 = time.time()
            gone["mask"], gone["clean"] = pm, erase._blend(erase.inpaint(clean, pm, d.get("device")), clean, pm)
            T["xoa_the"] = time.time() - t1
    U, llay = layout(U, M, ink, A["styles"], A["accent"], texts, base=clean if gone["clean"] is None else gone["clean"])
    log = geo_log + llog + slog + rlog0 + llay
    T["ghep"] = time.time() - t0
    t0 = time.time()
    progress("dựng + kiểm")
    base, mask = base_of(U, log)
    poster, res, rlog = render.render_units(page, base, U, M)
    errs = render.check_units(U, res)
    if empty:
        errs.append({"type": "empty_plan", "ops": []})
    log += rlog + [f"LỖI: {e}" for e in errs]
    poster_first, errs_first = poster, errs   # bản lượt đầu (debug: so trước / sau khi mã sửa lỗi)
    shown = True   # trang Chromium đang hiện đúng bản giữ (vòng sửa bị bỏ thì trang đang hiện bản sửa)
    for k in range(render.FIX_ROUNDS if render.score(errs)[0] else 0):   # SỬA BẰNG CODE: lỗi nặng đo được -> sửa, dựng lại, giữ khi bớt lỗi
        U2, flog2 = render.autofix_units(U, errs, res)
        if not flog2:
            break
        poster2, res2, _ = render.render_units(page, base, U2, M)
        errs2 = render.check_units(U2, res2)
        ok = render.score(errs2) < render.score(errs)
        shown = ok
        log += [f"-- sửa bằng code {k + 1} --"] + flog2 + [f"LỖI CÒN: {e}" for e in errs2] + \
            ["-> giữ bản sửa" if ok else "-> bản sửa không bớt lỗi, giữ bản trước"]
        if not ok:
            break
        poster, U, errs, res = poster2, U2, errs2, res2
        if not render.score(errs)[0]:
            break
    from .render import finish as rfinish
    if rfinish.enabled() and any(u["kind"] == "text" and u.get("els") for u in U):   # BÓNG ĐỔ MỀM (render/finish.py)
        if not shown:
            render.render_units(page, base, U, M)
        layer = rfinish.text_layer(page, base.shape[1], base.shape[0])
        if layer is not None:
            p2, flog = rfinish.finish(base, layer, U)
            if flog:   # không chữ nào cần bóng: giữ nguyên ảnh chụp trang
                poster, log = p2, log + flog
    T["dung"] = time.time() - t0
    vlm_miss = [k for k in A.get("missing") or [] if 0 <= k < len(texts)]
    if sorted(vlm_miss) != sorted(miss_i):
        log.append(f"câu khách không có trên poster: VLM khai {['T%d' % k for k in vlm_miss]}, đo bằng mã {['T%d' % k for k in miss_i]}")
    missing = [texts[i]["text"] for i in miss_i]
    kept = erase.kept_of({"ops": U})
    # DỰNG THẾ NÀO, từng phần tử (debug "chữ khó"): cỡ, co, nén ngang, xoay, theo đường cong
    rinfo = [{"id": r["id"], "unit": r.get("unit"), "fs": round(float(r.get("fs") or r.get("size") or 0), 1),
              "shrink": round(float(r.get("shrink") or 1), 2), "sx": r.get("sx", 1), "angle": round(float(r.get("angle") or 0), 1),
              "path": bool(r.get("path"))} for r in res or []]
    plan = {"style": A["styles"], "style_used": styles_used(U, A["styles"]), "assign": A["assign"], "accent": A["accent"], "ops": public(U), "fates": fate.table(U, M),
            "render": rinfo,
            "missing": missing, "notes": A.get("notes"), "errors": errs, "log": log, "slots": lines.public(M), "slots_mode": "L",
            "errors_first": errs_first,
            "erase": {"mask_pct": round(100 * float(mask.mean()), 2), "ink_pct": round(100 * float(ink.mean()), 2),
                      "kept": sorted(kept), "orphans": [l["id"] for l in M["L"] if l.get("orphan")], "log": elog}}
    r = {"draft": draft, "ink": ink, "clean": clean, "base": base, "mask": mask, "poster": poster, "poster_first": poster_first,
         "plan": plan, "M": M, "kept": kept, "timing": {k: round(v, 1) for k, v in T.items()}}
    return debug_images(r) if debug else r


def debug_images(r: dict) -> dict:
    """Ảnh debug của một kết quả design() (máy chủ: gọi SAU khi đã trả poster): ink_image (mặt nạ gỡ chữ), hisam_image (nét
    Hi-SAM thô), steps (nháp | dòng | nét + mặt nạ | nền | poster)."""
    from . import erase, lines
    draft = r["draft"]
    ink_img = erase.ink_image(draft, r["ink"], r["mask"], r["M"], r["kept"])
    steps = _row([draft, lines.marked_image(draft, r["M"]), ink_img, r["base"], r["poster"]])
    return {**r, "ink_image": ink_img, "hisam_image": erase.hisam_image(draft, r["ink"]), "steps": steps}


def _row(ims, h: int = 640) -> np.ndarray:
    """nháp | dòng | nét chữ + mặt nạ | nền đã gỡ | poster."""
    from PIL import Image
    ims = [Image.fromarray(np.asarray(x)).convert("RGB") for x in ims]
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    row = Image.new("RGB", (sum(im.width + 10 for im in ims), h), "white")
    x = 0
    for im in ims:
        row.paste(im, (x, 0))
        x += im.width + 10
    return np.asarray(row)
