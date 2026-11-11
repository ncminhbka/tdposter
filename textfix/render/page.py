"""Dựng các đơn vị chữ lên nền trong Chromium: mỗi phần tử một div (hoặc SVG textPath) tại khung nét nháp.
"""

from __future__ import annotations

import base64
import io
import re
import threading

import numpy as np

from .check import CONTRAST_FRAC, _low_contrast, _recolor
from .js import RENDER_JS

CONDENSE = 0.85   # chữ dài hơn ô theo chiều ngang: nén ngang (scaleX) tới 0.85 TRƯỚC khi co cỡ -- chữ giữ chiều cao như nháp


EMOJI = re.compile("[\U0001F000-\U0001FAFF️‍]")   # emoji (không có font emoji trong bộ font)


def shot(page, W: int, H: int) -> np.ndarray:
    """Ảnh khung #c (đặt ở góc trái trên, viewport đúng cỡ poster). CDP Page.captureScreenshot + optimizeForSpeed: vẫn PNG không
    mất dữ liệu, điểm ảnh trùng locator.screenshot (đo 12/11), chỉ nén zlib nhẹ hơn: ~110 ms thay ~480 ms cho 1 MP. Không có CDP
    (trình duyệt khác Chromium): locator.screenshot."""
    from PIL import Image
    try:
        cdp = getattr(page, "_tf_cdp", None)
        if cdp is None:
            cdp = page._tf_cdp = page.context.new_cdp_session(page)
        r = cdp.send("Page.captureScreenshot", {"format": "png", "optimizeForSpeed": True,
                                                 "clip": {"x": 0, "y": 0, "width": W, "height": H, "scale": 1}})
        raw = base64.b64decode(r["data"])
    except Exception:
        raw = page.locator("#c").screenshot()
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


_URI = threading.local()   # nền vừa mã hoá của luồng này: (mảng, data URI) -- vòng sửa dựng lại trên cùng nền


def to_data_uri(arr: np.ndarray) -> str:
    """Nền -> data URI PNG (nén mức 3: cùng điểm ảnh, nhanh gấp đôi mức 6). Cùng mảng với lần gọi trước trên luồng này -> dùng lại
    (mỗi poster dựng 1-4 lượt trên cùng một nền: đo tập, dựng, vòng sửa)."""
    from PIL import Image
    last = getattr(_URI, "last", None)
    if last is not None and last[0] is arr:
        return last[1]
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG", compress_level=3)
    uri = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    _URI.last = (arr, uri)
    return uri


def plain(h: str) -> str:
    """Chữ trần của HTML (bỏ thẻ, gộp khoảng trắng)."""
    import html as _html
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", h or "")).split())


JUST = {"left": "justify-content:flex-start;text-align:left", "right": "justify-content:flex-end;text-align:right",
        "center": "justify-content:center;text-align:center"}


def _el(u: dict, e: dict, k: int, force: list, size: float | None = None, sx: float | None = None, fit: str | None = None) -> dict:
    """Phần tử của đơn vị -> tham số RENDER_JS. fix (autofix_units): màu / viền tương phản, nén sâu hơn, khung cắt."""
    from .fonts import family
    fix = u.get("fix") or {}
    h = EMOJI.sub("", e["html"])   # không font nào trong bộ font có hình emoji -> ô vuông
    if sx:
        h = (f'<span style="display:inline-block;flex:none;white-space:inherit;transform:scaleX({sx:.3f});'
             f'transform-origin:{u["align"]} center">{h}</span>')
    cls = None
    if fix.get("color"):   # chữ chìm nền -> màu tương phản (+ quầng) cưỡng cho mọi phần tử trong khối
        cls = f"fx{k}"
        force.append(f"#c>.{cls},#c>.{cls} *{{color:{fix['color']}!important;-webkit-text-fill-color:{fix['color']}!important;"
                     + (f"text-shadow:0 0 .08em {fix['halo']},0 0 .16em {fix['halo']},0 0 .3em {fix['halo']}!important;" if fix.get("halo") else "")
                     + "}")
    return {"id": e["id"], "box": (fix.get("boxes") or {}).get(e["id"]) or e["box"], "html": h,
            "style": u["css"] + ";" + JUST[u["align"]] + ";" + ("" if e["path"] else "white-space:nowrap;"),
            "size": size or e["size"], "color": u.get("color") or "#161616", "font": family(u["font"]),
            "fit": fit or fix.get("fit") or "shrink", "angle": e["angle"], "cls": cls, "condense": fix.get("condense"),
            "nl": len(e["lines"]), "hfit": not e["multi"], "path": e["path"], "lead": False}


def render_units(page, plate: np.ndarray, U: list[dict], M: dict, floor: float | None = None) -> tuple[np.ndarray, list, list]:
    """Dựng các đơn vị chữ (compose.layout) lên NỀN (erase.settle) -> (ảnh, kết quả từng phần tử {id, size, shrink, over, ink, ...}, log).
    Một dòng / một khối: nén ngang <= CONDENSE rồi co cỡ (RENDER_JS). TẬP: một mức nén + một cỡ cho mọi mục theo mục chật nhất."""

    from .fonts import faces_for
    H, W = plate.shape[:2]
    T = [u for u in U if u["kind"] == "text" and u.get("els")]
    used = tuple(sorted({u["font"] for u in T} | {"bevietnam"}))
    fl = floor or 0.012 * min(W, H)
    log = []
    page.set_viewport_size({"width": W, "height": H})

    def run(els, force, bg=True):   # bg=False: chỉ đo chữ (bề rộng tự nhiên của tập), khỏi nạp nền
        back = f"background:url('{to_data_uri(plate)}') 0 0/{W}px {H}px no-repeat" if bg else ""
        page.set_content(f"<html><head><style>{faces_for(used)} *{{margin:0;padding:0;box-sizing:border-box}} #c{{position:relative;"
                         f"width:{W}px;height:{H}px;overflow:hidden;{back}}}"
                         f"{''.join(force)}</style></head><body><div id=c></div></body></html>")
        return page.evaluate(RENDER_JS, {"els": els, "floor": fl, "condense": CONDENSE})
    # TẬP: bề rộng tự nhiên của từng mục ở cỡ chung (khung giãn theo chữ) -> mục chật nhất quyết mức nén / cỡ của cả tập
    fitset = {}
    for name in sorted({u["set"] for u in T if u.get("set")}):
        us = [u for u in T if u.get("set") == name and not (u.get("fix") or {}).get("boxes")]
        if not us:
            continue
        nat = {r["id"]: r for r in run([_el(u, e, 0, [], fit="grow") for u in us for e in u["els"]], [], bg=False)}
        # sức chứa từng mục: bề NGANG (nén ngang cứu được) và bề CAO của khối nhiều dòng (chỉ co cỡ cứu được -- cỡ đúng nháp làm khối
        # hai dòng trong vòng tròn cao hơn khung, 15/10 v01)
        def fit_of(u):
            rw = min([(e["box"][2] - e["box"][0]) / max(1.0, nat[e["id"]]["box"][2] - nat[e["id"]]["box"][0])
                      for e in u["els"] if e["id"] in nat] or [1.0])
            rh = min([(e["box"][3] - e["box"][1]) / max(1.0, nat[e["id"]]["ink"][3] - nat[e["id"]]["ink"][1])
                      for e in u["els"] if e["id"] in nat and e["multi"]] or [1.0])
            # (hệ số cỡ: đủ thấp cho bề cao -- 0,98: đo ở cỡ gốc, co xong làm tròn px lệch 1-2 px; phần ngang còn thiếu nén tới
            # CONDENSE; rw)
            return min(1.0, 0.98 * rh if rh < 1 else 1.0, rw / CONDENSE), rw
        # CÙNG TẬP CÙNG CỠ (người dùng chốt 16/10): cả tập co theo mục chật nhất -- không mục nào rời tập. Mục lạc chỗ đã bị loại khi
        # lập tập (compose.layout.aligned_groups), bề rộng cột đã chia đều (compose.size.column_want)
        fits = {u["id"]: fit_of(u) for u in us}
        kk = min(fits[u["id"]][0] for u in us)
        sx = max(CONDENSE, min(1.0, min(fits[u["id"]][1] for u in us) / kk))   # nén ngang ở cỡ chung
        for u in us:
            fitset[u["id"]] = (kk, sx if sx < 0.999 else None)
        log.append(f"tập '{name}': cả tập nén ngang {sx:.2f}, cỡ x{kk:.2f} (theo mục chật nhất)")
    els, force, k = [], [], 0
    for u in T:
        for e in u["els"]:
            if u["id"] in fitset:
                kk, sx = fitset[u["id"]]
                els.append(_el(u, e, k, force, size=e["size"] * kk, sx=sx, fit="none"))
            else:
                els.append(_el(u, e, k, force))
            k += 1
    res = run(els, force)
    size0 = {e["id"]: e["size"] for u in T for e in u["els"]}
    for r in res:
        uid = r["id"].split("#")[0]
        r["unit"] = uid
        if uid in fitset:   # co theo tập: chỉ mục chật nhất là "phải co" -- too_small không tính các mục kéo theo
            r["shrink"], r["sx"], r["evened"] = fitset[uid][0], fitset[uid][1] or 1, True
        elif size0.get(r["id"]):
            r["shrink"] = r["size"] / size0[r["id"]]
        if r["shrink"] < 0.7:
            log.append(f"{r['id']}: co còn {r['shrink']:.2f} để vừa khung")
    img = shot(page, W, H)
    # NỀN THẬT sau chữ -> tỉ lệ tương phản chữ / nền (check_units: low_contrast). Trước 12/11: chụp lại trang với chữ trong suốt;
    # ảnh đó trùng từng điểm ảnh với chính nền dựng (đo 12/11), trừ chữ tô dải màu (background-clip:text vẫn hiện dải màu
    # trong ruột chữ) -- ở đó nền dựng mới là nền thật. Bớt một lượt chụp mỗi lần dựng.
    bg = plate
    color0 = {e["id"]: u.get("color") for u in T for e in u["els"]}
    drawn = {e["id"]: u.get("color_draft") for u in T for e in u["els"]}
    bad = lambda r: (r["low"] or 0) > (0.6 if r.get("px_based") else CONTRAST_FRAC)   # noqa: E731
    for r in res:
        r["low"] = _low_contrast(r, bg, img)
        if len(r.get("runs") or []) > 1 and not bad(r):   # khối nhiều màu: một đoạn chìm là lỗi dù cả khối trung bình đọc được
            for q in r["runs"]:
                q["low"] = _low_contrast(q, bg, img)
                if bad(q):
                    r["low"], r["px_based"] = q["low"], q.get("px_based", False)
                    break
    pal = list(dict.fromkeys(r["fg"] for r in res if r.get("fg") and not bad(r)))   # màu chữ đọc tốt trên poster
    for r in res:
        if bad(r):
            r["recolor"] = _recolor(r, bg, color0.get(r["id"]), pal, drawn=drawn.get(r["id"]))
    return img, res, log
