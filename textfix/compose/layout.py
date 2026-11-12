"""BƯỚC 4d -- BỐ CỤC: đơn vị chữ -> phần tử dựng (khung, cỡ, màu, lề, font, tập cùng vai), theo nháp.

  U, log = layout(U, M, ink, styles, accent, texts, base=clean)

VLM chỉ hiểu NGHĨA (dòng nào là câu nào, mỗi vai trông ra sao); code làm phần chính xác:
  1. chữ   = nguyên văn câu khách (compose.assign);
  2. chỗ   = hình bao NÉT Hi-SAM của dòng, nới vào nền trống (compose.size); dòng nghiêng / cong theo render.geometry;
  3. cỡ    = cao nét nháp / tỉ lệ cao chữ của font đã chọn; màu = màu đo của dòng (compose.style); kiểu theo VAI;
  4. lề    = mép nét nháp; ngắt dòng = theo tỉ lệ số ký tự OCR các dòng nháp (compose.text);
  5. TẬP   = cùng vai, cỡ nháp đôi một lệch <= SET_RATIO, thẳng lề / cùng hàng: một cỡ, một lề, một mức nén, một bề rộng cột
             (compose.size, render); một màu khi nháp tô cùng màu (style.one_color);
  6. co    = nén ngang <= render.CONDENSE rồi co cỡ, theo đơn vị (dòng / khối / tập).
Người dùng chốt 14/10: nhóm lấy từ vai câu khách (brief), không đoán từ ảnh; không cắt chữ, chỉ co; nháp thiếu câu thì chịu.
"""

from __future__ import annotations

import html as _html
import re
from collections import Counter

import numpy as np

from .geometry import SPLIT_RATIO, _gaps, _reading, _rows, _stacked, align_of, ink_box, lone_align, set_align
from .size import _line_size, _rooms
from .style import _sanitize, one_color, pick_color
from .text import _caps, break_like

SET_RATIO = 1.25    # tập cùng vai: cỡ nháp đôi một lệch <= 1,25 lần (probe 394b: trung vị +-1,4 gộp nhầm nhãn 22px với website 32px)
TALL = 1.7          # một khung OCR cao > 1,7 cỡ chữ = ôm nhiều dòng chữ (huy hiệu ba dòng 582e)
QUOTES = "\"“”«»'‘’"


# ---------------------------------------------------------------------------------------------------- hình


def layout(U: list[dict], M: dict, ink: np.ndarray, styles: dict, accent: list, texts: list[dict],
           base: np.ndarray | None = None) -> tuple[list[dict], list[str]]:
    """Đơn vị chữ -> số đo + phần tử dựng (u["els"]): khung nét, cỡ, màu, lề, tập, chữ (nguyên văn, ngắt theo nháp). Sửa U tại chỗ."""
    from ..render.fonts import advance
    from ..render.geometry import op_geom, path_of
    from ..render.style import HEAD, WEIGHT, font_key
    by, log = M["by"], []
    med = float(np.median([l["size"] for l in M["L"]])) if M.get("L") else 0.0
    T = [u for u in U if u["kind"] == "text"]
    for u in T:
        # THỨ TỰ ĐỌC: hàng trên xuống, trong hàng trái sang (sắp theo mép trên thì "50%" to đứng trước "Giảm" nhỏ cùng hàng: d07)
        Ls = [by[m] for m in u["marks"]]
        u["Ls"] = [Ls[i] for i in _reading([l["box"] for l in Ls])]
        u["role"] = texts[u["clients"][0]]["role"] if u["clients"] else (u.get("role_vlm") or "fill")
        u["text"] = [texts[k]["text"] for k in u["clients"]] or [u["text_vlm"]]
        u["size"] = float(np.median([l["size"] for l in u["Ls"]]))
        u["bent"] = any(l.get("path") or abs(l.get("angle") or 0) >= 2 or l.get("vertical") for l in u["Ls"])
        u["ib"] = []
        for l in u["Ls"]:
            b = None if u["bent"] else ink_box(ink, l)
            u["ib"].append([b[0], min(b[1], l["box"][1]), b[2], max(b[3], l["box"][3])] if b else list(l["box"]))
        u["box"] = [min(b[0] for b in u["ib"]), min(b[1] for b in u["ib"]), max(b[2] for b in u["ib"]), max(b[3] for b in u["ib"])]
        u["color"] = next((l["color"] for l in u["Ls"] if l.get("color")), None)
        u["color_draft"] = u["color"]   # màu đo trên nét nháp (bộ kiểm tương phản: đúng màu nháp thì giữ, thêm quầng)
        szs = [l["size"] for l in u["Ls"]]
        # CÁCH DỰNG: "lines" = mỗi hàng nháp một phần tử (câu khác cỡ / có split / các dòng không chồng thành khối);
        # "block" = khối nhiều dòng (tự ngắt theo nháp); "line" = một dòng
        if u["bent"]:
            u["mode"] = "bent"
        elif len(u["Ls"]) > 1 and (u["split"] or max(szs) / min(szs) > SPLIT_RATIO or not _stacked(u["Ls"])):
            u["mode"] = "lines"
        elif len(u["Ls"]) > 1 or (u["Ls"][0]["box"][3] - u["Ls"][0]["box"][1]) > TALL * u["size"]:
            u["mode"] = "block"
        else:
            u["mode"] = "line"
    # MỘT DÒNG NHIỀU CÂU (OCR gộp tên món + giá vào một dòng, 14/10 h02): cắt khung nét tại các khe trống rộng nhất giữa các cụm nét
    # -> mỗi câu một đơn vị theo vai của nó (giá vào tập giá, tên vào tập tên); dòng không có khe thì giữ một phần tử
    for u in list(T):
        if u["mode"] != "line" or len(u["clients"]) < 2:
            continue
        cuts = _gaps(ink, u["ib"][0], [texts[k]["text"] for k in u["clients"]], u["size"])
        if not cuts:
            continue
        b, xs = u["ib"][0], [u["ib"][0][0]] + cuts + [u["ib"][0][2]]
        subs = []
        for i, k in enumerate(u["clients"]):
            # khung mẩu ôm đúng nét của nó (không ôm nửa khe): lề / cỡ của mẩu theo chữ nháp (14/10 mix1 t02: tên khoá học căn trái
            # thành căn giữa vì khung mẩu kéo tới giữa khe)
            col = np.flatnonzero(ink[max(0, int(b[1])):int(b[3]), max(0, int(xs[i])):int(xs[i + 1])].any(axis=0))
            o = max(0, int(xs[i]))
            sx0, sx1 = (float(o + col[0]), float(o + col[-1] + 1)) if len(col) else (xs[i], xs[i + 1])
            v = {**u, "id": f"{u['id']}c{i}", "clients": [k], "text": [texts[k]["text"]], "role": texts[k]["role"],
                 "ib": [[sx0, b[1], sx1, b[3]]], "split": None}
            v["box"] = v["ib"][0]
            subs.append(v)
        U[U.index(u):U.index(u) + 1] = subs
        T[T.index(u):T.index(u) + 1] = subs
        log.append(f"{u['id']}: một dòng {len(subs)} câu -> cắt tại khe nét x={[round(c) for c in cuts]}")
    # TẬP CÙNG VAI: cỡ nháp đôi một lệch <= SET_RATIO (gom theo cỡ tăng dần), VÀ vẽ thành một danh sách: thẳng một lề với nhau
    # hoặc cùng hàng (compose.layout.aligned_groups) -- mục lạc chỗ (mẩu chữ ở chỗ khác cùng vai) không thuộc tập, thay cho luật
    # "rời tập khi phải co nhiều" (render.SET_FLOOR, gỡ 16/10: người dùng chốt cùng danh sách thì cùng cỡ, chữ giống hoàn toàn);
    # dòng lấp có vai VLM ghi thì cũng vào tập
    sets = {}
    for role in dict.fromkeys(u["role"] for u in T if u["role"] != "fill"):   # thứ tự gặp (set chuỗi: mỗi lần chạy một thứ tự)
        us = sorted((u for u in T if u["role"] == role and u["mode"] in ("line", "block")), key=lambda u: u["size"])
        grps, grp = [], []
        for u in us:
            if grp and u["size"] / grp[0]["size"] <= SET_RATIO:
                grp.append(u)
            else:
                grps.append(grp)
                grp = [u]
        grps.append(grp)
        for grp in grps:
            for g in aligned_groups(grp):
                sets[f"{role}#{len(sets)}"] = g
    for name, us in sets.items():
        s = float(np.median([l["size"] for u in us for l in u["Ls"]]))
        col = one_color([u["color"] for u in us])
        al = set_align([u["box"] for u in us], s)
        for u in us:
            u.update(set=name, size=s, align=al, **({"color": col} if col else {}))
        log.append(f"tập '{name}': {[u['id'] for u in us]} cỡ {s:.0f}px, lề {al}")
    allb = [(u["id"], b) for u in T for b in u["ib"]]
    for u in T:
        if "align" not in u:
            u["align"] = align_of(u["ib"], u["size"]) or lone_align(u["box"], [b for i, b in allb if i != u["id"]], u["size"])
    # KIỂU THEO VAI (VLM không ghi font / độ đậm: font theo cỡ, độ đậm theo độ dày nét đo được); trong một TẬP: font / độ đậm dùng
    # nhiều nhất (h02: giá cắt ra từ dòng gộp tên + giá đo nét cả dòng -> đậm hơn các giá khác)
    for u in T:
        st = styles.get(u["role"]) or {}
        u["font"] = font_key(st.get("font")) or (st.get("font_display") if med and u["size"] >= HEAD * med else st.get("font_default")) or "bevietnam"
        sw = [l["stroke"] / l["size"] for l in u["Ls"] if l.get("stroke") and l.get("size")]
        u["weight"] = int(st.get("weight") if isinstance(st.get("weight"), (int, float)) else WEIGHT(float(np.median(sw))) if sw else 400)
    for us in sets.values():
        fk, wt = Counter((u["font"], u["weight"]) for u in us).most_common(1)[0][0]
        for u in us:
            u["font"], u["weight"] = fk, wt
    # MÀU: hai trong ba nguồn đồng ý (pick_color) -- màu VLM theo vai không còn ghi đè màu đo từng dòng
    for u in T:
        vc = str((styles.get(u["role"]) or {}).get("color") or "")
        c, why = pick_color(u["Ls"][0].get("color"), (u["Ls"][0].get("ink") or {}).get("color"),
                            vc if re.fullmatch(r"#[0-9a-fA-F]{6}", vc) else None)
        if c and c != u["color"]:
            log.append(f"{u['id']}: màu {c} ({why})")
        u["color"] = c or u["color"]
    for us in sets.values():   # một tập một màu khi nháp tô cùng màu (style.one_color); nháp tô mỗi mục một màu thì giữ
        col = one_color([u["color"] for u in us])
        for u in us:
            u["color"] = col or u["color"]
    for u in T:   # sửa poster: màu cũ của vai giữ nguyên (edit/style.py); chìm nền thì bộ kiểm tương phản đổi như mọi poster
        st = styles.get(u["role"]) or {}
        if st.get("keep_color") and re.fullmatch(r"#[0-9a-fA-F]{6}", str(st.get("color") or "")):
            u["color"] = st["color"]
    for u in T:
        st = styles.get(u["role"]) or {}
        wt = u["weight"]
        case = {"upper": "uppercase", "lower": "lowercase", "title": "capitalize"}.get(str(st.get("case") or "").lower(), "none")
        tr = st.get("tracking")
        outer, inner = _sanitize(st.get("effect"))
        u["css"] = (f"font-weight:{int(wt)};font-style:{'italic' if st.get('italic') else 'normal'};text-transform:{case};"
                    + (f"letter-spacing:{float(tr):.3f}em;" if isinstance(tr, (int, float)) else "") + outer)
        u["inner"] = inner
        # bề rộng chữ theo font thật của đơn vị (ngắt dòng): chữ in hoa đo bản in hoa
        u["meas"] = lambda t, k=u["font"], up=case == "uppercase": advance(k, t.upper() if up else t)  # noqa: E731
        # chữ: nguyên văn; ngoặc kép FLUX vẽ quanh câu trích (OCR thấy dấu ngoặc ở đầu / cuối) -> giữ ngoặc (probe dc0d)
        first, last = (u["Ls"][0].get("ocr") or "").strip(), (u["Ls"][-1].get("ocr") or "").strip()
        if len(u["text"]) == 1 and u["role"] == "review_quote":   # chỉ câu trích (probe k03: mép cuộn ruy băng đọc ra dấu ngoặc)
            t = u["text"][0]
            if first[:1] in QUOTES and t[:1] not in QUOTES:
                t = "“" + t
            if last[-1:] in QUOTES and t[-1:] not in QUOTES:
                t = t + "”"
            u["text"] = [t]
        # SỐ BƯỚC FLUX vẽ trong dòng ("1. Tráng phin", mix1 i02): OCR gộp số vào khung, số bị gỡ cùng chữ -> đặt lại số theo THỨ TỰ
        # CÂU KHÁCH (FLUX hay đánh sai: 1, 3, 4), chỉ cho câu là bước (brief "ordered")
        if len(u["clients"]) == 1 and texts[u["clients"][0]].get("ordered") and re.match(r"\s*\d{1,2}\s*[.)]", first):
            n = 1 + sorted(j for j, t in enumerate(texts) if t.get("ordered")).index(u["clients"][0])
            u["text"] = [f"{n}. {u['text'][0]}"]
        u["accent"] = [a for a in accent if any(str(a.get("text")) in t for t in u["text"])]
    # CỠ THEO FONT THẬT: chữ dựng cao bằng chữ nháp -- cao nét đo / cao chữ của font đã chọn (danh mục chênh 0,59-0,86 em ở chữ
    # hoa: một hằng chung làm chữ thư pháp / chữ hẹp lệch cỡ 20-40%); hoa / thường theo câu khách, không theo chữ OCR đọc
    for u in T:
        u["lsize"] = [_line_size(l, u["font"], " ".join(u["text"]), "text-transform:uppercase" in u["css"]) for l in u["Ls"]]
        if not u.get("set"):
            u["size"] = float(np.median(u["lsize"]))
    for us in sets.values():
        s = float(np.median([x for u in us for x in u["lsize"]]))
        for u in us:
            u["size"] = s
    # PHẦN TỬ DỰNG
    for u in T:
        if u["mode"] == "bent":
            g = op_geom({"kind": "text", "marks": u["marks"]}, M)
            if g is None:
                u["els"] = []
                continue
            pieces = u["split"] or (break_like(" ".join(u["text"]), [max(1, len(l.get("ocr") or "")) for l in u["Ls"]])
                                   if len(u["Ls"]) > 1 and not path_of({"kind": "text", "marks": u["marks"]}, M) else [" ".join(u["text"])])
            u["els"] = [{"id": u["id"], "box": g["rect"], "angle": g["angle"], "path": path_of({"kind": "text", "marks": u["marks"]}, M),
                         "lines": pieces, "size": u["size"], "multi": len(pieces) > 1}]
        elif u["mode"] == "lines":
            pieces = break_like(" ".join(u["text"]), _caps(u), u["meas"])
            rank = list(range(len(u["Ls"])))   # thứ tự trong câu của mẩu ở mỗi dòng
            # một câu VLM ghi thành nhiều mục rồi được gộp (m04 "Du lịch Mây Ngàn -" | "0283 777 8899"): mỗi dòng giữ mẩu của nó
            if u.get("split_ok") and len(u["split"]) == len(u["Ls"]):
                rank = [u["marks"].index(l["id"]) for l in u["Ls"]]
                pieces = [u["split"][i] for i in rank]
            if len(pieces) < len(u["Ls"]):   # câu ít chữ hơn số dòng: dồn vào các dòng đầu
                pieces += [""] * (len(u["Ls"]) - len(pieces))
            u["els"] = []
            szs = u["lsize"]
            even = max(szs) / min(szs) <= SPLIT_RATIO   # các hàng nháp vẽ gần cùng cỡ: một cỡ chung; khác cỡ hẳn: mỗi hàng cỡ của nó
            for gi, g in enumerate(_rows(u["Ls"], u["ib"], u["lsize"])):
                txt = " ".join(p for p in (pieces[i] for i in sorted(g, key=lambda i: rank[i])) if p)
                if not txt:
                    continue
                b = [min(u["ib"][i][0] for i in g), min(u["ib"][i][1] for i in g), max(u["ib"][i][2] for i in g), max(u["ib"][i][3] for i in g)]
                sz = u["size"] if even else float(np.median([u["lsize"][i] for i in g]))
                # hàng mà khung OCR ôm nhiều dòng chữ (huy chương "vỗng / 2025" chung một khung, 14/10 e03): ngắt theo chiều cao
                n = int(round((b[3] - b[1]) / (1.25 * sz))) if b[3] - b[1] > TALL * sz else 1
                parts = break_like(txt, [1.0] * n, u["meas"]) if n > 1 else [txt]
                u["els"].append({"id": f"{u['id']}#{gi}", "box": b, "angle": 0.0, "path": None, "lines": parts, "size": sz,
                                 "multi": len(parts) > 1})
        elif u["mode"] == "block":
            if len(u["text"]) > 1:
                pieces = u["text"]
            elif len(u["Ls"]) > 1:
                pieces = break_like(u["text"][0], _caps(u), u["meas"])
            else:   # một khung OCR ôm nhiều dòng chữ: số dòng ước theo chiều cao
                n = max(1, int(round((u["box"][3] - u["box"][1]) / (1.25 * u["size"]))))
                pieces = break_like(u["text"][0], [1.0] * n, u["meas"])
            u["els"] = [{"id": u["id"], "box": u["box"], "angle": 0.0, "path": None, "lines": pieces, "size": u["size"], "multi": True}]
        else:
            u["els"] = [{"id": u["id"], "box": u["box"], "angle": 0.0, "path": None, "lines": [" ".join(u["text"]) if len(u["text"]) == 1
                         else " ".join(u["text"])], "size": u["size"], "multi": False}]
        for e in u["els"]:
            e["html"] = "<br>".join(_markup(x, u, texts) for x in e["lines"])
        u.pop("meas", None)
    if base is not None:
        _rooms(T, M, base, log, {m for u in U if u["kind"] == "skip" for m in u["marks"]})
    return U, log


def aligned_groups(us: list[dict]) -> list[list[dict]]:
    """Các nhóm >= 2 đơn vị nối nhau bằng quan hệ "cùng danh sách": hai khung thẳng một lề (trái / phải lệch <= EDGE_TOL cỡ, tâm
    lệch <= CENTER_TOL cỡ) hoặc cùng hàng (chồng dọc >= nửa cao khung thấp hơn: các thẻ / nút đứng ngang nhau)."""
    from .geometry import CENTER_TOL, EDGE_TOL

    def linked(a, b):
        A, B, sz = a["box"], b["box"], max(a["size"], b["size"])
        row = min(A[3], B[3]) - max(A[1], B[1]) >= 0.5 * min(A[3] - A[1], B[3] - B[1])
        return (row or abs(A[0] - B[0]) <= EDGE_TOL * sz or abs(A[2] - B[2]) <= EDGE_TOL * sz
                or abs((A[0] + A[2]) / 2 - (B[0] + B[2]) / 2) <= CENTER_TOL * sz)
    comp = list(range(len(us)))

    def root(i):
        while comp[i] != i:
            i = comp[i]
        return i
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            if linked(us[i], us[j]):
                comp[root(j)] = root(i)
    out = {}
    for i, u in enumerate(us):
        out.setdefault(root(i), []).append(u)
    return [g for g in out.values() if len(g) >= 2]


def _markup(text: str, u: dict, texts: list[dict]) -> str:
    """Một dòng chữ -> html: escape; từ nhấn (accent của VLM) -> span cỡ / màu / đậm; chữ tô dải màu -> span ôm chữ."""
    # từ nhấn tìm trong CHỮ (các đoạn chưa nhấn), không trong html đã dựng: 12/11 sme1 s22 nhấn "24" thành span màu #834e2d rồi
    # nhấn "4" khớp vào chính mã màu -> poster in "Thùng 4e2d;>24 lon"
    segs: list[tuple[str, str | None]] = [(text, None)]
    for a in u.get("accent") or []:
        t = str(a.get("text") or "")
        # cỡ phần nhấn không theo VLM: nháp nhấn bằng cỡ thì phần đó là dòng riêng với cỡ đo được (_rows)
        st = (f"color:{a['color']};" if re.fullmatch(r"#[0-9a-fA-F]{3,8}", str(a.get("color") or "")) else "") + \
             (f"font-weight:{int(a['weight'])};" if isinstance(a.get("weight"), (int, float)) else "")
        k = next((i for i, (s, sty) in enumerate(segs) if sty is None and t and t in s), None)
        if not st or k is None:
            continue
        s = segs[k][0]
        j = s.index(t)
        segs[k:k + 1] = [x for x in ((s[:j], None), (t, st), (s[j + len(t):], None)) if x[0]]
    h = "".join(_html.escape(s) if sty is None else f'<span style="{sty}">{_html.escape(s)}</span>' for s, sty in segs)
    if u.get("inner"):
        h = f'<span style="display:inline-block;{_html.escape(u["inner"], quote=True)}">{h}</span>'
    return h


def public(U: list[dict]) -> list[dict]:
    """Đơn vị -> "ops" của plan.json (UI debug đọc id / kind / marks / html như lệnh cũ) + số đo dựng."""
    out = []
    for u in U:
        o = {"id": u["id"], "kind": u["kind"], "marks": u["marks"], "client": [f"T{k}" for k in u["clients"]] or None, "why": u.get("why") or ""}
        if u["kind"] == "text":
            o.update(html="<br>".join(_html.escape(x) for e in u.get("els") or [] for x in e["lines"]), role=u.get("role"),
                     set=u.get("set"), align=u.get("align"), size=round(u.get("size") or 0, 1), mode=u.get("mode"),
                     font=u.get("font"), weight=u.get("weight"), **({"split": u["split"]} if u.get("split") else {}))
        out.append(o)
    return out
