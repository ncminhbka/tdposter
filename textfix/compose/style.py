"""MÀU và hiệu ứng: màu đo trên nét nháp đối chiếu màu VLM (hai trong ba nguồn), lọc CSS hiệu ứng VLM đưa.
"""

from __future__ import annotations

import re

import numpy as np

COLOR_AGREE = 25.0  # hai nguồn màu đồng ý: lệch <= 25 ΔE


EFFECT_KEYS = re.compile(r"^\s*(-webkit-)?(text-shadow|text-stroke(-width|-color)?|background(-image|-clip|-color)?|"
                         r"text-fill-color|filter|paint-order)\s*:", re.I)


def _rgb(hx) -> tuple | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hx or ""))
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None


def _sanitize(css: str) -> tuple[str, str]:
    """Hiệu ứng VLM ghi cho một vai (CSS) -> (phần đặt ở khung, phần đặt ở span ôm chữ). Chỉ nhận bóng / viền / dải màu; bỏ url()."""
    keep = [x.strip() for x in str(css or "").split(";") if EFFECT_KEYS.match(x) and "url(" not in x.lower() and "expression" not in x.lower()]
    grad = [x for x in keep if re.match(r"\s*(-webkit-)?(background|text-fill-color)", x, re.I)]
    clip = any("clip" in x.lower() and "text" in x.lower() for x in grad)
    # CHỮ TÔ DẢI MÀU (background-clip:text) phải ở span ôm chữ: ở khung thì dải màu chỉ phủ trong khung, nét tràn ra trong suốt
    # (09/10 dev3 g14, probe b21a "An khang thịnh vượng" mất hai đầu)
    # ruột chữ TRONG SUỐT thì dải màu mới hiện (VLM hay quên: 12/11 golden r07 ghi dải màu + "color" -> ruột màu đặc phủ kín dải
    # màu; viền vẽ đè ruột từng che bớt, viền vẽ dưới ruột -- render/js.py -- làm chữ thành một màu nâu)
    if clip and not any(re.match(r"\s*-webkit-text-fill-color", x, re.I) for x in grad):
        grad.append("-webkit-text-fill-color:transparent")
    if clip:   # viền / bóng của chữ dải màu -> filter drop-shadow (vẽ DƯỚI chữ); text-shadow / text-stroke vẽ ĐÈ lên dải màu
        fx = [x for x in keep if re.match(r"\s*(-webkit-)?(text-shadow|text-stroke)", x, re.I)]
        keep = [x for x in keep if x not in fx]
        drops = _drop_shadows(fx)
        if drops:
            grad += ["text-shadow:none", "-webkit-text-stroke:0", "filter:" + " ".join(drops)]
    inner = ";".join(grad) if clip else ""
    outer = ";".join(x for x in keep if not (clip and x in grad))
    return outer, inner


def _drop_shadows(fx: list[str]) -> list[str]:
    """text-stroke (độ dày + màu) và text-shadow -> chuỗi drop-shadow: viền = 4 lần dời +-w (vòng vuông bán kính w quanh chữ),
    rồi các bóng giữ nguyên lệch / nhoè / màu."""
    out = []
    w, col = None, None
    for x in fx:
        k, _, v = x.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k.endswith("text-stroke-width"):
            w = v
        elif k.endswith("text-stroke-color"):
            col = v
        elif k.endswith("text-stroke"):
            m = re.match(r"([\d.]+px)\s+(.+)", v)
            if m:
                w, col = m.group(1), m.group(2)
    if w and col and re.fullmatch(r"[\d.]+px", w) and float(w[:-2]) > 0:
        out += [f"drop-shadow({w} 0 0 {col})", f"drop-shadow(-{w} 0 0 {col})", f"drop-shadow(0 {w} 0 {col})",
                f"drop-shadow(0 -{w} 0 {col})"]
    for x in fx:
        k, _, v = x.partition(":")
        if k.strip().lower() != "text-shadow" or v.strip().lower() == "none":
            continue
        for sh in re.split(r",(?![^()]*\))", v):
            parts = re.findall(r"rgba?\([^)]*\)|hsla?\([^)]*\)|#[0-9a-fA-F]{3,8}|[-\d.]+(?:px|em)?|[a-zA-Z]+", sh.strip())
            lens = [p for p in parts if re.fullmatch(r"[-\d.]+(?:px|em)?", p)]
            cols = [p for p in parts if p not in lens]
            if len(lens) >= 2 and cols:
                out.append(f"drop-shadow({' '.join((lens + ['0'])[:3])} {cols[0]})")
    return out


def pick_color(glyph: str | None, ink: str | None, vlm: str | None) -> tuple[str | None, str]:
    """MÀU CHỮ từ ba nguồn, mỗi nguồn sai một kiểu (dev2 15/10, 58 dòng ba nguồn lệch > 25 ΔE, soát 14):
      glyph  gom cụm màu trong khung (slots / glyph) -- sai 3/14 (chữ trên nền cùng tông);
      ink    lõi nét Hi-SAM (lines) -- sai 7/14 (chữ mảnh lẫn màu nền ở mép, chữ neon ra màu quầng);
      vlm    màu VLM ghi cho cả VAI -- sai 6/14 (một vai trên hai nền khác màu).
    Hai nguồn đồng ý (<= COLOR_AGREE) -> màu đo của cặp đó (glyph trước); không cặp nào -> glyph, rồi ink, rồi vlm. Đúng 13/14.
    -> (màu, lý do)."""
    from ..color import hex_lab

    def near(a, b):
        try:
            return a and b and float(np.linalg.norm(hex_lab(a) - hex_lab(b))) <= COLOR_AGREE
        except Exception:
            return False
    if glyph and (near(glyph, ink) or near(glyph, vlm)):
        return glyph, "đo khung + " + ("nét" if near(glyph, ink) else "VLM") + " đồng ý"
    if ink and near(ink, vlm):
        return ink, "nét + VLM đồng ý"
    for c, why in ((glyph, "chỉ còn đo khung"), (ink, "chỉ còn đo nét"), (vlm, "chỉ còn VLM")):
        if c:
            return c, why + (" (ba nguồn lệch nhau)" if sum(bool(x) for x in (glyph, ink, vlm)) > 1 else "")
    return None, "không nguồn nào"


def one_color(colors: list) -> str | None:
    """MỘT MÀU CHO TẬP khi nháp tô các mục cùng một màu (lệch <= COLOR_AGREE: sai số đo) -> màu nhiều mục nhất; nháp cố ý tô mỗi
    mục một màu (final3 h31: ba giá đỏ / cam / vàng) -> None, mỗi mục giữ màu của nó (bám nháp, 16/10)."""
    from collections import Counter

    from ..color import hex_lab
    cs = [c for c in colors if c]
    if not cs:
        return None
    top = Counter(cs).most_common(1)[0][0]
    try:
        ok = all(float(np.linalg.norm(hex_lab(c) - hex_lab(top))) <= COLOR_AGREE for c in cs)
    except Exception:
        ok = False
    return top if ok else None

