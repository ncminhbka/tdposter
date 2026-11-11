"""Bộ font designer chọn (chỉ font giấy phép OFL, đủ dấu tiếng Việt) + @font-face nhúng base64 cho đúng các font dùng."""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

FONTS_DIR = Path(os.environ.get("TEXTFIX_FONTS", Path(__file__).resolve().parents[2] / "assets" / "fonts"))


# khoá -> (css family, KIỂU (VLM đọc trong danh mục), fallback, {độ đậm: tệp}). Tệp ở FONTS_DIR (assets/fonts/): CHỈ giấy phép OFL
# (dùng thương mại được), đủ 134 ký tự dấu tiếng Việt (02/10: bỏ mọi SVN-*).
# 12/11 (người dùng duyệt): MỖI KIỂU MỘT FONT -- 46 font (nhiều font gần giống nhau, VLM dồn 49% vào Montserrat) -> 21 kiểu khác
# hẳn nhau; thêm chân khối, dãn rộng, mảnh art-deco, truyện tranh, cổ điển, bút dạ, mono, stencil, pixel. Bản nghiêng THẬT nằm ở
# ITALIC (cùng họ, font-style:italic): VLM chỉ chọn kiểu + "italic"; font không có bản nghiêng thì Chromium tự nghiêng.
CATALOG = {
    "bevietnam": ("Be Vietnam Pro", "neutral sans, the safe default", "sans-serif",
                  {400: "BeVietnamPro-Regular.woff2", 600: "BeVietnamPro-SemiBold.woff2", 700: "BeVietnamPro-Bold.woff2",
                   800: "BeVietnamPro-ExtraBold.woff2", 900: "BeVietnamPro-Black.woff2"}),
    "montserrat": ("Montserrat", "geometric sans, round open letters", "sans-serif", {"100 900": "Montserrat.woff2"}),
    "unbounded": ("Unbounded", "extended wide sans, modern / tech headlines", "sans-serif", {"200 900": "Unbounded.woff2"}),
    "josefin": ("Josefin Sans", "thin elegant geometric sans, art-deco, spa / beauty / luxury", "sans-serif",
                {"100 700": "JosefinSans.woff2"}),
    "chakra": ("Chakra Petch", "squared techno sans, gaming / electronics", "sans-serif", {700: "ChakraPetch-Bold.woff2"}),
    "oswald": ("Oswald", "condensed sans, tall narrow headlines", "sans-serif", {"200 700": "Oswald.woff2"}),
    "barlowcond": ("Barlow Condensed", "heavy condensed sport sans (italic for speed / sport)", "sans-serif",
                   {800: "BarlowCondensed-ExtraBold.woff2", 900: "BarlowCondensed-Black.woff2"}),
    "baloo": ("Baloo 2", "rounded friendly sans, kids / snacks / cute", "sans-serif", {"400 800": "Baloo2.woff2"}),
    "bungee": ("Bungee", "blocky signage capitals, street / shop sign", "sans-serif", {400: "Bungee-Regular.woff2"}),
    "bangers": ("Bangers", "comic-book capitals, sale burst / fun", "sans-serif", {400: "Bangers-Regular.woff2"}),
    "playfair": ("Playfair Display", "high-contrast display serif, luxury / elegant", "serif", {"400 900": "PlayfairDisplay.woff2"}),
    "lora": ("Lora", "warm readable serif, café / bakery / book", "serif", {"400 700": "Lora.woff2"}),
    "alfaslab": ("Alfa Slab One", "heavy slab serif, bold retro / industrial", "serif", {400: "AlfaSlabOne-Regular.woff2"}),
    "chonburi": ("Chonburi", "vintage display serif, classic / heritage / traditional", "serif", {400: "Chonburi-Regular.woff2"}),
    "greatvibes": ("Great Vibes", "formal calligraphy script, wedding / luxury", "cursive", {400: "GreatVibes-Regular.woff2"}),
    "dancing": ("Dancing Script", "casual handwritten script", "cursive", {"400 700": "DancingScript.woff2"}),
    "lobster": ("Lobster", "bold retro script, diner / shop sign", "cursive", {400: "Lobster-Regular.woff2"}),
    "playpen": ("Playpen Sans", "handwritten marker, friendly notes / chalkboard", "cursive", {"100 800": "PlaypenSans.woff2"}),
    "spacemono": ("Space Mono", "monospace, typewriter / tech / code", "monospace",
                  {400: "SpaceMono-Regular.woff2", 700: "SpaceMono-Bold.woff2"}),
    "stencil": ("Saira Stencil One", "stencil capitals, military / industrial / street", "sans-serif",
                {400: "SairaStencilOne-Regular.woff2"}),
    "pixel": ("VT323", "pixel 8-bit, retro game", "monospace", {400: "VT323-Regular.woff2"}),
}

# bản nghiêng THẬT: cùng họ với CATALOG, font-style:italic (faces_for)
ITALIC = {
    "bevietnam": {700: "BeVietnamPro-BoldItalic.woff2", 800: "BeVietnamPro-ExtraBoldItalic.woff2", 900: "BeVietnamPro-BlackItalic.woff2"},
    "montserrat": {"100 900": "Montserrat-Italic.woff2"},
    "josefin": {"100 700": "JosefinSans-Italic.woff2"},
    "chakra": {700: "ChakraPetch-BoldItalic.woff2"},
    "barlowcond": {900: "BarlowCondensed-BlackItalic.woff2"},
    "playfair": {"400 900": "PlayfairDisplay-Italic.woff2"},
    "lora": {"400 700": "Lora-Italic.woff2"},
}


WIDTH = {   # em / ký tự (chữ thường lẫn hoa, IN HOA) ở độ đậm đậm nhất <= 800; scripts/font_widths.py
    'bevietnam': (0.54, 0.61),
    'montserrat': (0.55, 0.63),
    'unbounded': (0.65, 0.74),
    'josefin': (0.5, 0.61),
    'chakra': (0.5, 0.55),
    'oswald': (0.43, 0.47),
    'barlowcond': (0.39, 0.41),
    'baloo': (0.47, 0.52),
    'bungee': (0.6, 0.62),
    'bangers': (0.37, 0.37),
    'playfair': (0.48, 0.59),
    'lora': (0.5, 0.6),
    'alfaslab': (0.56, 0.67),
    'chonburi': (0.58, 0.64),
    'greatvibes': (0.33, 0.78),
    'dancing': (0.39, 0.55),
    'lobster': (0.4, 0.52),
    'playpen': (0.55, 0.63),
    'spacemono': (0.61, 0.61),
    'stencil': (0.48, 0.55),
    'pixel': (0.4, 0.4),
}


@lru_cache(maxsize=64)
def _metrics(key: str):
    """(bảng mã ký tự -> bề rộng em, bề rộng em trung bình) đọc từ tệp font của khoá (tệp đầu tiên; font biến thiên: bản mặc định)."""
    from fontTools.ttLib import TTFont
    f = TTFont(str(FONTS_DIR / next(iter(CATALOG[key][3].values()))))
    upm, hm = f["head"].unitsPerEm, f["hmtx"].metrics
    adv = {cp: hm[g][0] / upm for cp, g in f.getBestCmap().items() if g in hm}
    return adv, sum(adv.get(ord(c), 0.5) for c in "aeinost") / 7


def advance(key: str | None, text: str) -> float:
    """Bề rộng (em) của chữ theo font THẬT của khoá; khoá lạ -> ước 0,55 em / ký tự."""
    try:
        adv, avg = _metrics(key) if key in CATALOG else (None, None)
    except Exception:   # thiếu fontTools / brotli: ước theo số ký tự
        adv = None
    if adv is None:
        return 0.55 * len(text)
    return sum(adv.get(ord(c), avg) for c in text)


CAP_EM, X_EM = 0.72, 0.53   # cao chữ hoa / chữ thường (em) khi không đọc được font: trung vị danh mục


@lru_cache(maxsize=64)
def heights(key: str | None) -> tuple[float, float]:
    """(cao chữ hoa, cao chữ thường) theo em, đo trên HÌNH chữ "H" / "x" của font (số khai trong bảng OS/2 có font ghi sai:
    Bungee ghi chữ thường 0,50 mà chữ thường của nó là chữ hoa 0,72). Danh mục chênh 0,59-0,86 (hoa), 0,30-0,73 (thường): cỡ
    tính từ cao nét nháp phải chia cho số của ĐÚNG font sẽ dựng. Khoá lạ / thiếu fontTools -> (CAP_EM, X_EM)."""
    try:
        from fontTools.pens.boundsPen import BoundsPen
        from fontTools.ttLib import TTFont
        f = TTFont(str(FONTS_DIR / next(iter(CATALOG[key][3].values()))))
        upm, gs, cmap = f["head"].unitsPerEm, f.getGlyphSet(), f.getBestCmap()

        def top(ch):
            p = BoundsPen(gs)
            gs[cmap[ord(ch)]].draw(p)
            return p.bounds[3] / upm
        return round(top("H"), 3), round(top("x"), 3)
    except Exception:
        return CAP_EM, X_EM


def family(key: str) -> str:
    fam, _, fb, _ = CATALOG[key]
    return f"'{fam}', {fb}"


@lru_cache(maxsize=64)
def faces_for(keys: tuple) -> str:
    """@font-face (nhúng base64) cho đúng các khoá CATALOG cần."""
    out = []
    for key in keys:
        fam, _, _, files = CATALOG[key]
        for style, fs in (("normal", files), ("italic", ITALIC.get(key, {}))):
            for wt, fn in fs.items():
                p = FONTS_DIR / fn
                fmt = "woff2" if fn.endswith(".woff2") else "truetype"
                b64 = base64.b64encode(p.read_bytes()).decode()
                out.append(f"@font-face{{font-family:'{fam}';src:url('data:font/{fmt};base64,{b64}') format('{fmt}');"
                           f"font-weight:{wt};font-style:{style};unicode-range:U+0-9F,U+A1-10FFFF}}")
    return "\n".join(out)
