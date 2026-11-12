"""KIỂU CHỮ QUA CÁC LẦN SỬA (12/11): mỗi lần sửa chạy lại cả pipeline -- FLUX vẽ lại, VLM chọn lại font / độ đậm / màu từ đầu,
nên một lời sửa nhỏ ở chữ có thể đổi hẳn kiểu chữ (probe 12/11 e14: "nền màu cam" -> cả poster từ chữ có chân đậm thành chữ viết
tay mảnh). Kiểu chữ ĐÃ DÙNG của từng vai được lưu trong khối thông tin PNG (textfix/meta.py, "styles"); lần sửa sau giữ nguyên,
trừ phần lời sửa nhắc tới.

  styles = used(U, A_styles)               # sau khi dựng: kiểu từng vai -> khối thông tin
  keep = carry(old["styles"], request)     # lời sửa: vai nào giữ gì (font / màu mở khoá khi lời sửa nhắc)
  log = apply(A["styles"], keep)           # sau lời gọi VLM: đè kiểu VLM chọn bằng kiểu cũ

Cỡ và chỗ vẫn theo nháp mới (FLUX có thể dời bố cục). Màu giữ nguyên màu cũ; nền mới làm chữ chìm thì bộ kiểm tương phản (render/check)
đổi màu như mọi poster.
"""

from __future__ import annotations

import re

SHAPE = ("font", "weight", "italic", "case", "tracking", "effect")
# lời sửa NHẮC kiểu chữ -> không giữ phần đó (so trên chữ bỏ dấu)
FONT_CUE = re.compile(r"\b(font|phong chu|kieu chu|in hoa|viet hoa|in nghieng|nghieng|dam hon|manh hon|bo dam|thu phap|"
                      r"hieu ung|vien chu|bong chu|chu dam|chu manh|chu thang|chu hoa)\b")
COLOR_CUE = re.compile(r"\bmau\b[^.,;!?]{0,25}\b(chu|tieu de|ten|gia|dong|cau|slogan)\b"
                       r"|\b(chu|tieu de|ten|gia|slogan)\b[^.,;!?]{0,25}\bmau\b")
HEX = re.compile(r"#[0-9a-fA-F]{6}")
# lời sửa gọi tên VAI nào -> chỉ vai đó được đổi kiểu (12/11 quick1 q15: "đổi font tiêu đề" làm cả câu phụ và nút đổi font)
ROLE_WORDS = [(re.compile(r"\b(tieu de|ten quan|ten tiem|ten shop|ten cua hang|chu to|dong lon)\b"), ("headline",)),
              (re.compile(r"\b(nut|button)\b"), ("cta",)),
              (re.compile(r"\b(gia)\b"), ("price", "old_price")),
              (re.compile(r"\b(slogan|khau hieu|cau phu|dong phu)\b"), ("subheadline",)),
              (re.compile(r"\b(so dien thoai|hotline|lien he|dia chi)\b"), ("contact",))]


def named_roles(f: str) -> set | None:
    """Vai lời sửa (chữ bỏ dấu) gọi tên; None = không gọi tên vai nào (đổi kiểu cho mọi vai)."""
    got = {r for rx, roles in ROLE_WORDS if rx.search(f) for r in roles}
    return got or None


def used(U: list[dict], styles: dict) -> dict:
    """Kiểu đã dựng của từng vai (đơn vị đầu tiên mang câu khách của vai đó): font, độ đậm, màu đo thật + phần VLM ghi."""
    out: dict = {}
    for u in U:
        role = u.get("role")
        if u.get("kind") != "text" or not u.get("clients") or not role or role in out or not u.get("font"):
            continue
        st = styles.get(role) or {}
        s = {"font": u["font"], "weight": int(u.get("weight") or 400), "italic": bool(st.get("italic")),
             "case": st.get("case") or "none", "tracking": st.get("tracking"), "effect": str(st.get("effect") or "")}
        if HEX.fullmatch(str(u.get("color") or "")):
            s["color"] = u["color"]
        out[role] = s
    return out


def clean(styles) -> dict:
    """Kiểu đọc từ khối thông tin (ai cũng sửa được file): chỉ giữ khoá biết, kiểu dữ liệu đúng."""
    from ..render.fonts import CATALOG
    out: dict = {}
    for role, s in (styles or {}).items() if isinstance(styles, dict) else ():
        if not isinstance(s, dict) or s.get("font") not in CATALOG:
            continue
        c = {"font": s["font"], "weight": int(s["weight"]) if isinstance(s.get("weight"), (int, float)) else 400,
             "italic": bool(s.get("italic")), "case": s.get("case") if s.get("case") in ("upper", "none") else "none",
             "tracking": float(s["tracking"]) if isinstance(s.get("tracking"), (int, float)) else None,
             "effect": str(s.get("effect") or "")[:300]}
        if HEX.fullmatch(str(s.get("color") or "")):
            c["color"] = s["color"]
        out[str(role)[:40]] = c
    return out


def carry(old_styles, request: str) -> dict:
    """Kiểu cũ cần giữ ở lần sửa này: {vai: {khoá: giá trị}}; lời sửa nhắc font / màu chữ thì bỏ phần đó."""
    from ..brief.evidence import fold
    f = fold(request or "")
    roles = named_roles(f)
    free = lambda cue, role: bool(cue.search(f)) and (roles is None or role in roles)   # noqa: E731
    out = {}
    for role, s in clean(old_styles).items():
        keep_shape, keep_color = not free(FONT_CUE, role), not free(COLOR_CUE, role)
        k = {key: s[key] for key in SHAPE if keep_shape and key in s}
        if keep_color and s.get("color"):
            k["color"] = s["color"]
        if k:
            out[role] = k
    return out


def apply(styles: dict, keep: dict) -> list[str]:
    """Đè kiểu VLM vừa chọn bằng kiểu cũ (vai có trên poster mới). Màu cũ được đánh dấu keep_color: layout dùng thẳng."""
    log = []
    for role, k in (keep or {}).items():
        if role not in styles:
            continue
        st = styles[role] = dict(styles[role] or {})
        diff = [key for key in k if st.get(key) != k[key]]
        st.update(k)
        if "color" in k:
            st["keep_color"] = True
        if diff:
            log.append(f"kiểu cũ của vai {role}: giữ {diff} (lời sửa không nhắc)")
    return log
