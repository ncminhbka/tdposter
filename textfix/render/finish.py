"""BÓNG ĐỔ MỀM DO CODE THÊM (12/11): chữ SÁNG nằm trên ảnh rối hoặc tương phản yếu -> bóng mềm dưới chữ, tách chữ khỏi nền như
dân thiết kế vẫn làm (người dùng 12/11: "code có thể thêm hiệu ứng, miễn output đẹp"). Chữ tối trên nền sáng, chữ trên nút / thẻ
phẳng, chữ đã có hiệu ứng (bóng / viền của VLM, quầng tương phản của vòng sửa): không đổi.

  layer = text_layer(page, W, H)                         # RGBA: chỉ chữ của trang vừa dựng
  poster, log = finish(base, layer, U)                   # bóng vào nền, rồi ghép chữ lên

Bóng làm bằng numpy trên mặt nạ chữ thật (cùng nét đã dựng), không qua CSS: đo được nền dưới từng chữ trước khi quyết.
Đã thử và BỎ (12/11, 8 poster golden xem phóng to): chép chất liệu từ chữ nháp (dải màu khớp lõi nét -- nửa câu nhạt đi, vệt
loang), vân nền qua chữ (chữ trắng xỉn như bẩn), ánh sáng theo vùng (vệt sáng ngang chữ dưới đèn).
TEXTFIX_FINISH=0 tắt.
"""

from __future__ import annotations

import base64
import io
import os
import re

import numpy as np

RATIO = 4.5      # chữ sáng trên nền: tỉ lệ tương phản WCAG < 4,5 ...
BUSY = 4.0       # ... hoặc nền rối sát quanh chữ (độ lệch chuẩn cao tần >= 4 mức xám) -> bóng
CLEAR = 7.0      # tương phản >= 7: đọc rõ dù nền rối, không thêm (24 golden: bóng thừa dưới chữ nút đã rõ)
NEAR = 0.35      # "sát quanh chữ" = trong 0,35 cỡ chữ quanh nét (chữ trong nút: chỉ lòng nút, không lấy cả ảnh ngoài nút)
DARK = 0.15      # bóng = nền tối còn 15% (giữ sắc nền, không đen kịt)


def enabled() -> bool:
    return os.environ.get("TEXTFIX_FINISH", "1").strip() not in ("0", "", "off", "false")


def text_layer(page, W: int, H: int) -> np.ndarray | None:
    """Lớp chữ RGBA của trang vừa dựng (nền tắt, nền trang trong suốt). None nếu không chụp được (không có CDP)."""
    from PIL import Image
    try:
        cdp = getattr(page, "_tf_cdp", None)
        if cdp is None:
            cdp = page._tf_cdp = page.context.new_cdp_session(page)
        page.evaluate("() => { document.getElementById('c').style.background = 'none'; "
                      "document.documentElement.style.background = 'transparent'; document.body.style.background = 'transparent'; }")
        cdp.send("Emulation.setDefaultBackgroundColorOverride", {"color": {"r": 0, "g": 0, "b": 0, "a": 0}})
        try:
            r = cdp.send("Page.captureScreenshot", {"format": "png", "clip": {"x": 0, "y": 0, "width": W, "height": H, "scale": 1}})
        finally:
            cdp.send("Emulation.setDefaultBackgroundColorOverride", {})
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(r["data"]))).convert("RGBA"))
    except Exception:
        return None


def _lum(rgb) -> float:
    """Độ chói tương đối (WCAG) của màu RGB 0-255."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _rgb(hx: str) -> np.ndarray | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hx or ""))
    return np.array([int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)], np.float64) if m else None


def finish(base: np.ndarray, layer: np.ndarray, U: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Bóng mềm vào nền dưới các phần tử cần, rồi ghép lớp chữ -> (poster, log)."""
    import cv2
    H, W = base.shape[:2]
    a = layer[..., 3].astype(np.float32) / 255.0
    bf = base.astype(np.float32)
    g = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    sh = np.zeros((H, W), np.float32)
    log = []
    for u in U:
        fg = _rgb(u.get("color"))
        if u.get("kind") != "text" or fg is None or re.search(r"text-shadow|stroke", u.get("css") or "") \
                or (u.get("fix") or {}).get("halo"):
            continue
        for e in u.get("els") or []:
            b, sz = e.get("box"), float(e.get("size") or 0)
            if not b or sz <= 0:
                continue
            pad = sz
            x0, y0 = max(0, int(b[0] - pad)), max(0, int(b[1] - pad))
            x1, y1 = min(W, int(np.ceil(b[2] + pad))), min(H, int(np.ceil(b[3] + pad)))
            if x1 <= x0 or y1 <= y0:
                continue
            sl = (slice(y0, y1), slice(x0, x1))
            aa = a[sl]
            glyph = aa > 0.05
            if glyph.sum() < 20:
                continue
            r = max(2, int(round(NEAR * sz)))
            ring = cv2.dilate(glyph.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
            ring &= aa < 0.02
            if ring.sum() < 30:
                continue
            lf, lb = _lum(fg), _lum(np.median(bf[sl][ring], axis=0))
            ratio = (max(lf, lb) + 0.05) / (min(lf, lb) + 0.05)
            busy = float(np.std(hp[sl][ring]))
            if not (lf > lb and (ratio < RATIO or (busy >= BUSY and ratio < CLEAR))):
                continue
            blur = cv2.GaussianBlur(aa, (0, 0), max(1.5, 0.07 * sz))
            blur = cv2.warpAffine(blur, np.float32([[1, 0, 0.02 * sz], [0, 1, 0.045 * sz]]), (aa.shape[1], aa.shape[0]))
            op = 0.6 if ratio < 3 else 0.45
            sh[sl] = np.maximum(sh[sl], np.clip(1.4 * blur, 0, 1) * op)
            log.append(f"bóng đổ {e['id']}: tương phản {ratio:.1f}, nền rối {busy:.1f}")
    bf = bf * (1 - (1 - DARK) * sh[..., None])
    out = bf * (1 - a[..., None]) + layer[..., :3].astype(np.float32) * a[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), log
