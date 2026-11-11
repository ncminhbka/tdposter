"""Dòng ra ngoài: JSON cho plan.json, ảnh đánh dấu L# (VLM và debug), ghi chú "chữ khó" (cong, dọc, nhiều cỡ).
"""

from __future__ import annotations

import re

import numpy as np

GEO_NAME = {"arc": "cung tròn", "wave": "lượn sóng", "grow": "to dần / nhỏ dần"}


def _typo_notes(L: list, logs: list[str]) -> None:
    """CHỮ KHÓ, ghi cho debug (người dùng 14/10: "debug chưa in các ca typography khó được phát hiện và xử lý thế nào"): mỗi dòng
    l["typo"] = [{"kind", "how"}] -- loại chữ khó và bằng chứng mã đã dùng để nhận ra nó. Cách dựng ghi ở plan["render"]."""
    for l in L:
        notes = []
        lg = [x for x in logs if re.search(rf"(?<![A-Za-z0-9]){l['id']}(?![0-9])", x)]
        if l.get("path"):
            how = []
            if l.get("sag") is not None:
                how.append(f"vùng chữ của bộ dò võng {l['sag']:.2f} cao chữ")
            if l.get("merged"):
                how.append(f"gộp {len(l['merged'])} mẩu OCR {l['merged']}")
            c = l.get("curve") or {}
            if c.get("r"):
                how.append(f"khớp đường tròn bán kính {c['r']:.0f} px, chữ {'trên' if c.get('up') else 'dưới'} vòm")
            notes.append({"kind": f"chữ cong ({GEO_NAME.get(l.get('geo'), l.get('geo') or '?')})", "how": "; ".join(how) or "phần hình học (lines/curve.py)"})
        if l.get("vertical"):
            bw, bh = l["box"][2] - l["box"][0], l["box"][3] - l["box"][1]
            notes.append({"kind": "chữ dọc", "how": f"khung OCR cao gấp {bh / max(1.0, bw):.1f} lần rộng; đọc lại theo {l['angle']:+.0f}° ra {l.get('ocr')!r}"})
        a, ao, ia = float(l.get("angle") or 0.0), l.get("angle_ocr"), (l.get("ink") or {}).get("ink_angle")
        src = f"nét đo {ia:+.1f}°" if ia is not None else "nét không đủ mảnh để đo hướng"
        if not l.get("vertical") and not l.get("path") and abs(a) >= 2:
            how = f"{src}, OCR đo {ao:+.1f}°: cùng thấy nghiêng" if ao is not None else "vùng chữ của bộ dò nghiêng (dòng OCR sót)"
            notes.append({"kind": f"nghiêng {a:+.0f}°", "how": how})
        elif ao is not None and abs(ao) >= 2 and not l.get("vertical") and not l.get("path"):
            notes.append({"kind": "dựng thẳng", "how": f"OCR đo {ao:+.1f}° nhưng {src}: hai nguồn không cùng thấy nghiêng"})
        if l.get("missed") and not l.get("orphan"):
            notes.append({"kind": "dòng OCR sót", "how": "; ".join(x for x in lg if "OCR sót" in x)[:300] or "vùng chữ của bộ dò không có dòng OCR, bộ đọc đọc ra chữ"})
        if l.get("orphan"):
            notes.append({"kind": "cụm nét OCR không đọc", "how": "; ".join(x for x in lg if "cụm nét" in x)[:300] or "nét Hi-SAM ngoài mọi dòng OCR"})
        if (l.get("ink") or {}).get("ornament"):
            notes.append({"kind": "trang trí đầu dòng", "how": "mảnh nét đầu dòng nhỏ, giữa chiều cao, cách chữ (lines.lead): không thuộc chữ, giữ nguyên điểm ảnh nháp"})
        if l.get("lead_icon"):
            notes.append({"kind": "khung dời khỏi icon", "how": "khung bộ dò ôm cả thứ đứng trước chữ -> mép trái dời tới nét chữ đầu tiên"})
        if notes:
            l["typo"] = notes
        else:
            l.pop("typo", None)


def public(M: dict) -> dict:
    """Ô để ghi JSON."""
    return {"L": [{k: m[k] for k in ("id", "box", "ocr_box", "ocr", "size", "color", "angle", "curve", "runs", "geo", "path", "scale", "missed", "orphan", "lead_icon", "vertical", "shape", "row", "angle_ocr", "ink", "sag", "merged", "typo", "poly") if k in m}
                  for m in M["L"]]}


def marked_image(img: np.ndarray, M: dict, label: int | None = None, width: int = 2):
    """Nháp + khung đỏ L# mỗi dòng. label: cỡ chữ nhãn (px); width: nét khung -- ảnh gửi VLM bị thu nhỏ nên vẽ to hơn."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    s = label or max(12, img.shape[1] // 75)
    try:
        f = ImageFont.truetype("arial.ttf", s)
    except OSError:
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", s)
        except OSError:
            f = ImageFont.load_default()
    items = [(m, (230, 20, 20), width, "tl") for m in M["L"]]
    for m, col, wd, corner in items:
        if m.get("path"):   # ô chữ cong: đường chân thay cho khung (khung thẳng của một vòm trùm cả thứ nằm dưới vòm)
            pt = [tuple(q) for q in m["path"]]
            d.line(pt, fill=col, width=2 * width)
            tw, th = d.textbbox((0, 0), m["id"], font=f)[2:]
            tx, ty = pt[0][0], max(0, pt[0][1] - th - 4)
            d.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=col)
            d.text((tx + 1, ty), m["id"], fill=(255, 255, 255), font=f)
            continue
        x0, y0, x1, y1 = m["box"]
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)   # ảnh debug không được làm hỏng phiên bản
        d.rectangle([x0, y0, x1, y1], outline=col, width=wd)
        tw, th = d.textbbox((0, 0), m["id"], font=f)[2:]
        tx, ty = (x0, max(0, y0 - th - 2)) if corner == "tl" else (max(0, x1 - tw - 2), min(img.shape[0] - th - 2, y1 + 1))
        d.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill=col)
        d.text((tx + 1, ty), m["id"], fill=(255, 255, 255), font=f)
    return im
