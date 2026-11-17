"""POSTER NHIỀU Ô (17/11): router của chatbot gom 2-4 ảnh thành MỘT ảnh chia ô (diptych / triptych / lưới 2x2). FLUX chia khung
tốt (prod 17/11: đúng bố cục 36/38 lượt, đường chia lệch chia đều <= 1,5%, sửa giữ khung 8/8), nhưng dây chuyền coi cả ảnh là
một poster: câu vắt qua đường chia, câu dồn sang ô khác. Ở đây: tìm đường chia trên nháp, gắn mỗi dòng chữ vào ô của nó; ghép câu
và bố cục dùng ô làm ràng buộc (compose/vlm.schema allow, compose/size room).

  P = find(draft, n, prior)     # {n, cols, rows, rects (px, thứ tự đọc)} | None
  tag_lines(M, P)               # l["panel"]: 1..n, 0 = dòng chung (vắt đường chia / ngoài ô)
  named(fold(request), P)       # lời sửa gọi ô nào

Câu: "panel" 0 = chung cả ảnh (tiêu đề chung, hotline dưới đáy), 1..n = ô theo thứ tự đọc (trái -> phải, trên -> dưới).
"""

from __future__ import annotations

import re

import numpy as np

STEP = 10        # hai cột kề nhau lệch > 10 (kênh màu lớn nhất) = một điểm đổi màu
ACCEPT = 0.6     # đường chia: >= 60% chiều dài có điểm đổi màu (đo prod 17/11: đường thật 0,61-1,0)
CONTRAST = 0.35  # ... và trội hơn nền quanh nó (trung vị +-15% cạnh ảnh) >= 0,35: ảnh rối / nhiễu cột nào cũng "đổi màu"
                 # (prod 17/11: trội 0,45-1,0; ca bốn mùa nền lá rối 0,47)
WINDOW = 0.04    # tìm quanh vị trí chia đều +-4% cạnh ảnh
PRIOR = 0.02     # sửa poster: quanh đường chia cũ +-2%
INSIDE = 0.85    # dòng có >= 85% diện tích trong một ô -> thuộc ô đó


def layouts(n: int, W: int, H: int) -> list[tuple[int, int]]:
    """Bố cục ứng viên (cột, hàng): n=4 -> 2x2; n=2/3 -> chia theo cạnh dài trước (prod: 16:9 ra cột, 9:16 ra hàng), rồi cạnh kia."""
    if n == 4:
        return [(2, 2)]
    if n in (2, 3):
        return [(n, 1), (1, n)] if W >= H else [(1, n), (n, 1)]
    return []


def _profile(img: np.ndarray, axis: int) -> np.ndarray:
    """Tỉ lệ điểm đổi màu giữa hai cột (axis=1) / hai hàng (axis=0) kề nhau, theo vị trí ranh giới (1..W-1 / 1..H-1)."""
    a = img.astype(np.int16)
    d = np.abs(np.diff(a, axis=axis)).max(axis=2) > STEP
    return d.mean(axis=0 if axis == 1 else 1)


def _cut(prof: np.ndarray, at: float, half: float) -> tuple[int, float]:
    """Ranh giới mạnh nhất trong [at-half, at+half] (px) -> (vị trí ranh giới, điểm); điểm = 0 khi không trội hơn nền quanh nó."""
    n = len(prof) + 1
    lo, hi = max(1, int(at - half)), min(n - 1, int(at + half) + 1)
    if hi <= lo:
        return int(round(at)), 0.0
    i = lo - 1 + int(np.argmax(prof[lo - 1:hi - 1]))
    span = int(0.15 * n)
    base = float(np.median(prof[max(0, i - span):i + span + 1]))
    v = float(prof[i])
    return i + 1, v if v - base >= CONTRAST else 0.0


def find(img: np.ndarray, n: int, prior: dict | None = None) -> dict | None:
    """Đường chia ô trên ảnh (RGB HxWx3). prior = kết quả cũ (chuẩn hoá, xem public) khi sửa poster: chỉ tìm sát đường cũ.
    Mọi đường chia phải đạt ACCEPT; nhiều bố cục đạt -> bố cục điểm thấp nhất cao nhất. Không đạt -> None (chạy như 1 ô)."""
    if img is None or img.ndim != 3 or n < 2:
        return None
    H, W = img.shape[:2]
    px, py = _profile(img, 1), _profile(img, 0)
    best = None
    cands = [(int(prior["cols"]), int(prior["rows"]))] if prior else layouts(n, W, H)
    for cols, rows in cands:
        if cols * rows != n:
            continue
        if prior:
            xs_at = [float(f) * W for f in prior.get("xs") or []]
            ys_at = [float(f) * H for f in prior.get("ys") or []]
            hx, hy = PRIOR * W, PRIOR * H
        else:
            xs_at, ys_at = [W * k / cols for k in range(1, cols)], [H * k / rows for k in range(1, rows)]
            hx, hy = WINDOW * W, WINDOW * H
        if len(xs_at) != cols - 1 or len(ys_at) != rows - 1:
            continue
        xs = [_cut(px, x, hx) for x in xs_at]
        ys = [_cut(py, y, hy) for y in ys_at]
        score = min(s for _, s in xs + ys)
        if score >= ACCEPT and (best is None or score > best[0]):
            best = (score, cols, rows, [x for x, _ in xs], [y for y, _ in ys])
    if best is None:
        return None
    score, cols, rows, xs, ys = best
    X, Y = [0] + xs + [W], [0] + ys + [H]
    rects = [[X[c], Y[r], X[c + 1], Y[r + 1]] for r in range(rows) for c in range(cols)]
    return {"n": n, "cols": cols, "rows": rows, "rects": rects, "xs": xs, "ys": ys, "score": round(score, 2), "size": [W, H]}


def from_prior(prior: dict, W: int, H: int) -> dict:
    """Đường chia cũ (chuẩn hoá) dùng nguyên khi nháp mới không đo ra (FLUX sửa giữ khung 8/8 lượt prod)."""
    cols, rows = int(prior["cols"]), int(prior["rows"])
    xs = [int(round(float(f) * W)) for f in prior.get("xs") or []]
    ys = [int(round(float(f) * H)) for f in prior.get("ys") or []]
    X, Y = [0] + xs + [W], [0] + ys + [H]
    return {"n": cols * rows, "cols": cols, "rows": rows, "xs": xs, "ys": ys, "score": 0.0, "size": [W, H],
            "rects": [[X[c], Y[r], X[c + 1], Y[r + 1]] for r in range(rows) for c in range(cols)]}


def public(P: dict | None) -> dict | None:
    """Dạng chuẩn hoá (khối thông tin PNG / plan.json)."""
    if not P:
        return None
    W, H = P["size"]
    return {"n": P["n"], "cols": P["cols"], "rows": P["rows"], "xs": [round(x / W, 4) for x in P["xs"]],
            "ys": [round(y / H, 4) for y in P["ys"]]}


def clean(p) -> dict | None:
    """Khối panels đọc từ PNG (ai cũng sửa được file): đúng kiểu, đúng số đường, toạ độ trong (0, 1)."""
    try:
        cols, rows = int(p["cols"]), int(p["rows"])
        xs, ys = [float(v) for v in p.get("xs") or []], [float(v) for v in p.get("ys") or []]
    except (TypeError, KeyError, ValueError):
        return None
    if not (1 <= cols <= 4 and 1 <= rows <= 4 and 2 <= cols * rows <= 4 and len(xs) == cols - 1 and len(ys) == rows - 1):
        return None
    if any(not 0 < v < 1 for v in xs + ys) or xs != sorted(xs) or ys != sorted(ys):
        return None
    return {"n": cols * rows, "cols": cols, "rows": rows, "xs": xs, "ys": ys}


def panel_of(box, P: dict) -> int:
    """Ô chứa khung (px): >= INSIDE diện tích trong một ô -> 1..n, không thì 0 (vắt đường chia / chung)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    for i, (a, b, c, d) in enumerate(P["rects"], 1):
        ov = max(0.0, min(x1, c) - max(x0, a)) * max(0.0, min(y1, d) - max(y0, b))
        if ov / area >= INSIDE:
            return i
    return 0


def tag_lines(M: dict, P: dict) -> list[str]:
    """l["panel"] cho mọi dòng; -> log."""
    log, cross = [], []
    for ln in M["L"]:
        ln["panel"] = panel_of(ln["box"], P)
        if not ln["panel"]:
            cross.append(ln["id"])
    log.append(f"ô: {P['cols']}x{P['rows']} (điểm {P['score']}), dòng chung / vắt đường chia: {cross or 'không'}")
    return log


def allow(M: dict, texts: list[dict]) -> dict:
    """Câu từng dòng được nhận (khuôn VLM): dòng ô i -> câu ô i + câu chung; dòng chung -> mọi câu."""
    common = [k for k, t in enumerate(texts) if not t.get("panel")]
    out = {}
    for ln in M["L"]:
        p = ln.get("panel") or 0
        out[ln["id"]] = list(range(len(texts))) if not p else sorted(common + [k for k, t in enumerate(texts) if t.get("panel") == p])
    return out


def where(k: int, P: dict, en: bool = False) -> str:
    """Tên ô k: lời nhắn LLM sửa "ô 2 · giữa", VLM (en) "panel 2 = center"."""
    cols, rows = P["cols"], P["rows"]
    r, c = divmod(k - 1, cols)
    if en:
        h = {1: [""], 2: ["left", "right"], 3: ["left", "center", "right"]}[cols][c]
        v = {1: [""], 2: ["top", "bottom"], 3: ["top", "middle", "bottom"]}[rows][r]
        return f"panel {k} = " + "-".join(x for x in (v, h) if x)
    h = {1: [""], 2: ["trái", "phải"], 3: ["trái", "giữa", "phải"]}[cols][c]
    v = {1: [""], 2: ["trên", "dưới"], 3: ["trên", "giữa", "dưới"]}[rows][r]
    pos = " ".join(x for x in (v, h) if x)
    return f"ô {k} · {pos}" if pos else f"ô {k}"


ORD = {"nhat": 1, "mot": 1, "1": 1, "hai": 2, "nhi": 2, "2": 2, "ba": 3, "3": 3, "tu": 4, "bon": 4, "4": 4,
       "first": 1, "second": 2, "third": 3, "fourth": 4}
H_WORDS = {"trai": "L", "left": "L", "phai": "R", "right": "R", "giua": "M", "center": "M", "centre": "M", "middle": "M"}
V_WORDS = {"tren": "T", "top": "T", "upper": "T", "duoi": "B", "bottom": "B", "lower": "B", "giua": "M", "middle": "M"}
NOUN = r"(?:o|khung|panel|quadrant|frame)"   # không "ảnh": "thêm ảnh 2 ly" không phải ô 2


def named(f: str, P: dict | None) -> set[int] | None:
    """Ô lời sửa (chữ bỏ dấu) gọi tên -> {chỉ số 1..n}; None = không gọi ô nào."""
    if not P or P.get("n", 1) < 2:
        return None
    n, cols, rows = P["n"], P["cols"], P["rows"]
    got: set[int] = set()
    for m in re.finditer(rf"\b{NOUN}\s+(?:so\s+|thu\s+|number\s+)?(\d|nhat|mot|hai|nhi|ba|tu|bon)\b", f):
        k = ORD.get(m.group(1))
        if k and k <= n:
            got.add(k)
    for m in re.finditer(rf"\b(first|second|third|fourth)\s+{NOUN}\b", f):
        k = ORD[m.group(1)]
        if k <= n:
            got.add(k)
    # vị trí: "ô bên phải", "ô trên trái", "top-left panel", "khung giữa", "right panel"
    pos = r"(?:ben\s+)?(trai|phai|giua|tren|duoi|left|right|center|centre|middle|top|bottom|upper|lower)"
    pats = [rf"\b{NOUN}\s+{pos}(?:[\s-]+{pos})?\b", rf"\b{pos}(?:[\s-]+{pos})?[\s-]+{NOUN}\b"]
    for pat in pats:
        for m in re.finditer(pat, f):
            words = [w for w in m.groups() if w]
            h = next((H_WORDS[w] for w in words if w in H_WORDS and not (w in ("giua", "middle") and cols == 1)), None)
            v = next((V_WORDS[w] for w in words if w in V_WORDS and not (w in ("giua", "middle") and rows == 1)), None)
            if cols == 1:
                h = None
            if rows == 1:
                v = None
            cs = {"L": [0], "R": [cols - 1], "M": [cols // 2] if cols == 3 else []}.get(h, list(range(cols)))
            rs = {"T": [0], "B": [rows - 1], "M": [rows // 2] if rows == 3 else []}.get(v, list(range(rows)))
            if h is None and v is None:
                continue
            got |= {r * cols + c + 1 for r in rs for c in cs}
    return got or None
