"""SỬA POSTER, KHUNG GIỮ NGUYÊN: chỗ của câu cũ mà nháp mới bỏ trống -> trả một dòng về đó TRƯỚC bước VLM (11/11).

FLUX sửa poster đôi khi vẽ lại cái nút / dải ruy băng nhưng KHÔNG vẽ chữ trong đó (edit1 e03: nút "Đặt lịch ngay" trống) -> không
có nét chữ, không có dòng, VLM không có chỗ đặt câu, câu (chữ AI) bị bỏ, nút trống. Khối thông tin của poster cũ có khung từng
câu (textfix/meta.py, "box"): câu còn trong danh sách mà không dòng nào của nháp mới chạm khung cũ -> thêm một dòng ở đó, đánh
dấu "restored"; VLM thấy nó như mọi dòng khác và quyết (lời dặn compose/vlm.py: chỉ đặt câu khi câu đó chưa có dòng nào).
Chỉ khi khung ảnh không đổi (khung cũ còn đúng chỗ).
"""

from __future__ import annotations

import numpy as np

TOUCH = 0.3   # một dòng của nháp phủ >= 30% khung cũ (theo khung nhỏ hơn) = câu vẫn còn chỗ đó


def _cover(a: list, b: list) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def restore(M: dict, texts: list[dict], draft: np.ndarray) -> list[str]:
    """Thêm vào M["L"] một dòng cho mỗi khung cũ bị bỏ trống. -> nhật ký."""
    H, W = draft.shape[:2]
    log = []
    for k, t in enumerate(texts):
        for nb in t.get("box") or []:
            b = [nb[0] * W, nb[1] * H, nb[2] * W, nb[3] * H]
            if b[2] - b[0] < 4 or b[3] - b[1] < 4 or any(_cover(b, l["box"]) >= TOUCH for l in M["L"]):
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            patch = draft[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
            dark_bg = patch.size and float(patch.mean()) < 128
            new = {"id": f"L{1 + max([int(l['id'][1:]) for l in M['L']] or [0])}", "box": [float(v) for v in b],
                   "ocr_box": [float(v) for v in b], "poly": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], "ocr": "",
                   "base": None, "size": float(0.8 * (b[3] - b[1])), "color": "#ffffff" if dark_bg else "#222222", "angle": 0.0,
                   "caps": False, "stroke": None, "stroke_contrast": None, "effect": None, "restored": f"T{k}"}
            M["L"].append(new)
            log.append(f"{new['id']}: khung cũ của T{k} {t['text']!r} không còn dòng nào -> trả dòng về chỗ cũ")
    if log and "by" in M:
        M["by"] = {l["id"]: l for l in M["L"]}
    return log
