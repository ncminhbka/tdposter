"""Bóng đổ mềm do code thêm (textfix/render/finish.py): chỉ chữ sáng trên nền rối / tương phản yếu."""

import numpy as np

from textfix.render.finish import finish


def _layer(H, W, color, box):
    lay = np.zeros((H, W, 4), np.uint8)
    x0, y0, x1, y1 = box
    lay[y0:y1, x0:x1, :3] = color
    lay[y0:y1, x0:x1, 3] = 255
    return lay


def _unit(color, box, size=20.0, css=""):
    return {"id": "u1", "kind": "text", "color": color, "css": css, "els": [{"id": "u1#0", "box": list(box), "size": size}]}


def test_light_text_on_busy_photo_gets_a_soft_shadow_below():
    rng = np.random.default_rng(0)
    base = np.clip(rng.normal(150, 25, (120, 200, 3)), 0, 255).astype(np.uint8)   # ảnh rối, sáng vừa
    box = (40, 40, 160, 60)
    poster, log = finish(base, _layer(120, 200, (235, 235, 235), box), [_unit("#ebebeb", box)])
    assert log and "bóng đổ" in log[0]
    below = (slice(61, 64), slice(60, 140))   # ngay dưới chữ (bóng lệch xuống)
    assert poster[below].mean() < base[below].mean() - 10
    assert (poster[40:60, 40:160] == 235).all()   # chữ giữ nguyên màu


def test_dark_text_on_light_flat_ground_and_text_with_own_effect_are_left_alone():
    base = np.full((120, 200, 3), 245, np.uint8)
    box = (40, 40, 160, 60)
    _, log = finish(base, _layer(120, 200, (20, 20, 20), box), [_unit("#141414", box)])
    assert not log
    rng = np.random.default_rng(1)
    busy = np.clip(rng.normal(150, 25, (120, 200, 3)), 0, 255).astype(np.uint8)
    _, log = finish(busy, _layer(120, 200, (235, 235, 235), box), [_unit("#ebebeb", box, css="text-shadow:0 1px 2px #000")])
    assert not log


def test_accent_never_splits_the_markup_of_an_earlier_accent():
    from textfix.compose.layout import _markup
    u = {"accent": [{"text": "24", "color": "#834e2d"}, {"text": "4", "color": "#834e2d"}, {"text": "'", "color": "#123456"}]}
    h = _markup("Thùng 24 lon, hộp 4 cái 'ngon'", u, [])
    assert "4e2d;" not in h.replace('color:#834e2d;"', "")
    assert h.count("<span") == 3 and h.count("</span>") == 3
    assert ">24</span>" in h and ">4</span>" in h and "&#x27;" in h
