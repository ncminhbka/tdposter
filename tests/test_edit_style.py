"""Kiểu chữ qua các lần sửa (textfix/edit/style.py): lưu trong khối thông tin PNG, lần sửa sau giữ, trừ phần lời sửa nhắc."""

import numpy as np

from textfix import meta
from textfix.edit.style import apply, carry, clean, used

OLD = {"headline": {"font": "playfair", "weight": 800, "italic": False, "case": "none", "tracking": None,
                    "effect": "text-shadow:0 2px 4px #0006", "color": "#5a2d0c"},
       "cta": {"font": "montserrat", "weight": 700, "italic": False, "case": "upper", "tracking": 0.05, "effect": "",
               "color": "#ffffff"}}


def test_a_change_that_does_not_mention_lettering_keeps_font_weight_and_color():
    for req in ("đổi màu nền sang xanh lá pastel", "đổi giá thành 459k", "thêm bó hoa hồng bên cạnh", "doi mau nut sang do"):
        k = carry(OLD, req)
        assert k["headline"]["font"] == "playfair" and k["headline"]["weight"] == 800, req
        assert k["headline"]["color"] == "#5a2d0c" and k["cta"]["case"] == "upper", req


def test_a_change_of_text_color_or_font_frees_only_that_part():
    k = carry(OLD, "đổi màu chữ tiêu đề sang vàng")
    assert "color" not in k["headline"] and k["headline"]["font"] == "playfair"
    k = carry(OLD, "đổi font cho sang hơn")
    assert "font" not in k["headline"] and k["headline"]["color"] == "#5a2d0c"
    k = carry(OLD, "chu in hoa het, mau chu trang")
    assert not k


def test_old_style_overrides_what_the_vlm_picked_for_roles_still_on_the_poster():
    A = {"headline": {"font": "greatvibes", "weight": 400, "color": "#ffffff", "effect": ""}, "body": {"font": "lora"}}
    log = apply(A, carry(OLD, "đổi nền màu cam"))
    assert A["headline"]["font"] == "playfair" and A["headline"]["weight"] == 800 and A["headline"]["keep_color"]
    assert A["body"] == {"font": "lora"} and "cta" not in A and log


def test_styles_ride_in_the_png_and_bad_entries_are_dropped():
    st = dict(OLD, bad={"font": "comic-sans", "weight": 400}, weird="x")
    raw = meta.save_png(np.zeros((8, 8, 3), np.uint8), {"run_id": "r1", "texts": [{"text": "A", "role": "headline"}], "styles": st})
    got = meta.read(raw)["styles"]
    assert set(got) == {"headline", "cta"} and got["headline"]["color"] == "#5a2d0c"
    assert clean({"headline": {"font": "playfair", "color": "red; background:url(x)"}})["headline"].get("color") is None
    old = meta.read(meta.save_png(np.zeros((8, 8, 3), np.uint8), {"run_id": "r0", "texts": [{"text": "A"}]}))
    assert old["styles"] == {}


def test_used_takes_the_drawn_style_of_the_first_client_unit_of_each_role():
    U = [{"kind": "text", "role": "headline", "clients": [0], "font": "playfair", "weight": 800, "color": "#5a2d0c"},
         {"kind": "text", "role": "headline", "clients": [1], "font": "lora", "weight": 400, "color": "#000000"},
         {"kind": "text", "role": "fill", "clients": [], "font": "lora", "weight": 400, "color": "#000000"}]
    got = used(U, {"headline": {"italic": True, "case": "none", "effect": "text-shadow:0 1px 2px #000"}})
    assert got == {"headline": {"font": "playfair", "weight": 800, "italic": True, "case": "none", "tracking": None,
                                "effect": "text-shadow:0 1px 2px #000", "color": "#5a2d0c"}}
