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


def test_a_change_naming_one_role_frees_only_that_role():
    k = carry(OLD, "đổi font tiêu đề cho mềm mại hơn")
    assert "font" not in k["headline"] and k["cta"]["font"] == "montserrat"
    k = carry(OLD, "doi mau chu nut sang do")
    assert "color" not in k["cta"] and k["headline"]["color"] == "#5a2d0c"


def test_drawn_font_size_rides_in_the_png_and_a_size_request_frees_it():
    """quick1 q04 / q05: chữ phình dần qua các lần sửa -> cỡ đã dựng (fs / cao ảnh) vào khối thông tin, lần sửa sau không to hơn."""
    from textfix.edit.plan import free_sizes
    from textfix.meta import set_shown
    info = {"texts": [{"text": "Ứng tuyển ngay", "role": "cta"}, {"text": "Tuyển dụng", "role": "headline"}]}
    ops = [{"id": "u1", "kind": "text", "client": ["T0"], "marks": ["L1"], "html": "Ứng tuyển ngay"},
           {"id": "u2", "kind": "text", "client": ["T1"], "marks": ["L2"], "html": "Tuyển dụng"}]
    render = [{"id": "u1", "unit": "u1", "fs": 55.5}, {"id": "u2", "unit": "u2", "fs": 98.0}]
    set_shown(info, ops, None, (1024, 1024), render)
    assert info["texts"][0]["fs"] == round(55.5 / 1024, 5) and info["texts"][1]["fs"] == round(98 / 1024, 5)
    t = [dict(x) for x in info["texts"]]
    free_sizes(t, "nền màu cam")
    assert all("fs" in x for x in t)
    free_sizes(t, "chữ tiêu đề to hơn")
    assert "fs" in t[0] and "fs" not in t[1]
