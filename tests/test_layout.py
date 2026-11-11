"""Bố cục chữ (fill): cỡ theo font thật, nới khung vào nền trống, màu từ ba nguồn."""

from __future__ import annotations

import pytest


def test_font_heights_differ_by_family():
    from textfix.render.fonts import CAP_EM, X_EM, heights
    assert heights("oswald")[0] > 0.8 and heights("baloo")[0] < 0.62   # chữ hẹp cao / chữ tròn thấp: không dùng một hằng chung
    assert heights("bungee")[1] == pytest.approx(heights("bungee")[0], abs=0.01)   # font toàn chữ hoa
    assert heights("khong-co") == (CAP_EM, X_EM)


def test_room_grows_into_free_background_only():
    import numpy as np
    from textfix.compose.size import room
    lab = np.zeros((100, 400, 3))
    lab[:, 290:] = (50, 40, 0)                 # mép vỏ / hình ở x >= 290
    blocked = np.zeros((100, 400), bool)
    blocked[:, :60] = True                     # dòng khác ở x < 60
    box = [100, 40, 260, 60]                   # rộng 160 -> nới tối đa 40
    assert room(box, "left", lab, blocked)[2] == 286          # chạm vỏ ở 290 (chưa hết 40): chừa khe 0,2 cao dòng (4 px)
    assert room(box, "right", lab, blocked)[0] == 60           # nới trái đủ 40 px (không chạm gì)
    c = room(box, "center", lab, blocked)
    assert c[0] == 80 and c[2] == 280                          # đều hai bên, mỗi bên 20


def test_room_stops_short_of_a_neighbor_cut_from_the_same_line():
    # 20/10 run 5b98 v3: "0322 111 333" | "123 Hoàng Hoa Thám" cắt từ CÙNG một dòng nháp -- vùng dòng không chặn nhau; ô SĐT nới
    # giữa vào khe "|" tới sát ô địa chỉ
    import numpy as np
    from textfix.compose.size import NEIGHBOR_GAP, neighbors, room
    lab = np.zeros((100, 600, 3))
    phone = {"box": [100, 40, 260, 60], "size": 20.0}
    addr = {"box": [272, 40, 560, 60], "size": 20.0}
    nb = room(phone["box"], "center", lab, neighbors(phone, [phone, addr], 100, 600))
    assert addr["box"][0] - nb[2] >= NEIGHBOR_GAP * 20


@pytest.mark.parametrize("glyph, ink, vlm, want", [
    ("#e8d9a0", "#e0d4a2", "#000000", "#e8d9a0"),   # chữ vàng trên đen, VLM ghi màu vai "đen": hai số đo đồng ý
    ("#fdfdfd", "#8a8a8a", "#ffffff", "#fdfdfd"),   # chữ trắng mảnh: lõi nét lẫn nền, đo khung + VLM đồng ý
    ("#c9a45a", "#fffdf5", "#ffffff", "#fffdf5"),   # chữ trắng trên nền vàng: khung đo ra nền, nét + VLM đồng ý
    (None, None, "#123456", "#123456"),
])
def test_color_needs_two_sources(glyph, ink, vlm, want):
    from textfix.compose.style import pick_color
    assert pick_color(glyph, ink, vlm)[0] == want


def test_row_of_mixed_sizes_reads_left_to_right():
    from textfix.compose.geometry import _reading
    giam, big = [100, 120, 180, 160], [190, 60, 420, 165]   # "Giảm" nhỏ | "50%" to, cùng hàng: mép trên của chữ to cao hơn
    under = [100, 200, 420, 240]
    assert _reading([big, under, giam]) == [2, 0, 1]


def test_set_keeps_only_units_drawn_as_one_list():
    # final2 d63e: ba hàng danh sách thẳng lề trái + mẩu chữ cùng vai ở giữa lon (x 477): mẩu lạc chỗ không thuộc tập
    from textfix.compose.layout import aligned_groups
    u = lambda i, b: {"id": i, "box": b, "size": 36.0}   # noqa: E731
    rows = [u("u3", [100, 294, 337, 329]), u("u4", [101, 344, 301, 383]), u("u5", [99, 391, 292, 431])]
    G = aligned_groups(rows + [u("u6", [477, 553, 494, 576])])
    assert [[x["id"] for x in g] for g in G] == [["u3", "u4", "u5"]]
    # các thẻ đứng ngang một hàng (không thẳng lề nào) vẫn là một tập
    cards = [u("a", [50, 500, 200, 540]), u("b", [300, 505, 420, 545]), u("c", [520, 500, 700, 540])]
    assert len(aligned_groups(cards)) == 1


def test_column_width_is_shared_by_the_set():
    # cơm tấm f1cf: chữ giun "Trá" 67px cạnh "Cồm sũron bí chà" 333px, cùng cột lề trái -> mỗi tên món được nới tới 333px
    from textfix.compose.size import column_want
    e = lambda i, x1: {"id": i, "box": [100, 0, x1, 40], "path": None, "angle": 0.0, "multi": False}   # noqa: E731
    T = [{"set": "label#0", "align": "left", "size": 42.0, "els": [e("a", 433)]},
         {"set": "label#0", "align": "left", "size": 42.0, "els": [e("b", 167)]},
         {"set": None, "align": "left", "size": 42.0, "els": [e("c", 120)]}]
    assert column_want(T) == {"b": 266}


def test_set_keeps_each_color_when_the_draft_colors_differ():
    # final3 h31: ba giá FLUX tô đỏ / cam / vàng -> giữ màu từng dòng; cùng một màu (sai số đo) -> một màu
    from textfix.compose.style import one_color
    assert one_color(["#8a1017", "#e86f1a", "#fad611"]) is None
    assert one_color(["#1a1a1a", "#1c1b1a", "#1a1a1a"]) == "#1a1a1a"
