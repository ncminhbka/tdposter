"""Hướng và số đo dòng từ nét (textfix.lines) trên chữ vẽ tổng hợp: mỗi test là một cam kết."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from textfix.lines import Ink, measure_lines
from textfix.lines.angle import SNAP, direction, orient
from textfix.lines.group import Group
from textfix.lines.ink import own


def ink_of(text: str, size: int = 48, rotate: float = 0.0) -> np.ndarray:
    """Mặt nạ nét của một dòng chữ đen trên nền trắng, xoay `rotate` độ (ngược kim đồng hồ, như PIL)."""
    font = ImageFont.load_default(size=size)
    im = Image.new("L", (40 + size * len(text), 3 * size), 255)
    ImageDraw.Draw(im).text((20, size // 2), text, fill=0, font=font)
    im = im.rotate(rotate, expand=True, fillcolor=255)
    return np.asarray(im) < 128


def group_of(mask: np.ndarray) -> tuple[Ink, Group]:
    ink = Ink(mask)
    return ink, Group(ink, [i for i in range(1, ink.n) if ink.ok[i]])


def test_horizontal_line_has_no_tilt():
    _, g = group_of(ink_of("Hello world"))
    assert abs(direction(g)) < SNAP


@pytest.mark.parametrize("deg", [8.0, -12.0])
def test_tilted_line_angle(deg):
    _, g = group_of(ink_of("Summer sale now", rotate=deg))
    assert direction(g) == pytest.approx(-deg, abs=1.0)   # ảnh: góc dương = xuôi kim đồng hồ


def test_vertical_line():
    mask = ink_of("VERTICAL", rotate=90)
    _, g = group_of(mask)
    ys, xs = np.nonzero(mask)
    tall = np.ptp(ys) > np.ptp(xs)
    assert orient(direction(g), 0.0, bool(tall))[0] == "vertical"


def test_few_glyphs_have_no_ink_direction():
    _, g = group_of(ink_of("-5"))
    assert direction(g) is None


@pytest.mark.parametrize("ink_angle, ocr_angle, want", [
    (5.0, 0.0, "straight"),     # nét lệch vì chân thò ("200g"), OCR thấy ngang
    (0.0, 6.0, "straight"),     # OCR lệch ở từ ngắn ("MC"), nét thấy ngang
    (-8.0, 7.0, "straight"),    # hai nguồn ngược chiều
    (-8.0, -6.5, "tilt"),       # cùng thấy nghiêng: góc theo nét
    (None, -15.0, "tilt"),      # nét không nói hướng ("-50"): chỉ còn OCR
    (None, 1.0, "straight"),
    (88.0, 0.0, "vertical"),
])
def test_orient_needs_both_sources(ink_angle, ocr_angle, want):
    shape, ang = orient(ink_angle, ocr_angle, tall=True)
    assert shape == want
    if want == "tilt" and ink_angle is not None:
        assert ang == pytest.approx(ink_angle)


def test_angles_are_lines_not_directions():
    # 95,9 độ và -84,1 độ là cùng một đường
    assert orient(95.9, -85.4, tall=True)[0] == "vertical"


def test_overlapping_regions_do_not_share_a_glyph():
    mask = ink_of("Hello")
    ink = Ink(mask)
    xs = np.nonzero(mask)[1]
    x0, x1 = int(xs.min()), int(xs.max())
    a, b = np.zeros(mask.shape, bool), np.zeros(mask.shape, bool)
    a[:, : x0 + (x1 - x0) * 2 // 3] = True   # hai vùng chồng nhau ở phần ba giữa dòng
    b[:, x0 + (x1 - x0) // 3:] = True
    ia, ib = own(ink, [a, b])
    assert ia and ib and not set(ia) & set(ib)


def test_measure_lines_reads_height_and_color():
    mask = ink_of("Hello world", size=60)
    draft = np.where(mask[..., None], np.array([200, 30, 30], np.uint8), np.uint8(255))
    H, W = mask.shape
    (g,) = measure_lines(draft, Ink(mask), [np.ones((H, W), bool)], [0.0], [[0, 0, W, H]])
    assert g["shape"] == "straight"
    ys = np.nonzero(mask)[0]
    assert 0.5 * np.ptp(ys) < g["h_cap"] <= np.ptp(ys) + 1
    r, gg, b = (int(g["color"][i:i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and gg < 80 and b < 80


def test_rows_link_lines_on_one_baseline_with_different_sizes():
    from textfix.lines.runs import rows
    boxes = [[100, 500, 180, 560], [190, 400, 450, 565], [460, 510, 520, 562], [100, 700, 300, 740], [400, 702, 600, 742]]
    heights = [50, 150, 45, 35, 35]          # "Giảm" | "30%" | "tất"; tên | giá cùng cỡ
    bases = [560, 563, 561, 740, 741]
    assert rows(boxes, heights, bases) == [0, 0, 0, None, None]


def test_bullet_is_ornament_and_box_starts_at_text():
    from textfix.lines.lead import lead
    mask = np.pad(ink_of("Hello world"), ((0, 0), (80, 0)))   # chỗ cho chấm đứng trước
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    x0, cy, h = int(xs.min()), int((ys.min() + ys.max()) / 2), int(np.ptp(ys))
    yy, xx = np.ogrid[:H, :W]
    dot = (yy - cy) ** 2 + (xx - (x0 - h)) ** 2 <= (h // 6) ** 2   # chấm tròn đứng trước, cách một khe
    ink, g = group_of(mask | dot)
    orn, new_x0 = lead(g, ink, box_x0=x0 - h - h // 6)
    assert len(orn) == 1 and ink.stats[orn[0], 0] < x0
    assert new_x0 is not None and x0 - 0.2 * h <= new_x0 <= x0


def test_no_ornament_no_trim():
    from textfix.lines.lead import lead
    mask = ink_of("Hello world")
    ink, g = group_of(mask)
    x0 = float(np.nonzero(mask)[1].min())
    assert lead(g, ink, box_x0=x0 - 2) == ([], None)


def test_missed_piece_inside_an_ocr_line_is_not_a_new_line():
    # 11/11 edit1 e14: vùng "Lương 3" [313,616,513,676] nằm trong dòng "Lương 30k/giờ" [304,605,729,685] (tâm dòng lọt ra ngoài
    # vùng của bộ dò) -> không tạo dòng thứ hai
    from textfix.lines.curve import INSIDE_LINE, _inside
    assert _inside([313, 616, 513, 676], [304, 605, 729, 685]) >= INSIDE_LINE
    assert _inside([100, 100, 300, 160], [250, 100, 600, 160]) < INSIDE_LINE   # chỉ chạm một phần: vẫn là dòng sót
