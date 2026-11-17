"""Poster nhiều ô (textfix/panels.py): tìm đường chia, gắn dòng vào ô, ràng buộc ghép câu / bố cục, khối thông tin, luồng sửa."""

import glob
import os

import numpy as np
import pytest

from textfix import meta, panels


def grid(cols, rows, W=600, H=600, gutter=0, colors=None, noise=0):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 245, np.uint8)
    colors = colors or [(200, 80, 80), (80, 160, 90), (70, 90, 200), (220, 190, 60)]
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c * W // cols, (c + 1) * W // cols
            y0, y1 = r * H // rows, (r + 1) * H // rows
            img[y0 + gutter:y1 - gutter, x0 + gutter:x1 - gutter] = colors[(r * cols + c) % len(colors)]
    if noise:
        img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    return img


@pytest.mark.parametrize("cols,rows,W,H", [(2, 1, 600, 600), (3, 1, 900, 500), (1, 3, 500, 900), (2, 2, 600, 600)])
def test_find_equal_panels_with_or_without_a_gutter(cols, rows, W, H):
    for gutter in (0, 6):
        P = panels.find(grid(cols, rows, W, H, gutter), cols * rows)
        assert P and (P["cols"], P["rows"]) == (cols, rows), (gutter, P)
        assert all(abs(x - W * k / cols) <= 8 for k, x in enumerate(P["xs"], 1))
        assert len(P["rects"]) == cols * rows and P["rects"][0][:2] == [0, 0] and P["rects"][-1][2:] == [W, H]


def test_pastel_panels_are_found_by_color_not_brightness():
    """prod 17/11 giày 2x2: ô pastel sáng gần nhau -- đo độ sáng không ra đường chia."""
    pastel = [(240, 214, 222), (214, 236, 228), (226, 218, 242), (212, 226, 244)]
    P = panels.find(grid(2, 2, colors=pastel), 4)
    assert P and (P["cols"], P["rows"]) == (2, 2)


def test_a_plain_image_or_one_without_a_cut_has_no_panels():
    rng = np.random.default_rng(1)
    assert panels.find(np.full((400, 400, 3), 128, np.uint8), 2) is None
    assert panels.find(rng.integers(0, 255, (400, 400, 3)).astype(np.uint8), 3) is None
    assert panels.find(grid(2, 1), 3) is None   # hai ô, brief nói ba: không bịa đường chia


def test_prior_keeps_the_old_layout_and_searches_close_to_it():
    img = grid(3, 1, 900, 500)
    prior = {"n": 3, "cols": 3, "rows": 1, "xs": [0.3333, 0.6667], "ys": []}
    P = panels.find(img, 3, prior)
    assert P and P["cols"] == 3 and abs(P["xs"][0] - 300) <= 2
    far = {"n": 3, "cols": 3, "rows": 1, "xs": [0.2, 0.8], "ys": []}
    assert panels.find(img, 3, far) is None
    Q = panels.from_prior(far, 900, 500)
    assert Q["xs"] == [180, 720] and Q["rects"][1] == [180, 0, 720, 500]


def test_lines_get_the_panel_that_holds_them_and_crossing_lines_are_shared():
    P = panels.find(grid(2, 1), 2)
    M = {"L": [{"id": "L1", "box": [40, 40, 260, 80]}, {"id": "L2", "box": [340, 40, 560, 80]},
               {"id": "L3", "box": [200, 500, 420, 540]}]}
    panels.tag_lines(M, P)
    assert [ln["panel"] for ln in M["L"]] == [1, 2, 0]
    texts = [{"text": "A", "panel": 1}, {"text": "B", "panel": 2}, {"text": "Hotline", "panel": 0}]
    al = panels.allow(M, texts)
    assert al == {"L1": [0, 2], "L2": [1, 2], "L3": [0, 1, 2]}


def test_vlm_schema_limits_each_line_to_its_panel():
    from textfix.compose.vlm import schema
    M = {"L": [{"id": "L1"}, {"id": "L2"}]}
    texts = [{"text": "A", "role": "headline"}, {"text": "B", "role": "headline"}]
    S = schema(M, texts, short=True, allow={"L1": [0], "L2": [1]})
    lines = S["properties"]["lines"]["properties"]
    assert lines["L1"]["properties"]["client"]["items"]["enum"] == ["T0"]
    assert lines["L2"]["properties"]["client"]["items"]["enum"] == ["T1"]
    plain = schema(M, texts, short=True)   # poster thường: khuôn như cũ
    assert plain["properties"]["lines"]["properties"]["L1"]["properties"]["client"]["items"]["enum"] == ["T0", "T1"]


def test_text_list_for_the_vlm_names_panels_only_on_split_images():
    from textfix.compose.call import _texts
    assert _texts([{"text": "A", "role": "cta"}]) == "T0 (cta): A"
    assert _texts([{"text": "A", "role": "cta", "panel": 2}, {"text": "B", "role": "info", "panel": 0}]) == \
        "T0 (cta, panel 2): A\nT1 (info, whole image): B"


def test_room_stops_at_the_panel_cut():
    from textfix.compose.size import outside_panel
    m = outside_panel([300, 0, 600, 600], 600, 600)
    assert m[:, :304].all() and not m[10, 310:600].any()   # lùi 1,5% ở đường chia, mép ảnh không lùi


@pytest.mark.parametrize("req,want", [
    ("o ben phai doi thanh mua 3 tang 1", {2}), ("doi gia o 2 thanh 55k", {2}), ("o thu nhat them chu moi", {1}),
    ("change the price in the right panel", {2}), ("doi mau nen", None), ("them anh 2 ly tra sua", None)])
def test_named_panel_two_columns(req, want):
    assert panels.named(req, {"n": 2, "cols": 2, "rows": 1}) == want


def test_named_panel_grid_and_rows():
    g = {"n": 4, "cols": 2, "rows": 2}
    assert panels.named("o tren phai doi gia", g) == {2}
    assert panels.named("bottom-left panel", g) == {3}
    assert panels.named("khung duoi", g) == {3, 4}
    assert panels.named("doi anh o giua thanh tra dao", {"n": 3, "cols": 3, "rows": 1}) == {2}
    assert panels.named("o giua", {"n": 3, "cols": 1, "rows": 3}) == {2}


def test_brief_keeps_panels_and_one_copy_per_panel():
    from textfix.brief import make_brief
    raw = {"intent": "poster", "panels": 3, "prompt_en": 'Three panels: "Trà sữa Mây" "Giảm 20%" "Freeship"',
           "texts": [{"text": "Trà sữa Mây", "role": "headline", "source": "user", "panel": 1},
                     {"text": "Giảm 20%", "role": "offer", "source": "user", "panel": 1},
                     {"text": "Trà sữa Mây", "role": "headline", "source": "user", "panel": 2},
                     {"text": "Freeship", "role": "offer", "source": "user", "panel": 9}]}
    B = make_brief(lambda *a, **k: raw, "ba poster trà sữa Mây: giảm 20%, freeship")
    assert B["panels"] == 3
    assert [(t["text"], t["panel"]) for t in B["texts"]] == [("Trà sữa Mây", 1), ("Giảm 20%", 1), ("Trà sữa Mây", 2), ("Freeship", 0)]
    one = make_brief(lambda *a, **k: {**raw, "panels": 1}, "ba poster trà sữa Mây: giảm 20%, freeship")
    assert "panels" not in one and all("panel" not in t for t in one["texts"])


def test_panels_ride_in_the_png_and_bad_blocks_are_dropped():
    info = {"run_id": "r1", "texts": [{"text": "A", "panel": 2}, {"text": "B", "panel": 7}],
            "panels": {"n": 2, "cols": 2, "rows": 1, "xs": [0.5], "ys": []}}
    got = meta.read(meta.save_png(np.zeros((8, 8, 3), np.uint8), info))
    assert got["panels"]["n"] == 2 and [t["panel"] for t in got["texts"]] == [2, 0]
    bad = meta.read(meta.save_png(np.zeros((8, 8, 3), np.uint8), {**info, "panels": {"cols": 3, "rows": 1, "xs": [0.5], "ys": []}}))
    assert "panels" not in bad and all("panel" not in t for t in bad["texts"])
    plain = meta.read(meta.save_png(np.zeros((8, 8, 3), np.uint8), {"run_id": "r0", "texts": [{"text": "A"}]}))
    assert "panels" not in plain


def test_edit_keeps_old_panels_and_moves_a_price_change_into_the_named_panel():
    from textfix.edit.plan import make_edit
    P = {"n": 2, "cols": 2, "rows": 1, "xs": [0.5], "ys": []}
    old = {"run_id": "r", "texts": [{"text": "Trà sữa 25k", "role": "price", "source": "user", "panel": 1},
                                    {"text": "Trà sữa 25k", "role": "price", "source": "user", "panel": 2}],
           "prompt_en": "two panels", "aspect": "1:1", "panels": P}
    seen = {}

    def llm(system, msgs, schema=None):
        seen["msg"], seen["schema"] = msgs[0], schema
        return {"texts": [{"text": "Trà sữa 30k", "role": "price", "source": "user", "old": 0, "panel": 2},
                          {"text": "Trà sữa 25k", "role": "price", "source": "user", "old": 1, "panel": 2},
                          {"text": "Hotline 0909 123 456", "role": "contact", "source": "user", "old": -1, "panel": 0}],
                "prompt_en": "The poster from image 1 ...", "aspect": None}
    B = make_edit(llm, "đổi giá ô bên phải thành 30k, thêm hotline 0909 123 456", old)
    assert "[ô 1 · trái]" in seen["msg"] and "panel" in seen["schema"]["properties"]["texts"]["items"]["required"]
    assert [(t["text"], t["panel"]) for t in B["texts"]] == [("Trà sữa 25k", 1), ("Trà sữa 30k", 2), ("Hotline 0909 123 456", 0)]


def test_edit_of_a_plain_poster_uses_the_old_schema():
    from textfix.edit.instructions import SCHEMA, schema_for
    assert schema_for({"texts": []}) is SCHEMA


@pytest.mark.skipif(not glob.glob("output/grid_probe/*.png"), reason="ảnh prod 17/11 không có tại máy")
def test_real_flux_grids_from_prod():
    """24 ảnh FLUX prod 17/11 (scratchpad grid_probe): đúng bố cục đã dặn, trừ ca giày pastel lượt 2 (đường chia mờ)."""
    from PIL import Image
    want = {"n2_11": (2, 1), "n2_916": (1, 2), "n3_169": (3, 1), "n3_11": (3, 1), "n3_916": (1, 3), "n4": (2, 2), "t2_11": (2, 1),
            "t3_169": (3, 1), "t4_11": (2, 2), "t3_916": (1, 3)}
    bad = []
    for p in sorted(glob.glob("output/grid_probe/*.png")):
        name = os.path.basename(p)
        key = next((k for k in want if name.startswith(k)), None)
        if not key:
            continue
        cols, rows = want[key]
        P = panels.find(np.asarray(Image.open(p).convert("RGB")), cols * rows)
        if not P or (P["cols"], P["rows"]) != (cols, rows):
            bad.append(name)
    assert len(bad) <= 2, bad


def test_edit_matches_old_texts_by_words_when_the_llm_numbers_them_by_panel():
    """notebook 17/11 p13: ảnh 4 ô, LLM ghi old = số ô 1..4 thay cho T0..T3 -> mọi câu lệch một, giá đổi nhầm món."""
    from textfix.edit.plan import make_edit
    P = {"n": 4, "cols": 2, "rows": 2, "xs": [0.5], "ys": [0.5]}
    names = ["Cà phê sữa 25k", "Nước cam 30k", "Trà chanh 20k", "Sinh tố bơ 35k"]
    old = {"run_id": "r", "texts": [{"text": t, "role": "list_item", "source": "user", "panel": i + 1} for i, t in enumerate(names)],
           "prompt_en": "grid", "aspect": "1:1", "panels": P}
    out = [("Cà phê sữa 25k", 1), ("Nước cam 32k", 2), ("Trà chanh 20k", 3), ("Sinh tố bơ 35k", 4)]
    llm = lambda s, m, schema=None: {"texts": [{"text": t, "role": "list_item", "source": "user", "old": o, "panel": o}   # noqa: E731
                                               for t, o in out], "prompt_en": "The poster from image 1", "aspect": None}
    B = make_edit(llm, "sửa giá nước cam trên ảnh này thành 32k", old)
    assert [(t["text"], t["panel"]) for t in B["texts"]] == [("Cà phê sữa 25k", 1), ("Nước cam 32k", 2), ("Trà chanh 20k", 3),
                                                             ("Sinh tố bơ 35k", 4)]


def test_panel_counts_in_the_chat_are_not_printed():
    """notebook 17/11 p11: "làm 1 ảnh chia 3 ô ..., ô 2 đắp bột" -> code thêm "1 ảnh chia", "3 ô cho tiệm nail Bông", "Ô 2 đắp bột"."""
    from textfix.brief import make_brief
    raw = {"intent": "poster", "panels": 3, "prompt_en": "three panels",
           "texts": [{"text": "Tiệm nail Bông", "role": "headline", "source": "user", "panel": 0},
                     {"text": "Sơn gel 120k", "role": "list_item", "source": "user", "panel": 1},
                     {"text": "Đắp bột 250k", "role": "list_item", "source": "user", "panel": 2},
                     {"text": "Vẽ móng 50k", "role": "list_item", "source": "user", "panel": 3}]}
    B = make_brief(lambda *a, **k: raw, "làm 1 ảnh chia 3 ô cho tiệm nail Bông: ô 1 sơn gel 120k, ô 2 đắp bột 250k, ô 3 vẽ móng 50k, tông hồng")
    assert [t["text"] for t in B["texts"]] == ["Tiệm nail Bông", "Sơn gel 120k", "Đắp bột 250k", "Vẽ móng 50k"]
