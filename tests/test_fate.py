"""Số phận dòng (compose.fate), khuôn trả lời mỗi dòng một mục (compose.vlm.schema / compose.parse) và chữ lấp
(compose.assign.filler) với VLM giả: mỗi test là một luật. 16/10: không còn hỏi lại VLM; dòng FLUX vẽ thừa viết chữ VLM đoán."""

from __future__ import annotations

import numpy as np
import pytest

from textfix.compose.parse import parse

from textfix.compose.assign import resolve
from textfix.compose import fate, vlm


def line(i: int, y: int, **kw) -> dict:
    return {"id": f"L{i}", "box": [100, y, 500, y + 40], "poly": None, "ocr": f"chu {i}", "size": 40.0, "color": "#000000",
            "angle": 0.0, **kw}


def page(*L) -> dict:
    return {"L": list(L), "size": [800, 800], "by": {l["id"]: l for l in L}}


TEXTS = [{"text": "Khai trương", "role": "headline"}, {"text": "Giảm 30%", "role": "offer"}]


def entry(marks, kind="text", client=None, text=None) -> dict:
    return {"lines": marks, "client": client, "kind": kind, "text": text, "why": "test"}


def fates(M, entries):
    A = parse({"assign": entries, "styles": {}})
    U, log, _ = resolve(A, M, TEXTS)
    return fate.table(U, M)


def test_vlm_skip_is_trusted():
    M = page(line(1, 100), line(2, 200))
    t = fates(M, [entry(["L1"], client="T0"), entry(["L2"], kind="skip")])
    assert t["L1"]["fate"] == "text" and t["L2"]["fate"] == "skip"


def test_omitted_line_is_kept():
    # VLM không theo khuôn (khuôn JSON bắt buộc đủ dòng): dòng không nhắc tới -> giữ, không hỏi lại
    M = page(line(1, 100), line(2, 200))
    t = fates(M, [entry(["L1"], client="T0")])
    assert t["L2"]["fate"] == "keep" and "bỏ sót" in t["L2"]["why"]


def test_repeated_text_is_written_once():
    # FLUX vẽ tiêu đề hai lần (xa nhau), VLM ghép cùng một câu vào cả hai: viết MỘT lần, chỗ kia gỡ trống (11/11; trước: viết lặp)
    M = page(line(1, 100, ocr="Khai truong"), line(2, 300, ocr="Khoi trung"))
    A = parse({"assign": [entry(["L1"], client="T0"), entry(["L2"], client="T0"), entry([], client="T1")], "styles": {}})
    U, _, _ = resolve(A, M, TEXTS)
    t = fate.table(U, M)
    assert sorted([t["L1"]["fate"], t["L2"]["fate"]]) == ["skip", "text"]
    assert [u["text_vlm"] or TEXTS[u["clients"][0]]["text"] for u in U if u["kind"] == "text"] == ["Khai trương"]


def test_ask_is_one_call_with_a_schema_requiring_every_line(monkeypatch):
    calls = []

    def call(system, parts, meta=None, fresh=False, schema=None):
        calls.append(schema)
        return {"lines": {"L1": {"kind": "text", "client": ["T0"], "part": None, "text": None, "role": None, "why": ""},
                          "L2": {"kind": "keep", "client": [], "part": None, "text": None, "role": None, "why": ""}}}
    monkeypatch.setattr("textfix.compose.call.marked_image", lambda img, M, **k: __import__("PIL.Image").Image.fromarray(img))
    M = page(line(1, 100), line(2, 200))
    img = np.zeros((64, 64, 3), np.uint8)
    A = vlm.ask(img, M, "prompt", TEXTS, call)
    S = calls[0]["properties"]
    assert len(calls) == 1 and S["lines"]["required"] == ["L1", "L2"]
    assert S["lines"]["properties"]["L1"]["properties"]["client"]["items"]["enum"] == ["T0", "T1"]
    assert sorted(S["styles"]["required"]) == ["headline", "offer"]
    assert [(u["marks"], u["kind"], u["clients"]) for u in A["assign"]] == [(["L1"], "text", [0]), (["L2"], "keep", [])]
    # 12/11: khuôn không còn "why" / "notes" (token sinh ra code không dùng)
    assert "why" not in S["lines"]["properties"]["L1"]["properties"] and "notes" not in S


def test_vlm_images_are_downscaled_with_readable_labels(monkeypatch):
    import io

    from PIL import Image
    seen = {}

    def call(system, parts, meta=None, fresh=False, schema=None):
        seen["parts"] = parts
        return {"lines": {"L1": {"kind": "text", "client": ["T0"], "part": None, "text": None, "role": None}}}
    monkeypatch.setenv("TEXTFIX_VLM_MAX_SIDE", "512")
    vlm.ask(np.zeros((1024, 768, 3), np.uint8), page(line(1, 100)), "prompt", TEXTS, call)
    ims = [Image.open(io.BytesIO(p)) for p in seen["parts"] if isinstance(p, bytes)]
    assert [im.size for im in ims] == [(384, 512), (384, 512)]
    assert "images shown at 384x512" in seen["parts"][0] and "Poster size: 768x1024px" in seen["parts"][0]


def L(kind="text", client=(), part=None, text=None) -> dict:
    return {"kind": kind, "client": list(client), "part": part, "text": text, "role": None, "why": ""}


def test_parts_of_a_stacked_text_become_one_unit_and_code_breaks_it():
    # câu khách FLUX vẽ hai dòng cùng cỡ xếp chồng: một đơn vị, không split (code ngắt câu theo sức chứa)
    M = page(line(1, 100), line(2, 145))
    A = parse({"lines": {"L1": L(client=["T0"], part="Khai"), "L2": L(client=["T0"], part="trương")}}, M)
    assert [(u["marks"], u["split"]) for u in A["assign"]] == [(["L1", "L2"], None)]


def test_parts_of_different_size_keep_the_split():
    # "Giảm" nhỏ trên "30%" to: một đơn vị, mỗi dòng mẩu của nó
    M = page({**line(1, 100), "size": 20.0, "box": [100, 100, 200, 120]}, {**line(2, 125), "size": 80.0, "box": [100, 125, 400, 205]})
    A = parse({"lines": {"L2": L(client=["T1"], part="30%"), "L1": L(client=["T1"], part="Giảm")}}, M)
    assert [(u["marks"], u["split"]) for u in A["assign"]] == [(["L1", "L2"], ["Giảm", "30%"])]


def test_same_text_on_two_lines_without_parts_is_a_repeat():
    # cùng câu cho hai dòng mà không có mẩu: dòng lặp -> hai mục, compose.assign xử câu ghép hai lần
    M = page(line(1, 100), line(2, 300))
    A = parse({"lines": {"L1": L(client=["T0"]), "L2": L(client=["T0"])}}, M)
    assert [u["marks"] for u in A["assign"]] == [["L1"], ["L2"]]


@pytest.mark.parametrize("filler, want", [
    ("Giảm 30%", 1),                 # chữ lấp chính là câu khách chưa có chỗ -> viết câu đó
    ("Khai trương", None),           # FLUX vẽ lặp tiêu đề: viết theo
    ("Mở cửa 7:00", None),           # VLM đọc giờ trên nháp: viết theo (FLUX bịa thì viết theo, 16/10)
])
def test_filler_becomes_a_missing_client_text_else_is_written(filler, want):
    from collections import Counter

    from textfix.compose.assign import filler as verdict
    assert verdict(filler, TEXTS, Counter({0: 1})) == want


def test_fragment_inside_a_kept_logo_is_kept():
    # final2 d63e: Hi-SAM tách mẩu logo trên lon (OCR không đọc) thành dòng riêng; VLM ghép chữ lấp -> theo dòng logo: giữ
    M = page({**line(7, 0), "box": [469, 394, 576, 748], "angle": -85.0},
             {**line(6, 0), "box": [477, 553, 494, 576], "orphan": True, "ocr": ""}, line(1, 100))
    t = fates(M, [entry(["L1"], client="T0"), entry(["L7"], kind="keep"), entry(["L6"], text="Hàng thật")])
    assert t["L6"]["fate"] == "keep" and t["L7"]["fate"] == "keep"


def test_empty_split_line_beside_its_row_stays():
    # final2 g18: "Giảm" | "50%" cùng hàng, VLM dồn cả câu vào ô "50%" và để trống ô "Giảm"; "mow" là mẩu OCR trùng trong dòng dưới
    M = page({**line(1, 820), "box": [209, 832, 387, 922], "ocr": "Giam"}, {**line(2, 820), "box": [385, 822, 563, 922], "ocr": "5%"},
             {**line(3, 920), "box": [215, 921, 554, 977], "ocr": "moi dich vu"}, {**line(4, 935), "box": [223, 935, 324, 970], "ocr": "mow"})
    texts = [{"text": "Giảm 50% mọi dịch vụ", "role": "offer"}]
    A = parse({"assign": [{"lines": ["L1", "L2", "L3", "L4"], "client": "T0", "kind": "text",
                                "split": ["", "Giảm 50%", "mọi dịch vụ", ""]}]})
    U, _, _ = resolve(A, M, texts)
    t = fate.table(U, M)
    assert t["L1"]["fate"] == "text" and t["L4"]["fate"] == "skip"


def test_split_pieces_follow_the_text_not_the_reading_order():
    # final3 h13: "Giảm đến" nhỏ nằm trên "70%" to; thứ tự đọc coi là cùng hàng và đặt "70%" trước -> mẩu xếp theo câu, giữ
    M = page({**line(3, 0), "box": [291, 329, 468, 377], "size": 49.0}, {**line(4, 0), "box": [165, 342, 767, 592], "size": 251.0})
    texts = [{"text": "Giảm đến 70%", "role": "offer"}]
    A = parse({"lines": {"L3": L(client=["T0"], part="Giảm đến"), "L4": L(client=["T0"], part="70%")}}, M)
    U, _, _ = resolve(A, M, texts)
    u = next(u for u in U if u["kind"] == "text")
    assert u["split_ok"] and dict(zip(u["marks"], u["split"])) == {"L3": "Giảm đến", "L4": "70%"}



def _stack(y0: int, sizes, ocrs) -> list[dict]:
    out, y = [], y0
    for i, (s, o) in enumerate(zip(sizes, ocrs), 1):
        out.append(line(i, y, size=float(s), ocr=o, box=[100, y, 600, y + s]))
        y += s + 5
    return out


def test_pieces_of_one_text_in_different_sizes_are_one_unit():
    # 12/11 chat1 c10: câu vẽ thành 3 dòng xếp chồng, dòng cuối to 1,46 lần, VLM gán cả câu cho từng dòng -> trước viết lặp
    # 3 lần; tổng chữ OCR ~ độ dài câu = các dòng là MẨU của câu -> một đơn vị (dựng "lines", mỗi dòng giữ cỡ nháp)
    Ls = _stack(100, [70, 79, 102], ["Kahi duaing", "shop quuan ao", "Linh Chi"])
    M, T = page(*Ls), [{"text": "Khai trương shop quần áo Linh Chi", "role": "headline"}]
    A = parse({"assign": [entry([l["id"]], client="T0") for l in Ls], "styles": {}})
    U, _, _ = resolve(A, M, T)
    assert [u["marks"] for u in U if u["kind"] == "text"] == [["L1", "L2", "L3"]]


def test_whole_text_drawn_again_below_is_written_once():
    # FLUX vẽ lặp CẢ câu ngay dưới, cỡ khác: mỗi dòng mang gần trọn câu (OCR / câu ~ 2) -> không gộp thành một câu chia dòng;
    # câu viết một lần, dòng lặp gỡ trống (11/11)
    Ls = _stack(100, [60, 100], ["Tiem nail Hong", "Tiem nail Hong"])
    M, T = page(*Ls), [{"text": "Tiệm nail Hồng", "role": "headline"}]
    A = parse({"assign": [entry([l["id"]], client="T0") for l in Ls], "styles": {}})
    U, _, _ = resolve(A, M, T)
    assert [len(u["marks"]) for u in U if u["kind"] == "text"] == [1]


def test_every_part_being_the_whole_text_is_a_repeat_not_a_split():
    # 11/11 edit1 e12: VLM ghi "part" = nguyên câu cho hai dòng cách xa nhau -> nguyên câu ở MỘT chỗ, chỗ kia gỡ trống (trước:
    # code ngắt câu làm đôi "Combo 4" | "người 459k" qua hai chỗ)
    M = page({**line(3, 321), "ocr": "Combo 4nguoi459k"}, {**line(6, 824), "ocr": "Combo4nghoi459k"})
    T = [{"text": "Combo 4 người 459k", "role": "offer"}]
    A = parse({"lines": {"L3": L(client=["T0"], part="Combo 4 người 459k"), "L6": L(client=["T0"], part="Combo 4 người 459k")}}, M)
    U, log, _ = resolve(A, M, T)
    us = [u for u in U if u["kind"] == "text"]
    assert len(us) == 1 and len(us[0]["marks"]) == 1 and not us[0].get("split")
    assert (us[0]["text_vlm"] or T[us[0]["clients"][0]]["text"]) == "Combo 4 người 459k"


# ---------------------------------------------------------------------------------------------- sửa poster (11/11 prod)
def test_filler_equal_to_a_removed_old_text_is_wiped():
    # e12 / f12: người dùng bỏ dòng ngày; FLUX vẽ lại dòng đó, VLM đoán chữ = câu cũ -> không viết lại
    M = page(line(1, 100, ocr="Khai truong"), line(2, 300, ocr="Uu dai mua Noel"))
    A = parse({"lines": {"L1": L(client=["T0"]), "L2": L(text="Ưu đãi mùa Noel")}}, M)
    U, log, _ = resolve(A, M, TEXTS, gone=["Ưu đãi mùa Noel"])
    assert fate.table(U, M)["L2"]["fate"] == "skip" and any("đã bỏ" in x for x in log)
    U, _, _ = resolve(A, M, TEXTS)   # không sửa poster: chữ lấp vẫn viết theo nháp (16/10)
    assert fate.table(U, M)["L2"]["fate"] == "text"


def test_restored_line_gets_its_text_even_when_vlm_skips_it():
    # e03 prod: FLUX xoá khối hotline; dòng trả về chỗ cũ của T1, VLM bảo skip -> vẫn đặt T1 (không mất chữ khách)
    M = page(line(1, 100, ocr="Khai truong"), {**line(2, 300, ocr=""), "restored": "T1"})
    A = parse({"lines": {"L1": L(client=["T0"]), "L2": L(kind="skip")}}, M)
    U, log, missing = resolve(A, M, TEXTS)
    u = next(u for u in U if "L2" in u["marks"])
    assert u["kind"] == "text" and u["clients"] == [1] and missing == []
    # T1 đã có chỗ khác: dòng trả về theo VLM (không viết lặp)
    M2 = page(line(1, 100), line(3, 200, ocr="Giam 30%"), {**line(2, 300, ocr=""), "restored": "T1"})
    A2 = parse({"lines": {"L1": L(client=["T0"]), "L3": L(client=["T1"]), "L2": L(kind="skip")}}, M2)
    assert fate.table(resolve(A2, M2, TEXTS)[0], M2)["L2"]["fate"] == "skip"


def test_filler_echoing_a_placed_text_or_inventing_numbers_is_wiped():
    # 11/11 hard1 h22 "Lẩu Nấm Rừng" dưới "Nhà hàng Lẩu Nấm Rừng"; sme1 s05 bảng giờ FLUX vẽ, VLM điền giờ bịa
    T = [{"text": "Nhà hàng Lẩu Nấm Rừng", "role": "headline"}, {"text": "Mở cửa 10h - 23h", "role": "info"}]
    M = page(line(1, 100, ocr="Nha hang Lau"), line(2, 300, ocr="Lau Nam Rung"), line(3, 500, ocr="Mo cua 10h"),
             line(4, 700, ocr="Sang 7:00"), line(5, 900, ocr="Khong gian"))
    A = parse({"lines": {"L1": L(client=["T0"]), "L2": L(text="Lẩu Nấm Rừng"), "L3": L(client=["T1"]),
                         "L4": L(text="Sáng 7:00 - 12:00"), "L5": L(text="Không gian ấm cúng")}}, M)
    U, log, _ = resolve(A, M, T)
    t = fate.table(U, M)
    assert t["L2"]["fate"] == t["L4"]["fate"] == "skip"
    assert t["L5"]["fate"] == "text"   # chữ lấp không số, không lặp: vẫn viết theo nháp
    assert any("lặp một câu" in x for x in log) and any("số khách không đưa" in x for x in log)


def test_filler_where_ocr_reads_the_text_takes_the_text():
    # golden r10: VLM ghép lệch, câu "Tận hưởng ngay" vào ô "15k", nút đọc "Tan huong ngay" là dòng lấp -> câu về nút, ô kia gỡ
    T = [{"text": "Thực đơn chè", "role": "headline"}, {"text": "Tận hưởng ngay", "role": "cta"}, {"text": "15k", "role": "price"}]
    M = page(line(1, 100, ocr="Thuc don che"), {**line(2, 500, ocr="15k"), "size": 52.0},
             {**line(3, 800, ocr="Tan huong ngay"), "size": 40.0})
    A = parse({"lines": {"L1": L(client=["T0"]), "L2": L(client=["T1"]), "L3": L(text="Tận hưởng ngay")}}, M)
    U, log, miss = resolve(A, M, T)
    t = fate.table(U, M)
    assert t["L3"]["fate"] == "text" and t["L2"]["fate"] == "skip" and miss == [2]
    assert [u["clients"] for u in U if "L3" in u["marks"]] == [[1]]


def test_two_buttons_with_the_same_words_both_keep_them():
    # final3 h21: câu nút VLM ghép hai chỗ, chỗ thua gỡ trống -> nút trống. Nút bấm (cta) vẫn viết lặp theo nháp.
    T = [{"text": "Khai trương", "role": "headline"}, {"text": "Đặt lịch ngay", "role": "cta"}]
    M = page(line(1, 100, ocr="Khai truong"), line(2, 500, ocr="Dat lich ngay"), line(3, 800, ocr="Dat lich nqay"))
    A = parse({"lines": {"L1": L(client=["T0"]), "L2": L(client=["T1"]), "L3": L(client=["T1"])}}, M)
    U, _, _ = resolve(A, M, T)
    t = fate.table(U, M)
    assert t["L2"]["fate"] == t["L3"]["fate"] == "text"


def test_texts_the_client_numbered_follow_the_reading_order():
    from textfix.compose.assign import order
    texts = [{"text": "Nội quy"}] + [{"text": f"{i}. Luật số {i}"} for i in range(1, 5)]
    M = {"by": {f"L{i}": {"box": [10, 100 * i, 300, 100 * i + 30]} for i in range(1, 6)}}
    U = [{"id": f"u{i}", "kind": "text", "marks": [f"L{i}"], "clients": [k], "split": None}
         for i, k in zip(range(1, 6), [0, 1, 4, 2, 3])]
    log = order(U, M, texts)
    assert log and [u["clients"][0] for u in U] == [0, 1, 2, 3, 4]
