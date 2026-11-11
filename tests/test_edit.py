"""Sửa poster (textfix/edit, textfix/meta.py): khối thông tin trong PNG, LLM sửa (code kiểm lại), đọc chữ ảnh lạ. LLM / VLM giả."""

from __future__ import annotations

import io

import numpy as np
import pytest

from textfix import meta
from textfix.brief import make_brief
from textfix.edit import make_edit, nearest_ratio
from textfix.edit.read import collect

OLD = {"run_id": "run_a", "depth": 0, "aspect": "9:16", "product": False, "prompt_en": 'A pink nail salon poster with "Tiệm nail Hồng Nhung".',
       "texts": [{"text": "Tiệm nail Hồng Nhung", "role": "headline", "source": "user"},
                 {"text": "Giảm 30% sơn gel", "role": "offer", "source": "user"},
                 {"text": "Khai trương ngày 5/11", "role": "date", "source": "user"},
                 {"text": "Đặt lịch ngay", "role": "cta", "source": "ai"}]}


def llm_says(texts, prompt_en=None, aspect=None):
    seen = {}

    def llm(system, parts, schema=None):
        seen.update(system=system, message=parts[0], schema=schema)
        pe = prompt_en if prompt_en is not None else "The poster from image 1 with " + ", ".join(f'"{t[0]}"' for t in texts) + "."
        return {"texts": [{"text": t[0], "role": t[1], "source": t[2], "old": t[3]} for t in texts], "prompt_en": pe,
                "aspect": aspect}
    return llm, seen


KEEP = [("Tiệm nail Hồng Nhung", "headline", "user", 0), ("Khai trương ngày 5/11", "date", "user", 2),
        ("Đặt lịch ngay", "cta", "ai", 3)]


# ---------------------------------------------------------------------------------------------- khối thông tin PNG
def test_meta_absent_for_plain_png_jpeg_and_junk():
    from PIL import Image
    for fmt in ("PNG", "JPEG"):
        b = io.BytesIO()
        Image.new("RGB", (4, 4)).save(b, fmt)
        assert meta.read(b.getvalue()) is None
    assert meta.read(b"") is None and meta.read(b"\x89PNG\r\n\x1a\nxx") is None


def test_meta_survives_reencode_free_roundtrip_but_not_recompression():
    from PIL import Image
    raw = meta.save_png(np.zeros((8, 8, 3), np.uint8), meta.info("run_x", {"texts": OLD["texts"]}, "1:1", False))
    assert meta.read(raw)["texts"] == OLD["texts"]
    b = io.BytesIO()
    Image.open(io.BytesIO(raw)).save(b, "PNG")   # kho ảnh / ứng dụng nén lại: mất khối -> ảnh lạ
    assert meta.read(b.getvalue()) is None


def test_nearest_ratio():
    assert (nearest_ratio(1024, 1024), nearest_ratio(768, 1344), nearest_ratio(1920, 1080), nearest_ratio(900, 1200)) == \
        ("1:1", "9:16", "16:9", "3:4")


# ---------------------------------------------------------------------------------------------- LLM sửa
def test_edit_changes_one_text_keeps_the_rest():
    llm, seen = llm_says([KEEP[0], ("Giảm 50% sơn gel", "offer", "user", 1), *KEEP[1:]])
    B = make_edit(llm, "đổi thành giảm 50%", OLD)
    assert [t["text"] for t in B["texts"]] == ["Tiệm nail Hồng Nhung", "Giảm 50% sơn gel", "Khai trương ngày 5/11", "Đặt lịch ngay"]
    assert B["texts"][3]["source"] == "ai" and B["mode"] == "poster" and B["edit"]["parent"] == "run_a"
    assert B["edit"]["depth"] == 1 and B["edit"]["aspect"] is None and B["kho"] == []
    assert "T1 (offer, user): Giảm 30% sơn gel" in seen["message"] and "đổi thành giảm 50%" in seen["message"]
    assert B["prompt_en"].count('"Giảm 50% sơn gel"') == 1
    assert B["edit"]["gone"] == ["Giảm 30% sơn gel"]   # câu cũ đã đổi: FLUX vẽ lại thì không viết (assign._drop_gone)


def test_edit_with_invented_number_reverts_to_old_text():
    llm, _ = llm_says([KEEP[0], ("Giảm 70% sơn gel", "offer", "user", 1), *KEEP[1:]])
    B = make_edit(llm, "làm nền màu xanh", OLD)
    assert B["texts"][1]["text"] == "Giảm 30% sơn gel" and B["log"]["reverted"]
    assert '"Giảm 70% sơn gel"' not in B["prompt_en"] and '"Giảm 30% sơn gel"' in B["prompt_en"]   # prompt khớp danh sách


def test_text_dropped_without_remove_cue_is_restored_in_place():
    llm, _ = llm_says(KEEP)   # LLM chép sót T1, lời sửa chỉ nói về màu
    B = make_edit(llm, "làm nền màu xanh dương", OLD)
    assert [t["text"] for t in B["texts"]] == [t["text"] for t in OLD["texts"]] and B["log"]["restored"]


def test_change_color_request_is_not_a_removal():
    llm, _ = llm_says(KEEP)   # "đổi" không phải ý bỏ câu: câu chép sót vẫn được trả lại
    assert len(make_edit(llm, "đổi màu nền sang xanh", OLD)["texts"]) == 4


def test_replacement_marked_as_new_text_is_not_duplicated():
    llm, _ = llm_says([KEEP[0], ("Mua 1 tặng 1", "offer", "user", -1), *KEEP[1:]])
    B = make_edit(llm, "thay ưu đãi thành mua 1 tặng 1", OLD)
    assert [t["text"] for t in B["texts"]] == ["Tiệm nail Hồng Nhung", "Mua 1 tặng 1", "Khai trương ngày 5/11", "Đặt lịch ngay"]


def test_text_removed_on_request():
    llm, _ = llm_says(KEEP)
    B = make_edit(llm, "bỏ dòng giảm giá đi", OLD)
    assert "Giảm 30% sơn gel" not in [t["text"] for t in B["texts"]] and "restored" not in B["log"]


def test_new_text_needs_its_facts_in_the_request():
    llm, _ = llm_says([*OLD_ALL(), ("Hotline 0909 123 456", "contact", "user", -1), ("Gọi 0988 000 111", "contact", "user", -1)])
    B = make_edit(llm, "thêm hotline 0909123456", OLD)
    got = [t["text"] for t in B["texts"]]
    assert "Hotline 0909 123 456" in got and "Gọi 0988 000 111" not in got


def test_new_support_copy_follows_brief_rules():
    llm, _ = llm_says([*OLD_ALL(), ("Không gian thư giãn", "body", "ai", -1), ("Tặng quà 100k", "body", "ai", -1)])
    B = make_edit(llm, "viết thêm một câu slogan", OLD)
    got = [t["text"] for t in B["texts"]]
    assert "Không gian thư giãn" in got and "Tặng quà 100k" not in got


def test_all_texts_removed_gives_image_mode_and_aspect_change():
    llm, _ = llm_says([], prompt_en="The poster from image 1 without any lettering.", aspect="16:9")
    B = make_edit(llm, "xoá hết chữ, chuyển sang khổ ngang", OLD)
    assert B["mode"] == "image" and B["texts"] == [] and B["edit"]["aspect"] == "16:9" and "No text" in B["prompt_en"]


def test_edit_prompt_always_references_image_1():
    llm, _ = llm_says(OLD_ALL(), prompt_en='A poster with "Tiệm nail Hồng Nhung" on blue.')
    assert "image 1" in make_edit(llm, "nền xanh", OLD)["prompt_en"]


def OLD_ALL():
    return [(t["text"], t["role"], t["source"], i) for i, t in enumerate(OLD["texts"])]


# ---------------------------------------------------------------------------------------------- brief: ảnh lạ cần sửa
def test_brief_edit_intent_only_with_an_attached_image():
    def llm(system, parts):
        llm.msg = parts[0]
        return {"intent": "edit", "texts": [], "prompt_en": ""}
    assert make_brief(llm, "sửa giá trên ảnh này thành 99k", product_image=True)["mode"] == "edit"
    assert "intent \"edit\"" in llm.msg
    B = make_brief(llm, "sửa giá thành 99k", product_image=False)   # không ảnh: không có gì để sửa -> tạo mới
    assert B["mode"] != "edit" and "intent \"edit\"" not in llm.msg


# ---------------------------------------------------------------------------------------------- đọc chữ ảnh lạ
def test_read_groups_lines_into_texts_and_keep_marks_product():
    M = {"L": [{"id": "L1"}, {"id": "L2"}, {"id": "L3"}, {"id": "L4"}, {"id": "L5"}]}
    raw = {"description_en": "A red sale poster.", "lines": {
        "L1": {"kind": "text", "text": "SIÊU SALE", "role": "headline", "group": 1},
        "L2": {"kind": "text", "text": "Giảm đến 50% toàn bộ", "role": "offer", "group": 2},
        "L3": {"kind": "text", "text": "sản phẩm", "role": "offer", "group": 2},
        "L4": {"kind": "keep", "text": "COCA", "role": "label", "group": 3},
        "L5": {"kind": "skip", "text": "", "role": "body", "group": 4}}}
    R = collect(raw, M)
    assert R["texts"] == [{"text": "SIÊU SALE", "role": "headline", "source": "user"},
                          {"text": "Giảm đến 50% toàn bộ sản phẩm", "role": "offer", "source": "user"}]
    assert R["product"] is True and R["prompt_en"] == "A red sale poster."


def test_read_photo_without_lines_skips_vlm(monkeypatch):
    from textfix import lines
    from textfix.edit import read
    monkeypatch.setattr(lines, "build", lambda img, ink: {"L": [], "size": [4, 4]})
    called = []
    R = read.read_image(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), bool), lambda *a, **k: called.append(1))
    assert R["texts"] == [] and not called


@pytest.mark.parametrize("ids", [["L1", "L2"]])
def test_read_schema_requires_every_line(ids):
    from textfix.edit.read import schema
    S = schema({"L": [{"id": i} for i in ids]})
    assert S["properties"]["lines"]["required"] == ids


# ---------------------------------------------------------------------------------------------- câu ở nhiều chỗ (e03)
def test_shown_counts_client_units_and_repeats_and_reaches_the_edit_message():
    ops = [{"kind": "text", "client": ["T0"], "html": "Spa Hoa Sen"},
           {"kind": "text", "client": ["T1"], "html": "Đặt lịch ngay"},
           {"kind": "text", "client": None, "html": "Đặt lịch<br>ngay"},        # viết lặp theo nháp (không client)
           {"kind": "skip", "client": None}]
    info = meta.info("run_x", {"texts": [{"text": "Spa Hoa Sen", "role": "headline", "source": "user", "shown": 3},
                                        {"text": "Đặt lịch ngay", "role": "cta", "source": "ai"}]}, "1:1", False)
    meta.set_shown(info, ops)
    assert "shown" not in info["texts"][0] and info["texts"][1]["shown"] == 2
    raw = meta.save_png(np.zeros((4, 4, 3), np.uint8), info)
    old = meta.read(raw)
    llm, seen = llm_says([(t["text"], t["role"], t["source"], i) for i, t in enumerate(old["texts"])])
    make_edit(llm, "nền xanh", old)
    assert "T1 (cta, ai, shown 2 times): Đặt lịch ngay" in seen["message"] and "shown N times" in seen["system"]


# ---------------------------------------------------------------------------------------------- chỗ cũ bị bỏ trống (e03 lượt 2)
def test_boxes_are_recorded_per_text():
    ops = [{"kind": "text", "client": ["T0"], "marks": ["L1", "L2"], "html": "x"},
           {"kind": "text", "client": ["T0", "T1"], "marks": ["L3"], "html": "y"}]   # một dòng hai câu: không ghi khung
    L = [{"id": "L1", "box": [100, 100, 300, 150]}, {"id": "L2", "box": [120, 160, 280, 200]}, {"id": "L3", "box": [0, 0, 10, 10]}]
    info = meta.set_shown({"texts": [{"text": "a"}, {"text": "b"}]}, ops, L, (1000, 500))
    assert info["texts"][0]["box"] == [[0.1, 0.2, 0.3, 0.4]] and "box" not in info["texts"][1]


def test_restore_adds_a_line_only_where_the_old_place_is_empty():
    from textfix.edit.slots import restore
    draft = np.full((1000, 1000, 3), 240, np.uint8)
    M = {"L": [{"id": "L1", "box": [100, 100, 900, 200]}]}
    texts = [{"text": "Tiêu đề", "box": [[0.1, 0.1, 0.9, 0.2]]},          # còn dòng ở đó
             {"text": "Đặt lịch ngay", "box": [[0.3, 0.85, 0.7, 0.92]]},   # nút trống
             {"text": "Mới"}]                                               # câu mới: không có khung cũ
    log = restore(M, texts, draft)
    assert [l["id"] for l in M["L"]] == ["L1", "L2"] and M["L"][1]["restored"] == "T1" and len(log) == 1
    assert M["L"][1]["box"] == [300.0, 850.0, 700.0, 920.0] and M["L"][1]["color"] == "#222222"
