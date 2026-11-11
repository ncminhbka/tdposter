"""Bước LLM đầu luồng, với LLM giả trả JSON dựng sẵn: mỗi test là một cam kết của textfix.brief."""

from __future__ import annotations

import pytest

from textfix.brief import make_brief
from textfix.brief.evidence import Evidence, fold
from textfix.brief.prompt import TOKEN_LIMIT, count_tokens


def fake_llm(texts, prompt_en, intent="poster"):
    resp = {"intent": intent, "texts": [{"text": t, "role": r, "source": s} for t, r, s in texts], "prompt_en": prompt_en}
    return lambda system, parts: resp


def brief(texts, prompt_en, prompt="poster", **kw):
    return make_brief(fake_llm(texts, prompt_en, **kw), prompt)


def by_text(B):
    return {t["text"]: t for t in B["texts"]}


NAIL = "làm poster cho tiệm nail Hồng Nhung, khai trương ngày 5/11, giảm 30% tất cả dịch vụ sơn gel, màu hồng pastel"


# ---------------------------------------------------------------- đối chiếu lời khách

def test_fold_ignores_case_and_diacritics():
    assert fold("Khai Trương ĐẸP") == fold("khai truong dep")


def test_numbers_match_with_or_without_spaces():
    ev = Evidence(["sdt 0909 123 456, giá 185.000đ"])
    assert ev.unknown_facts("Hotline 0909123456") == []
    assert ev.unknown_facts("Giá 185.000đ") == []
    assert ev.unknown_facts("Giá 190.000đ") == ["190000"]


# ---------------------------------------------------------------- chốt danh sách chữ

def test_texts_found_without_quotes_are_user_texts():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Khai trương ngày 5/11", "date", "user")],
              'A poster "Tiệm nail Hồng Nhung" with a ribbon "Khai trương ngày 5/11".', NAIL)
    assert [t["source"] for t in B["texts"]] == ["user", "user", "user"]
    assert by_text(B)["Giảm 30% tất cả dịch vụ sơn gel"]["source"] == "user"   # số khách LLM bỏ sót: code lấy lại


def test_invented_fact_is_dropped_from_texts_and_prompt():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Hotline 0909 123 456", "contact", "user")],
              'A poster "Tiệm nail Hồng Nhung". A card reads "Hotline 0909 123 456".', NAIL)
    assert "Hotline 0909 123 456" not in by_text(B)
    assert "0909" not in B["prompt_en"]


def test_llm_copy_error_snaps_back_to_user_wording_in_texts_and_prompt():
    B = brief([("Khánh trương ngày 5/11", "date", "user")], 'A ribbon "Khánh trương ngày 5/11".', NAIL)
    assert "Khai trương ngày 5/11" in by_text(B)
    assert '"Khai trương ngày 5/11"' in B["prompt_en"]


def test_llm_may_add_diacritics_to_unaccented_input():
    B = brief([("Bún bò Cô Ba", "headline", "user"), ("Giá 45.000đ", "price", "user")],
              'A sign "Bún bò Cô Ba", a tag "Giá 45.000đ".', "banner quan bun bo co ba, gia 45.000d")
    assert set(by_text(B)) == {"Bún bò Cô Ba", "Giá 45.000đ"}
    assert all(t["source"] == "user" for t in B["texts"])


def test_support_copy_kept_without_facts_or_promises():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Đặt lịch ngay", "cta", "ai"),
               ("Giảm giá cực sốc", "offer", "ai"), ("Top 1 Sài Gòn", "label", "ai")],
              'A poster "Tiệm nail Hồng Nhung", a button "Đặt lịch ngay", a badge "Giảm giá cực sốc", "Top 1 Sài Gòn".', NAIL)
    assert by_text(B)["Đặt lịch ngay"]["source"] == "ai"
    assert "Giảm giá cực sốc" not in by_text(B) and "Top 1 Sài Gòn" not in by_text(B)
    assert "Giảm giá cực sốc" not in B["prompt_en"]


def test_support_copy_capped():
    ai = [(f"Dòng lợi ích chung {c}", "body", "ai") for c in "abcdefgh"]
    B = brief(ai, "A poster.", "poster tiệm bánh")
    assert sum(t["source"] == "ai" for t in B["texts"]) == 6


def test_promise_added_to_user_text_is_dropped():
    B = brief([("Giảm 30% tất cả dịch vụ sơn gel, tặng quà", "offer", "user")], "A poster.", NAIL)
    assert B["texts"] == [] or "tặng" not in B["texts"][0]["text"]


def test_user_quoted_string_always_present():
    B = brief([], "A summer poster.", 'thêm dòng "Chỉ 3 ngày"')
    assert {"Chỉ 3 ngày"} <= set(by_text(B))
    assert '"Chỉ 3 ngày"' in B["prompt_en"]


def test_quoted_style_word_is_not_printed():
    B = brief([], "A Y2K-style t-shirt with a vintage aesthetic.", 'áo phong cách "Y2K"')
    assert "Y2K" not in by_text(B)


def test_steps_are_ordered_list_items():
    B = brief([("Bước một rửa tay", "step", "user")], 'A guide "Bước một rửa tay".', "hướng dẫn: bước một rửa tay")
    assert B["texts"][0]["role"] == "list_item" and B["texts"][0]["ordered"]


def test_star_rating_is_a_picture_not_text():
    B = brief([("5 sao", "label", "user")], 'A review card with "5 sao".', "ảnh feedback, hiển thị đánh giá 5 sao")
    assert B["texts"] == [] and "gold rating stars" in B["prompt_en"]


def test_llm_step_numbers_are_removed_not_treated_as_invented_facts():
    # brief_probe 15/10 accept2 l02: "1. Chụp ảnh sản phẩm lỗi" bị bỏ như số bịa, thêm lại không số -> lẫn lộn
    B = brief([("1. Chụp ảnh sản phẩm lỗi", "step", "user"), ("Bước 2: Gửi qua Zalo của shop", "step", "user")],
              'A guide "1. Chụp ảnh sản phẩm lỗi", "Bước 2: Gửi qua Zalo của shop".',
              "hướng dẫn đổi trả: chụp ảnh sản phẩm lỗi, gửi qua Zalo của shop")
    steps = [t["text"] for t in B["texts"] if t.get("ordered")]
    assert steps == ["Chụp ảnh sản phẩm lỗi", "Gửi qua Zalo của shop"]
    assert '"Chụp ảnh sản phẩm lỗi"' in B["prompt_en"] and "1." not in B["prompt_en"]


def test_link_written_in_quotes_is_the_users():
    B = brief([("www.ironfit.vn", "contact", "user")], 'A gym poster with "www.ironfit.vn".', 'thêm website "www.ironfit.vn"')
    assert "www.ironfit.vn" in by_text(B) and "facts_not_from_user" not in B["log"]


def test_inserted_texts_are_a_separate_sentence():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], "A pink salon poster with soft light", NAIL)
    assert 'soft light. The poster also shows the texts "Tiệm nail Hồng Nhung", "Khai trương ngày 5/11", '            '"Giảm 30% tất cả dịch vụ sơn gel".' in B["prompt_en"]


def test_llm_typo_in_prompt_fixed_in_place():
    # brief_probe 15/10 dev2 d22: texts "COMBO GIA ĐÌNH", prompt viết "COMBO GIA ĐỊNH"
    B = brief([("COMBO GIA ĐÌNH", "headline", "user")], 'A hotpot banner with "COMBO GIA ĐỊNH" on top. Warm light.',
              'banner "COMBO GIA ĐÌNH"')
    assert B["prompt_en"].startswith('A hotpot banner with "COMBO GIA ĐÌNH" on top.')
    assert "quoted_added" not in B["log"]


def test_plain_image_has_no_texts():
    B = brief([], "A cow in a field.", "ảnh một con bò", intent="image")
    assert B["mode"] == "image" and B["texts"] == []


# ---------------------------------------------------------------- hợp đồng của prompt

def test_every_text_quoted_exactly_once():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user"), ("Đặt lịch ngay", "cta", "ai")],
              'A poster "Tiệm nail Hồng Nhung". The sign repeats "Tiệm nail Hồng Nhung". Soft light.', NAIL)
    assert B["prompt_en"].count('"Tiệm nail Hồng Nhung"') == 1
    assert B["prompt_en"].count('"Đặt lịch ngay"') == 1


def test_missing_texts_inserted_after_first_sentence():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], "A pink salon poster. Soft light. Glossy nails.", NAIL)
    first, second = B["prompt_en"].split(". ")[:2]
    assert first == "A pink salon poster" and '"Tiệm nail Hồng Nhung"' in second


def test_single_quotes_and_curly_quotes_unified():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], "A poster 'Tiệm nail Hồng Nhung' with “Tiệm nail Hồng Nhung”.", NAIL)
    assert B["prompt_en"].count('"Tiệm nail Hồng Nhung"') == 1 and "'" not in B["prompt_en"]


def test_foreign_quote_removed_but_rest_of_sentence_kept():
    B = brief([("Bún bò Cô Ba", "headline", "user")], 'A rustic banner "Bún bò Cô Ba" and a tag "Giá 50.000đ", steam rising.',
              "banner quán bún bò cô ba")
    assert "50.000" not in B["prompt_en"] and '"Bún bò Cô Ba"' in B["prompt_en"] and "rustic" in B["prompt_en"]


def test_english_descriptor_in_quotes_is_unquoted():
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], 'A poster "Tiệm nail Hồng Nhung" in a "before/after" layout.', NAIL)
    assert "before/after layout" in B["prompt_en"] and '"before/after"' not in B["prompt_en"]


def test_ratio_removed_outside_quotes_only():
    B = brief([("Mở cửa 8:30", "info", "user")], 'A 9:16 poster, "Mở cửa 8:30", 4k.', "poster mở cửa 8:30")
    assert "9:16" not in B["prompt_en"] and "4k" not in B["prompt_en"] and '"Mở cửa 8:30"' in B["prompt_en"]
    assert ", ." not in B["prompt_en"]


def test_long_prompt_fits_token_limit_keeping_texts():
    long = 'A poster "Tiệm nail Hồng Nhung". ' + " ".join(
        f"Sentence {i} describes soft pink light, glossy textures, pastel flowers and elegant decor." for i in range(60))
    B = brief([("Tiệm nail Hồng Nhung", "headline", "user")], long, NAIL)
    assert count_tokens(B["prompt_en"]) <= TOKEN_LIMIT and '"Tiệm nail Hồng Nhung"' in B["prompt_en"]


def test_non_object_answer_raises():
    with pytest.raises(ValueError):
        make_brief(lambda s, p: ["not", "an", "object"], "poster")


def test_ordinal_is_not_a_time_or_a_number():
    # final3 h08: "6:00" / "17:30" bị coi là số thứ tự "6:" -> "00"; giá / khoảng cũng không phải số thứ tự
    from textfix.brief.texts import ORDINAL
    assert [ORDINAL.sub("", t) for t in ("Bước 1: Rửa tay", "1. Tráng phin", "2) Chọn món")] == ["Rửa tay", "Tráng phin", "Chọn món"]
    assert not any(ORDINAL.match(t) for t in ("6:00", "17:30", "1.500.000đ", "3 - 5 ngày"))


# ---------------------------------------------------------------- viết hoa, lời dặn (12/11: chỉ còn lời chat)

def test_support_copy_title_case_becomes_sentence_case():
    B = brief([("Quán Cà Phê", "headline", "ai"), ("Không Gian Yên Tĩnh", "body", "ai"), ("Ghé quán hôm nay", "cta", "ai")],
              'A café poster "Quán Cà Phê", "Không Gian Yên Tĩnh" and a button "Ghé quán hôm nay".', "poster quán cà phê")
    assert [t["text"] for t in B["texts"]] == ["Quán cà phê", "Không gian yên tĩnh", "Ghé quán hôm nay"]
    assert '"Quán cà phê"' in B["prompt_en"] and "Quán Cà Phê" not in B["prompt_en"]


def test_user_title_case_and_caps_are_kept():
    from textfix.brief.texts import sentence_case
    B = brief([("Tiệm Nail Hồng Nhung", "headline", "user")], 'A poster "Tiệm Nail Hồng Nhung".', "poster Tiệm Nail Hồng Nhung")
    assert by_text(B)["Tiệm Nail Hồng Nhung"]["source"] == "user"   # lời khách: giữ nguyên cách viết hoa
    assert sentence_case("ƯU ĐÃI HÔM NAY") == "ƯU ĐÃI HÔM NAY" and sentence_case("Ưu đãi") == "Ưu đãi"


def test_system_prompt_is_chat_only():
    import re

    from textfix.brief.instructions import SCHEMA, SYSTEM
    assert not re.search(r"\b(form (field|value)s?|the form|variants?|overridden)\b", SYSTEM, re.I)
    assert SCHEMA["required"] == ["intent", "texts", "prompt_en"]
    assert "NEVER name a font" in SYSTEM and "sentence case" in SYSTEM


def test_request_message_has_prompt_and_canvas():
    from textfix.brief.instructions import request_message
    m = request_message("poster cà phê", False, "9:16")
    assert m.startswith("Prompt:\nposter cà phê") and "tall vertical canvas" in m and "variants" not in m


# ---------------------------------------------------------------------------------------------- câu khách bị cắt cụt (11/11)
def _cut(prompt, *texts):
    from textfix.brief.texts import TextPlan, complete_cut_texts
    plan = TextPlan(texts=[{"text": t, "role": "body", "source": "user"} for t in texts])
    complete_cut_texts(plan, prompt)
    return [t["text"] for t in plan.texts], plan


def test_text_cut_at_a_connector_is_completed_to_the_clause_end():
    got, plan = _cut("đăng tuyển 2 bạn phục vụ part-time cho quán cơm, lương 25k/giờ, liên hệ 0909 123 456",
                     "Tuyển 2 bạn phục vụ part-time cho", "Lương 25k/giờ")
    assert got == ["Tuyển 2 bạn phục vụ part-time cho quán cơm", "Lương 25k/giờ"]
    assert plan.rewrites == [('"Tuyển 2 bạn phục vụ part-time cho"', '"Tuyển 2 bạn phục vụ part-time cho quán cơm"')]


@pytest.mark.parametrize("prompt, text", [
    ("làm poster cho tiệm nail Hồng Nhung, khai trương ngày 5/11", "Tiệm nail Hồng Nhung"),   # chữ đệm "làm poster cho" bỏ đúng
    ("Khai trương ngày 5/11, giảm 30% tất cả dịch vụ sơn gel", "Giảm 30% tất cả dịch vụ sơn gel"),
    ("poster cho quán", "Poster cho quán"),                                                   # không dừng ở chữ nối
    ("ưu đãi cho, khách mới", "Ưu đãi cho"),                                                  # khách tự ngắt vế ở đó
    ("tuyen 2 ban phuc vu cho quan com", "Tuyển 2 bạn phục vụ cho"),                          # khách gõ không dấu: không nối chữ không dấu
])
def test_texts_that_are_not_cut_stay(prompt, text):
    assert _cut(prompt, text)[0] == [text]


def test_completion_skips_when_the_rest_is_another_text():
    assert _cut("bánh mì cho bé, quán cơm ngon", "Bánh mì cho", "bé")[0] == ["Bánh mì cho", "bé"]


def test_number_the_llm_dropped_comes_back_with_its_label():
    # 11/11 sme1 s01 / s08: LLM bỏ giá khách đưa
    B = brief([("Bánh Mì Thịt", "headline", "user")], 'A poster "Bánh Mì Thịt".', "lam poster ban banh mi thit 15k")
    assert "15k" in by_text(B)
    B = brief([("Cho thuê phòng trọ", "headline", "user"), ("Liên hệ anh Bình 0933 222 111", "contact", "user")],
              'A poster "Cho thuê phòng trọ", "Liên hệ anh Bình 0933 222 111".',
              "cho thuê phòng trọ, giá 3tr5/tháng, điện 3.5k/số. LH anh Bình 0933 222 111")
    assert {"Giá 3tr5/tháng", "Điện 3.5k/số"} <= set(by_text(B))
    assert '"Giá 3tr5/tháng"' in B["prompt_en"]


def test_design_numbers_and_user_numbering_are_not_texts():
    B = brief([("Quán phở Hà", "headline", "user")], 'A poster "Quán phở Hà".', "quán phở Hà, khổ 16:9, chữ 40px, đánh giá 5 sao")
    assert list(by_text(B)) == ["Quán phở Hà"]
    B = brief([("1. Tắm tráng trước khi xuống hồ", "list_item", "user"), ("2. Mặc đồ bơi", "list_item", "user")],
              'A board "1. Tắm tráng trước khi xuống hồ", "2. Mặc đồ bơi".', "nội quy: 1. Tắm tráng trước khi xuống hồ. 2. Mặc đồ bơi")
    assert list(by_text(B)) == ["1. Tắm tráng trước khi xuống hồ", "2. Mặc đồ bơi"]   # khách tự đánh số: giữ
