"""Danh sách / bảng nháp vẽ (compose.rows): nhận hàng bằng hình học; câu ghép hai lần về đúng hàng; hàng FLUX vẽ thừa viết chữ
lấp VLM đoán (người dùng chốt 16/10)."""

from __future__ import annotations

from textfix.compose import rows
from textfix.compose.assign import resolve
from textfix.compose.parse import parse


def line(i: int, box: list, **kw) -> dict:
    return {"id": f"L{i}", "box": list(box), "poly": None, "ocr": f"chu {i}", "size": float(box[3] - box[1]), "color": "#000000",
            "angle": 0.0, **kw}


def page(*L, size=(800, 1000)) -> dict:
    return {"L": list(L), "size": list(size), "by": {l["id"]: l for l in L}}


def menu(n: int, y0: int = 100, pitch: int = 60) -> list[dict]:
    """n hàng tên món (lề trái x=100) | giá (lề trái x=500), cao 40."""
    out = []
    for r in range(n):
        y = y0 + r * pitch
        out += [line(2 * r + 1, [100, y, 100 + 120 + 30 * (r % 3), y + 40]), line(2 * r + 2, [500, y, 640, y + 40])]
    return out


def entry(marks, client=None, kind="text", text=None, role=None) -> dict:
    return {"lines": marks, "client": client, "kind": kind, "text": text, "role": role, "why": "test"}


def test_menu_columns_pair_into_table_rows():
    G = rows.lists(page(*menu(4)))
    assert len(G) == 1 and G[0]["cols"] == 2
    assert G[0]["rows"] == [["L1", "L2"], ["L3", "L4"], ["L5", "L6"], ["L7", "L8"]]


def test_two_stacked_lines_are_not_a_list():
    # tiêu đề + phụ đề cùng cỡ, cùng tâm: không phải danh sách (hàng thừa của danh sách mới gỡ cả hàng)
    assert rows.lists(page(line(1, [200, 100, 600, 140]), line(2, [220, 160, 580, 200]))) == []


def test_far_block_breaks_the_column():
    # ba hàng đều nhịp 60, dòng thứ tư cách 300: khối khác
    L = [line(1, [100, 100, 300, 140]), line(2, [100, 160, 300, 200]), line(3, [100, 220, 300, 260]), line(4, [100, 560, 300, 600])]
    G = rows.lists(page(*L))
    assert [g["rows"] for g in G] == [[["L1"], ["L2"], ["L3"]]]


TEXTS = [{"text": "Cơm sườn", "role": "label"}, {"text": "35.000đ", "role": "price"},
         {"text": "Canh khổ qua", "role": "label"}, {"text": "15.000đ", "role": "price"},
         {"text": "Trà đá", "role": "label"}, {"text": "3.000đ", "role": "price"}]


def test_extra_row_is_written_with_the_vlm_guess_but_no_made_up_price():
    # FLUX vẽ 4 hàng cho 3 món: VLM đoán hàng 3 là "Canh rau muống" 20.000đ -> tên món viết theo (FLUX bịa thì viết theo), giá
    # khách không đưa thì gỡ trống (11/11: không bịa số)
    M = page(*menu(4))
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T1"), entry(["L3"], "T2"), entry(["L4"], "T3"),
                          entry(["L5"], text="Canh rau muống", role="label"), entry(["L6"], text="20.000đ", role="price"),
                          entry(["L7"], "T4"), entry(["L8"], "T5")]})
    U, log, miss = resolve(A, M, TEXTS)
    by = {m: u for u in U for m in u["marks"]}
    assert all(by[f"L{i}"]["kind"] == "text" for i in range(1, 9) if i != 6) and not miss
    assert by["L5"]["text_vlm"] == "Canh rau muống" and by["L6"]["kind"] != "text"


def test_duplicate_price_goes_to_the_row_of_its_dish():
    # cơm tấm f1cf: VLM ghi "3.000đ" (T5) cho giá hàng Trà đá (L6) và cả giá hàng thừa (L8) -- hai hàng khác nhau không gộp
    # thành một khối; hàng có tên món liền kề (T4 Trà đá) nhận giá
    M = page(*menu(4))
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T1"), entry(["L3"], "T2"), entry(["L4"], "T3"),
                          entry(["L5"], "T4"), entry(["L6"], "T5"), entry(["L7"], text="Chè đậu xanh"), entry(["L8"], "T5")]})
    U, log, _ = resolve(A, M, TEXTS)
    by = {m: u for u in U for m in u["marks"]}
    assert by["L6"]["clients"] == [5] and by["L6"]["marks"] == ["L6"]
    assert by["L7"]["text_vlm"] == "Chè đậu xanh" and not by["L8"]["clients"]


def test_a_text_drawn_over_stacked_lines_is_one_block():
    # final3 h22: lời khen FLUX vẽ ba dòng xếp chồng (trông như cột danh sách), VLM ghi câu cho cả ba -> một khối, không phải ba câu
    M = page(line(1, [300, 100, 700, 134]), line(2, [310, 140, 690, 174]), line(3, [305, 180, 695, 214]))
    A = parse({"assign": [entry(["L1"], "T0"), entry(["L2"], "T0"), entry(["L3"], "T0")]})
    U, log, _ = resolve(A, M, [{"text": "Niềng răng 18 tháng, giờ tôi cười thoải mái trong mọi tấm ảnh", "role": "review_quote"}])
    assert [(u["marks"], u["clients"]) for u in U if u["kind"] == "text"] == [(["L1", "L2", "L3"], [0])]

