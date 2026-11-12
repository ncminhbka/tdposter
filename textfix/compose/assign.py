"""BƯỚC 4c -- GHÉP: câu khách <-> dòng nháp thành ĐƠN VỊ (một câu nhiều dòng, một dòng nhiều câu, câu ghép hai lần,
thứ tự các bước, số thứ tự đầu hàng). Chữ luôn là nguyên văn câu khách.
"""

from __future__ import annotations

import re
from collections import Counter

from ..lines.fold import _flat
from . import rows
from .geometry import SPLIT_RATIO, _reading, _stacked

SPLIT_OCR = 1.5     # câu gán cho các dòng xếp chồng: tổng chữ OCR / độ dài câu <= 1,5 = các dòng là mẩu của câu (chia dòng 0,7-1,38;
                    # FLUX vẽ lặp cả câu 1,68-2,0 trên mọi tập đã lưu 12/11) -> gộp dù lệch cỡ
PIECE_SIM = 0.8    # các mẩu chữ VLM nối lại giống câu khách >= 0,8 = một câu chia nhiều chỗ (VLM viết lệch dấu: "TRƢƠNG", i03)


FILL_SIM = 0.8      # chữ lấp VLM viết giống một câu khách >= 0,8 (bỏ dấu, chữ thường): là câu đó, không phải chữ lấp


OCR_SAYS = 0.5      # chữ OCR một chỗ giống câu khác >= 0,5 (và hơn câu VLM ghép): VLM ghép nhầm chỗ (_misread)


INSIDE = 0.8        # dòng nằm >= 80% trong khung một dòng khác lớn hơn = mẩu của dòng đó (mẩu logo, OCR đọc trùng)


def resolve(A: dict, M: dict, texts: list[dict], gone: list[str] = ()) -> tuple[list[dict], list[str], list[int]]:
    """Đơn vị dựng từ phần ghép của VLM; số phận dòng theo compose.fate. -> (đơn vị, nhật ký, câu khách không có trên poster).
    gone: câu của poster cũ đã bỏ / đã đổi (sửa poster) -- chữ lấp trùng một câu này là FLUX chép lại poster cũ, không viết."""
    by, log = M["by"], []
    G = rows.lists(M)   # danh sách / bảng nháp vẽ (hình học): câu ghép hai lần về đúng hàng, không gộp ô ở hai hàng
    rof = rows.row_of(G)
    from difflib import SequenceMatcher
    sim = lambda a, b: SequenceMatcher(None, _flat(a), _flat(b)).ratio()  # noqa: E731
    U = []
    for u in A.get("assign") or []:
        u = {**u, "marks": [m for m in u["marks"] if m in by], "clients": [k for k in u["clients"] if 0 <= k < len(texts)]}
        if u["marks"]:
            U.append(u)
    # MỘT DÒNG MỘT ĐƠN VỊ: viết > giữ > xoá; ngang nhau: đơn vị đứng trước (VLM ghi thêm keep cho dòng đã có chữ -> hai lớp chữ)
    rank = lambda u: 0 if u["kind"] == "text" and (u["clients"] or u["text_vlm"]) else 1 if u["kind"] == "keep" else 2  # noqa: E731
    owner = {}
    for i, u in sorted(enumerate(U), key=lambda t: (rank(t[1]), t[0])):
        for m in u["marks"]:
            owner.setdefault(m, i)
    for i, u in enumerate(U):
        lost = [m for m in u["marks"] if owner[m] != i]
        if lost:
            log.append(f"{u['id']}: dòng {lost} đã thuộc đơn vị khác -> bỏ khỏi đơn vị này")
            u["marks"] = [m for m in u["marks"] if owner[m] == i]
    U = [u for u in U if u["marks"]]
    # MỘT CÂU CHIA Ở NHIỀU ĐƠN VỊ (VLM ghi mỗi mẩu một mục: "Du lịch Mây Ngàn -" | "0283 777 8899", 14/10 mix1 m04): các mẩu nối
    # lại theo thứ tự đọc ra đúng câu -> gộp thành một đơn vị nhiều dòng, mỗi dòng một mẩu
    for k in {u["clients"][0] for u in U if len(u["clients"]) == 1 and u["split"]}:
        us = [u for u in U if u["kind"] == "text" and u["clients"] == [k] and u["split"] and len(u["split"]) == len(u["marks"])]
        if len(us) < 2:
            continue
        us = [us[i] for i in _reading([by[u["marks"][0]]["box"] for u in us])]
        if sim("".join(p for u in us for p in u["split"]), texts[k]["text"]) < PIECE_SIM:
            continue
        us[0].update(marks=[m for u in us for m in u["marks"]], split=[p for u in us for p in u["split"]], split_ok=True)
        U = [u for u in U if u not in us[1:]]
        log.append(f"{us[0]['id']}: câu T{k} chia ở {[u['id'] for u in us]} -> một đơn vị {us[0]['marks']}")
    # MỌI MẨU ĐỀU LÀ NGUYÊN CÂU (VLM: câu này vẽ ở hai chỗ, 11/11 edit1 e12 "Combo 4 người 459k" ở L3 và L6 cách nhau 460px): là
    # VIẾT LẶP, không phải một câu chia dòng -> mỗi dòng một đơn vị; khối "một câu ở nhiều đơn vị" dưới đây quyết gộp (xếp
    # chồng liền nhau) hay viết lặp theo nháp. Trước: nối mẩu không ra câu -> split bỏ -> code ngắt câu làm đôi qua hai chỗ.
    for u in list(U):
        sp, k = u.get("split"), (u["clients"] or [None])[0]
        if (u["kind"] == "text" and len(u["clients"]) == 1 and sp and len(sp) == len(u["marks"]) >= 2
                and all(_flat(p) == _flat(texts[k]["text"]) for p in sp)):
            i = next(j for j, x in enumerate(U) if x is u)
            U[i:i + 1] = [{**u, "id": u["id"] if j == 0 else f"{u['id']}r{j}", "marks": [m], "split": None}
                          for j, m in enumerate(u["marks"])]
            log.append(f"{u['id']}: mọi mẩu là nguyên câu T{k} -> viết lặp ở {u['marks']}, mỗi dòng một đơn vị")
    # SPLIT trong MỘT đơn vị: chỉ dùng mảnh rỗng (VLM nói dòng đó không thuộc câu -- 14/10 mix1 m01: dòng thứ ba là tên máy FLUX
    # tự vẽ) -> dòng rời đơn vị, xoá trống (VLM đã nhìn dòng và nói không có chữ khách nào ở đó). Ngắt câu vào các dòng là việc
    # của code (bề rộng nét, layout); split chỉ còn cho chữ cong và phải nối lại đúng câu khách.
    # ... TRỪ dòng để trống đứng CÙNG HÀNG, ngay cạnh một dòng có chữ của đơn vị, không chồng lên nó (final2 g18: VLM dồn "Giảm 50%"
    # vào ô "50%", để trống ô "Giảm" bên cạnh -> chữ co còn 0,32): đó là một phần của hàng, giữ lại, code ngắt câu theo sức chứa.
    # Dòng để trống CHỒNG lên dòng khác (OCR đọc trùng một mẩu) vẫn rời đơn vị.
    for u in list(U):
        if u["split"] and len(u["split"]) == len(u["marks"]) and any(not p.strip() for p in u["split"]):
            full = [m for m, p in zip(u["marks"], u["split"]) if p.strip()]
            empty = [m for m, p in zip(u["marks"], u["split"]) if not p.strip()]
            stay = [m for m in empty if any(_beside(by[m]["box"], by[f]["box"]) for f in full)
                    and not any(_cover_min(by[m]["box"], by[o]["box"]) > 0.5 for o in u["marks"] if o != m)]
            out = [m for m in empty if m not in stay]
            if stay:
                u["split"], u["split_ok"] = None, False
                log.append(f"{u['id']}: split để trống dòng {stay} cùng hàng cạnh dòng có chữ -> giữ trong đơn vị, code ngắt câu")
            else:
                u["split"] = [p for p in u["split"] if p.strip()]
            u["marks"] = [m for m in u["marks"] if m not in out]
            if out:
                U.append({**u, "id": u["id"] + "x", "kind": "skip", "marks": out, "clients": [], "split": None, "text_vlm": "", "vlm_out": True})
                log.append(f"{u['id']}: split để trống dòng {out} -> dòng đó rời đơn vị, xoá trống")
        # MẨU VLM CHIA nối lại đúng câu (xếp theo chỗ của mẩu trong câu, không theo thứ tự đọc: final3 h13 "Giảm đến" nhỏ nằm trên
        # "70%" to bị coi là cùng hàng, "70%" đứng trước) -> mỗi dòng đúng mẩu VLM đọc (split_ok); không nối lại đúng -> code ngắt
        if u["split"] and not u.get("split_ok"):
            if len(u["clients"]) == 1 and len(u["split"]) == len(u["marks"]):
                u["marks"], u["split"] = _text_order(u["marks"], u["split"], texts[u["clients"][0]]["text"])
            if len(u["clients"]) != 1 or _flat("".join(u["split"])) != _flat(texts[u["clients"][0]]["text"]):
                u["split"] = None
            elif len(u["split"]) == len(u["marks"]):
                u["split_ok"] = True
    U = [u for u in U if u["marks"]]
    # MỘT CÂU Ở NHIỀU ĐƠN VỊ: đơn vị có chữ OCR giống câu nhất nhận câu; đơn vị kia: còn câu CÙNG VAI chưa có chỗ thì nhận câu đó
    # (FLUX vẽ lặp một hàng: 5f35 "Gội dưỡng" hai lần, "Đính đá" chưa có chỗ), không thì thành dòng lấp (chữ VLM đoán; không có
    # thì viết lặp đúng câu đó theo nháp)
    used = Counter(k for u in U for k in u["clients"])
    ocr = lambda u: " ".join(by[m].get("ocr") or "" for m in u["marks"])  # noqa: E731
    for k, n in list(used.items()):
        if n < 2:
            continue
        us = [u for u in U if k in u["clients"]]
        # các chỗ của câu XẾP CHỒNG liền nhau, cùng cỡ (FLUX vẽ một câu thành hai dòng, VLM ghi mỗi dòng một mục -- 14/10 dev2
        # d31: dòng dưới giữ chữ giun ngay dưới câu đã viết) -> một đơn vị nhiều dòng
        # (một hàng list FLUX vẽ lặp mà còn câu cùng vai chưa có chỗ thì không gộp: hàng lặp nhận câu đó, như dưới)
        if all(u["clients"] == [k] and u["kind"] == "text" for u in us) and not any(
                not used[j] and texts[j]["role"] == texts[k]["role"] for j in range(len(texts))):
            Ls = sorted((by[m] for u in us for m in u["marks"]), key=lambda l: (l["box"][1], l["box"][0]))
            # ... không gộp hai ô ở HAI HÀNG ĐỦ Ô của một bảng (giá ở hai hàng giá, compose.rows.across_table_rows)
            # LỆCH CỠ vẫn gộp khi các dòng chỉ là MẨU của câu (12/11 chat1 c10: "Khai trương / shop quần áo / Linh Chi", dòng cuối to
            # 1,46 lần -> trước bị viết lặp cả câu 3 lần): tổng độ dài chữ OCR các dòng ~ độ dài câu (chia dòng 0,7-1,38 trên mọi tập
            # đã lưu) chứ không ~ số dòng x câu (FLUX vẽ lặp: 1,68-2,0). Dựng "lines": mỗi dòng giữ cỡ nháp, code chia chữ theo OCR
            sizes = [l["size"] for l in Ls]
            pieces = sum(len((l.get("ocr") or "").strip()) for l in Ls) / max(1, len(texts[k]["text"])) <= SPLIT_OCR
            if (_stacked(Ls) and (max(sizes) / min(sizes) <= SPLIT_RATIO or pieces)
                    and not rows.across_table_rows(G, [l["id"] for l in Ls])):
                us[0].update(marks=[l["id"] for l in Ls], split=None)
                U = [u for u in U if not any(u is x for x in us[1:])]
                used[k] = 1
                log.append(f"câu T{k}: {[u['id'] for u in us]} xếp chồng liền nhau -> một đơn vị {us[0]['marks']}")
                continue
        # chỗ đúng của câu: hàng bảng có câu liền kề (k-1 / k+1: tên món của giá) > chữ VLM giống câu > chữ OCR giống câu
        # ... nhưng chỗ mà OCR đọc ra hẳn một câu KHÁC thì thua (golden r10: VLM ghép lệch cả bảng, ô "15k" nhận câu nút "Tận
        # hưởng ngay" -- hàng bảng thắng thì nút bị gỡ trống)
        best = max(us, key=lambda u: (not _misread(u, k, M, texts), _paired(u, k, U, rof),
                                      sim(u["text_vlm"], texts[k]["text"]), sim(ocr(u), texts[k]["text"])))
        for u in us:
            if u is best:
                continue
            free = [j for j in range(len(texts)) if not used[j] and texts[j]["role"] == texts[k]["role"]]
            u["clients"] = [j for j in u["clients"] if j != k]
            used[k] -= 1
            if free and not u["clients"]:
                u["clients"] = [free[0]]
                used[free[0]] += 1
                log.append(f"{u['id']}: câu T{k} đã ở {best['id']} -> nhận câu cùng vai chưa có chỗ T{free[0]}")
            else:
                u["role_vlm"] = u["role_vlm"] or texts[k]["role"]
                # FLUX vẽ lặp câu mà VLM không đoán chữ khác -> gỡ trống (11/11, thay luật 16/10 "viết lặp theo nháp": hard1 + sme1
                # 5 / 48 poster mang một câu hai lần -- người dùng duyệt). TRỪ NÚT BẤM (vai cta) ở chỗ OCR không đọc ra câu khác:
                # nút trống trông như poster hỏng (final3 h21) -> vẫn viết lặp
                if not u["clients"] and not u["text_vlm"] and texts[k]["role"] == "cta" and not _misread(u, k, M, texts):
                    u["text_vlm"] = texts[k]["text"]
                    log.append(f"{u['id']}: nút T{k} đã ở {best['id']} -> FLUX vẽ hai nút, viết lặp theo nháp")
                else:
                    log.append(f"{u['id']}: câu T{k} đã ở {best['id']} -> " + (f"dòng lấp {u['text_vlm']!r}" if u["text_vlm"]
                                                                              else "FLUX vẽ lặp, gỡ trống"))
    # DÒNG LẤP (VLM đoán FLUX định viết gì ở dòng vẽ thừa, kể cả hàng danh sách / bảng -- người dùng chốt 16/10: FLUX bịa thì
    # viết theo): compose.assign.filler -- là một câu khách chưa có chỗ thì viết câu đó; còn lại viết đúng chữ VLM đoán
    used = Counter(k for u in U for k in u["clients"])
    for u in U:
        if u["kind"] == "text" and not u["clients"] and u["text_vlm"]:
            k = filler(u["text_vlm"], texts, used)
            if k is not None:
                u["clients"], used[k] = [k], used[k] + 1
                log.append(f"{u['id']}: chữ lấp {u['text_vlm']!r} là câu khách chưa có chỗ T{k} -> viết câu đó")
    log += _echo_or_made_up(U, M, texts)
    log += _drop_gone(U, gone)
    log += _restored(U, M, texts)
    # đơn vị "text" không còn chữ (câu đã có chỗ khác, chữ lấp bị bỏ): dòng của nó về luật 4 của compose.fate (skip)
    U = [u for u in U if not (u["kind"] == "text" and not u["clients"] and not u["text_vlm"])]
    log += inside(U, M)
    from . import fate
    extra, flog = fate.leftovers(U, M, fate.mentioned(A))
    U += extra
    log += flog
    log += order(U, M, texts)
    log += _markers(U, M, texts)
    used = Counter(k for u in U for k in u["clients"])
    missing = [j for j in range(len(texts)) if not used[j] and texts[j].get("source") != "ai"]
    return U, log, missing


def _echo_or_made_up(U: list[dict], M: dict, texts: list[dict]) -> list[str]:
    """Chữ lấp VLM đoán cho dòng FLUX vẽ thừa (11/11, người dùng duyệt sau hard1 + sme1 trên prod):
      - LẶP: chính là / là mẩu của một câu khách đã có chỗ ("Lẩu Nấm Rừng" dưới "Nhà hàng Lẩu Nấm Rừng", "60 phút") -> gỡ trống
        (trừ hai nút bấm cùng chữ: viết lặp).
        Chính là câu đó mà chỗ của câu OCR đọc ra một câu khác (VLM ghép nhầm chỗ, chỗ thật thành dòng lấp) -> câu về dòng lấp,
        chỗ nhầm gỡ trống;
      - BỊA SỐ: mang số không câu khách nào có (sme1 s05: bảng giờ FLUX vẽ, VLM điền "Tất 8:00-7:30") -> gỡ trống.
    Chữ lấp không số, không lặp (một lợi ích nữa, tên món FLUX vẽ thêm) vẫn viết theo nháp (luật 16/10)."""
    from difflib import SequenceMatcher
    from ..brief.evidence import Evidence
    log = []
    facts = Evidence([t["text"] for t in texts])
    for u in U:
        f = _flat(u.get("text_vlm") or "")
        if u["kind"] != "text" or u["clients"] or not f:
            continue
        held = [(k, v) for v in U if v is not u and v["kind"] == "text" for k in v["clients"]]
        same = next(((k, v) for k, v in held if SequenceMatcher(None, f, _flat(texts[k]["text"])).ratio() >= FILL_SIM), None)
        if same and len(same[1]["clients"]) == 1 and _misread(same[1], same[0], M, texts) and not _misread(u, same[0], M, texts):
            k, v = same
            u.update(clients=[k], text_vlm="", split=None)
            v.update(clients=[], text_vlm="", split=None)
            log.append(f"{u['id']}: chữ lấp là câu T{k}, OCR ở {v['id']} đọc ra câu khác -> câu về {u['id']}, {v['id']} gỡ trống")
        elif same and texts[same[0]]["role"] == "cta" and not _misread(u, same[0], M, texts):
            continue   # hai nút cùng chữ: viết lặp (nút trống trông như poster hỏng)
        elif same or any(f in _flat(texts[k]["text"]) for k, _ in held):
            log.append(f"{u['id']}: chữ lấp {u['text_vlm']!r} lặp một câu khách đã có chỗ -> gỡ trống")
            u["text_vlm"] = ""
        elif facts.unknown_facts(u["text_vlm"]):
            log.append(f"{u['id']}: chữ lấp {u['text_vlm']!r} mang số khách không đưa -> gỡ trống")
            u["text_vlm"] = ""
    return log


def _misread(u: dict, k: int, M: dict, texts: list[dict]) -> bool:
    """Chữ OCR các dòng của u đọc ra một câu khách KHÁC k (>= OCR_SAYS, giống hơn k): VLM ghép câu k vào nhầm chỗ."""
    from difflib import SequenceMatcher
    o = _flat(" ".join(M["by"][m].get("ocr") or "" for m in u["marks"]))
    if not o:
        return False
    r = [SequenceMatcher(None, o, _flat(t["text"])).ratio() for t in texts]
    return any(j != k and r[j] >= OCR_SAYS and r[j] > r[k] for j in range(len(texts)))


def _drop_gone(U: list[dict], gone: list[str]) -> list[str]:
    """SỬA POSTER: FLUX sửa từ poster cũ nên hay vẽ lại cả dòng người dùng vừa bỏ / vừa đổi; luật "FLUX vẽ thừa thì viết theo"
    (16/10) viết lại đúng câu cũ (11/11 prod: e12 dòng ngày đã bỏ, f12 "Khai trương mùa Noel" đã bỏ vẫn hiện). Chữ lấp còn lại
    (không là câu nào của danh sách mới) giống một câu cũ đã đi -> dòng đó gỡ trống."""
    from difflib import SequenceMatcher
    log = []
    gf = [_flat(g) for g in gone if _flat(g)]
    for u in U:
        f = _flat(u.get("text_vlm") or "")
        if u["kind"] != "text" or u["clients"] or not f or not gf:
            continue
        if any(SequenceMatcher(None, f, g).ratio() >= FILL_SIM or (len(f) >= 4 and f in g) for g in gf):
            log.append(f"{u['id']}: chữ lấp {u['text_vlm']!r} là câu người dùng đã bỏ / đã đổi -> gỡ trống")
            u["text_vlm"] = ""
    return log


def _restored(U: list[dict], M: dict, texts: list[dict]) -> list[str]:
    """SỬA POSTER: dòng trả về chỗ cũ của câu Tk (edit/slots.py: nháp sửa bỏ trống chỗ đó) mà Tk chưa có chỗ nào -> Tk ở dòng đó,
    VLM nói gì cũng vậy (11/11 prod e03: FLUX xoá cả khối hotline / web, VLM bỏ dòng trả về vì "chỉ là nền" -> mất chữ khách)."""
    log = []
    used = Counter(k for u in U for k in u["clients"])
    for l in M["L"]:
        r = str(l.get("restored") or "")
        k = int(r[1:]) if r[1:].isdigit() else -1
        if not 0 <= k < len(texts) or used[k]:
            continue
        u = next((u for u in U if l["id"] in u["marks"]), None)
        if u is not None and len(u["marks"]) > 1:
            u["marks"] = [m for m in u["marks"] if m != l["id"]]
            u = None
        if u is None:
            u = {"id": f"r_{l['id']}", "marks": [l["id"]], "why": "chỗ cũ"}
            U.append(u)
        u.update(kind="text", clients=[k], split=None, text_vlm="", role_vlm=None)
        used[k] += 1
        log.append(f"{l['id']}: câu T{k} không có chỗ nào khác -> đặt vào chỗ cũ (dòng trả về)")
    return log


def filler(text: str, texts: list[dict], used: Counter) -> int | None:
    """Chữ lấp VLM đoán cho một dòng FLUX vẽ thừa -> câu khách k khi chữ lấp chính là câu khách chưa có chỗ, không thì None.
    (Chữ lấp lặp câu đã có / mang số bịa: _echo_or_made_up gỡ.)"""
    from difflib import SequenceMatcher
    f = _flat(text)
    near = [j for j, t in enumerate(texts) if not used[j] and f and (SequenceMatcher(None, f, _flat(t["text"])).ratio() >= FILL_SIM
                                                                     or (len(f) >= 4 and f in _flat(t["text"])))]
    return near[0] if near else None


def _paired(u: dict, k: int, U: list[dict], rof: dict) -> bool:
    """Đơn vị u nằm ở hàng bảng có dòng viết câu liền kề k-1 / k+1 (tên món của giá k, giá của tên k)."""
    cells = {rof[m] for m in u["marks"] if m in rof}
    return any(rof.get(m) in cells and set(v["clients"]) & {k - 1, k + 1} for v in U if v is not u for m in v["marks"])


def inside(U: list[dict], M: dict) -> list[str]:
    """MẨU CỦA DÒNG KHÁC: dòng không mang câu khách nằm >= INSIDE trong khung một dòng thẳng lớn hơn (mẩu logo Hi-SAM tách ra
    thành cụm nét riêng, OCR đọc trùng một mẩu) theo số phận dòng đó: dòng kia giữ (chữ trên sản phẩm, logo) -> giữ; không ->
    gỡ (chữ của dòng kia viết lại ở khung của nó). -> nhật ký."""
    log = []
    owner = {m: u for u in U for m in u["marks"]}
    area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))  # noqa: E731
    for l in M["L"]:
        m, u = l["id"], owner.get(l["id"])
        if u is not None and u["kind"] == "text" and u["clients"]:
            continue
        b = l["box"]
        par = [p for p in M["L"] if p is not l and not p.get("path") and area(p["box"]) > area(b)
               and _cover_min(b, p["box"]) * min(area(b), area(p["box"])) >= INSIDE * area(b)]
        if not par:
            continue
        p = max(par, key=lambda p: area(p["box"]))
        pu = owner.get(p["id"])
        kind = "keep" if pu is not None and pu["kind"] == "keep" else "skip"
        if u is not None and u["kind"] == kind and len(u["marks"]) == 1:
            continue
        if u is not None:
            u["marks"] = [x for x in u["marks"] if x != m]
        nu = {"id": f"i_{m}", "kind": kind, "marks": [m], "clients": [], "split": None, "text_vlm": "", "role_vlm": None,
              "why": f"nằm trong dòng {p['id']}"}
        U.append(nu)
        owner[m] = nu
        log.append(f"{m}: nằm trong dòng {p['id']} -> {kind} theo dòng đó")
    U[:] = [u for u in U if u["marks"]]
    return log


def _text_order(marks: list, pieces: list, text: str) -> tuple[list, list]:
    """Cặp (dòng, mẩu) xếp theo chỗ của mẩu trong câu khách (bỏ dấu, chữ thường); có mẩu không nằm trong câu -> giữ nguyên."""
    f = _flat(text)
    at = [f.find(_flat(p)) for p in pieces]
    if any(a < 0 for a in at):
        return marks, pieces
    order = sorted(range(len(pieces)), key=lambda i: at[i])
    return [marks[i] for i in order], [pieces[i] for i in order]


def _beside(a: list, b: list) -> bool:
    """Hai khung cùng hàng (chồng dọc >= nửa khung thấp hơn) và sát nhau (khe ngang <= 1 cao khung cao hơn)."""
    h = max(a[3] - a[1], b[3] - b[1])
    return (min(a[3], b[3]) - max(a[1], b[1]) >= 0.5 * min(a[3] - a[1], b[3] - b[1])
            and max(a[0] - b[2], b[0] - a[2]) <= h)


def _cover_min(a: list, b: list) -> float:
    """Diện tích giao / diện tích khung nhỏ hơn."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h) / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _markers(U: list[dict], M: dict, texts: list[dict]) -> list[str]:
    """SỐ THỨ TỰ FLUX VẼ TRƯỚC CÁC HÀNG (số trong vòng tròn / ô vuông đứng riêng một dòng, 14/10 dev2 d20 / d19): chữ giun số
    bị xoá thì để lại vòng tròn trống, giữ thì số sai (1, 4, 2, 3, 4). -> viết lại số: dòng chỉ 1-2 chữ số, ngay bên trái một
    hàng chữ đã viết câu khách (cùng hàng, khe <= 3 cỡ). Số = thứ tự câu khách nếu là các bước (brief "ordered"), không thì thứ
    tự đọc các hàng có số. Không phải các dòng số đứng riêng khác (giá, ngày)."""
    by = M["by"]
    rows = [u for u in U if u["kind"] == "text" and u["clients"]]
    found = []
    for u, m in [(u, m) for u in U if not (u["kind"] == "text" and (u["clients"] or u["text_vlm"])) for m in u["marks"]]:
        l = by[m]
        if not re.fullmatch(r"\s*\d{1,2}\s*[.)]?\s*", l.get("ocr") or ""):
            continue
        b, s = l["box"], l["size"]
        cy = (b[1] + b[3]) / 2
        nxt = [r for r in rows for m in r["marks"][:1] if by[m]["box"][1] <= cy <= by[m]["box"][3]
               and 0 <= by[m]["box"][0] - b[2] <= 3 * max(s, by[m]["size"])]
        if nxt:
            found.append((u, m, nxt[0]))
    if len(found) < 2:   # một số lẻ loi không phải đánh số hàng
        return []
    ordered = sorted(j for j, t in enumerate(texts) if t.get("ordered"))
    found.sort(key=lambda p: (by[p[1]]["box"][1], by[p[1]]["box"][0]))
    out = []
    for i, (u, m, r) in enumerate(found):
        k = r["clients"][0]
        n = 1 + ordered.index(k) if k in ordered else i + 1
        u["marks"] = [x for x in u["marks"] if x != m]
        U.append({"id": f"n_{m}", "kind": "text", "marks": [m], "clients": [], "split": None, "text_vlm": str(n),
                  "role_vlm": "step_no", "why": "số thứ tự trước hàng"})
        out.append(f"{m}: số FLUX vẽ trước hàng {r['id']} -> viết lại '{n}'")
    U[:] = [u for u in U if u["marks"]]
    return out


NUMBERED = re.compile(r"\s*\d{1,2}\s*[.)]\s")   # câu khách tự đánh số: "1. ...", "2) ..."


def order(U: list[dict], M: dict, texts: list[dict]) -> list[str]:
    """CÁC BƯỚC theo thứ tự khách (người dùng chốt 14/10, kể cả lưới 2x2 và khi lệch hình minh hoạ FLUX vẽ): đơn vị viết đúng một
    câu "ordered" (brief: bước của poster hướng dẫn), xếp theo thứ tự đọc -> hoán CÂU giữa các đơn vị cho tăng dần. Khung, cỡ,
    kiểu của đơn vị giữ nguyên. Danh sách khác (ưu điểm, nút) để VLM gán theo ảnh.
    Câu khách TỰ ĐÁNH SỐ ("1. Tắm tráng ...", "2) ...") cũng vậy (12/11 hard1 h13: VLM gán câu 5 vào hàng thứ hai, poster đọc
    1, 5, 2, 3, 4, 6)."""
    by = M["by"]
    num = lambda k: bool(texts[k].get("ordered") or NUMBERED.match(texts[k]["text"]))   # noqa: E731
    us = [u for u in U if u["kind"] == "text" and len(u["clients"]) == 1 and num(u["clients"][0])]
    if len(us) < 2:
        return []
    box = lambda u: [min(by[m]["box"][0] for m in u["marks"]), min(by[m]["box"][1] for m in u["marks"]),  # noqa: E731
                     max(by[m]["box"][2] for m in u["marks"]), max(by[m]["box"][3] for m in u["marks"])]
    seq = [us[i] for i in _reading([box(u) for u in us])]
    cur = [u["clients"][0] for u in seq]
    if cur == sorted(cur):
        return []
    for u, k in zip(seq, sorted(cur)):
        if u["clients"][0] != k:
            u["clients"], u["split"] = [k], None
    return [f"các bước: thứ tự đọc {['T%d' % k for k in cur]} -> theo thứ tự khách {['T%d' % k for k in sorted(cur)]}"]
