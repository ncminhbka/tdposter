"""Kho ảnh tham chiếu (textfix/kho.py, textfix/kho_sync.py) và chỗ nối vào brief: không cần mạng, kho đối tượng giả trong bộ nhớ."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from textfix import kho
from textfix.brief import make_brief
from textfix.kho_sync import sha1, safe_rel, sync, sync_from_env

E = [{"id": "bun-thang", "file": "img/bun-thang.jpg", "tags": ["bun thang"], "aliases": ["bún thang"], "source": "CC BY"},
     {"id": "pho-bo", "file": "img/pho-bo.jpg", "tags": ["beef pho"], "aliases": ["phở bò"], "source": "CC0"},
     {"id": "ao-dai", "file": "img/ao-dai.jpg", "tags": ["ao dai"], "aliases": ["áo dài"], "source": "CC BY-SA"}]


def _jpg(color=(200, 120, 40)) -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Thư mục kho có đủ ảnh cho E, đặt làm kho_dir()."""
    (tmp_path / "img").mkdir()
    for e in E:
        (tmp_path / e["file"]).write_bytes(_jpg())
    (tmp_path / "index.json").write_text(json.dumps({"entries": E}), encoding="utf-8")
    monkeypatch.setenv("TEXTFIX_KHO_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- khớp đúng từ khoá

def test_match_accents_and_case_free(store):
    assert kho.match("Poster quán BUN THANG Hà Nội") == ["bun-thang"]
    assert kho.match("bún thang") == ["bun-thang"]


def test_match_whole_words_only(store):
    assert kho.match("bún thanh long") == []        # "bún thang" không nằm trong "bún thanh"
    assert kho.match("phở gà") == []                 # cụm chung "phở" không có trong kho: không kéo ảnh phở bò
    assert kho.match("ao dai") == ["ao-dai"]


def test_match_accented_words_need_exact_accents(store, tmp_path):
    tet = [{"id": "banh-tet", "file": "img/bun-thang.jpg", "tags": ["banh tet"], "aliases": ["bánh tét"], "source": "x"},
           {"id": "com", "file": "img/bun-thang.jpg", "tags": ["young green rice flakes"], "aliases": ["cốm"], "source": "x"}]
    assert kho.match("banner đặt bánh Tết", tet, store) == []      # bánh ngày Tết, không phải bánh tét
    assert kho.match("CƠM NHÀ MIỀN TÂY", tet, store) == []         # cơm không phải cốm
    assert kho.match("bánh tét nhân đậu", tet, store) == ["banh-tet"]
    assert kho.match("banh tet va com lang vong", tet, store) == ["banh-tet", "com"]   # gõ không dấu: so dạng bỏ dấu


COM = [{"id": "com", "file": "img/bun-thang.jpg", "tags": ["young green rice flakes"], "aliases": ["cốm làng vòng", "cốm"],
        "source": "x"}]


@pytest.mark.parametrize("text, want", [
    ("quán bún thang Hà Nội", [("bun-thang", True)]),        # cụm hai từ, đủ dấu: chắc
    ("quan BUN THANG Ha Noi", [("bun-thang", False)]),       # gõ không dấu: không chắc
    ("người mặc áo dài", [("ao-dai", True)]),
])
def test_match_detail_sure(store, text, want):
    assert [(h["id"], h["sure"]) for h in kho.match_detail(text)] == want


@pytest.mark.parametrize("text, want", [
    ("Poster trà sữa cốm giảm 20%", [("com", False, "cốm")]),           # alias một từ: LLM quyết
    ("cốm làng Vòng mùa thu", [("com", True, "cốm làng vòng")]),
    ("quan com tam Sai Gon", [("com", False, "com")]),                  # "com" không dấu khớp "cốm" bỏ dấu: không chắc
    ("cơm tấm sườn bì", []),
])
def test_match_detail_single_word_and_unaccented_are_unsure(store, text, want):
    assert [(h["id"], h["sure"], h["word"]) for h in kho.match_detail(text, COM, store)] == want


def test_match_follows_order_of_appearance(store):
    assert kho.match("người mặc áo dài ăn phở bò và bún thang") == ["ao-dai", "pho-bo", "bun-thang"]


def test_match_skips_entry_without_image(store):
    (store / "img" / "pho-bo.jpg").unlink()
    assert kho.match("phở bò") == []


def test_empty_or_broken_index_is_empty_kho(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXTFIX_KHO_DIR", str(tmp_path))
    assert kho.match("bún thang") == []
    (tmp_path / "index.json").write_text("{không phải json", encoding="utf-8")
    assert kho.load() == [] and kho.match("bún thang") == []


# ---------------------------------------------------------------- kiểm mục lục

def test_problems_clean(store):
    assert kho.problems(E, store) == []


def test_problems_found(store):
    bad = [*E, {"id": "pho-bo", "file": "img/x.jpg", "aliases": ["Bún Thang"], "source": ""}, {"file": "img/y.jpg"}]
    out = " | ".join(kho.problems(bad, store))
    for msg in ("trùng id", "không có file img/x.jpg", "thiếu source", "trỏ vào hai mục", "thiếu id"):
        assert msg in out
    one = [{"id": "com", "file": "img/bun-thang.jpg", "aliases": ["cốm"], "tags": ["com"], "source": "x"}]
    assert "chỉ có cụm một từ" in " | ".join(kho.problems(one, store))


# ---------------------------------------------------------------- số ảnh và ảnh vẽ

def test_numbered_product_is_image_1(store):
    ids = ["bun-thang", "pho-bo", "ao-dai"]
    assert [n for n, _ in kho.numbered(ids, product=False)] == [1, 2, 3]
    assert [(n, e["id"]) for n, e in kho.numbered(ids, product=True)] == [(2, "bun-thang"), (3, "pho-bo"), (4, "ao-dai")]


def test_numbered_cut_at_max_refs(store, monkeypatch):
    monkeypatch.setattr(kho, "MAX_REFS", 2)
    assert [e["id"] for _, e in kho.numbered(["bun-thang", "pho-bo", "ao-dai"], product=True)] == ["bun-thang"]


def test_refs_for_draw_product_first_then_kho(store):
    product = np.zeros((8, 8, 3), np.uint8)
    refs = kho.refs_for_draw({"kho": ["pho-bo", "ao-dai"]}, product)
    assert len(refs) == 3 and refs[0] is product and refs[1].shape == (24, 32, 3)
    assert kho.refs_for_draw({}, None) == []
    assert len(kho.refs_for_draw({"kho": ["bun-thang", "pho-bo", "ao-dai", "x"]}, product)) == kho.MAX_REFS


def test_refs_for_draw_skips_broken_image(store):
    (store / "img" / "pho-bo.jpg").write_bytes(b"hong")
    assert len(kho.refs_for_draw({"kho": ["pho-bo", "ao-dai"]}, None)) == 1


# ---------------------------------------------------------------- đồng bộ từ kho đối tượng

def remote(entries, files):
    """Kho đối tượng giả: {khoá: bytes}; ghi lại các khoá đã đọc."""
    objs = {"textfix-kho/index.json": json.dumps({"entries": entries}).encode()}
    objs.update({"textfix-kho/" + k: v for k, v in files.items()})
    calls = []

    def get(key):
        calls.append(key)
        if key not in objs:
            raise KeyError(key)
        return objs[key]
    return get, calls


def test_sync_downloads_then_skips_unchanged(tmp_path):
    img = _jpg()
    entries = [{**E[0], "sha1": sha1(img)}]
    get, calls = remote(entries, {"img/bun-thang.jpg": img})
    r = sync(get, "textfix-kho/", tmp_path)
    assert r == {"status": "ok", "entries": 1, "downloaded": 1, "dropped": []}
    assert kho.match("bún thang", root=tmp_path) == ["bun-thang"]
    calls.clear()
    assert sync(get, "textfix-kho", tmp_path)["downloaded"] == 0
    assert calls == ["textfix-kho/index.json"]


def test_sync_redownloads_changed_image(tmp_path):
    old, new = _jpg((0, 0, 0)), _jpg((255, 255, 255))
    sync(remote([{**E[0], "sha1": sha1(old)}], {"img/bun-thang.jpg": old})[0], "textfix-kho/", tmp_path)
    r = sync(remote([{**E[0], "sha1": sha1(new)}], {"img/bun-thang.jpg": new})[0], "textfix-kho/", tmp_path)
    assert r["downloaded"] == 1 and (tmp_path / "img/bun-thang.jpg").read_bytes() == new


def test_sync_drops_missing_bad_or_unsafe_entries(tmp_path):
    img = _jpg()
    entries = [{**E[0], "sha1": sha1(img)}, {**E[1], "sha1": "0" * 40}, E[2],
               {"id": "tron", "file": "../../etc/passwd", "aliases": ["trốn"], "source": "x"}]
    get, _ = remote(entries, {"img/bun-thang.jpg": img, "img/pho-bo.jpg": img})
    r = sync(get, "textfix-kho/", tmp_path)
    assert r["entries"] == 1 and sorted(r["dropped"]) == ["ao-dai", "pho-bo", "tron"]
    assert [e["id"] for e in kho.load(tmp_path)] == ["bun-thang"]


def test_sync_failure_keeps_old_cache(tmp_path):
    img = _jpg()
    sync(remote([E[0]], {"img/bun-thang.jpg": img})[0], "textfix-kho/", tmp_path)

    def down(key):
        raise ConnectionError("không tới được kho đối tượng")
    r = sync(down, "textfix-kho/", tmp_path)
    assert r["status"] == "error" and r["entries"] == 1
    assert kho.match("bún thang", root=tmp_path) == ["bun-thang"]


def test_sync_from_env_off_without_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXTFIX_KHO_DIR", str(tmp_path))
    assert sync_from_env({})["status"] == "off"


def test_sync_from_env_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXTFIX_KHO_DIR", str(tmp_path))
    r = sync_from_env({"TEXTFIX_KHO_ENDPOINT": "http://127.0.0.1:1", "TEXTFIX_KHO_BUCKET": "b"})
    assert r["status"] == "error" and r["entries"] == 0


def test_safe_rel():
    assert safe_rel("img/a.jpg")
    assert not any(safe_rel(p) for p in ("", "/etc/x", "../x", "img/../../x", "C:/x", "img\\x"))


# ---------------------------------------------------------------- nối vào brief

def llm_seen(prompt_en):
    seen = {}

    def llm(system, parts):
        seen["msg"] = parts[0]
        return {"intent": "image", "texts": [], "prompt_en": prompt_en, "variants": [], "overridden": []}
    return llm, seen


def test_brief_numbers_kho_after_product(store):
    llm, seen = llm_seen("A bowl of the bun thang from image 2 on a wooden table.")
    B = make_brief(llm, "ảnh tô bún thang", product_image=True)
    assert B["kho"] == ["bun-thang"]
    assert "reference image 2 shows an authentic bun thang" in seen["msg"]
    assert "refs_cited" not in B["log"] and B["log"]["kho"] == ["bun thang = image 2"]


def test_brief_cites_reference_llm_forgot(store):
    llm, _ = llm_seen("A steaming bowl of Vietnamese noodle soup. Soft morning light.")
    B = make_brief(llm, "poster quán Bún thang bà Hai")
    assert B["kho"] == ["bun-thang"]
    assert "the bun thang from image 1" in B["prompt_en"]
    assert B["prompt_en"].index("image 1") < B["prompt_en"].index("Soft morning")   # chèn sớm: FLUX đọc phần đầu


@pytest.fixture
def com_store(store):
    (store / "index.json").write_text(json.dumps({"entries": E + COM}), encoding="utf-8")
    return store


def test_brief_unsure_ref_dropped_when_llm_does_not_cite_it(com_store):
    llm, seen = llm_seen("A cup of green milk tea with tapioca pearls on a pastel table. Soft light.")
    B = make_brief(llm, "Poster trà sữa cốm giảm 20%")
    assert "MAY show an authentic young green rice flakes" in seen["msg"] and "'cốm'" in seen["msg"]
    assert B["kho"] == [] and "image" not in B["prompt_en"]
    assert B["log"]["kho_llm_dropped"] == ["young green rice flakes ('cốm')"] and "kho" not in B["log"]


def test_brief_unsure_ref_kept_when_llm_cites_it(com_store):
    llm, _ = llm_seen("A cup of milk tea topped with the young green rice flakes from image 1. Soft light.")
    B = make_brief(llm, "Poster trà sữa cốm giảm 20%")
    assert B["kho"] == ["com"] and B["log"]["kho_llm_kept"] == ["young green rice flakes ('cốm')"]


def test_brief_unaccented_com_tam_dropped(com_store):
    llm, _ = llm_seen("A plate of broken rice with grilled pork chop. Warm light.")
    B = make_brief(llm, "poster quan com tam Sai Gon")
    assert B["kho"] == [] and "image" not in B["prompt_en"]


def test_brief_renumbers_after_dropping(com_store):
    # ảnh sản phẩm = image 1; cốm (không chắc, đứng trước) = image 2, LLM không nhắc -> bỏ; bún thang (chắc) image 3 -> image 2
    llm, _ = llm_seen("The product from image 1 next to a bowl of the bun thang from image 3. Soft light.")
    B = make_brief(llm, "trà sữa cốm và bún thang", product_image=True)
    assert B["kho"] == ["bun-thang"]
    assert "the bun thang from image 2" in B["prompt_en"] and "image 3" not in B["prompt_en"]
    assert "The product from image 1" in B["prompt_en"] and B["log"]["kho"] == ["bun thang = image 2"]


def test_brief_without_kho_match_unchanged(store):
    llm, seen = llm_seen("A cat on a sofa.")
    B = make_brief(llm, "ảnh con mèo")
    assert B["kho"] == [] and "reference image" not in seen["msg"] and "kho" not in B["log"]
