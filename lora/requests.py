"""BƯỚC 1 -- LLM sinh yêu cầu cho bộ dữ liệu (JSONL), phủ đủ nguyên âm x thanh tiếng Việt.

  python -m lora.requests --n 800 --out data/lora/requests.jsonl

Mỗi dòng một việc cho lora.drive:
  {"id", "kind": "poster",  "prompt", "aspect"}                      kiểu 1: tạo mới
  {"id", "kind": "product", "prompt", "aspect", "product_prompt"}    kiểu 2: vẽ ảnh sản phẩm KHÔNG chữ trước, rồi poster cho nó
Lời sửa (kiểu 3) do lora.drive xin LLM sau khi có poster thật (cần biết chữ đang có trên poster).

~30% lô viết SƠ SÀI (--sparse-frac) như người dùng thật -> brief tự thêm câu "ai" đúng tỉ lệ prod.
Độ phủ: sau mỗi lô, đếm (nguyên âm, thanh) trong các câu chữ đã sinh (chỉ câu của khách; câu AI xem thống kê của lora.collect); lô sau xin LLM dùng các từ chứa tổ hợp còn hiếm
(RARE_WORDS + chữ HOA có dấu). LLM / khoá: như máy chủ (TEXTFIX_LLM_URL, TEXTFIX_MODEL; textfix.config.models).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

VOWELS = "aăâeêioôơuưy"
TONES = {"": "ngang", "́": "sắc", "̀": "huyền", "̉": "hỏi", "̃": "ngã", "̣": "nặng"}
RATIOS = ["1:1", "9:16", "16:9", "3:4", "4:3"]
RATIO_P = [0.35, 0.25, 0.15, 0.15, 0.10]
SECTORS = ["quán phở", "tiệm bánh mì", "quán cà phê", "trà sữa", "tiệm bánh ngọt", "quán lẩu", "quán nướng", "cơm văn phòng",
           "tiệm nail", "spa", "salon tóc", "phòng gym", "lớp yoga", "trung tâm tiếng Anh", "lớp vẽ thiếu nhi", "nhà thuốc",
           "phòng khám nha khoa", "shop quần áo", "shop giày", "mỹ phẩm", "điện thoại", "điện máy", "nội thất", "vật liệu xây dựng",
           "sửa xe máy", "rửa xe", "giặt ủi", "hoa tươi", "thú cưng", "siêu thị mini", "trái cây", "hải sản", "đặc sản vùng miền",
           "homestay", "tour du lịch", "tuyển dụng", "sự kiện khai trương", "lễ hội", "bất động sản", "gói cước internet",
           "khóa học online", "tiệm vàng", "đồ chơi trẻ em", "văn phòng phẩm", "bánh trung thu", "mứt Tết", "nước mắm", "trà",
           "bia", "nước giải khát"]
PRODUCTS = ["chai nước mắm thủy tinh", "hộp trà xanh giấy kraft", "lon nước ngọt", "chai dầu gội", "hũ mứt gừng", "túi cà phê rang",
            "hộp bánh trung thu", "chai sữa tắm", "lọ kem dưỡng da", "đôi giày thể thao", "chiếc túi xách da", "bình giữ nhiệt",
            "hộp cơm giấy", "ly trà sữa trân châu", "bó hoa hồng", "chiếc điện thoại", "tai nghe không dây", "nồi cơm điện",
            "chai mật ong", "gói mì", "chai bia", "hộp sữa tươi", "túi gạo", "chai nước suối", "hũ kim chi", "hộp bánh quy",
            "son môi", "lọ nước hoa", "đồng hồ đeo tay", "chiếc ba lô"]
STYLES = ["tối giản hiện đại", "rực rỡ vui tươi", "sang trọng tông vàng đen", "pastel nhẹ nhàng", "truyền thống Tết đỏ vàng",
          "trẻ trung năng động", "ấm cúng mộc mạc", "công nghệ tông xanh", "thiên nhiên xanh lá", "cổ điển retro"]
# từ có tổ hợp nguyên âm x thanh hiếm trên poster (ngã / hỏi / nặng trên ơ ư ă â ê ô y)
RARE_WORDS = ["ngẫu hứng", "kỹ thuật", "mỹ phẩm", "sữa tươi", "bữa trưa", "lẩu thái", "bưởi da xanh", "quẩy", "vẽ tranh", "ẵm bé",
              "nghỉ lễ", "tưởng thưởng", "phở bò tái", "trứng muối", "nướng mọi", "đặc sản", "giảm giá sốc", "miễn phí", "ưu đãi",
              "khuyến mãi", "rộn ràng", "hội ngộ", "thưởng thức", "sẵn hàng", "mỗi ngày", "chữa lành", "dưỡng ẩm", "tẩy trang",
              "nhẫn cưới", "đổi trả", "bảo hành", "lỡ hẹn", "vỡ òa", "dỗ dành", "kỷ niệm", "bỗng dưng", "rực rỡ", "mãi mãi",
              "ngỡ ngàng", "nữa", "cũng", "đẫm", "lững thững", "bẽn lẽn", "nỗi nhớ", "giữ ấm", "hãy", "những", "ngủ ngon", "uống"]

SYSTEM = ("Bạn viết yêu cầu mà chủ cửa hàng nhỏ ở Việt Nam gõ cho trợ lý thiết kế poster. Mỗi yêu cầu: tiếng Việt có dấu đầy đủ, "
          "tự nhiên như tin nhắn chat, nêu NGUYÊN VĂN các dòng chữ muốn in (tên quán, khẩu hiệu, giá, giờ, địa chỉ, hotline...). "
          "Đa dạng: số dòng chữ 1-6, câu ngắn lẫn câu dài, có yêu cầu viết HOA tiêu đề, có giá tiền / số điện thoại / ngày giờ. "
          "Không dùng emoji, không dùng từ tiếng Anh trừ khi tự nhiên (sale, combo, freeship).")
SPARSE = ("Viết như người dùng lười gõ: chỉ nêu tên cửa hàng / sản phẩm và MỘT ý (giảm giá, khai trương, món mới...), không liệt "
          "kê dòng chữ, không mô tả bố cục. Ví dụ: \"làm poster cho tiệm bánh mì Cô Ba, mới khai trương\".")
# MẬT ĐỘ CHỮ (người dùng 19/11: yêu cầu thật thường nhiều chữ, có câu dài; ảnh nhiều chữ = nhiều âm tiết mỗi mẫu học)
DENSITY = {"ít": "1-3 dòng chữ ngắn",
           "vừa": "4-6 dòng chữ",
           "dày": ("7-10 dòng chữ theo khuôn quảng cáo đầy đủ: tiêu đề, tên thương hiệu / sản phẩm, một câu khách hàng nói (15-30 chữ, "
                   "có con số như \"98% khách hàng...\"), một câu liệt kê tính năng / thành phần (15-30 chữ), một câu ưu đãi có giá / "
                   "phần trăm / thời hạn, cộng thêm hotline hoặc địa chỉ")}
DENSITY_P = {"ít": 0.35, "vừa": 0.40, "dày": 0.25}
LETTERING = (" Mỗi yêu cầu nêu thêm mong muốn về chữ cho một hai dòng: kiểu chữ (viết tay, thư pháp, chữ đậm hiện đại, chữ có chân "
             "cổ điển...), cỡ (thật to, nhỏ gọn) hoặc chỗ đặt (góc trên bên trái, giữa, dưới cùng).")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["prompt"],
              "properties": {"prompt": {"type": "string"}}}}}}


def tone_of(ch: str) -> tuple[str, str] | None:
    """Ký tự -> (nguyên âm gốc thường, tên thanh); không phải nguyên âm tiếng Việt -> None. 'Ặ' -> ('ă', 'nặng')."""
    d = unicodedata.normalize("NFD", ch.lower())
    marks = [c for c in d[1:] if c in TONES]
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES))
    if base not in VOWELS:
        return None
    return base, TONES[marks[0] if marks else ""]


def coverage(texts: list[str]) -> Counter:
    """Đếm (nguyên âm, thanh) trong các câu; khoá ('đ', '') đếm chữ đ."""
    c = Counter()
    for s in texts:
        for ch in unicodedata.normalize("NFC", s):
            if ch.lower() == "đ":
                c[("đ", "")] += 1
            k = tone_of(ch)
            if k:
                c[k] += 1
    return c


def rare(cov: Counter, k: int = 8) -> list[str]:
    """Các từ RARE_WORDS chứa tổ hợp (nguyên âm, thanh) đang ít nhất."""
    def score(w: str) -> float:
        ks = [tone_of(ch) for ch in w]
        return min((cov[x] for x in ks if x), default=10 ** 9)
    return sorted(RARE_WORDS, key=lambda w: (score(w), random.random()))[:k]


def quoted(prompt: str) -> list[str]:
    """Câu trong ngoặc kép của yêu cầu (ước lượng độ phủ; lời chat không ngoặc thì lấy cả câu)."""
    import re
    q = re.findall(r"[\"“”']([^\"“”']{2,})[\"“”']", prompt)
    return q or [prompt]


def batch(llm, sector: str, style: str, words: list[str], n: int, product: str | None, sparse: bool = False,
          density: str = "vừa", lettering: bool = False) -> list[str]:
    """sparse: yêu cầu sơ sài như phần lớn người dùng thật -> LLM brief tự viết thêm câu (source "ai": tiêu đề, nút, nhãn mục,
    lợi ích chung), đúng như trên prod. Không có lô sơ sài thì tập train ít câu AI hơn prod."""
    what = f"poster quảng cáo cho sản phẩm trong ảnh khách gửi kèm ({product})" if product else f"poster cho {sector}"
    if sparse:
        msg = f"Viết {n} yêu cầu khác nhau: {what}. {SPARSE} Trả JSON {{\"items\": [{{\"prompt\": ...}}]}}."
        raw = llm(SYSTEM, [msg], schema=SCHEMA, fresh=True)
        return [str(it.get("prompt") or "").strip() for it in (raw or {}).get("items") or [] if str(it.get("prompt") or "").strip()]
    msg = (f"Viết {n} yêu cầu khác nhau: {what}, phong cách {style}. Mỗi yêu cầu có {DENSITY[density]}. "
           f"Trong các dòng chữ muốn in, dùng tự nhiên vài từ trong: "
           f"{', '.join(words)}. Một yêu cầu có tiêu đề viết HOA có dấu.{LETTERING if lettering else ''} Trả JSON {{\"items\": [{{\"prompt\": ...}}]}}.")
    raw = llm(SYSTEM, [msg], schema=SCHEMA, fresh=True)   # fresh: không lấy lại câu trả lời cũ trong bộ đệm LLM
    return [str(it.get("prompt") or "").strip() for it in (raw or {}).get("items") or [] if str(it.get("prompt") or "").strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="tổng số yêu cầu")
    ap.add_argument("--product-frac", type=float, default=0.25, help="phần yêu cầu kiểu 2 (có ảnh sản phẩm)")
    ap.add_argument("--sparse-frac", type=float, default=0.3, help="phần lô viết SƠ SÀI (brief tự thêm câu như trên prod)")
    ap.add_argument("--density", default=",".join(str(v) for v in DENSITY_P.values()),
                    help="tỉ lệ lô ít / vừa / dày chữ (lô không sơ sài)")
    ap.add_argument("--lettering-frac", type=float, default=0.15,
                    help="phần lô có mong muốn kiểu / cỡ / vị trí chữ (brief chép vào prompt_en; router của chatbot hay gửi)")
    ap.add_argument("--per-call", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/lora/requests.jsonl")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from textfix import config
    from textfix.llm import json_call
    config.load_env()
    m = config.models()["llm"]
    llm = json_call(m["url"], m["model"], m["key_env"], tag="lora_requests", temperature=1.0)
    random.seed(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    cov = coverage([t for r in rows for t in quoted(r["prompt"])])
    seen = {r["prompt"] for r in rows}
    with out.open("a", encoding="utf-8") as f:
        while len(rows) < a.n:
            product = random.choice(PRODUCTS) if random.random() < a.product_frac else None
            sector, style = random.choice(SECTORS), random.choice(STYLES)
            sparse = random.random() < a.sparse_frac
            density = random.choices(list(DENSITY), [float(x) for x in a.density.split(",")])[0]
            try:
                got = batch(llm, sector, style, rare(cov), a.per_call, product, sparse, density,
                            random.random() < a.lettering_frac)
            except Exception as e:   # LLM hỏng một lô: bỏ lô đó
                print(f"lô hỏng: {type(e).__name__}: {e}")
                continue
            for p in got:
                if p in seen or len(rows) >= a.n:
                    continue
                seen.add(p)
                r = {"id": hashlib.sha1(p.encode()).hexdigest()[:10], "kind": "product" if product else "poster", "prompt": p,
                     "aspect": random.choices(RATIOS, RATIO_P)[0], "sparse": sparse, "density": None if sparse else density}
                if product:   # nửa có nhãn / logo như ảnh khách thật (Dove, Coca...), nửa trơn
                    labeled = random.random() < 0.5
                    r["product_prompt"] = (f"Ảnh chụp sản phẩm {product}, chụp studio, nền đơn giản, ánh sáng mềm. CHỈ có sản phẩm. "
                                           + ("Bao bì có nhãn và logo in như hàng bày bán thật, nhưng ngoài sản phẩm không có chữ nào."
                                              if labeled else "Không có chữ, không logo, không khẩu hiệu nào trên ảnh."))
                    r["labeled"] = labeled
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                cov.update(coverage(quoted(p)))
            print(f"{len(rows)}/{a.n} yêu cầu; tổ hợp ít nhất: "
                  f"{sorted(((v, k) for k in ((x, t) for x in VOWELS for t in TONES.values()) for v in [cov[k]]))[:5]}")


if __name__ == "__main__":
    main()
