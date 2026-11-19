"""Tải ảnh sản phẩm THẬT có chữ (bao bì) cho dữ liệu kiểu 2 của pilot LoRA -- từ Open Food / Beauty / Products Facts.

Nguồn: ảnh mặt trước do cộng đồng chụp, giấy phép CC BY-SA 3.0 (cho phép dùng huấn luyện, phải ghi nguồn -> SOURCES.csv).
Ưu tiên sản phẩm bán ở Việt Nam (nhãn tiếng Việt); đa dạng: mỗi thương hiệu <= 2 ảnh, chia đều theo nhóm hàng; cạnh ngắn >= 512 px.

  python -m lora.fetch_products --out data/lora/products --n 150
  python -m lora.drive ... --products data/lora/products

Sau khi tải nên XEM LƯỚT (contact sheet: --sheet) và xoá ảnh hỏng (mờ, cầm tay che nhãn, chỉ có mặt sau / bảng dinh dưỡng).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

UA = "textfix-dataset/0.1 (research)"
SITES = {"food": "world.openfoodfacts.org", "beauty": "world.openbeautyfacts.org", "products": "world.openproductsfacts.org"}
LICENSE = {"food": "CC BY-SA 3.0, Open Food Facts contributors", "beauty": "CC BY-SA 3.0, Open Beauty Facts contributors",
           "products": "CC BY-SA 3.0, Open Products Facts contributors"}
# nhóm hàng thô theo categories_tags -> để chia đều (mì gói không chiếm cả tập)
GROUPS = [("noodle", ("noodle", "instant")), ("drink", ("beverage", "drink", "water", "juice", "tea", "coffee", "soda")),
          ("dairy", ("dair", "milk", "yogurt", "cheese")), ("snack", ("snack", "biscuit", "cookie", "chip", "cracker", "candy",
          "confection", "chocolate", "cake")), ("sauce", ("sauce", "condiment", "seasoning", "spice", "fish-sauce", "oil")),
          ("canned", ("canned", "meat", "fish", "seafood")), ("grain", ("rice", "cereal", "flour", "bread"))]


def get_json(url: str, tries: int = 3) -> dict:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))
    return {}


def get_bytes(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except Exception:
        return None


def group_of(source: str, cats: list[str]) -> str:
    if source != "food":
        return source
    s = " ".join(cats).lower()
    for g, keys in GROUPS:
        if any(k in s for k in keys):
            return g
    return "other"


def candidates(source: str, country: str, pages: int) -> list[dict]:
    out = []
    for page in range(1, pages + 1):
        url = (f"https://{SITES[source]}/api/v2/search?countries_tags_en={country}&page={page}&page_size=100"
               "&fields=code,product_name,brands,categories_tags,image_front_url,lang&sort_by=unique_scans_n")
        try:
            ps = get_json(url).get("products", [])
        except Exception as e:
            print(f"  {source} trang {page}: lỗi {type(e).__name__}")
            break
        out += [dict(p, source=source) for p in ps if p.get("image_front_url")]
        if len(ps) < 100:
            break
        time.sleep(7)   # API tìm kiếm giới hạn ~10 lượt / phút
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lora/products")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--country", default="vietnam")
    ap.add_argument("--food-pages", type=int, default=6)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--per-brand", type=int, default=2)
    ap.add_argument("--sheet", action="store_true", help="chỉ dựng contact sheet từ ảnh đã tải (xem lướt để xoá ảnh hỏng)")
    a = ap.parse_args()
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.sheet:
        return contact(out)
    cands = candidates("food", a.country, a.food_pages) + candidates("beauty", a.country, 2) + candidates("products", a.country, 2)
    print(f"ứng viên: {len(cands)}")
    # chia đều theo nhóm: vòng tròn qua các nhóm, mỗi nhóm theo thứ tự lượt quét (phổ biến trước)
    buckets: dict[str, list[dict]] = {}
    seen = set()
    for p in cands:
        if p["code"] in seen:
            continue
        seen.add(p["code"])
        buckets.setdefault(group_of(p["source"], p.get("categories_tags") or []), []).append(p)
    print("nhóm:", {g: len(v) for g, v in buckets.items()})
    brands: Counter = Counter()
    rows, k = [], 0
    while len(rows) < a.n and any(buckets.values()):
        g = sorted(buckets)[k % len(buckets)]
        k += 1
        if not buckets[g]:
            continue
        p = buckets[g].pop(0)
        brand = (p.get("brands") or "").split(",")[0].strip().lower() or p["code"]
        if brands[brand] >= a.per_brand:
            continue
        url = p["image_front_url"]
        full = url.rsplit(".", 2)[0] + ".full.jpg" if url.count(".") >= 2 else url   # front_vi.38.400.jpg -> front_vi.38.full.jpg
        raw = get_bytes(full) or get_bytes(url)
        if not raw:
            continue
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        if min(im.size) < a.min_side:
            continue
        if max(im.size) > 2048:
            im.thumbnail((2048, 2048), Image.LANCZOS)
        name = f"{p['source']}_{p['code']}.jpg"
        im.save(out / name, quality=92)
        brands[brand] += 1
        rows.append({"file": name, "group": g, "product_name": p.get("product_name", ""), "brands": p.get("brands", ""),
                     "image_url": full, "page": f"https://{SITES[p['source']]}/product/{p['code']}",
                     "license": LICENSE[p["source"]]})
        if len(rows) % 25 == 0:
            print(f"  đã tải {len(rows)}")
    with open(out / "SOURCES.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"xong: {len(rows)} ảnh -> {out} (nhóm: {dict(Counter(r['group'] for r in rows))}); nguồn + giấy phép: SOURCES.csv")
    contact(out)


def contact(out: Path, cell: int = 200, cols: int = 10) -> None:
    """Lưới ảnh thu nhỏ có số thứ tự -> sheet_XX.jpg, để xem lướt và xoá ảnh hỏng."""
    from PIL import Image, ImageDraw
    files = sorted(out.glob("*.jpg"))
    (out / "_sheets").mkdir(exist_ok=True)   # thư mục con: drive --products chỉ đọc tệp ngay trong out/
    per = cols * 5
    for s in range(0, len(files), per):
        part = files[s:s + per]
        S = Image.new("RGB", (cols * cell, ((len(part) + cols - 1) // cols) * (cell + 18)), "white")
        d = ImageDraw.Draw(S)
        for i, f in enumerate(part):
            im = Image.open(f).convert("RGB")
            im.thumbnail((cell, cell))
            x, y = (i % cols) * cell, (i // cols) * (cell + 18)
            S.paste(im, (x + (cell - im.width) // 2, y))
            d.text((x + 4, y + cell + 2), f"{s + i}: {f.stem[-13:]}", fill=(0, 0, 0))
        S.save(out / "_sheets" / f"sheet_{s // per + 1:02d}.jpg", quality=85)


if __name__ == "__main__":
    main()
