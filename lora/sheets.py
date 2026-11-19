"""BƯỚC 6 -- bảng so sánh bằng mắt, cùng prompt, cùng seed, cùng ảnh tham chiếu:

  python -m lora.sheets --ds data/lora/ds --lora data/lora/run_r32/lora_step03000.safetensors --out data/lora/sheets_r32
  python -m lora.sheets ... --variants base --base-steps 28 --limit 16   # CHẨN ĐOÁN, chỉ khi distill + LoRA không tốt lên
  python -m lora.sheets --ds ... --no-lora --out ...       # vẽ trước ảnh model gốc (trong lúc train); lần sau bỏ qua ảnh đã có
  python -m lora.sheets --ds ... --lora A.safetensors,B.safetensors --out ...   # so hai LoRA (cùng rank) trên cùng bảng
  python -m lora.sheets ... --split train --limit 12       # mẫu ĐÃ học: không đổi gì -> chưa học được; đổi mà eval không -> thiếu dữ liệu

Mỗi hàng một mẫu của tập eval (không bao giờ dùng để học):
  [TextFix (chữ đúng) | distill 4 bước | distill + LoRA]   (mặc định: prod chỉ chạy distill)
Bản base (50 bước CFG, ~46 s / ảnh / A30) chỉ để chẩn đoán: base + LoRA tốt mà distill + LoRA không -> LoRA học được nhưng không
chuyển sang distill (nhánh 5B); cả hai không -> LoRA không học được.
dưới hàng: kiểu (new / product / edit), các câu chữ đúng. Thêm --notext prompt KHÔNG chữ: LoRA không được tự sinh chữ lạ.
Ra: <out>/sheet_<kiểu>_NN.jpg (vài hàng một trang) + <out>/img/ (từng ảnh). Mỗi biến thể nạp một lần; LoRA bật / tắt bằng
net.set_mult trên cùng model (không nạp lại).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEXT = ["A cozy coffee shop interior with warm morning light, wooden tables and plants, photorealistic.",
          "A bowl of Vietnamese pho with fresh herbs on a rustic table, top view, food photography.",
          "A modern living room with a grey sofa and a family watching TV, natural light.",
          "A bouquet of pink roses in a glass vase on a white background, studio photo.",
          "A red sports shoe floating on a pastel blue background, product photography.",
          "A mountain lake at sunrise with mist over the water, landscape photo.",
          "A golden retriever puppy sitting on green grass, sunny day.",
          "A bottle of shampoo with water splashes on a clean white background.",
          "A street food stall at night in Hanoi with warm lanterns, cinematic.",
          "A minimalist flat illustration of a city skyline at dusk, pastel colors.",
          "A plate of banh mi sandwiches with pickled vegetables, close-up.",
          "Fresh fruits arranged on a marble counter: mango, dragon fruit, pineapple.",
          "A spa room with candles, towels and orchids, soft light.",
          "A young woman jogging in a park in the morning, motion blur background.",
          "A cup of bubble milk tea with tapioca pearls on a pink background.",
          "A wooden boat on a calm river with limestone mountains, Ninh Binh style.",
          "A gaming laptop with RGB keyboard on a dark desk, tech product shot.",
          "A traditional Vietnamese ao dai hanging in a bright boutique.",
          "Golden autumn leaves on a cobblestone street, shallow depth of field.",
          "A birthday cake with candles on a table decorated with balloons."]
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
         "/usr/share/fonts/noto/NotoSans-Regular.ttf", str(ROOT / "assets" / "fonts" / "BeVietnamPro-Regular.woff2"),
         "C:/Windows/Fonts/arial.ttf"]


def font(size: int):
    from PIL import ImageFont
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_rgb(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


def tile(img, h: int):
    from PIL import Image
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def zoom_box(mask_path: Path, H: int, W: int) -> tuple[int, int, int, int] | None:
    """Khung vùng chữ dựng lớn nhất (thành phần liên thông lớn nhất của text.png), nới 15% -> để phóng to cùng chỗ ở mọi cột."""
    import cv2
    from PIL import Image
    if not mask_path.exists():
        return None
    m = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
    if m.shape != (H, W) or not m.any():
        return None
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))   # gộp các chữ của một dòng
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in st[k, :4])
    px, py = int(0.15 * w) + 8, int(0.15 * h) + 8
    return max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)


def sheet(rows: list[dict], cols: list[str], out: Path, h: int = 512) -> None:
    """rows: {"label", "texts", "imgs": [ảnh | None theo cols], "box": khung phóng to | None} -> một ảnh JPG. Có "box" -> dưới mỗi
    ảnh thêm ô phóng to vùng chữ lớn nhất (cùng toạ độ ở mọi cột) để soi dấu."""
    from PIL import Image, ImageDraw
    f_head, f_txt = font(28), font(24)
    tiles = [[tile(i, h) if i is not None else None for i in r["imgs"]] for r in rows]
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    colw = [max(max((t[c].width for t in tiles if t[c] is not None), default=h), int(probe.textlength(cols[c], font=f_head)) + 4)
            for c in range(len(cols))]   # cột không hẹp hơn tên cột
    zh = h // 2 if any(r.get("box") for r in rows) else 0
    cap = 36 + 30 * max(1, max(len(r["texts"]) for r in rows))
    W, H = sum(colw) + 10 * (len(cols) + 1), 50 + len(rows) * (h + zh + cap + 30)
    S = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(S)
    x = 10
    for c, name in enumerate(cols):
        d.text((x, 10), name, fill=(200, 0, 0), font=f_head)
        x += colw[c] + 10
    y = 50
    for r, ts in zip(rows, tiles):
        x = 10
        for c, t in enumerate(ts):
            if t is not None:
                S.paste(t, (x, y))
                img = r["imgs"][c]
                if zh and r.get("box") is not None and img is not None:
                    x0, y0, x1, y1 = r["box"]
                    crop = Image.fromarray(np.asarray(img)[y0:y1, x0:x1])
                    crop = crop.resize((min(colw[c], round(crop.width * zh / crop.height)), zh), Image.LANCZOS)
                    S.paste(crop, (x, y + h + 6))
            x += colw[c] + 10
        yt = y + h + (zh + 12 if zh else 6)
        d.text((10, yt), r["label"], fill=(90, 90, 90), font=f_txt)
        for k, s_ in enumerate(r["texts"]):
            d.text((10, yt + 30 + 30 * k), f"• {s_}", fill=(0, 0, 0), font=f_txt)
        y += h + zh + cap + 30
    S.save(out, quality=88)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/lora/ds")
    ap.add_argument("--lora", default="", help="một hoặc nhiều tệp LoRA, cách nhau dấu phẩy (CÙNG rank); bỏ trống với --no-lora")
    ap.add_argument("--no-lora", action="store_true", help="chỉ vẽ ảnh model gốc (chạy trước, trong lúc đang train)")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval", help="eval (chưa từng học) | train (xem học thuộc)")
    ap.add_argument("--variants", default="distill", help="distill (mặc định) | base (chẩn đoán) | distill,base")
    ap.add_argument("--base-steps", type=int, default=50, help="số bước bản base khi chẩn đoán (28 đủ để nhìn chữ, nhanh gần gấp đôi)")
    ap.add_argument("--limit", type=int, default=60, help="số mẫu tối đa")
    ap.add_argument("--notext", type=int, default=20)
    ap.add_argument("--rows", type=int, default=4, help="số hàng mỗi trang")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights-dir", default=None)
    a = ap.parse_args()
    import os

    from PIL import Image

    from lora import net
    from textfix.draw.flux import Flux
    loras = [] if a.no_lora else [x.strip() for x in a.lora.split(",") if x.strip()]
    if not a.no_lora and not loras:
        raise SystemExit("cần --lora <tệp> (hoặc --no-lora)")
    tags = ["gốc"] + [Path(x).parent.name + "/" + Path(x).stem.replace("lora_", "") for x in loras]
    os.environ.pop("TEXTFIX_FLUX_LORA", None)   # bảng tự bật / tắt LoRA trên cùng model, không gộp lúc nạp
    ds, out = Path(a.ds), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ds / f"manifest_{a.split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = sorted(rows, key=lambda m: (m["kind"], m["id"]))[: a.limit]
    jobs = [{"id": m["id"], "kind": m["kind"], "prompt": m["prompt_en"], "W": m["W"], "H": m["H"], "seed": m["seed"],
             "texts": m["texts"], "refs": [ds / a.split / m["id"] / r for r in m["refs"]],
             "target": ds / a.split / m["id"] / "target.png", "text_mask": ds / a.split / m["id"] / "text.png"} for m in rows]
    jobs += [{"id": f"notext_{k:02d}", "kind": "notext", "prompt": p, "W": 1024, "H": 1024, "seed": 1000 + k, "texts": [],
              "refs": [], "target": None, "text_mask": None} for k, p in enumerate(NOTEXT[: a.notext])]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]

    def img_path(j, v, tag):
        return out / "img" / f"{j['id']}_{v}_{tag.replace('/', '-')}.jpg"
    for v in variants:
        F = Flux(v, steps=a.base_steps if v == "base" else None, device=a.device, weights_dir=a.weights_dir).load()
        for li, tag in enumerate(tags):
            if li > 0:
                net.load(loras[li - 1], F.dit, mult=a.mult)
            elif loras:
                net.load(loras[0], F.dit, mult=0.0)   # gắn khung LoRA, tắt -> model gốc
            todo = [j for j in jobs if not img_path(j, v, tag).exists()]
            for k, j in enumerate(todo, 1):
                refs = [load_rgb(p_) for p_ in j["refs"]] or None
                img = F.generate(j["prompt"], j["W"], j["H"], int(j["seed"] or 0), refs=refs)
                Image.fromarray(img).save(img_path(j, v, tag), quality=92)
                if k % 10 == 0:
                    print(f"  {v} {tag}: {k}/{len(todo)}", flush=True)
            if loras:
                net.set_mult(F.dit, 0.0)
        del F
        gc.collect()
        torch.cuda.empty_cache()
    cols = ["TextFix (chữ đúng)"] + [f"{v}{'' if tag == 'gốc' else ' + ' + tag}" for v in variants for tag in tags]
    by_kind: dict[str, list] = {}
    for j in jobs:
        imgs = [load_rgb(j["target"]) if j["target"] else None]
        for v in variants:
            for tag in tags:
                f = img_path(j, v, tag)
                imgs.append(load_rgb(f) if f.exists() else None)
        box = zoom_box(j["text_mask"], j["H"], j["W"]) if j["text_mask"] else None
        by_kind.setdefault(j["kind"], []).append({"label": f"{a.split} · {j['kind']} · {j['id']} · seed {j['seed']}",
                                                  "texts": j["texts"], "imgs": imgs, "box": box})
    for kind, rs in by_kind.items():
        for p_ in range(0, len(rs), a.rows):
            sheet(rs[p_:p_ + a.rows], cols, out / f"sheet_{a.split}_{kind}_{p_ // a.rows + 1:02d}.jpg")
    print(f"bảng so sánh: {out}/sheet_*.jpg ({sum(len(r) for r in by_kind.values())} hàng)")


if __name__ == "__main__":
    main()
