"""BƯỚC 6 -- bảng so sánh bằng mắt, cùng prompt, cùng seed, cùng ảnh tham chiếu:

  python -m lora.sheets --ds data/lora/ds --lora data/lora/run_r32/lora_step03000.safetensors --out data/lora/sheets_r32

Mỗi hàng một mẫu của tập eval (không bao giờ dùng để học):
  [TextFix (chữ đúng) | distill 4 bước | distill + LoRA | base 50 bước | base + LoRA]
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


def sheet(rows: list[dict], cols: list[str], out: Path, h: int = 512) -> None:
    """rows: {"label", "texts", "imgs": [ảnh hoặc None theo cols]} -> một ảnh JPG."""
    from PIL import Image, ImageDraw
    f_head, f_txt = font(28), font(24)
    tiles = [[tile(i, h) if i is not None else None for i in r["imgs"]] for r in rows]
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    colw = [max(max((t[c].width for t in tiles if t[c] is not None), default=h), int(probe.textlength(cols[c], font=f_head)) + 4)
            for c in range(len(cols))]   # cột không hẹp hơn tên cột
    cap = 36 + 30 * max(1, max(len(r["texts"]) for r in rows))
    W, H = sum(colw) + 10 * (len(cols) + 1), 50 + len(rows) * (h + cap + 20)
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
            x += colw[c] + 10
        d.text((10, y + h + 6), r["label"], fill=(90, 90, 90), font=f_txt)
        for k, s in enumerate(r["texts"]):
            d.text((10, y + h + 36 + 30 * k), f"• {s}", fill=(0, 0, 0), font=f_txt)
        y += h + cap + 20
    S.save(out, quality=88)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/lora/ds")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="distill,base")
    ap.add_argument("--limit", type=int, default=60, help="số mẫu eval tối đa (base 50 bước chậm)")
    ap.add_argument("--notext", type=int, default=20)
    ap.add_argument("--rows", type=int, default=4, help="số hàng mỗi trang")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights-dir", default=None)
    a = ap.parse_args()
    import os

    from lora import net
    from textfix.draw.flux import Flux
    os.environ.pop("TEXTFIX_FLUX_LORA", None)   # bảng tự bật / tắt LoRA trên cùng model, không gộp lúc nạp
    ds, out = Path(a.ds), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ds / "manifest_eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = sorted(rows, key=lambda m: (m["kind"], m["id"]))[: a.limit]
    jobs = [{"id": m["id"], "kind": m["kind"], "prompt": m["prompt_en"], "W": m["W"], "H": m["H"], "seed": m["seed"],
             "texts": m["texts"], "refs": [ds / "eval" / m["id"] / r for r in m["refs"]],
             "target": ds / "eval" / m["id"] / "target.png"} for m in rows]
    jobs += [{"id": f"notext_{k:02d}", "kind": "notext", "prompt": p, "W": 1024, "H": 1024, "seed": 1000 + k, "texts": [],
              "refs": [], "target": None} for k, p in enumerate(NOTEXT[: a.notext])]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    for v in variants:
        F = Flux(v, device=a.device, weights_dir=a.weights_dir).load()
        net.load(a.lora, F.dit, mult=0.0)
        for k, j in enumerate(jobs, 1):
            refs = [load_rgb(p) for p in j["refs"]] or None
            for on in (0, 1):
                f = out / "img" / f"{j['id']}_{v}_{on}.jpg"
                if f.exists():
                    continue
                net.set_mult(F.dit, a.mult if on else 0.0)
                img = F.generate(j["prompt"], j["W"], j["H"], int(j["seed"] or 0), refs=refs)
                from PIL import Image
                Image.fromarray(img).save(f, quality=92)
            if k % 10 == 0:
                print(f"  {v}: {k}/{len(jobs)}", flush=True)
        del F
        gc.collect()
        torch.cuda.empty_cache()
    cols = ["TextFix (chữ đúng)"] + [f"{v}{' + LoRA' if on else ''}" for v in variants for on in (0, 1)]
    by_kind: dict[str, list] = {}
    for j in jobs:
        imgs = [load_rgb(j["target"]) if j["target"] else None]
        for v in variants:
            for on in (0, 1):
                f = out / "img" / f"{j['id']}_{v}_{on}.jpg"
                imgs.append(load_rgb(f) if f.exists() else None)
        by_kind.setdefault(j["kind"], []).append({"label": f"{j['kind']} · {j['id']} · seed {j['seed']}", "texts": j["texts"],
                                                  "imgs": imgs})
    for kind, rs in by_kind.items():
        for p in range(0, len(rs), a.rows):
            sheet(rs[p:p + a.rows], cols, out / f"sheet_{kind}_{p // a.rows + 1:02d}.jpg")
    print(f"bảng so sánh: {out}/sheet_*.jpg ({sum(len(r) for r in by_kind.values())} hàng)")


if __name__ == "__main__":
    main()
