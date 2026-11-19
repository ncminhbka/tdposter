"""BƯỚC 4 -- mã hoá trước bộ dữ liệu: chữ (Qwen3-4B) + ảnh (VAE) -> một tệp .pt mỗi mẫu. Lúc huấn luyện chỉ còn DiT trên card.

  python -m lora.cache --ds data/lora/ds --device cuda:0

Khớp đúng đường suy luận của textfix/draw/flux.py:
  - chữ: Flux._encode -> te([prompt]) (đệm tới 512 token, 3 tầng Qwen3 [9, 18, 27] -> 7680 chiều), bf16;
  - ảnh đích: ae.encode(ảnh [-1, 1]) -> latent 128 x H/16 x W/16 (đã chuẩn hoá, mean của VAE: tất định);
  - ảnh tham chiếu: Flux._encode_refs -> default_prep (trần 2024^2 điểm ảnh nếu 1 ảnh, 1024^2 nếu nhiều), id thời gian 10 + 10k.
Caption rút gọn (ctx_short): chỉ câu đầu + các câu có chữ trong ngoặc kép + câu nhắc ảnh tham chiếu (bỏ tả cảnh, như
textfix.brief.prompt._fit) -- train dùng một phần nhỏ bước để LoRA bám vào chữ trong ngoặc kép, không vào khuôn câu của brief.
Mỗi tệp: {ctx (512, 7680) bf16, ctx_short | None, x0 (128, h16, w16) bf16, refs (m, 128) bf16 | None, ref_ids (m, 4) | None,
          frac (h16*w16,) float16 = tỉ lệ điểm ảnh chữ trong từng ô 16x16, kind, id}. Thêm ds/cache/_empty.pt = ctx của prompt rỗng
(bỏ prompt khi học -- giữ nhánh không prompt của CFG bản base).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def short_caption(pe: str) -> str | None:
    """Caption rút gọn: câu đầu + câu có ngoặc kép + câu nhắc ảnh tham chiếu. Không ngắn đi đáng kể -> None."""
    from textfix.brief.prompt import IMAGE_REF, _sentences
    s = _sentences(pe)
    keep = [x for i, x in enumerate(s) if i == 0 or '"' in x or IMAGE_REF.search(x)]
    out = " ".join("".join(keep).split())
    return out if len(out) < 0.8 * len(pe) else None


def to_tensor(img):
    """Ảnh PIL RGB -> tensor (3, H, W) trong [-1, 1] (như flux2.sampling.default_images_prep)."""
    import torchvision
    return 2 * torchvision.transforms.ToTensor()(img.convert("RGB")) - 1


def main() -> None:
    import numpy as np
    import torch
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/lora/ds")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights-dir", default=None, help="thư mục trọng số FLUX.2 klein (mặc định: tìm trong ~/persistent-data)")
    ap.add_argument("--shard", default="0/1", help="i/n: chỉ mã hoá mẫu thứ k với k % n == i (chạy song song trên nhiều card)")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    from flux2 import util
    from flux2.sampling import default_prep, prc_img

    from lora.flow import token_weights
    from textfix.draw.flux import find_weights
    found = find_weights([Path(a.weights_dir)] if a.weights_dir else None)
    print("trọng số:", {k: v for k, v in found.items()})
    name = "flux.2-klein-base-4b"
    te = util.load_text_encoder(name, device=a.device)
    ae = util.load_ae(name, device=a.device).eval()
    ae_dtype = next(ae.parameters()).dtype
    ds = Path(a.ds)
    (ds / "cache").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def ctx_of(prompt: str):
        return te([prompt]).to(torch.bfloat16)[0].cpu()

    @torch.no_grad()
    def latent(t):
        return ae.encode(t[None].to(a.device, ae_dtype))[0].to(torch.bfloat16)

    if si == 0:
        torch.save({"ctx": ctx_of("")}, ds / "cache" / "_empty.pt")
    for split in ("train", "eval"):
        mf = ds / f"manifest_{split}.jsonl"
        if not mf.exists():
            continue
        rows = [json.loads(x) for x in mf.read_text(encoding="utf-8").splitlines() if x.strip()]
        outd = ds / "cache" / split
        outd.mkdir(parents=True, exist_ok=True)
        for k, m in enumerate(rows, 1):
            f = outd / f"{m['id']}.pt"
            if f.exists() or (k - 1) % sn != si:
                continue
            d = ds / split / m["id"]
            x0 = latent(to_tensor(Image.open(d / "target.png"))).cpu()
            h16, w16 = x0.shape[1], x0.shape[2]
            def frac_of(name: str):
                """Tỉ lệ điểm ảnh của mặt nạ trong từng ô 16x16 (h16*w16,), float16; không có tệp -> None."""
                if not (d / name).exists():
                    return None
                m = torch.from_numpy(np.asarray(Image.open(d / name).convert("L"), dtype=np.float32) / 255.0)
                return (token_weights(m, h16, w16, 2.0) - 1.0).to(torch.float16)   # text_weight 2 -> trọng số - 1 = tỉ lệ
            frac = frac_of("text.png")
            if frac is None:
                frac = frac_of("mask.png")   # bộ dữ liệu cũ (một mặt nạ)
            ef = frac_of("erase.png")
            kf = frac_of("unver.png")
            if kf is None:
                kf = frac_of("keep.png")
            refs = ids = None
            if m["refs"]:
                limit = 2024 ** 2 if len(m["refs"]) == 1 else 1024 ** 2
                toks, idl = [], []
                for j, rp in enumerate(m["refs"]):
                    t = default_prep(img=Image.open(d / rp).convert("RGB"), limit_pixels=limit)
                    z = latent(t)
                    x, xi = prc_img(z, t_coord=torch.tensor([10 + 10 * j]))
                    toks.append(x.cpu())
                    idl.append(xi.cpu())
                refs, ids = torch.cat(toks), torch.cat(idl)
            short = short_caption(m["prompt_en"])
            torch.save({"id": m["id"], "kind": m["kind"], "ctx": ctx_of(m["prompt_en"]), "x0": x0, "frac": frac,
                        "ctx_short": ctx_of(short) if short else None, "keep_frac": kf, "erase_frac": ef,
                        "refs": refs, "ref_ids": ids}, f)
            if k % 50 == 0:
                print(f"  {split}: {k}/{len(rows)}", flush=True)
        print(f"{split}: {len(rows)} mẫu đã mã hoá -> {outd}")


if __name__ == "__main__":
    main()
