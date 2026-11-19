"""BƯỚC 5 -- huấn luyện LoRA trên DiT FLUX.2 klein BASE 4B (rồi chạy trên bản distill: lora.sheets / TEXTFIX_FLUX_LORA).

  python -m lora.train --ds data/lora/ds --out data/lora/run_r32 --rank 32 --steps 3000 --device cuda:0
  python -m lora.train ... --overfit 50 --steps 500        # chặng 3: học thuộc 50 mẫu (kiểm đường dữ liệu / VRAM)

Mỗi bước (lora.flow): mẫu ngẫu nhiên (cân theo kiểu 1/2/3 nếu --kind-weights), eps ~ N(0, 1), t (logit-normal dịch theo lịch base
+ một phần đúng 4 mức t của distill), x_t = (1-t)·x0 + t·eps, [x_t, ảnh tham chiếu] -> DiT -> pred[:, :n], loss = MSE(pred, eps - x0)
nhân trọng số vùng chữ (--text-weight). Bỏ prompt với xác suất --cond-drop (giữ nhánh không prompt mà CFG của bản base dùng).
LoRA float32, DiT bf16 đóng băng, gradient checkpointing từng khối. Lưu LoRA (safetensors) mỗi --save-every bước + bước cuối.
Theo dõi: log.jsonl (loss, MSE trong / ngoài vùng chữ, giây / bước, VRAM đỉnh).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_items(ds: Path, split: str, overfit: int) -> list[Path]:
    files = sorted((ds / "cache" / split).glob("*.pt"))
    if not files:
        raise SystemExit(f"chưa có {ds}/cache/{split}/*.pt -- chạy python -m lora.cache trước")
    return files[:overfit] if overfit else files


def nanmean(xs) -> float | None:
    v = [x for x in xs if x == x]   # bỏ NaN (mẫu không có token chữ / không có token nền)
    return round(sum(v) / len(v), 5) if v else None


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Khởi động tuyến tính rồi cosine về 10%."""
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/lora/ds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None, help="mặc định = rank")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--accum", type=int, default=4, help="cộng dồn gradient (mỗi mẫu một lượt, cỡ ảnh khác nhau)")
    ap.add_argument("--text-weight", type=float, default=6.0, help="trọng số loss token vùng chữ (ngoài chữ = 1)")
    ap.add_argument("--keep-weight", type=float, default=0.3,
                    help="trọng số dòng chữ FLUX vẽ mà TextFix giữ nguyên (nhãn sản phẩm, con dấu: không ai kiểm đúng sai)")
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--short-caption", type=float, default=0.15,
                    help="phần bước dùng caption rút gọn (chỉ câu có chữ) -- LoRA bám vào chữ trong ngoặc kép, không vào khuôn brief")
    ap.add_argument("--p-distill", type=float, default=0.25, help="phần bước lấy đúng 4 mức t của bản distill")
    ap.add_argument("--kind-weights", default="new=1,product=1,edit=1", help="cân lấy mẫu theo kiểu")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--overfit", type=int, default=0, help=">0: chỉ học N mẫu đầu (chặng 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--resume", default=None, help="tệp LoRA để học tiếp")
    a = ap.parse_args()

    from flux2 import util
    from flux2.sampling import prc_img, prc_txt

    from lora import flow, net
    from textfix.draw.flux import find_weights
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=1), encoding="utf-8")

    ds = Path(a.ds)
    files = load_items(ds, "train", a.overfit)
    kind_of = {}   # kiểu từng mẫu lấy từ manifest (không mở tệp cache: mỗi tệp ~8 MB)
    for line in (ds / "manifest_train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            kind_of[m["id"]] = m["kind"]
    kinds = {f: kind_of.get(f.stem, "new") for f in files}
    kw = dict(x.split("=") for x in a.kind_weights.split(","))
    weights = [float(kw.get(kinds[f], 1.0)) for f in files]
    empty = torch.load(ds / "cache" / "_empty.pt", map_location="cpu", weights_only=False)["ctx"]
    print(f"{len(files)} mẫu: " + ", ".join(f"{k} {sum(1 for f in files if kinds[f] == k)}" for k in sorted(set(kinds.values()))))

    find_weights([Path(a.weights_dir)] if a.weights_dir else None)
    torch.backends.cuda.matmul.allow_tf32 = True
    dit = util.load_flow_model("flux.2-klein-base-4b", device=a.device)
    names = net.inject(dit, a.rank, a.alpha)
    if a.resume:
        net.load(a.resume, dit)
    nblk = flow.checkpointing(dit)
    params = net.parameters(dit)
    print(f"LoRA rank {a.rank}: {len(names)} lớp, {sum(p.numel() for p in params) / 1e6:.1f}M tham số; checkpoint {nblk} khối")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    dev = torch.device(a.device)
    log = (out / "log.jsonl").open("a", encoding="utf-8")
    ema, t0 = None, time.time()
    dit.train()
    for step in range(a.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, a.lr, a.warmup, a.steps)
        parts_acc = []
        for _ in range(a.accum):
            item = torch.load(random.choices(files, weights)[0], map_location="cpu", weights_only=False)
            x0, ids = prc_img(item["x0"].to(dev))
            x0, ids = x0[None], ids[None].to(dev)
            n = x0.shape[1]
            u = random.random()
            if u < a.cond_drop:
                ctx = empty
            elif u < a.cond_drop + a.short_caption and item.get("ctx_short") is not None:
                ctx = item["ctx_short"]
            else:
                ctx = item["ctx"]
            c, c_ids = prc_txt(ctx.to(dev))
            c, c_ids = c[None], c_ids[None].to(dev)
            refs = item["refs"][None].to(dev) if item["refs"] is not None else None
            ref_ids = item["ref_ids"][None].to(dev) if item["refs"] is not None else None
            eps = torch.randn(x0.shape, generator=g).to(dev, x0.dtype)
            t = flow.sample_t(n, g, a.p_distill)
            pred = flow.predict(dit, flow.noisy(x0, eps, t).to(torch.bfloat16), ids, t, c, c_ids, refs, ref_ids)
            kf = item.get("keep_frac")
            w = flow.weights(item["frac"].to(dev), kf.to(dev) if kf is not None else None, a.text_weight, a.keep_weight)
            loss, parts = flow.loss_fn(pred, eps - x0, w)
            (loss / a.accum).backward()
            parts_acc.append({"loss": loss.item(), **parts, "t": t, "kind": item["kind"]})
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        L = sum(p["loss"] for p in parts_acc) / len(parts_acc)
        ema = L if ema is None else 0.98 * ema + 0.02 * L
        row = {"step": step + 1, "loss": round(L, 5), "ema": round(ema, 5), "lr": opt.param_groups[0]["lr"],
               "text": nanmean(p["text"] for p in parts_acc), "bg": nanmean(p["bg"] for p in parts_acc),
               "s_per_step": round((time.time() - t0) / (step + 1), 2),
               "vram_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 1) if dev.type == "cuda" else 0}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (step + 1) % 25 == 0 or step == 0:
            print(f"bước {step + 1}/{a.steps} loss {L:.4f} (ema {ema:.4f}, chữ {row['text']}, nền {row['bg']}) "
                  f"{row['s_per_step']} s/bước, VRAM {row['vram_gb']} GB", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            p = out / f"lora_step{step + 1:05d}.safetensors"
            net.save(str(p), dit, meta={"base": "flux.2-klein-base-4b", "step": step + 1, "text_weight": a.text_weight,
                                        "samples": len(files)})
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
