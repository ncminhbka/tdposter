"""BƯỚC 3 -- lượt chạy của máy chủ -> bộ dữ liệu sạch cho LoRA.

  python -m lora.collect --runs data/lora/runs --out data/lora/ds [--eval-frac 0.08]

Mỗi lượt (thư mục run_*) đi qua bộ lọc (`reject`, không cần OCR: chữ TextFix chép nguyên văn nên đúng từ cách dựng):
  - lượt tạo mới / sửa poster của mình (bỏ ảnh lạ, bỏ ảnh không chữ);
  - không lỗi dựng còn lại (plan.errors), không câu khách bị thiếu (plan.missing);
  - không dòng nháp "keep" mang câu khách (dòng giữ nét nháp có thể sai dấu -> poster không sạch);
  - vùng khác nhau giữa nháp và poster (mặt nạ) trong khoảng hợp lý.
Mẫu đạt -> ds/<split>/<id>/: target.png (poster TextFix), draft.png, mask.png (vùng chữ, trắng = chữ), ref0.png, ref1.png ...
(đúng thứ tự FLUX đã nhận: sửa = [poster cũ, ảnh sản phẩm mới?]; tạo mới = [ảnh sản phẩm?, ảnh kho...]), meta.json.
ds/manifest_<split>.jsonl liệt kê mẫu. Tách train / eval theo lời yêu cầu gốc (một yêu cầu và các lần sửa của nó cùng một phía).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIFF = 3            # điểm ảnh poster khác nháp > 3 (trên thang 255, kênh lớn nhất) = vùng TextFix đã đổi
GROW = 8            # nới mặt nạ 8 px (viền khử răng cưa, bóng đổ ngắn của chữ dựng)
MASK_MIN, MASK_MAX = 0.003, 0.6   # phần ảnh bị đổi: < 0,3% = gần như không có chữ; > 60% = poster khác hẳn nháp (lỗi)


def load_png(p: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"))


def text_mask(draft: np.ndarray, poster: np.ndarray, grow: int = GROW) -> np.ndarray:
    """Vùng TextFix đã đổi (bool HxW): |poster - nháp| > DIFF ở kênh lớn nhất, nới `grow` px."""
    import cv2
    m = (np.abs(poster.astype(np.int16) - draft.astype(np.int16)).max(-1) > DIFF).astype(np.uint8)
    if grow > 0 and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m.astype(bool)


def reject(req: dict, brief: dict, plan: dict | None, allow_kept_client: bool = False) -> str | None:
    """Lý do bỏ lượt (None = giữ). Chỉ đọc request.json / brief.json / plan.json."""
    mode = req.get("mode")
    if mode not in ("new", "edit"):
        return f"luồng {mode}"
    if plan is None:
        return "không có plan.json"
    if plan.get("mode") == "image" or not brief.get("texts"):
        return "ảnh không chữ"
    if plan.get("errors"):
        return "lỗi dựng: " + ",".join(sorted({str(e.get("type")) for e in plan["errors"]}))
    if plan.get("missing"):
        return "thiếu câu khách"
    if not allow_kept_client and any(o.get("kind") == "keep" and o.get("client") is not None for o in plan.get("ops") or []):
        return "dòng nháp giữ nguyên mang câu khách"
    return None


def refs_of(run: Path, req: dict, brief: dict) -> list[np.ndarray] | None:
    """Ảnh tham chiếu đúng như FLUX đã nhận (server/app.py generate + engine.draw). Thiếu tệp -> None (bỏ mẫu)."""
    product = load_png(run / "product.png") if (run / "product.png").exists() else None
    if req.get("mode") == "edit":
        if not (run / "source.png").exists():
            return None
        return [load_png(run / "source.png")] + ([product] if product is not None else [])
    from textfix import kho
    want = len(brief.get("kho") or [])
    refs = kho.refs_for_draw(brief, product)
    if len(refs) != (product is not None) + min(want, kho.MAX_REFS - (product is not None)):
        return None   # ảnh kho không còn trên máy -> không dựng lại được đúng đầu vào
    return refs


def chain_root(runs: Path, run_id: str, cache: dict) -> str:
    """Lượt gốc của chuỗi sửa (request.json: parent) -- để tách train / eval theo yêu cầu gốc."""
    seen = []
    cur = run_id
    while cur and cur not in cache:
        seen.append(cur)
        try:
            req = json.loads((runs / cur / "request.json").read_text(encoding="utf-8"))
        except OSError:
            break
        cur = req.get("parent")
    root = cache.get(cur, cur or seen[-1])
    for s in seen:
        cache[s] = root
    return root


def main() -> None:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/lora/runs")
    ap.add_argument("--out", default="data/lora/ds")
    ap.add_argument("--eval-frac", type=float, default=0.08, help="phần yêu cầu gốc để riêng cho bảng so sánh")
    ap.add_argument("--allow-kept-client", action="store_true", help="giữ cả lượt có dòng nháp giữ nguyên mang câu khách")
    a = ap.parse_args()
    from textfix.config import load_env
    load_env()
    runs, out = Path(a.runs), Path(a.out)
    why, roots = Counter(), {}
    manifests = {"train": [], "eval": []}
    for run in sorted(p for p in runs.iterdir() if p.is_dir() and p.name.startswith("run_")):
        try:
            req = json.loads((run / "request.json").read_text(encoding="utf-8"))
            brief = json.loads((run / "brief.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            why["thiếu request / brief"] += 1
            continue
        pf = run / "v0" / "plan.json"
        plan = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else None
        r = reject(req, brief, plan, a.allow_kept_client)
        if r is None and not ((run / "v0" / "draft.png").exists() and (run / "v0" / "poster.png").exists()):
            r = "thiếu draft.png / poster.png"
        if r:
            why[r.split(":")[0]] += 1
            continue
        draft, poster = load_png(run / "v0" / "draft.png"), load_png(run / "v0" / "poster.png")
        if draft.shape != poster.shape or draft.shape[0] % 16 or draft.shape[1] % 16:
            why["cỡ nháp / poster lệch"] += 1
            continue
        mask = text_mask(draft, poster)
        if not MASK_MIN <= mask.mean() <= MASK_MAX:
            why[f"mặt nạ ngoài [{MASK_MIN}, {MASK_MAX}]"] += 1
            continue
        refs = refs_of(run, req, brief)
        if refs is None:
            why["thiếu ảnh tham chiếu"] += 1
            continue
        root = chain_root(runs, run.name, roots)
        split = "eval" if int(hashlib.sha1(root.encode()).hexdigest(), 16) % 10_000 < a.eval_frac * 10_000 else "train"
        d = out / split / run.name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "v0" / "poster.png", d / "target.png")
        shutil.copyfile(run / "v0" / "draft.png", d / "draft.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(d / "mask.png")
        for k, im in enumerate(refs):
            Image.fromarray(im).save(d / f"ref{k}.png")
        kind = "edit" if req["mode"] == "edit" else ("product" if req.get("has_product_image") else "new")
        meta = {"id": run.name, "kind": kind, "prompt_en": (plan or {}).get("prompt_en") or brief["prompt_en"],
                "seed": brief.get("seed"), "W": int(poster.shape[1]), "H": int(poster.shape[0]),
                "texts": [t["text"] for t in brief.get("texts") or []], "refs": [f"ref{k}.png" for k in range(len(refs))],
                "root": root, "mask_frac": round(float(mask.mean()), 4), "request": req.get("prompt")}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        manifests[split].append(meta)
        why["GIỮ"] += 1
    for split, rows in manifests.items():
        (out / f"manifest_{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                     encoding="utf-8")
    kinds = Counter((s, r["kind"]) for s, rows in manifests.items() for r in rows)
    print("lý do:", dict(why.most_common()))
    print("mẫu:", dict(sorted(kinds.items())))


if __name__ == "__main__":
    main()
