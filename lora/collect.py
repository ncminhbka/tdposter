"""BƯỚC 3 -- lượt chạy của máy chủ -> bộ dữ liệu sạch cho LoRA.

  python -m lora.collect --runs data/lora/runs --out data/lora/ds [--eval-frac 0.08]

Mỗi lượt (thư mục run_*) đi qua bộ lọc (`reject`, không cần OCR: chữ TextFix chép nguyên văn nên đúng từ cách dựng):
  - lượt tạo mới / sửa poster của mình (bỏ ảnh lạ, bỏ ảnh không chữ);
  - không lỗi dựng còn lại (plan.errors), không câu khách bị thiếu (plan.missing);
  - không dòng nháp "keep" mang câu khách (dòng giữ nét nháp có thể sai dấu -> poster không sạch);
  - vùng khác nhau giữa nháp và poster (mặt nạ) trong khoảng hợp lý.
Mẫu đạt -> ds/<split>/<id>/: target.png (poster TextFix), draft.png, mask.png (vùng chữ, trắng = chữ), keep.png (dòng "keep":
chữ FLUX vẽ mà TextFix giữ nguyên, vd nhãn sản phẩm -- train hạ trọng số), ref0.png, ref1.png ...
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


def keep_mask(plan: dict, H: int, W: int, pad: int = 4) -> np.ndarray:
    """Vùng các dòng "keep" (bool HxW): chữ FLUX tự vẽ mà TextFix giữ nguyên -- nhãn / logo in trên ảnh sản phẩm, con dấu, biển
    phụ. Không ai kiểm chúng đúng sai (thường méo) -> train hạ trọng số (--keep-weight) để LoRA không củng cố chữ méo."""
    m = np.zeros((H, W), bool)
    L = {l["id"]: l for l in ((plan.get("slots") or {}).get("L") or [])}
    for o in plan.get("ops") or []:
        if o.get("kind") != "keep":
            continue
        for lid in o.get("marks") or []:
            if lid in L and L[lid].get("box"):
                x0, y0, x1, y1 = (int(round(v)) for v in L[lid]["box"])
                m[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = True
    return m


def reject(req: dict, brief: dict, plan: dict | None, allow_kept_client: bool = False, tokens=None) -> str | None:
    """Lý do bỏ lượt (None = giữ). Chỉ đọc request.json / brief.json / plan.json. tokens: hàm đếm token prompt (brief.count_tokens)
    -- prompt quá 512 token thì bộ mã hoá Qwen3 CẮT ĐUÔI im lặng: caption mất câu mà ảnh đích vẫn có câu đó -> LoRA học vẽ chữ
    không có trong prompt. Bỏ."""
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
    if tokens is not None:
        from textfix.brief.prompt import TOKEN_LIMIT
        if tokens(plan.get("prompt_en") or brief.get("prompt_en") or "") > TOKEN_LIMIT:
            return "prompt quá 512 token (bị cắt đuôi)"
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
    from textfix.brief import count_tokens
    from textfix.config import load_env
    from textfix.draw.flux import find_weights
    load_env()
    find_weights()   # đặt TEXT_ENCODER_PATH -> count_tokens dùng đúng tokenizer Qwen3 của FLUX (không có thì ước dư an toàn)
    runs, out = Path(a.runs), Path(a.out)
    why, roots = Counter(), {}
    by_lines = {}   # số dòng chữ của brief -> [đạt, tổng]: poster nhiều chữ hay bị nháp vẽ thiếu dòng -> tỉ lệ đạt thấp
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
        r = reject(req, brief, plan, a.allow_kept_client, count_tokens)
        nl = len(brief.get("texts") or [])
        if nl and req.get("mode") in ("new", "edit") and not (plan or {}).get("mode") == "image":
            by_lines.setdefault(min(nl, 10), [0, 0])[1] += 1
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
        km = keep_mask(plan, *mask.shape) & ~mask   # vùng chữ TextFix đã sửa thắng
        Image.fromarray((km * 255).astype(np.uint8)).save(d / "keep.png")
        for k, im in enumerate(refs):
            Image.fromarray(im).save(d / f"ref{k}.png")
        kind = "edit" if req["mode"] == "edit" else ("product" if req.get("has_product_image") else "new")
        meta = {"id": run.name, "kind": kind, "prompt_en": (plan or {}).get("prompt_en") or brief["prompt_en"],
                "seed": brief.get("seed"), "W": int(poster.shape[1]), "H": int(poster.shape[0]),
                "texts": [t["text"] for t in brief.get("texts") or []],
                "sources": [t.get("source") or "user" for t in brief.get("texts") or []],   # "ai" = câu LLM brief tự viết
                "refs": [f"ref{k}.png" for k in range(len(refs))],
                "root": root, "mask_frac": round(float(mask.mean()), 4), "keep_frac": round(float(km.mean()), 4),
                "request": req.get("prompt")}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        manifests[split].append(meta)
        why["GIỮ"] += 1
        by_lines.setdefault(min(len(meta["texts"]), 10), [0, 0])[0] += 1
    for split, rows in manifests.items():
        (out / f"manifest_{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                     encoding="utf-8")
    kinds = Counter((s, r["kind"]) for s, rows in manifests.items() for r in rows)
    print("lý do:", dict(why.most_common()))
    print("mẫu:", dict(sorted(kinds.items())))
    report_texts([r for rows in manifests.values() for r in rows])
    print("tỉ lệ đạt theo số dòng chữ:", ", ".join(f"{k}{'+' if k == 10 else ''} dòng {v[0]}/{v[1]}"
                                                   for k, v in sorted(by_lines.items())))


def report_texts(rows: list[dict], top: int = 20) -> dict:
    """Thống kê câu chữ của bộ dữ liệu: tỉ lệ câu LLM brief tự viết (source "ai") và các câu lặp nhiều nhất. Vài câu AI chung
    chung ("Đặt lịch ngay", "Điểm nổi bật") chiếm quá nhiều dòng -> LoRA dồn sức học vào chúng; khi đó mới hạ trọng số mẫu."""
    lines = [(t, s) for r in rows for t, s in zip(r["texts"], r.get("sources") or ["user"] * len(r["texts"]))]
    if not lines:
        return {}
    n_ai = sum(s == "ai" for _, s in lines)
    common = Counter(t for t, _ in lines).most_common(top)
    share = sum(c for _, c in common[:10]) / len(lines)
    print(f"câu chữ: {len(lines)} dòng, {100 * n_ai / len(lines):.0f}% do LLM brief tự viết; 10 câu lặp nhiều nhất chiếm "
          f"{100 * share:.0f}% số dòng" + ("  <-- LỆCH: cân nhắc hạ trọng số các mẫu này" if share > 0.3 else ""))
    print("lặp nhiều nhất:", ", ".join(f"{t!r} x{c}" for t, c in common))
    return {"lines": len(lines), "ai_share": n_ai / len(lines), "top10_share": share}


if __name__ == "__main__":
    main()
