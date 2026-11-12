#!/usr/bin/env python3
"""THỬ HIỆU ỨNG CHỮ (12/11): dựng lại các poster vừa chạy trên CÙNG nháp + mặt nạ nét, VLM dặn theo TEXTFIX_EFFECT=enhance (được
thêm một hiệu ứng nhẹ cho tiêu đề / giá / ưu đãi, compose/vlm.ENHANCE_EFFECT), đặt cạnh poster máy chủ đã trả (lời dặn hiện hành).
Chạy trên máy có VLM + TEXTFIX_RUNS của máy chủ, ngay sau scripts/edit_rounds.py (máy chủ xoá lượt cũ sau TEXTFIX_MAX_RUNS):

  python scripts/style_probe.py output/edit_rounds/<giờ>        # các lượt "gốc" trong report.json (bỏ lượt _r2, _r3 ...)

Ra output/style_probe/<giờ>/: <ca>.jpg (trái: đang chạy | phải: enhance), effects.json (hiệu ứng từng vai của hai bản),
và MỘT zip trong output/runs_zip/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)


def side(a, b, path: Path, h: int = 720) -> None:
    from PIL import Image
    ims = [Image.fromarray(x) if not isinstance(x, Image.Image) else x for x in (a, b)]
    ims = [im.convert("RGB").resize((round(im.width * h / im.height), h)) for im in ims]
    w = Image.new("RGB", (ims[0].width + ims[1].width + 10, h), "white")
    w.paste(ims[0], (0, 0))
    w.paste(ims[1], (ims[0].width + 10, 0))
    w.save(path, quality=85)


def styles_of(P: dict) -> dict:
    """Hiệu ứng (css) VLM ghi cho mỗi vai (plan.json "style")."""
    st = P.get("style") if isinstance(P.get("style"), dict) else {}
    return {r: (v or {}).get("effect") for r, v in st.items()}


def main(argv: list[str] | None = None) -> int:
    from textfix.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--device", default=os.environ.get("TEXTFIX_PROBE_DEVICE", ""))
    a = ap.parse_args(argv)
    os.environ["TEXTFIX_EFFECT"] = "enhance"

    import numpy as np
    from PIL import Image

    from textfix import config
    from textfix.draw import hisam
    from textfix.engine import design
    from textfix.llm import json_call
    from textfix.render.browser import Browser

    dev = a.device
    if not dev:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    runs = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
    reps = json.loads((a.report_dir / "report.json").read_text(encoding="utf-8"))
    todo = [(rp["id"], st["run_id"]) for rp in reps if "_r" not in rp["id"]
            for st in rp["steps"][:1] if st.get("kind") == "gốc" and st.get("run_id")]
    m = config.models()
    vlm = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")
    stamp = time.strftime("%m%d_%H%M%S")
    out = ROOT / "output" / "style_probe" / f"{a.report_dir.name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    effects, bad = {}, 0
    print(f"{len(todo)} poster, LaMa trên {dev}, VLM {m['vlm']['model']} (TEXTFIX_EFFECT=enhance)")
    with Browser() as B:
        for cid, run_id in todo:
            rd = runs / run_id
            try:
                draft = np.asarray(Image.open(rd / "v0" / "draft.png").convert("RGB"))
                ink = hisam.from_png(rd / "v0" / "ink.png", (draft.shape[1], draft.shape[0]))
                Bf = json.loads((rd / "brief.json").read_text(encoding="utf-8"))
                req = json.loads((rd / "request.json").read_text(encoding="utf-8"))
                P0 = json.loads((rd / "v0" / "plan.json").read_text(encoding="utf-8"))
                old = Image.open(rd / "v0" / "poster.png")
                r = design({"draft": draft, "ink": ink, "timing": {}, "device": dev}, Bf, vlm, B.page,
                           product=bool(req.get("has_product_image")), debug=False)
            except Exception as e:   # lượt đã bị máy chủ xoá / thiếu tệp / VLM hỏng: ghi lại, chạy ca sau
                bad += 1
                print(f"{cid} {run_id}: LỖI {type(e).__name__}: {e}")
                continue
            side(old, r["poster"], out / f"{cid}.jpg")
            effects[cid] = {"run_id": run_id, "draft": styles_of(P0), "enhance": styles_of(r["plan"])}
            print(f"{cid}: {effects[cid]['enhance']}")
    (out / "effects.json").write_text(json.dumps(effects, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "output" / "runs_zip").mkdir(parents=True, exist_ok=True)
    z = ROOT / "output" / "runs_zip" / f"style_probe_{a.report_dir.name}_{stamp}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.iterdir():
            zf.write(f, f.name)
    print(f"xong {len(effects)}/{len(todo)} (lỗi {bad}) | zip: {z} ({z.stat().st_size / 2 ** 20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
