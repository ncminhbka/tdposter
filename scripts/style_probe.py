#!/usr/bin/env python3
"""DỰNG LẠI TRÊN CÙNG NHÁP, NHIỀU BIẾN THỂ VLM (12/11): các poster vừa chạy, cùng nháp + mặt nạ nét máy chủ đã dùng, gọi lại VLM
(không đọc bộ đệm) dưới mỗi biến thể rồi dựng. Chạy trên máy có VLM + TEXTFIX_RUNS của máy chủ, ngay sau scripts/edit_rounds.py
(máy chủ xoá lượt cũ sau TEXTFIX_MAX_RUNS):

  python scripts/style_probe.py output/edit_rounds/<giờ>                  # lượt "gốc" trong report.json (bỏ _r2, _r3 ...)
  python scripts/style_probe.py output/edit_rounds/<giờ> --variants base,compact

Biến thể (compose/vlm.py, mặc định đều TẮT trên máy chủ):
  base     lời dặn hiện hành, gọi lại -> độ lệch tự nhiên giữa hai lần gọi VLM (mốc để so các biến thể khác)
  enhance  TEXTFIX_EFFECT=enhance: VLM được THÊM một hiệu ứng nhẹ cho tiêu đề / giá / ưu đãi (thẩm mỹ)
  compact  TEXTFIX_VLM_COMPACT=1: bỏ trường rỗng trong câu trả lời (ít token sinh ra -> nhanh hơn?)
  slots    TEXTFIX_VLM_IMAGES=slots: chỉ gửi ảnh có khung L# (nửa token ảnh -> nhanh hơn?)
Ra output/style_probe/<...>/: <ca>.jpg (máy chủ | các biến thể), probe.json (từng ca), summary.txt (giây / token VLM, cách gán
khớp lời gọi của máy chủ bao nhiêu phần trăm dòng, lỗi bộ kiểm) và MỘT zip trong output/runs_zip/.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stt
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

VARIANTS = {"base": {}, "enhance": {"TEXTFIX_EFFECT": "enhance"}, "compact": {"TEXTFIX_VLM_COMPACT": "1"},
            "slots": {"TEXTFIX_VLM_IMAGES": "slots"}}
KEYS = sorted({k for v in VARIANTS.values() for k in v})


def strip(ims: list, labels: list[str], path: Path, h: int = 480) -> None:
    from PIL import Image, ImageDraw
    ims = [(Image.fromarray(x) if not isinstance(x, Image.Image) else x).convert("RGB") for x in ims]
    ims = [im.resize((round(im.width * h / im.height), h)) for im in ims]
    w = Image.new("RGB", (sum(im.width for im in ims) + 10 * (len(ims) - 1), h + 22), "white")
    d, x = ImageDraw.Draw(w), 0
    for im, lb in zip(ims, labels):
        w.paste(im, (x, 22))
        d.text((x + 4, 4), lb, fill=(0, 0, 0))
        x += im.width + 10
    w.save(path, quality=82)


def styles_of(P: dict) -> dict:
    """Hiệu ứng (css) VLM ghi cho mỗi vai (plan.json "style")."""
    st = P.get("style") if isinstance(P.get("style"), dict) else {}
    return {r: (v or {}).get("effect") for r, v in st.items()}


def fingerprint(P: dict) -> dict:
    """Cách gán của một lần VLM: L# -> (loại, câu khách)."""
    out = {}
    for u in P.get("assign") or []:
        for i in u.get("marks") or []:
            out[str(i)] = (u.get("kind"), tuple(sorted(int(c) for c in u.get("clients") or [])))
    return out


def agree(a: dict, b: dict) -> float | None:
    ks = set(a) | set(b)
    return round(sum(a.get(k) == b.get(k) for k in ks) / len(ks), 3) if ks else None


def vlm_calls(calls: list[dict]) -> dict:
    c = [x for x in calls if x.get("tag") == "designer" and not x.get("cached")]
    return {"s": sum(x.get("s") or 0 for x in c), "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in c),
            "completion_tokens": sum(x.get("completion_tokens") or 0 for x in c), "attempts": sum(x.get("attempts") or 1 for x in c)}


def main(argv: list[str] | None = None) -> int:
    from textfix.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--device", default=os.environ.get("TEXTFIX_PROBE_DEVICE", ""))
    a = ap.parse_args(argv)
    names = [v.strip() for v in a.variants.split(",") if v.strip() in VARIANTS]

    import numpy as np
    from PIL import Image

    from textfix import config, erase
    from textfix.draw import hisam
    from textfix.engine import design
    from textfix.llm import collect, json_call
    from textfix.render.browser import Browser
    try:   # LaMa một lần mỗi poster cho mọi biến thể (cùng nháp, cùng mặt nạ)
        import golden
        erase.inpaint = golden.cached_inpaint(erase.inpaint)
    except Exception:
        pass

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
    vlm0 = json_call(m["vlm"]["url"], m["vlm"]["model"], m["vlm"]["key_env"], tag="designer")

    def vlm(system, parts, meta=None, fresh=False, schema=None):   # luôn gọi thật: đo giây / token, không lấy câu trả lời cũ
        return vlm0(system, parts, meta, True, schema)

    stamp = time.strftime("%m%d_%H%M%S")
    out = ROOT / "output" / "style_probe" / f"{a.report_dir.name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    res, bad = {}, 0
    print(f"{len(todo)} poster x {names}, LaMa trên {dev}, VLM {m['vlm']['model']}")
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
            except Exception as e:   # lượt đã bị máy chủ xoá / thiếu tệp
                bad += 1
                print(f"{cid} {run_id}: thiếu tệp {type(e).__name__}: {e}")
                continue
            f0 = fingerprint(P0)
            row = {"run_id": run_id, "server": {"effects": styles_of(P0), "errors": len(P0.get("errors") or []),
                                                "missing": len(P0.get("missing") or []), **vlm_calls(P0.get("calls") or [])}}
            ims, labels, fps = [old], ["server"], {}
            for v in names:
                for k in KEYS:
                    os.environ.pop(k, None)
                os.environ.update(VARIANTS[v])
                try:
                    with collect() as calls:
                        r = design({"draft": draft, "ink": ink, "timing": {}, "device": dev}, Bf, vlm, B.page,
                                   product=bool(req.get("has_product_image")), debug=False)
                except Exception as e:
                    row[v] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    print(f"{cid} {v}: LỖI {row[v]['error']}")
                    continue
                P = r["plan"]
                fps[v] = fingerprint(P)
                row[v] = {"effects": styles_of(P), "agree_server": agree(fingerprint(P), f0), "errors": len(P.get("errors") or []),
                          "missing": len(P.get("missing") or []), **vlm_calls(calls)}
                ims.append(r["poster"])
                labels.append(v)
            for k in KEYS:
                os.environ.pop(k, None)
            for v in fps:   # biến thể so với base (cùng lúc gọi lại): lệch hơn base-vs-máy chủ mới là do biến thể
                if v != "base" and "base" in fps:
                    row[v]["agree_base"] = agree(fps[v], fps["base"])
            res[cid] = row
            strip(ims, labels, out / f"{cid}.jpg")
            print(cid, {v: (row[v].get("s"), row[v].get("completion_tokens"), row[v].get("agree_server")) for v in names if v in row})
    lines = [f"{len(res)}/{len(todo)} poster (thiếu tệp {bad}); VLM: giây | token đọc vào | token sinh ra | cách gán khớp máy chủ"
             " (tỉ lệ dòng cùng loại + cùng câu) | lỗi bộ kiểm | câu thiếu -- TRUNG VỊ / tổng"]
    for v in ["server"] + names:
        rs = [row[v] for row in res.values() if v in row and "error" not in row[v]]
        if not rs:
            continue
        med = lambda k: stt.median([x[k] for x in rs if x.get(k) is not None]) if any(x.get(k) is not None for x in rs) else None   # noqa: E731
        lines.append(f"{v:8s} n={len(rs):2d}  s {med('s')}  in {med('prompt_tokens')}  out {med('completion_tokens')}  "
                     f"khớp {med('agree_server')} / base {med('agree_base')}  lỗi {sum(x['errors'] for x in rs)}  thiếu {sum(x['missing'] for x in rs)}  "
                     f"hiệu ứng {sum(any(e for e in x['effects'].values()) for x in rs)}/{len(rs)}"
                     + f"  hỏng {sum('error' in row.get(v, {}) for row in res.values())}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    (out / "probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "output" / "runs_zip").mkdir(parents=True, exist_ok=True)
    z = ROOT / "output" / "runs_zip" / f"style_probe_{a.report_dir.name}_{stamp}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.iterdir():
            zf.write(f, f.name)
    print(f"zip: {z} ({z.stat().st_size / 2 ** 20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
