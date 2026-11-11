#!/usr/bin/env python3
"""KIỂM OCR TRÊN GPU (TEXTFIX_OCR_GPU, cần onnxruntime-gpu): chạy bước đo dòng (lines.build: OCR + hình học) trên vài nháp, một
lượt CPU một lượt GPU (mỗi lượt một tiến trình riêng), in giây / nháp và so kết quả (số dòng, chữ đọc được, khung).

  python scripts/ocr_gpu_check.py                 # 8 nháp mới nhất trong TEXTFIX_RUNS (hoặc output/runs), GPU 0
  python scripts/ocr_gpu_check.py --gpu 1 -n 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def drafts(n: int) -> list[str]:
    runs = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
    fs = sorted(glob.glob(str(runs / "run_*" / "v0" / "draft.png")), key=os.path.getmtime)[-n:]
    return [f for f in fs if os.path.exists(f.replace("draft.png", "ink.png"))]


def child(files: list[str]) -> None:
    """Một lượt (CPU hoặc GPU theo TEXTFIX_OCR_GPU): -> JSON {providers, s (giây / nháp), lines}."""
    import numpy as np
    from PIL import Image

    from textfix.draw import hisam
    from textfix.lines import build, ocr
    eng = ocr._engine()
    prov = eng.text_detector.infer.session.get_providers()
    data = []
    for f in files:
        d = np.asarray(Image.open(f).convert("RGB"))
        data.append((d, hisam.from_png(f.replace("draft.png", "ink.png"), (d.shape[1], d.shape[0]))))
    build(*data[0])   # làm nóng (lượt đầu nạp / tối ưu mô hình)
    ocr._CACHE.clear()
    t, out = time.time(), []
    for d, ink in data:
        M = build(d, ink)
        out.append([[l.get("ocr"), [round(v) for v in l["box"]]] for l in M["L"]])
    print(json.dumps({"providers": prov, "s": (time.time() - t) / len(data), "lines": out}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--child", nargs="*")
    a = ap.parse_args()
    if a.child is not None:
        child(a.child)
        return 0
    import onnxruntime as ort
    print("onnxruntime", ort.__version__, "| thiết bị", ort.get_device(), "| providers", ort.get_available_providers())
    files = drafts(a.n)
    if not files:
        print("không thấy nháp nào (cần TEXTFIX_RUNS/run_*/v0/draft.png + ink.png: chạy máy chủ vài yêu cầu trước)")
        return 1
    res = {}
    for name, gpu in (("CPU", ""), ("GPU", a.gpu)):
        env = {**os.environ, "TEXTFIX_OCR_GPU": gpu, "PYTHONIOENCODING": "utf-8"}
        p = subprocess.run([sys.executable, __file__, "--child", *files], env=env, capture_output=True, text=True, errors="replace",
                           encoding="utf-8")
        line = next((x for x in p.stdout.splitlines()[::-1] if x.startswith("{")), None)
        if p.returncode or not line:
            print(f"{name}: LỖI\n{p.stderr[-2000:]}")
            return 1
        res[name] = json.loads(line)
        print(f"{name}: {res[name]['s']:.2f} s / nháp (đo dòng), chạy trên {res[name]['providers']}")
    c, g = res["CPU"]["lines"], res["GPU"]["lines"]
    same_n = sum(len(x) == len(y) for x, y in zip(c, g))
    same_t = sum(a_[0] == b_[0] for x, y in zip(c, g) for a_, b_ in zip(x, y))
    tot = sum(min(len(x), len(y)) for x, y in zip(c, g))
    shift = max((max(abs(p - q) for p, q in zip(a_[1], b_[1])) for x, y in zip(c, g) for a_, b_ in zip(x, y)), default=0)
    print(f"{len(files)} nháp: cùng số dòng {same_n}/{len(files)}, cùng chữ đọc được {same_t}/{tot} dòng, khung lệch tối đa {shift} px")
    if "CUDAExecutionProvider" not in res["GPU"]["providers"]:
        print("GPU KHÔNG chạy (onnxruntime lùi về CPU): xem hướng dẫn cài onnxruntime-gpu")
        return 1
    print(f"nhanh hơn {res['CPU']['s'] / max(1e-6, res['GPU']['s']):.1f} lần")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
