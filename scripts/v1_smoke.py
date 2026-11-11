#!/usr/bin/env python3
"""THỬ API v1 (POST /api/v1/images/generations, docs/API_V1.md): gửi N yêu cầu CÙNG LÚC, in mã trả về / giây / cỡ ảnh / giây từng bước (X-Timing) từng ca,
tỉ lệ quá hạn (504) và bận (503). Lời chat lấy từ một tập bench (mặc định bench/chat1.json, chỉ dùng "prompt").

  python scripts/v1_smoke.py --url http://127.0.0.1:8088 -n 3                     # máy chủ tại chỗ (notebook)
  python scripts/v1_smoke.py --url ... -n 2 --png --bad                            # nhận PNG thô; thêm các ca cố tình sai
  python scripts/v1_smoke.py --url ... -n 1 --image-file bench/refs/coca_cola_can.jpg   # + một ca gửi ảnh base64
Khoá x-api-key: TEXTFIX_API_KEY hoặc dòng x-api-key=... trong .env. Ảnh lưu ở output/v1_smoke/<giờ>/.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

PATH = "/api/v1/images/generations"
RATIOS = ["1:1", "9:16", "16:9", "3:4", "4:3"]
BAD = [("prompt rỗng", {"prompt": "  "}, "ERR_EMPTY_PROMPT"),
       ("tỉ lệ lạ", {"prompt": "poster cà phê", "aspect_ratio": "5:7"}, "ERR_INVALID_ASPECT_RATIO"),
       ("thân sai khuôn", {"prompt": ["x"]}, "ERR_INVALID_REQUEST")]


def call(url: str, key: str | None, body: dict, png: bool, timeout: float) -> dict:
    h = {"Content-Type": "application/json", "Accept": "image/png" if png else "application/json"}
    if key:
        h["x-api-key"] = key
    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data, status, hdr = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        data, status, hdr = e.read(), e.code, dict(e.headers)
    except (urllib.error.URLError, OSError) as e:
        return {"http": 0, "error_code": f"NETWORK: {e}", "s": round(time.time() - t0, 1)}
    g = lambda k: hdr.get(k) or hdr.get(k.lower())   # noqa: E731
    out = {"http": status, "s": round(time.time() - t0, 1), "request_id": g("X-Request-Id"), "run_id": g("X-Run-Id"),
           "timing": g("X-Timing")}
    if hdr.get("Content-Type", hdr.get("content-type", "")).startswith("image/"):
        out.update(image=data, ms=hdr.get("X-Execution-Time-Ms"))
        return out
    try:
        j = json.loads(data)
    except ValueError:
        return {**out, "error_code": f"KHÔNG PHẢI JSON: {data[:120]!r}"}
    if j.get("status") == "SUCCESS":
        d = j["data"]
        out.update(image=base64.b64decode(d["image_base64"]), size=f"{d['width']}x{d['height']}", ms=d["execution_time_ms"],
                   tokens=d["tokens"])
    else:
        out.update(error_code=j.get("error_code") or j, message=(j.get("message") or "")[:160])
    return out


def main(argv: list[str] | None = None) -> int:
    from textfix.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('TEXTFIX_PORT', '8088')}")
    ap.add_argument("-n", type=int, default=3, help="số yêu cầu gửi cùng lúc")
    ap.add_argument("--set", type=Path, default=ROOT / "bench" / "chat1.json")
    ap.add_argument("--image-file", default="", help="thêm một ca gửi ảnh này dạng base64 (images_base64), vd bench/refs/coca_cola_can.jpg")
    ap.add_argument("--png", action="store_true", help="nhận PNG thô (Accept: image/png) thay JSON base64")
    ap.add_argument("--bad", action="store_true", help="thêm các ca cố tình sai, kiểm mã lỗi")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args(argv)
    url = a.url.rstrip("/").removesuffix("/api") + PATH
    key = os.environ.get("TEXTFIX_API_KEY") or os.environ.get("x-api-key")
    prompts = [c["request"]["prompt"] for c in json.loads(a.set.read_text(encoding="utf-8")) if c["request"].get("prompt")]
    cases = [(f"p{i + 1:02d}", {"prompt": prompts[i % len(prompts)], "aspect_ratio": RATIOS[i % len(RATIOS)]}, None)
             for i in range(a.n)]
    if a.image_file:
        b64 = base64.b64encode(Path(a.image_file).read_bytes()).decode()
        cases.append(("img64", {"prompt": "Poster giới thiệu sản phẩm này, nền sáng, hiện đại", "images_base64": [b64]}, None))
    if a.bad:
        cases += BAD + [("base64 hỏng", {"prompt": "poster cà phê", "images_base64": ["không phải ảnh"]}, "ERR_INVALID_IMAGE")]
    out_dir = ROOT / "output" / "v1_smoke" / time.strftime("%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{url}: gửi {len(cases)} yêu cầu cùng lúc ({'PNG thô' if a.png else 'JSON base64'}) ...")
    with ThreadPoolExecutor(max_workers=len(cases)) as ex:
        res = list(ex.map(lambda c: call(url, key, c[1], a.png, a.timeout), cases))
    ok_s, bad = [], 0
    for (name, body, want), r in zip(cases, res):
        if r.get("image"):
            (out_dir / f"{name}.png").write_bytes(r["image"])
            ok_s.append(r["s"])
        verdict = ""
        if want:
            verdict = "  ĐÚNG" if r.get("error_code") == want else f"  SAI (muốn {want})"
            bad += verdict.startswith("  SAI")
        print(f"{name:4} {r['http']:3} {r['s']:5.1f}s {r.get('size') or ''} {r.get('error_code') or 'OK'}{verdict}"
              f"  [{body.get('aspect_ratio', '1:1')}] {str(body.get('prompt'))[:50]}" + (f" -- {r['message']}" if r.get("message") else "")
              + (f"\n     {r['run_id']} {r['timing']}" if r.get("timing") else ""))
    codes = [r["http"] for r in res]
    print(f"xong {len(ok_s)}/{len(cases)} | 504: {codes.count(504)} | 503: {codes.count(503)} | khác: "
          f"{sum(c not in (200, 503, 504) for c in codes)}" + (f" | giây trung vị {statistics.median(ok_s):.1f}, max {max(ok_s):.1f}"
                                                               if ok_s else "") + f" | ảnh: {out_dir}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
