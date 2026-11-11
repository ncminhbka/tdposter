#!/usr/bin/env python3
"""MÁY CHỦ: chạy một TẬP DEV (bench/chat1.json: lời chat như người dùng gõ cho chatbot) qua API v1 của máy chủ đang chạy TẠI
CHỖ -- đủ đường đi thật: LLM -> nháp -> nét chữ -> dòng -> gỡ chữ + VLM -> dựng -> sửa bằng code. Tệp của từng lượt (TEXTFIX_RUNS,
header X-Run-Id) nằm ngay trên máy này; xong thì BÁO CÁO TẠI CHỖ (scripts/batch_report.py): bảng ra terminal (so với lượt
trước), trang HTML, một zip nhẹ vài MB. Thêm --full để gói đầy đủ thành các zip < 24 MB.

  bash scripts/start_server.sh                    # máy chủ phải chạy sẵn (cổng TEXTFIX_PORT, mặc định 8088)
  python scripts/dev_batch.py bench/chat1.json --fresh        # cả tập, gửi -j 4 yêu cầu cùng lúc
  python scripts/dev_batch.py bench/chat1.json --only c03,c07 --fresh
  python scripts/dev_batch.py bench/chat1.json                # chạy lại: ca đã xong trong manifest thì bỏ qua

Manifest: output/dev/<tên tập>_manifest.json {ca: {run_id, status, http, error, elapsed, timing}} -- gói kèm vào zip.
Đo tốc độ API public (không có tệp lượt chạy): scripts/v1_smoke.py --url https://<cổng>/textfix (in X-Timing từng ca).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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


def timing_of(header: str | None) -> dict:
    out = {}
    for kv in (header or "").split(";"):
        k, _, v = kv.partition("=")
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def run_case(url: str, key: str | None, c: dict, timeout: float) -> dict:
    """Một ca -> {run_id, status, http, error, elapsed, timing}. Ảnh trả về không cần (đã có trong thư mục lượt chạy)."""
    body = {"prompt": c["request"]["prompt"], "aspect_ratio": c["request"].get("aspect_ratio", "1:1")}
    if c.get("image"):
        body["images_base64"] = [base64.b64encode((ROOT / c["image"]).read_bytes()).decode()]
    h = {"Content-Type": "application/json", "Accept": "image/png", **({"x-api-key": key} if key else {})}
    req = urllib.request.Request(url + PATH, data=json.dumps(body, ensure_ascii=False).encode(), headers=h, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            st, hdr, err = r.status, dict(r.headers), None
    except urllib.error.HTTPError as e:
        st, hdr = e.code, dict(e.headers)
        try:
            j = json.loads(e.read())
            err = f"{j.get('error_code')}: {j.get('message')}"
        except ValueError:
            err = f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return {"status": "error", "http": 0, "error": f"mạng: {e}", "elapsed": round(time.time() - t0, 1)}
    g = lambda k: hdr.get(k) or hdr.get(k.lower())   # noqa: E731
    return {"run_id": g("X-Run-Id"), "status": "done" if st == 200 else "error", "http": st, "error": err,
            "elapsed": round(time.time() - t0, 1), "timing": timing_of(g("X-Timing")), "tags": c.get("tags")}


def main(argv: list[str] | None = None) -> int:
    from textfix.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?", type=Path, default=ROOT / "bench" / "chat1.json")
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('TEXTFIX_PORT', '8088')}")
    ap.add_argument("--only", default="", help="chỉ các ca này (phẩy)")
    ap.add_argument("-j", type=int, default=4, help="số yêu cầu gửi cùng lúc")
    ap.add_argument("--timeout", type=float, default=180, help="giây chờ mỗi yêu cầu")
    ap.add_argument("--fresh", action="store_true", help="CHẠY LẠI các ca được chọn dù manifest ghi đã xong (sau khi cập nhật mã): "
                                                         "manifest cũ lưu thành <tên>_manifest_<giờ>.json")
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--full", action="store_true", help="gói thêm ĐẦY ĐỦ kết quả (nháp, bản xoá cỡ gốc) thành các zip < 24 MB")
    a = ap.parse_args(argv)
    a.url = a.url.rstrip("/").removesuffix("/api")
    key = os.environ.get("TEXTFIX_API_KEY") or os.environ.get("x-api-key")
    cases = [c for c in json.loads(a.set.read_text(encoding="utf-8")) if (c.get("request") or {}).get("prompt")]
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        cases = [c for c in cases if c["id"] in keep]
    name = a.set.stem
    mf = ROOT / "output" / "dev" / f"{name}_manifest.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    M = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    save = lambda: mf.write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")   # noqa: E731
    if a.fresh and any(c["id"] in M for c in cases):
        old = mf.with_name(f"{name}_manifest_{time.strftime('%m%d_%H%M%S')}.json")
        old.write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
        for c in cases:
            M.pop(c["id"], None)
        save()
        print(f"--fresh: quên {len(cases)} ca trong manifest (bản cũ: {old.name}), gửi lại")
    try:
        with urllib.request.urlopen(a.url + "/api/health", timeout=10) as r:
            h = json.loads(r.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"KHÔNG gọi được máy chủ {a.url} ({e}) -- chạy: bash scripts/start_server.sh, chờ /api/health ready")
        return 1
    if not h.get("ready"):
        print("máy chủ chưa sẵn sàng:", "lỗi nạp model (xem server.log)" if h.get("failed") else "đang nạp model, chờ rồi chạy lại")
        return 1
    todo = [c for c in cases if (M.get(c["id"]) or {}).get("status") != "done"]
    print(f"{a.url}: {len(todo)} ca, gửi {a.j} cùng lúc ...")

    def one(c):
        m = run_case(a.url, key, c, a.timeout)
        M[c["id"]] = m
        save()
        t = m.get("timing") or {}
        print(f"{c['id']}: {m['http']} {m['elapsed']:5.1f}s {m.get('run_id') or ''} " +
              (m["error"] or " ".join(f"{k}={v:g}" for k, v in t.items())))
    with ThreadPoolExecutor(max_workers=max(1, a.j)) as ex:
        list(ex.map(one, todo))
    time.sleep(3)   # ảnh debug ghi SAU khi trả poster (luồng phụ của máy chủ)
    st = {}
    for c in cases:
        s = (M.get(c["id"]) or {}).get("status")
        st[s] = st.get(s, 0) + 1
    print("trạng thái:", st)
    ts = [(M.get(c["id"]) or {}).get("timing") or {} for c in cases]
    ts = [t for t in ts if t]
    if ts:
        import statistics
        keys = list(dict.fromkeys(k for t in ts for k in t))
        print("giây trung vị:", " ".join(f"{k}={statistics.median([t[k] for t in ts if k in t]):.1f}" for k in keys))
    if a.no_ship:
        return 0
    # BÁO CÁO TẠI CHỖ (mặc định): bảng ra terminal + một trang HTML + MỘT zip nhẹ (scripts/batch_report.py)
    from scripts.batch_report import main as light   # noqa: E402
    rc = light([str(a.set)])
    if not a.full:
        print("(gói đầy đủ nháp / bản xoá cỡ gốc -- chỉ khi cần đo lại ô: thêm --full)")
        return rc
    runs = sorted({m["run_id"] for cid, m in M.items() if m.get("run_id") and cid in {c["id"] for c in cases}})
    from scripts.runs_report import main as report   # noqa: E402
    return report(["--only", ",".join(runs), "--again", "--jpg", "--extra", str(mf), "--name", name])


if __name__ == "__main__":
    raise SystemExit(main())
