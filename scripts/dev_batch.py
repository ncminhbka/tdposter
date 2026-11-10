#!/usr/bin/env python3
"""MÁY CHỦ: chạy một TẬP DEV (bench/dev2.json: yêu cầu như người dùng điền trên UI) qua máy chủ đang chạy -- đủ đường đi thật:
LLM brief -> nháp -> xoá -> dọn -> ô -> VLM -> dựng -> sửa bằng code -- rồi gói ĐẦY ĐỦ kết quả (request, brief, ảnh sản phẩm, nháp,
bản xoá thô / đã dọn, lượt đầu, poster, plan.json). Xong thì BÁO CÁO TẠI CHỖ (scripts/batch_report.py): bảng kết quả ra terminal
(có so với lượt trước), một trang HTML xem qua máy chủ, một zip nhẹ vài MB. Thêm --full để gói đầy đủ thành các zip < 24 MB.

  bash scripts/start_server.sh                    # máy chủ phải chạy sẵn (cổng TEXTFIX_PORT, mặc định 8088)
  %run scripts/dev_batch.py                       # bench/dev2.json, gửi hết một lần (máy chủ tự xếp hàng, 2 GPU gối đầu)
  %run scripts/dev_batch.py --only d05,d19        # vài ca
  %run scripts/dev_batch.py                       # chạy lại sau khi notebook đứt: đọc manifest, chờ tiếp các job cũ, KHÔNG gửi lại
  %run scripts/dev_batch.py bench/dev3.json --only g01,g02 --fresh   # sau khi cập nhật mã: chạy LẠI các ca đã có trong manifest
  python scripts/dev_batch.py bench/chat1.json --url https://<máy chủ>/textfix --fetch   # API public: khoá x-api-key
                                                  # trong .env (hoặc TEXTFIX_API_KEY), tải zip debug từng lượt về output/runs/

Manifest: output/dev/<tên tập>_manifest.json {ca: {run_id, job_id, status, error, elapsed}} -- gói kèm vào zip (ca -> lượt chạy).
Zip: output/runs_zip/<tên tập>_<giờ>_NN.zip, mỗi ảnh (v<i>/) nằm gọn một zip; giải nén vào output/runs/ ở máy cá nhân.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)


HEADERS = {"Content-Type": "application/json"}   # + x-api-key khi gọi API public (--url https://..., khoá trong .env)


def _call(url: str, body: dict | None = None, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers=HEADERS, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _fetch_run(url: str, run_id: str, runs: Path) -> str:
    """Máy chủ ở xa: tải zip debug của lượt chạy (GET /api/runs/<run_id>/zip) và giải nén vào runs/ -- batch_report đọc như lượt
    chạy tại chỗ (poster là JPG trong zip, batch_report nhận cả hai)."""
    import io
    import zipfile
    if (runs / run_id).is_dir():
        return "đã có"
    req = urllib.request.Request(f"{url}/api/runs/{run_id}/zip", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(runs)
    return "đã tải"


def main(argv: list[str] | None = None) -> int:
    from textfix.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?", type=Path, default=ROOT / "bench" / "dev2.json")
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('TEXTFIX_PORT', '8088')}")
    ap.add_argument("--only", default="", help="chỉ các ca này (phẩy)")
    ap.add_argument("--timeout", type=float, default=3 * 3600, help="giây chờ tối đa cả tập")
    ap.add_argument("--resubmit", action="store_true", help="gửi lại các ca lỗi / mất job (máy chủ khởi động lại)")
    ap.add_argument("--fresh", action="store_true", help="CHẠY LẠI các ca được chọn dù manifest ghi đã xong (sau khi cập nhật mã): "
                                                         "manifest cũ lưu thành <tên>_manifest_<giờ>.json, lượt chạy cũ vẫn còn trong output/runs")
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--full", action="store_true", help="gói thêm ĐẦY ĐỦ kết quả (nháp, bản xoá cỡ gốc) thành các zip < 24 MB như trước")
    ap.add_argument("--fetch", action="store_true", help="máy chủ ở xa (API public): tải zip debug từng lượt về output/runs/")
    a = ap.parse_args(argv)
    a.url = a.url.rstrip("/").removesuffix("/api")
    key = os.environ.get("TEXTFIX_API_KEY") or os.environ.get("x-api-key")   # .env: x-api-key=... (load_env nạp mọi khoá)
    if key:
        HEADERS["x-api-key"] = key
    cases = json.loads(a.set.read_text(encoding="utf-8"))
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
        h = _call(a.url + "/api/health", timeout=10)
    except (urllib.error.URLError, OSError) as e:
        print(f"KHÔNG gọi được máy chủ {a.url} ({e}) -- chạy: bash scripts/start_server.sh, chờ /api/health ready")
        return 1
    if not h.get("ready"):
        print("máy chủ chưa sẵn sàng:", h.get("error") or "đang nạp model, chờ rồi chạy lại")
        return 1
    print(f"máy chủ sẵn sàng: GPU {h.get('gpus')}, VLM {((h.get('models') or {}).get('vlm') or {}).get('model')}")

    n_img = 0
    for c in cases:   # GỬI (ca đã có job thì bỏ qua -- chạy lại script = chờ tiếp)
        m = M.get(c["id"])
        if m and not (a.resubmit and m.get("status") in ("error", "lost")):
            continue
        body = dict(c["request"])
        if c.get("image"):
            body["image_base64"] = base64.b64encode((ROOT / c["image"]).read_bytes()).decode()
        try:
            r = _call(a.url + "/api/jobs", body)
        except urllib.error.HTTPError as e:
            M[c["id"]] = {"status": "error", "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
            print(f"{c['id']}: LỖI gửi {M[c['id']]['error']}")
            continue
        M[c["id"]] = {"job_id": r["job_id"], "run_id": r["run_id"], "status": "queued", "tags": c.get("tags"),
                      "num_images": body.get("num_images", 1), "submitted": time.time()}
        n_img += int(body.get("num_images") or 1)
        print(f"{c['id']}: gửi -> {r['run_id']}")
        save()
    print(f"đã gửi {n_img} ảnh mới; chờ xong ...")

    t0, last = time.time(), ""
    while time.time() - t0 < a.timeout:   # CHỜ
        busy = 0
        for cid in [c["id"] for c in cases]:
            m = M.get(cid)
            if not m or m.get("status") in ("done", "error", "lost"):
                continue
            try:
                j = _call(f"{a.url}/api/jobs/{m['job_id']}", timeout=30)
            except urllib.error.HTTPError as e:
                if e.code == 404:   # máy chủ khởi động lại: job mất (ảnh đã xong vẫn còn trong output/runs)
                    m.update(status="lost", error="máy chủ không còn job (khởi động lại?) -- --resubmit để gửi lại")
                    continue
                raise
            except (urllib.error.URLError, OSError):
                busy += 1
                continue
            m["status"] = j["status"]
            if j["status"] in ("done", "error"):
                m.update(error=j.get("error"), elapsed=j.get("elapsed"))
                print(f"{cid}: {j['status']} sau {j.get('elapsed')} s" + (f" -- {j['error']}" if j.get("error") else ""))
            else:
                busy += 1
        save()
        if not busy:
            break
        msg = f"còn {busy} ca chạy ({round(time.time() - t0)} s)"
        if msg[:12] != last[:12] or int(time.time() - t0) % 60 < 6:
            print(msg)
            last = msg
        time.sleep(5)
    st = {}
    for m in M.values():
        st[m.get("status")] = st.get(m.get("status"), 0) + 1
    print("trạng thái:", st)
    if a.fetch:
        runs = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
        runs.mkdir(parents=True, exist_ok=True)
        for c in cases:
            m = M.get(c["id"]) or {}
            if m.get("run_id"):
                try:
                    print(f"{c['id']}: {_fetch_run(a.url, m['run_id'], runs)} {m['run_id']}")
                except (urllib.error.URLError, OSError) as e:
                    print(f"{c['id']}: KHÔNG tải được {m['run_id']}: {e}")
    if a.no_ship:
        return 0
    # BÁO CÁO TẠI CHỖ (mặc định): bảng ra terminal + một trang HTML xem qua máy chủ + MỘT zip nhẹ (scripts/batch_report.py)
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
