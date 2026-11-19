"""BƯỚC 2 -- gọi API máy chủ TextFix để sinh các lượt chạy (đầu vào của lora.collect).

  bash lora/serve_flux4b.sh                       # máy chủ: FLUX 4B tại chỗ, lưu MỌI lượt vào data/lora/runs (không xoá)
  python -m lora.drive --requests data/lora/requests.jsonl --url http://127.0.0.1:8088 -j 6
  python -m lora.drive --bench "bench/*.json" -j 8          # pilot nhanh: 376 yêu cầu thật + 79 lời sửa có sẵn trong bench/

Mỗi yêu cầu:
  kind=poster   tạo poster (kiểu 1); rồi với xác suất --edit-p: LLM viết MỘT lời sửa từ chữ đang có -> gửi lại PNG + lời sửa (kiểu 3)
  kind=product  vẽ ảnh sản phẩm KHÔNG chữ (máy chủ trả ảnh FLUX khi brief không có chữ) -> poster cho ảnh đó (kiểu 2) -> sửa
Ghi data/lora/drive.jsonl (mỗi lượt gọi một dòng: id, bước, run_id, mã trả về, giây) -- chạy lại thì bỏ các id đã xong.
Máy chủ tự lưu mọi tệp cần (request.json, brief.json, v0/draft.png, v0/poster.png, product.png, source.png) trong TEXTFIX_RUNS.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EDIT_SYSTEM = ("Bạn là chủ cửa hàng vừa nhận poster. Viết MỘT tin nhắn ngắn tiếng Việt có dấu yêu cầu sửa poster đó, như chat "
               "thật. Chọn đúng một kiểu được giao. Khi đổi / thêm chữ thì ghi NGUYÊN VĂN chữ mới.")
EDIT_KINDS = ["đổi một con số (giá, giờ, ngày, phần trăm giảm) thành số khác",
              "thêm một dòng chữ mới (hotline, địa chỉ, giờ mở cửa, khẩu hiệu ngắn)",
              "bỏ một dòng chữ đang có",
              "viết lại tiêu đề thành câu khác cùng ý",
              "đổi màu nền hoặc tông màu poster, giữ nguyên chữ",
              "đổi đồ vật / món trong ảnh thành thứ khác cùng loại, giữ nguyên chữ",
              "sửa một từ trong một dòng chữ"]
EDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["request"],
               "properties": {"request": {"type": "string"}}}
_LOCK = threading.Lock()


API_RATIOS = ("1:1", "9:16", "16:9", "3:4", "4:3")
NEAREST = {"4:5": "3:4", "2:3": "9:16", "5:4": "4:3", "3:2": "16:9"}


def bench_requests(pattern: str) -> list[dict]:
    """Yêu cầu từ các tệp bench: {request: {prompt, aspect_ratio, image_file?}} hoặc {base: {...}, edits: [{prompt, image_file?}]}.
    Bỏ ca chỉ có form (không prompt), ca ảnh lạ (source_file, base null) và tệp prompt mức FLUX. Tỉ lệ ngoài 5 tỉ lệ API -> gần
    nhất; không ghi -> chọn theo id (cố định, trộn đủ tỉ lệ). Ca có `edits` -> đúng các lời sửa đó, nối tiếp, thay lời sửa LLM."""
    import glob
    import hashlib
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for x in data if isinstance(data, list) else []:
            if not isinstance(x, dict):
                continue
            r = x.get("request") if isinstance(x.get("request"), dict) else x.get("base")
            if not isinstance(r, dict) or not str(r.get("prompt") or "").strip():
                continue
            rid = hashlib.sha1(f"{Path(f).stem}:{x.get('id')}:{r['prompt']}".encode()).hexdigest()[:10]
            ar = r.get("aspect_ratio")
            ar = ar if ar in API_RATIOS else NEAREST.get(ar) or API_RATIOS[int(rid, 16) % len(API_RATIOS)]
            row = {"id": rid, "kind": "bench", "prompt": r["prompt"].strip(), "aspect": ar, "source": f"{Path(f).stem}/{x.get('id')}"}
            if r.get("image_file"):
                row["image_file"] = r["image_file"]
            if "edits" in x:
                row["edits"] = [{"prompt": e["prompt"], **({"image_file": e["image_file"]} if e.get("image_file") else {})}
                                for e in x.get("edits") or [] if isinstance(e, dict) and e.get("prompt")]
            rows.append(row)
    return rows


def post(url: str, key: str | None, prompt: str, aspect: str, images: list[bytes] | None = None, timeout: float = 180) -> dict:
    from v1_smoke import call
    body = {"prompt": prompt, "aspect_ratio": aspect}
    if images:
        body["images_base64"] = [base64.b64encode(b).decode() for b in images]
    return call(url, key, body, True, timeout)


def texts_of(png: bytes) -> list[str]:
    from textfix.meta import read
    return [t["text"] for t in (read(png) or {}).get("texts") or []]


def mode_of(png: bytes) -> str | None:
    from textfix.meta import read
    return (read(png) or {}).get("mode")


def edit_request(llm, texts: list[str], kind: str) -> str:
    msg = ("Chữ đang có trên poster:\n" + "\n".join(f"- {t}" for t in texts) + f"\n\nKiểu sửa được giao: {kind}.\n"
           "Trả JSON {\"request\": ...}.")
    return str((llm(EDIT_SYSTEM, [msg], schema=EDIT_SCHEMA, fresh=True) or {}).get("request") or "").strip()


def run_one(r: dict, a, url: str, key: str | None, llm, log) -> None:
    rng = random.Random(r["id"])

    def rec(step: str, res: dict, **kw) -> None:
        row = {"id": r["id"], "step": step, "kind": r["kind"], "run_id": res.get("run_id"), "http": res.get("http"),
               "s": res.get("s"), **kw}
        if not res.get("image"):
            row["error"] = res.get("error_code")
        with _LOCK:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()

    images = None
    if r.get("image_file"):   # ảnh kèm theo: bench/refs/*.jpg hoặc ảnh sản phẩm thật (lora.requests --products)
        images = [(ROOT / r["image_file"]).read_bytes()]
    elif r["kind"] == "product":
        res = post(url, key, r["product_prompt"], "1:1", timeout=a.timeout)
        ok = bool(res.get("image")) and mode_of(res["image"]) == "image"
        rec("product", res, ok=ok)
        if not ok:   # máy chủ thêm chữ vào ảnh sản phẩm -> bỏ yêu cầu này
            return
        images = [res["image"]]
    res = post(url, key, r["prompt"], r["aspect"], images, timeout=a.timeout)
    rec("poster", res)
    png = res.get("image")
    if r.get("edits") is not None:   # bench: đúng các lời sửa có sẵn, nối tiếp
        for k, e in enumerate(r["edits"], 1):
            if not png:
                return
            imgs = [png] + ([(ROOT / e["image_file"]).read_bytes()] if e.get("image_file") else [])
            res = post(url, key, e["prompt"], r["aspect"], imgs, timeout=a.timeout)
            rec(f"edit{k}", res, request=e["prompt"])
            png = res.get("image")
        return
    for depth in range(a.edit_depth):
        if not png or rng.random() >= a.edit_p:
            return
        texts = texts_of(png)
        if not texts:
            return
        try:
            req = edit_request(llm, texts, rng.choice(EDIT_KINDS))
        except Exception as e:
            rec(f"edit{depth + 1}", {"http": 0, "error_code": f"LLM: {type(e).__name__}"})
            return
        if not req:
            return
        res = post(url, key, req, r["aspect"], [png], timeout=a.timeout)
        rec(f"edit{depth + 1}", res, request=req)
        png = res.get("image")


def main() -> None:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="data/lora/requests.jsonl")
    ap.add_argument("--bench", default=None, help='glob tệp bench, vd "bench/*.json" -- dùng thay --requests')
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('TEXTFIX_PORT', '8088')}")
    ap.add_argument("--log", default="data/lora/drive.jsonl")
    ap.add_argument("-j", type=int, default=6, help="số yêu cầu cùng lúc (máy chủ nhận tối đa TEXTFIX_MAX_QUEUE)")
    ap.add_argument("--edit-p", type=float, default=0.35, help="xác suất sửa sau mỗi poster (0,35 + 2 lần -> ~32% lượt là sửa)")
    ap.add_argument("--edit-depth", type=int, default=2, help="số lần sửa nối tiếp tối đa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800, help="bằng hạn của máy chủ dữ liệu (serve_flux4b.sh: 1800 s)")
    ap.add_argument("--products", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.products:   # cũ: bốc ảnh ngẫu nhiên -> yêu cầu ("poster nước mắm") lệch ảnh (Oreo). Yêu cầu phải viết cho đúng ảnh.
        raise SystemExit("--products đã bỏ: sinh yêu cầu gắn với từng ảnh bằng "
                         "`python -m lora.requests --products <thư mục> --out <tệp>` rồi `lora.drive --requests <tệp>`")
    from textfix import config
    from textfix.llm import json_call
    from v1_smoke import PATH
    config.load_env()
    m = config.models()["llm"]
    llm = json_call(m["url"], m["model"], m["key_env"], tag="lora_edit", temperature=1.0)
    key = os.environ.get("TEXTFIX_API_KEY") or os.environ.get("x-api-key")
    url = a.url.rstrip("/").removesuffix("/api") + PATH
    rows = bench_requests(a.bench) if a.bench else \
        [json.loads(x) for x in Path(a.requests).read_text(encoding="utf-8").splitlines() if x.strip()]
    logp = Path(a.log)
    logp.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["id"] for x in logp.read_text(encoding="utf-8").splitlines() if x.strip()} if logp.exists() else set()
    todo = [r for r in rows if r["id"] not in done][: a.limit or None]
    print(f"{len(todo)} yêu cầu ({len(done)} đã xong) -> {url}, {a.j} cùng lúc")
    with logp.open("a", encoding="utf-8") as log, ThreadPoolExecutor(a.j) as ex:
        for k, _ in enumerate(ex.map(lambda r: run_one(r, a, url, key, llm, log), todo), 1):
            if k % 20 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
