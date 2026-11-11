#!/usr/bin/env python3
"""THỬ SỬA POSTER qua API v1 (docs/API_V1.md, mục "Sửa poster"): mỗi ca tạo poster gốc (hoặc lấy ảnh có sẵn), rồi gửi LẠI ảnh
vừa nhận kèm lời sửa -- đúng như chatbot làm -- một hay nhiều lần nối tiếp. Báo cáo + MỘT zip nhẹ để gửi Claude chấm.

  python scripts/edit_rounds.py                                  # bench/edit1.json, máy chủ tại chỗ
  python scripts/edit_rounds.py --url http://127.0.0.1:8088 --only e01,e13 -j 4

Mỗi ca: base (lời chat tạo poster gốc, có thể kèm ảnh sản phẩm) hoặc source_file (ảnh lạ có sẵn); strip: bỏ khối thông tin của
poster gốc (giả ảnh lạ -- đo độ đọc chữ của VLM vì biết chữ thật); edits: các lời sửa nối tiếp, mỗi lời có thể kèm image_file
(ảnh sản phẩm mới) và expect (edit | edit_foreign | new, mặc định edit).
Kiểm tự động mỗi bước: đúng luồng (theo khối thông tin của ảnh trả về), câu cũ nào còn nguyên / đổi / mất / thêm, thời gian.
Ảnh lạ: so chữ VLM đọc (TEXTFIX_RUNS/<run>/read.json, cần chạy cùng máy với máy chủ) với chữ thật.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

from v1_smoke import PATH, call  # noqa: E402

from textfix import meta  # noqa: E402
from textfix.brief.evidence import numbers  # noqa: E402


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def strip_info(raw: bytes) -> bytes:
    """Cùng điểm ảnh, không khối thông tin (giả ảnh bị kho / ứng dụng nén lại)."""
    from PIL import Image
    b = io.BytesIO()
    Image.open(io.BytesIO(raw)).convert("RGB").save(b, "PNG")
    return b.getvalue()


def route(info: dict | None, prev: dict | None) -> str:
    """Luồng máy chủ đã chạy, đọc từ khối thông tin của ảnh trả về."""
    if not info:
        return "?"
    if info.get("parent"):
        return "edit" if prev and info["parent"] == prev.get("run_id") else "edit(?)"
    return "edit_foreign" if info.get("depth") else "new"


def diff(old: list[dict], new: list[dict]) -> dict:
    o, n = [t["text"] for t in old], [t["text"] for t in new]
    return {"kept": [x for x in o if x in n], "gone": [x for x in o if x not in n], "added": [x for x in n if x not in o]}


def read_score(truth: list[dict], got: list[dict]) -> dict:
    """Độ đọc chữ ảnh lạ: câu đúng nguyên văn, và mọi số của chữ thật có đọc ra đủ không."""
    t, g = [x["text"] for x in truth], [x["text"] for x in got]
    nums = lambda xs: sorted(d for x in xs for d, _ in numbers(x))   # noqa: E731
    return {"exact": sum(x in g for x in t), "of": len(t), "numbers_ok": nums(t) == nums(g),
            "missed": [x for x in t if x not in g], "read": g}


def run_case(url: str, key: str | None, c: dict, out: Path, runs: Path, timeout: float) -> dict:
    rep = {"id": c["id"], "why": c.get("why", ""), "steps": []}
    imgs: list[bytes] = []
    prev_info = None
    if c.get("base"):
        b = c["base"]
        body = {"prompt": b["prompt"], "aspect_ratio": b.get("aspect_ratio", "1:1")}
        if b.get("image_file"):
            body["images_base64"] = [b64((ROOT / b["image_file"]).read_bytes())]
        r = call(url, key, body, False, timeout)
        rep["steps"].append({"kind": "gốc", "prompt": b["prompt"], "http": r["http"], "s": r["s"], "run_id": r.get("run_id"),
                             "timing": r.get("timing"), "error": r.get("error_code")})
        if not r.get("image"):
            return rep
        prev_info = meta.read(r["image"])
        rep["steps"][-1]["texts"] = [t["text"] for t in (prev_info or {}).get("texts", [])]
        cur = strip_info(r["image"]) if c.get("strip") else r["image"]
        truth = (prev_info or {}).get("texts", [])
        if c.get("strip"):
            prev_info = None
    else:
        cur, truth = (ROOT / c["source_file"]).read_bytes(), []
    imgs.append(cur)
    for e in c["edits"]:
        body = {"prompt": e["prompt"], "images_base64": [b64(cur)] + ([b64((ROOT / e["image_file"]).read_bytes())]
                                                                     if e.get("image_file") else [])}
        if e.get("aspect_ratio"):
            body["aspect_ratio"] = e["aspect_ratio"]
        r = call(url, key, body, False, timeout)
        st = {"kind": "sửa", "prompt": e["prompt"], "http": r["http"], "s": r["s"], "run_id": r.get("run_id"), "timing": r.get("timing"),
              "error": r.get("error_code"), "size": r.get("size")}
        rep["steps"].append(st)
        if not r.get("image"):
            break
        info = meta.read(r["image"])
        st["route"], st["expect"] = route(info, prev_info), e.get("expect", "edit")
        st["route_ok"] = st["route"] == st["expect"]
        old_texts = (prev_info or {}).get("texts") or []
        rd = runs / (r.get("run_id") or "_")
        if (rd / "read.json").exists():   # ảnh lạ: chữ VLM đọc được là "chữ cũ"
            got = json.loads((rd / "read.json").read_text(encoding="utf-8"))
            old_texts = got.get("texts") or []
            if truth:
                st["read"] = read_score(truth, old_texts)
        st["texts"] = [t["text"] for t in (info or {}).get("texts", [])]
        st["diff"] = diff(old_texts, (info or {}).get("texts", []))
        for f in ("request.json", "brief.json", "read.json", "v0/plan.json"):
            if (rd / f).exists():
                (out / "runs" / rd.name / f).parent.mkdir(parents=True, exist_ok=True)
                (out / "runs" / rd.name / f).write_bytes((rd / f).read_bytes())
        cur, prev_info = r["image"], info
        imgs.append(cur)
    strip(imgs, out / f"{c['id']}.jpg")
    return rep


def strip(imgs: list[bytes], path: Path, h: int = 640) -> None:
    """Ảnh gốc | sửa 1 | sửa 2 ... cạnh nhau (jpg nhẹ)."""
    from PIL import Image
    ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in imgs]
    ims = [im.resize((round(im.width * h / im.height), h)) for im in ims]
    w = Image.new("RGB", (sum(im.width for im in ims) + 10 * (len(ims) - 1), h), "white")
    x = 0
    for im in ims:
        w.paste(im, (x, 0))
        x += im.width + 10
    w.save(path, quality=85)


def main(argv: list[str] | None = None) -> int:
    from textfix.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?", type=Path, default=ROOT / "bench" / "edit1.json")
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('TEXTFIX_PORT', '8088')}")
    ap.add_argument("--only", default="")
    ap.add_argument("-j", type=int, default=4, help="số ca chạy cùng lúc (các lần sửa của một ca chạy nối tiếp)")
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args(argv)
    url = a.url.rstrip("/").removesuffix("/api") + PATH
    key = os.environ.get("TEXTFIX_API_KEY") or os.environ.get("x-api-key")
    runs = Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs"))
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    cases = [c for c in json.loads(a.set.read_text(encoding="utf-8")) if not only or c["id"] in only]
    stamp = time.strftime("%m%d_%H%M%S")
    out = ROOT / "output" / "edit_rounds" / stamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"{url}: {len(cases)} ca, {a.j} ca cùng lúc ...")
    with ThreadPoolExecutor(max_workers=a.j) as ex:
        reps = list(ex.map(lambda c: run_case(url, key, c, out, runs, a.timeout), cases))
    lines, bad_route, reads = [], 0, []
    for rp in reps:
        lines.append(f"\n{rp['id']}  {rp['why']}")
        for st in rp["steps"]:
            lines.append(f"  {st['kind']} {st['http']} {st['s']:5.1f}s {st.get('route', '')}{'' if st.get('route_ok', True) else ' SAI LUỒNG (muốn ' + st['expect'] + ')'}"
                         f"  {st['prompt'][:60]}" + (f"  LỖI {st['error']}" if st.get("error") else ""))
            bad_route += not st.get("route_ok", True)
            if st.get("diff"):
                d = st["diff"]
                lines.append(f"      giữ {len(d['kept'])} | mất {d['gone']} | thêm {d['added']}")
            if st.get("read"):
                rs = st["read"]
                reads.append(rs)
                lines.append(f"      ĐỌC ẢNH LẠ: đúng nguyên văn {rs['exact']}/{rs['of']}, số {'đủ' if rs['numbers_ok'] else 'SAI'}"
                             f", sai: {rs['missed']}")
            if st.get("timing"):
                lines.append(f"      {st['run_id']} {st['timing']}")
    ok = [st for rp in reps for st in rp["steps"] if st["kind"] == "sửa" and st["http"] == 200]
    tot = [st for rp in reps for st in rp["steps"]]
    # sai luồng ở ảnh lạ = ảnh cũ bị coi là ảnh sản phẩm -> poster mới vẽ lại poster cũ, chữ hỏng: LỖI NẶNG
    eds = [st for rp in reps for st in rp["steps"] if st["kind"] == "sửa" and "route" in st]
    for want in ("edit", "edit_foreign", "new"):
        xs = [st for st in eds if st["expect"] == want]
        if xs:
            lines.append(f"luồng {want}: đúng {sum(st['route_ok'] for st in xs)}/{len(xs)}")
    lines.append(f"\nxong {sum(st['http'] == 200 for st in tot)}/{len(tot)} lượt | sai luồng {bad_route}"
                 + (f" | đọc ảnh lạ: {sum(r['exact'] for r in reads)}/{sum(r['of'] for r in reads)} câu nguyên văn, "
                    f"số đủ {sum(r['numbers_ok'] for r in reads)}/{len(reads)} ảnh" if reads else "")
                 + (f" | giây sửa trung vị {sorted(st['s'] for st in ok)[len(ok) // 2]:.1f}" if ok else ""))
    text = "\n".join(lines)
    print(text)
    (out / "report.txt").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(reps, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "output" / "runs_zip").mkdir(parents=True, exist_ok=True)
    z = ROOT / "output" / "runs_zip" / f"edit_report_{stamp}.zip"   # cùng chỗ với chat1_report (dev_batch)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out))
    print(f"zip: {z} ({z.stat().st_size / 2 ** 20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
