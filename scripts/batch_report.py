#!/usr/bin/env python3
"""BÁO CÁO MỘT TẬP DEV ngay trên máy chủ, không phải tải các zip đầy đủ về (chạy từ terminal hay notebook đều được):

  python scripts/batch_report.py bench/accept1.json            # sau khi dev_batch chạy xong (dev_batch tự gọi)

1. IN BẢNG ra terminal: mỗi ca một dòng -- lỗi còn lại, chữ co nhỏ nhất, số câu khách mất, các cờ nghi hỏng (textfix/tools/audit.py),
   và SO VỚI LƯỢT TRƯỚC của chính ca đó (manifest cũ do `dev_batch --fresh` lưu lại): tốt lên / xấu đi / như cũ.
2. GHI CÁC TỜ ẢNH GHÉP: output/runs/_reports/<tập>/sheets/<tập>_01.jpg ... -- mỗi tờ 3 ca, mỗi ca  nháp | poster  cạnh nhau, có
   mã ca + lỗi còn. Bấm đúp tệp .jpg trong cây thư mục JupyterLab là xem được, không cần trình duyệt mở cổng máy chủ.
2b. GHI MỘT TRANG HTML TỰ ĐỦ: output/runs/_reports/<tập>.html -- nháp cạnh poster, cờ, lỗi, lượt trước; ảnh NHÚNG trong tệp nên
   tải riêng tệp này về mở bằng trình duyệt là xem được (dưới 24 MB), bấm ảnh để phóng to; hoặc mở ngay trong JupyterLab.
3. GÓI MỘT ZIP: output/runs_zip/<tập>_report_<giờ>.zip (dưới 24 MB): mỗi ca nháp + poster (JPG) + ink.png (mặt nạ nét chữ Hi-SAM) + plan.json /
   brief.json / request.json + manifest -- đủ để Claude chấm bằng mắt VÀ dựng lại bước gỡ chữ tại máy cá nhân. Tải MỘT tệp này
   từ cây thư mục JupyterLab (chuột phải > Download). Ảnh PNG gốc từng điểm ảnh thì mới cần `dev_batch.py ... --full`.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

THUMB = 520   # px cạnh dài ảnh thu nhỏ của nháp
POSTER = 1024  # px cạnh dài poster trong zip nhẹ: đủ để soi viền chữ, bóng chữ cũ
SHEET_H, SHEET_N = 720, 3   # tờ ảnh ghép: cao mỗi ảnh (px), số ca mỗi tờ


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarize(rd: Path) -> list[dict]:
    """Một lượt chạy -> mỗi ảnh một bản tóm tắt {v, mode, errors, shrink, missing, n_texts, flags, files}."""
    from textfix.tools import audit
    B = _load(rd / "brief.json") or {}
    out = []
    for vd in sorted(rd.glob("v*")):
        P = _load(vd / "plan.json")
        if P is None:
            continue
        log = P.get("log") or []
        shr = [float(x) for l in log for x in re.findall(r"co còn ([0-9.]+)", l)]
        img = lambda *names: next((vd / n for n in names if (vd / n).exists()), None)   # noqa: E731
        out.append({"v": vd.name, "mode": P.get("mode") or B.get("mode") or "poster",
                    "errors": [e.get("type") for e in P.get("errors") or []], "shrink": min(shr) if shr else 1.0,
                    "missing": len(P.get("missing") or []), "n_texts": len(B.get("texts") or []),
                    "flags": audit.flags(P) if (P.get("slots") or {}).get("L") is not None else [],
                    "draft": img("draft.png", "draft.jpg"), "poster": img("poster.png", "poster.jpg"), "dir": vd})
    return out


def _verdict(new: dict, old: dict | None) -> str:
    if old is None:
        return ""
    a, b = (len(old["flags"]), len(old["errors"]), -old["shrink"]), (len(new["flags"]), len(new["errors"]), -new["shrink"])
    return "TỐT LÊN" if b < a else "XẤU ĐI" if b > a else "như cũ"


def sheets(rows: list, out: Path, name: str) -> list[Path]:
    """Tờ ảnh ghép để XEM MẮT ngay trong JupyterLab: mỗi ca một hàng  nháp | poster  + một dòng chữ (mã ca, lỗi còn, số cờ)."""
    from PIL import Image, ImageDraw, ImageFont
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob(f"{name}_*.jpg"):
        f.unlink()
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:   # Pillow cũ
        font = ImageFont.load_default()
    panels = []
    for key, cid, s, old, vd, th, run in rows:
        ims = []
        # có lượt trước (cùng seed, ví dụ hướng cũ trong scripts/ab_lonly.sh): nháp | poster lượt trước | poster lượt này
        srcs = [s["draft"], old["poster"] if old else None, s["poster"]]
        for src in srcs:
            if src is None:
                continue
            try:
                im = Image.open(src).convert("RGB")
                ims.append(im.resize((round(im.width * SHEET_H / im.height), SHEET_H), Image.LANCZOS))
            except (OSError, AttributeError, TypeError):
                pass
        if not ims:
            continue
        pn = Image.new("RGB", (sum(i.width for i in ims) + 12 * (len(ims) - 1), SHEET_H + 44), (28, 28, 28))
        x = 0
        for i in ims:
            pn.paste(i, (x, 44))
            x += i.width + 12
        cap = f"{key}   {'nhap | luot truoc | luot nay' if old and old['poster'] else 'nhap | poster'}   loi: {', '.join(s['errors']) or 'khong'}   co nghi hong: {len(s['flags'])}   chu co: {s['shrink']:.2f}"
        ImageDraw.Draw(pn).text((8, 8), cap, fill=(255, 255, 255), font=font)
        panels.append(pn)
    files = []
    for k in range(0, len(panels), SHEET_N):
        grp = panels[k:k + SHEET_N]
        sh = Image.new("RGB", (max(p.width for p in grp), sum(p.height for p in grp) + 16 * (len(grp) - 1)), (28, 28, 28))
        y = 0
        for p in grp:
            sh.paste(p, (0, y))
            y += p.height + 16
        f = out / f"{name}_{k // SHEET_N + 1:02d}.jpg"
        sh.save(f, quality=86)
        files.append(f)
    return files


def _thumb(src: Path, dst: Path, size: int = THUMB) -> bool:
    from PIL import Image
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((size, size))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=82)
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", type=Path)
    ap.add_argument("--runs", type=Path, default=Path(os.environ.get("TEXTFIX_RUNS", ROOT / "output" / "runs")))
    ap.add_argument("--manifest", type=Path, default=None, help="mặc định output/dev/<tập>_manifest.json")
    ap.add_argument("--prev", type=Path, default=None, help="manifest lượt trước (mặc định: bản <tập>_manifest_<giờ>.json mới nhất)")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args(argv)
    name = a.set.stem
    cases = {c["id"]: c for c in json.loads(a.set.read_text(encoding="utf-8"))}
    mf = a.manifest or ROOT / "output" / "dev" / f"{name}_manifest.json"
    M = _load(mf) or {}
    olds = sorted(mf.parent.glob(f"{name}_manifest_*.json"))
    Mp = _load(a.prev or olds[-1]) if (a.prev or olds) else {}
    rep = a.runs / "_reports"
    tdir = rep / name
    rows, n_flag, n_err, verd = [], 0, 0, {"TỐT LÊN": 0, "XẤU ĐI": 0, "như cũ": 0}
    print(f"\n{'ca':5} {'so lượt trước':13} {'co':>5} {'mất':>5}  lỗi còn / cờ nghi hỏng")
    for cid in sorted(M):
        m = M[cid]
        if not m.get("run_id"):
            print(f"{cid:5} KHÔNG chạy được: {m.get('error')}")
            continue
        S = summarize(a.runs / m["run_id"])
        Sp = summarize(a.runs / Mp[cid]["run_id"]) if (Mp or {}).get(cid, {}).get("run_id") else []
        if not S:
            print(f"{cid:5} {m.get('status')}: không có plan.json ({m.get('error') or ''})")
        for k, s in enumerate(S):
            old = Sp[k] if k < len(Sp) else None
            vd = _verdict(s, old)
            if vd:
                verd[vd] += 1
            n_flag += bool(s["flags"])
            n_err += bool(s["errors"])
            key = cid if len(S) == 1 else f"{cid}_{s['v']}"
            note = "ảnh thường" if s["mode"] == "image" else "; ".join(s["flags"]) or ("lỗi: " + ", ".join(s["errors"]) if s["errors"] else "-")
            print(f"{key:5} {vd:13} {s['shrink']:5.2f} {s['missing']:2}/{s['n_texts']:<2}  {note[:150]}")
            th = {}
            for kind in ("draft", "poster"):
                if s[kind] is not None and _thumb(s[kind], tdir / f"{key}_{kind}.jpg", POSTER if kind == "poster" else THUMB):
                    th[kind] = (f"{name}/{key}_{kind}.jpg", f"../{m['run_id']}/{s['v']}/{s[kind].name}")
            if old and old["poster"] is not None and _thumb(old["poster"], tdir / f"{key}_prev.jpg"):
                th["prev"] = (f"{name}/{key}_prev.jpg", f"../{Mp[cid]['run_id']}/{old['v']}/{old['poster'].name}")
            rows.append((key, cid, s, old, vd, th, m["run_id"]))
    total = len(rows)
    head = (f"{name}: {total} ảnh | có cờ nghi hỏng {n_flag} | bộ kiểm còn lỗi {n_err}"
            + (f" | so lượt trước: tốt lên {verd['TỐT LÊN']}, xấu đi {verd['XẤU ĐI']}, như cũ {verd['như cũ']}" if Mp else " | chưa có lượt trước để so"))
    print("\n" + head)
    # ---- trang HTML
    # ảnh NHÚNG THẲNG vào trang (data URI): trang là MỘT tệp tự đủ -- mở trong JupyterLab hay tải về mở bằng trình duyệt đều thấy
    # ảnh (13/10: trang cũ trỏ tới tệp ảnh bên cạnh, mở trong JupyterLab ảnh thành icon vỡ)
    import base64
    uri = lambda rel: "data:image/jpeg;base64," + base64.b64encode((rep / rel).read_bytes()).decode()   # noqa: E731
    cell = lambda t, label: (f'<img loading="lazy" src="{uri(t[0])}"><div class="cap">{label}</div>'   # noqa: E731
                             if t else f'<div class="cap">(không có {label})</div>')
    body = []
    for key, cid, s, old, vd, th, run in rows:
        why = html.escape(cases.get(cid, {}).get("why", ""))
        fl = "".join(f"<li>{html.escape(f)}</li>" for f in s["flags"]) or "<li class='ok'>không có cờ</li>"
        cls = {"TỐT LÊN": "up", "XẤU ĐI": "down"}.get(vd, "")
        body.append(f'<section class="{"flag" if s["flags"] else ""}"><h3>{key} <span class="{cls}">{vd}</span> <small>{run}</small></h3>'
                    f'<p class="why">{why}</p><div class="row"><div>{cell(th.get("draft"), "nháp")}</div><div>{cell(th.get("poster"), "poster")}</div>'
                    + (f'<div>{cell(th.get("prev"), "poster lượt trước")}</div>' if old else "")
                    + f'<div class="info"><ul>{fl}</ul><p>lỗi còn: {html.escape(", ".join(s["errors"]) or "không")}<br>chữ co nhỏ nhất: {s["shrink"]:.2f}'
                      f'<br>câu khách thiếu (VLM khai): {s["missing"]} / {s["n_texts"]}</p>'
                    + (f'<p class="old">lượt trước: {len(old["flags"])} cờ, lỗi {html.escape(", ".join(old["errors"]) or "không")}, co {old["shrink"]:.2f}</p>' if old else "")
                    + "</div></div></section>")
    page = ("<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>TextFix - {name}</title><style>body{{font:15px/1.45 system-ui,sans-serif;margin:16px;background:#f4f4f2;color:#222}}"
            "section{background:#fff;border:1px solid #ddd;border-left:6px solid #9c9;border-radius:8px;padding:10px 14px;margin:0 0 14px}"
            "section.flag{border-left-color:#e08a00}h3{margin:0 0 4px}small{color:#888;font-weight:400}.why{color:#555;margin:0 0 8px}"
            ".row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}img{max-width:560px;max-height:640px;border:1px solid #ccc;cursor:zoom-in}"
            "img.big{max-width:none;max-height:none;cursor:zoom-out}"
            ".cap{color:#666;font-size:13px;text-align:center}.info{min-width:240px;max-width:420px}ul{margin:0 0 8px;padding-left:18px}"
            ".ok{color:#2a7}.up{color:#1a8f3c}.down{color:#c62828}.old{color:#666}</style></head><body>"
            f"<h2>{html.escape(head)}</h2><p>Lập lúc {time.strftime('%d/%m %H:%M')}. Cờ nghi hỏng là dấu hiệu để chọn poster cần xem, "
            "không phải điểm chất lượng (xem textfix/tools/audit.py). Bấm ảnh để phóng to / thu nhỏ.</p>" + "".join(body)
            + "<script>document.addEventListener('click',e=>{if(e.target.tagName==='IMG')e.target.classList.toggle('big')})</script></body></html>")
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"{name}.html").write_text(page, encoding="utf-8")
    sh = sheets(rows, tdir / "sheets", name)
    print(f"\nXEM MẮT: {len(sh)} tờ ảnh ghép (nháp | poster) trong {tdir / 'sheets'} -- bấm đúp tệp .jpg trong cây thư mục JupyterLab")
    hp = rep / f"{name}.html"
    print(f"MỘT TỆP TỰ ĐỦ (ảnh nằm trong tệp): {hp}  ({hp.stat().st_size / 2**20:.1f} MB) -- chuột phải > Download rồi mở bằng trình duyệt,"
          f" hoặc mở ngay trong JupyterLab (bấm Trust HTML)")
    if a.no_zip:
        return 0
    # ---- MỘT zip đủ để chấm VÀ dựng lại tại máy cá nhân (13/10: người dùng muốn tải ít zip nhất): mỗi ca  nháp + bản xoá + poster
    # (JPG) + plan / brief / request. Trang HTML, tờ ảnh ghép, ảnh thu nhỏ KHÔNG gói -- xem trên máy chủ, và dựng lại được từ đây.
    # Vượt 23 MB thì hạ dần chất lượng JPG (nháp để đo màu nên giữ cao nhất có thể). Mặt nạ nét chữ: PNG 1 bit, vài chục KB.
    import io
    from PIL import Image
    zdir = ROOT / "output" / "runs_zip"
    zdir.mkdir(parents=True, exist_ok=True)
    zp = zdir / f"{name}_report_{time.strftime('%m%d_%H%M%S')}.zip"

    def jpg(src: Path, q: int) -> bytes:
        b = io.BytesIO()
        Image.open(src).convert("RGB").save(b, "JPEG", quality=q, subsampling=0 if q >= 90 else 2)
        return b.getvalue()

    for q in (92, 88, 84, 78, 70):
        items, total = [], 0
        for key, cid, s_, old, vd, th, run in rows:
            rd, v = a.runs / run, s_["dir"]
            for kind in ("draft", "poster"):
                src = next((v / n for n in (f"{kind}.png", f"{kind}.jpg") if (v / n).exists()), None)
                if src is not None:
                    data = jpg(src, q if kind != "poster" else min(q, 86))
                    items.append((f"runs/{run}/{v.name}/{kind}.jpg", data))
                    total += len(data)
            if (v / "ink.png").exists():
                data = (v / "ink.png").read_bytes()
                items.append((f"runs/{run}/{v.name}/ink.png", data))
                total += len(data)
            # poster + lệnh của LƯỢT TRƯỚC (so hai hướng trên cùng nháp): vẫn một zip
            if old and old.get("poster") is not None:
                data = jpg(old["poster"], min(q, 86))
                items.append((f"prev/{cid}/{old['v']}/poster.jpg", data))
                total += len(data)
                if (old["dir"] / "plan.json").exists():
                    items.append((f"prev/{cid}/{old['v']}/plan.json", (old["dir"] / "plan.json").read_bytes()))
        if total <= 23 * 2**20:
            break
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(mf, mf.name)
        z.writestr("README.txt", f"{head} | JPG chất lượng {q} | giải nén: runs/<lượt>/... -> output/{name}/<lượt>/...")
        for arc, data in items:
            z.writestr(arc, data, compress_type=zipfile.ZIP_STORED)
        for run in sorted({r[6] for r in rows}):
            rd = a.runs / run
            for f in [rd / "brief.json", rd / "request.json", *sorted(rd.glob("v*/plan.json"))]:
                if f.exists():
                    z.write(f, f"runs/{run}/{f.relative_to(rd).as_posix()}")
    print(f"MỘT tệp để tải về cho Claude (nháp + poster + nét chữ + plan, JPG chất lượng {q}): {zp}  ({zp.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
