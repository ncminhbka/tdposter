#!/usr/bin/env python3
"""THỬ TAY API v1: gõ prompt -> poster (PNG lưu ở output/try/, tự mở lên xem). Sửa poster vừa ra bằng lệnh /sua.

  python -X utf8 scripts/try_api.py                      # máy chủ: TEXTFIX_TRY_URL trong .env, không có thì 127.0.0.1:8088
  python -X utf8 scripts/try_api.py --url http://127.0.0.1:8088

Lệnh trong lúc chạy (dòng không bắt đầu bằng "/" = prompt TẠO MỚI):
  /sua <lời sửa>      sửa poster vừa ra (gửi lại đúng PNG đó + lời sửa, như chatbot)
  /anh <đường dẫn>    đính kèm ảnh (sản phẩm / ảnh lạ cần sửa) cho các lần tạo mới sau; /anh  (trống) = bỏ ảnh
  /khung 16:9         đổi tỉ lệ (1:1, 9:16, 16:9, 3:4, 4:3)
  /mo <đường dẫn>     chọn một PNG cũ làm "poster vừa ra" (để /sua tiếp)
  /q                  thoát
Khoá x-api-key: TEXTFIX_API_KEY hoặc dòng x-api-key=... trong .env (không in ra).
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", line_buffering=True)

from textfix.config import load_env  # noqa: E402

load_env()
from v1_smoke import PATH, RATIOS, call  # noqa: E402

from textfix.meta import read  # noqa: E402


def show(path: Path) -> None:
    """Mở ảnh bằng trình xem mặc định của máy."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as e:
        print(f"  (không mở được ảnh: {e})")


def b64_file(path: str) -> str:
    return base64.b64encode(Path(path.strip().strip('"')).read_bytes()).decode()


def report(r: dict, out: Path, tag: str) -> bytes | None:
    if not r.get("image"):
        print(f"  LỖI {r.get('http')} {r.get('error_code')}: {r.get('message') or ''} ({r.get('s')} s, run {r.get('run_id')})")
        return None
    f = out / f"{time.strftime('%H%M%S')}_{tag}.png"
    f.write_bytes(r["image"])
    m = read(r["image"]) or {}
    print(f"  {r['http']} · {r['s']} s · run {r.get('run_id')} · {m.get('mode')} · khung {m.get('aspect')}"
          + (f" · {m['panels']['n']} ô" if m.get("panels") else ""))
    for t in m.get("texts") or []:
        print(f"    - [{t.get('role')}{', ô ' + str(t['panel']) if t.get('panel') else ''}] {t['text']}")
    if r.get("timing"):
        print(f"    bước: {r['timing']}")
    print(f"  -> {f}")
    show(f)
    return r["image"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("TEXTFIX_TRY_URL") or "http://127.0.0.1:8088")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    url = a.url.rstrip("/").removesuffix("/api") + PATH
    key = os.environ.get("TEXTFIX_API_KEY") or os.environ.get("x-api-key")
    out = ROOT / "output" / "try"
    out.mkdir(parents=True, exist_ok=True)
    aspect, attach, last = "1:1", None, None
    print(f"Máy chủ: {a.url.split('//')[-1].split('/')[0]} · khung {aspect} · gõ /q để thoát, xem lệnh ở đầu tệp scripts/try_api.py")
    while True:
        try:
            line = input(f"\n[{aspect}{' + ảnh' if attach else ''}{' | có poster để /sua' if last else ''}] prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        try:
            if cmd == "/q":
                return
            if cmd == "/khung":
                if arg.strip() in RATIOS:
                    aspect = arg.strip()
                else:
                    print(f"  tỉ lệ phải là một trong {', '.join(RATIOS)}")
                continue
            if cmd == "/anh":
                attach = b64_file(arg) if arg.strip() else None
                print("  đã đính kèm ảnh" if attach else "  đã bỏ ảnh")
                continue
            if cmd == "/mo":
                last = Path(arg.strip().strip('"')).read_bytes()
                print("  đã chọn poster để sửa")
                continue
            if cmd == "/sua":
                if not last or not arg.strip():
                    print("  cần có poster vừa ra và lời sửa: /sua <lời sửa>")
                    continue
                body = {"prompt": arg.strip(), "aspect_ratio": aspect, "images_base64": [base64.b64encode(last).decode()]}
                tag = "sua"
            elif cmd.startswith("/"):
                print("  lệnh lạ; xem đầu tệp scripts/try_api.py")
                continue
            else:
                body = {"prompt": line, "aspect_ratio": aspect, **({"images_base64": [attach]} if attach else {})}
                tag = "moi"
        except OSError as e:
            print(f"  không đọc được tệp: {e}")
            continue
        print("  đang tạo ...")
        img = report(call(url, key, body, True, a.timeout), out, tag)
        if img is not None:
            last = img


if __name__ == "__main__":
    main()
