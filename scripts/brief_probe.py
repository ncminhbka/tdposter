#!/usr/bin/env python3
"""MÁY CHỦ: chạy RIÊNG bước LLM đầu luồng (textfix/brief/) trên các tập dev / accept -- không vẽ, không cần GPU, chỉ cần
TEXTFIX_LLM_URL / TEXTFIX_MODEL như máy chủ. Mỗi ca: request -> brief (texts, prompt_en, log, số token FLUX). Gói một zip nhẹ để
so với brief của lượt chạy trước tại máy cá nhân. Không chạy tập thi (final*).

  %run scripts/brief_probe.py                          # bench/chat1.json (các ca chỉ có lời chat)
  %run scripts/brief_probe.py bench/chat1.json --only c03,c05
  %run scripts/brief_probe.py --fresh                  # bỏ qua cache LLM (gọi lại thật, đo độ ổn định)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETS = ["bench/chat1.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=SETS)
    ap.add_argument("--only", default="", help="các ca, cách nhau dấu phẩy")
    ap.add_argument("--fresh", action="store_true", help="không đọc cache LLM")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from textfix import config
    from textfix.tools import ship
    from textfix.brief import SCHEMA, count_tokens, make_brief
    from textfix.llm import json_call
    config.load_env()
    m = config.models()
    call = json_call(m["llm"]["url"], m["llm"]["model"], m["llm"]["key_env"], tag="brief", schema=SCHEMA)
    llm = (lambda system, parts: call(system, parts, fresh=True)) if a.fresh else call
    only = {x for x in a.only.split(",") if x}

    def one(c: dict) -> dict:
        r = c["request"]
        t0 = time.time()
        try:
            B = make_brief(llm, r["prompt"], bool(c.get("image")), aspect=r.get("aspect_ratio", "1:1"))
        except Exception as e:
            return {"id": c["id"], "error": f"{type(e).__name__}: {e}"}
        return {"id": c["id"], "s": round(time.time() - t0, 1), "tokens": count_tokens(B["prompt_en"]),
                "words": len(B["prompt_en"].split()), **B}

    out_dir = ROOT / "output" / "brief_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    files = []
    for s in a.sets:
        cases = [c for c in json.loads((ROOT / s).read_text(encoding="utf-8"))
                 if (c.get("request") or {}).get("prompt") and (not only or c["id"] in only)]
        if not cases:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            res = list(ex.map(one, cases))
        name = Path(s).stem
        f = out_dir / f"{name}_{stamp}.json"
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        files.append((f, f.name))
        err = [x for x in res if "error" in x]
        ok = [x for x in res if "error" not in x]
        cnt = lambda k: sum(bool(x["log"].get(k)) for x in ok)   # noqa: E731
        print(f"{name}: {len(res)} ca, {len(err)} lỗi, {time.time() - t0:.0f}s | chữ bịa số bị bỏ {cnt('facts_not_from_user')}, "
              f"sửa chép sai {cnt('snapped')}, viết hoa lại {cnt('recased')}, ngoặc khách thêm lại {cnt('added')}, "
              f"chèn chữ vào prompt {cnt('quoted_added')}, cắt cho vừa 512 {cnt('trimmed_sentences')}, "
              f"token max {max((x['tokens'] for x in ok), default=0)}, "
              f"giây trung vị {sorted(x['s'] for x in ok)[len(ok) // 2] if ok else 0}, "
              f"số từ prompt trung vị {sorted(x['words'] for x in ok)[len(ok) // 2] if ok else 0}")
        for x in err:
            print("  LỖI", x["id"], x["error"])
    if files:
        ship.offer(ship.pack(files, ROOT / "output" / "runs_zip", f"brief_probe_{stamp}"))


if __name__ == "__main__":
    main()
