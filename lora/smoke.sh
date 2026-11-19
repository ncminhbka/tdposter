#!/usr/bin/env bash
# SMOKE TEST lora/ với TRỌNG SỐ THẬT (~10 phút, cần 1 card trống): collect -> cache -> train 3 bước -> sheets 2 ảnh, trên vài lượt
# CÓ SẴN của máy chủ (mặc định output/runs). Mục đích: lộ lỗi mã / đường dẫn trọng số / VRAM TRƯỚC khi bỏ một giờ sinh dữ liệu.
#   bash lora/smoke.sh                    # lượt trong output/runs, card cuda:0
#   DEV=cuda:1 bash lora/smoke.sh data/lora/runs
# Lượt cũ (nháp 9B, chưa có base.png) vẫn dùng được cho smoke: collect lùi về mặt nạ kiểu cũ ("mask_mode": "diff").
set -eu
cd "$(dirname "$0")/.."
RUNS="${1:-output/runs}"; N="${N:-6}"; DEV="${DEV:-cuda:0}"
S=data/lora/smoke
rm -rf "$S"; mkdir -p "$S/runs"
echo "== 1. test tại chỗ"
python -m pytest -q tests/test_lora.py
echo "== 2. chọn $N lượt có đủ draft.png + poster.png + plan.json trong $RUNS"
python - "$RUNS" "$S/runs" "$N" <<'EOF'
import json, shutil, sys
from pathlib import Path
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
got = 0
for r in sorted(src.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
    v = r / "v0"
    if not all((v / f).exists() for f in ("draft.png", "poster.png", "plan.json")) or not (r / "brief.json").exists():
        continue
    try:
        if json.loads((r / "request.json").read_text(encoding="utf-8")).get("mode") != "new":
            continue
    except (OSError, ValueError):
        continue
    shutil.copytree(r, dst / r.name)
    got += 1
    if got >= n:
        break
print(f"chép {got} lượt")
sys.exit(0 if got else 1)
EOF
echo "== 3. collect"
python -m lora.collect --runs "$S/runs" --out "$S/ds" --eval-frac 0
echo "== 4. cache ($DEV)"
python -m lora.cache --ds "$S/ds" --device "$DEV"
echo "== 5. train 3 bước ($DEV)"
python -m lora.train --ds "$S/ds" --out "$S/train" --rank 8 --steps 3 --accum 1 --save-every 3 --warmup 1 --device "$DEV"
echo "== 6. sheets (distill, 2 mẫu train + 1 prompt không chữ)"
python -m lora.sheets --ds "$S/ds" --split train --limit 2 --notext 1 --lora "$S/train/lora_step00003.safetensors" \
    --out "$S/sheets" --device "$DEV"
ls -la "$S/sheets"/sheet_*.jpg
echo "SMOKE XANH -- chuỗi chạy được với trọng số thật. Xem VRAM / giây mỗi bước trong $S/train/log.jsonl"
