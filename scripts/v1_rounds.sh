#!/usr/bin/env bash
# CHẠY HẾT CÁC VÒNG THỬ API v1 trên notebook (docs/API_V1.md), MỘT lệnh:
#   bash scripts/v1_rounds.sh                          # cất bộ đệm LLM, bật lại máy chủ, vòng 1-3, tóm tắt
#   bash scripts/v1_rounds.sh --draw flux4b            # ép FLUX.2-klein 4B TẠI CHỖ trên mọi card (bỏ FLUX 9B ở máy khác)
#   bash scripts/v1_rounds.sh --llm-port 8887 --llm-model Qwen/Qwen3.5-35B-A3B   # LLM + VLM ở cổng 8887 CÙNG MÁY với FLUX 9B;
#                                                      # máy chỉ một model thì bỏ --llm-model được (lấy từ /v1/models)
#   bash scripts/v1_rounds.sh --llm-url http://<máy>:<cổng>/v1 [--llm-model <tên>]   # LLM + VLM ở chỗ khác
#   bash scripts/v1_rounds.sh --keep-cache             # giữ bộ đệm LLM (nhanh hơn, số giây không còn là số thật)
#   bash scripts/v1_rounds.sh --no-batch               # bỏ vòng 2 (chat1, ~10-15 phút)
# Vòng 1: khuôn API (3 ca thật + các ca cố tình sai, 2 ca PNG thô, 1 ca ảnh sản phẩm). Vòng 2: chất lượng bench/chat1.json
# (dev_batch, zip báo cáo). Vòng 3: tải đồng thời 8 rồi 16 yêu cầu. Cuối: lời gọi LLM / VLM hỏng, mã trả về, giây trung vị.
# Mọi thứ in ra màn hình và ghi vào output/v1_rounds/<giờ>/rounds.log (+ server.log). Gửi Claude: rounds.log + zip chat1_report.
set -u
cd "$(dirname "$0")/.."
KEEP=0; BATCH=1; DRAW=""; LLM_PORT=""; LLM_URL=""; LLM_MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --keep-cache) KEEP=1; shift ;;
        --no-batch) BATCH=0; shift ;;
        --draw) DRAW="$2"; shift 2 ;;
        --llm-port) LLM_PORT="$2"; shift 2 ;;
        --llm-url) LLM_URL="$2"; shift 2 ;;
        --llm-model) LLM_MODEL="$2"; shift 2 ;;
        *) echo "không hiểu tham số $1"; exit 2 ;;
    esac
done
[ -f "$HOME/textfix_env.sh" ] && source "$HOME/textfix_env.sh"
PORT="${TEXTFIX_PORT:-8088}"
URL="http://127.0.0.1:$PORT"
OUT="output/v1_rounds/$(date +%m%d_%H%M)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/rounds.log") 2>&1

step() { echo; echo "==================== $* ($(date +%H:%M:%S))"; }

step "0. mã + cấu hình"
git log --oneline -1 2>/dev/null || echo "(không phải thư mục git)"
# LLM / VLM khác: đặt SAU khi nạp textfix_env.sh (tệp env không đè lại); cổng trên CÙNG MÁY với FLUX 9B
if [ -n "$LLM_PORT" ]; then
    HOST=$(python -c "import os, urllib.parse as u; print(u.urlsplit(os.environ.get('TEXTFIX_FLUX_URL', '')).hostname or '')")
    [ -z "$HOST" ] && { echo "--llm-port cần TEXTFIX_FLUX_URL (máy FLUX 9B) trong textfix_env.sh để biết máy nào"; exit 2; }
    LLM_URL="http://$HOST:$LLM_PORT/v1"
fi
if [ -n "$LLM_URL" ]; then
    if [ -z "$LLM_MODEL" ]; then
        LLM_MODEL=$(python - "$LLM_URL" <<'EOF'
import json, os, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/models"
key = os.environ.get("TEXTFIX_LLM_API_KEY") or "none"
try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"}), timeout=10) as r:
        ids = [m["id"] for m in json.load(r).get("data", [])]
except Exception as e:
    print(f"KHÔNG đọc được {url}: {e}", file=sys.stderr); sys.exit(1)
print(f"model ở {url}: {ids}", file=sys.stderr)
if len(ids) > 1:   # cổng chung nhiều model (12/11: gemma, qwen, FLUX, ASR cùng một cổng) -- không đoán
    print("máy có nhiều model: chọn bằng --llm-model <tên>, ví dụ --llm-model " + ids[0], file=sys.stderr); sys.exit(1)
print(ids[0] if ids else "")
EOF
) || exit 1
        [ -z "$LLM_MODEL" ] && { echo "máy $LLM_URL không khai model nào -- thêm --llm-model <tên>"; exit 2; }
    fi
    export TEXTFIX_LLM_URL="$LLM_URL" TEXTFIX_MODEL="$LLM_MODEL"
    unset TEXTFIX_VLM_URL TEXTFIX_VLM_MODEL   # một model làm cả LLM lẫn VLM
fi
[ -n "$DRAW" ] && export TEXTFIX_DRAW="$DRAW"
echo "LLM + VLM: ${TEXTFIX_MODEL:-?} | vẽ nháp: ${TEXTFIX_DRAW:-flux}$([ "${TEXTFIX_DRAW:-flux}" = flux ] && [ -n "${TEXTFIX_FLUX_URL:-}" ] && echo ' (máy khác)')"
if [ "$KEEP" = 0 ] && [ -d "$HOME/.cache/textfix" ]; then
    mv "$HOME/.cache/textfix" "$HOME/.cache/textfix.bak_$(date +%m%d_%H%M%S)" && echo "đã cất bộ đệm LLM (số giây là số thật)"
fi
bash scripts/start_server.sh
python - "$URL" <<'EOF' || { echo "MÁY CHỦ KHÔNG LÊN -- xem server.log"; tail -30 server.log; exit 1; }
import json, sys, time, urllib.request
url, t0 = sys.argv[1], time.time()
while time.time() - t0 < 900:
    try:
        h = json.load(urllib.request.urlopen(url + "/api/health", timeout=5))
        if h.get("ready"):
            print(f"sẵn sàng sau {time.time() - t0:.0f}s"); sys.exit(0)
        if h.get("failed"):
            print("LỖI NẠP (xem server.log)"); sys.exit(1)
    except OSError:
        pass
    time.sleep(3)
print("quá 15 phút chưa sẵn sàng"); sys.exit(1)
EOF
grep -E "làm nóng xong|Sẵn sàng|sẵn sàng \(" server.log | tail -4

step "1. khuôn API v1"
python scripts/v1_smoke.py --url "$URL" -n 3 --bad --image-file bench/refs/coca_cola_can.jpg
python scripts/v1_smoke.py --url "$URL" -n 2 --png

if [ "$BATCH" = 1 ]; then
    step "2. chất lượng: bench/chat1.json"
    python scripts/dev_batch.py bench/chat1.json --url "$URL" --fresh
fi

step "3. tải đồng thời"
python scripts/v1_smoke.py --url "$URL" -n 8
python scripts/v1_smoke.py --url "$URL" -n 16

step "TÓM TẮT"
echo "LLM + VLM: ${TEXTFIX_MODEL:-?} | vẽ nháp: ${TEXTFIX_DRAW:-flux}"
echo "lời gọi LLM / VLM hỏng (gọi lại): $(grep -c 'hỏng sau' server.log)"
grep "hỏng sau" server.log | tail -8
echo "v1 trả về theo mã:"
grep -oE "POST /api/v1/images/generations HTTP/1.1\" [0-9]+" server.log | awk '{print $NF}' | sort | uniq -c
grep -E " lỗi (5|4)[0-9]{2} ERR" server.log | tail -8
cp server.log "$OUT/server.log"
echo
echo "GỬI CLAUDE: $OUT/rounds.log"
ls -t output/runs_zip/chat1_report_*.zip 2>/dev/null | head -1 | sed 's/^/           + /'
