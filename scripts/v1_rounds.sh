#!/usr/bin/env bash
# CHẠY HẾT CÁC VÒNG THỬ API v1 trên notebook (docs/API_V1.md), MỘT lệnh:
#   bash scripts/v1_rounds.sh              # cất bộ đệm LLM, bật lại máy chủ, vòng 1-3, tóm tắt
#   bash scripts/v1_rounds.sh --keep-cache # giữ bộ đệm LLM (nhanh hơn, số giây không còn là số thật)
#   bash scripts/v1_rounds.sh --no-batch   # bỏ vòng 2 (chat1, ~10-15 phút)
# Vòng 1: khuôn API (3 ca thật + 4 ca cố tình sai, 2 ca PNG thô). Vòng 2: chất lượng bench/chat1.json (dev_batch, zip báo cáo).
# Vòng 3: tải đồng thời 8 rồi 16 yêu cầu. Cuối: đếm lời gọi LLM / VLM hỏng, mã trả về của v1.
# Mọi thứ in ra màn hình và ghi vào output/v1_rounds/<giờ>/rounds.log (+ server.log). Gửi Claude: rounds.log + zip chat1_report.
set -u
cd "$(dirname "$0")/.."
KEEP=0; BATCH=1
for a in "$@"; do
    case "$a" in --keep-cache) KEEP=1 ;; --no-batch) BATCH=0 ;; *) echo "không hiểu tham số $a"; exit 2 ;; esac
done
[ -f "$HOME/textfix_env.sh" ] && source "$HOME/textfix_env.sh"
PORT="${TEXTFIX_PORT:-8088}"
URL="http://127.0.0.1:$PORT"
OUT="output/v1_rounds/$(date +%m%d_%H%M)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/rounds.log") 2>&1

step() { echo; echo "==================== $* ($(date +%H:%M:%S))"; }

step "0. mã + máy chủ"
git log --oneline -1 2>/dev/null || echo "(không phải thư mục git)"
if [ "$KEEP" = 0 ] && [ -d "$HOME/.cache/textfix" ]; then
    mv "$HOME/.cache/textfix" "$HOME/.cache/textfix.bak_$(date +%m%d_%H%M%S)" && echo "đã cất bộ đệm LLM (số giây là số thật)"
fi
bash scripts/start_server.sh
python - "$URL" <<'EOF' || { echo "MÁY CHỦ KHÔNG LÊN -- xem server.log"; tail -20 server.log; exit 1; }
import json, sys, time, urllib.request
url, t0 = sys.argv[1], time.time()
while time.time() - t0 < 600:
    try:
        h = json.load(urllib.request.urlopen(url + "/api/health", timeout=5))
        if h.get("ready"):
            print(f"sẵn sàng sau {time.time() - t0:.0f}s"); sys.exit(0)
        if h.get("failed"):
            print("LỖI NẠP (xem server.log)"); sys.exit(1)
    except OSError:
        pass
    time.sleep(3)
print("quá 10 phút chưa sẵn sàng"); sys.exit(1)
EOF
grep -E "làm nóng xong|Sẵn sàng" server.log | tail -2

step "1. khuôn API v1"
python scripts/v1_smoke.py --url "$URL" -n 3 --bad
python scripts/v1_smoke.py --url "$URL" -n 2 --png

if [ "$BATCH" = 1 ]; then
    step "2. chất lượng: bench/chat1.json"
    python scripts/dev_batch.py bench/chat1.json --url "$URL" --fresh
fi

step "3. tải đồng thời"
python scripts/v1_smoke.py --url "$URL" -n 8
python scripts/v1_smoke.py --url "$URL" -n 16

step "TÓM TẮT"
echo "lời gọi LLM / VLM hỏng (gọi lại): $(grep -c 'hỏng sau' server.log)"
grep "hỏng sau" server.log | tail -8
echo "v1 trả về theo mã:"
grep -oE "POST /api/v1/images/generations HTTP/1.1\" [0-9]+" server.log | awk '{print $NF}' | sort | uniq -c
grep -E "v1 .* lỗi (5|4)[0-9]{2}" server.log | tail -8
cp server.log "$OUT/server.log"
echo
echo "GỬI CLAUDE: $OUT/rounds.log"
ls -t output/runs_zip/chat1_report_*.zip 2>/dev/null | head -1 | sed 's/^/           + /'
