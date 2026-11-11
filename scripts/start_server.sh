#!/usr/bin/env bash
# Chạy máy chủ demo textfix nền (nohup), log ở server.log. Dừng: pkill -f server/app.py
#   bash scripts/start_server.sh            # cổng 8088 (hoặc TEXTFIX_PORT trong .env)
#   bash scripts/start_server.sh --model base
#   bash scripts/start_server.sh --draw flux4b # ÉP FLUX.2-klein 4B TẠI CHỖ trên mọi card (bỏ TEXTFIX_FLUX_URL của 9B ở máy khác)
#   bash scripts/start_server.sh --draw qwen   # vẽ nháp bằng Qwen-Image TẠI CHỖ (mỗi card một bản, cùng Hi-SAM / LaMa) thay FLUX
#                                              # gọi ngoài: TEXTFIX_QWEN_DIR (mặc định ~/persistent-data/Qwen-Image-2.1-Turbo),
#                                              # TEXTFIX_QWEN_MEM=offload|cuda (textfix/draw/qwen.py). Bỏ cờ: quay về FLUX
set -e
cd "$(dirname "$0")/.."
# image notebook đặt libcuda lệch driver lên đầu LD_LIBRARY_PATH (lỗi 803 ngày 01/10) -> bỏ thư mục compat
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --draw) export TEXTFIX_DRAW="$2"; shift 2 ;;
        --draw=*) export TEXTFIX_DRAW="${1#--draw=}"; shift ;;
        *) args+=("$1"); shift ;;
    esac
done
export TEXTFIX_DRAW="${TEXTFIX_DRAW:-flux}"
pkill -f "server/app.py" 2>/dev/null && sleep 2 || true
nohup python server/app.py "${args[@]}" > server.log 2>&1 &
echo "đã chạy (pid $!, vẽ nháp: $TEXTFIX_DRAW) -- xem log: tail -f server.log ; kiểm tra: curl -s localhost:${TEXTFIX_PORT:-8088}/api/health"
