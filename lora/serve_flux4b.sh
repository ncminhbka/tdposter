#!/usr/bin/env bash
# MÁY CHỦ SINH DỮ LIỆU LoRA (docs/LORA_PLAN.md, chặng 2): FLUX.2-klein 4B TẠI CHỖ trên mọi card, mọi lượt chạy lưu vào
# data/lora/runs và KHÔNG bị xoá (máy chủ thường chỉ giữ TEXTFIX_MAX_RUNS=300 lượt mới nhất).
#   bash lora/serve_flux4b.sh --llm-port 8887 --llm-model Qwen/Qwen3.5-35B-A3B
#   bash lora/serve_flux4b.sh --llm-url http://<máy>:<cổng>/v1 --llm-model <tên>
# Dừng: pkill -f server/app.py (rồi chạy lại máy chủ thường bằng scripts/start_server.sh hoặc scripts/v1_rounds.sh)
set -eu
cd "$(dirname "$0")/.."
[ -f "$HOME/textfix_env.sh" ] && source "$HOME/textfix_env.sh"
LLM_PORT=""; LLM_URL=""; LLM_MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --llm-port) LLM_PORT="$2"; shift 2 ;;
        --llm-url) LLM_URL="$2"; shift 2 ;;
        --llm-model) LLM_MODEL="$2"; shift 2 ;;
        *) echo "tham số lạ: $1"; exit 2 ;;
    esac
done
if [ -n "$LLM_PORT" ]; then   # LLM + VLM ở cổng này trên CÙNG MÁY với FLUX 9B (như scripts/v1_rounds.sh --llm-port)
    HOST=$(python -c "import os, urllib.parse as u; print(u.urlsplit(os.environ.get('TEXTFIX_FLUX_URL', '')).hostname or '')")
    [ -z "$HOST" ] && { echo "--llm-port cần TEXTFIX_FLUX_URL trong ~/textfix_env.sh để biết máy nào"; exit 2; }
    LLM_URL="http://$HOST:$LLM_PORT/v1"
fi
if [ -n "$LLM_URL" ]; then
    [ -z "$LLM_MODEL" ] && { echo "thêm --llm-model <tên>"; exit 2; }
    export TEXTFIX_LLM_URL="$LLM_URL" TEXTFIX_MODEL="$LLM_MODEL"
    unset TEXTFIX_VLM_URL TEXTFIX_VLM_MODEL
fi
export TEXTFIX_RUNS="$PWD/data/lora/runs" TEXTFIX_MAX_RUNS=100000000
mkdir -p "$TEXTFIX_RUNS"
echo "LLM + VLM: ${TEXTFIX_MODEL:-?} | vẽ nháp: flux4b (tại chỗ) | lượt chạy: $TEXTFIX_RUNS"
bash scripts/start_server.sh --draw flux4b
echo "chờ máy chủ sẵn sàng: curl -s localhost:${TEXTFIX_PORT:-8088}/api/health  (nạp 4B trên mọi card + làm nóng: vài phút)"
