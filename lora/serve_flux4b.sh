#!/usr/bin/env bash
# MÁY CHỦ SINH DỮ LIỆU LoRA (docs/LORA_PLAN.md, docs/LORA_CHANGES.md): FLUX.2-klein 4B distill TẠI CHỖ, chế độ dữ liệu.
#   bash lora/serve_flux4b.sh --llm-port 8887 --llm-model Qwen/Qwen3.5-35B-A3B            # mọi card
#   bash lora/serve_flux4b.sh --llm-port 8887 --llm-model Qwen/Qwen3.5-35B-A3B --gpus 0   # chỉ card 0 (card 1 rảnh)
# Khác máy chủ thường:
#   - lượt chạy vào data/lora/runs và KHÔNG bị xoá (máy chủ thường chỉ giữ 300 lượt mới nhất);
#   - TEXTFIX_DATA_MODE=1: lưu thêm v0/base.png (nền dựng không nén mất -> tách chữ dựng khỏi vùng LaMa lấp) và ghi model vẽ
#     vào request.json (collect chỉ nhận lượt của 4B distill không LoRA);
#   - hạn trả lời 1800 s thay 55 s, hàng đợi 32: từ v0.0.9 lượt quá hạn bị DỪNG -> với hạn 55 s, lượt đứng chờ khi sinh hàng
#     loạt sẽ bị huỷ mất.
# draft.png / ink.png / base.png ghi NỀN sau khi trả lời: drive xong CHỜ 60 s rồi mới dừng (pkill -f server/app.py).
set -eu
cd "$(dirname "$0")/.."
[ -f "$HOME/textfix_env.sh" ] && source "$HOME/textfix_env.sh"
LLM_PORT=""; LLM_URL=""; LLM_MODEL=""; GPUS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --llm-port) LLM_PORT="$2"; shift 2 ;;
        --llm-url) LLM_URL="$2"; shift 2 ;;
        --llm-model) LLM_MODEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
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
[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
unset TEXTFIX_FLUX_LORA   # dữ liệu phải là nháp của model GỐC
export TEXTFIX_RUNS="$PWD/data/lora/runs" TEXTFIX_MAX_RUNS=100000000 TEXTFIX_DATA_MODE=1 \
       TEXTFIX_V1_DEADLINE=1800 TEXTFIX_MAX_QUEUE=32 TEXTFIX_JOB_WORKERS=32
mkdir -p "$TEXTFIX_RUNS"
echo "LLM + VLM: ${TEXTFIX_MODEL:-?} | vẽ nháp: flux4b (tại chỗ) | card: ${CUDA_VISIBLE_DEVICES:-tất cả} | lượt chạy: $TEXTFIX_RUNS"
bash scripts/start_server.sh --draw flux4b
echo "chờ máy chủ sẵn sàng: curl -s localhost:${TEXTFIX_PORT:-8088}/api/health  (nạp 4B + làm nóng: vài phút)"
