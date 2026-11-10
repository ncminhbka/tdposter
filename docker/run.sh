#!/usr/bin/env bash
# Lệnh chính của container: in thông tin GPU, rồi chạy máy chủ ở tiền cảnh (log ra `docker logs`).
set -e
cd "$(dirname "$0")/.."
PY="${TEXTFIX_PY:-python}"
# image notebook đặt libcuda lệch driver (thư mục .../compat) lên đầu LD_LIBRARY_PATH (lỗi 803 trên máy notebook): mặc định bỏ,
# như scripts/start_server.sh. Máy driver cũ cần compat để chạy CUDA mới hơn: TEXTFIX_KEEP_COMPAT=1 giữ nguyên.
if [ "${TEXTFIX_KEEP_COMPAT:-0}" != "1" ]; then
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/compat' | paste -sd: -)"
fi
# Chromium ĐÓNG TRONG IMAGE: tệp env chép từ notebook có thể mang PLAYWRIGHT_CHROME_PATH của notebook (~/persistent-data/...),
# không có trong container -> mọi poster lỗi design_failed (11/11). Đường dẫn không chạy được -> dùng bản của image, báo rõ.
IMAGE_CHROME=/opt/textfix/chrome-linux64/chrome
if [ -n "${PLAYWRIGHT_CHROME_PATH:-}" ] && [ ! -x "$PLAYWRIGHT_CHROME_PATH" ] && [ -x "$IMAGE_CHROME" ]; then
    echo "[textfix] CHÚ Ý: PLAYWRIGHT_CHROME_PATH=$PLAYWRIGHT_CHROME_PATH không có trong container -> dùng Chromium của image ($IMAGE_CHROME)"
    export PLAYWRIGHT_CHROME_PATH="$IMAGE_CHROME"
fi
if [ -z "${TEXTFIX_FLUX_URL:-}" ]; then
    echo "LỖI: chưa đặt TEXTFIX_FLUX_URL (image không chứa FLUX; xem docker/textfix.env.example)" >&2
    exit 1
fi
# HẠN MỨC CPU của container (Kubernetes limits.cpu, docker --cpus): numpy / OpenCV / torch / onnxruntime (OCR) mặc định mở luồng
# theo số lõi CẢ MÁY -> vượt hạn mức, bị bóp liên tục: bước đo dòng chậm ~10 lần (16/10, pod trên máy L40S). Đặt số luồng = hạn
# mức. TEXTFIX_THREADS đặt tay thì thắng; biến *_NUM_THREADS người deploy đã đặt thì giữ.
if [ -z "${TEXTFIX_THREADS:-}" ]; then
    q=""; p=""
    # "|| true": đọc hỏng (tệp lạ, không xuống dòng) không được làm dừng script (set -e) -> coi như không giới hạn
    if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q p < /sys/fs/cgroup/cpu.max || true
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true)
        p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || true)
    fi
    if [[ "$q" =~ ^[0-9]+$ ]] && [[ "$p" =~ ^[0-9]+$ ]] && [ "$q" -gt 0 ] && [ "$p" -gt 0 ]; then
        TEXTFIX_THREADS=$(( (q + p - 1) / p ))
    fi
fi
if [ -n "${TEXTFIX_THREADS:-}" ]; then
    export TEXTFIX_THREADS
    for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS OPENCV_FOR_THREADS_NUM; do
        export "$v=${!v:-$TEXTFIX_THREADS}"
    done
    echo "[textfix] CPU: hạn mức $TEXTFIX_THREADS lõi (máy có $(nproc --all)) -> số luồng thư viện = $TEXTFIX_THREADS"
else
    echo "[textfix] CPU: không giới hạn ($(nproc) lõi)"
fi
# máy lạ: in ngay torch / CUDA / driver / từng card để biết máy có hợp không (GPU không dùng được thì Hi-SAM + LaMa chạy CPU)
"$PY" - <<'EOF' || true
import subprocess, torch
print(f"[textfix] torch {torch.__version__}, build CUDA {torch.version.cuda}, GPU dùng được: {torch.cuda.is_available()}")
try:
    print("[textfix] " + subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
                                         "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
          .replace("\n", "\n[textfix] "))
except Exception as e:
    print(f"[textfix] không chạy được nvidia-smi ({e}) -- container không thấy GPU? (docker run --gpus all)")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, tot = torch.cuda.mem_get_info(i)
        warn = "  << dưới 8 GB trống: Hi-SAM + LaMa có thể hết bộ nhớ (TEXTFIX_GPUS để bỏ card này)" if free < 8 * 2 ** 30 else ""
        print(f"[textfix] cuda:{i} {torch.cuda.get_device_name(i)}: trống {free / 2 ** 30:.1f} / {tot / 2 ** 30:.1f} GB{warn}")
else:
    print("[textfix] CHÚ Ý: không dùng được GPU -> Hi-SAM + LaMa chạy CPU (chậm). Driver quá cũ cho torch này? thử TEXTFIX_KEEP_COMPAT=1")
EOF
# tên máy FLUX / LLM / VLM phải phân giải được TRONG container (mạng docker không có DNS nội bộ của notebook)
"$PY" - <<'EOF' || true
import os, socket
from urllib.parse import urlparse
for k in ("TEXTFIX_FLUX_URL", "TEXTFIX_LLM_URL", "TEXTFIX_VLM_URL", "TEXTFIX_KHO_ENDPOINT"):
    host = urlparse(os.environ.get(k) or "").hostname
    if not host:
        continue
    try:
        print(f"[textfix] {k}: {host} -> {socket.gethostbyname(host)}")
    except OSError as e:
        print(f"[textfix] LỖI {k}: không phân giải được {host} ({e}) -- docker run thêm --add-host {host}:<IP> hoặc --dns <máy DNS>")
EOF
# thư mục mount vào /data phải cho uid 1000 (người chạy máy chủ trong container) ghi được; docker tự tạo thư mục mount -> chủ root
RUNS="${TEXTFIX_RUNS:-/data/runs}"
if ! mkdir -p "$RUNS" 2>/dev/null || [ ! -w "$RUNS" ]; then
    echo "LỖI: không ghi được $RUNS (container chạy uid $(id -u)). Trên máy thật: sudo chown -R 1000:100 <thư mục mount vào /data>" >&2
    echo "     (không có sudo: docker run --rm --user root --entrypoint chown -v <thư mục>:/data <image> -R 1000:100 /data)" >&2
    exit 1
fi
export TEXTFIX_KHO_DIR="${TEXTFIX_KHO_DIR:-/data/kho}"   # đệm kho ảnh tham chiếu (docs/KHO.md): giữ qua các lần chạy lại
exec "$PY" server/app.py --port "${TEXTFIX_PORT:-8004}" "$@"
