#!/usr/bin/env bash
# Build image demo textfix ngay trên notebook (docker của máy thật qua /run/docker.sock). Chạy từ gốc repo:
#   bash docker/build.sh                 # -> image textfix:demo
#   TAG=textfix:demo2 bash docker/build.sh
# Gom một thư mục build riêng (không gửi output/, bench/ ... cho docker): code + trọng số Hi-SAM / LaMa CHÉP từ
# ~/persistent-data + Chromium, rồi docker build. Thư mục build để lại: lần sau chỉ chép lại code (xoá hẳn: rm -rf "$CTX").
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DATA="${TEXTFIX_DATA:-$HOME/persistent-data}"
CTX="${CTX:-$HOME/work/textfix-docker-build}"
TAG="${TAG:-textfix:demo}"
HISAM="${TEXTFIX_HISAM:-$DATA/hisam}"
LAMA="${TEXTFIX_LAMA:-$DATA/lama/big-lama.pt}"
CHROME_DIR="$(dirname "${PLAYWRIGHT_CHROME_PATH:-$DATA/chrome-linux64/chrome}")"

die() { echo "LỖI: $*" >&2; exit 1; }
size_is() { [ "$(stat -c %s "$1" 2>/dev/null || echo 0)" = "$2" ] || die "$1 sai kích thước (cần $2 byte) -- tải lại (docs/DEPLOY.md)"; }

echo "1. Kiểm trọng số và Chromium"
[ -f "$HISAM/Hi-SAM/hi_sam/modeling/build.py" ] || die "thiếu mã Hi-SAM: $HISAM/Hi-SAM/hi_sam"
size_is "$HISAM/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth" 1249524607
size_is "$HISAM/sam_tss_l_textseg.pth" 122756163
size_is "$LAMA" 205669692
[ -x "$CHROME_DIR/chrome" ] || die "thiếu Chromium: $CHROME_DIR/chrome (python scripts/check_server.py --fix giải nén zip trong persistent-data)"
BASE="$(sed -n 's/^ARG BASE=//p' docker/Dockerfile)"
docker image inspect "$BASE" >/dev/null 2>&1 || die "docker không thấy image gốc (sửa ARG BASE trong docker/Dockerfile)"
PY="$(command -v python)"
case "$PY" in "$HOME"/*) die "python của notebook ($PY) nằm trong thư mục nhà, không có trong image gốc";; esac
echo "   python: $PY"
"$PY" - <<'EOF'
import torch, os
where = os.path.dirname(torch.__file__)
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda}) ở {where}")
if where.startswith(os.path.expanduser("~")):
    print("   CHÚ Ý: torch của notebook cài trong thư mục nhà, không nằm trong image gốc -> pip trong docker sẽ cài torch từ Nexus;"
          " nếu container báo lỗi CUDA thì ghim đúng bản này trong docker/requirements.txt")
EOF

echo "2. Gom thư mục build: $CTX"
# code: luôn chép mới; trọng số + Chromium (1,9 GB): chỉ chép khi chưa có hoặc khác kích thước (build lại sau khi sửa code nhanh)
same() { [ -f "$2" ] && [ "$(stat -c %s "$1")" = "$(stat -c %s "$2")" ]; }
put() { same "$1" "$2" || { mkdir -p "$(dirname "$2")"; cp "$1" "$2"; echo "   chép $(basename "$2")"; }; }
rm -rf "$CTX/app"
mkdir -p "$CTX/app/scripts"
cp -r textfix server assets docker "$CTX/app/"
cp scripts/check_server.py scripts/start_server.sh "$CTX/app/scripts/"
# Dockerfile dồn image gốc thành một lớp (FROM scratch) -> mất ENV / SHELL / ENTRYPOINT của image gốc: chép lại từ inspect
docker image inspect "$BASE" > "$CTX/base.json"
"$PY" - "$CTX/base.json" docker/Dockerfile "$CTX/Dockerfile" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))[0]["Config"]
q = lambda v: json.dumps(v).replace("$", "\\$")          # Dockerfile thay $X trong ENV: giữ nguyên giá trị gốc
out = [f"ENV {e.split('=', 1)[0]}={q(e.split('=', 1)[1])}" for e in c.get("Env") or []]
if c.get("Shell"):
    out.append("SHELL " + json.dumps(c["Shell"]))
if c.get("Entrypoint"):
    out.append("ENTRYPOINT " + json.dumps(c["Entrypoint"]))
src = open(sys.argv[2], encoding="utf-8").read()
assert "# @BASE_CONFIG@" in src
open(sys.argv[3], "w", encoding="utf-8").write(src.replace("# @BASE_CONFIG@", "\n".join(out)))
print(f"   cấu hình image gốc: {len(c.get('Env') or [])} biến môi trường, entrypoint {c.get('Entrypoint')}")
EOF
rm -f "$CTX/base.json"
cp docker/requirements.txt "$CTX/requirements.txt"
# GHIM đúng bản đang chạy tốt trên notebook (pip trong docker không có ràng buộc thì lấy bản mới nhất: playwright lệch đời
# Chromium, openai / opencv lên bản lớn mới) -> constraints.txt cho pip trong Dockerfile.
# KHÔNG ghim torch / torchvision: lấy từ IMAGE GỐC (đúng bản CUDA của nó). Notebook có thể chạy image khác image gốc (11/11:
# notebook dựng lại có torch 2.11+cu128, image gốc cu130) và bản "+cuXXX" không có trên Nexus -> pip trong docker không giải được.
"$PY" -m pip freeze 2>/dev/null | grep -iE "^(numpy|opencv-python-headless|opencv-python|pillow|einops|rapidocr-onnxruntime|onnxruntime|playwright|openai|httpx|fastapi|uvicorn|pydantic|packaging|fonttools|shapely|pyclipper)=="     > "$CTX/constraints.txt" || true
echo "   ghim theo notebook: $(paste -sd' ' "$CTX/constraints.txt")"
grep -qi "^playwright==" "$CTX/constraints.txt" || echo "   CHÚ Ý: notebook không có playwright? pip trong docker sẽ lấy bản mới nhất"
rm -rf "$CTX/models/hisam/Hi-SAM/hi_sam" && mkdir -p "$CTX/models/hisam/Hi-SAM" && cp -r "$HISAM/Hi-SAM/hi_sam" "$CTX/models/hisam/Hi-SAM/"
put "$HISAM/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth" "$CTX/models/hisam/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth"
put "$HISAM/sam_tss_l_textseg.pth" "$CTX/models/hisam/sam_tss_l_textseg.pth"
put "$LAMA" "$CTX/models/lama/big-lama.pt"
same "$CHROME_DIR/chrome" "$CTX/chrome-linux64/chrome" || { rm -rf "$CTX/chrome-linux64"; cp -a "$CHROME_DIR" "$CTX/chrome-linux64"; echo "   chép Chromium"; }
find "$CTX" -name __pycache__ -prune -exec rm -rf {} +
du -sh "$CTX"

echo "3. docker build -> $TAG"
# container build nằm trên mạng mặc định của docker máy thật: không có DNS nội bộ của notebook (lỗi "Temporary failure in name
# resolution" khi pip gọi Nexus) -> build bằng mạng máy thật, và ghi cứng IP Nexus phân giải ngay trong notebook
NET=(--network "${BUILD_NETWORK:-host}")
NEXUS_IP="$(getent hosts nexus.kcntt.net | awk '{print $1; exit}')"
[ -n "$NEXUS_IP" ] && NET+=(--add-host "nexus.kcntt.net:$NEXUS_IP") && echo "   nexus.kcntt.net -> $NEXUS_IP"
docker build "${NET[@]}" --build-arg PY="$PY" -t "$TAG" "$CTX"
docker image ls "$TAG"
echo "Xong. Chạy: xem docs/DOCKER.md (docker run ... -p 8004:8004 ...)"
