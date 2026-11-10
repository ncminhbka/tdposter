"""Cấu hình qua biến môi trường (export trong terminal, hoặc tệp .env ở gốc repo -- biến đã export thắng .env).

  TEXTFIX_LLM_URL, TEXTFIX_MODEL, TEXTFIX_LLM_API_KEY     LLM / VLM (OpenAI-compatible, vLLM); một model nhìn được ảnh làm cả hai
  TEXTFIX_VLM_URL, TEXTFIX_VLM_MODEL, TEXTFIX_VLM_API_KEY  VLM riêng (không đặt: dùng LLM)
  TEXTFIX_LLM_TIMEOUT, _MAX_TOKENS, _RETRIES               giây chờ MỘT lần gọi (40), trần độ dài trả lời (4096), số lần gọi lại (2)
  TEXTFIX_FLUX_URL, TEXTFIX_FLUX_API_KEY, ..._CONCURRENCY  FLUX ở máy khác (flux_api.py, docs/DEPLOY.md)
  TEXTFIX_WEIGHTS, TEXTFIX_DEVICE_DIT / _TE / _AE          FLUX.2-klein tại chỗ (xem flux.py)
  TEXTFIX_DRAW=qwen, TEXTFIX_QWEN_DIR / _MEM / ...         vẽ nháp bằng Qwen-Image tại chỗ thay FLUX (draw/qwen.py)
  PLAYWRIGHT_CHROME_PATH                                   Chromium có sẵn (máy chủ không tải được bản của Playwright)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRAWERS = ("flux", "qwen")


def load_env() -> None:
    """Nạp .env vào biến môi trường (không ghi đè biến đã có, không in giá trị), rồi áp model vẽ (draw_profile)."""
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    draw_profile()


def draw_profile() -> str:
    """MODEL VẼ NHÁP (TEXTFIX_DRAW, mặc định flux; start_server.sh --draw qwen). qwen: Qwen-Image nạp TẠI CHỖ trên mỗi card,
    cùng card với Hi-SAM / LaMa (draw/qwen.py) -> bỏ TEXTFIX_FLUX_URL của tiến trình này (.env của FLUX 9B giữ nguyên, bỏ cờ
    là quay về)."""
    name = (os.environ.get("TEXTFIX_DRAW") or "flux").strip().lower()
    if name not in DRAWERS:
        raise RuntimeError(f"TEXTFIX_DRAW phải là một trong {', '.join(DRAWERS)}, không phải {name!r}")
    if name == "qwen":
        os.environ.pop("TEXTFIX_FLUX_URL", None)
    return name


def models() -> dict:
    """{llm: {url, model, key_env}, vlm: {...}} từ biến môi trường."""
    url, model = os.environ.get("TEXTFIX_LLM_URL"), os.environ.get("TEXTFIX_MODEL")
    vurl = os.environ.get("TEXTFIX_VLM_URL") or url
    vmodel = os.environ.get("TEXTFIX_VLM_MODEL") or model
    return {"llm": {"url": url, "model": model, "key_env": "TEXTFIX_LLM_API_KEY"},
            "vlm": {"url": vurl, "model": vmodel, "key_env": "TEXTFIX_VLM_API_KEY"}}
