"""HỢP ĐỒNG API v1 cho chatbot (docs/API_V1.md): MỘT endpoint đồng bộ POST /api/v1/images/generations.

Thân: {prompt, image_urls?, aspect_ratio?}. Trả: JSON {status, code, message, request_id, data: {image_base64, ...}} hoặc ảnh PNG thô
(Accept: image/png). Lỗi: khuôn {status: "ERROR", code, error_code, message, request_id, data: null}, mã cố định ở bảng dưới.
Không phụ thuộc model hay dây chuyền: server/app.py dùng, test (tests/test_api_v1.py) gọi thẳng. Lõi vẫn là job của API demo
(server/contract.py, app.run_job): v1 chỉ là lớp vỏ.
"""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

V1_RATIOS = ["1:1", "9:16", "16:9", "3:4", "4:3"]   # các tỉ lệ chatbot dùng; đều có trong textfix/draw/flux.py SIZES


class V1Request(BaseModel):
    model_config = ConfigDict(extra="allow")   # khoá lạ: bỏ qua, ghi log
    prompt: str = ""
    image_urls: List[str] = []
    # ảnh gửi thẳng trong thân (base64 hoặc data URI) -- máy chủ prod không ra được kho ảnh của router (11/11). Thắng image_urls.
    images_base64: List[str] = []
    image_base64: Optional[str] = None   # một ảnh: như images_base64 = [image_base64]
    aspect_ratio: str = "1:1"

    def b64_images(self) -> list[str]:
        return [s for s in ([self.image_base64] if self.image_base64 else []) + list(self.images_base64) if (s or "").strip()]


class V1Error(Exception):
    """Lỗi trả cho người gọi v1: mã HTTP + error_code cố định (docs/API_V1.md)."""

    def __init__(self, http: int, error_code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.http, self.error_code, self.message, self.retry_after = http, error_code, message, retry_after


# mã lỗi của job (docs/API.md) -> lỗi v1. Còn lại: 500 ERR_IMAGE_GENERATION_FAILED (mã nội bộ giữ trong message để tra log)
JOB_ERRORS = {"draw_moderated": (422, "ERR_IMAGE_SAFETY_VIOLATION")}
# lỗi đầu vào của contract.fetch_image (422 của API demo) -> lỗi tải ảnh của v1
IMAGE_ERRORS = ("invalid_image_url", "image_fetch_failed", "invalid_image")


def check(req: V1Request) -> None:
    """Đầu vào sai -> 400 (trước khi gọi mạng / xếp hàng)."""
    if not (req.prompt or "").strip():
        raise V1Error(400, "ERR_EMPTY_PROMPT", "prompt rỗng")
    if (req.aspect_ratio or "1:1") not in V1_RATIOS:
        raise V1Error(400, "ERR_INVALID_ASPECT_RATIO", f"aspect_ratio {req.aspect_ratio!r} không thuộc {', '.join(V1_RATIOS)}")


def check_b64_image(s: str) -> str:
    """Ảnh base64 trong thân -> chuỗi base64 thuần, đã kiểm đọc được (cùng giới hạn dung lượng với ảnh tải theo link).
    Sai -> 400 ERR_INVALID_IMAGE."""
    import base64
    import binascii

    from contract import decode_image
    raw_s = s.split(",", 1)[1] if s.lstrip().startswith("data:") and "," in s else s
    limit = int(float(os.environ.get("TEXTFIX_IMAGE_MAX_MB", "15")) * 2 ** 20)
    try:
        raw = base64.b64decode("".join(raw_s.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise V1Error(400, "ERR_INVALID_IMAGE", f"ảnh base64 không giải mã được: {type(e).__name__}") from e
    if len(raw) > limit:
        raise V1Error(400, "ERR_INVALID_IMAGE", f"ảnh lớn hơn {limit // 2 ** 20} MB")
    b64 = base64.b64encode(raw).decode()
    try:
        decode_image(b64)
    except Exception as e:
        raise V1Error(400, "ERR_INVALID_IMAGE", f"không đọc được ảnh: {type(e).__name__}") from e
    return b64


def job_error(err: dict | None) -> V1Error:
    """Lỗi của job ({code, stage, message}) -> V1Error. Thông điệp chỉ mã + bước hỏng: chi tiết ngoại lệ có thể chứa địa chỉ
    nội bộ (LLM / FLUX), chỉ ghi log máy chủ (tra theo request_id)."""
    err = err or {}
    code = err.get("code") or "internal_error"
    http, ec = JOB_ERRORS.get(code, (500, "ERR_IMAGE_GENERATION_FAILED"))
    return V1Error(http, ec, f"{code}" + (f" ({err['stage']})" if err.get("stage") else ""))


def image_error(detail: dict) -> V1Error:
    """422 của contract.fetch_image -> 400 ERR_IMAGE_DOWNLOAD_FAILED."""
    return V1Error(400, "ERR_IMAGE_DOWNLOAD_FAILED", f"{detail.get('code')}: {detail.get('message', '')}")


def body_error(e: V1Error, request_id: str) -> dict:
    return {"status": "ERROR", "code": e.http, "error_code": e.error_code, "message": e.message, "request_id": request_id,
            "data": None}


def tokens() -> float:
    """Điểm compute mỗi ảnh (trừ ví): bản đầu là một mức cố định, TEXTFIX_V1_TOKENS (mặc định 1.0)."""
    return float(os.environ.get("TEXTFIX_V1_TOKENS", "1.0"))


def body_success(request_id: str, image_b64: str, aspect: str, width: int, height: int, ms: int) -> dict:
    return {"status": "SUCCESS", "code": 200, "message": "Image generated successfully", "request_id": request_id,
            "data": {"image_base64": image_b64, "mime_type": "image/png", "aspect_ratio": aspect, "width": width,
                     "height": height, "tokens": tokens(), "execution_time_ms": ms}}


def headers(request_id: str, aspect: str, ms: int) -> dict:
    """Header kèm ảnh PNG thô (Accept: image/png)."""
    return {"X-Request-Id": request_id, "X-Aspect-Ratio": aspect, "X-Token-Cost": f"{tokens():g}", "X-Execution-Time-Ms": str(ms)}
