"""HỢP ĐỒNG API (docs/API_V1.md): MỘT endpoint đồng bộ POST /api/v1/images/generations cho chatbot.

Thân: {prompt, images_base64?, aspect_ratio?}. Trả: JSON {status, code, message, request_id, data:
{image_base64, ...}} hoặc ảnh PNG thô (Accept: image/png). Lỗi: khuôn {status: "ERROR", code, error_code, message, request_id,
data: null}, mã cố định ở docs/API_V1.md.
Không phụ thuộc model hay dây chuyền: server/app.py dùng, test (tests/test_api.py) gọi thẳng.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict

RATIOS = ["1:1", "9:16", "16:9", "3:4", "4:3"]   # các tỉ lệ chatbot dùng; đều có trong textfix/draw/flux.py SIZES


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")   # khoá lạ: bỏ qua, ghi log
    prompt: str = ""
    # ảnh tham chiếu gửi thẳng trong thân (base64 hoặc data URI). 12/11: đội chatbot chốt chỉ trường này -- bỏ image_base64 và
    # image_urls (máy chủ prod không ra được kho ảnh của router nên tải theo link cũng không dùng được)
    images_base64: List[str] = []
    aspect_ratio: str = "1:1"

    def b64_images(self) -> list[str]:
        return [s for s in self.images_base64 if (s or "").strip()]


class V1Error(Exception):
    """Lỗi trả cho người gọi: mã HTTP + error_code cố định (docs/API_V1.md)."""

    def __init__(self, http: int, error_code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.http, self.error_code, self.message, self.retry_after = http, error_code, message, retry_after


class ApiError(Exception):
    """Lỗi của dây chuyền: mã nội bộ (llm_failed, draw_moderated, vlm_failed...), bước hỏng, chi tiết (chỉ ghi log)."""

    def __init__(self, code: str, stage: str, message: str = ""):
        super().__init__(message or code)
        self.code, self.stage, self.message = code, stage, message or code


def draw_error(e: BaseException, stage: str) -> ApiError:
    """Trạm VẼ hỏng: máy FLUX lọc nội dung -> draw_moderated (không phải lỗi của TextFix), còn lại draw_failed."""
    from textfix.draw.remote import FluxModerated
    return ApiError("draw_moderated" if isinstance(e, FluxModerated) else "draw_failed", stage, f"{type(e).__name__}: {e}")


# bước thiết kế (engine.design báo qua progress) -> mã lỗi khi hỏng ở bước đó
DESIGN_STAGES = (("đo dòng", "lines_failed"), ("gỡ chữ", "erase_failed"), ("VLM", "vlm_failed"), ("dựng", "render_failed"))
# mã lỗi nội bộ -> lỗi trả về. Còn lại: 500 ERR_IMAGE_GENERATION_FAILED
JOB_ERRORS = {"draw_moderated": (422, "ERR_IMAGE_SAFETY_VIOLATION")}


def job_error(e: ApiError) -> V1Error:
    """Thông điệp chỉ mã + bước hỏng: chi tiết ngoại lệ có thể chứa địa chỉ nội bộ (LLM / FLUX), chỉ ghi log (tra theo request_id)."""
    http, ec = JOB_ERRORS.get(e.code, (500, "ERR_IMAGE_GENERATION_FAILED"))
    return V1Error(http, ec, e.code + (f" ({e.stage})" if e.stage else ""))


def check(req: Request) -> None:
    """Đầu vào sai -> 400 (trước khi gọi mạng / xếp hàng)."""
    if not (req.prompt or "").strip():
        raise V1Error(400, "ERR_EMPTY_PROMPT", "prompt rỗng")
    if (req.aspect_ratio or "1:1") not in RATIOS:
        raise V1Error(400, "ERR_INVALID_ASPECT_RATIO", f"aspect_ratio {req.aspect_ratio!r} không thuộc {', '.join(RATIOS)}")


def max_bytes() -> int:
    return int(float(os.environ.get("TEXTFIX_IMAGE_MAX_MB", "15")) * 2 ** 20)


def check_b64_image(s: str) -> np.ndarray:
    """Ảnh base64 trong thân (hoặc data URI) -> ảnh RGB (cạnh dài <= 1024). Sai -> 400 ERR_INVALID_IMAGE."""
    raw_s = s.split(",", 1)[1] if s.lstrip().startswith("data:") and "," in s else s
    try:
        raw = base64.b64decode("".join(raw_s.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise V1Error(400, "ERR_INVALID_IMAGE", f"ảnh base64 không giải mã được: {type(e).__name__}") from e
    if len(raw) > max_bytes():
        raise V1Error(400, "ERR_INVALID_IMAGE", f"ảnh lớn hơn {max_bytes() // 2 ** 20} MB")
    try:
        return decode_image(raw)
    except Exception as e:
        raise V1Error(400, "ERR_INVALID_IMAGE", f"không đọc được ảnh: {type(e).__name__}") from e


def decode_image(raw: bytes) -> np.ndarray:
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1024, 1024))
    return np.asarray(im)


# ---------------------------------------------------------------------------------------------------- trả lời
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


def timing_header(timing: dict) -> str:
    """X-Timing: giây từng bước, "llm=3.5;ve_nhap=2.4;..." -- đo tốc độ qua đúng đường chatbot gọi."""
    return ";".join(f"{k}={v:g}" for k, v in timing.items() if isinstance(v, (int, float)))
