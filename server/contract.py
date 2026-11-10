"""HỢP ĐỒNG API của máy chủ (docs/API.md): form từng loại, kiểm đầu vào (422 có mã cố định), mã lỗi trả cho người gọi.

Không phụ thuộc model hay dây chuyền: server/app.py dùng, test (tests/test_api.py) và script (scripts/brief_probe.py) gọi thẳng.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

# trường NỘI DUNG từng loại form (khớp UI); trường chung cửa hàng gửi cho mọi loại
CATEGORY_FIELDS = {
    "promo": ["title", "discount", "applied_product", "date_start", "date_end"],
    "product_intro": ["title", "product_name", "product_desc", "highlights", "price"],
    "opening": ["title", "opening_date", "opening_promo", "booking_contact"],
    "feedback": ["title", "feedback_target", "feedback_quote", "feedback_highlights", "special_offer"],
    "recruitment": ["title", "job_position", "job_desc", "apply_deadline", "apply_method"],
    "guide": ["title"],
}
REQUIRED = {"promo": ["discount"], "product_intro": ["product_name", "product_desc"], "opening": ["opening_date"],
            "feedback": ["feedback_target", "feedback_quote"], "recruitment": ["job_position", "apply_deadline", "apply_method"],
            "guide": []}   # như form của app: hướng dẫn không trường nào bắt buộc (không bước nào: LLM viết theo tiêu đề / prompt)
COMMON = ["store_name", "phone", "address", "website_link"]
MAX_IMAGES = 4
RATIOS = ["1:1", "4:5", "9:16", "16:9"]   # 4 tỉ lệ của app (bộ vẽ có thêm 3:4, 2:3, 4:3 -- flux.SIZES -- app không dùng)


class ApiError(Exception):
    """Lỗi trả cho người gọi API: mã cố định (docs/API.md), bước hỏng, thông điệp."""

    def __init__(self, code: str, stage: str, message: str = ""):
        super().__init__(message or code)
        self.code, self.stage, self.message = code, stage, message or code

    def as_dict(self) -> dict:
        return {"code": self.code, "stage": self.stage, "message": self.message}


def failure(code: str, stage: str, e: BaseException | None = None) -> dict:
    return {"code": code, "stage": stage, "message": f"{type(e).__name__}: {e}" if e else code}


def draw_error(e: BaseException, stage: str) -> dict:
    """Trạm VẼ hỏng: máy FLUX lọc nội dung -> draw_moderated (không phải lỗi của TextFix), còn lại draw_failed."""
    from textfix.draw.remote import FluxModerated
    return failure("draw_moderated" if isinstance(e, FluxModerated) else "draw_failed", stage, e)


# bước thiết kế (engine.design báo qua progress) -> mã lỗi khi hỏng ở bước đó
DESIGN_STAGES = (("đo dòng", "lines_failed"), ("gỡ chữ", "erase_failed"), ("VLM", "vlm_failed"), ("dựng", "render_failed"))


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")   # khoá lạ không làm hỏng yêu cầu: báo lại trong "ignored_fields"
    category: str = "promo"
    # thông tin chung theo loại
    title: Optional[str] = ""
    discount: Optional[str] = ""
    applied_product: Optional[str] = ""
    date_start: Optional[str] = ""
    date_end: Optional[str] = ""
    product_name: Optional[str] = ""
    product_desc: Optional[str] = ""
    highlights: Optional[str] = ""
    price: Optional[str] = ""
    opening_date: Optional[str] = ""
    opening_promo: Optional[str] = ""
    booking_contact: Optional[str] = ""
    feedback_target: Optional[str] = ""
    feedback_quote: Optional[str] = ""
    feedback_highlights: Optional[str] = ""
    special_offer: Optional[str] = ""
    job_position: Optional[str] = ""
    job_desc: Optional[str] = ""
    apply_deadline: Optional[str] = ""
    apply_method: Optional[str] = ""
    guide_steps: Optional[List[str]] = None
    prompt: Optional[str] = ""
    image_base64: Optional[str] = None   # ảnh sản phẩm tải lên (data URI hoặc base64)
    image_url: Optional[str] = None      # ... hoặc link ảnh public (https, host trong TEXTFIX_IMAGE_HOSTS): máy chủ tự tải
    # cửa hàng
    store_name: Optional[str] = ""
    phone: Optional[str] = ""
    address: Optional[str] = ""
    website_link: Optional[str] = ""
    # hiển thị / thiết kế
    style_pref: Optional[str] = "auto"
    primary_color: Optional[str] = None
    aspect_ratio: str = "1:1"
    num_images: int = 1
    seed: Optional[int] = None


def form_of(req: GenerateRequest) -> dict:
    """Chỉ trường thuộc loại form đang chọn + trường cửa hàng, bỏ ô trống. Các bước hướng dẫn -> step_1, step_2..."""
    cat = req.category if req.category in CATEGORY_FIELDS else "promo"
    out: dict = {"category": cat}
    for k in CATEGORY_FIELDS[cat] + COMMON:
        v = (getattr(req, k) or "").strip()
        if v:
            out[k] = v
    if cat == "guide":
        for i, s in enumerate([s.strip() for s in req.guide_steps or [] if s and s.strip()], 1):
            out[f"step_{i}"] = s
    return out


def missing_fields(req: GenerateRequest, form: dict) -> list[str]:
    """Trường bắt buộc -- chỉ khi KHÔNG có prompt (có prompt thì prompt quyết)."""
    if (req.prompt or "").strip():
        return []
    return [k for k in REQUIRED.get(form["category"], []) if k not in form]


def ignored_fields(req: GenerateRequest) -> list[str]:
    """Trường có giá trị nhưng không thuộc loại form đang chọn, và khoá lạ: bỏ qua, báo lại cho người gọi."""
    own = set(CATEGORY_FIELDS.get(req.category, [])) | set(COMMON) | ({"guide_steps"} if req.category == "guide" else set())
    content = {k for fs in CATEGORY_FIELDS.values() for k in fs} | {"guide_steps"}
    return [k for k in sorted(content - own) if getattr(req, k, None)] + sorted((req.model_extra or {}).keys())


def validate(req: GenerateRequest) -> None:
    """Đầu vào sai -> 422 với mã cố định (người gọi sửa được; không tự đoán thay)."""
    if req.category not in CATEGORY_FIELDS:
        raise HTTPException(422, {"code": "invalid_category", "value": req.category, "allowed": sorted(CATEGORY_FIELDS)})
    if req.aspect_ratio not in RATIOS:
        raise HTTPException(422, {"code": "invalid_aspect_ratio", "value": req.aspect_ratio, "allowed": RATIOS})
    if not 1 <= int(req.num_images or 0) <= MAX_IMAGES:
        raise HTTPException(422, {"code": "invalid_num_images", "value": req.num_images, "allowed": [1, MAX_IMAGES]})
    if req.image_base64 and req.image_url:
        raise HTTPException(422, {"code": "invalid_image", "message": "chỉ gửi một trong image_base64 / image_url"})
    if req.image_url:
        check_image_url(req.image_url)   # chỉ kiểm khuôn + host (không gọi mạng); tải ở fetch_image
    if req.image_base64:
        try:
            decode_image(req.image_base64)
        except Exception as e:
            raise HTTPException(422, {"code": "invalid_image", "message": f"{type(e).__name__}: {e}"}) from e


# ---------------------------------------------------------------------------------------------------- ảnh theo link
# Người gọi (chatbot) gửi link ảnh public thay base64; máy chủ tự tải. Link do người ngoài đưa -> chặn để máy chủ không bị dùng
# gọi vào mạng nội bộ: chỉ https, host trong danh sách cho phép, mọi IP phân giải ra phải là IP công khai, không theo chuyển hướng,
# giới hạn thời gian và dung lượng.
def image_hosts() -> list[str]:
    """Host được phép (đuôi tên miền, phẩy): TEXTFIX_IMAGE_HOSTS, mặc định amazonaws.com (S3 public). "*" = mọi host công khai."""
    return [h.strip().lower().lstrip(".") for h in os.environ.get("TEXTFIX_IMAGE_HOSTS", "amazonaws.com").split(",") if h.strip()]


def check_image_url(url: str) -> str:
    """Khuôn link + host cho phép -> host. Sai -> 422 invalid_image_url."""
    def bad(msg: str):
        raise HTTPException(422, {"code": "invalid_image_url", "message": msg, "allowed_hosts": image_hosts()})
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme != "https" or not u.hostname:
        bad("image_url phải là link https://...")
    host = u.hostname.lower()
    if "*" not in image_hosts() and not any(host == h or host.endswith("." + h) for h in image_hosts()):
        bad(f"host {host} không nằm trong danh sách cho phép")
    return host


def _public_host(host: str, port: int) -> None:
    try:
        ips = {a[4][0] for a in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)}
    except OSError as e:
        raise HTTPException(422, {"code": "image_fetch_failed", "message": f"không phân giải được {host}: {e}"}) from e
    for ip in ips:
        if not ipaddress.ip_address(ip.split("%")[0]).is_global:
            raise HTTPException(422, {"code": "invalid_image_url", "message": f"{host} trỏ vào địa chỉ không công khai"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None   # chuyển hướng -> HTTPError 3xx (không đi theo: có thể dẫn vào host không được phép)


def fetch_image(url: str) -> bytes:
    """Tải ảnh theo link (đã qua check_image_url): TEXTFIX_IMAGE_TIMEOUT giây (15), tối đa TEXTFIX_IMAGE_MAX_MB (15).
    Hỏng -> 422 image_fetch_failed / invalid_image_url; tải được nhưng không phải ảnh -> 422 invalid_image."""
    host = check_image_url(url)
    # CDN nội bộ (DNS trong mạng công ty trả IP 10.x): TEXTFIX_IMAGE_ALLOW_PRIVATE=1 bỏ kiểm IP cho host ĐÃ trong danh sách cho
    # phép; không áp dụng khi danh sách là "*" (khi đó link bất kỳ có thể trỏ vào mạng nội bộ)
    if not (os.environ.get("TEXTFIX_IMAGE_ALLOW_PRIVATE") == "1" and "*" not in image_hosts()):
        _public_host(host, urllib.parse.urlsplit(url.strip()).port or 443)
    limit = int(float(os.environ.get("TEXTFIX_IMAGE_MAX_MB", "15")) * 2 ** 20)
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url.strip(), headers={"User-Agent": "textfix-image-fetch"})
    try:
        with opener.open(req, timeout=float(os.environ.get("TEXTFIX_IMAGE_TIMEOUT", "15"))) as r:
            raw = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        raise HTTPException(422, {"code": "image_fetch_failed", "message": f"HTTP {e.code} khi tải ảnh"}) from e
    except (urllib.error.URLError, OSError) as e:
        raise HTTPException(422, {"code": "image_fetch_failed", "message": f"{type(e).__name__}: {e}"}) from e
    if len(raw) > limit:
        raise HTTPException(422, {"code": "image_fetch_failed", "message": f"ảnh lớn hơn {limit // 2 ** 20} MB"})
    try:
        decode_image(base64.b64encode(raw).decode())
    except Exception as e:
        raise HTTPException(422, {"code": "invalid_image", "message": f"link không trả về ảnh đọc được: {type(e).__name__}"}) from e
    return raw


def decode_image(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    from PIL import Image
    raw = base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1024, 1024))
    return np.asarray(im)
