"""Lớp API (server/app.py) không cần model: đầu vào sai trả 422 có mã cố định, model chưa nạp trả 503 (docs/API.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402

client = TestClient(server.app)   # không "with": không chạy lifespan, không nạp model


def post(**body):
    return client.post("/api/jobs", json=body)


@pytest.mark.parametrize("body, code", [
    ({"category": "flyer", "discount": "20%"}, "invalid_category"),
    ({"category": "promo", "discount": "20%", "aspect_ratio": "5:7"}, "invalid_aspect_ratio"),
    ({"category": "promo", "discount": "20%", "num_images": 9}, "invalid_num_images"),
    ({"category": "promo", "discount": "20%", "image_base64": "không phải ảnh"}, "invalid_image"),
    ({"category": "promo"}, "missing_required_fields"),
    ({"category": "recruitment", "job_position": "Pha chế"}, "missing_required_fields"),
])
def test_bad_input_is_422_with_code(body, code):
    r = post(**body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == code


@pytest.mark.parametrize("body, code", [
    ({"image_url": "http://bucket.s3.amazonaws.com/a.jpg"}, "invalid_image_url"),          # không https
    ({"image_url": "https://evil.example.com/a.jpg"}, "invalid_image_url"),                # host ngoài danh sách
    ({"image_url": "https://amazonaws.com.evil.io/a.jpg"}, "invalid_image_url"),           # đuôi giả
    ({"image_url": "https://b.s3.amazonaws.com/a.jpg", "image_base64": "xx"}, "invalid_image"),   # cả hai trường
])
def test_bad_image_url_is_422_without_network(body, code):
    r = post(category="promo", prompt="poster giảm giá", **body)
    assert r.status_code == 422 and r.json()["detail"]["code"] == code


def _png() -> bytes:
    import io

    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(b, "PNG")
    return b.getvalue()


class _Resp:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, n: int = -1) -> bytes:
        return self.data if n < 0 else self.data[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_net(monkeypatch, ip: str, data: bytes):
    import contract
    monkeypatch.setattr(contract.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (ip, 443))])
    opened = []

    class Opener:
        def open(self, req, timeout=None):
            opened.append(req.full_url)
            return _Resp(data)
    monkeypatch.setattr(contract.urllib.request, "build_opener", lambda *a: Opener())
    return opened


def test_image_url_is_fetched(monkeypatch):
    import contract
    opened = _fake_net(monkeypatch, "52.216.1.1", _png())
    assert contract.fetch_image("https://b.s3.ap-southeast-1.amazonaws.com/x.png") == _png()
    assert opened == ["https://b.s3.ap-southeast-1.amazonaws.com/x.png"]


@pytest.mark.parametrize("ip, data, code", [
    ("10.0.0.5", None, "invalid_image_url"),          # host hợp lệ nhưng phân giải vào mạng nội bộ: không tải
    ("169.254.169.254", None, "invalid_image_url"),   # địa chỉ metadata của máy ảo
    ("52.216.1.1", b"<html>not an image</html>", "invalid_image"),
    ("52.216.1.1", "big", "image_fetch_failed"),   # quá giới hạn dung lượng
], ids=["private", "metadata", "not_image", "too_big"])
def test_image_url_rejected(monkeypatch, ip, data, code):
    import contract
    monkeypatch.setenv("TEXTFIX_IMAGE_MAX_MB", "1")
    data = b"x" * (2 * 2 ** 20) if data == "big" else data
    opened = _fake_net(monkeypatch, ip, data or _png())
    with pytest.raises(server.HTTPException) as e:
        contract.fetch_image("https://b.s3.amazonaws.com/x.png")
    assert e.value.detail["code"] == code
    if data is None:
        assert opened == []   # chặn trước khi gọi mạng


def test_required_fields_listed():
    r = post(category="recruitment", job_position="Pha chế")
    assert r.json()["detail"]["fields"] == ["apply_deadline", "apply_method"]


@pytest.mark.parametrize("body", [
    {"category": "guide", "title": "Đặt hàng 3 bước"},          # hướng dẫn: không trường nào bắt buộc (như app)
    {"category": "promo", "prompt": "poster giảm giá cuối tuần"},   # có prompt: prompt quyết, không đòi trường
])
def test_valid_input_reaches_model_check(body):
    r = post(**body)
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_not_ready"


def test_fields_of_other_categories_are_reported():
    req = server.GenerateRequest(category="promo", discount="20%", job_position="Pha chế", foo="x")
    assert server.ignored_fields(req) == ["job_position", "foo"]


def test_remote_flux_draw_threads_follow_concurrency_not_gpus(monkeypatch):
    # FLUX ở máy khác: số luồng vẽ = số nháp gửi cùng lúc, chia vòng lên card (chỉ để chạy Hi-SAM / LaMa); không card: CPU
    monkeypatch.setenv("TEXTFIX_FLUX_URL", "https://api.test/v1/flux-2-klein-9b")
    monkeypatch.setenv("TEXTFIX_FLUX_CONCURRENCY", "3")
    monkeypatch.setattr(server, "gpu_list", lambda: [0, 1])
    assert server.draw_devices() == ["cuda:0", "cuda:1", "cuda:0"]
    monkeypatch.setattr(server, "gpu_list", lambda: [])
    assert server.draw_devices() == ["cpu"] * 3


def test_flux_moderation_has_its_own_code():
    from textfix.draw.remote import FluxError, FluxModerated
    assert server.draw_error(FluxModerated("Content Moderated"), "FLUX vẽ nháp")["code"] == "draw_moderated"
    assert server.draw_error(FluxError("HTTP 500"), "FLUX vẽ nháp")["code"] == "draw_failed"


# ---------------------------------------------------------------------------------------------- lấy ảnh: /api/jobs/{id}/poster
def fake_job(monkeypatch, tmp_path, status="done", posters=("done",)):
    """Job giả trong bộ nhớ máy chủ + tệp poster thật trong thư mục lượt chạy tạm."""
    import io

    from PIL import Image
    monkeypatch.setattr(server, "RUNS", tmp_path)
    P = []
    for i, st in enumerate(posters):
        p = {"index": i, "status": st}
        if st == "done":
            (tmp_path / "run_x" / f"v{i}").mkdir(parents=True)
            buf = io.BytesIO()
            Image.new("RGB", (8, 6), (10 * i, 0, 0)).save(buf, "PNG")
            (tmp_path / "run_x" / f"v{i}" / "poster.png").write_bytes(buf.getvalue())
            p["poster_url"] = f"outputs/run_x/v{i}/poster.png"
        if st == "error":
            p["error"] = {"code": "vlm_failed", "stage": "VLM", "message": "x"}
        P.append(p)
    job = {"id": "j1", "run_id": "run_x", "status": status, "progress": "", "created": 0, "error": None,
           "result": {"job_id": "j1", "width": 8, "height": 6, "posters": P}}
    monkeypatch.setitem(server.JOBS, "j1", job)
    return job


def test_poster_is_one_get_with_the_job_id(monkeypatch, tmp_path):
    fake_job(monkeypatch, tmp_path, posters=("done", "done"))
    assert client.get("/api/jobs/j1").json()["poster"] == "/api/jobs/j1/poster"
    r = client.get("/api/jobs/j1/poster")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    b = client.get("/api/jobs/j1/poster", params={"index": 1, "format": "base64"}).json()
    assert (b["index"], b["width"], b["mime"]) == (1, 8, "image/png") and len(b["image_base64"]) > 20


def test_poster_not_ready_failed_and_missing(monkeypatch, tmp_path):
    fake_job(monkeypatch, tmp_path, status="running", posters=("drawing",))
    assert client.get("/api/jobs/j1/poster").json()["detail"]["code"] == "not_ready"
    assert "poster" not in client.get("/api/jobs/j1").json()
    fake_job(monkeypatch, tmp_path / "b", posters=("done", "error"))
    r = client.get("/api/jobs/j1/poster", params={"index": 1})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "poster_failed"
    assert client.get("/api/jobs/j1/poster", params={"index": 5}).json()["detail"]["code"] == "poster_not_found"
    assert client.get("/api/jobs/nope/poster").status_code == 404
    (tmp_path / "b" / "run_x" / "v0" / "poster.png").unlink()
    assert client.get("/api/jobs/j1/poster").status_code == 410


def test_ocr_threads_follow_container_cpu_limit(monkeypatch):
    # docker/run.sh đặt TEXTFIX_THREADS theo hạn mức CPU -> onnxruntime của OCR mở đúng ngần ấy luồng (không theo số lõi cả máy)
    pytest.importorskip("rapidocr_onnxruntime")
    from rapidocr_onnxruntime import utils as ru

    from textfix.lines import ocr
    monkeypatch.setattr(ru, "SessionOptions", ru.SessionOptions)   # trả lại như cũ sau test
    monkeypatch.setenv("TEXTFIX_THREADS", "3")
    ocr._limit_threads()
    ocr._limit_threads()   # gọi lại không bọc chồng
    o = ru.SessionOptions()
    assert o.intra_op_num_threads == 3 and ru.SessionOptions.textfix_base.__name__ == "SessionOptions"
