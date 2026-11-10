"""API v1 cho chatbot (server/contract_v1.py, POST /api/v1/images/generations) không cần model: dây chuyền thay bằng `work` giả."""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402

client = TestClient(server.app)   # không "with": không nạp model
URL = "/api/v1/images/generations"


def _png(w=8, h=6) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def ready(monkeypatch, tmp_path):
    """Máy chủ 'đã nạp model'; `work` giả: outcome["error"] = mã lỗi job, outcome["sleep"] = giây chạy. Trả các yêu cầu đã nhận."""
    monkeypatch.setitem(server.STATE, "flux", [object()])
    monkeypatch.setitem(server.STATE, "vlm", object())
    monkeypatch.setitem(server.STATE, "ready", True)
    monkeypatch.setitem(server.STATE, "error", None)
    monkeypatch.setattr(server, "RUNS", tmp_path)
    outcome, seen = {}, []

    def work(job, req):
        seen.append(req)
        time.sleep(outcome.get("sleep", 0))
        if outcome.get("error"):
            job.update(status="error", error={"code": outcome["error"], "stage": "x", "message": "hỏng"})
        else:
            (tmp_path / job["run_id"] / "v0").mkdir(parents=True)
            (tmp_path / job["run_id"] / "v0" / "poster.png").write_bytes(_png())
            job.update(status="done", result={"width": 8, "height": 6, "posters": [
                {"index": 0, "status": "done", "poster_url": f"outputs/{job['run_id']}/v0/poster.png"}]})
        server.ORDER.remove(job["id"])
    monkeypatch.setattr(server, "work", work)
    return outcome, seen


def test_success_json_has_spec_shape(ready):
    _, seen = ready
    r = client.post(URL, json={"prompt": "Poster quán cà phê", "aspect_ratio": "9:16"})
    assert r.status_code == 200
    b = r.json()
    assert (b["status"], b["code"]) == ("SUCCESS", 200) and b["request_id"] == r.headers["x-request-id"]
    d = b["data"]
    assert base64.b64decode(d["image_base64"]) == _png()
    assert (d["mime_type"], d["aspect_ratio"], d["width"], d["height"], d["tokens"]) == ("image/png", "9:16", 8, 6, 1.0)
    assert isinstance(d["execution_time_ms"], int)
    assert (seen[0].prompt, seen[0].aspect_ratio, seen[0].num_images, seen[0].category) == ("Poster quán cà phê", "9:16", 1, "promo")


def test_png_bytes_and_request_id_echo(ready):
    r = client.post(URL, json={"prompt": "x"}, headers={"Accept": "image/png", "X-Request-Id": "abc-123"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == _png()
    assert (r.headers["x-request-id"], r.headers["x-aspect-ratio"], r.headers["x-token-cost"]) == ("abc-123", "1:1", "1")
    assert int(r.headers["x-execution-time-ms"]) >= 0


@pytest.mark.parametrize("ratio", ["1:1", "9:16", "16:9", "3:4", "4:3"])
def test_all_spec_ratios_accepted(ready, ratio):
    assert client.post(URL, json={"prompt": "x", "aspect_ratio": ratio}).status_code == 200


@pytest.mark.parametrize("body, http, code", [
    ({"prompt": "   "}, 400, "ERR_EMPTY_PROMPT"),
    ({"aspect_ratio": "1:1"}, 400, "ERR_EMPTY_PROMPT"),
    ({"prompt": "x", "aspect_ratio": "4:5"}, 400, "ERR_INVALID_ASPECT_RATIO"),
    ({"prompt": "x", "image_urls": ["https://evil.example.com/a.png"]}, 400, "ERR_IMAGE_DOWNLOAD_FAILED"),
    ({"prompt": "x", "image_urls": "không phải mảng"}, 400, "ERR_INVALID_REQUEST"),
    ({"prompt": ["x"]}, 400, "ERR_INVALID_REQUEST"),
])
def test_bad_input_has_spec_error_shape(ready, body, http, code):
    r = client.post(URL, json=body, headers={"X-Request-Id": "rid"})
    b = r.json()
    assert r.status_code == http and (b["status"], b["code"], b["error_code"], b["request_id"], b["data"]) == \
        ("ERROR", http, code, "rid", None)


def test_body_not_json_is_400(ready):
    r = client.post(URL, content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error_code"] == "ERR_INVALID_REQUEST"


def test_demo_api_still_answers_422(ready):
    assert client.post("/api/jobs", json={"category": ["x"]}).status_code == 422   # chỉ v1 đổi khuôn lỗi kiểm đầu vào


def test_first_image_url_becomes_product_image(ready, monkeypatch):
    _, seen = ready
    got = []
    monkeypatch.setattr(server, "fetch_image", lambda u: got.append(u) or _png(4, 4))
    r = client.post(URL, json={"prompt": "x", "image_urls": ["https://a/1.png", "https://a/2.png"]})
    assert r.status_code == 200 and got == ["https://a/1.png"]   # bản đầu: chỉ ảnh đầu
    assert base64.b64decode(seen[0].image_base64) == _png(4, 4)
    assert seen[0].model_extra["v1_image_urls"] == ["https://a/1.png", "https://a/2.png"]   # ghi vào request.json để tra


@pytest.mark.parametrize("field, value", [
    ("images_base64", lambda b: [b, "bỏ qua ảnh sau"]),
    ("images_base64", lambda b: ["data:image/png;base64," + b]),
    ("image_base64", lambda b: b),
])
def test_base64_image_in_body(ready, monkeypatch, field, value):
    # máy chủ prod không ra được kho ảnh: ảnh gửi thẳng base64, thắng image_urls (không tải link nào)
    _, seen = ready
    monkeypatch.setattr(server, "fetch_image", lambda u: pytest.fail("không được tải link khi đã có base64"))
    b = base64.b64encode(_png(4, 4)).decode()
    r = client.post(URL, json={"prompt": "x", field: value(b), "image_urls": ["https://a/1.png"]})
    assert r.status_code == 200 and base64.b64decode(seen[0].image_base64) == _png(4, 4)


@pytest.mark.parametrize("value", ["không phải base64 !!", base64.b64encode(b"<html>").decode()])
def test_bad_base64_image_is_400(ready, value):
    r = client.post(URL, json={"prompt": "x", "images_base64": [value]})
    assert r.status_code == 400 and r.json()["error_code"] == "ERR_INVALID_IMAGE"


@pytest.mark.parametrize("err, http, code", [
    ("draw_moderated", 422, "ERR_IMAGE_SAFETY_VIOLATION"),
    ("vlm_failed", 500, "ERR_IMAGE_GENERATION_FAILED"),
    ("llm_failed", 500, "ERR_IMAGE_GENERATION_FAILED"),
])
def test_job_errors_are_mapped(ready, err, http, code):
    ready[0]["error"] = err
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == http and r.json()["error_code"] == code and err in r.json()["message"]
    assert "hỏng" not in r.json()["message"]   # chi tiết ngoại lệ (có thể có địa chỉ nội bộ) chỉ ở log máy chủ


def test_deadline_gives_504(ready, monkeypatch):
    ready[0]["sleep"] = 1.0
    monkeypatch.setenv("TEXTFIX_V1_DEADLINE", "0.2")
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == 504 and r.json()["error_code"] == "ERR_UPSTREAM_TIMEOUT"
    time.sleep(1.0)   # job giả chạy nốt


def test_deadline_counts_image_download(ready, monkeypatch):
    # hạn tính từ lúc nhận yêu cầu: tải ảnh chậm ăn vào hạn -> 504 trước khi mở job, tổng thời gian không vượt hạn
    _, seen = ready
    monkeypatch.setenv("TEXTFIX_V1_DEADLINE", "6")
    monkeypatch.setattr(server, "fetch_image", lambda u: time.sleep(1.5) or _png())
    t0 = time.time()
    r = client.post(URL, json={"prompt": "x", "image_urls": ["https://a/1.png"]})
    assert r.status_code == 504 and r.json()["error_code"] == "ERR_UPSTREAM_TIMEOUT" and seen == []   # không mở job
    assert time.time() - t0 < 6


def test_busy_and_not_ready_are_503_with_retry_after(ready, monkeypatch):
    monkeypatch.setenv("TEXTFIX_MAX_QUEUE", "2")
    monkeypatch.setattr(server, "ORDER", ["a", "b"])
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == 503 and r.json()["error_code"] == "ERR_SERVICE_BUSY" and int(r.headers["retry-after"]) > 0
    monkeypatch.setattr(server, "ORDER", [])
    monkeypatch.setitem(server.STATE, "flux", [])
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == 503 and r.json()["error_code"] == "ERR_SERVICE_UNAVAILABLE"
    # làm nóng hỏng (Chromium không mở được): FLUX / VLM đã nạp nhưng KHÔNG nhận việc (11/11)
    monkeypatch.setitem(server.STATE, "flux", [object()])
    monkeypatch.setitem(server.STATE, "error", "RuntimeError: chromium")
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == 503 and r.json()["error_code"] == "ERR_SERVICE_UNAVAILABLE" and "chromium" not in r.text
    assert server.ready() is False   # /api/health báo đúng giá trị này
