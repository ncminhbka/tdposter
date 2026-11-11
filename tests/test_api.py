"""API (server/app.py, server/contract.py: POST /api/v1/images/generations) không cần model: dây chuyền thay bằng `generate` giả."""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402
import contract  # noqa: E402

client = TestClient(server.app)   # không "with": không nạp model
URL = "/api/v1/images/generations"


def _png(w=8, h=6) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def ready(monkeypatch, tmp_path):
    """Máy chủ 'đã nạp model'; `generate` giả: outcome["error"] = ApiError, outcome["sleep"] = giây chạy. Trả các lời gọi."""
    monkeypatch.setitem(server.STATE, "flux", [object()])
    monkeypatch.setitem(server.STATE, "vlm", object())
    monkeypatch.setitem(server.STATE, "ready", True)
    monkeypatch.setitem(server.STATE, "error", None)
    monkeypatch.setattr(server, "RUNS", tmp_path)
    outcome, seen = {}, []

    def generate(run_id, req, product, rid, old=None, old_img=None):
        seen.append({"req": req, "product": product, "rid": rid, "old": old, "old_img": old_img})
        time.sleep(outcome.get("sleep", 0))
        if outcome.get("error"):
            raise outcome["error"]
        return _png(), 8, 6, {"llm": 3.5, "vlm": 5.0}, (old or {}).get("aspect") or req.aspect_ratio
    monkeypatch.setattr(server, "generate", generate)
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
    assert (seen[0]["req"].prompt, seen[0]["req"].aspect_ratio, seen[0]["product"]) == ("Poster quán cà phê", "9:16", None)
    assert r.headers["x-run-id"].startswith("run_") and r.headers["x-timing"] == "llm=3.5;vlm=5"
    assert server.ORDER == []   # lượt xong rời hàng đợi


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
    ({"prompt": "x", "images_base64": "không phải mảng"}, 400, "ERR_INVALID_REQUEST"),
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


@pytest.mark.parametrize("path", ["/", "/api/jobs", "/api/generate", "/docs", "/openapi.json", "/api/runs/run_x/zip"])
def test_only_one_api(ready, path):
    # 12/11: chỉ còn API cho chatbot -- không UI, không job, không zip debug, không trang tài liệu tự sinh
    assert client.get(path).status_code in (404, 405) and client.post(path, json={}).status_code in (404, 405)


def test_health_has_no_internal_details(ready):
    assert client.get("/api/health").json() == {"ready": True, "failed": False, "queue": 0}


@pytest.mark.parametrize("value", [lambda b: [b], lambda b: ["data:image/png;base64," + b], lambda b: [b, b]])
def test_first_base64_image_becomes_product_image(ready, value):
    _, seen = ready
    b = base64.b64encode(_png(4, 4)).decode()
    r = client.post(URL, json={"prompt": "x", "images_base64": value(b)})
    assert r.status_code == 200 and seen[0]["product"].shape == (4, 4, 3)   # bản đầu: chỉ ảnh đầu


def test_old_image_fields_are_ignored(ready):
    # 12/11: đội chatbot chốt chỉ images_base64 -- image_base64 / image_urls là khoá lạ: bỏ qua, không lỗi, không tải gì
    _, seen = ready
    b = base64.b64encode(_png(4, 4)).decode()
    r = client.post(URL, json={"prompt": "x", "image_base64": b, "image_urls": ["https://a/1.png"]})
    assert r.status_code == 200 and seen[0]["product"] is None


def test_bad_later_image_is_also_400(ready):
    b = base64.b64encode(_png(4, 4)).decode()
    r = client.post(URL, json={"prompt": "x", "images_base64": [b, "hỏng !!"]})
    assert r.status_code == 400 and r.json()["error_code"] == "ERR_INVALID_IMAGE"


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
    ready[0]["error"] = contract.ApiError(err, "x", "hỏng ở http://10.0.0.1")
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == http and r.json()["error_code"] == code and err in r.json()["message"]
    assert "10.0.0.1" not in r.text and r.headers["x-run-id"].startswith("run_")   # chi tiết chỉ ở log máy chủ


def test_unexpected_error_is_500(ready):
    ready[0]["error"] = RuntimeError("bất ngờ")
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == 500 and r.json()["error_code"] == "ERR_IMAGE_GENERATION_FAILED" and "bất ngờ" not in r.text


def test_deadline_gives_504(ready, monkeypatch):
    ready[0]["sleep"] = 1.0
    monkeypatch.setenv("TEXTFIX_V1_DEADLINE", "0.2")
    r = client.post(URL, json={"prompt": "x"})
    assert r.status_code == 504 and r.json()["error_code"] == "ERR_UPSTREAM_TIMEOUT"
    time.sleep(1.0)   # lượt giả chạy nốt


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
    assert client.get("/api/health").json() == {"ready": False, "failed": True, "queue": 0}


# ---------------------------------------------------------------------------------------------- máy chủ
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
    assert contract.draw_error(FluxModerated("Content Moderated"), "FLUX vẽ nháp").code == "draw_moderated"
    assert contract.draw_error(FluxError("HTTP 500"), "FLUX vẽ nháp").code == "draw_failed"


def test_png_bytes_is_lossless():
    from PIL import Image
    a = np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8)
    assert (np.asarray(Image.open(io.BytesIO(server.png_bytes(a)))) == a).all()


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


# ---------------------------------------------------------------------------------------------- sửa poster (11/11)
def _poster_b64(run_id="run_old", aspect="9:16", w=12, h=20) -> str:
    from textfix import meta
    a = np.full((h, w, 3), 120, np.uint8)
    info = meta.info(run_id, {"mode": "poster", "prompt_en": "a poster", "texts": [{"text": "Giảm 20%", "role": "offer",
                                                                                    "source": "user"}]}, aspect, False)
    return base64.b64encode(server.png_bytes(a, info)).decode()


def test_our_poster_is_recognized_as_edit_and_keeps_its_aspect(ready):
    _, seen = ready
    r = client.post(URL, json={"prompt": "đổi thành giảm 30%", "images_base64": [_poster_b64()], "aspect_ratio": "1:1"})
    assert r.status_code == 200 and r.json()["data"]["aspect_ratio"] == "9:16"   # chatbot gửi 1:1 mặc định: giữ khung cũ
    s = seen[0]
    assert s["old"]["run_id"] == "run_old" and s["old"]["texts"][0]["text"] == "Giảm 20%" and s["product"] is None
    assert s["old_img"].shape == (20, 12, 3)   # ảnh cũ giữ cỡ (không thu về 1024 khi nhỏ hơn)


def test_poster_found_in_any_position_other_image_is_new_product(ready):
    _, seen = ready
    prod = base64.b64encode(_png(4, 4)).decode()
    r = client.post(URL, json={"prompt": "thay sản phẩm bằng ảnh này", "images_base64": [prod, _poster_b64()]})
    assert r.status_code == 200 and seen[0]["old"]["run_id"] == "run_old" and seen[0]["product"].shape == (4, 4, 3)


def test_image_without_info_is_product_image(ready):
    _, seen = ready
    r = client.post(URL, json={"prompt": "x", "images_base64": [base64.b64encode(_png(4, 4)).decode()]})
    assert r.status_code == 200 and seen[0]["old"] is None and seen[0]["product"] is not None


def test_returned_png_carries_poster_info():
    from textfix import meta
    a = np.zeros((8, 8, 3), np.uint8)
    B = {"mode": "poster", "prompt_en": "p", "texts": [{"text": "Tiệm nail Hồng Nhung", "role": "headline", "source": "user",
                                                        "ordered": True, "extra": 1}]}
    parent = {"run_id": "run_a", "depth": 1}
    raw = base64.b64decode(base64.b64encode(server.png_bytes(a, meta.info("run_b", B, "4:3", True, parent=parent))))
    d = meta.read(raw)
    assert (d["run_id"], d["parent"], d["depth"], d["aspect"], d["product"]) == ("run_b", "run_a", 2, "4:3", True)
    assert d["texts"] == [{"text": "Tiệm nail Hồng Nhung", "role": "headline", "source": "user", "ordered": True}]


def _fake_draw_worker(tasks):
    """Luồng VẼ giả: nhận việc từ DRAW_Q, trả nháp đen (có draft sẵn: chỉ 'Hi-SAM' -> nét rỗng)."""
    def run():
        while True:
            t = server.DRAW_Q.get()
            if t is None:
                return
            tasks.append(t)
            if t.get("draft") is not None:
                t["fut"].set_result({"draft": t["draft"], "ink": np.zeros(t["draft"].shape[:2], bool), "timing": {}})
            else:
                t["fut"].set_result({"draft": np.zeros((t["H"], t["W"], 3), np.uint8), "timing": {"ve_nhap": 1.0}})
    import threading
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


@pytest.mark.parametrize("own", [True, False])
def test_generate_edit_wiring(monkeypatch, tmp_path, own):
    from textfix import edit, meta
    from textfix.edit.instructions import SCHEMA as EDIT_SCHEMA
    monkeypatch.setattr(server, "RUNS", tmp_path)

    def llm(system, parts, schema=None):
        if schema is EDIT_SCHEMA:
            llm.edit_msg = parts[0]
            return {"texts": [], "prompt_en": "The poster from image 1 without lettering.", "aspect": None}
        return {"intent": "edit", "texts": [], "prompt_en": ""}
    monkeypatch.setitem(server.STATE, "llm", llm)
    monkeypatch.setitem(server.STATE, "vlm", object())
    monkeypatch.setattr(edit, "read_image", lambda img, ink, vlm: {"texts": [{"text": "Giảm 20%", "role": "offer", "source": "user"}],
                                                                   "prompt_en": "a red poster", "product": True, "lines": 1})
    tasks = []
    _fake_draw_worker(tasks)
    img = np.full((1344, 768, 3), 50, np.uint8)
    old = {"run_id": "run_old", "depth": 2, "aspect": "4:3", "product": False, "prompt_en": "p",
           "texts": [{"text": "Giảm 20%", "role": "offer", "source": "user"}]}
    req = contract.Request(prompt="bỏ hết chữ", images_base64=["x"], aspect_ratio="1:1")
    try:
        if own:
            png, W, H, timing, aspect = server.generate("run_new", req, None, "rid", old=old, old_img=img)
        else:
            png, W, H, timing, aspect = server.generate("run_new", req, img, "rid")
    finally:
        server.DRAW_Q.put(None)
    info = meta.read(png)
    draw = tasks[-1]
    assert draw["refs"][0] is img and len(draw["refs"]) == 1 and draw["product"] is None
    assert info["mode"] == "image" and info["texts"] == [] and "Giảm 20%" in llm.edit_msg
    if own:   # poster của mình: khung cũ, nối chuỗi sửa
        assert (aspect, info["parent"], info["depth"], info["product"]) == ("4:3", "run_old", 3, False)
    else:     # ảnh lạ: đọc chữ trước (một việc chỉ-Hi-SAM trên trạm vẽ), khung gần nhất với ảnh
        assert tasks[0]["draft"] is img and (aspect, info["parent"], info["depth"], info["product"]) == ("9:16", None, 1, True)
        assert "doc_anh" in timing and (tmp_path / "run_new" / "read.json").exists()
    rq = json.loads((tmp_path / "run_new" / "request.json").read_text(encoding="utf-8"))
    assert rq["mode"] == ("edit" if own else "edit_foreign") and "aspect_ratio" in rq["ignored_fields"]
