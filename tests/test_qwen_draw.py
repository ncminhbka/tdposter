"""Qwen-Image vẽ nháp tại chỗ (textfix/draw/qwen.py, start_server.sh --draw qwen): pipeline giả, không cần GPU."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


class FakePipe:
    """Giả pipeline diffusers: ghi lại tham số, trả ảnh lệch cỡ."""

    def __init__(self, with_image: bool):
        self.calls = []
        self.with_image = with_image

    def __call__(self, prompt, width, height, generator=None, num_inference_steps=None, true_cfg_scale=None,
                 negative_prompt=None, image=None, use_kv_cache=False):
        from PIL import Image
        self.calls.append({"prompt": prompt, "width": width, "height": height, "steps": num_inference_steps,
                           "cfg": true_cfg_scale, "image": image, "kv": use_kv_cache})
        return type("R", (), {"images": [Image.new("RGB", (width + 8, height), (10, 20, 30))]})()


def drawer(monkeypatch, with_image=True, **kw):
    from textfix.draw.qwen import QwenDraw
    for k in ("TEXTFIX_QWEN_STEPS", "TEXTFIX_QWEN_CFG", "TEXTFIX_QWEN_MEM", "TEXTFIX_QWEN_SCALE"):
        monkeypatch.delenv(k, raising=False)
    Q = QwenDraw(device="cpu", **kw)
    Q.pipe = FakePipe(with_image)
    import inspect
    Q.params = set(inspect.signature(Q.pipe.__call__).parameters) - (set() if with_image else {"image"})
    Q.image_arg = "image" if with_image else ""
    return Q


def test_defaults_are_the_pipeline_own(monkeypatch):
    Q = drawer(monkeypatch)
    img = Q.generate("a poster", 64, 48, 7)
    assert img.shape == (48, 64, 3)   # vẽ ở 2 lần cỡ (khung gốc ~4 MP của model) rồi thu về cỡ đã hẹn
    c = Q.pipe.calls[0]
    assert (c["width"], c["height"]) == (128, 96) and c["kv"] is True   # model card: use_kv_cache=True
    assert (c["steps"], c["cfg"], c["image"]) == (None, None, None)   # không đặt: để Turbo dùng lịch 8 bước lưu sẵn


def test_scale_1_draws_at_pipeline_size(monkeypatch):
    monkeypatch.setenv("TEXTFIX_QWEN_SCALE", "1")
    from textfix.draw.qwen import QwenDraw
    assert QwenDraw(device="cpu").scale == 1
    monkeypatch.setenv("TEXTFIX_QWEN_SCALE", "0")
    with pytest.raises(ValueError, match="TEXTFIX_QWEN_SCALE"):
        QwenDraw(device="cpu")


def test_steps_cfg_and_refs(monkeypatch):
    Q = drawer(monkeypatch, steps=8, cfg=1.0)
    Q.generate("x", 64, 48, 1, refs=[np.zeros((10, 10, 3), np.uint8)])
    c = Q.pipe.calls[0]
    assert (c["steps"], c["cfg"]) == (8, 1.0) and c["image"].size == (10, 10)


def test_refs_dropped_when_pipeline_takes_no_image(monkeypatch):
    Q = drawer(monkeypatch, with_image=False)
    kw = Q.call_args("x", 64, 48, 1, [np.zeros((10, 10, 3), np.uint8)])
    assert "image" not in kw and Q._warned


def test_bad_mem_and_missing_dir(monkeypatch, tmp_path):
    from textfix.draw.qwen import QwenDraw
    with pytest.raises(ValueError, match="TEXTFIX_QWEN_MEM"):
        QwenDraw(mem="tpu")
    with pytest.raises(FileNotFoundError, match="model_index.json"):
        QwenDraw(model=tmp_path, device="cpu").load()


def test_draw_profile_qwen_ignores_remote_flux(monkeypatch):
    from textfix import config
    from textfix.draw import remote
    monkeypatch.setenv("TEXTFIX_FLUX_URL", "http://flux9b/v1/chat/completions")
    monkeypatch.setenv("TEXTFIX_DRAW", "flux")
    assert config.draw_profile() == "flux" and remote.enabled()
    monkeypatch.setenv("TEXTFIX_DRAW", "qwen")
    assert config.draw_profile() == "qwen" and not remote.enabled()   # vẽ tại chỗ: không gọi FLUX ngoài
    monkeypatch.setenv("TEXTFIX_DRAW", "sdxl")
    with pytest.raises(RuntimeError, match="TEXTFIX_DRAW"):
        config.draw_profile()


def test_draw_profile_flux4b_forces_local_flux(monkeypatch):
    from textfix import config
    from textfix.draw import remote
    monkeypatch.setenv("TEXTFIX_FLUX_URL", "http://flux9b.test/v1/chat/completions")
    monkeypatch.setenv("TEXTFIX_DRAW", "flux4b")
    assert config.draw_profile() == "flux4b" and not remote.enabled()   # FLUX 4B nạp tại chỗ, không gọi 9B ở máy khác
