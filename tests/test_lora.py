"""lora/ (docs/LORA_PLAN.md): lớp LoRA, toán huấn luyện, độ phủ dấu, lọc dữ liệu -- trên DiT FLUX.2 thật nhưng TÍ HON (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from flux2.model import Flux2, Klein4BParams  # noqa: E402
from flux2.sampling import compute_empirical_mu, get_schedule, prc_img, prc_txt  # noqa: E402

from lora import flow, net  # noqa: E402
from lora.collect import reject, text_mask  # noqa: E402
from lora.requests import coverage, tone_of  # noqa: E402


def tiny(seed: int = 0) -> Flux2:
    torch.manual_seed(seed)
    return Flux2(Klein4BParams(context_in_dim=32, hidden_size=256, num_heads=2, depth=1, depth_single_blocks=1)).eval()


def inputs(seed: int = 1, h: int = 3, w: int = 4, L: int = 5, refs: bool = False):
    g = torch.Generator().manual_seed(seed)
    x, ids = prc_img(torch.randn(128, h, w, generator=g))
    c, cids = prc_txt(torch.randn(L, 32, generator=g))
    out = {"x_t": x[None], "x_ids": ids[None], "ctx": c[None], "ctx_ids": cids[None]}
    if refs:
        r, rids = prc_img(torch.randn(128, 2, 2, generator=g), t_coord=torch.tensor([10]))
        out.update(refs=r[None], ref_ids=rids[None])
    return out


def run(dit, I, t: float = 0.5):
    return flow.predict(dit, I["x_t"], I["x_ids"], t, I["ctx"], I["ctx_ids"], I.get("refs"), I.get("ref_ids"))


def randomize(dit):
    with torch.no_grad():
        for m in net.layers(dit).values():
            m.up.normal_(0, 0.02)


# ----------------------------------------------------------------------------------------------------- net.py
def test_inject_covers_every_attention_and_mlp_layer_and_starts_as_identity():
    dit, I = tiny(), inputs()
    before = run(dit, I)
    names = net.inject(dit, rank=4)
    # khối đôi: qkv + proj + 2 lớp MLP, cho cả luồng ảnh và luồng chữ = 8; khối đơn: linear1 + linear2 = 2
    assert len(names) == 10 and all(not p.requires_grad for n, p in dit.named_parameters() if not n.endswith((".down", ".up")))
    assert torch.equal(run(dit, I), before)


def test_mult_zero_is_base_model_and_merge_equals_wrapped():
    dit, I = tiny(), inputs()
    base = run(dit, I)
    net.inject(dit, rank=4, alpha=8)
    randomize(dit)
    on = run(dit, I)
    assert not torch.allclose(on, base)
    net.set_mult(dit, 0.0)
    assert torch.equal(run(dit, I), base)
    net.set_mult(dit, 1.0)
    assert net.merge(dit) == 10 and not net.layers(dit)
    assert torch.allclose(run(dit, I), on, atol=1e-5)


def test_save_load_roundtrip_and_mismatch(tmp_path):
    dit, I = tiny(), inputs()
    net.inject(dit, rank=4)
    randomize(dit)
    want = run(dit, I)
    p = str(tmp_path / "l.safetensors")
    net.save(p, dit, meta={"step": 7})
    fresh = tiny()
    info = net.load(p, fresh)
    assert info["rank"] == 4 and info["step"] == 7 and torch.allclose(run(fresh, I), want, atol=1e-6)
    other = Flux2(Klein4BParams(context_in_dim=32, hidden_size=256, num_heads=2, depth=2, depth_single_blocks=1))
    with pytest.raises(ValueError):   # khác số khối -> không lặng lẽ nạp thiếu
        net.inject(other, rank=4)
        net.load(p, other)


# ----------------------------------------------------------------------------------------------------- flow.py
def test_schedule_matches_flux2_sampling():
    for n in (1024, 4096, 5376):
        assert flow.schedule_mu(n, 4) == pytest.approx(compute_empirical_mu(n, 4))
        assert flow.distill_steps(n) == pytest.approx(get_schedule(4, n)[:-1], rel=1e-5)


def test_sample_t_in_range_and_hits_distill_steps():
    g = torch.Generator().manual_seed(0)
    ts = [flow.sample_t(4096, g, p_distill=0.5) for _ in range(400)]
    assert all(0 < t <= 1 for t in ts)   # bước đầu của bản distill là đúng t = 1 (nhiễu thuần)
    d = flow.distill_steps(4096)
    hits = sum(any(abs(t - x) < 1e-9 for x in d) for t in ts)
    assert 120 < hits < 280


def test_token_weights_follow_text_pixels_in_prc_img_order():
    m = torch.zeros(32, 48)
    m[:16, 32:] = 1.0            # ô (hàng 0, cột 2) toàn chữ
    m[16:, :8] = 1.0             # ô (hàng 1, cột 0) nửa chữ
    w = flow.token_weights(m, 2, 3, text_weight=5.0)
    assert w.tolist() == pytest.approx([1, 1, 5, 3, 1, 1])


def test_loss_weights_text_tokens():
    pred, target = torch.zeros(1, 4, 2), torch.tensor([[[1.0, 1.0], [0, 0], [0, 0], [0, 0]]])
    w = torch.tensor([3.0, 1.0, 1.0, 1.0])
    loss, parts = flow.loss_fn(pred, target, w)
    assert float(loss) == pytest.approx(3 / 6) and parts["text"] == pytest.approx(1.0) and parts["bg"] == pytest.approx(0.0)


def test_predict_keeps_only_target_tokens_and_reference_tokens_matter():
    dit = tiny()
    I, R = inputs(), inputs(refs=True)
    assert run(dit, R).shape == I["x_t"].shape
    assert not torch.allclose(run(dit, R), run(dit, I))


def test_training_step_reduces_loss_and_only_touches_lora():
    dit, I = tiny(), inputs(refs=True)
    net.inject(dit, rank=8)
    frozen = {n: p.clone() for n, p in dit.named_parameters() if not n.endswith((".down", ".up"))}
    params = net.parameters(dit)
    opt = torch.optim.Adam(params, lr=3e-3)
    g = torch.Generator().manual_seed(3)
    x0 = I["x_t"]
    eps = torch.randn(x0.shape, generator=g)
    w = torch.ones(x0.shape[1])
    w[:4] = 6.0
    losses = []
    for _ in range(40):
        pred = flow.predict(dit, flow.noisy(x0, eps, 0.6), I["x_ids"], 0.6, I["ctx"], I["ctx_ids"], I["refs"], I["ref_ids"])
        loss, _ = flow.loss_fn(pred, eps - x0, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    assert all(torch.equal(p, frozen[n]) for n, p in dit.named_parameters() if n in frozen)


def test_checkpointing_gives_same_gradients():
    I = inputs()
    grads = []
    for ck in (False, True):
        dit = tiny()
        net.inject(dit, rank=4)
        randomize(dit)
        if ck:
            assert flow.checkpointing(dit) == 2
        dit.train()
        loss, _ = flow.loss_fn(run(dit, I), torch.zeros_like(I["x_t"]), torch.ones(I["x_t"].shape[1]))
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in net.parameters(dit)]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


# ----------------------------------------------------------------------------------------------------- dữ liệu
def test_tone_and_coverage_handle_uppercase_and_d():
    assert tone_of("Ặ") == ("ă", "nặng") and tone_of("ỹ") == ("y", "ngã") and tone_of("ơ") == ("ơ", "ngang")
    assert tone_of("b") is None and tone_of("đ") is None
    c = coverage(["Đặc sản", "ỨNG DỤNG"])
    assert c[("đ", "")] == 1 and c[("ă", "nặng")] == 1 and c[("ư", "sắc")] == 1 and c[("u", "nặng")] == 1


def test_text_mask_marks_only_changed_pixels():
    draft = np.full((64, 64, 3), 100, np.uint8)
    poster = draft.copy()
    poster[10:20, 10:30] = 250
    poster[50, 50] = 102          # khác 2 < ngưỡng: nén / làm tròn, không phải chữ
    m = text_mask(draft, poster, grow=2)
    assert m[15, 20] and m[8, 20] and not m[5, 20] and not m[50, 50]


def test_reject_rules():
    req, brief = {"mode": "new"}, {"texts": [{"text": "Phở bò 50k"}]}
    ok = {"errors": [], "missing": [], "ops": [{"kind": "text"}, {"kind": "keep", "client": None}]}
    assert reject(req, brief, ok) is None
    assert reject({"mode": "edit_foreign"}, brief, ok).startswith("luồng")
    assert reject(req, brief, {**ok, "errors": [{"type": "overflow"}]}).startswith("lỗi dựng")
    assert reject(req, brief, {**ok, "missing": [1]}) == "thiếu câu khách"
    kept = {**ok, "ops": [{"kind": "keep", "client": 0}]}
    assert reject(req, brief, kept) and reject(req, brief, kept, allow_kept_client=True) is None
    assert reject(req, {"texts": []}, ok) == "ảnh không chữ"
    long = {**ok, "prompt_en": "x"}
    assert reject(req, brief, long, tokens=lambda s: 501).startswith("prompt quá 512")
    assert reject(req, brief, long, tokens=lambda s: 500) is None


# ----------------------------------------------------------------------------------------------------- chạy trọn script
def _png(path, arr):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_collect_script_builds_dataset_from_runs(tmp_path, monkeypatch):
    import json
    import sys
    from lora import collect
    runs = tmp_path / "runs"
    base = np.full((64, 48, 3), 120, np.uint8)
    poster = base.copy()
    poster[8:20, 6:40] = 20                        # chữ TextFix dựng
    plan = {"prompt_en": 'A poster with the headline "Phở bò".', "errors": [], "missing": [], "ops": [{"kind": "text"}]}
    for rid, req, extra in [("run_a", {"mode": "new"}, {}),
                            ("run_b", {"mode": "edit", "parent": "run_a"}, {"source.png": base}),
                            ("run_c", {"mode": "new"}, {"plan": {**plan, "missing": [0]}})]:
        d = runs / rid
        (d / "v0").mkdir(parents=True)
        (d / "request.json").write_text(json.dumps({**req, "prompt": "x"}), encoding="utf-8")
        (d / "brief.json").write_text(json.dumps({"prompt_en": "p", "seed": 5, "texts": [{"text": "Phở bò"}], "kho": []}),
                                      encoding="utf-8")
        (d / "v0" / "plan.json").write_text(json.dumps(extra.get("plan", plan)), encoding="utf-8")
        _png(d / "v0" / "draft.png", base)
        _png(d / "v0" / "poster.png", poster)
        if "source.png" in extra:
            _png(d / "source.png", extra["source.png"])
    out = tmp_path / "ds"
    monkeypatch.setattr(sys, "argv", ["collect", "--runs", str(runs), "--out", str(out), "--eval-frac", "0"])
    collect.main()
    rows = [json.loads(x) for x in (out / "manifest_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["id"]: r["kind"] for r in rows} == {"run_a": "new", "run_b": "edit"}          # run_c thiếu câu -> bỏ
    b = next(r for r in rows if r["id"] == "run_b")
    assert b["refs"] == ["ref0.png"] and b["root"] == "run_a" and b["prompt_en"] == plan["prompt_en"]
    assert (out / "train" / "run_b" / "ref0.png").exists() and 0.3 < b["mask_frac"] < 0.6   # ảnh 64x48: nới 8 px chiếm nhiều


def test_train_script_runs_end_to_end_on_tiny_model(tmp_path, monkeypatch):
    """lora.train với DiT tí hon thay cho trọng số thật: đọc cache, ảnh tham chiếu, bỏ prompt, cộng dồn, lưu LoRA nạp lại được."""
    import json
    import sys

    from flux2 import util
    from lora import train
    from textfix.draw import flux as tflux
    ds = tmp_path / "ds"
    (ds / "cache" / "train").mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    rows = []
    for k, kind in enumerate(["new", "product", "edit"]):
        refs = None if kind == "new" else prc_img(torch.randn(128, 2, 2, generator=g), t_coord=torch.tensor([10]))
        torch.save({"id": f"r{k}", "kind": kind, "ctx": torch.randn(6, 32, generator=g).to(torch.bfloat16),
                    "x0": torch.randn(128, 3, 4, generator=g).to(torch.bfloat16),
                    "frac": torch.tensor([0, 0, 1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float16),
                    "refs": None if refs is None else refs[0].to(torch.bfloat16), "ref_ids": None if refs is None else refs[1]},
                   ds / "cache" / "train" / f"r{k}.pt")
        rows.append({"id": f"r{k}", "kind": kind})
    torch.save({"ctx": torch.zeros(6, 32).to(torch.bfloat16)}, ds / "cache" / "_empty.pt")
    (ds / "manifest_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(util, "load_flow_model", lambda name, device="cpu": tiny().to(torch.bfloat16))
    monkeypatch.setattr(tflux, "find_weights", lambda extra=None: {})
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--ds", str(ds), "--out", str(out), "--rank", "4", "--steps", "3",
                                      "--accum", "2", "--save-every", "2", "--device", "cpu", "--cond-drop", "0.5"])
    train.main()
    saved = sorted(p.name for p in out.glob("lora_step*.safetensors"))
    assert saved == ["lora_step00002.safetensors", "lora_step00003.safetensors"]
    log = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == 3 and all(r["loss"] > 0 for r in log)
    info = net.load(str(out / saved[-1]), tiny())
    assert info["rank"] == 4 and info["step"] == 3 and info["samples"] == 3


def test_cache_script_encodes_targets_refs_and_text_fraction(tmp_path, monkeypatch):
    """lora.cache với bộ mã hoá chữ / VAE giả: đúng cỡ latent, đúng id thời gian ảnh tham chiếu (10, 20), tỉ lệ chữ từng ô 16x16."""
    import json
    import sys

    from flux2 import util
    from lora import cache
    from textfix.draw import flux as tflux

    class AE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def encode(self, x):
            return torch.nn.functional.avg_pool2d(x, 16).repeat(1, 43, 1, 1)[:, :128]

    monkeypatch.setattr(util, "load_text_encoder", lambda name, device="cpu": (lambda txt: torch.ones(1, 512, 32)))
    monkeypatch.setattr(util, "load_ae", lambda name, device="cpu": AE())
    monkeypatch.setattr(tflux, "find_weights", lambda extra=None: {})
    ds = tmp_path / "ds"
    d = ds / "train" / "run_x"
    _png(d / "target.png", np.full((64, 96, 3), 200, np.uint8))
    mask = np.zeros((64, 96), np.uint8)
    mask[:16, :16] = 255
    mask[16:32, 16:24] = 255
    _png(d / "mask.png", np.stack([mask] * 3, -1))
    _png(d / "ref0.png", np.full((128, 128, 3), 50, np.uint8))
    _png(d / "ref1.png", np.full((96, 64, 3), 90, np.uint8))
    meta = {"id": "run_x", "kind": "edit", "prompt_en": "p", "refs": ["ref0.png", "ref1.png"]}
    (ds / "manifest_train.jsonl").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["cache", "--ds", str(ds), "--device", "cpu"])
    cache.main()
    c = torch.load(ds / "cache" / "train" / "run_x.pt", weights_only=False)
    assert c["x0"].shape == (128, 4, 6) and c["ctx"].shape == (512, 32) and c["kind"] == "edit"
    assert c["frac"].reshape(4, 6)[0, 0] == 1 and c["frac"].reshape(4, 6)[1, 1] == 0.5 and c["frac"].sum() == 1.5
    assert c["refs"].shape == (8 * 8 + 6 * 4, 128) and sorted(set(c["ref_ids"][:, 0].tolist())) == [10, 20]
    assert (ds / "cache" / "_empty.pt").exists()


def test_report_texts_flags_repeated_ai_copy():
    from lora.collect import report_texts
    rows = [{"texts": ["Đặt lịch ngay", f"Tiệm nail số {k}"], "sources": ["ai", "user"]} for k in range(10)]
    r = report_texts(rows)
    assert r["ai_share"] == 0.5 and r["top10_share"] > 0.3


def test_short_caption_keeps_quoted_and_reference_sentences():
    from lora.cache import short_caption
    pe = ('A warm poster for a pho shop with the headline "Phở Bà Tư". Steam rises from a bowl under soft window light. '
          'A red button reads "Gọi món ngay". The bowl looks exactly like the bowl from image 1. Wooden table, green herbs.')
    s = short_caption(pe)
    assert '"Phở Bà Tư"' in s and '"Gọi món ngay"' in s and "image 1" in s and "Steam" not in s and "Wooden" not in s
    assert short_caption('Only "Sale".') is None


def test_keep_lines_are_down_weighted_but_fixed_text_wins():
    from lora.collect import keep_mask
    plan = {"slots": {"L": [{"id": "L1", "box": [10, 10, 30, 20]}, {"id": "L2", "box": [0, 0, 5, 5]}]},
            "ops": [{"kind": "keep", "marks": ["L1"]}, {"kind": "text", "marks": ["L2"]}]}
    m = keep_mask(plan, 40, 40, pad=0)
    assert m[15, 20] and not m[2, 2] and m.sum() == 20 * 10
    w = flow.weights(torch.tensor([1.0, 0.0, 0.0, 0.5]), torch.tensor([1.0, 1.0, 0.0, 1.0]), 6.0, 0.3)
    assert w.tolist() == pytest.approx([6.0, 0.3, 1.0, 1 + 2.5 - 0.35])
