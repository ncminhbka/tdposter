"""FLUX.2-klein: vẽ nháp poster (prompt + ảnh sản phẩm tuỳ chọn -> ảnh).

  F = Flux("distill", device="cuda:0").load()   # DiT + VAE + bộ mã hoá chữ (Qwen3-4B) nạp một lần, giữ trong VRAM
  W, H = SIZES["4:5"]
  draft = F.generate(prompt_en, W, H, seed, refs=[product])   # refs: ảnh sản phẩm khách tải (tuỳ chọn)

Lấy mẫu đúng như mã tham chiếu của BFL (flux2/sampling.py): lịch `get_schedule`, Euler; bản distill 4 bước không CFG, bản base
50 bước CFG 4.0 (flux2/util.py). Ảnh tham chiếu như `encode_image_refs`: mã hoá VAE, ghép token sau ảnh đang khử nhiễu với trục
thời gian RoPE t = 10, 20, ...; chỉ lấy dự đoán của phần ảnh đang khử nhiễu. Khác mã tham chiếu ở một điểm: DiT, VAE, bộ mã hoá
chữ có thể đặt trên các card khác nhau.

Một bản Flux không dùng đồng thời từ nhiều luồng: máy chủ chạy mỗi card một bản, một luồng (server/app.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# tỉ lệ khung -> cỡ sinh (rộng, cao): ~1 MP, bội số 16 (VAE nén 16 lần). Nguồn duy nhất của các tỉ lệ API nhận.
SIZES = {"1:1": (1024, 1024), "4:5": (896, 1120), "3:4": (864, 1152), "2:3": (832, 1248), "9:16": (768, 1344),
         "16:9": (1344, 768), "4:3": (1152, 864)}

# biến thể -> (tên cấu hình flux2.util, tệp trọng số, biến môi trường chỉ tệp, số bước, CFG hoặc None)
MODELS = {"distill": ("flux.2-klein-4b", "flux-2-klein-4b.safetensors", "KLEIN_4B_MODEL_PATH", 4, None),
          "base": ("flux.2-klein-base-4b", "flux-2-klein-base-4b.safetensors", "KLEIN_4B_BASE_MODEL_PATH", 50, 4.0)}
HOME = Path(os.path.expanduser("~"))
SEARCH = [r / "persistent-data" / d for r in (HOME, Path("/home/jovyan"), Path("/"))
          for d in ("FLUX.2-klein-4B", "FLUX.2-klein-base-4B", "")]
VAE_FILE = Path("vae") / "diffusion_pytorch_model.safetensors"
# mã suy luận FLUX.2 của Black Forest Labs (Apache-2.0) KHÔNG đi kèm TextFix: chỉ cần khi vẽ tại chỗ
FLUX2_MISSING = ("thiếu gói flux2 (mã suy luận FLUX.2 của Black Forest Labs, đi kèm repo ở flux2/ từ 12/11): git pull, hoặc chép src/flux2 "
                 "của github.com/black-forest-labs/flux2 vào gốc repo, hoặc pip install git+https://github.com/black-forest-labs/flux2.git"
                 " (docs/SERVER.md). FLUX ở máy khác (TEXTFIX_FLUX_URL) thì không cần.")


def find_weights(extra: list[Path] | None = None) -> dict:
    """Tìm trọng số DiT (distill / base), VAE, bộ mã hoá chữ trong các thư mục tìm kiếm -> {distill, base, vae, text_encoder}.
    Biến môi trường đã đặt thì thắng. Tìm thấy thì ĐẶT biến môi trường tương ứng, vì flux2.util đọc đường dẫn từ đó."""
    roots = [p for p in list(extra or []) + SEARCH if p.is_dir()]
    found = {}

    def first(*candidates):
        return next((str(c) for c in candidates if c.is_file()), None)

    for key, (_, fname, env, _, _) in MODELS.items():
        hit = first(*([Path(os.environ[env])] if os.environ.get(env) else []),
                    *(r / fname for r in roots), *(r / "transformer" / fname for r in roots))
        if hit:
            os.environ[env] = found[key] = hit
    vae = os.environ.get("AE_MODEL_PATH") or first(*(r / VAE_FILE for r in roots))
    if vae:
        os.environ["AE_MODEL_PATH"] = found["vae"] = vae
    te = os.environ.get("TEXT_ENCODER_PATH") or next(
        (str(r / "text_encoder") for r in roots if (r / "text_encoder" / "config.json").is_file()), None)
    if te:
        os.environ["TEXT_ENCODER_PATH"] = found["text_encoder"] = te
    return found


class Flux:
    """DiT + VAE + bộ mã hoá chữ nạp sẵn.
    device: đặt cả ba phần lên một card (máy chủ: mỗi card một bản). Không truyền thì theo TEXTFIX_DEVICE_DIT / _TE / _AE
    (mặc định cuda:0 cả ba)."""

    def __init__(self, variant: str = "distill", steps: int | None = None, weights_dir: str | None = None,
                 device: str | None = None):
        self.variant = variant
        self.name, _, _, default_steps, self.cfg = MODELS[variant]
        self.steps = steps or default_steps
        self.dev_dit = device or os.environ.get("TEXTFIX_DEVICE_DIT", "cuda:0")
        self.dev_te = device or os.environ.get("TEXTFIX_DEVICE_TE", self.dev_dit)
        self.dev_ae = device or os.environ.get("TEXTFIX_DEVICE_AE", self.dev_dit)
        self.weights = find_weights([Path(weights_dir)] if weights_dir else None)
        self.dit = self.ae = self.te = None

    def load(self) -> "Flux":
        import torch

        try:
            from flux2 import util
        except ModuleNotFoundError as e:
            if e.name != "flux2":
                raise
            raise ModuleNotFoundError(FLUX2_MISSING, name="flux2") from e
        if self.variant not in self.weights:
            raise FileNotFoundError(f"không thấy {MODELS[self.variant][1]} -- đặt {MODELS[self.variant][2]} hoặc --weights-dir")
        torch.backends.cuda.matmul.allow_tf32 = True
        self.dit = util.load_flow_model(self.name, device=self.dev_dit).eval()
        self.ae = util.load_ae(self.name, device=self.dev_ae).eval()
        self.ae_dtype = next(self.ae.parameters()).dtype
        self.te = util.load_text_encoder(self.name, device=self.dev_te)
        self._force_bf16()
        self._uncond = self._encode("") if self.cfg else None
        return self

    def _force_bf16(self) -> None:
        """DiT và bộ mã hoá chữ chạy bf16. transformers đổi kiểu số mặc định của torch (toàn cục) khi nạp bộ mã hoá chữ, nên
        bản nạp song song với bản khác có thể lẫn tham số float32 -> lỗi "float != BFloat16" lúc vẽ. Ép lại cho chắc."""
        import torch
        for mod in (self.dit, self.te):
            if any(p.dtype == torch.float32 for p in mod.parameters()):
                mod.to(torch.bfloat16)

    def info(self) -> dict:
        return {"variant": self.variant, "model": self.name, "steps": self.steps, "cfg": self.cfg, "weights": self.weights,
                "devices": {"dit": self.dev_dit, "te": self.dev_te, "ae": self.dev_ae}}

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[np.ndarray] | None = None) -> np.ndarray:
        """Vẽ một ảnh w x h (bội số 16) -> RGB uint8 (h, w, 3). Cùng prompt, cỡ, seed, ảnh tham chiếu -> cùng ảnh."""
        if w % 16 or h % 16:
            raise ValueError(f"cỡ ảnh phải là bội số 16: {w}x{h}")
        return self._sample(self._encode(prompt), w, h, seed, self._encode_refs(refs) if refs else None)

    # ------------------------------------------------------------------------------------------------ mã hoá
    def _encode(self, prompt: str):
        import torch

        from flux2.sampling import prc_txt
        with torch.no_grad():
            ctx = self.te([prompt]).to(torch.bfloat16)[0].to(self.dev_dit)
        ctx, ids = prc_txt(ctx)
        return ctx.unsqueeze(0), ids.unsqueeze(0).to(self.dev_dit)

    def _encode_refs(self, imgs: list[np.ndarray]):
        """Ảnh tham chiếu -> (token, id), như flux2.sampling.encode_image_refs nhưng VAE ở thiết bị riêng."""
        import torch
        from PIL import Image

        from flux2.sampling import default_prep, prc_img
        limit = 2024 ** 2 if len(imgs) == 1 else 1024 ** 2   # trần điểm ảnh của mã tham chiếu
        toks, ids = [], []
        with torch.no_grad():
            for k, im in enumerate(imgs):
                t = default_prep(img=Image.fromarray(im), limit_pixels=limit)
                z = self.ae.encode(t[None].to(self.dev_ae, self.ae_dtype))[0]
                x, xi = prc_img(z, t_coord=torch.tensor([10 + 10 * k]))
                toks.append(x)
                ids.append(xi)
        return torch.cat(toks)[None].to(self.dev_dit, torch.bfloat16), torch.cat(ids)[None].to(self.dev_dit)

    # ------------------------------------------------------------------------------------------------ khử nhiễu
    def _sample(self, ctx, w: int, h: int, seed: int, refs=None) -> np.ndarray:
        """Euler theo flux2.sampling.get_schedule. Bản có CFG chạy một batch [có prompt, prompt rỗng]."""
        import torch

        from flux2.sampling import get_schedule, prc_img
        dev, cfg = self.dev_dit, self.cfg
        g = torch.Generator(device=dev).manual_seed(int(seed))
        z = torch.randn(1, 128, h // 16, w // 16, generator=g, device=dev, dtype=torch.bfloat16)
        x, x_ids = prc_img(z[0])
        x, x_ids = x.unsqueeze(0), x_ids.unsqueeze(0).to(dev)
        n = x.shape[1]
        nb = 2 if cfg else 1
        c, c_ids = ctx
        if cfg:
            c, c_ids = torch.cat([c, self._uncond[0]]), torch.cat([c_ids, self._uncond[1]])
        ts = get_schedule(num_steps=self.steps, image_seq_len=n)
        with torch.no_grad():
            for t_curr, t_prev in zip(ts[:-1], ts[1:]):
                xi, ii = (torch.cat([x, refs[0]], 1), torch.cat([x_ids, refs[1]], 1)) if refs is not None else (x, x_ids)
                pred = self.dit(x=xi.repeat(nb, 1, 1), x_ids=ii.repeat(nb, 1, 1),
                                timesteps=torch.full((nb,), t_curr, dtype=x.dtype, device=dev), ctx=c, ctx_ids=c_ids,
                                guidance=None)[:, :n]   # klein không có nhúng guidance
                if cfg:
                    pred = pred[1:2] + cfg * (pred[0:1] - pred[1:2])   # không prompt + cfg * (có prompt - không prompt)
                x = (x + (t_prev - t_curr) * pred).to(torch.bfloat16)
            z = x[0].transpose(0, 1).reshape(1, 128, h // 16, w // 16)
            img = self.ae.decode(z.to(self.dev_ae, self.ae_dtype)).float()
        out = ((img[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        torch.cuda.empty_cache()
        return out
