"""Toán huấn luyện flow-matching cho DiT FLUX.2, khớp ĐÚNG đường suy luận của textfix/draw/flux.py (_sample):

- Quy ước của flux2/sampling.py: t chạy 1 -> 0, mỗi bước img += (t_prev - t_curr) * pred -> pred = dx/dt. Với
  x_t = (1 - t) * x0 + t * eps thì dx/dt = eps - x0: ĐÍCH = eps - x0.
- Ảnh tham chiếu (kiểu 2-3): token ảnh tham chiếu NỐI SAU token ảnh đang khử nhiễu, id thời gian 10, 20, ... (prc_img t_coord),
  đi qua `Flux2.forward` thường (cùng mức t với ảnh đích, attention đầy đủ) -> chỉ lấy dự đoán `[:, :n]`. Không dùng
  forward_kv_extract (đường đó cho ảnh tham chiếu t = 0 cố định, khác đường máy chủ đang chạy).
- Mức nhiễu khi học: logit-normal rồi dịch theo độ dài chuỗi như lịch suy luận (compute_empirical_mu + time shift của base 50
  bước); thêm một phần đúng 4 mức t của bản distill để LoRA học kỹ đúng những bước bản distill dùng.
- Loss: MSE theo token, nhân trọng số mặt nạ vùng chữ (1 ngoài chữ, `text_weight` trong chữ), chia tổng trọng số.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def schedule_mu(n_tokens: int, steps: int) -> float:
    """Như flux2.sampling.compute_empirical_mu (chép lại để test không cần gói flux2)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if n_tokens > 4300:
        return float(a2 * n_tokens + b2)
    m_200 = a2 * n_tokens + b2
    m_10 = a1 * n_tokens + b1
    a = (m_200 - m_10) / 190.0
    return float(a * steps + (m_200 - 200.0 * a))


def time_shift(t: torch.Tensor, mu: float) -> torch.Tensor:
    """Như flux2.sampling.generalized_time_snr_shift với sigma = 1."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1))


def distill_steps(n_tokens: int, steps: int = 4) -> list[float]:
    """Các mức t bản distill khử nhiễu (bỏ t = 0 cuối lịch) -- như flux2.sampling.get_schedule."""
    ts = torch.linspace(1, 0, steps + 1)
    return time_shift(ts[:-1].clamp(min=1e-6), schedule_mu(n_tokens, steps)).tolist()


def sample_t(n_tokens: int, g: torch.Generator | None = None, p_distill: float = 0.25, base_steps: int = 50) -> float:
    """Một mức t trong (0, 1): với xác suất p_distill lấy đúng một trong 4 mức của bản distill, còn lại logit-normal dịch theo
    lịch của bản base."""
    if p_distill > 0 and torch.rand(1, generator=g).item() < p_distill:
        ts = distill_steps(n_tokens)
        return float(ts[int(torch.randint(len(ts), (1,), generator=g).item())])
    u = torch.sigmoid(torch.randn(1, generator=g)).clamp(1e-4, 1 - 1e-4)
    return float(time_shift(u, schedule_mu(n_tokens, base_steps)).item())


def token_weights(mask: torch.Tensor, h16: int, w16: int, text_weight: float) -> torch.Tensor:
    """Mặt nạ vùng chữ cỡ ảnh (H, W, giá trị 0..1) -> trọng số từng token (h16 * w16,), thứ tự như prc_img ((h w) hàng trước).
    Token phủ một phần chữ nhận trọng số theo tỉ lệ điểm ảnh chữ trong ô 16x16."""
    m = mask.float()[None, None]
    frac = F.adaptive_avg_pool2d(m, (h16, w16))[0, 0]
    return (1.0 + (text_weight - 1.0) * frac).reshape(-1)


def noisy(x0: torch.Tensor, eps: torch.Tensor, t: float) -> torch.Tensor:
    return (1.0 - t) * x0 + t * eps


def loss_fn(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """pred / target (1, n, C), w (n,) -> (loss có trọng số, {text, bg}: MSE trong / ngoài vùng chữ để theo dõi)."""
    per = (pred.float() - target.float()).pow(2).mean(-1)[0]          # (n,)
    w = w.to(per.device, per.dtype)
    loss = (per * w).sum() / w.sum()
    with torch.no_grad():
        text = w > 1.0 + 1e-6
        parts = {"text": float(per[text].mean()) if text.any() else float("nan"),
                 "bg": float(per[~text].mean()) if (~text).any() else float("nan")}
    return loss, parts


def predict(dit, x_t: torch.Tensor, x_ids: torch.Tensor, t: float, ctx: torch.Tensor, ctx_ids: torch.Tensor,
            refs: torch.Tensor | None = None, ref_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Một lượt DiT như textfix/draw/flux.py._sample: [ảnh đích, ảnh tham chiếu] -> dự đoán của phần ảnh đích. Mọi tensor có
    chiều batch 1: x_t (1, n, 128), x_ids (1, n, 4), ctx (1, L, D), ctx_ids (1, L, 4), refs (1, m, 128), ref_ids (1, m, 4)."""
    n = x_t.shape[1]
    xi, ii = (torch.cat([x_t, refs], 1), torch.cat([x_ids, ref_ids], 1)) if refs is not None else (x_t, x_ids)
    ts = torch.full((1,), t, dtype=x_t.dtype, device=x_t.device)
    return dit(x=xi, x_ids=ii, timesteps=ts, ctx=ctx, ctx_ids=ctx_ids, guidance=None)[:, :n]


def checkpointing(dit) -> int:
    """Gradient checkpointing từng khối (Flux2.forward gọi block.forward_kv_extract): đổi tính lại lấy VRAM. -> số khối."""
    from torch.utils.checkpoint import checkpoint
    n = 0
    for block in list(dit.double_blocks) + list(dit.single_blocks):
        f = block.forward_kv_extract
        if getattr(f, "_ckpt", False):
            continue

        def wrapped(*a, _f=f, **k):
            return checkpoint(_f, *a, use_reentrant=False, **k)
        wrapped._ckpt = True
        block.forward_kv_extract = wrapped
        n += 1
    return n
