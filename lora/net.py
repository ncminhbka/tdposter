"""LoRA tự viết cho DiT FLUX.2 (flux2/model.py), không phụ thuộc peft.

Mọi lớp Linear của DiT được GỌI như module (img_attn.qkv(x), linear1(x), img_mlp là nn.Sequential), nên chỉ cần thay module bằng
LoRALinear: y = W x + (up @ down) x * alpha / rank * mult. `up` khởi tạo 0 -> gắn xong model ra y hệt bản gốc.

  names = inject(dit, rank=32, alpha=32)        # gắn vào các lớp TARGETS
  save(path, dit, meta={...}); load(path, dit)  # chỉ trọng số LoRA (safetensors) + metadata rank / alpha / targets
  set_mult(dit, 0.0)                            # tắt LoRA (so sánh trước / sau trên cùng model)
  merge(dit)                                    # gộp vào W, gỡ lớp bọc: suy luận không chậm thêm (máy chủ)
"""

from __future__ import annotations

import json
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

# lớp được gắn LoRA: attention + MLP của cả hai luồng (ảnh, chữ) trong khối đôi, và hai lớp của khối đơn
TARGETS = (r"double_blocks\.\d+\.(img|txt)_attn\.(qkv|proj)$", r"double_blocks\.\d+\.(img|txt)_mlp\.[02]$",
           r"single_blocks\.\d+\.linear[12]$")


class LoRALinear(nn.Module):
    """Bọc một nn.Linear (đóng băng). Tham số LoRA giữ float32 (bản chính để tối ưu), tính theo kiểu số của đầu vào."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank, self.alpha, self.mult = int(rank), float(alpha), 1.0
        dev = base.weight.device
        self.down = nn.Parameter(torch.empty(self.rank, base.in_features, device=dev, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(base.out_features, self.rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank * self.mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.mult == 0.0:
            return y
        return y + F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype)) * self.scale

    def delta(self) -> torch.Tensor:
        """Phần cộng vào W (float32)."""
        return (self.up @ self.down) * self.scale


def _match(name: str, patterns) -> bool:
    return any(re.search(p, name) for p in patterns)


def _set(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


def inject(model: nn.Module, rank: int, alpha: float | None = None, targets=TARGETS) -> list[str]:
    """Gắn LoRA vào mọi nn.Linear có tên khớp `targets`; đóng băng mọi tham số khác. -> tên các lớp đã gắn."""
    model.requires_grad_(False)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and _match(n, targets)]
    if not names:
        raise ValueError("không lớp nào khớp TARGETS -- model không phải DiT FLUX.2?")
    for n in names:
        _set(model, n, LoRALinear(model.get_submodule(n), rank, rank if alpha is None else alpha))
    return names


def layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in layers(model).values() for p in (m.down, m.up)]


def set_mult(model: nn.Module, mult: float) -> None:
    for m in layers(model).values():
        m.mult = float(mult)


def state(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for n, m in layers(model).items():
        out[f"{n}.down"] = m.down.detach().float().cpu().contiguous()
        out[f"{n}.up"] = m.up.detach().float().cpu().contiguous()
    return out


def save(path: str, model: nn.Module, meta: dict | None = None) -> None:
    from safetensors.torch import save_file
    L = layers(model)
    if not L:
        raise ValueError("model chưa gắn LoRA")
    m0 = next(iter(L.values()))
    info = {"rank": m0.rank, "alpha": m0.alpha, "layers": len(L), **(meta or {})}
    save_file(state(model), path, metadata={"textfix_lora": json.dumps(info, ensure_ascii=False)})


def read(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as f:
        info = json.loads((f.metadata() or {}).get("textfix_lora", "{}"))
    return load_file(path), info


def load(path: str, model: nn.Module, mult: float = 1.0) -> dict:
    """Nạp LoRA từ tệp; model chưa gắn thì gắn theo rank / alpha trong tệp. Tên lớp phải khớp hết (sai model -> lỗi)."""
    sd, info = read(path)
    names = sorted({k.rsplit(".", 1)[0] for k in sd})
    if not layers(model):
        for n in names:
            _set(model, n, LoRALinear(model.get_submodule(n), int(info["rank"]), float(info["alpha"])))
        model.requires_grad_(False)
    L = layers(model)
    missing = [n for n in names if n not in L]
    if missing or len(names) != len(L):
        raise ValueError(f"LoRA không khớp model: thiếu {missing[:3]}, tệp {len(names)} lớp, model {len(L)} lớp")
    with torch.no_grad():
        for n in names:
            L[n].down.copy_(sd[f"{n}.down"])
            L[n].up.copy_(sd[f"{n}.up"])
    set_mult(model, mult)
    return info


def merge(model: nn.Module) -> int:
    """Gộp LoRA vào trọng số gốc và gỡ lớp bọc (không đảo ngược được: muốn bật / tắt thì dùng set_mult). -> số lớp đã gộp."""
    L = layers(model)
    with torch.no_grad():
        for n, m in L.items():
            w = m.base.weight
            w.copy_((w.float() + m.delta().to(w.device)).to(w.dtype))
            _set(model, n, m.base)
    return len(L)


def load_and_merge(path: str, model: nn.Module, mult: float = 1.0) -> dict:
    """Máy chủ: nạp rồi gộp ngay (TEXTFIX_FLUX_LORA, textfix/draw/flux.py)."""
    info = load(path, model, mult)
    merge(model)
    return info
