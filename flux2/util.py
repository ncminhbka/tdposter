import base64
import io
import os
import sys
from pathlib import Path

import torch.nn as _nn


def _load_state_dict_verbose(module: "_nn.Module", sd: dict, label: str) -> None:
    """
    Loads a state dict and ALWAYS reports missing/unexpected keys explicitly,
    even when falling back to strict=False. A silent strict=False fallback can
    hide a broken key-conversion (e.g. Diffusers->BFL VAE remap) behind a model
    that "loads fine" but is missing real weights.
    """
    try:
        module.load_state_dict(sd, strict=True, assign=True)
        print(f"  -> [{label}] Loaded with strict=True (0 missing, 0 unexpected).")
    except Exception as e:
        print(f"Warning: Strict state dict loading failed for [{label}] ({e}), retrying non-strict...")
        result = module.load_state_dict(sd, strict=False, assign=True)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        print(f"  -> [{label}] Non-strict load: {len(missing)} missing key(s), {len(unexpected)} unexpected key(s).")
        if missing:
            print(f"     Missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"     Unexpected (first 10): {unexpected[:10]}")
        if missing or unexpected:
            print(
                "     ⚠️  This checkpoint did NOT fully match the target architecture. "
                "Outputs may be silently degraded (e.g. VAE decode quality). "
                "Do not treat this as a successful load without reviewing the lists above."
            )

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import load_mistral_small_embedder, load_qwen3_embedder

FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b-kv": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B-kv",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b-kv.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_KV_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
        "use_kv_cache": True,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}


def find_persistent_data_root() -> str | None:
    if "FLUX_CHECKPOINT_DIR" in os.environ and os.path.exists(os.environ["FLUX_CHECKPOINT_DIR"]):
        return os.environ["FLUX_CHECKPOINT_DIR"]

    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "persistent-data", "FLUX.2-klein-base-4B"),
        "/home/jovyan/persistent-data/FLUX.2-klein-base-4B",
        "/persistent-data/FLUX.2-klein-base-4B",
        os.path.join(home, "persistent-data"),
        "/home/jovyan/persistent-data",
        "/persistent-data",
    ]

    cwd = Path.cwd()
    for base in [cwd, Path(__file__).resolve().parent.parent.parent]:
        curr = base
        for _ in range(5):
            candidates.append(str(curr / "persistent-data" / "FLUX.2-klein-base-4B"))
            candidates.append(str(curr / "persistent-data"))
            if curr.parent == curr:
                break
            curr = curr.parent

    for c in candidates:
        if c and os.path.exists(c):
            if os.path.exists(os.path.join(c, "FLUX.2-klein-base-4B")):
                return os.path.join(c, "FLUX.2-klein-base-4B")
            if (
                os.path.exists(os.path.join(c, "flux-2-klein-base-4b.safetensors"))
                or os.path.exists(os.path.join(c, "vae"))
                or os.path.exists(os.path.join(c, "text_encoder"))
            ):
                return c
    return None


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        weight_path = None
        if config["model_path"] in os.environ and os.path.exists(os.environ[config["model_path"]]):
            weight_path = os.environ[config["model_path"]]
        else:
            p_root = find_persistent_data_root()
            if p_root:
                candidates = [
                    os.path.join(p_root, config["filename"]),
                    os.path.join(p_root, "transformer", "diffusion_pytorch_model.safetensors"),
                ]
                for cp in candidates:
                    if os.path.exists(cp):
                        weight_path = cp
                        print(f"Found local FLUX.2 weights at: {cp}")
                        break

        if weight_path is None:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except Exception as e:
                print(
                    f"Failed to access model repository on HuggingFace and local file not found ({config['filename']}). "
                    f"Error: {e}. Please set environment variable {config['model_path']} to local file path."
                )
                sys.exit(1)

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        _load_state_dict_verbose(model, sd, label=f"DiT:{model_name}")
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(model_name: str, device: str | torch.device = "cuda"):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    return config["text_encoder_load_fn"](device=device)


def convert_diffusers_vae_to_bfl(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in sd.items():
        new_k = k
        if new_k.startswith("vae."):
            new_k = new_k[4:]

        # quant_conv / post_quant_conv
        if new_k.startswith("quant_conv."):
            new_k = "encoder." + new_k
        elif new_k.startswith("post_quant_conv."):
            new_k = "decoder." + new_k

        # conv_norm_out -> norm_out
        new_k = new_k.replace("conv_norm_out.", "norm_out.")

        # Encoder: down_blocks.{i} -> down.{i} (same order)
        if "encoder.down_blocks." in new_k:
            new_k = new_k.replace("encoder.down_blocks.", "encoder.down.")
            new_k = new_k.replace(".downsamplers.0.conv.", ".downsample.conv.")
            new_k = new_k.replace(".conv_shortcut.", ".nin_shortcut.")
            new_k = new_k.replace(".resnets.", ".block.")

        # Decoder: up_blocks.{i} -> up.{3 - i} (reversed order in BFL)
        elif "decoder.up_blocks." in new_k:
            parts = new_k.split(".")
            # parts example: ['decoder', 'up_blocks', '0', 'resnets', '0', ...]
            level_idx = int(parts[2])
            bfl_level = 3 - level_idx
            parts[1] = "up"
            parts[2] = str(bfl_level)
            new_k = ".".join(parts)

            new_k = new_k.replace(".upsamplers.0.conv.", ".upsample.conv.")
            new_k = new_k.replace(".conv_shortcut.", ".nin_shortcut.")
            new_k = new_k.replace(".resnets.", ".block.")

        # mid_block -> mid
        if ".mid_block." in new_k:
            new_k = new_k.replace(".mid_block.resnets.0.", ".mid.block_1.")
            new_k = new_k.replace(".mid_block.resnets.1.", ".mid.block_2.")
            new_k = new_k.replace(".mid_block.attentions.0.group_norm.", ".mid.attn_1.norm.")
            new_k = new_k.replace(".mid_block.attentions.0.to_q.", ".mid.attn_1.q.")
            new_k = new_k.replace(".mid_block.attentions.0.to_k.", ".mid.attn_1.k.")
            new_k = new_k.replace(".mid_block.attentions.0.to_v.", ".mid.attn_1.v.")
            new_k = new_k.replace(".mid_block.attentions.0.to_out.0.", ".mid.attn_1.proj_out.")

        # Shape adjustment: Diffusers attention Linear (512, 512) -> BFL Conv2d (512, 512, 1, 1)
        if any(att in new_k for att in [".attn_1.q.weight", ".attn_1.k.weight", ".attn_1.v.weight", ".attn_1.proj_out.weight"]):
            if len(v.shape) == 2:
                v = v.view(v.shape[0], v.shape[1], 1, 1)

        new_sd[new_k] = v
    return new_sd


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    weight_path = None

    if "AE_MODEL_PATH" in os.environ and os.path.exists(os.environ["AE_MODEL_PATH"]):
        weight_path = os.environ["AE_MODEL_PATH"]
    else:
        p_root = find_persistent_data_root()
        if p_root:
            candidates = [
                os.path.join(p_root, "vae", "diffusion_pytorch_model.safetensors"),
                os.path.join(p_root, config["filename_ae"]),
                os.path.join(p_root, "vae", config["filename_ae"]),
            ]
            for cp in candidates:
                if os.path.exists(cp):
                    weight_path = cp
                    print(f"Found local AutoEncoder weights at: {cp}")
                    break

    if weight_path is None:
        # download from huggingface
        try:
            ae_repo = config.get("ae_repo_id", config["repo_id"])
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=ae_repo,
                filename=config["filename_ae"],
                repo_type="model",
            )
        except Exception as e:
            print(
                f"Failed to access AE repository on HuggingFace and local file not found ({config['filename_ae']}). "
                f"Error: {e}. Please set environment variable AE_MODEL_PATH to local file path."
            )
            sys.exit(1)

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    
    # Auto-convert diffusers format to BFL format if detected
    if any(k.startswith("encoder.down_blocks.") or k.startswith("quant_conv.") or k.startswith("decoder.up_blocks.") for k in sd.keys()):
        print("  -> Detected Diffusers format VAE keys, converting to BFL AutoEncoder format...")
        sd = convert_diffusers_vae_to_bfl(sd)
    elif any(k.startswith("vae.") for k in sd.keys()):
        sd = {k.replace("vae.", ""): v for k, v in sd.items()}

    _load_state_dict_verbose(ae, sd, label=f"AutoEncoder:{model_name}")

    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
