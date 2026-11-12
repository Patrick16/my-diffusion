"""Service settings. Read from the environment variables MY_DIFFUSION_* or from the .env file.

All model paths are relative to the models/ folder in the project root. That folder consists
of links to the real files (see scripts/link_models.ps1), so the code does not know
where the models are on a particular machine."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MY_DIFFUSION_", env_file=".env", extra="ignore")

    device: str = "cuda"

    # --- where things live ---
    models_dir: Path = PROJECT_ROOT / "models"
    output_dir: Path = PROJECT_ROOT / "outputs"

    checkpoint: str = "krea2_turbo_mxfp8.safetensors"   # file in models/checkpoints
    vae: str = "qwen_vae.safetensors"                   # file in models/vae (next to it lies qwen_vae_config.json)
    text_encoder: str = "Qwen3-VL-4B-Instruct"          # folder in models/text_encoders

    # turbo | raw | auto. The default steps and guidance depend on the kind (see engine.KIND_DEFAULTS).
    # auto: "raw" in the file name -> raw, otherwise turbo. The kind cannot be determined from the weights themselves.
    checkpoint_kind: Literal["auto", "turbo", "raw"] = "auto"

    # How to multiply the network's matrices (details at the top of weights.py):
    #   exact   - expand the compressed weights to bf16 and multiply as usual; works everywhere,
    #             the result matches ComfyUI bit for bit;
    #   fp8     - multiply mxfp8 layers directly in compressed form on the tensor cores, except the layers that
    #             the checkpoint author asked to compute exactly (as ComfyUI does). About 30% faster;
    #   fp8_all - the same for all mxfp8 layers, despite the author's marks. The fastest: ~40% of the time
    #             of sampling saved at 720p. The image has the same quality, but differs from exact in details.
    # Fast modes require an NVIDIA 50-series GPU (on another one, loading an mxfp8 checkpoint
    # will fail - then use exact). Checkpoints in other formats are always computed as exact.
    # Default is fp8_all: chosen after comparing images (scripts/profile_step.py --real).
    matmul: Literal["exact", "fp8", "fp8_all"] = "fp8_all"

    # Which kernel computes the network's attention (details at the top of transformer.py):
    #   cudnn   - the kernel from NVIDIA's cuDNN library: 2-2.4x faster, the result differs
    #             from default only at the rounding level;
    #   default - whatever torch picks; for comparison with ComfyUI.
    attention: Literal["cudnn", "default"] = "cudnn"

    # Keep model weights in "pinned" RAM: copying from it to the
    # GPU is 6-9x faster, and switching models on the GPU takes fractions of a second instead of
    # several seconds. Cost: ~17 GB of RAM that the system cannot give to other programs
    # or page out to the swap file. If RAM is short, turn it off.
    pin_memory: bool = True

    enhancer_max_new_tokens: int = 220

    # --- API ---
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def checkpoint_path(self) -> Path:
        return self.models_dir / "checkpoints" / self.checkpoint

    @property
    def vae_path(self) -> Path:
        return self.models_dir / "vae" / self.vae

    @property
    def text_encoder_dir(self) -> Path:
        return self.models_dir / "text_encoders" / self.text_encoder

    @property
    def lora_dir(self) -> Path:
        return self.models_dir / "loras"
