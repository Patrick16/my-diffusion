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
