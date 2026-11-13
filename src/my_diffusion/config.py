"""Service settings. Read from the environment variables MY_DIFFUSION_* or from the .env file.

All model paths are relative to the models/ folder in the project root. That folder consists
of links to the real files (see scripts/link_models.ps1), so the code does not know
where the models are on a particular machine."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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
    #   default - whatever torch picks; for comparison with ComfyUI;
    #   sage2   - SageAttention 2: an approximate kernel on quantized values, faster on large pictures and on
    #             edits with references, but the picture differs from cudnn. Needs the optional package:
    #             `uv sync --group sage2`. Never chosen automatically.
    attention: Literal["cudnn", "default", "sage2"] = "cudnn"
    # Only with attention=sage2: sequences (text + picture + reference tokens) shorter than this still go to cudnn.
    # Below ~6000 tokens sage2 saves under 10% of the sampling time, and with cudnn those pictures stay exactly
    # the same as without sage2. 6000 leaves 480 and 720 text-to-image and one-reference edits at 480 on cudnn;
    # 1080 (8200 tokens) and edits at 720 (7500+) use sage2. 0 = sage2 everywhere.
    attention_sage2_min_tokens: int = Field(default=6000, ge=0)

    # Keep model weights in "pinned" RAM: copying from it to the
    # GPU is 6-9x faster, and switching models on the GPU takes fractions of a second instead of
    # several seconds. Cost: ~17 GB of RAM that the system cannot give to other programs
    # or page out to the swap file. If RAM is short, turn it off.
    pin_memory: bool = True

    # LoRA streaming (as in Wan2GP): the attached LoRA pairs stay in pinned RAM, and each network block copies
    # its own pairs to the GPU right before it runs and drops them right after (weights.streamed_adapters).
    # The GPU then holds one block's pairs (~30 MB at rank 128) instead of the whole LoRA (0.85 GB at rank 128,
    # 1.7 GB at 256); the price is copying the LoRA over PCIe on every pass of the network. The result is bit
    # for bit the same. False keeps every pair on the GPU while the LoRA is attached (the old way, for comparison).
    lora_streaming: bool = True

    enhancer_max_new_tokens: int = 220

    # --- reference editing (engine.generate with references) ---
    # Editing needs three parts together: this LoRA on the network, the pictures in the Qwen prompt
    # (vision tower) and their clean latents in the network's sequence. The LoRA is added automatically
    # with this weight unless the request names it itself (weight 0 there switches it off).
    edit_lora: str = "krea2_identity_edit_v1_2"         # file in models/loras
    edit_lora_weight: float = 1.0
    vision_max_side: int = 768   # longer side of a picture before the vision tower; the encoder never goes above 768
    max_references: int = 2      # reference 1 = the scene (the output frame), reference 2 = a subject or person

    # --- background removal (background.py) ---
    rembg_model: str = "rembg/u2net.onnx"               # file in models/ (u2net ONNX, run by onnxruntime on the CPU)

    # --- job queue and scheduler (store.py, jobs.py) ---
    # Which GPUs run jobs. One "slot" per device, each with its own engine and its own copy of the models.
    # None - only `device` (one slot: the normal setup). "auto" - every visible CUDA device.
    # A list such as ["cuda:0", "cuda:1"] - exactly those. A GPU runs one generation at a time: one generation
    # fills it (up to ~14 GB of VRAM at 1080p), so more jobs per card would only make each one slower.
    # Cost of every extra slot: another copy of the weights in RAM (~19 GB pinned).
    devices: list[str] | Literal["auto"] | None = None
    # Jobs and their state live in SQLite, so the queue survives a restart.
    database_path: Path = PROJECT_ROOT / "data" / "jobs.sqlite3"
    # Pictures sent with a job (init_image, mask, references) live here, one folder per job. They are kept after the
    # job ends so clients can show and reuse them (GET /result/{id}/inputs/{name}), and deleted `inputs_keep_days`
    # days after the job was queued (checked at start and every hour). 0 = delete them as soon as the job ends.
    inputs_dir: Path = PROJECT_ROOT / "data" / "inputs"
    inputs_keep_days: float = Field(default=7.0, ge=0)
    # How many jobs may wait at once (all users together), and per user (queued + running; admins are exempt).
    queue_max_pending: int = Field(default=16, ge=1)
    queue_max_per_user: int = Field(default=3, ge=1)

    # --- access (auth.py) ---
    # Client name -> secret key, sent as "Authorization: Bearer <key>". Empty = access control is OFF and every
    # request is an admin: fine for a server on 127.0.0.1 used by one person, not for anything shared.
    # Example: MY_DIFFUSION_API_KEYS='{"web": "<32+ random chars>", "telegram": "<another one>"}'
    api_keys: dict[str, str] = Field(default_factory=dict)
    # Clients whose key gives admin rights: they see and cancel every job. The others are "services" that act
    # for an end user named in the X-User-Id header (the Telegram bot passes the Telegram user id there).
    admin_clients: list[str] = Field(default_factory=list)

    # --- API ---
    host: str = "127.0.0.1"
    port: int = 8000
    # Interactive API documentation: /docs (Swagger UI), /redoc (ReDoc), /scalar (Scalar) and the schema at
    # /openapi.json. The pages need no key (the key is entered in them), but they show the whole API surface,
    # so a server open to the internet may turn them off.
    api_docs: bool = True

    @field_validator("api_keys")
    @classmethod
    def _strong_keys(cls, keys: dict[str, str]) -> dict[str, str]:
        weak = [name for name, secret in keys.items() if len(secret) < 16]
        if weak:
            raise ValueError(f"API keys must be at least 16 characters long; too short: {weak}")
        if len(set(keys.values())) != len(keys):
            raise ValueError("every client needs its own API key")
        return keys

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
    def rembg_model_path(self) -> Path:
        return self.models_dir / self.rembg_model

    @property
    def lora_dir(self) -> Path:
        return self.models_dir / "loras"
