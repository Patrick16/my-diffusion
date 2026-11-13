"""Service settings. Read from the environment variables MY_DIFFUSION_* or from the .env file, and on top of
them from the override file that the config API writes (see load_settings at the bottom).

All model paths are relative to the models/ folder in the project root. That folder consists
of links to the real files (see scripts/link_models.ps1), so the code does not know
where the models are on a particular machine. The checkpoints and LoRA folders can be set separately.

The Settings class is also the registry of the config API (GET /config, PATCH /config in api.py): every field
is declared with _setting(...), which records next to the default how a change of the field takes effect
(`apply`), whether the API may change it, its group and a one-line description for the Settings page. The
endpoints know nothing about particular fields, so a new field needs nothing but its declaration here."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# How a change of a setting takes effect (the config API reports it for every field):
#   live          - at once, without reloading anything. Settings the engine reads are swapped on the GPU slot's
#                   thread between two generations, so a running generation ends with the values it started with;
#   reload_models - the models are read again; allowed only while no job waits or runs;
#   restart       - saved to the override file, used from the next start of the server.
APPLY_CLASSES = ("live", "reload_models", "restart")
GROUPS = ("paths", "models", "performance", "editing", "queue", "access", "server")


def _setting(
    default: Any = ..., *, apply: str, group: str, description: str, editable: bool = True, secret: bool = False,
    kind: str | None = None, effective: str | None = None, **kwargs: Any,
) -> Any:
    """A Settings field with the metadata of the config API.

    apply, group - see APPLY_CLASSES and GROUPS. editable=False: shown by GET /config but never changed through
    the API (set it in the environment). secret=True: the value is never shown either. kind="directory": a folder
    the API checks against the allowed roots. effective: the name of the property that gives the value really
    used when the field itself is None. Other arguments go to pydantic's Field (ge, le, default_factory ...)."""
    if apply not in APPLY_CLASSES or group not in GROUPS:
        raise ValueError(f"unknown apply class {apply!r} or group {group!r}")
    extra = {"apply": apply, "group": group, "editable": editable and not secret, "secret": secret}
    if kind is not None:
        extra["kind"] = kind
    if effective is not None:
        extra["effective"] = effective
    if default is ...:
        return Field(description=description, json_schema_extra=extra, **kwargs)
    return Field(default, description=description, json_schema_extra=extra, **kwargs)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MY_DIFFUSION_", env_file=".env", extra="ignore")

    device: str = _setting(
        "cuda", apply="restart", group="server",
        description="torch device of the single GPU slot (used when `devices` is not set)",
    )

    # --- where things live ---
    models_dir: Path = _setting(
        PROJECT_ROOT / "models", apply="reload_models", group="paths", kind="directory",
        description="Folder with the models: vae/, text_encoders/, rembg/ and, unless set separately, checkpoints/ and loras/",
    )
    # The checkpoints and LoRA folders. None = the folder of that name under models_dir (the layout made by
    # scripts/link_models.ps1). The code reads the folder really used from checkpoint_dir / lora_dir below.
    checkpoints_dir: Path | None = _setting(
        None, apply="reload_models", group="paths", kind="directory", effective="checkpoint_dir",
        description="Folder with the checkpoints; empty = checkpoints/ under models_dir",
    )
    loras_dir: Path | None = _setting(
        None, apply="live", group="paths", kind="directory", effective="lora_dir",
        description="Folder with the LoRAs; empty = loras/ under models_dir",
    )
    # Read-only for the config API: pictures are found in it by name, so changing it would hide the old ones.
    output_dir: Path = _setting(
        PROJECT_ROOT / "outputs", apply="restart", group="paths", editable=False,
        description="Where generated PNGs are saved",
    )
    # Folders the config API may show (GET /fs) and accept as models_dir / checkpoints_dir / loras_dir, besides
    # the model folders already in use. Empty = only those. Never changed through the API itself: otherwise the
    # API could widen its own sandbox. Example: MY_DIFFUSION_FS_ROOTS='["D:\\ComfyUI\\models", "D:\\AI"]'
    fs_roots: list[Path] = _setting(
        apply="restart", group="paths", editable=False, default_factory=list,
        description="Extra folders the config API may browse and accept as model folders",
    )

    checkpoint: str = _setting(   # file in the checkpoints folder
        "krea2_turbo_mxfp8.safetensors", apply="reload_models", group="models",
        description="Checkpoint loaded at start: a file in the checkpoints folder (.safetensors or .gguf)",
    )
    vae: str = _setting(   # file in models/vae (next to it lies qwen_vae_config.json)
        "qwen_vae.safetensors", apply="reload_models", group="models", description="VAE: a file in models/vae",
    )
    text_encoder: str = _setting(   # folder in models/text_encoders
        "Qwen3-VL-4B-Instruct", apply="reload_models", group="models",
        description="Text encoder and prompt enhancer: a folder in models/text_encoders",
    )

    # turbo | raw | auto. The default steps and guidance depend on the kind (see engine.KIND_DEFAULTS).
    # auto: "raw" in the file name -> raw, otherwise turbo. The kind cannot be determined from the weights themselves.
    checkpoint_kind: Literal["auto", "turbo", "raw"] = _setting(
        "auto", apply="live", group="models",
        description="Checkpoint kind, sets the default steps and guidance; auto = 'raw' in the file name means raw",
    )

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
    matmul: Literal["exact", "fp8", "fp8_all"] = _setting(
        "fp8_all", apply="live", group="performance",
        description="How the network's matrices are multiplied: exact (works everywhere), fp8 or fp8_all "
        "(faster; mxfp8 checkpoints on an NVIDIA 50-series GPU)",
    )

    # Which kernel computes the network's attention (details at the top of transformer.py):
    #   cudnn   - the kernel from NVIDIA's cuDNN library: 2-2.4x faster, the result differs
    #             from default only at the rounding level;
    #   default - whatever torch picks; for comparison with ComfyUI;
    #   sage2   - SageAttention 2: an approximate kernel on quantized values, faster on large pictures and on
    #             edits with references, but the picture differs from cudnn. Needs the optional package:
    #             `uv sync --group sage2`. Never chosen automatically.
    attention: Literal["cudnn", "default", "sage2"] = _setting(
        "cudnn", apply="live", group="performance",
        description="Attention kernel: cudnn (fast, exact), default (torch's choice) or sage2 (approximate, "
        "faster on large pictures; needs the optional package)",
    )
    # Only with attention=sage2: sequences (text + picture + reference tokens) shorter than this still go to cudnn.
    # Below ~6000 tokens sage2 saves under 10% of the sampling time, and with cudnn those pictures stay exactly
    # the same as without sage2. 6000 leaves 480 and 720 text-to-image and one-reference edits at 480 on cudnn;
    # 1080 (8200 tokens) and edits at 720 (7500+) use sage2. 0 = sage2 everywhere.
    attention_sage2_min_tokens: int = _setting(
        6000, apply="live", group="performance", ge=0,
        description="Only with attention=sage2: sequences shorter than this many tokens still use cudnn; 0 = sage2 everywhere",
    )

    # Keep model weights in "pinned" RAM: copying from it to the
    # GPU is 6-9x faster, and switching models on the GPU takes fractions of a second instead of
    # several seconds. Cost: ~17 GB of RAM that the system cannot give to other programs
    # or page out to the swap file. If RAM is short, turn it off.
    pin_memory: bool = _setting(
        True, apply="reload_models", group="performance",
        description="Keep the weights in pinned RAM: models switch on the GPU several times faster, at ~17 GB of RAM",
    )

    # LoRA streaming (as in Wan2GP): the attached LoRA pairs stay in pinned RAM, and each network block copies
    # its own pairs to the GPU right before it runs and drops them right after (weights.streamed_adapters).
    # The GPU then holds one block's pairs (~30 MB at rank 128) instead of the whole LoRA (0.85 GB at rank 128,
    # 1.7 GB at 256); the price is copying the LoRA over PCIe on every pass of the network. The result is bit
    # for bit the same. False keeps every pair on the GPU while the LoRA is attached (the old way, for comparison).
    lora_streaming: bool = _setting(
        True, apply="live", group="performance",
        description="Stream LoRA pairs to the GPU block by block: less VRAM, slightly slower",
    )

    enhancer_max_new_tokens: int = _setting(
        220, apply="live", group="models", ge=1, description="Length limit of the prompt enhancer's answer, in tokens",
    )

    # --- reference editing (engine.generate with references) ---
    # Editing needs three parts together: this LoRA on the network, the pictures in the Qwen prompt
    # (vision tower) and their clean latents in the network's sequence. The LoRA is added automatically
    # with this weight unless the request names it itself (weight 0 there switches it off).
    edit_lora: str = _setting(   # file in the LoRA folder
        "krea2_identity_edit_v1_2", apply="live", group="editing",
        description="Edit LoRA added automatically to requests with references: a file in the LoRA folder",
    )
    edit_lora_weight: float = _setting(
        1.0, apply="live", group="editing", description="Weight of the edit LoRA unless the request names it itself",
    )
    # longer side of a picture before the vision tower; the encoder never goes above 768
    vision_max_side: int = _setting(
        768, apply="live", group="editing", ge=16,
        description="Longer side of a reference picture before the vision tower, in pixels",
    )
    # reference 1 = the scene (the output frame), reference 2 = a subject or person
    max_references: int = _setting(
        2, apply="live", group="editing", ge=0, description="Maximum number of reference pictures in a request",
    )

    # --- background removal (background.py) ---
    rembg_model: str = _setting(   # file in models/ (u2net ONNX, run by onnxruntime on the CPU)
        "rembg/u2net.onnx", apply="live", group="editing",
        description="Background-removal model: a file path inside models_dir",
    )

    # --- job queue and scheduler (store.py, jobs.py) ---
    # Which GPUs run jobs. One "slot" per device, each with its own engine and its own copy of the models.
    # None - only `device` (one slot: the normal setup). "auto" - every visible CUDA device.
    # A list such as ["cuda:0", "cuda:1"] - exactly those. A GPU runs one generation at a time: one generation
    # fills it (up to ~14 GB of VRAM at 1080p), so more jobs per card would only make each one slower.
    # Cost of every extra slot: another copy of the weights in RAM (~19 GB pinned).
    devices: list[str] | Literal["auto"] | None = _setting(
        None, apply="restart", group="server",
        description='GPUs that run jobs: empty = `device` only, "auto" = every CUDA device, or a list such as '
        '["cuda:0", "cuda:1"]; each holds its own copy of the models',
    )
    # Jobs and their state live in SQLite, so the queue survives a restart.
    # Read-only for the config API, like inputs_dir: the existing jobs and pictures are found there.
    database_path: Path = _setting(
        PROJECT_ROOT / "data" / "jobs.sqlite3", apply="restart", group="queue", editable=False,
        description="The job database (SQLite)",
    )
    # Pictures sent with a job (init_image, mask, references) live here, one folder per job. They are kept after the
    # job ends so clients can show and reuse them (GET /result/{id}/inputs/{name}), and deleted `inputs_keep_days`
    # days after the job was queued (checked at start and every hour). 0 = delete them as soon as the job ends.
    inputs_dir: Path = _setting(
        PROJECT_ROOT / "data" / "inputs", apply="restart", group="queue", editable=False,
        description="Where the pictures sent with jobs are kept",
    )
    inputs_keep_days: float = _setting(
        7.0, apply="live", group="queue", ge=0,
        description="Days to keep a job's input pictures; 0 = delete them when the job ends",
    )
    # How many jobs may wait at once (all users together), and per user (queued + running; admins are exempt).
    queue_max_pending: int = _setting(
        16, apply="live", group="queue", ge=1, description="Jobs that may wait at once, all users together",
    )
    queue_max_per_user: int = _setting(
        3, apply="live", group="queue", ge=1, description="Jobs one user may have waiting or running; admins are exempt",
    )

    # --- access (auth.py) ---
    # Client name -> secret key, sent as "Authorization: Bearer <key>". Empty = access control is OFF and every
    # request is an admin: fine for a server on 127.0.0.1 used by one person, not for anything shared.
    # Example: MY_DIFFUSION_API_KEYS='{"web": "<32+ random chars>", "telegram": "<another one>"}'
    # Neither this nor admin_clients is ever changed through the config API, and the keys are never shown by it:
    # secrets stay in the environment, and an admin API that edits its own access list is a way to lock oneself out.
    api_keys: dict[str, str] = _setting(
        apply="restart", group="access", secret=True, default_factory=dict,
        description="Client name -> secret key; empty = access control is off",
    )
    # Clients whose key gives admin rights: they see and cancel every job. The others are "services" that act
    # for an end user named in the X-User-Id header (the Telegram bot passes the Telegram user id there).
    admin_clients: list[str] = _setting(
        apply="restart", group="access", editable=False, default_factory=list,
        description="Clients whose key gives admin rights",
    )

    # --- API ---
    # For reference only: the server itself does not read them, the address is given to uvicorn on the command line.
    host: str = _setting(
        "127.0.0.1", apply="restart", group="server", editable=False,
        description="Bind address (informational: the address is set on the uvicorn command line)",
    )
    port: int = _setting(
        8000, apply="restart", group="server", editable=False, ge=1, le=65535,
        description="Port (informational: the port is set on the uvicorn command line)",
    )
    # Interactive API documentation: /docs (Swagger UI), /redoc (ReDoc), /scalar (Scalar) and the schema at
    # /openapi.json. The pages need no key (the key is entered in them), but they show the whole API surface,
    # so a server open to the internet may turn them off.
    api_docs: bool = _setting(
        True, apply="restart", group="server",
        description="Serve the API documentation pages (/docs, /scalar, /redoc, /openapi.json)",
    )
    # Where the config API saves changed settings (see load_settings below). Only from the environment.
    settings_file: Path = _setting(
        PROJECT_ROOT / "data" / "settings.override.json", apply="restart", group="server", editable=False,
        description="File where settings changed through the API are saved; read on top of the environment",
    )

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
    def checkpoint_dir(self) -> Path:
        """The checkpoints folder really used: checkpoints_dir, or checkpoints/ under models_dir."""
        return self.checkpoints_dir if self.checkpoints_dir is not None else self.models_dir / "checkpoints"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.checkpoint

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
        """The LoRA folder really used: loras_dir, or loras/ under models_dir."""
        return self.loras_dir if self.loras_dir is not None else self.models_dir / "loras"


def inside_folder(folder: Path, name: str) -> Path:
    """Path to the model file `name` inside folder (".safetensors" may be left out); does not allow escaping it via '..'."""
    folder = folder.resolve()
    path = (folder / name).resolve()
    if not path.is_file():
        # the name is given without an extension; we append it rather than replace: names can contain dots ("style_v1.2.1")
        path = path.with_name(path.name + ".safetensors")
    if not path.is_relative_to(folder):
        raise ValueError(f"path {name!r} is outside {folder}")
    if not path.is_file():
        raise FileNotFoundError(f"no file {name!r} in {folder}")
    return path


def setting_meta(name: str) -> dict:
    """The config API metadata of a field (see _setting). KeyError for a name that is not a setting."""
    return Settings.model_fields[name].json_schema_extra


for _name, _field in Settings.model_fields.items():
    # A field declared without _setting() would be invisible to the config API or, worse, shown without its rules.
    if not isinstance(_field.json_schema_extra, dict) or _field.json_schema_extra.get("apply") not in APPLY_CLASSES:
        raise TypeError(f"Settings.{_name} must be declared with _setting(...): the config API needs its metadata")


# ---------------------------------------------------------------------- the override file
# Settings changed through the config API are saved in a JSON file (settings_file), a flat object such as
# {"matmul": "exact", "queue_max_pending": 32}, and read on top of everything else:
#
#     override file  >  environment variables  >  .env  >  defaults
#
# The user's .env is never written. The file is stronger than the environment because otherwise a value in .env
# would silently undo what an admin has just saved on the Settings page; PATCH /config "reset" removes a key and
# the value falls back to the environment. Only settings the API may change are read from the file (no secrets),
# and a broken file never keeps the server from starting: what cannot be used is skipped and reported.

@dataclass
class LoadedSettings:
    settings: Settings
    overrides: dict[str, Any] = field(default_factory=dict)   # the part of the file in use, as JSON values
    problems: list[str] = field(default_factory=list)         # what was skipped, and why


def read_overrides(path: Path) -> tuple[dict[str, Any], list[str]]:
    """The override file -> (the entries that name changeable settings, problems). A missing file is empty."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, []
    except (OSError, ValueError) as exc:
        return {}, [f"{path} could not be read and is ignored: {exc}"]
    if not isinstance(raw, dict):
        return {}, [f"{path} must hold a JSON object; ignored"]
    usable: dict[str, Any] = {}
    problems: list[str] = []
    for name, value in raw.items():
        if name not in Settings.model_fields:
            problems.append(f"{name}: not a setting; ignored")
        elif not setting_meta(name)["editable"]:
            problems.append(f"{name}: cannot be set in the override file (use the environment); ignored")
        else:
            usable[name] = value
    return usable, problems


def write_overrides(path: Path, values: dict[str, Any]) -> None:
    """Saves the override file atomically: a reader sees the old file or the new one, never a half-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(values, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validation_messages(exc: ValidationError) -> dict[str, str]:
    """A pydantic error -> {field name: what is wrong}, one line per field."""
    messages: dict[str, str] = {}
    for error in exc.errors():
        name = str(error["loc"][0]) if error["loc"] else "settings"
        messages[name] = "; ".join(filter(None, [messages.get(name), error["msg"]]))
    return messages


def load_settings(**source: Any) -> LoadedSettings:
    """The settings the server runs with: environment and .env, then the override file on top.

    `source` is passed to Settings (e.g. _env_file=None in tests). An entry of the file that does not validate
    is dropped and reported, the rest still applies; an invalid environment raises as before."""
    loaded = Settings(**source)
    overrides, problems = read_overrides(loaded.settings_file)
    while overrides:
        try:
            loaded = Settings(**overrides, **source)   # arguments are stronger than the environment
            break
        except ValidationError as exc:
            bad = {name: text for name, text in validation_messages(exc).items() if name in overrides}
            if not bad:
                raise
            for name, text in bad.items():
                problems.append(f"{name}: {text}; ignored")
                del overrides[name]
    for problem in problems:
        logger.warning("settings override file: %s", problem)
    return LoadedSettings(loaded, overrides, problems)
