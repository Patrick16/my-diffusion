"""Shape of the API requests and responses.

A generation is a job (see api.py): the answer is the job's status, never the picture in base64. The PNG is fetched
separately, with GET /result/{id}/image, after the access check."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .resolutions import ASPECT_RATIOS, RESOLUTION_TIERS


class LoraSpec(BaseModel):
    name: str = Field(description="File in models/loras; a subfolder and no extension are allowed: 'people/anna'")
    weight: float = Field(default=1.0, ge=-4.0, le=4.0, description="Strength: 1.0 - as trained, 0 - off")


class OutpaintMargins(BaseModel):
    """How far to extend reference 1 on each side, in percent of its own height (top, bottom) or width (left, right)."""

    top: float = Field(default=0.0, ge=0.0, le=300.0)
    bottom: float = Field(default=0.0, ge=0.0, le=300.0)
    left: float = Field(default=0.0, ge=0.0, le=300.0)
    right: float = Field(default=0.0, ge=0.0, le=300.0)


class GenerateRequest(BaseModel):
    # Shown by the documentation pages as the starting body of "Try it".
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "prompt": "a fox in the snow, golden hour",
        "resolution": "720",
        "aspect_ratio": "16:9",
        "seed": 42,
        "loras": [{"name": "KreaAmateur_V2", "weight": 0.8}],
    }]})

    prompt: str
    negative_prompt: str = Field(default="", description="Only works when guidance_scale != 1 (that is, on raw checkpoints)")

    resolution: str = Field(default="720", description=f"One of: {list(RESOLUTION_TIERS)}")
    aspect_ratio: str = Field(default="16:9", description=f"One of: {list(ASPECT_RATIOS)}; 'auto' is allowed only with references in 'scene' mode (the exact proportions of reference 1)")

    seed: int | None = Field(default=None, description="None - random")
    # None - the default for the checkpoint kind (turbo: 8 steps / 1.0, raw: 28 / 4.5)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=1.0, le=20.0, description="CFG; 1.0 - off")
    # Limits CFG to a part of the run. Steps outside the interval call the network once instead of twice.
    guidance_start: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of the steps after which CFG starts; 0 - from the first step")
    guidance_end: float = Field(default=1.0, ge=0.0, le=1.0, description="Fraction of the steps after which CFG stops; 1 - until the last step")
    shift: float | None = Field(default=None, description="Shift of the schedule (mu). None - automatic from the image size")

    loras: list[LoraSpec] = Field(default_factory=list)
    enhance_prompt: bool = False

    # Image-to-image and inpaint. Without init_image it is plain text-to-image.
    init_image: str | None = Field(default=None, description="Base64 of a PNG or JPEG; a 'data:image/...;base64,' prefix is accepted")
    strength: float | None = Field(default=None, gt=0.0, le=1.0, description="None - the mode default: 0.6 image-to-image, 1.0 inpaint. Needs init_image")
    mask: str | None = Field(default=None, description="Base64 of a PNG: white = regenerate, black = keep. Needs init_image")
    feather: int = Field(default=2, ge=0, le=32, description="Softness of the mask edge, in latent pixels (1 latent pixel = 8 image pixels)")

    # Reference editing. Cannot be combined with init_image / mask.
    references: list[str] = Field(default_factory=list, description="Base64 PNG/JPEG reference pictures. Up to config max_references")
    reference_mode: str = Field(default="scene", pattern="^(scene|subjects)$", description="'scene': reference 1 is the main scene / landscape, the others are people or objects; 'subjects': all references are people or objects (aspect_ratio 'auto' is not allowed)")
    outpaint: OutpaintMargins | None = Field(default=None, description="Extend reference 1 by these percentages and generate the new area around it. Needs references in 'scene' mode; replaces aspect_ratio (the frame is reference 1 plus the margins at the resolution's pixel budget). All zeros = off")
    keep_original: bool = Field(default=False, description="Only with outpaint: keep reference 1's pixels unchanged and generate only the margins (a few pixels at the inner edges are blended to hide the seam). Off: the model also lightly redraws reference 1")
    remove_background: str = Field(default="none", pattern="^(none|subjects)$", description="'subjects': cut the background out of the people / objects before use - references 2.. in 'scene' mode, all references in 'subjects' mode")


class GenerationResult(BaseModel):
    """What a finished generation reports. The picture itself is fetched with GET /result/{id}/image."""

    width: int
    height: int
    seed: int
    prompt: str
    enhanced_prompt: str | None
    steps: int
    guidance_scale: float
    guidance_interval: list[float] = Field(description="[start, end] - the part of the run where CFG was applied, as fractions of the steps")
    shift: float
    checkpoint: str
    matmul: str = Field(description="Matrix multiplication mode used: exact, fp8 or fp8_all (see MY_DIFFUSION_MATMUL)")
    loras: list[LoraSpec]
    timings: dict[str, float]
    mode: str = Field(default="text2image", description="text2image, image2image, inpaint or edit")
    strength: float | None = None
    feather: int | None = None
    references: int = 0
    reference_mode: str | None = None
    outpaint: OutpaintMargins | None = Field(default=None, description="The margins actually applied; None when outpainting was off")
    remove_background: str = Field(default="none", description="'subjects' if the background of some references was removed")
    keep_original: bool = Field(default=False, description="True if reference 1's pixels were kept in an outpaint")
    edit_lora_rank: int | None = None


class GenerateResponse(GenerationResult):
    """The engine's own result, with the server-side path of the PNG (CLI and scripts)."""

    image_path: str


class JobInfo(BaseModel):
    """A generation job: the answer of POST /generate, GET /result/{id}, POST /cancel/{id} and GET /jobs."""

    job_id: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    cancel_requested: bool = Field(description="True once a cancel was asked; a running job stops at its next step")
    owner: str = Field(description="'<client>:<user>' that created the job")
    position: int | None = Field(default=None, description="1 = next to start; None once the job is not waiting")
    step: int | None = Field(default=None, description="Network steps finished so far (running jobs)")
    total_steps: int | None = None
    created_at: float = Field(description="Unix time the job entered the queue")
    started_at: float | None = None
    finished_at: float | None = None
    slot: str | None = Field(default=None, description="The GPU the job ran on")
    params: dict = Field(description="The request as submitted, without the pictures")
    result: GenerationResult | None = Field(default=None, description="Set when status is done")
    image_url: str | None = Field(default=None, description="Where to fetch the PNG, when status is done")
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="The pictures sent with the job, name -> URL: 'init', 'mask', 'ref1', 'ref2'. Kept for a limited "
        "time (MY_DIFFUSION_INPUTS_KEEP_DAYS); an expired one answers 404",
    )
    error: str | None = Field(default=None, description="Set when status is failed")


class EnhancePromptRequest(BaseModel):
    prompt: str
    seed: int | None = None


class EnhancePromptResponse(BaseModel):
    prompt: str
    enhanced_prompt: str


class LoadCheckpointRequest(BaseModel):
    name: str = Field(description="File in models/checkpoints")


class ConfigChange(BaseModel):
    """PATCH /config: which settings to change. The names and their rules are listed by GET /config."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"set": {"matmul": "exact", "queue_max_pending": 32}, "reset": [], "dry_run": False}]},
    )

    set: dict[str, Any] = Field(default_factory=dict, description="Setting name -> its new value")
    reset: list[str] = Field(
        default_factory=list,
        description="Settings whose saved value is removed: they fall back to the environment, .env or the default",
    )
    dry_run: bool = Field(default=False, description="Only check the change and report what it would do")
