"""Shape of the API requests and responses.

The response to a generation is not the image bytes but the path to a PNG in outputs/: the server and the client
are on one machine, so there is no point in pushing megabytes of base64 through HTTP. The image itself can be
fetched via GET /outputs/<file name>."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .resolutions import ASPECT_RATIOS, RESOLUTION_TIERS


class LoraSpec(BaseModel):
    name: str = Field(description="File in models/loras; a subfolder and no extension are allowed: 'people/anna'")
    weight: float = Field(default=1.0, ge=-4.0, le=4.0, description="Strength: 1.0 - as trained, 0 - off")


class GenerateRequest(BaseModel):
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


class GenerateResponse(BaseModel):
    image_path: str
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
    edit_lora_rank: int | None = None


class EnhancePromptRequest(BaseModel):
    prompt: str
    seed: int | None = None


class EnhancePromptResponse(BaseModel):
    prompt: str
    enhanced_prompt: str


class LoadCheckpointRequest(BaseModel):
    name: str = Field(description="File in models/checkpoints")
