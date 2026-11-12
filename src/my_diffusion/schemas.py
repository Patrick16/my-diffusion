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
    aspect_ratio: str = Field(default="16:9", description=f"One of: {list(ASPECT_RATIOS)}")

    seed: int | None = Field(default=None, description="None - random")
    # None - the default for the checkpoint kind (turbo: 8 steps / 1.0, raw: 28 / 4.5)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=1.0, le=20.0, description="CFG; 1.0 - off")
    shift: float | None = Field(default=None, description="Shift of the schedule (mu). None - automatic from the image size")

    loras: list[LoraSpec] = Field(default_factory=list)
    enhance_prompt: bool = False


class GenerateResponse(BaseModel):
    image_path: str
    width: int
    height: int
    seed: int
    prompt: str
    enhanced_prompt: str | None
    steps: int
    guidance_scale: float
    shift: float
    checkpoint: str
    matmul: str = Field(description="Matrix multiplication mode used: exact, fp8 or fp8_all (see MY_DIFFUSION_MATMUL)")
    loras: list[LoraSpec]
    timings: dict[str, float]


class EnhancePromptRequest(BaseModel):
    prompt: str
    seed: int | None = None


class EnhancePromptResponse(BaseModel):
    prompt: str
    enhanced_prompt: str


class LoadCheckpointRequest(BaseModel):
    name: str = Field(description="File in models/checkpoints")
