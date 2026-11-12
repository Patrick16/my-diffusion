"""LoRA specification for a generation request.

Applied in comfy_workflow.py as a chain of LoraLoaderModelOnly nodes -
the path must be the file name RELATIVE to ComfyUI/models/loras (the same
as shown in the node's dropdown list in ComfyUI itself)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoraSpec(BaseModel):
    path: str  # file name relative to ComfyUI/models/loras
    weight: float = Field(default=1.0, ge=-2.0, le=2.0)
