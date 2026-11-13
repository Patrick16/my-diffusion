"""HTTP layer on top of the engine. No ML logic here: validate the request, call the engine
in a separate thread (a generation takes a long time, and the server must keep
responding to /health), and turn errors into understandable HTTP codes.

    uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000

Interactive documentation of all endpoints: http://127.0.0.1:8000/docs"""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import imaging
from .config import Settings
from .engine import KIND_DEFAULTS, Krea2Engine
from .resolutions import all_combinations
from .schemas import (
    EnhancePromptRequest,
    EnhancePromptResponse,
    GenerateRequest,
    GenerateResponse,
    LoadCheckpointRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = Settings()
engine = Krea2Engine(settings)
_load_error: str | None = None

# All GPU work runs in ONE persistent thread. There is one GPU, so computing
# in parallel is impossible anyway (requests just wait in this thread's queue), and persistence
# of the thread matters for speed: the cuDNN attention kernel remembers its setup for each sequence length
# separately in each thread (~0.2-0.4 s, see transformer.py), and in a "random"
# thread from a shared pool it would pay for it again.
_gpu_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load in the background: the server starts responding right away, and /health shows "loading".
    async def load() -> None:
        global _load_error
        try:
            await _in_gpu_thread(engine.load)
        except Exception as exc:  # noqa: BLE001 - we show the reason in /health instead of silently crashing the server
            logging.exception("could not load the models")
            _load_error = f"{type(exc).__name__}: {exc}"

    task = asyncio.create_task(load())
    yield
    task.cancel()


app = FastAPI(title="my-diffusion", description="Local backend for Krea 2 (text-to-image)", lifespan=lifespan)
settings.output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")


def _require_loaded() -> None:
    if _load_error:
        raise HTTPException(500, f"Models failed to load: {_load_error}")
    if not engine.loaded:
        raise HTTPException(503, "Models are still loading, try again in a few seconds")


async def _in_gpu_thread(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(_gpu_thread, functools.partial(func, *args, **kwargs))


async def _run(func, *args, **kwargs):
    """Calls the engine, translating its errors into HTTP codes."""
    try:
        return await _in_gpu_thread(func, *args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except ImportError as exc:   # an optional package is missing on this machine: a server problem, not a bad request
        raise HTTPException(503, str(exc)) from exc


@app.get("/health")
async def health() -> dict:
    status = "error" if _load_error else "ok" if engine.loaded else "loading"
    return {
        "status": status, "error": _load_error, "checkpoint": engine.checkpoint_name,
        "kind": engine.kind if engine.loaded else None,
        "matmul": engine.matmul, "native_layers": engine.native_layers, "attention": settings.attention,
    }


@app.get("/resolutions")
async def resolutions() -> dict:
    """Table: resolution -> format -> (width, height)."""
    return {
        tier: {ratio: {"width": d.width, "height": d.height} for ratio, d in ratios.items()}
        for tier, ratios in all_combinations().items()
    }


@app.get("/checkpoints")
async def checkpoints() -> dict:
    return {"current": engine.checkpoint_name, "available": engine.list_files("checkpoints"), "defaults": KIND_DEFAULTS}


@app.post("/checkpoints/load")
async def load_checkpoint(request: LoadCheckpointRequest) -> dict:
    """Switch the checkpoint without restarting the server."""
    _require_loaded()

    def switch() -> None:
        with engine._lock:
            engine.load_checkpoint(request.name)

    await _run(switch)
    return {"current": engine.checkpoint_name, "kind": engine.kind}


@app.get("/loras")
async def loras() -> list[str]:
    return engine.list_files("loras")


@app.post("/enhance-prompt", response_model=EnhancePromptResponse)
async def enhance_prompt(request: EnhancePromptRequest) -> EnhancePromptResponse:
    _require_loaded()
    enhanced = await _run(engine.enhance_prompt, request.prompt, request.seed)
    return EnhancePromptResponse(prompt=request.prompt, enhanced_prompt=enhanced)


_DATA_URI_PREFIX = re.compile(r"^data:[^;,]*;base64,", re.IGNORECASE)


def _decode_image(value: str, field: str) -> Image.Image:
    """Base64 (optionally as a data URI) -> RGB picture. Errors become 400s that name the field.
    Decoding happens here, not in the engine, so bad input fails before the GPU is touched."""
    payload = _DATA_URI_PREFIX.sub("", value.strip())
    try:
        data = base64.b64decode(payload, validate=True)  # binascii.Error is a ValueError
    except ValueError as exc:
        raise HTTPException(400, f"{field} is not valid base64") from exc
    try:
        return imaging.load_rgb(data)
    except ValueError as exc:
        raise HTTPException(400, f"{field}: {exc}") from exc


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Text-to-image, or with init_image: image-to-image (the picture is noised to `strength` and
    denoised again), or with init_image + mask: inpaint (only the white part of the mask is
    regenerated, the rest is kept). Images are plain base64 PNG/JPEG inside the JSON body.
    Input is validated before the models are checked, so a bad request gets 400 even while loading.

    With `references` (1 or 2 pictures) it is reference editing, which cannot be combined with init_image or
    mask: the picture is built from the references. Three parts work together: the edit LoRA is added
    automatically (reduced rank for larger outputs or two references), the references are also shown to the
    vision encoder in the prompt, and their tokens are appended to the image tokens as context. Size limits
    (for example 1080 with a reference) come back as 400. aspect_ratio "auto" takes the format nearest to reference 1."""
    if len(request.references) > settings.max_references:
        raise HTTPException(400, f"at most {settings.max_references} references are supported, got {len(request.references)}")
    if request.references and (request.init_image is not None or request.mask is not None):
        raise HTTPException(400, "references cannot be combined with init_image or mask")
    if request.aspect_ratio == "auto" and not request.references:
        raise HTTPException(400, "aspect_ratio 'auto' needs references")
    if request.aspect_ratio == "auto" and request.reference_mode != "scene":
        raise HTTPException(400, "aspect_ratio 'auto' needs reference_mode 'scene'")
    outpaint = request.outpaint
    if outpaint is not None and any(outpaint.model_dump().values()):
        if not request.references:
            raise HTTPException(400, "outpaint needs references")
        if request.reference_mode != "scene":
            raise HTTPException(400, "outpaint needs reference_mode 'scene'")
    elif request.keep_original:
        raise HTTPException(400, "keep_original needs outpaint (non-zero margins)")
    if request.mask is not None and request.init_image is None:
        raise HTTPException(400, "mask needs init_image")
    if request.strength is not None and request.init_image is None:
        raise HTTPException(400, "strength needs init_image")
    init_image = _decode_image(request.init_image, "init_image") if request.init_image is not None else None
    mask = _decode_image(request.mask, "mask") if request.mask is not None else None
    references = [_decode_image(value, f"references[{i}]") for i, value in enumerate(request.references)]

    _require_loaded()
    result = await _run(
        engine.generate,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        resolution=request.resolution,
        aspect_ratio=request.aspect_ratio,
        seed=request.seed,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        guidance_interval=(request.guidance_start, request.guidance_end),
        shift=request.shift,
        loras=[(lora.name, lora.weight) for lora in request.loras],
        enhance=request.enhance_prompt,
        init_image=init_image,
        strength=request.strength,
        mask=mask,
        feather=request.feather,
        references=references,
        reference_mode=request.reference_mode,
        outpaint=None if outpaint is None else (outpaint.top, outpaint.bottom, outpaint.left, outpaint.right),
        remove_background=request.remove_background,
        keep_original=request.keep_original,
    )
    return GenerateResponse(**result)
