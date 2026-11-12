"""HTTP layer on top of the engine. No ML logic here: validate the request, call the engine
in a separate thread (a generation takes a long time, and the server must keep
responding to /health), and turn errors into understandable HTTP codes.

    uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000

Interactive documentation of all endpoints: http://127.0.0.1:8000/docs"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load in the background: the server starts responding right away, and /health shows "loading".
    async def load() -> None:
        global _load_error
        try:
            await asyncio.to_thread(engine.load)
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


async def _run(func, *args, **kwargs):
    """Calls the engine, translating its errors into HTTP codes."""
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/health")
async def health() -> dict:
    status = "error" if _load_error else "ok" if engine.loaded else "loading"
    return {
        "status": status, "error": _load_error, "checkpoint": engine.checkpoint_name,
        "kind": engine.kind if engine.loaded else None,
        "matmul": engine.matmul, "native_layers": engine.native_layers,
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
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
        shift=request.shift,
        loras=[(lora.name, lora.weight) for lora in request.loras],
        enhance=request.enhance_prompt,
    )
    return GenerateResponse(**result)
