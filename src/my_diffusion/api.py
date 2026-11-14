"""HTTP layer on top of the engine. No ML logic here: check who is calling, validate the request, hand it to the
scheduler, and turn errors into understandable HTTP codes.

    uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5

Run exactly ONE server process (no `--workers N`): every process loads its own copy of the models.
Interactive documentation of all endpoints: http://127.0.0.1:8000/docs

The flow:
    POST /generate            -> 202 with the job id; the job waits in the queue
    GET  /result/{id}         -> status, position in the queue, progress, result or error
    GET  /result/{id}/events  -> the same as /result/{id}, but pushed as server-sent events (no polling)
    GET  /result/{id}/image   -> the PNG, once the status is "done"
    POST /cancel/{id}         -> cancel: a waiting job at once, a running one at its next step
    GET  /jobs                -> your jobs (an admin sees everybody's)

Settings at run time (admins only; see runtime_config.py and the bottom of this file):
    GET   /config             -> every setting: its value, where it comes from, how a change takes effect
    PATCH /config             -> change settings: applied at once where possible, saved to the override file
    GET   /fs                 -> folders of the server, for choosing the model folders (allowed roots only)

Access: see auth.py. A job is visible only to the client user who created it and to admins; for anyone else it
does not exist (404, not 403, so job ids cannot be probed)."""

from __future__ import annotations

import asyncio
import base64
import functools
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import imaging, runtime_config
from .auth import AuthError, Caller, resolve_caller
from .config import Settings, load_settings, write_overrides
from .engine import KIND_DEFAULTS, Krea2Engine
from .jobs import JobNotCancellable, JobNotFound, QueueFull, Scheduler, Slot
from .resolutions import all_combinations
from .schemas import (
    ConfigChange,
    EnhancePromptRequest,
    EnhancePromptResponse,
    GenerateRequest,
    GenerationResult,
    JobInfo,
    LoadCheckpointRequest,
)
from .store import InputFiles, JobRecord, JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# The environment and .env, with the settings saved by the config API on top (config.load_settings).
_loaded_settings = load_settings()
# One object for the whole life of the process: the config API changes its fields in place.
settings = _loaded_settings.settings
_overrides: dict = dict(_loaded_settings.overrides)        # the override file's content in use
_config_problems: list[str] = list(_loaded_settings.problems)
_config_lock: asyncio.Lock | None = None                    # one PATCH /config at a time; created at startup


def resolve_devices(s: Settings) -> list[str]:
    """settings.devices -> the list of devices to put a slot on (see config.py)."""
    if s.devices is None:
        return [s.device]
    if s.devices == "auto":
        import torch

        found = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return found or [s.device]
    if not s.devices:
        raise ValueError("MY_DIFFUSION_DEVICES is an empty list; leave it unset for the default device")
    return list(dict.fromkeys(s.devices))   # the same device twice would only load the models twice


def _bind_device(device: str) -> None:
    """Thread initializer of a slot: make its GPU the thread's current CUDA device. The engine places every
    tensor on settings.device explicitly; this covers the calls that use the current device (memory statistics,
    synchronize, empty_cache)."""
    try:
        import torch

        if torch.device(device).type == "cuda" and torch.device(device).index is not None and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device))
    except Exception:  # noqa: BLE001 - a failure here would break the thread pool; the engine will report it instead
        logger.exception("could not bind a worker thread to %s", device)


DEVICES = resolve_devices(settings)
# One engine per device, each with the settings of its own device.
engines: dict[str, Krea2Engine] = {d: Krea2Engine(settings.model_copy(update={"device": d})) for d in DEVICES}
engine = engines[DEVICES[0]]   # the first one also serves the listings and the prompt enhancer
# One persistent thread per GPU: CUDA's current device and the cuDNN attention setup are per thread.
slots = [
    Slot(d, ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu{i}", initializer=_bind_device, initargs=(d,)))
    for i, d in enumerate(DEVICES)
]

_load_error: str | None = None
scheduler: Scheduler | None = None   # created at startup (lifespan)
inputs: InputFiles | None = None


def _job_finished(job_id: str) -> None:
    """The scheduler's hook for a job that ended. The pictures stay for `inputs_keep_days` (see _clean_inputs)."""
    if settings.inputs_keep_days == 0:
        assert inputs is not None
        inputs.remove(job_id)


def _clean_inputs() -> int:
    """Deletes the folders of jobs queued more than `inputs_keep_days` ago. A job that is still waiting or running
    keeps its pictures however old it is: it needs them. Returns how many folders were deleted."""
    if inputs is None or scheduler is None:
        return 0
    removed = 0
    for job_id in inputs.folders_older_than(settings.inputs_keep_days * 86400):
        job = scheduler.store.get(job_id)
        if job is None or job.finished:   # no job at all: a leftover of a crash between saving and queueing
            inputs.remove(job_id)
            removed += 1
    return removed


async def _clean_inputs_forever(every: float = 3600.0) -> None:
    while True:
        try:
            removed = await asyncio.to_thread(_clean_inputs)
            if removed:
                logger.info("deleted the input pictures of %d old job(s)", removed)
        except Exception:  # noqa: BLE001 - housekeeping must never take the server down
            logger.exception("cleaning old input pictures failed")
        await asyncio.sleep(every)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, inputs, _config_lock
    _config_lock = asyncio.Lock()
    store = JobStore(settings.database_path)
    inputs = InputFiles(settings.inputs_dir)
    scheduler = Scheduler(
        store, slots, _run_job, settings.queue_max_pending, settings.queue_max_per_user, on_finished=_job_finished,
    )
    # Jobs are accepted at once (they wait in the queue), but none starts before the models are loaded.
    scheduler.start(paused=True)
    if not settings.api_keys:
        logger.warning("access control is OFF (no MY_DIFFUSION_API_KEYS): every caller is an admin")
    if len(slots) > 1:
        logger.info("%d GPU slots: %s; every slot holds its own copy of the models", len(slots), ", ".join(DEVICES))

    async def load() -> None:
        global _load_error
        try:
            await asyncio.gather(*(_on_slot(slot, engines[slot.name].load) for slot in slots))
        except Exception as exc:  # noqa: BLE001 - we show the reason in /health instead of silently crashing the server
            logger.exception("could not load the models")
            _load_error = _error_text(exc)
        else:
            scheduler.resume()

    task = asyncio.create_task(load())
    cleaner = asyncio.create_task(_clean_inputs_forever())
    yield
    task.cancel()
    cleaner.cancel()
    await scheduler.stop()
    store.close()
    scheduler = None


# ---------------------------------------------------------------------- the API's own documentation
_DESCRIPTION = """
Local backend for **Krea 2**: text-to-image, image-to-image, inpaint, reference editing, outpainting.

## The flow

1. `POST /generate` - queue a generation; the answer (`202`) is the job with its `job_id`.
2. `GET /result/{job_id}` - poll: `queued` (with `position`), `running` (with `step` / `total_steps`),
   then `done`, `failed` (with `error`) or `cancelled`.
3. `GET /result/{job_id}/image` - the PNG, once the job is `done`.
4. `POST /cancel/{job_id}` - cancel a waiting job at once, a running one at its next step.

Pictures in requests are base64 PNG/JPEG; a `data:image/...;base64,` prefix is accepted.

## Access

When the server has API keys configured, every endpoint except `/health` and `/resolutions` needs
**`Authorization: Bearer <key>`** (the *Authorize* button on this page). Service clients such as a Telegram bot
also send **`X-User-Id`**: the end user they act for. A job is visible only to its owner and to admin clients;
for anyone else it answers `404`. Without configured keys every caller is an admin and no key is needed.
"""

_TAGS = [
    {"name": "jobs", "description": "Queue a generation, follow it, fetch the picture, cancel it"},
    {"name": "models", "description": "Checkpoints, LoRAs and the prompt enhancer"},
    {"name": "service", "description": "Health and reference tables; no key needed"},
    {"name": "config", "description": "Server settings at run time; admins only"},
]


def _app_version() -> str:
    try:
        return package_version("my-diffusion")
    except PackageNotFoundError:   # running from a source tree that is not installed
        return "0.0.0"


def docs_urls(enabled: bool) -> dict:
    """FastAPI's documentation addresses, or None for each when the documentation is turned off."""
    if enabled:
        return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}
    return {"docs_url": None, "redoc_url": None, "openapi_url": None}


app = FastAPI(
    title="my-diffusion",
    version=_app_version(),
    description=_DESCRIPTION,
    openapi_tags=_TAGS,
    lifespan=lifespan,
    **docs_urls(settings.api_docs),
)
settings.output_dir.mkdir(parents=True, exist_ok=True)
# Note: there is no public /outputs folder any more. Pictures are served per job, after the access check.

# Scalar: a third documentation page next to Swagger UI and ReDoc. The script comes from a CDN, pinned to an
# exact version, with its SHA-256 checked by the browser (integrity): a tampered file on the CDN is refused
# instead of running on a page where an API key is typed in. To update: pick a release at least two weeks old
# and take the hash of dist/browser/standalone.js from https://data.jsdelivr.com/v1/packages/npm/@scalar/api-reference@<version>
SCALAR_VERSION = "1.76.0"
SCALAR_SRI = "sha256-3LAnFSvyC/L23NZuj5+wc8Gp4nTPC5S5OafZmriUcGA="
_SCALAR_PAGE = """<!doctype html>
<html>
<head>
  <title>my-diffusion API</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
  <div id="app"></div>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@{version}/dist/browser/standalone.js"
          integrity="{sri}" crossorigin="anonymous"></script>
  <script>
    Scalar.createApiReference('#app', {{
      url: '{openapi}',
      // A local server: none of Scalar's cloud features. "Ask AI" and the SDK / MCP generators would send the
      // API description to Scalar's servers, and telemetry is on by default.
      showDeveloperTools: 'never',
      agent: {{ disabled: true }},
      mcp: {{ disabled: true }},
      telemetry: false,
      persistAuth: false,   // the API key typed into the page is not kept in the browser's storage
    }})
  </script>
</body>
</html>
"""

if settings.api_docs:
    @app.get("/scalar", include_in_schema=False)
    async def scalar() -> HTMLResponse:
        return HTMLResponse(_SCALAR_PAGE.format(version=SCALAR_VERSION, sri=SCALAR_SRI, openapi=app.openapi_url))


class ErrorDetail(BaseModel):
    detail: str


def _errors(*codes: int) -> dict:
    """The error answers of an endpoint, for the documentation. Every error body is {"detail": "..."}."""
    meaning = {
        400: "Malformed request (or, for a service client, a missing / bad X-User-Id)",
        401: "API key missing or unknown (only when keys are configured)",
        403: "Admin rights are required",
        404: "No such job - or it belongs to someone else",
        409: "Conflict with the job's or the queue's state",
        422: "The request does not validate",
        429: "The queue, or this user's share of it, is full",
        503: "The job queue or the models are not ready",
    }
    return {code: {"model": ErrorDetail, "description": meaning[code]} for code in codes}


# ---------------------------------------------------------------------- access
# Declared as OpenAPI security schemes (not plain headers), so the documentation pages show an "Authorize"
# button that sends them with every call. auto_error=False: without configured keys none is needed.
_bearer = HTTPBearer(
    auto_error=False, scheme_name="ApiKey",
    description="The client's API key from MY_DIFFUSION_API_KEYS. Not needed when no keys are configured.",
)
_user_header = APIKeyHeader(
    name="X-User-Id", auto_error=False, scheme_name="UserId",
    description="Service clients only (e.g. the Telegram bot): the end user the request is made for. "
    "1-64 characters: letters, digits, '_', '.', '@', '-'. Optional for admin clients.",
)


def _caller(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)] = None,
    x_user_id: Annotated[str | None, Security(_user_header)] = None,
) -> Caller:
    authorization = None if credentials is None else f"{credentials.scheme} {credentials.credentials}"
    try:
        return resolve_caller(settings, authorization, x_user_id)
    except AuthError as exc:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        raise HTTPException(exc.status, str(exc), headers=headers) from None


def _admin(caller: Annotated[Caller, Depends(_caller)]) -> Caller:
    if not caller.is_admin:
        raise HTTPException(403, "admin rights are required")
    return caller


CallerDep = Annotated[Caller, Depends(_caller)]
AdminDep = Annotated[Caller, Depends(_admin)]


# ---------------------------------------------------------------------- helpers
def _models_loaded() -> bool:
    return all(e.loaded for e in engines.values())


def _scheduler() -> Scheduler:
    if scheduler is None or not scheduler.alive:
        raise HTTPException(503, "the job queue is not running; see the server log")
    return scheduler


def _require_loaded() -> None:
    if _load_error:
        raise HTTPException(500, f"Models failed to load: {_load_error}")
    if not _models_loaded():
        raise HTTPException(503, "Models are still loading, try again in a few seconds")


async def _on_slot(slot: Slot, func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(slot.executor, functools.partial(func, *args, **kwargs))


def _http_error(exc: BaseException) -> BaseException:
    """Engine error -> the HTTP exception that describes it. Anything unknown is returned unchanged (a 500)."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, str(exc))
    if isinstance(exc, (ValueError, NotImplementedError)):
        return HTTPException(400, str(exc))
    if isinstance(exc, ImportError):   # an optional package is missing on this machine: a server problem, not a bad request
        return HTTPException(503, str(exc))
    return exc


async def _run(slot: Slot, func, *args, **kwargs):
    """Calls the engine directly on a slot's thread (not through the queue), translating its errors into HTTP codes."""
    try:
        return await _on_slot(slot, func, *args, **kwargs)
    except Exception as exc:
        mapped = _http_error(exc)
        if mapped is exc:
            raise
        raise mapped from exc


# ---------------------------------------------------------------------- service endpoints
@app.get("/health", tags=["service"])
async def health() -> dict:
    """Model loading state (loading / ok / error), the checkpoint, and the queue: waiting jobs and the GPU slots."""
    status = "error" if _load_error else "ok" if _models_loaded() else "loading"
    queue = None
    if scheduler is not None:
        busy = scheduler.busy_slots()
        queue = {
            "waiting": scheduler.store.count(("queued",)),
            "slots": [{"device": slot.name, "job": busy.get(slot.name)} for slot in scheduler.slots],
            "scheduler_alive": scheduler.alive,
            "paused": scheduler.paused,
        }
    return {
        "status": status, "error": _load_error, "checkpoint": engine.checkpoint_name,
        "kind": engine.kind if engine.loaded else None,
        "matmul": engine.matmul, "native_layers": engine.native_layers, "attention": settings.attention,
        "access_control": bool(settings.api_keys),
        "queue": queue,
    }


@app.get("/resolutions", tags=["service"])
async def resolutions() -> dict:
    """Table: resolution -> format -> (width, height)."""
    return {
        tier: {ratio: {"width": d.width, "height": d.height} for ratio, d in ratios.items()}
        for tier, ratios in all_combinations().items()
    }


@app.get("/checkpoints", tags=["models"], responses=_errors(401))
async def checkpoints(caller: CallerDep) -> dict:
    """The current checkpoint, the available files and the default steps / guidance per checkpoint kind."""
    return {"current": engine.checkpoint_name, "available": engine.list_files("checkpoints"), "defaults": KIND_DEFAULTS}


@app.post("/checkpoints/load", tags=["models"], responses=_errors(401, 403, 404, 409, 503))
async def load_checkpoint(request: LoadCheckpointRequest, caller: AdminDep) -> dict:
    """Switch the checkpoint on every GPU without restarting the server. Admins only. Refused while jobs are
    waiting or running: a queued job must run on the checkpoint it was submitted for."""
    _require_loaded()
    if not _scheduler().is_idle:
        raise HTTPException(409, "jobs are waiting or running; cancel them or wait until the queue is empty, then switch")

    def switch(target: Krea2Engine) -> None:
        with target._lock:
            target.load_checkpoint(request.name)

    await asyncio.gather(*(_run(slot, switch, engines[slot.name]) for slot in slots))
    return {"current": engine.checkpoint_name, "kind": engine.kind}


@app.get("/loras", tags=["models"], responses=_errors(401))
async def loras(caller: CallerDep) -> list[str]:
    """LoRA files in the LoRA folder, as names to use in a request's `loras` list."""
    return engine.list_files("loras")


@app.post("/enhance-prompt", response_model=EnhancePromptResponse, tags=["models"], responses=_errors(400, 401, 503))
async def enhance_prompt(request: EnhancePromptRequest, caller: CallerDep) -> EnhancePromptResponse:
    """Not queued: runs on the first GPU's thread between jobs (a few seconds)."""
    _require_loaded()
    enhanced = await _run(slots[0], engine.enhance_prompt, request.prompt, request.seed)
    return EnhancePromptResponse(prompt=request.prompt, enhanced_prompt=enhanced)


# ---------------------------------------------------------------------- generation jobs
_DATA_URI_PREFIX = re.compile(r"^data:[^;,]*;base64,", re.IGNORECASE)
_PICTURE_FIELDS = ("init_image", "mask", "references")


def _decode_upload(value: str, field: str) -> bytes:
    """Base64 (optionally as a data URI) -> the image file's bytes, checked to be a readable picture.
    Errors become 400s that name the field. The bytes are kept as sent (compressed); the picture is decoded
    again only when the job runs, so a waiting job does not hold a decoded picture in RAM."""
    payload = _DATA_URI_PREFIX.sub("", value.strip())
    try:
        data = base64.b64decode(payload, validate=True)  # binascii.Error is a ValueError
    except ValueError as exc:
        raise HTTPException(400, f"{field} is not valid base64") from exc
    try:
        imaging.load_rgb(data)
    except ValueError as exc:
        raise HTTPException(400, f"{field}: {exc}") from exc
    return data


def _check_generate(request: GenerateRequest) -> None:
    """Checks that need neither the pictures decoded nor the models loaded: answered with 400 at once."""
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


def _run_job(slot: Slot, job: JobRecord, cancel_event, progress) -> dict:
    """The scheduler's runner: rebuilds the request from the store and calls this slot's engine. Runs in the
    slot's thread. Re-validating the stored params fills in fields added to the schema after the job was queued."""
    assert inputs is not None
    request = GenerateRequest.model_validate(job.params)

    def picture(name: str):
        return imaging.load_rgb(inputs.load(job.id, name))

    names = job.inputs
    outpaint = request.outpaint
    return engines[slot.name].generate(
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
        init_image=picture(names["init_image"]) if "init_image" in names else None,
        strength=request.strength,
        mask=picture(names["mask"]) if "mask" in names else None,
        feather=request.feather,
        references=[picture(name) for name in names.get("references", [])],
        reference_mode=request.reference_mode,
        outpaint=None if outpaint is None else (outpaint.top, outpaint.bottom, outpaint.left, outpaint.right),
        remove_background=request.remove_background,
        keep_original=request.keep_original,
        cancel_event=cancel_event,
        on_progress=progress,
        output_name=f"{job.id}.png",
    )


def _input_files(job: JobRecord) -> dict[str, str]:
    """The public name of each stored input picture -> its file name: init, mask, ref1, ref2 ..."""
    names = job.inputs
    files: dict[str, str] = {}
    if "init_image" in names:
        files["init"] = names["init_image"]
    if "mask" in names:
        files["mask"] = names["mask"]
    for i, name in enumerate(names.get("references", []), start=1):
        files[f"ref{i}"] = name
    return files


def _input_urls(job: JobRecord) -> dict[str, str]:
    return {name: f"/result/{job.id}/inputs/{name}" for name in _input_files(job)}


def _job_info(job: JobRecord, positions: dict[str, int] | None = None) -> JobInfo:
    s = _scheduler()
    if job.status == "queued":
        position = positions.get(job.id) if positions is not None else s.position(job.id)
    else:
        position = None
    progress = s.progress(job.id) if job.status == "running" else None
    return JobInfo(
        job_id=job.id,
        status=job.status,
        cancel_requested=job.cancel_requested,
        owner=job.owner,
        position=position,
        step=None if progress is None else progress[0],
        total_steps=None if progress is None else progress[1],
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        slot=job.slot,
        params=job.params,
        result=None if job.result is None else GenerationResult(**job.result),
        image_url=f"/result/{job.id}/image" if job.status == "done" else None,
        inputs=_input_urls(job),
        error=job.error,
    )


def _visible_job(job_id: str, caller: Caller) -> JobRecord:
    try:
        job = _scheduler().get(job_id)
    except JobNotFound:
        job = None
    if job is None or not caller.can_see(job.owner):   # someone else's job looks exactly like a missing one
        raise HTTPException(404, f"no job with id {job_id!r}")
    return job


@app.post("/generate", response_model=JobInfo, status_code=202, tags=["jobs"], responses=_errors(400, 401, 429, 503))
async def generate(request: GenerateRequest, caller: CallerDep) -> JobInfo:
    """Puts a generation in the queue and answers at once with the job (status 202). Poll GET /result/{id};
    when the status is "done", fetch the picture from GET /result/{id}/image.

    Modes: text-to-image; with init_image - image-to-image (the picture is noised to `strength` and denoised
    again); with init_image + mask - inpaint (only the white part of the mask is regenerated). With `references`
    (1 or 2 pictures) - reference editing, not combinable with init_image or mask. Pictures are base64 PNG/JPEG.

    The request is checked before it is queued: a malformed one gets 400 at once. Problems only the engine can
    see (an unknown LoRA, a size that does not fit the GPU) make the job "failed", with the reason in `error`.
    429 when the queue, or this user's share of it, is full."""
    _check_generate(request)
    pictures: dict[str, bytes | list[bytes]] = {}
    if request.init_image is not None:
        pictures["init_image"] = _decode_upload(request.init_image, "init_image")
    if request.mask is not None:
        pictures["mask"] = _decode_upload(request.mask, "mask")
    if request.references:
        pictures["references"] = [_decode_upload(value, f"references[{i}]") for i, value in enumerate(request.references)]
    if _load_error:
        raise HTTPException(500, f"Models failed to load: {_load_error}")
    s = _scheduler()
    assert inputs is not None
    params = request.model_dump(exclude=set(_PICTURE_FIELDS))
    job_id = uuid.uuid4().hex
    names = inputs.save(job_id, pictures) if pictures else {}
    try:
        job = s.submit(caller.owner, params, names, exempt=caller.is_admin, job_id=job_id)
    except QueueFull as exc:
        inputs.remove(job_id)
        raise HTTPException(429, str(exc)) from exc
    except BaseException:
        inputs.remove(job_id)
        raise
    return _job_info(job)


@app.get("/result/{job_id}", response_model=JobInfo, tags=["jobs"], responses=_errors(401, 404))
async def result(job_id: str, caller: CallerDep) -> JobInfo:
    """The job's status: queued (with its position), running (with progress), done (with image_url), failed
    (with error) or cancelled."""
    return _job_info(_visible_job(job_id, caller))


EVENT_KEEPALIVE_SECONDS = 15.0   # a comment line this often keeps proxies and idle-timeouts from closing the stream


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.get(
    "/result/{job_id}/events",
    response_class=StreamingResponse,
    tags=["jobs"],
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent events"}, **_errors(401, 404)},
)
async def result_events(job_id: str, caller: CallerDep) -> StreamingResponse:
    """Server-sent events instead of polling GET /result/{id}.

    The first event is the job as it is now; after that, a new event whenever something changes: the job's
    position in the queue, its progress step, `cancel_requested`, or its end. Every event's data is the same JSON as
    GET /result/{id}. Events named `update` leave the stream open; the last one is named after the final status
    (`done`, `failed` or `cancelled`) and the server then closes the stream. A `: keepalive` comment line is sent
    every 15 s of silence.

    Send the key like for every other call (`Authorization: Bearer ...`). The browser's `EventSource` cannot set
    headers: use `fetch` and read the response body as a stream. A client that reconnects simply gets the current
    state first."""
    _visible_job(job_id, caller)
    s = _scheduler()

    async def stream():
        wake = s.subscribe()
        last = None
        try:
            while True:
                wake.clear()   # before looking: a change that lands while we look wakes us again
                job = s.get(job_id)
                info = _job_info(job)
                data = info.model_dump_json()
                if data != last:
                    last = data
                    yield _sse(job.status if job.finished else "update", data)
                if job.finished or s.stopping:
                    return
                try:
                    await asyncio.wait_for(wake.wait(), EVENT_KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            s.unsubscribe(wake)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}   # no buffering in nginx-like proxies
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


_INPUT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


@app.get(
    "/result/{job_id}/inputs/{name}",
    response_class=FileResponse,
    tags=["jobs"],
    responses={200: {"content": {"image/*": {}}, "description": "The picture as it was sent"}, **_errors(401, 404)},
)
async def result_input(job_id: str, name: str, caller: CallerDep) -> FileResponse:
    """A picture that was sent with the job, exactly as sent: `init` (init_image), `mask`, `ref1`, `ref2`
    (references, counted from 1). `GET /result/{id}` lists them in `inputs`. Pictures are kept for a limited time
    after the job (MY_DIFFUSION_INPUTS_KEEP_DAYS, 7 days by default); after that, or if the job had no such picture,
    the answer is 404."""
    job = _visible_job(job_id, caller)
    stored = _input_files(job).get(name)
    assert inputs is not None
    path = None if stored is None else inputs.path(job.id, stored)
    if path is None or not path.is_file():
        raise HTTPException(404, f"job {job_id} has no input picture {name!r} (it may have expired)")
    return FileResponse(path, media_type=_INPUT_TYPES.get(path.suffix[1:], "application/octet-stream"))


@app.get(
    "/result/{job_id}/image",
    response_class=FileResponse,
    tags=["jobs"],
    responses={200: {"content": {"image/png": {}}, "description": "The PNG"}, **_errors(401, 404, 409)},
)
async def result_image(job_id: str, caller: CallerDep) -> FileResponse:
    """The PNG of a finished job. 409 while it is not done."""
    job = _visible_job(job_id, caller)
    if job.status != "done" or not job.image_name:
        raise HTTPException(409, f"the job is {job.status}; the picture exists only once it is done")
    path = settings.output_dir / job.image_name
    if not path.is_file():
        raise HTTPException(404, "the picture file no longer exists on the server")
    return FileResponse(path, media_type="image/png", filename=job.image_name)


@app.post("/cancel/{job_id}", response_model=JobInfo, tags=["jobs"], responses=_errors(401, 404, 409))
async def cancel(job_id: str, caller: CallerDep) -> JobInfo:
    """A waiting job is cancelled at once. A running one gets cancel_requested=true and stops at its next step
    (then its status becomes "cancelled"; if the picture was already finished, it stays "done").
    409 for a job that has already finished."""
    _visible_job(job_id, caller)
    try:
        job = _scheduler().cancel(job_id)
    except JobNotCancellable:
        raise HTTPException(409, f"job {job_id} has already finished") from None
    return _job_info(job)


@app.get("/jobs", response_model=list[JobInfo], tags=["jobs"], responses=_errors(401, 403))
async def list_jobs(
    caller: CallerDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    owner: Annotated[str | None, Query(description="Admins only: one owner's jobs ('telegram:123')")] = None,
) -> list[JobInfo]:
    """Your jobs, newest first. Admins see everybody's, or one owner's with ?owner=."""
    if owner is not None and not caller.is_admin:
        raise HTTPException(403, "only admins can list other users' jobs")
    s = _scheduler()
    positions = {job.id: i for i, job in enumerate(s.order(), start=1)}
    who = owner if caller.is_admin else caller.owner
    return [_job_info(job, positions) for job in s.store.list(who, limit)]


# ---------------------------------------------------------------------- settings at run time
# How a change is applied (the rules themselves are in runtime_config.py and in the field metadata of config.py):
#   1. the whole request is checked first; anything wrong is a 422 and nothing has changed;
#   2. settings the engines read are switched on every GPU slot's own thread, under the engine's lock, so
#      between two generations (Krea2Engine.apply_settings). A running generation keeps the settings it
#      started with, and the call answers once the change is in effect;
#   3. then the shared `settings` object and the scheduler's limits are updated and the override file is saved.
# If a slot cannot apply the change, or the file cannot be written, everything is put back.

# Settings only this module reads: they need no trip to the GPU threads. Purely an optimisation - a setting
# missing here is still applied correctly, it just waits for the running generation first.
_API_ONLY = {"queue_max_pending", "queue_max_per_user", "inputs_keep_days"}


def _describe_config() -> dict:
    return runtime_config.describe(settings, _overrides, _config_problems, runtime_config.read_baseline())


def _error_text(exc: BaseException) -> str:
    """"ValueError: the reason", or the type alone when the exception carries no text (a MemoryError)."""
    text = str(exc).strip().rstrip(".")
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _sync_load_state(error: BaseException | None = None) -> None:
    """After the engines were touched: bring /health's error and the scheduler in line with whether the models
    are there. Models that failed to load at start and load now clear the error and start the queue; models
    lost by a failed reload set it and pause the queue."""
    global _load_error
    if _models_loaded():
        if _load_error is not None:
            _load_error = None
            if scheduler is not None:
                scheduler.resume()
    elif error is not None or _load_error is None:
        _load_error = _error_text(error) if error is not None else "the models are not loaded"
        if scheduler is not None:
            scheduler.pause()


async def _switch_engines(per_slot: dict[str, Settings]) -> list:
    """apply_settings on the thread of every slot named in per_slot, all at once; one result (the reload plan)
    or exception per slot, in the order of `slots`."""
    return await asyncio.gather(
        *(_on_slot(slot, engines[slot.name].apply_settings, per_slot[slot.name]) for slot in slots if slot.name in per_slot),
        return_exceptions=True,
    )


async def _change_config(change: ConfigChange, caller: Caller):
    assert _config_lock is not None
    async with _config_lock:
        baseline = runtime_config.read_baseline()
        try:
            plan = runtime_config.plan_change(
                settings, _overrides, change.set, change.reset, baseline.settings, runtime_config.allowed_roots(settings),
            )
        except runtime_config.ChangeRefused as exc:
            return JSONResponse({"detail": f"the settings were not changed: {exc}", "errors": exc.errors}, status_code=422)

        # Every engine has its own copy of the settings, with its own device.
        before = {slot.name: engines[slot.name].settings for slot in slots}
        after = {name: plan.running.model_copy(update={"device": old.device}) for name, old in before.items()}
        touch_engines = bool(set(plan.applied) - _API_ONLY)
        reload = "none"
        if touch_engines:
            reload = runtime_config.heaviest_reload(engines[name].reload_plan(new) for name, new in after.items())
        # Without models no job can run, so a reload has nothing to wait for (and may be what repairs the server).
        if reload != "none" and _models_loaded() and not _scheduler().is_idle:
            raise HTTPException(
                409, "this change reloads the models, and jobs are waiting or running; cancel them or wait until "
                "the queue is empty",
            )

        def answer() -> dict:
            return {
                "applied": plan.applied, "restart_required": plan.restart_required, "unchanged": plan.unchanged,
                "reloaded": reload, "warnings": plan.warnings, "dry_run": change.dry_run, "config": _describe_config(),
            }

        if change.dry_run:
            return answer()

        previous = {name: getattr(settings, name) for name in plan.applied}
        if touch_engines:
            results = await _switch_engines(after)
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                # The slots that did switch go back; a slot that failed has put its old settings back itself.
                switched = {slot.name for slot, result in zip(slots, results) if not isinstance(result, BaseException)}
                await _switch_engines({name: before[name] for name in switched})
                _sync_load_state(failures[0])
                logger.warning("config change by %s could not be applied: %s", caller.owner, _error_text(failures[0]))
                lost = "" if _models_loaded() else " The models are not loaded now: see /health."
                raise HTTPException(
                    400, f"the settings could not be applied and were left as they were: "
                    f"{_error_text(failures[0])}.{lost}",
                )
            reload = runtime_config.heaviest_reload(results)
        try:
            for name in plan.applied:
                setattr(settings, name, getattr(plan.running, name))
            if scheduler is not None:
                scheduler.set_limits(settings.queue_max_pending, settings.queue_max_per_user)
            write_overrides(settings.settings_file, plan.overrides)
        except Exception as exc:
            for name, value in previous.items():
                setattr(settings, name, value)
            if scheduler is not None:
                scheduler.set_limits(settings.queue_max_pending, settings.queue_max_per_user)
            if touch_engines:
                await _switch_engines(before)
                _sync_load_state()
            logger.exception("config change by %s could not be saved", caller.owner)
            raise HTTPException(500, f"the settings could not be saved and were left as they were: {exc}") from exc
        _overrides.clear()
        _overrides.update(plan.overrides)
        if touch_engines:
            _sync_load_state()
        # No secret can appear here: secret settings are never changed through the API.
        logger.info(
            "config changed by %s: applied %s, for the next restart %s, models reloaded: %s",
            caller.owner, {name: plan.overrides.get(name, "<reset>") for name in plan.applied},
            {name: plan.overrides.get(name, "<reset>") for name in plan.restart_required}, reload,
        )
        return answer()


_CONFIG_ERRORS = {
    **_errors(400, 401, 403, 409),
    422: {"description": "A value does not validate; `errors` has a message per setting. Nothing was changed"},
}


@app.get("/config", tags=["config"], responses=_errors(401, 403))
async def get_config(caller: AdminDep) -> dict:
    """Every setting of the server with what a Settings page needs. Admins only.

    For each one: `value` (what the server runs with), `default`, `source` (`override` - saved through this API,
    `env`, `dotenv` or `default`), `type` with `choices` / `minimum` / `maximum` / `nullable`, `description`,
    `editable`, and `apply` - how a change takes effect: `live` (at once), `reload_models` (the models are read
    again; only with an empty queue) or `restart` (saved, used from the next start; `pending_restart` tells that
    the saved value differs from the running one). Secrets are never included: for `api_keys` only the client
    names are given, in `access_control`."""
    return _describe_config()


@app.patch("/config", tags=["config"], responses=_CONFIG_ERRORS)
async def patch_config(change: ConfigChange, caller: AdminDep):
    """Change settings. Admins only. All or nothing: the whole request is checked first (`422` with a message per
    setting in `errors`), and if applying fails, everything is put back (`400`).

    - `live` settings take effect at once. Those the engine reads are switched between two generations: while a
      generation runs, the call waits for it to end.
    - `reload_models` settings reload the checkpoint or all the models on every GPU. Refused with `409` while
      jobs are waiting or running.
    - `restart` settings are only saved; they take effect at the next start of the server.

    Changes are saved to the override file (not to `.env`) and survive a restart. `reset` removes a saved value,
    so the setting falls back to the environment or the default. `dry_run` only checks and reports.

    The answer: `applied` (in effect now), `restart_required` (saved, waiting for a restart), `unchanged`,
    `reloaded` (`none` / `checkpoint` / `all`), `warnings`, and `config` - the same as `GET /config`."""
    # Shielded: a client that hangs up must not stop the change half-way (the engines switched, nothing saved).
    return await asyncio.shield(asyncio.ensure_future(_change_config(change, caller)))


@app.get("/fs", tags=["config"], responses={**_errors(400, 401, 403), 404: {"model": ErrorDetail, "description": "No such folder"}})
async def browse_folders(
    caller: AdminDep,
    path: Annotated[str | None, Query(description="Absolute path of a folder; without it the allowed roots are listed")] = None,
) -> dict:
    """The subfolders of a folder on the server, for choosing the model folders. Admins only.

    Only folders are listed (`model_files` counts the `.safetensors` / `.gguf` files lying directly in `path`),
    and only inside the allowed roots: the model folders in use plus `MY_DIFFUSION_FS_ROOTS`. Paths are compared
    by their real location, so `..` and links cannot lead out of the roots (`403`). The same roots limit the
    folders that `PATCH /config` accepts."""
    try:
        return await asyncio.to_thread(runtime_config.list_directory, path, runtime_config.allowed_roots(settings))
    except runtime_config.FsError as exc:
        raise HTTPException(exc.status, str(exc)) from None
