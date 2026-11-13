# HTTP API and command line

The same engine is reachable two ways: an HTTP server and a CLI. Both accept the same parameters.

## HTTP server

```bash
uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5
```

The server documents itself. The same OpenAPI schema is shown three ways; pick the one you like:

| Address | What |
|---|---|
| http://127.0.0.1:8000/docs | Swagger UI: every endpoint with a *Try it out* form |
| http://127.0.0.1:8000/scalar | Scalar: the same, with ready-made request snippets (curl, Python, JavaScript ...) |
| http://127.0.0.1:8000/redoc | ReDoc: read-only, convenient for reading |
| http://127.0.0.1:8000/openapi.json | the schema itself, for generating a client (OpenAPI 3.1) |

When API keys are configured, press *Authorize* (Swagger) or pick the auth type (Scalar) and paste the key; a
service client's `X-User-Id` goes there too. The pages load their scripts from the jsdelivr CDN, so they need
internet access in the browser; Scalar is pinned to an exact version and checked by its hash, and its cloud
features (AI assistant, SDK generator, telemetry) are switched off. `MY_DIFFUSION_API_DOCS=false` removes all
four addresses.

| Endpoint | What it does |
|---|---|
| `GET /health` | `loading` / `ok` / `error`, the current checkpoint |
| `POST /generate` | queue a generation; answers at once (`202`) with the job id |
| `GET /result/{job_id}` | the job: status, position in the queue, progress, result or error |
| `GET /result/{job_id}/events` | the same job pushed as server-sent events, instead of polling ([details](job-queue.md#events-instead-of-polling)) |
| `GET /result/{job_id}/image` | the finished PNG |
| `POST /cancel/{job_id}` | cancel a waiting or running job |
| `GET /jobs` | your jobs (an admin sees everybody's) |

All endpoints except `/health` and `/resolutions` need an API key once keys are configured; see
[jobs, queue and access](job-queue.md).
| `POST /enhance-prompt` | only rewrite the prompt |
| `GET /resolutions` | table of sizes |
| `GET /checkpoints`, `POST /checkpoints/load` | list and switch the checkpoint without a restart |
| `GET /loras` | list of LoRAs |

Example request:

```json
{
  "prompt": "a fox in the snow, golden hour",
  "resolution": "720",
  "aspect_ratio": "16:9",
  "seed": 42,
  "loras": [{"name": "KreaAmateur_V2", "weight": 0.8}],
  "enhance_prompt": false
}
```

Other request fields (`init_image`, `mask`, `references`, `reference_mode`, `outpaint`,
`keep_original`, `remove_background`, `steps`, `guidance_scale`, `guidance_start`, `guidance_end`,
`matmul`, `attention` ...) are described in [image modes](image-modes.md), [checkpoints](checkpoints.md)
and [performance](performance.md).

## Command line

```bash
uv run my-diffusion-generate "a fox in the snow, golden hour" --resolution 720 --aspect-ratio 16:9
```

| Flag | Meaning |
|---|---|
| `--resolution {480,720,1080}` | size tier, default `720` |
| `--aspect-ratio` | one of the ratios in [resolutions](resolutions.md), or `auto` (with `--reference`, scene mode); default `16:9` |
| `--seed`, `--count` | seed; `--count N` makes N images, the seed grows by 1 each time |
| `--steps`, `--guidance-scale` | override the defaults of the checkpoint kind |
| `--guidance-start`, `--guidance-end` | CFG interval as fractions of the steps (see [checkpoints](checkpoints.md)) |
| `--shift` | schedule shift `mu`; default is automatic |
| `--lora NAME[:WEIGHT]` | repeatable, see [LoRA](loras.md) |
| `--enhance` | expand the prompt with Qwen first |
| `--checkpoint` | file in `models/checkpoints` |
| `--matmul`, `--attention` | override the modes from the settings (see [performance](performance.md)) |
| `--init-image`, `--strength`, `--mask`, `--feather` | image-to-image and inpaint |
| `--reference` (up to 2), `--reference-mode`, `--outpaint T B L R`, `--keep-original`, `--remove-background` | reference editing; see [image modes](image-modes.md) |

Full help: `uv run my-diffusion-generate --help`.

## Prompt enhancer

`"enhance_prompt": true` in the request or `--enhance` in the CLI: Qwen3-VL expands a short idea into
a detailed description in English, and the generation then runs on that description. It is the same
model that serves as the text encoder, so it takes no extra memory. Separately, without generating:
`POST /enhance-prompt`.
