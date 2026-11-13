# HTTP API and command line

The same engine is reachable two ways: an HTTP server and a CLI. Both accept the same parameters.

## HTTP server

```bash
uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000
```

Interactive documentation is served at http://127.0.0.1:8000/docs.

| Endpoint | What it does |
|---|---|
| `GET /health` | `loading` / `ok` / `error`, the current checkpoint |
| `POST /generate` | generation; the response is the path to a PNG, the parameters and timings |
| `GET /outputs/<file>` | fetch a finished image |
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
