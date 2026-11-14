# my-diffusion

A thin local backend for generating images with the **Krea 2** model. It runs its own native
engine in one process: no ComfyUI and no Wan2GP at runtime. It is exposed as an HTTP API and as a
command line.

## Features

- **Text-to-image** with frame resolution (480 / 720 / 1080) and aspect ratio
- **Image-to-image** and **inpaint** with a mask
- **Reference editing**: one or two reference pictures, scene or subjects mode
- **Outpainting**: extend a picture on any side, optionally keeping the original pixels
- **Background removal** for reference people and objects
- **LoRA** with per-request weights; the set can change without reloading the model
- **Prompt enhancer**: Qwen3-VL expands a short idea into a detailed prompt
- **Checkpoints** in mxfp8, fp8, int8, and GGUF q8_0 formats; switch them without a restart
- **Fast inference** on NVIDIA GPUs: quantized matrix multiplication, pinned memory, cuDNN attention
- **Jobs**: `POST /generate` returns a job id at once; a scheduler runs the queue on one or more GPUs, fairly between users; jobs can be cancelled while waiting or running and survive a restart (SQLite)
- **Access control**: API keys per client; users see only their own jobs and pictures, admins see all
- **HTTP API** with an OpenAPI schema and interactive docs (Swagger UI at `/docs`, Scalar at `/scalar`, ReDoc at `/redoc`), and a **CLI**

## Requirements

- Windows (the model-linking script is PowerShell; tested on Windows 11)
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with CUDA. Tested on an RTX 5080 Laptop with 16 GB VRAM (peak use about 14 GB at 1080p).
  The fast multiplication modes need a 50-series (Blackwell) GPU; other cards use `exact` mode.
- About 64 GB of RAM was used for testing. Pinned memory keeps about 19 GB of RAM in use; it can be
  turned off (see [configuration](guides/configuration.md)).
- Model files (see [Models](#models)); the checkpoint alone is about 13 GB.

## Installation

Clone the repository and install the dependencies (torch is taken from the PyTorch CUDA 12.8 index):

```bash
uv sync
```

Link the models into the project (see [Models](#models)):

```bash
pwsh scripts/link_models.ps1
```

## Running

Generate an image from the command line:

```bash
uv run my-diffusion-generate "a fox in the snow, golden hour" --resolution 720 --aspect-ratio 16:9
```

Start the HTTP server (interactive documentation at http://127.0.0.1:8000/docs or http://127.0.0.1:8000/scalar):

```bash
uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5
```

Optional settings go into a `.env` file: copy `.env.example` to `.env` and edit it.

## Models

The models are not downloaded or copied. The code only knows about the `models/` folder in the
project root, and that folder consists of links (junctions) to the real folders - they are created by
`scripts/link_models.ps1`. On another machine: fix the path in the script (the `-ModelsHub`
parameter) or simply put the files into the same structure.

| Link in the project | What should be there |
|---|---|
| `models/checkpoints/` | Krea 2 checkpoints: `.safetensors` or `.gguf` |
| `models/loras/` | LoRAs for Krea 2, subfolders allowed |
| `models/vae/` | `qwen_vae.safetensors` and `qwen_vae_config.json` |
| `models/text_encoders/Qwen3-VL-4B-Instruct/` | `*_quanto_bf16_int8.safetensors`, `config.json`, tokenizer files |
| `models/rembg/` | `u2net.onnx` (only for background removal) |

Reference editing also needs the edit LoRA `krea2_identity_edit_v1_2` in `models/loras`.

## Supported

| Area | Support |
|---|---|
| Resolutions | 480, 720, 1080 pixel budget; 16:9, 4:3, 3:2, 21:9, 1:1 and portrait variants ([details](guides/resolutions.md)) |
| Checkpoints | mxfp8, float8_e4m3fn, int8 tensorwise, bare fp8, GGUF q8_0; turbo and raw kinds ([details](guides/checkpoints.md)) |
| LoRA | Standard LoRA files with 7 key naming schemes; LoKr/LoHa not supported ([details](guides/loras.md)) |
| Modes | text-to-image, image-to-image, inpaint, reference editing, outpainting, background removal ([details](guides/image-modes.md)) |
| Interfaces | HTTP API (`/generate`, `/result/{id}`, `/cancel/{id}`, `/checkpoints/load`, ...) and CLI ([details](guides/api-and-cli.md)) |
| Hardware | NVIDIA CUDA GPUs; fast modes on 50-series only ([details](guides/performance.md)) |

Known limitations:

- Reference pictures cannot be combined with `init_image` or `mask`.
- Edits above 720 (one reference) or 480 (two) are slow rather than impossible: the network is streamed from RAM as needed. `MY_DIFFUSION_EDIT_LIMITS` (`strict` / `warn` / `off`, default `warn`) and `MY_DIFFUSION_EDIT_MAX_PIXELS` decide whether such jobs are accepted ([details](guides/image-modes.md)).
- A GPU runs one generation at a time; more GPUs run more jobs at once, but each holds its own copy of the models in RAM (multi-GPU is not tested yet). Run the server as a single process.
- Without `MY_DIFFUSION_API_KEYS` access control is off and every caller is an admin: keep such a server on `127.0.0.1`.
- The prompt enhancer supports only Qwen3-VL-4B.

## Documentation

- [Image modes](guides/image-modes.md): image-to-image, inpaint, reference editing, outpainting, background removal
- [API and CLI](guides/api-and-cli.md): endpoints, request fields, command-line flags, prompt enhancer
- [Configuration](guides/configuration.md): all `MY_DIFFUSION_*` variables
- [Checkpoints](guides/checkpoints.md): formats, turbo and raw, guidance
- [Resolutions](guides/resolutions.md): pixel budgets and aspect ratios
- [LoRA](guides/loras.md): using LoRA files
- [Jobs, queue and access](guides/job-queue.md): the generate / result / cancel flow, API keys, fairness, several GPUs, restarts
- [Performance](guides/performance.md): measurements, memory, attention kernel, multiplication modes
- [Architecture](guides/architecture.md): how the engine works, design decisions, rejected experiments
- [Development](guides/development.md): tests, verification scripts, what is not implemented yet
