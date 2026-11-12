# my-diffusion

A thin local backend for generating images with the **Krea 2** model (text-to-image).
Its own native engine: no ComfyUI and no Wan2GP at runtime, the model runs inside our
process. Features: frame resolution and aspect ratio, LoRA with weights, a prompt enhancer,
all formats of quantized checkpoints, an HTTP API and a CLI.

## Quick start

```bash
uv sync
```

```bash
pwsh scripts/link_models.ps1
```

```bash
uv run my-diffusion-generate "a fox in the snow, golden hour" --resolution 720 --aspect-ratio 16:9
```

HTTP server (endpoint documentation at http://127.0.0.1:8000/docs):

```bash
uv run uvicorn my_diffusion.api:app --host 127.0.0.1 --port 8000
```

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

## How it works

The path of one request: text -> Qwen -> noise -> network x N steps -> VAE -> PNG.
Every module starts with an explanation of what it does and why.

| File | Role |
|---|---|
| `text_encoder.py` | Qwen3-VL-4B. Turns the prompt into hidden states of 12 layers (this is what Krea 2 "reads"). It is also the prompt enhancer. |
| `transformer.py` | The Krea 2 network itself: 28 blocks, text and image in one sequence. A port from ComfyUI. |
| `weights.py` | Reading checkpoints and the `QLinear` layer: the weights stay on the GPU in compressed form (1 byte per number) and are unpacked for the duration of a single multiplication. |
| `sampler.py` | The noise schedule and Euler steps: from noise to latent. |
| `vae.py` | Latent -> pixels. The only thing taken from `diffusers`. |
| `lora.py` | Parsing LoRA files (7 key naming schemes) and attaching them to the layers. |
| `memory.py` | One large model on the GPU at a time; switching without unnecessary copies. |
| `pinned.py` | Pinned RAM: the weights move from it to the GPU in fractions of a second. |
| `engine.py` | Combines everything into `generate()`. Text cache, LoRA cache, timing measurements. |
| `resolutions.py` | Resolution + aspect ratio -> width and height. |
| `api.py`, `schemas.py`, `cli.py` | HTTP and the command line on top of the engine. |

### Why earlier attempts produced noise

Before this engine, we tried `diffusers.Krea2Pipeline` and a port of the network from ComfyUI -
both produced noise. The network itself is not complicated; it is easy to get something wrong around it. What has to be
done right (and what is now verified by scripts):

1. **Weight format.** `krea2_turbo_mxfp8` is not an ordinary fp8: every 32 numbers have their own multiplier
   2^(e-127), and the table of exponents is also rearranged in 128x4 tiles. Our unpacking is checked
   bit for bit against the ComfyUI library (`scripts/check_weights.py`).
2. **The schedule shift** sits in the exponent: `mu=1.15` is a shift by a factor of 3.16.
3. **The noise level** is fed to the network as a number 0..1, not 0..1000.
4. **Latent normalization** per channel before the VAE: `z = latent * std + mean`.
5. **Text**: 12 layers of Qwen (`hidden_states[2,5,...,35]`); the 34-token service prefix
   is cut off after the forward pass, not before.

## Resolutions and formats

The resolution sets a **pixel budget** - the area of a 16:9 frame at that height. The other formats
get the same area, so the time and memory are the same for all formats of one
resolution. The sides are multiples of 16.

| | 480 | 720 | 1080 |
|---|---|---|---|
| 16:9 | 848x480 | 1280x720 | 1920x1088 |
| 4:3 | 736x560 | 1104x832 | 1664x1248 |
| 3:2 | 784x528 | 1168x784 | 1760x1168 |
| 21:9 | 976x416 | 1472x624 | 2192x944 |
| 1:1 | 640x640 | 960x960 | 1440x1440 |

Portrait formats (9:16, 3:4, 2:3, 9:21) use the same numbers with the sides swapped.
To check the table: `uv run python -m my_diffusion.resolutions`.

## Checkpoints

All formats found in `models/checkpoints` are supported:

| Format | How it is stored | Example |
|---|---|---|
| `mxfp8` | fp8 + a 2^e multiplier per block of 32 numbers | `krea2_turbo_mxfp8.safetensors` |
| `float8_e4m3fn` | fp8 + one multiplier per layer | `finepornV4INT8NVFP4BF16_v4.safetensors` |
| `int8_tensorwise` | int8 + a multiplier per row | `intorealismKrea2_v20.safetensors` |
| bare fp8 | fp8 without multipliers | `dafK2T_v1.safetensors` |
| `q8_0` (GGUF) | int8 + a float16 multiplier per block of 32 numbers | `Wild_Krea-2-turbo_NSFW-Q8_0.gguf` |

The loader is strict: an unknown format or an "extra" tensor is an error, not silent garbage.

**Turbo or raw.** The kind of checkpoint cannot be determined from the weights, and the settings depend on it:
turbo - 8 steps without CFG, raw - 28 steps and CFG 4.5. By default the kind is taken from the file name
(`raw` in the name -> raw, otherwise turbo); it can be overridden with the `MY_DIFFUSION_CHECKPOINT_KIND`
variable or with the `steps` / `guidance_scale` parameters in the request.

`guidance_scale` here is the usual CFG, where 1.0 means "off". In Wan2GP the scale is shifted
by one: their `guidance 3.5` is our `4.5`.

## LoRA

```json
"loras": [{"name": "KreaAmateur_V2", "weight": 0.8}, {"name": "sliders/zoom_krea2_loraholic", "weight": -1.0}]
```

`name` is the path relative to `models/loras`; the extension can be omitted. `weight` is the strength:
1.0 as trained, a negative value reverses the effect. The LoRA set and its weights can be changed between
requests without reloading the model: a LoRA does not get merged into the weights, it is computed alongside
as a small addition.

To check which files are read: `uv run python scripts/check_loras.py`.
Not supported: LoKr/LoHa and ControlNet-like files (the loader says so directly).

## Prompt enhancer

`"enhance_prompt": true` in the request or `--enhance` in the CLI: Qwen3-VL expands a short
idea into a detailed description in English, and the generation then runs on that description. It is the same model
that serves as the text encoder, so it takes no extra memory. Separately, without
generating: `POST /enhance-prompt`.

## API

| Endpoint | What it does |
|---|---|
| `GET /health` | `loading` / `ok` / `error`, the current checkpoint |
| `POST /generate` | generation; the response is the path to a PNG, the parameters and timings |
| `GET /outputs/<file>` | fetch a finished image |
| `POST /enhance-prompt` | only rewrite the prompt |
| `GET /resolutions` | table of sizes |
| `GET /checkpoints`, `POST /checkpoints/load` | list and switch the checkpoint without a restart |
| `GET /loras` | list of LoRAs |

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

## Verification scripts

Every stage of building the engine is verified separately - handy if something breaks after
updating the libraries or on another machine.

| Script | What it confirms |
|---|---|
| `scripts/check_weights.py` | the checkpoint loads into the model in strict mode |
| `scripts/check_text_encoder.py` | Qwen is loaded correctly: it encodes the prompt and writes coherent text |
| `scripts/smoke_test.py` | the whole path "by hand", linearly, without the engine |
| `scripts/check_loras.py` | which LoRAs are read, which are not, and why |
| `scripts/check_checkpoints.py` | every checkpoint produces an image |
| `scripts/profile_step.py` | where the step time goes; with `--real` - a comparison of the multiplication modes on the real checkpoint |

## Measurements (RTX 5080 Laptop 16 GB, 64 GB RAM, turbo, 8 steps)

| Resolution | Sampling | VAE | Peak VRAM |
|---|---|---|---|
| 480 (736x560) | ~7 s | ~1 s | - |
| 720 (1280x720) | ~15 s | ~1.5 s | 13.3 GB |
| 1080 (1920x1088) | ~38 s | ~1 s (bf16) | 13.8 GB |

The table above is the initial state of the engine (exact multiplication, ordinary memory, the default
attention kernel). After three optimizations - fast multiplication, pinned memory and a fast attention
kernel, described below - a whole request takes:

| Resolution | New prompt | Same prompt, different seed | Of which sampling |
|---|---|---|---|
| 480 (736x560) | 4.9 s | 3.9 s | 2.8 s |
| 720 (1280x720) | 9.9 s | 8.9 s | 7.2 s |
| 1080 (1920x1088) | ~23 s | 22.2 s | 19.8 s |

For comparison: a repeated request at 720 used to take about 20 s.
The first load of a checkpoint depends on the disk: 12.6 GB are read sequentially.

### Pinned memory (`MY_DIFFUSION_PIN_MEMORY`)

One large model lives on the GPU at a time, so every request moves the models between
RAM and the GPU. From ordinary RAM this is ~4 GB/s (the network takes about 3 s), from
pinned memory - 26-36 GB/s (about 0.5 s). That is why the weights of all three models live
in pinned memory. The cost: about 19 GB of RAM that the system cannot give to other programs
or page out to swap. If RAM is short - `MY_DIFFUSION_PIN_MEMORY=false`.

### Attention kernel (`MY_DIFFUSION_ATTENTION`)

torch computes attention with one of several ready-made "kernels". By default on Windows torch uses the
`efficient` kernel (FlashAttention is not included in the Windows build). The same build has a kernel from
NVIDIA's cuDNN library - it is 2-2.4 times faster, and we put it first (`cudnn`):

| Kernel | Sampling 480 / 720 / 1080 (`fp8_all` mode) |
|---|---|
| `default` | 3.0 / 8.3 / 26.3 s |
| `cudnn` (default) | 2.8 / 7.2 / 19.8 s |

The accuracy of the kernels is the same; they differ from each other only at the rounding level (in the `exact` mode
the pictures differ by 41-44 dB PSNR - not visible to the eye). A peculiarity of `cudnn`: for each new
sequence length (a new resolution or a prompt of a different length) it is prepared once, taking
0.2-0.4 s. To compare with ComfyUI, set `MY_DIFFUSION_ATTENTION=default`.

### Multiplication mode (`MY_DIFFUSION_MATMUL`)

Exact mode (`exact`): the compressed weights of each layer are unpacked into bf16 and multiplied as usual. NVIDIA 50-series
GPUs can multiply mxfp8 matrices directly in compressed form - these are the `fp8` and `fp8_all` modes. `fp8_all` is enabled by default:

| Mode | Sampling 480 / 720 / 1080 | What changes |
|---|---|---|
| `exact` | ~7 / 15.6 / 39.1 s | the result matches ComfyUI bit for bit |
| `fp8` | 5.0 / 11.3 / 32.8 s | the input of 160 of the 256 layers is rounded to fp8; the other 96 are computed exactly, as the checkpoint author asks (ComfyUI does the same) |
| `fp8_all` (default) | 3.1 / 9.1 / 26.0 s | the same for all 256 layers; the picture is of the same quality, but differs from `exact` in details |

The larger the picture, the smaller the gain: at 1080p a third of the step is attention, which this
mode does not speed up. Fast modes work only with mxfp8 checkpoints (other formats
are computed as `exact`) and only on NVIDIA 50-series GPUs - on another card set
`MY_DIFFUSION_MATMUL=exact`. The peak VRAM is the same. To compare the modes on your own prompt: `uv run python scripts/profile_step.py --real` -
the pictures are saved side by side in `outputs/ab_*.png`.

## Configuration

Environment variables `MY_DIFFUSION_*` or a `.env` file (see `.env.example`).

## Not implemented yet

- image-to-image, inpainting, editing from references
- LoKr/LoHa
- request queue: generations run strictly one at a time
- other Qwen models for the enhancer (Qwen3.5-4B-Abliterated, qwen3_8b): links to them
  in `models/text_encoders` are created, but the loader currently supports only Qwen3-VL-4B
