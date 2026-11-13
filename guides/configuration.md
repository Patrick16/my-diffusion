# Configuration

Settings are read from environment variables `MY_DIFFUSION_*` or from a `.env` file in the project
root. Copy `.env.example` to `.env` and adjust it. Every variable is optional: without `.env` the
defaults from `src/my_diffusion/config.py` are used.

## Paths and devices

| Variable | Default | Meaning |
|---|---|---|
| `MY_DIFFUSION_DEVICE` | `cuda` | torch device |
| `MY_DIFFUSION_MODELS_DIR` | `./models` | folder with links to the models (created by `scripts/link_models.ps1`) |
| `MY_DIFFUSION_OUTPUT_DIR` | `./outputs` | where generated PNGs are saved |

## Models

| Variable | Default | Meaning |
|---|---|---|
| `MY_DIFFUSION_CHECKPOINT` | `krea2_turbo_mxfp8.safetensors` | file in `models/checkpoints` (`.safetensors` or `.gguf`) |
| `MY_DIFFUSION_CHECKPOINT_KIND` | `auto` | `auto`, `turbo` or `raw`; sets the default steps and guidance (see [checkpoints](checkpoints.md)) |
| `MY_DIFFUSION_VAE` | `qwen_vae.safetensors` | file in `models/vae` |
| `MY_DIFFUSION_TEXT_ENCODER` | `Qwen3-VL-4B-Instruct` | folder in `models/text_encoders` |
| `MY_DIFFUSION_ENHANCER_MAX_NEW_TOKENS` | `220` | length limit of the prompt enhancer's output |

## Performance

| Variable | Default | Meaning |
|---|---|---|
| `MY_DIFFUSION_MATMUL` | `fp8_all` | `exact`, `fp8` or `fp8_all`; fast modes need mxfp8 checkpoints and an NVIDIA 50-series GPU, otherwise use `exact` |
| `MY_DIFFUSION_ATTENTION` | `cudnn` | `cudnn` or `default` (the torch choice, for comparison with ComfyUI) |
| `MY_DIFFUSION_PIN_MEMORY` | `true` | keep weights in pinned RAM (faster model switching, ~19 GB of RAM); set `false` if RAM is short |
| `MY_DIFFUSION_LORA_STREAMING` | `true` | stream LoRA pairs to the GPU block by block (less VRAM, slightly slower) |

Details and measurements: [performance](performance.md).

## Reference editing and background removal

| Variable | Default | Meaning |
|---|---|---|
| `MY_DIFFUSION_EDIT_LORA` | `krea2_identity_edit_v1_2` | edit LoRA in `models/loras`, added automatically for references |
| `MY_DIFFUSION_EDIT_LORA_WEIGHT` | `1.0` | its weight; a request can name the LoRA itself, weight 0 switches it off |
| `MY_DIFFUSION_VISION_MAX_SIDE` | `768` | longer side of a picture before the vision tower |
| `MY_DIFFUSION_MAX_REFERENCES` | `2` | maximum number of reference pictures |
| `MY_DIFFUSION_REMBG_MODEL` | `rembg/u2net.onnx` | background-removal model in `models/` |

## Server

| Variable | Default | Meaning |
|---|---|---|
| `MY_DIFFUSION_HOST` | `127.0.0.1` | bind address |
| `MY_DIFFUSION_PORT` | `8000` | port |

## Example `.env`

```bash
MY_DIFFUSION_CHECKPOINT=krea2_turbo_mxfp8.safetensors
MY_DIFFUSION_MATMUL=fp8_all
MY_DIFFUSION_ATTENTION=cudnn
MY_DIFFUSION_PIN_MEMORY=true
```
