# Performance and memory

## Measurements

Hardware: RTX 5080 Laptop 16 GB, 64 GB RAM, turbo checkpoint, 8 steps.

Initial state of the engine (exact multiplication, ordinary memory, default attention kernel):

| Resolution | Sampling | VAE | Peak VRAM |
|---|---|---|---|
| 480 (736x560) | ~7 s | ~1 s | - |
| 720 (1280x720) | ~15 s | ~1.5 s | 13.3 GB |
| 1080 (1920x1088) | ~38 s | ~1 s (bf16) | 13.8 GB |

After three optimizations - fast multiplication, pinned memory and a fast attention kernel, described
below - a whole request takes:

| Resolution | New prompt | Same prompt, different seed | Of which sampling |
|---|---|---|---|
| 480 (736x560) | 4.9 s | 3.9 s | 2.8 s |
| 720 (1280x720) | 9.9 s | 8.9 s | 7.2 s |
| 1080 (1920x1088) | ~23 s | 22.2 s | 19.8 s |

For comparison: a repeated request at 720 used to take about 20 s. The first load of a checkpoint
depends on the disk: 12.6 GB are read sequentially.

## Pinned memory (`MY_DIFFUSION_PIN_MEMORY`)

One large model lives on the GPU at a time, so every request moves the models between RAM and the
GPU. From ordinary RAM this is ~4 GB/s (the network takes about 3 s), from pinned memory - 26-36 GB/s
(about 0.5 s). That is why the weights of all three models live in pinned memory. The cost: about
19 GB of RAM that the system cannot give to other programs or page out to swap. If RAM is short -
`MY_DIFFUSION_PIN_MEMORY=false`.

## Attention kernel (`MY_DIFFUSION_ATTENTION`)

torch computes attention with one of several ready-made "kernels". By default on Windows torch uses
the `efficient` kernel (FlashAttention is not included in the Windows build). The same build has a
kernel from NVIDIA's cuDNN library - it is 2-2.4 times faster, and we put it first (`cudnn`):

| Kernel | Sampling 480 / 720 / 1080 (`fp8_all` mode) |
|---|---|
| `default` | 3.0 / 8.3 / 26.3 s |
| `cudnn` (default) | 2.8 / 7.2 / 19.8 s |

The accuracy of the kernels is the same; they differ from each other only at the rounding level (in
the `exact` mode the pictures differ by 41-44 dB PSNR - not visible to the eye). A peculiarity of
`cudnn`: for each new sequence length (a new resolution or a prompt of a different length) it is
prepared once, taking 0.2-0.4 s. To compare with ComfyUI, set `MY_DIFFUSION_ATTENTION=default`.

### Optional: `sage2` (SageAttention 2)

`sage2` is a third kernel, off by default and not installed by default. Unlike the two above it is an
approximation: it quantizes queries and keys to int8 and the attention weights and values to fp8, and
multiplies them on the tensor cores. One attention call becomes 1.9-2.5 times faster than `cudnn`, but
attention is only a part of a step, so the sampling gains less - and more on long sequences:

| Case (tokens) | `cudnn` | `sage2` | Gain |
|---|---|---|---|
| 480 (1600) | 2.9 s | 2.8 s | 3% |
| 720 (3600) | 7.3 s | 6.9 s | 6% |
| 1080 (8200) | 20.4 s | 17.8 s | 13% |
| edit at 720, one reference (7600-7800) | 21.6-23.8 s | 19.7-20.3 s | 9-15% |

The picture changes: with the same seed `sage2` gives a different variant of it (PSNR 17-34 dB against
`cudnn`). That is not a loss of quality: the exact `default` kernel moves the picture just as far from
`cudnn` (19-29 dB), because sampling amplifies any small difference in the first steps. In the checked
pictures fine detail, lettering and the identity of a face in an edit were the same as with `cudnn`.
The peak VRAM is the same, so `sage2` does not make larger sizes fit.

To use it (Windows, NVIDIA GPU):

```
uv sync --group sage2
MY_DIFFUSION_ATTENTION=sage2
```

`MY_DIFFUSION_ATTENTION_SAGE2_MIN_TOKENS` (default 6000) keeps short sequences on `cudnn`, where the gain is
small: 480 and 720 text-to-image and one-reference edits at 480 then give exactly the same pictures as without
`sage2`; 1080 and edits at 720 use it. Note that a plain `uv sync` or `uv run` removes the optional package
again - use `uv sync --group sage2` / `uv run --group sage2`. To compare the kernels on your own machine:
`uv run --group sage2 python scripts/check_attention.py run`, then `... summary`.

## Multiplication mode (`MY_DIFFUSION_MATMUL`)

Exact mode (`exact`): the compressed weights of each layer are unpacked into bf16 and multiplied as
usual. NVIDIA 50-series GPUs can multiply mxfp8 matrices directly in compressed form - these are the
`fp8` and `fp8_all` modes. `fp8_all` is enabled by default:

| Mode | Sampling 480 / 720 / 1080 | What changes |
|---|---|---|
| `exact` | ~7 / 15.6 / 39.1 s | the result matches ComfyUI bit for bit |
| `fp8` | 5.0 / 11.3 / 32.8 s | the input of 160 of the 256 layers is rounded to fp8; the other 96 are computed exactly, as the checkpoint author asks (ComfyUI does the same) |
| `fp8_all` (default) | 3.1 / 9.1 / 26.0 s | the same for all 256 layers; the picture is of the same quality, but differs from `exact` in details |

The larger the picture, the smaller the gain: at 1080p a third of the step is attention, which this
mode does not speed up. Fast modes work only with mxfp8 checkpoints (other formats are computed as
`exact`) and only on NVIDIA 50-series GPUs - on another card set `MY_DIFFUSION_MATMUL=exact`. The peak
VRAM is the same. To compare the modes on your own prompt: `uv run python scripts/profile_step.py --real`
- the pictures are saved side by side in `outputs/ab_*.png`.

## Reference editing memory (edit LoRA)

The edit LoRA runs at rank 128 for every edit (128 vs 256 was measured as visually similar). It is
streamed per block from pinned RAM (`MY_DIFFUSION_LORA_STREAMING`, default `true`), which costs about
5% per network pass. Measured on a 16 GB laptop GPU, turbo, 8 steps: an edit takes about 9 s of
sampling at 480 (one reference) and about 20 s at 720. Two references work at 480 only; 1080 is not
supported. See [image modes](image-modes.md).
