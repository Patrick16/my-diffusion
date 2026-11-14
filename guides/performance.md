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
sampling at 480 (one reference) and about 20 s at 720. Larger sizes and two references are accepted
too (`MY_DIFFUSION_EDIT_LIMITS`), see below. See [image modes](image-modes.md).

## Large sizes: what limits them and what was measured

The tiers that strict mode refuses above (`EDIT_MAX_TIER` in `engine.py`) are a policy, not the card's limit
any more (see [configuration](configuration.md): `MY_DIFFUSION_EDIT_LIMITS`). There is still no 1440 tier in the
presets; the table below lifted it for the measurements. Measured on the same 16 GB laptop GPU with 14.7 GB free, turbo, 8 steps, with
`scripts/check_memory.py`, which lifts the rules for itself. Times are for the sampling alone; this
session ran about 30% slower than the tables above (the laptop was in another power plan), so compare the
rows with each other, not with those tables.

| Case | Tokens | Before | Now | With 8 blocks streamed |
|---|---|---|---|---|
| text-to-image 1080 (1920x1088) | 8180 | 13.9 GB, 27 s | the same picture | 10.7 GB, +1.2 s |
| text-to-image 1440 (2560x1440) | 14420 | out of memory | 14.0 GB, 56 s | |
| edit 720, one reference | 7559 | 14.0 GB, 28 s | the same picture | |
| edit 720, two references | 10084 | out of memory | 13.9 GB, 39 s | |
| outpaint 1080 (2016x1024) | 13902 | out of memory | 14.1 GB, 60 s | |
| edit 1080, one reference | 16679 | out of memory | 14.3 GB, 77 s | 11.5 GB, 77 s |
| edit 1080, two references | 21895 | out of memory | out of memory | 12.1 GB, 113 s |
| outpaint 1440 (2688x1360) | 24262 | out of memory | out of memory | 12.3 GB, 131 s |
| edit 1440, one reference | 29159 | out of memory | out of memory | 12.8 GB, 169 s |

Two things changed, and one mechanism was added.

**Working memory of a block.** The weights take 12.6 GB, so a block has about 2 GB to work in. Three of
its parts used to compute over the whole sequence at once and needed many copies of it for a moment: the
MLP (its hidden layer is 2.7 times wider than the stream), RoPE and the normalizations (both in float32).
They now go over a long sequence in pieces of 4096 tokens (`CHUNK_TOKENS` in `transformer.py`), and the
attention frees its tensors as soon as they are used. Sampling time is the same. The normalizations and
RoPE give the same bits in pieces; so does the compressed (fp8) matrix product, so text-to-image pictures
do not change. The product of a LoRA pair is rounded differently in pieces, which would give every edit
another variant of its picture - so the MLP is cut only above 10000 tokens (`MLP_WHOLE_TOKENS`), longer
than anything that fitted before: every picture that could be made before comes out bit for bit the same.

**Streamed weights** (`Krea2Engine.stream_weights(n)`). The engine now uses it on its own: before sampling it
counts the sequence (text + picture + reference tokens), takes the free video memory from torch (plus what the
network already holds on the GPU) and, if the rule of thumb below says the peak will not fit, streams the fewest
blocks that fit plus one spare, with a 0.5 GB margin (`limits.plan_streaming`). After the job the previous
state is put back, so the next small job does not pay for it. The job's result says how many blocks it used
(`streamed_blocks`) and how long the sequence was (`tokens`). If a job still runs out of memory, it fails with a
clear message, the memory is freed and the next job runs (`GenerationOutOfMemory`). A block's
weights (0.42 GB) can stay in pinned RAM and be copied to the GPU only while that block runs, like the
LoRA pairs with LoRA streaming. A copy takes about 22 ms, so one block costs 0.18 s per 8-step generation:
1.2 s for 8 blocks, 5-6 s for all 28, at any size - and the picture is bit for bit the same. With all 28
streamed the sampling of a 1440p edit peaks at 4.5 GB instead of not fitting at all. Four blocks are
enough for every case of the table. So the sizes are limited by time, not memory: beyond 16 thousand
tokens the attention takes most of a step.

A rule of thumb for the peak during sampling, in GB, for a sequence of T thousand tokens with n blocks
streamed: `12.7 + 0.09 * T - 0.42 * (n - 1)` for n >= 1 (one streamed block is on the GPU at any moment),
`12.7 + 0.09 * T` for n = 0; it held within 0.3 GB for the cases above. torch takes about 0.5 GB more from
the driver than it uses, so on this machine (14.7 GB free) a peak above ~14.2 GB is the edge: the 1080p
edit with one reference fits without streaming, but only just. With all 28 streamed the peak is about
`1.3 + 0.1 * T`. (The streamed column of the table was measured one small saving earlier and is 0.2-0.3 GB
higher than the rule.)

**The VAE** runs alone on the GPU and fits up to 1440p in one pass: decode 12.1 GB / 3.4 s, encoding a
1440p reference 9.5 GB. 4K does not (encode runs out of memory). Tiling (diffusers'
`enable_tiling`, tiles of 1024 px every 896) brings a 1440p decode to 4.4 GB and 2.2 s and a 4K decode to
4.5 GB and 5.3 s; the tiled picture is 46-47 dB PSNR from the whole one, which the eye does not see. The
engine does not tile yet. Measured with `scripts/check_vae_sizes.py`.

**Cheaper references** were tried with `scripts/check_reference_cost.py` on 720 edits:

| Variant | Step | Face similarity to the reference |
|---|---|---|
| as it is | 3.5 s | 0.70 |
| reference encoded at half the side (a quarter of its tokens) | 2.2 s | 0.70 |
| the same, positions stretched over the frame | 2.2 s | 0.55 |
| reference keys / values computed once and reused ("K/V cache") | 2.0 s | 0.27 |
| the same, recomputed on every step | 3.5 s | 0.21 |

The reference tokens cannot be cached: they must see the noisy picture and the text on every step, and
when they do not the subject is lost. A half-size reference keeps the subject and saves 40% of the step;
it is one face and one seed so far, not a feature yet.

`MY_DIFFUSION_LORA_STREAMING=false` keeps the edit LoRA on the GPU (0.83 GB more): with 14.7 GB free a 720
edit then runs out of memory, so leave it on.
