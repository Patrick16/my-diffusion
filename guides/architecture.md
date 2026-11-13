# Architecture

## Request path

text -> Qwen -> noise -> network x N steps -> VAE -> PNG.

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
| `background.py` | Background removal (u2net on onnxruntime, CPU). |
| `resolutions.py` | Resolution + aspect ratio -> width and height. |
| `api.py`, `schemas.py`, `cli.py` | HTTP and the command line on top of the engine. |
| `config.py` | Settings from `MY_DIFFUSION_*` variables and `.env`. |

## Why earlier attempts produced noise

Before this engine, we tried `diffusers.Krea2Pipeline` and a port of the network from ComfyUI - both
produced noise. The network itself is not complicated; it is easy to get something wrong around it.
What has to be done right (and what is now verified by scripts):

1. **Weight format.** `krea2_turbo_mxfp8` is not an ordinary fp8: every 32 numbers have their own
   multiplier 2^(e-127), and the table of exponents is also rearranged in 128x4 tiles. Our unpacking is
   checked bit for bit against the ComfyUI library (`scripts/check_weights.py`).
2. **The schedule shift** sits in the exponent: `mu=1.15` is a shift by a factor of 3.16.
3. **The noise level** is fed to the network as a number 0..1, not 0..1000.
4. **Latent normalization** per channel before the VAE: `z = latent * std + mean`.
5. **Text**: 12 layers of Qwen (`hidden_states[2,5,...,35]`); the 34-token service prefix is cut off
   after the forward pass, not before.

## Tried and rejected

- **Skipping steps with a cache** (reusing the deep blocks' contribution from the previous step when
  the first block's output changed little, as in first-block cache / TeaCache). Measured on the turbo
  checkpoint: at 8 steps there is nothing to reuse (the reused part is 23-40% off), and at 28 steps a
  cached run stays numerically closer to the full run than a plain run with the same number of network
  passes, but the image is visibly broken by a noisy mosaic texture, while plain 8-10 steps give a clean
  image. Use fewer `steps` instead.
- **Storing a checkpoint as "base + low-rank correction"** (to switch checkpoints in seconds instead of
  reading another 13 GB). Measured with `scripts/compare_checkpoints.py` on five Krea 2 turbo
  checkpoints: the difference from the base one is mostly spread over the whole matrix, not low-rank.
  Even 1024 of 6144 singular directions leave 1.5-5% of the weights unexplained, while rounding into an
  8-bit format alone accounts for 0.6-3%. One checkpoint is the base within rounding, one has a strong
  low-rank part plus a spread remainder, two have no low-rank part at all. The minutes come from the cold
  disk, not from the format: from the OS file cache a checkpoint loads in about 7 s.
