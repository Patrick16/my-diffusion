"""Profile: where the step time goes and how the multiplication modes (exact / fp8 / fp8_all) differ.

We measure every optimization first, then evaluate it. The script has two modes.

1) No arguments - synthetic: one network block with random weights (no disk or checkpoint needed,
   runs in seconds). Shows the time of the block by parts for three resolutions and three multiplication
   modes. The step time = 28 such blocks; checked against the real network - it matches.

       uv run python scripts/profile_step.py

2) --real - the real checkpoint: the same seeds are computed with each multiplication mode.
   Prints the step time, peak video memory and the difference of the result from the exact mode, and saves the pictures
   side by side into outputs/ab_<size>_s<seed>.png (left to right in the --modes order).
   The numbers are numbers, but the eye decides: open the file and compare.

       uv run python scripts/profile_step.py --real [--resolution 720] [--seeds 42 43] [--modes exact fp8 fp8_all]"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.attention import sdpa_kernel

from my_diffusion import sampler, transformer
from my_diffusion.config import Settings
from my_diffusion.resolutions import ASPECT_RATIOS, RESOLUTION_TIERS, resolve_dimensions
from my_diffusion.transformer import (
    ATTENTION_KERNELS, CUDNN_FIRST, Krea2Config, Krea2Transformer, SingleStreamBlock, apply_rope, build_rope, set_attention_kernel,
)
from my_diffusion.weights import MATMUL_MODES, QLinear, load_into, quantize_mxfp8, read_checkpoint, set_matmul, shared_input

# In the real krea2_turbo_mxfp8 checkpoint the author marked these layers as "multiply only exactly".
FULL_PRECISION = {"attn.gate", "attn.wo", "mlp.down"}
TEXT_TOKENS = 60


def timeit(fn, repeats: int = 5) -> float:
    """Average call time in milliseconds. synchronize is required: the GPU executes
    commands asynchronously, and without waiting we would only measure queuing them."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def relative_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---------------------------------------------------------------------- synthetic
def synthetic(device: str, attention: str) -> None:
    cfg = Krea2Config()
    torch.manual_seed(0)
    block = SingleStreamBlock(cfg).to(device)
    set_attention_kernel(block, attention)
    for name, layer in block.named_modules():
        if isinstance(layer, QLinear):
            packed = quantize_mxfp8(torch.randn(layer.out_features, layer.in_features, device=device) * 0.02)
            layer.set_weight(packed.data, "mxfp8", packed.scales.view(torch.uint8), None, full_precision=name in FULL_PRECISION)
            del packed
    attn, mlp = block.attn, block.mlp
    params = sum(m.weight.numel() for m in block.modules() if isinstance(m, QLinear))
    print(f"one block: {params / 1e6:.0f} M weights in mxfp8; blocks in the network {cfg.layers}; attention kernel: {attention}\n")
    print(f"{'size':>6} {'mode':>8} | {'block, ms':>8} {'step, s':>7} | {'MLP':>6} {'lin. attn':>10} {'attn':>6} {'other':>6} | {'diff from exact':>16}")

    for tier, (gh, gw) in {"480": (35, 46), "720": (45, 80), "1080": (68, 120)}.items():
        n = TEXT_TOKENS + gh * gw
        x = torch.randn(1, n, cfg.features, device=device, dtype=torch.bfloat16)
        tvec = torch.randn(1, 1, 6 * cfg.features, device=device, dtype=torch.bfloat16) * 0.1
        rope = build_rope(torch.zeros(1, n, 3, device=device), (32, 48, 48), cfg.theta)
        q = torch.randn(1, cfg.heads, n, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, cfg.kvheads, n, 128, device=device, dtype=torch.bfloat16)

        def attention_core() -> None:
            qq, kk = apply_rope(attn.qknorm.qnorm(q), rope), apply_rope(attn.qknorm.knorm(k), rope)
            kk = kk.repeat_interleave(cfg.heads // cfg.kvheads, dim=1)
            if attn.kernel == "sage2":
                transformer._sageattn(qq, kk, kk, tensor_layout="HND", is_causal=False)
            elif attn.kernel == "cudnn":
                with sdpa_kernel(CUDNN_FIRST, set_priority=True):
                    F.scaled_dot_product_attention(qq, kk, kk)
            else:
                F.scaled_dot_product_attention(qq, kk, kk)

        def attention_linears() -> None:
            xs = shared_input(x, attn.wq, attn.wk, attn.wv, attn.gate)
            attn.wq(xs), attn.wk(xs), attn.wv(xs), attn.gate(xs), attn.wo(x)

        reference = None
        for mode in MATMUL_MODES:
            set_matmul(block, mode, device)
            out = block(x, tvec, rope)
            reference = out if reference is None else reference
            t_block = timeit(lambda: block(x, tvec, rope))
            t_mlp, t_lin, t_att = timeit(lambda: mlp(x)), timeit(attention_linears), timeit(attention_core)
            print(f"{tier:>6} {mode:>8} | {t_block:8.1f} {t_block * cfg.layers / 1000:7.2f} | {t_mlp:6.1f} {t_lin:10.1f} {t_att:6.1f} "
                  f"{t_block - t_mlp - t_lin - t_att:6.1f} | {relative_diff(out, reference):16.1e}")
        print()
        del x, q, k, rope, reference, out
        torch.cuda.empty_cache()
    print("'diff from exact' here is measured on random weights - it checks the wiring, not the image quality.")


# ---------------------------------------------------------------------- real checkpoint
def real(args, device: str) -> None:
    from my_diffusion.text_encoder import QwenVL
    from my_diffusion.vae import LATENT_CHANNELS, VAE_SCALE, LatentDecoder

    settings = Settings()
    width, height = resolve_dimensions(args.resolution, args.aspect_ratio)

    t0 = time.time()
    qwen = QwenVL(settings.text_encoder_dir)
    qwen.load()
    qwen.to(device)
    cond = qwen.encode(args.prompt)
    qwen.to("cpu")
    del qwen
    torch.cuda.empty_cache()
    print(f"text: {cond.shape[1]} tokens, {time.time() - t0:.0f}s")

    t0 = time.time()
    model = Krea2Transformer()
    load_into(model, read_checkpoint(settings.checkpoint_path))
    set_attention_kernel(model.blocks, args.attention)
    model.to(device)
    formats = sorted({m.fmt for m in model.modules() if isinstance(m, QLinear)})
    print(f"checkpoint {settings.checkpoint}: layer formats {formats}, {time.time() - t0:.0f}s\n")

    lat_h, lat_w = height // VAE_SCALE, width // VAE_SCALE
    mu = sampler.auto_mu((lat_h // 2) * (lat_w // 2))
    sigmas = sampler.make_sigmas(args.steps, mu)

    latents: dict[tuple[str, int], torch.Tensor] = {}
    print(f"{width}x{height}, {args.steps} steps, seed {args.seeds}, attention kernel {args.attention}")
    print(f"{'mode':>8} {'fp8 layers':>9} | {'step, s':>7} {'sampling, s':>11} {'peak VRAM, GB':>12}")
    for mode in args.modes:
        native_layers = set_matmul(model, mode, device)
        # warm-up: the first call spends time on one-time setup of the GPU libraries
        sampler.sample(model, sampler.make_noise(0, LATENT_CHANNELS, lat_h, lat_w).to(device), cond, sigmas[:2])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step_times: list[float] = []
        for seed in args.seeds:
            marks = [time.perf_counter()]

            def on_step(i: int, total: int) -> None:
                torch.cuda.synchronize()
                marks.append(time.perf_counter())

            noise = sampler.make_noise(seed, LATENT_CHANNELS, lat_h, lat_w).to(device)
            latents[mode, seed] = sampler.sample(model, noise, cond, sigmas, on_step=on_step).cpu()
            step_times += [b - a for a, b in zip(marks, marks[1:])]
        per_step = sum(step_times) / len(step_times)
        print(f"{mode:>8} {native_layers:>9} | {per_step:7.2f} {per_step * args.steps:11.1f} {torch.cuda.max_memory_allocated() / 2**30:12.1f}")

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    vae = LatentDecoder(settings.vae_path)
    vae.load()
    vae.to(device)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.modes[0]
    print(f"\ndifference from '{base}': latent - relative, image - PSNR in dB (higher = closer; 40+ is usually not visible to the eye)")
    for seed in args.seeds:
        images = {mode: vae.decode(latents[mode, seed]) for mode in args.modes}
        sheet = Image.new("RGB", (width * len(images), height))
        for i, image in enumerate(images.values()):
            sheet.paste(image, (i * width, 0))
        path = settings.output_dir / f"ab_{width}x{height}_s{seed}.png"
        sheet.save(path)
        for mode in args.modes[1:]:
            a = torch.frombuffer(bytearray(images[mode].tobytes()), dtype=torch.uint8).float()
            b = torch.frombuffer(bytearray(images[base].tobytes()), dtype=torch.uint8).float()
            mse = ((a - b) ** 2).mean().item()
            psnr = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
            print(f"  seed {seed}, {mode:>8}: latent {relative_diff(latents[mode, seed], latents[base, seed]):.3f}, image {psnr:.1f} dB")
        print(f"  seed {seed}: {path}  (left to right: {', '.join(args.modes)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="use the real checkpoint (otherwise synthetic)")
    ap.add_argument("--prompt", default="a fox walking in fresh snow at golden hour, natural light, detailed fur")
    ap.add_argument("--resolution", choices=list(RESOLUTION_TIERS), default="720")
    ap.add_argument("--aspect-ratio", choices=list(ASPECT_RATIOS), default="16:9")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--modes", nargs="+", choices=MATMUL_MODES, default=list(MATMUL_MODES), help="the first one - the others are compared against it")
    ap.add_argument("--attention", choices=ATTENTION_KERNELS, default=Settings().attention, help="attention kernel (default from the settings)")
    args = ap.parse_args()

    device = Settings().device
    with torch.inference_mode():
        real(args, device) if args.real else synthetic(device, args.attention)


if __name__ == "__main__":
    main()
