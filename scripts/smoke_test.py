"""Step 3 check: end-to-end generation of one image "by hand", without the engine and the API.

Written deliberately linearly so that the whole path is visible:
text -> Qwen -> noise -> network x N steps -> VAE -> PNG.

    uv run python scripts/smoke_test.py [--mu comfy] [--width 1280 --height 720] [--seed 42]"""

from __future__ import annotations

import argparse
import time

import torch

from my_diffusion import sampler
from my_diffusion.config import Settings
from my_diffusion.text_encoder import QwenVL
from my_diffusion.transformer import Krea2Transformer
from my_diffusion.vae import LATENT_CHANNELS, VAE_SCALE, LatentDecoder
from my_diffusion.weights import load_into, read_checkpoint


def vram() -> str:
    return f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB (peak {torch.cuda.max_memory_allocated() / 2**30:.1f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a fox walking in fresh snow at golden hour, natural light, detailed fur")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mu", default="auto", help="auto | comfy | number")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    settings = Settings()
    dev = settings.device

    # 1. text -> Qwen hidden states
    t0 = time.time()
    qwen = QwenVL(settings.text_encoder_dir)
    qwen.load()
    qwen.to(dev)
    cond = qwen.encode(args.prompt)
    print(f"[1] text: {tuple(cond.shape)}, {time.time() - t0:.1f}s, {vram()}")
    qwen.to("cpu")
    del qwen
    torch.cuda.empty_cache()

    # 2. network to the GPU
    t0 = time.time()
    model = Krea2Transformer()
    load_into(model, read_checkpoint(settings.checkpoint_path))
    model.to(dev)
    print(f"[2] network loaded: {time.time() - t0:.1f}s, {vram()}")

    # 3. noise -> latent
    h, w = args.height // VAE_SCALE, args.width // VAE_SCALE
    tokens = (h // 2) * (w // 2)
    mu = sampler.auto_mu(tokens) if args.mu == "auto" else sampler.MU_COMFY if args.mu == "comfy" else float(args.mu)
    sigmas = sampler.make_sigmas(args.steps, mu)
    print(f"    {tokens} image tokens, mu={mu:.3f}, sigmas={[round(s, 3) for s in sigmas]}")
    noise = sampler.make_noise(args.seed, LATENT_CHANNELS, h, w).to(dev)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    step_t = [time.time()]
    latent = sampler.sample(model, noise, cond, sigmas, on_step=lambda i, n: step_t.append(time.time()))
    per_step = [round(b - a, 2) for a, b in zip(step_t, step_t[1:])]
    print(f"[3] sampling: {time.time() - t0:.1f}s, per step {per_step}, {vram()}")
    print(f"    latent: mean {latent.mean().item():.3f}, std {latent.std().item():.3f}, nan={bool(latent.isnan().any())}")

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    # 4. latent -> pixels
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    vae = LatentDecoder(settings.vae_path)
    vae.load()
    vae.to(dev)
    image = vae.decode(latent)
    print(f"[4] VAE: {time.time() - t0:.1f}s, {vram()}")

    out = args.out or str(settings.output_dir / f"smoke_{args.width}x{args.height}_s{args.seed}_mu{args.mu}.png")
    image.save(out)
    print("saved:", out)


if __name__ == "__main__":
    main()
