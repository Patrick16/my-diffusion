"""The VAE at large sizes: peak video memory and time of encode and decode, whole and in tiles.

    uv run python scripts/check_vae_sizes.py [SIZES...] [--source PICTURE]

SIZES: 1080 (1920x1088), 1440 (2560x1440), 4k (3840x2160); default: all three. The source picture (default
outputs/memory/final_t2i_1440.png, made by scripts/check_memory.py; any large picture will do) is resized to each
size, encoded and decoded back:
    whole          the engine's way: one pass over the full picture (bf16 above 1.3 megapixels)
    tiles T/S      diffusers' tiling: tiles of T pixels every S pixels, blended over the overlap

For each it prints the time, the peak memory, the PSNR of the round trip against the source, and for the tiles how
far their result is from the whole one (the latent: mean absolute difference, in units of the latent's spread; the
decoded picture: PSNR). Pictures: outputs/memory/vae_<size>_<variant>.png.

Only the VAE is loaded. The process is capped at the video memory free at start, so an overrun is an OOM."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.vae import LatentDecoder

WORK = PROJECT_ROOT / "outputs" / "memory"
GIB = 2**30
SIZES = {"1080": (1920, 1088), "1440": (2560, 1440), "4k": (3840, 2160)}
TILES = ((1024, 896), (768, 640), (512, 384))   # tile side, stride (pixels)


def psnr(a: Image.Image, b: Image.Image) -> float:
    mse = float(((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def measured(call, *args):
    """(result or None on OOM, peak GB, seconds)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        out = call(*args)
    except torch.OutOfMemoryError:
        return None, torch.cuda.max_memory_allocated() / GIB, 0.0
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() / GIB, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", nargs="*", help=f"any of {list(SIZES)}; default: all")
    ap.add_argument("--source", type=Path, default=WORK / "final_t2i_1440.png")
    args = ap.parse_args()
    if unknown := [name for name in args.sizes if name not in SIZES]:
        ap.error(f"unknown size {unknown}; available: {list(SIZES)}")
    WORK.mkdir(parents=True, exist_ok=True)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.97, free / total))
    print(f"video memory: {free / GIB:.1f} GB free of {total / GIB:.1f} GB; the process is capped at the free part", flush=True)
    vae = LatentDecoder(Settings().vae_path)
    vae.load()
    vae.to("cuda")
    source = Image.open(args.source).convert("RGB")
    for name in args.sizes or list(SIZES):
        w, h = SIZES[name]
        image = source.resize((w, h), Image.LANCZOS)
        vae.vae.disable_tiling()
        latent, peak, sec = measured(vae.encode, image)
        print(f"{name} {w}x{h}: encode whole: " + ("OOM" if latent is None else f"{sec:.2f}s") + f", peak {peak:.2f} GB", flush=True)
        whole = None
        if latent is not None:
            whole, peak, sec = measured(vae.decode, latent)
            result = "OOM" if whole is None else f"{sec:.2f}s, PSNR vs source {psnr(whole, image):.1f} dB"
            print(f"{name}: decode whole: {result}, peak {peak:.2f} GB", flush=True)
            if whole is not None:
                whole.save(WORK / f"vae_{name}_whole.png")
        for tile, stride in TILES:
            vae.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                                  tile_sample_stride_height=stride, tile_sample_stride_width=stride)
            tiled_latent, peak_e, sec_e = measured(vae.encode, image)
            if tiled_latent is None:
                print(f"{name}: tiles {tile}/{stride}: encode OOM, peak {peak_e:.2f} GB", flush=True)
                continue
            line = f"{name}: tiles {tile}/{stride}: encode {sec_e:.2f}s peak {peak_e:.2f} GB"
            if latent is not None:
                line += f" (latent differs from whole by {float((tiled_latent - latent).abs().mean()):.4f})"
            # decode the whole-encoded latent when there is one: then the two decodes differ only by the tiling
            tiled, peak_d, sec_d = measured(vae.decode, latent if latent is not None else tiled_latent)
            if tiled is None:
                line += f"; decode OOM peak {peak_d:.2f} GB"
            else:
                line += f"; decode {sec_d:.2f}s peak {peak_d:.2f} GB, PSNR vs source {psnr(tiled, image):.1f} dB"
                if whole is not None:
                    line += f", vs whole decode {psnr(tiled, whole):.1f} dB"
                tiled.save(WORK / f"vae_{name}_tiles{tile}.png")
            print(line, flush=True)
    vae.vae.disable_tiling()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
