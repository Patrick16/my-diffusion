"""Check the VAE encoder: picture -> latent -> picture should give almost the same picture.

Also shows the latent statistics. After correct normalization the latent should have a mean near 0
and a spread near 1 (the same space the network works in); a wrong normalization shows up here at once.

    uv run python scripts/check_vae_roundtrip.py [--width 1280 --height 720]"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image, ImageDraw

from my_diffusion.config import Settings
from my_diffusion.vae import LatentDecoder


def make_test_picture(width: int, height: int) -> Image.Image:
    """A colour gradient with a few sharp shapes (sharp edges are the hard case for a VAE)."""
    x = np.linspace(0, 1, width)[None, :, None]
    y = np.linspace(0, 1, height)[:, None, None]
    r = np.broadcast_to(x, (height, width, 1))
    g = np.broadcast_to(y, (height, width, 1))
    b = np.broadcast_to(1 - (x + y) / 2, (height, width, 1))
    image = Image.fromarray((np.concatenate([r, g, b], axis=2) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 8, height // 6, width // 3, height // 2), fill=(240, 240, 30))
    draw.ellipse((width // 2, height // 4, width * 3 // 4, height * 3 // 4), fill=(20, 20, 20))
    draw.line((0, height - 1, width, 0), fill=(255, 255, 255), width=4)
    draw.polygon([(width // 6, height * 5 // 6), (width // 3, height * 2 // 3), (width // 2, height * 5 // 6)], fill=(200, 30, 120))
    return image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    settings = Settings()
    vae = LatentDecoder(settings.vae_path)
    vae.load()
    vae.to(settings.device)

    original = make_test_picture(args.width, args.height)

    t0 = time.time()
    latent = vae.encode(original)
    print(f"[1] encode: {time.time() - t0:.2f}s, latent {tuple(latent.shape)}, dtype {latent.dtype}, device {latent.device}")
    print(f"    latent: mean {latent.mean().item():.3f}, std {latent.std().item():.3f}, nan={bool(latent.isnan().any())}")

    t0 = time.time()
    restored = vae.decode(latent)
    print(f"[2] decode: {time.time() - t0:.2f}s")

    diff = np.abs(np.asarray(original, dtype=np.int16) - np.asarray(restored, dtype=np.int16))
    print(f"[3] mean absolute error: {diff.mean():.2f} of 255 (max {diff.max()})")


if __name__ == "__main__":
    main()
