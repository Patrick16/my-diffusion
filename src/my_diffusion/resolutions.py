"""Resolution tiers + aspect ratios -> concrete (width, height).

Idea (borrowed from Wan2GP, shared/resolutions.py, but greatly simplified):
instead of listing 24 pairs of numbers by hand (3 tiers x 8 formats),
we set for each tier a "pixel budget" - the area of a 16:9 frame at this
height (so 720 -> 1280x720 -> 921600px, 1080 -> 1920x1080 -> 2073600px -
exactly the familiar HD/FHD). For any other format we take the same area
and solve the system: width*height = budget, width/height = aspect_ratio.

So all formats on one tier get the same "weight" for the GPU
(the same number of latent tokens), while the tiers remain intuitive
numbers 480/720/1080.

The sides must be multiples of 16: the VAE compresses the image 8 times per side, and the network
cuts the latent into 2x2 patches (8 * 2 = 16). Therefore both sides are rounded to the
nearest multiple of 16, and the area comes out slightly different for different formats."""

from __future__ import annotations

import math
from typing import NamedTuple

BLOCK_SIZE = 16

# Tier -> nominal height of a 16:9 frame, which sets the pixel budget.
RESOLUTION_TIERS: dict[str, int] = {
    "480": 480,
    "720": 720,
    "1080": 1080,
}

# Format "width:height" -> width/height ratio.
ASPECT_RATIOS: dict[str, float] = {
    "1:1": 1 / 1,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "21:9": 21 / 9,
    "9:21": 9 / 21,
}


class Dimensions(NamedTuple):
    width: int
    height: int


def _round_to_block(value: float, block_size: int = BLOCK_SIZE) -> int:
    """Rounds to the nearest multiple of block_size, never down to 0."""
    return max(block_size, round(value / block_size) * block_size)


def _pixel_budget(tier: str) -> int:
    height_16_9 = RESOLUTION_TIERS[tier]
    width_16_9 = height_16_9 * ASPECT_RATIOS["16:9"]
    return round(height_16_9 * width_16_9)


def resolve_dimensions(tier: str, aspect_ratio: str) -> Dimensions:
    """tier ('480'|'720'|'1080') + aspect_ratio ('16:9', ...) -> (width, height).

    The frame area is preserved within one tier; the sides are multiples of 16."""
    if tier not in RESOLUTION_TIERS:
        raise ValueError(f"Unknown resolution tier: {tier!r}. Available: {list(RESOLUTION_TIERS)}")
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Unknown aspect ratio: {aspect_ratio!r}. Available: {list(ASPECT_RATIOS)}")

    budget = _pixel_budget(tier)
    ratio = ASPECT_RATIOS[aspect_ratio]

    raw_height = math.sqrt(budget / ratio)
    raw_width = raw_height * ratio

    return Dimensions(width=_round_to_block(raw_width), height=_round_to_block(raw_height))


def resolve_dimensions_from_size(tier: str, width: float, height: float) -> Dimensions:
    """tier + the size of a given picture -> (width, height) with the tier's pixel budget and the
    picture's own aspect ratio (Wan2GP's calculate_new_dimensions with fit_canvas=0).

    Why: snapping to the nearest preset (see nearest_aspect_ratio) distorts or crops a picture
    whose proportions are not in the list. Here the aspect is kept as exactly as the 16-pixel
    grid allows, while the GPU cost stays the same as for any other frame of this tier."""
    if tier not in RESOLUTION_TIERS:
        raise ValueError(f"Unknown resolution tier: {tier!r}. Available: {list(RESOLUTION_TIERS)}")
    if not (width > 0 and height > 0 and math.isfinite(width) and math.isfinite(height)):
        raise ValueError(f"picture size must be positive, got {width}x{height}")
    scale = math.sqrt(_pixel_budget(tier) / (width * height))
    return Dimensions(width=_round_to_block(width * scale), height=_round_to_block(height * scale))


def nearest_aspect_ratio(width: int, height: int) -> str:
    """The ASPECT_RATIOS entry closest to width/height (used by aspect_ratio="auto" in editing).

    Closeness is measured on the logarithm of the ratio, so 2:1 and 1:2 are equally far from 1:1:
    a plain difference of ratios would treat wide and tall pictures unequally."""
    if width <= 0 or height <= 0:
        raise ValueError(f"picture size must be positive, got {width}x{height}")
    target = math.log(width / height)
    return min(ASPECT_RATIOS, key=lambda name: abs(math.log(ASPECT_RATIOS[name]) - target))


def all_combinations() -> dict[str, dict[str, Dimensions]]:
    """The whole tier -> format -> (w, h) table. Handy for /resolutions in the API and for logs."""
    return {
        tier: {ratio: resolve_dimensions(tier, ratio) for ratio in ASPECT_RATIOS}
        for tier in RESOLUTION_TIERS
    }


if __name__ == "__main__":
    # Quick visual check: python -m my_diffusion.resolutions
    for tier, ratios in all_combinations().items():
        print(f"--- {tier}p ---")
        for ratio, dims in ratios.items():
            megapixels = dims.width * dims.height / 1_000_000
            print(f"  {ratio:>5}: {dims.width}x{dims.height}  ({megapixels:.2f} MP)")
