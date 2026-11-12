"""Generation from the command line, without the HTTP server. Convenient for checks.

    uv run my-diffusion-generate "fox in the snow" --resolution 720 --aspect-ratio 16:9
    uv run my-diffusion-generate "portrait" --lora KreaAmateur_V2:0.8 --lora sliders/zoom_krea2_loraholic:-1 --count 3"""

from __future__ import annotations

import argparse
import logging

from .config import Settings
from .engine import Krea2Engine
from .resolutions import ASPECT_RATIOS, RESOLUTION_TIERS


def _parse_lora(value: str) -> tuple[str, float]:
    # "name:0.8" or just "name" (weight 1.0). We look for the colon from the right: it must not appear in the name, and the weight is always the last part.
    name, sep, weight = value.rpartition(":")
    if not sep:
        return value, 1.0
    return name, float(weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image with Krea 2")
    parser.add_argument("prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--resolution", choices=list(RESOLUTION_TIERS), default="720")
    parser.add_argument("--aspect-ratio", choices=list(ASPECT_RATIOS), default="16:9")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None, help="shift of the schedule mu; default is auto")
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]", help="can be given several times")
    parser.add_argument("--enhance", action="store_true", help="first expand the prompt with Qwen")
    parser.add_argument("--checkpoint", default=None, help="file in models/checkpoints")
    parser.add_argument("--count", type=int, default=1, help="how many images in a row (seed increases by 1 each time)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    settings = Settings()
    if args.checkpoint:
        settings.checkpoint = args.checkpoint
    engine = Krea2Engine(settings)
    engine.load()

    for i in range(args.count):
        result = engine.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
            seed=None if args.seed is None else args.seed + i,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            loras=[_parse_lora(v) for v in args.lora],
            enhance=args.enhance,
        )
        print(f"-> {result['image_path']}  ({result['width']}x{result['height']}, seed={result['seed']})")
        if result["enhanced_prompt"]:
            print(f"   enhanced: {result['enhanced_prompt']}")


if __name__ == "__main__":
    main()
