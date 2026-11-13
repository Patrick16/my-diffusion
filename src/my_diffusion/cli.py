"""Generation from the command line, without the HTTP server. Convenient for checks.

    uv run my-diffusion-generate "fox in the snow" --resolution 720 --aspect-ratio 16:9
    uv run my-diffusion-generate "portrait" --lora KreaAmateur_V2:0.8 --lora sliders/zoom_krea2_loraholic:-1 --count 3"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Settings
from .engine import Krea2Engine
from .imaging import load_rgb
from .resolutions import ASPECT_RATIOS, RESOLUTION_TIERS
from .transformer import ATTENTION_KERNELS
from .weights import MATMUL_MODES


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
    parser.add_argument("--aspect-ratio", choices=[*ASPECT_RATIOS, "auto"], default="16:9", help="'auto' (only with --reference, scene mode) takes the exact proportions of reference 1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-start", type=float, default=0.0, help="fraction of the steps after which CFG starts (default 0)")
    parser.add_argument("--guidance-end", type=float, default=1.0, help="fraction of the steps after which CFG stops (default 1); lower is faster")
    parser.add_argument("--shift", type=float, default=None, help="shift of the schedule mu; default is auto")
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:WEIGHT]", help="can be given several times")
    parser.add_argument("--enhance", action="store_true", help="first expand the prompt with Qwen")
    parser.add_argument("--checkpoint", default=None, help="file in models/checkpoints")
    parser.add_argument("--count", type=int, default=1, help="how many images in a row (seed increases by 1 each time)")
    parser.add_argument("--matmul", choices=MATMUL_MODES, default=None, help="matrix multiplication mode; default from the settings")
    parser.add_argument("--attention", choices=ATTENTION_KERNELS, default=None, help="attention kernel; default from the settings")
    parser.add_argument("--init-image", type=Path, default=None, metavar="PATH", help="start from this picture (image-to-image); it is cropped to the requested size")
    parser.add_argument("--strength", type=float, default=None, help="with --init-image: how much to change the picture, 0 < strength <= 1 (default 0.6 for image-to-image, 1.0 for inpaint; 1.0 = start from pure noise)")
    parser.add_argument("--mask", type=Path, default=None, metavar="PATH", help="inpaint (needs --init-image): white = regenerate, black = keep; cropped like the picture")
    parser.add_argument("--feather", type=int, default=2, help="inpaint only: soft mask edge in latent pixels, 1 = 8 image pixels (default 2; 0 = hard edge)")
    parser.add_argument("--reference", action="append", type=Path, default=[], metavar="PATH", help="reference editing, can be given up to 2 times: 1 = the scene, 2 = a subject or person; cannot be combined with --init-image or --mask")
    parser.add_argument("--reference-mode", choices=["scene", "subjects"], default="scene", help="'scene': --reference 1 is the main scene, the rest are people / objects; 'subjects': all references are people / objects")
    parser.add_argument("--outpaint", nargs=4, type=float, default=None, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), help="extend reference 1 by these percentages of its sides and generate the new area (needs --reference, scene mode; replaces --aspect-ratio)")
    parser.add_argument("--remove-background", action="store_true", help="cut the background out of the people / objects: references 2.. in scene mode, all in subjects mode")
    args = parser.parse_args()
    if args.outpaint is not None and any(args.outpaint):
        if not args.reference:
            parser.error("--outpaint needs --reference")
        if args.reference_mode != "scene":
            parser.error("--outpaint needs --reference-mode scene")
        if any(v < 0 for v in args.outpaint):
            parser.error("--outpaint margins must not be negative")
    if args.aspect_ratio == "auto" and args.reference_mode != "scene":
        parser.error("--aspect-ratio auto needs --reference-mode scene")
    if len(args.reference) > 2:
        parser.error(f"at most 2 --reference pictures are supported, got {len(args.reference)}")
    if args.reference and (args.init_image or args.mask):
        parser.error("--reference cannot be combined with --init-image or --mask")
    if args.aspect_ratio == "auto" and not args.reference:
        parser.error("--aspect-ratio auto needs --reference")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    settings = Settings()
    if args.checkpoint:
        settings.checkpoint = args.checkpoint
    if args.matmul:
        settings.matmul = args.matmul
    if args.attention:
        settings.attention = args.attention
    engine = Krea2Engine(settings)
    engine.load()
    init_image = None if args.init_image is None else load_rgb(args.init_image.read_bytes())
    mask = None if args.mask is None else load_rgb(args.mask.read_bytes())   # grayscale is taken inside mask_to_latent

    references = [load_rgb(p.read_bytes()) for p in args.reference]

    for i in range(args.count):
        result = engine.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
            seed=None if args.seed is None else args.seed + i,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            guidance_interval=(args.guidance_start, args.guidance_end),
            shift=args.shift,
            loras=[_parse_lora(v) for v in args.lora],
            enhance=args.enhance,
            init_image=init_image,
            strength=args.strength,
            mask=mask,
            feather=args.feather,
            references=references,
            reference_mode=args.reference_mode,
            outpaint=None if args.outpaint is None else tuple(args.outpaint),
            remove_background="subjects" if args.remove_background else "none",
        )
        if result["mode"] == "edit":
            mode = f", edit refs={result['references']} lora_rank={result['edit_lora_rank']}"
        elif result["mode"] == "inpaint":
            mode = ", inpaint"
        elif result["mode"] == "image2image":
            mode = f", image2image strength={result['strength']}"
        else:
            mode = ""
        print(f"-> {result['image_path']}  ({result['width']}x{result['height']}, seed={result['seed']}{mode})")
        if result["enhanced_prompt"]:
            print(f"   enhanced: {result['enhanced_prompt']}")


if __name__ == "__main__":
    main()
