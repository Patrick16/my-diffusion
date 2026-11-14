"""GPU check of the inpaint refinement (inpaint_iterations, langevin.py) against the plain inpaint.

    PYTHONPATH=src python scripts/check_inpaint_iterations.py small=PATH big=PATH [case=head,remove,half,band]
                                                              [seeds=42,43] [iterations=0,2,5] [feather=2]

small - a 4:3 picture with one subject on the left two thirds (the checks were made with a 736x560 fox walking
        toward the camera); big - a 16:9 side view of a subject that crosses the middle of the frame (a 1920x1088
        fox). Both are cropped to the run's size. The masks are fixed shapes in the frame, so other pictures
        work only if the subject sits in the same place.

Cases (turbo defaults, strength 1.0):
    head    - small, 480 4:3: an ellipse over the head; the new head must join the body that is kept
    remove  - small, 480 4:3: an ellipse over the whole subject; background must continue through the hole
    half    - big, 720 16:9: the right half of the frame; the rear of the subject is completed from its front
    band    - big, 720 16:9: a vertical band through the body; it must connect on both sides

Per run: RESULT {case, seed, iterations, sample_s, calls} and a picture outputs/check_inpaint_iter_<case>_s<seed>_n<N>.png.
Per case and seed: a sheet (…_sheet.png) with the source, the mask outline and every variant side by side.
What to look for: a second head or an extra leg (the plain inpaint on half / band), a separate picture inside
the mask (many iterations on remove). There is no number for that - the sheets are judged by eye."""

from __future__ import annotations

import json
import logging
import sys

import torch
from PIL import Image, ImageDraw, ImageFilter

from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine
from my_diffusion.imaging import cover_crop
from my_diffusion.langevin import network_calls

OUT = PROJECT_ROOT / "outputs"


def arg(name: str, default: str | None = None) -> str | None:
    return next((a[len(name) + 1:] for a in sys.argv[1:] if a.startswith(name + "=")), default)


def report(**row) -> None:
    print("RESULT " + json.dumps(row), flush=True)


def shape_mask(size: tuple[int, int], box: tuple[float, float, float, float], ellipse: bool) -> Image.Image:
    """White shape on black; box is (left, top, right, bottom) in fractions of the frame."""
    w, h = size
    pixels = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    (draw.ellipse if ellipse else draw.rectangle)(pixels, fill=255)
    return mask


def cases(small: Image.Image | None, big: Image.Image | None) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if small is not None:
        frame = cover_crop(small.convert("RGB"), 736, 560)
        found["head"] = dict(image=frame, mask=shape_mask(frame.size, (0.15, 0.07, 0.53, 0.62), True), resolution="480", aspect_ratio="4:3",
                             prompt="A red fox wearing a blue knitted winter hat walks through the snow in golden evening light.")
        found["remove"] = dict(image=frame, mask=shape_mask(frame.size, (0.15, 0.07, 0.98, 0.97), True), resolution="480", aspect_ratio="4:3",
                               prompt="An empty snowy field in golden evening light, soft blue shadows on the snow.")
    if big is not None:
        frame = cover_crop(big.convert("RGB"), 1280, 720)
        found["half"] = dict(image=frame, mask=shape_mask(frame.size, (0.5, 0.0, 1.0, 1.0), False), resolution="720", aspect_ratio="16:9",
                             prompt="A red fox walking through the snow in golden evening light, side view, bushy tail.")
        found["band"] = dict(image=frame, mask=shape_mask(frame.size, (0.4375, 0.0, 0.625, 1.0), False), resolution="720", aspect_ratio="16:9",
                             prompt="A red fox walking through the snow in golden evening light, side view.")
    return found


def sheet(case: dict, images: dict[str, Image.Image], path) -> None:
    source = case["image"].copy()
    outline = case["mask"].point(lambda p: 255 if p > 127 else 0).filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p else 0)
    source.paste(Image.new("RGB", source.size, (255, 0, 255)), (0, 0), outline)
    tiles = [("source + mask", source), *images.items()]
    w, h = source.size
    canvas = Image.new("RGB", (len(tiles) * w, h + 18), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for k, (name, image) in enumerate(tiles):
        draw.text((k * w + 4, 3), name, fill=(255, 255, 255))
        canvas.paste(image, (k * w, 18))
    canvas.save(path)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    small, big = arg("small"), arg("big")
    if small is None and big is None:
        sys.exit(__doc__)
    found = cases(None if small is None else Image.open(small), None if big is None else Image.open(big))
    wanted = (arg("case") or ",".join(found)).split(",")
    seeds = [int(v) for v in arg("seeds", "42,43").split(",")]
    variants = [int(v) for v in arg("iterations", "0,2,5").split(",")]
    feather = int(arg("feather", "2"))
    torch.cuda.set_per_process_memory_fraction(0.97)   # an overrun is an error, not a silent spill into shared memory
    engine = Krea2Engine(Settings())
    engine.load()
    engine.generate(prompt="warm-up", resolution="480", aspect_ratio="1:1", seed=1, steps=2)   # moves the network to the GPU
    for name in wanted:
        if name not in found:
            sys.exit(f"case {name!r} needs its picture: head / remove take small=PATH, half / band take big=PATH")
        case = found[name]
        for seed in seeds:
            images: dict[str, Image.Image] = {}
            for n in variants:
                result = engine.generate(
                    prompt=case["prompt"], resolution=case["resolution"], aspect_ratio=case["aspect_ratio"], seed=seed,
                    init_image=case["image"], mask=case["mask"], feather=feather, inpaint_iterations=n,
                    output_name=f"check_inpaint_iter_{name}_s{seed}_n{n}.png",
                )
                seconds = result["timings"]["sample"]
                report(case=name, seed=seed, iterations=n, size=[result["width"], result["height"]], sample_s=seconds,
                       calls=network_calls(result["steps"], n, from_pure_noise=True), file=result["image_path"])
                images[f"iterations {n}  {seconds:.1f}s"] = Image.open(result["image_path"]).convert("RGB")
            sheet(case, images, OUT / f"check_inpaint_iter_{name}_s{seed}_sheet.png")


if __name__ == "__main__":
    main()
