"""GPU check of outpainting with keep_original (reference 1's pixels kept, only the margins generated).

    PYTHONPATH=src python scripts/check_keep_original.py [stages...] fox=PATH cat=PATH [blend=N]

Stages (default: all; turbo, seed 42, pictures in outputs/check_keep_*.png):
    fox    - the fox extended 25% right at 480, without and with keep_original (same seed)
    two    - [fox, cat], 25% right at 480, keep_original (two references)
    sides  - the fox extended 15% top, 10% bottom, 20% left, 20% right at 720, without and with keep_original
blend=N overrides engine.KEEP_BLEND (the mixed band at the box's inner edges, in latent cells) to compare values.

Numbers per outpaint run (mean absolute difference, 0..255, against reference 1 resized to its box):
    mad_box    - over the whole box
    mad_kept   - over the box shrunk by the blend band at its inner edges (the area held exactly to the original)
    mad_band   - over the band only
    vae_floor  - the same, kept area, for a plain VAE encode -> decode of the canvas: the best keep_original can do
The sheet next to each picture (…_sheet.png) shows the result with the box outlined and a 3x crop across the seam."""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from my_diffusion import engine as engine_module
from my_diffusion import imaging
from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine

OUT = PROJECT_ROOT / "outputs"
FOX_PROMPT = "Extend the snowy landscape around the fox: more snow, soft golden light, the same style."
TWO_PROMPT = "The cat from image 2 sits in the snow next to the fox from image 1. Extend the snowy landscape."


def arg(name: str, default=None):
    return next((a[len(name) + 1:] for a in sys.argv[1:] if a.startswith(name + "=")), default)


def report(name: str, **row) -> None:
    print("RESULT " + json.dumps({"check": name, **row}), flush=True)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return round(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), 2)


def masks(frame_w: int, frame_h: int, box, blend: int):
    """Pixel masks of the box and of the area held exactly (weight 0), from the engine's own weight map."""
    w = imaging.outpaint_keep_weight(frame_w, frame_h, box, blend)[0, 0].numpy()
    w = np.kron(w, np.ones((8, 8)))
    left, top, bw, bh = box
    in_box = np.zeros((frame_h, frame_w), bool)
    in_box[top:top + bh, left:left + bw] = True
    kept = in_box & (w == 0)
    return in_box, kept, in_box & ~kept


def measure(image: Image.Image, scene: Image.Image, box, blend: int, floor: Image.Image | None) -> dict:
    fw, fh = image.size
    left, top, bw, bh = box
    canvas = np.asarray(imaging.place_in_frame(scene.convert("RGB").resize((bw, bh), Image.LANCZOS), fw, fh, box))
    got = np.asarray(image)
    in_box, kept, band = masks(fw, fh, box, blend)
    row = {"box": list(box), "mad_box": mad(got[in_box], canvas[in_box]), "mad_kept": mad(got[kept], canvas[kept]),
           "mad_band": mad(got[band], canvas[band]) if band.any() else None}
    if floor is not None:
        row["vae_floor"] = mad(np.asarray(floor)[kept], canvas[kept])
    return row


def sheet(image: Image.Image, box, path) -> None:
    """The result with the box outlined, plus 3x crops across the inner edges."""
    left, top, bw, bh = box
    outlined = image.copy()
    ImageDraw.Draw(outlined).rectangle((left, top, left + bw - 1, top + bh - 1), outline=(255, 0, 255))
    crops = []
    fw, fh = image.size
    if left + bw < fw:   # right edge
        x, y = left + bw, top + bh // 2
        crops.append(image.crop((x - 48, y - 48, x + 48, y + 48)))
    if left > 0:
        x, y = left, top + bh // 2
        crops.append(image.crop((x - 48, y - 48, x + 48, y + 48)))
    if top > 0:
        x, y = left + bw // 2, top
        crops.append(image.crop((x - 48, y - 48, x + 48, y + 48)))
    crops = [c.resize((288, 288), Image.NEAREST) for c in crops]
    out = Image.new("RGB", (outlined.width + 288 * len(crops), max(outlined.height, 288)), "white")
    out.paste(outlined, (0, 0))
    for k, c in enumerate(crops):
        out.paste(c, (outlined.width + 288 * k, 0))
    out.save(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    fox = Image.open(arg("fox", OUT / "check_ckpt_krea2_turbo_mxfp8.png")).convert("RGB")
    cat = Image.open(arg("cat", OUT / "check_enhance_cat.png")).convert("RGB")
    if arg("blend") is not None:
        engine_module.KEEP_BLEND = int(arg("blend"))
    blend = engine_module.KEEP_BLEND
    engine = Krea2Engine(Settings())
    t0 = time.time()
    engine.load()
    report("load", seconds=round(time.time() - t0, 1), blend=blend)

    def vae_floor(scene: Image.Image, margins, resolution: str) -> Image.Image:
        fw, fh, box = imaging.outpaint_frame(*scene.size, imaging.Margins(*margins), resolution)
        canvas = imaging.place_in_frame(scene.resize(box[2:], Image.LANCZOS), fw, fh, box)
        engine.gpu.use("vae")
        return engine.vae.decode(engine.vae.encode(canvas))

    def run(name: str, scene: Image.Image, margins, resolution: str, floor, **kw) -> None:
        t0 = time.time()
        result = engine.generate(prompt=kw.pop("prompt", FOX_PROMPT), seed=42, resolution=resolution, outpaint=margins, **kw)
        image = Image.open(result["image_path"]).convert("RGB")
        path = OUT / f"check_keep_{name}.png"
        image.save(path)
        _, _, box = imaging.outpaint_frame(*scene.size, imaging.Margins(*margins), resolution)
        sheet(image, box, OUT / f"check_keep_{name}_sheet.png")
        report(name, path=str(path), size=[result["width"], result["height"]], keep_original=result["keep_original"],
               references=result["references"], edit_lora_rank=result["edit_lora_rank"], wall=round(time.time() - t0, 1),
               **measure(image, scene, box, blend, floor))

    stages = [a for a in sys.argv[1:] if "=" not in a] or ["fox", "two", "sides"]
    if "fox" in stages:
        m = (0, 0, 0, 25)
        floor = vae_floor(fox, m, "480")
        run("fox_right25_off", fox, m, "480", floor, references=[fox])
        run(f"fox_right25_keep_b{blend}", fox, m, "480", floor, references=[fox], keep_original=True)
    if "two" in stages:
        m = (0, 0, 0, 25)
        run(f"two_refs_right25_keep_b{blend}", fox, m, "480", vae_floor(fox, m, "480"), references=[fox, cat],
            keep_original=True, prompt=TWO_PROMPT)
    if "sides" in stages:
        m = (15, 10, 20, 20)
        floor = vae_floor(fox, m, "720")
        run("fox_sides_off", fox, m, "720", floor, references=[fox])
        run(f"fox_sides_keep_b{blend}", fox, m, "720", floor, references=[fox], keep_original=True)


if __name__ == "__main__":
    main()
