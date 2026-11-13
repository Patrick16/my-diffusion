"""GPU check of outpainting and background removal in reference editing.

    PYTHONPATH=src python scripts/check_outpaint.py [stages...] [cat=PATH] [cut=0]

Stages (default: all, in this order; without "make" the pictures of an earlier run are reused). One model load, then (turbo, seed 42 unless noted, pictures in outputs/check_outpaint_*.png):
    make      - text-to-image: the scene (an empty pier, 480 16:9) and a person on a busy market street (480 3:4)
    cut       - background removal of the person and the cat (cat=PATH, default outputs/check_enhance_cat.png):
                before / after side by side, CPU time
    a0 / a    - "scene" mode, [scene, person], remove_background none / subjects
    b         - "subjects" mode, [person, cat], remove_background subjects
    c         - "scene" mode, [scene, cat], outpaint 25% right + remove_background subjects (480: two references);
                cut=0 runs it without background removal
    d         - one reference [scene], outpaint 20% top + 30% left, at 720
    missing   - remove_background with a missing model file: FileNotFoundError before any GPU work

For the outpaint runs, keep_mad = mean absolute difference (0..255) between the result inside the source box and
reference 1 resized to that box: low means the scene was kept where it was placed."""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import torch
from PIL import Image

from my_diffusion import background, imaging
from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine

OUT = PROJECT_ROOT / "outputs"
GIB = 2**30
SCENE_PROMPT = ("A wide photo of an empty wooden pier on a calm mountain lake at golden hour, pine forest on the shore, "
                "no people, no animals.")
PERSON_PROMPT = ("Full-body photo of a young man in a red jacket and blue jeans standing in a crowded colorful street "
                 "market, many people, stalls and signs behind him, sharp background.")


def arg(name: str, default=None):
    return next((a[len(name) + 1:] for a in sys.argv[1:] if a.startswith(name + "=")), default)


def mad(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean())


def report(name: str, **row) -> None:
    print("RESULT " + json.dumps({"check": name, **row}), flush=True)


def run(engine: Krea2Engine, name: str, **kw) -> tuple[Image.Image, dict]:
    t0 = time.time()
    result = engine.generate(seed=kw.pop("seed", 42), **kw)
    peak = torch.cuda.max_memory_allocated() / GIB   # reset by the engine right before sampling
    image = Image.open(result["image_path"]).convert("RGB")
    path = OUT / f"check_outpaint_{name}.png"
    image.save(path)
    report(
        name, path=str(path), size=[result["width"], result["height"]], mode=result["mode"],
        reference_mode=result["reference_mode"], outpaint=result["outpaint"], remove_background=result["remove_background"],
        edit_lora_rank=result["edit_lora_rank"], wall=round(time.time() - t0, 2), peak_vram_gb=round(peak, 2),
        timings=result["timings"],
    )
    return image, result


def keep_mad(image: Image.Image, scene: Image.Image, margins, resolution: str) -> dict:
    """How well the scene survived inside its box (the box is recomputed exactly as the engine does)."""
    w, h, (left, top, bw, bh) = imaging.outpaint_frame(*scene.size, imaging.Margins(*margins), resolution)
    assert image.size == (w, h), (image.size, (w, h))
    inside = image.crop((left, top, left + bw, top + bh))
    return {"box": [left, top, bw, bh], "keep_mad": round(mad(inside, scene.resize((bw, bh), Image.LANCZOS)), 2)}


def side_by_side(*pictures: Image.Image, height: int = 480) -> Image.Image:
    scaled = [p.resize((round(p.width * height / p.height), height), Image.LANCZOS) for p in pictures]
    sheet = Image.new("RGB", (sum(p.width for p in scaled), height), "white")
    x = 0
    for p in scaled:
        sheet.paste(p, (x, 0))
        x += p.width
    return sheet


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    cat = Image.open(arg("cat", OUT / "check_enhance_cat.png")).convert("RGB")
    settings = Settings()
    engine = Krea2Engine(settings)
    t0 = time.time()
    engine.load()
    report("load", seconds=round(time.time() - t0, 1), checkpoint=engine.checkpoint_name, matmul=engine.matmul)

    stages = [a for a in sys.argv[1:] if "=" not in a] or ["make", "cut", "a", "b", "c", "d", "missing"]
    if "make" in stages:
        scene, _ = run(engine, "make_scene", prompt=SCENE_PROMPT, resolution="480", aspect_ratio="16:9", seed=7)
        person, _ = run(engine, "make_person", prompt=PERSON_PROMPT, resolution="480", aspect_ratio="3:4", seed=8)
    else:   # the pictures of an earlier run
        scene = Image.open(OUT / "check_outpaint_make_scene.png").convert("RGB")
        person = Image.open(OUT / "check_outpaint_make_person.png").convert("RGB")
    cut = arg("cut", "1") == "1"   # cut=0: the outpaint stage c without background removal (no onnxruntime needed)

    for name, picture in (("person", person), ("cat", cat)) if "cut" in stages else ():
        t0 = time.time()
        cutout = background.remove_background(picture, path=settings.rembg_model_path)
        seconds = time.time() - t0
        side_by_side(picture, cutout).save(OUT / f"check_outpaint_cut_{name}.png")
        report(f"cut_{name}", path=str(OUT / f"check_outpaint_cut_{name}.png"), size=list(picture.size),
               cpu_seconds=round(seconds, 2))

    two = "The man from image 2 stands on the wooden pier in image 1, full body, natural light."
    if "a" in stages:
        run(engine, "a0_scene_two_refs_nocut", prompt=two, resolution="480", aspect_ratio="16:9", references=[scene, person])
        run(engine, "a_scene_two_refs_cut", prompt=two, resolution="480", aspect_ratio="16:9", references=[scene, person],
            remove_background="subjects")
    if "b" in stages:
        run(engine, "b_subjects_two_refs_cut",
            prompt="The man from image 1 holds the cat from image 2 in his arms, standing in a cozy living room.",
            resolution="480", aspect_ratio="16:9", references=[person, cat], reference_mode="subjects",
            remove_background="subjects")

    if "c" in stages:
        margins_c = (0, 0, 0, 25)
        image, _ = run(engine, f"c_outpaint_right25_two_refs_{'cut' if cut else 'nocut'}",
                       prompt="The cat from image 2 sits on the wooden pier in image 1. Extend the lake and the forest to the right.",
                       resolution="480", references=[scene, cat], remove_background="subjects" if cut else "none",
                       outpaint=margins_c)
        report("c_keep", **keep_mad(image, scene, margins_c, "480"))

    if "d" in stages:
        margins_d = (20, 0, 30, 0)
        image, _ = run(engine, "d_outpaint_top20_left30_one_ref",
                       prompt="Extend the scene: more evening sky above and more lake and pine forest on the left.",
                       resolution="720", references=[scene], outpaint=margins_d)
        report("d_keep", **keep_mad(image, scene, margins_d, "720"))

    if "missing" not in stages:
        return
    settings.rembg_model = "rembg/missing.onnx"
    t0 = time.time()
    try:
        engine.generate(prompt=two, resolution="480", references=[scene, person], remove_background="subjects")
        report("missing", ok=False, error=None)
    except FileNotFoundError as e:
        report("missing", ok=True, seconds=round(time.time() - t0, 3), error=str(e))


if __name__ == "__main__":
    main()
