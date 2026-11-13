"""GPU check of the BFS face/head and body swap LoRAs (Krea 2) on people.

    PYTHONPATH=src python scripts/check_swap.py [stages...]

Stages (default: all; the people are generated once and reused by later runs; pictures in outputs/check_swap_*.png):
    make      - text-to-image: person A (a man in a red jacket, full body) and person B (a woman in a white
                shirt, full body, plain background); 480, 3:4
    head      - head swap: scene = A, reference 2 = B, LoRA krea2_bfs_v1.1 (trigger "head_swap: ...")
    head_base - the same request without the swap LoRA (only the edit LoRA): what the prompt alone does
    body      - body swap: scene = A, reference 2 = B, LoRA krea2_bodyswap_v1 (trigger "body_swap: ...")
    body_base - the same request without the swap LoRA

Same seed in every pair, so the only difference between "X" and "X_base" is the swap LoRA.
Also prints the LoRA file check (identical files by sha256) and the report of every run as RESULT lines."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time

from PIL import Image

from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine

OUT = PROJECT_ROOT / "outputs"
LORAS = PROJECT_ROOT / "models" / "loras"
PERSON_A = ("Full-body photo of a young man in a red jacket and blue jeans standing in a crowded colorful street "
            "market, many people, stalls and signs behind him, sharp background.")
PERSON_B = ("Full-body photo of a young woman with curly dark hair, a white t-shirt and grey trousers, standing "
            "against a plain light grey wall, soft even light.")
HEAD_PROMPT = ("head_swap: replace the head with the reference head. Image 1 is the scene: put the head of the "
               "woman from image 2 on the man in image 1, keep his body, pose, clothes and the market behind him.")
BODY_PROMPT = ("body_swap: replace the person with the reference person. Image 1 is the scene: the woman from "
               "image 2 stands in the market of image 1 in the same pose, keep the background.")


def report(name: str, **row) -> None:
    print("RESULT " + json.dumps({"check": name, **row}, ensure_ascii=False), flush=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(engine: Krea2Engine, name: str, **kw) -> Image.Image:
    t0 = time.time()
    result = engine.generate(**kw)
    image = Image.open(result["image_path"]).convert("RGB")
    path = OUT / f"check_swap_{name}.png"
    image.save(path)
    report(name, path=str(path), mode=result["mode"], loras=result["loras"],
           edit_lora_rank=result["edit_lora_rank"], seed=result["seed"], wall=round(time.time() - t0, 2),
           timings=result["timings"])
    return image


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)

    # The two LoRA names from the request: are they the same file?
    head_a, body_a = LORAS / "krea2_bfs_v1.1.safetensors", LORAS / "krea2_bodyswap_v1.safetensors"
    report("files", bfs_v1_1=sha256(head_a)[:16], bodyswap_v1=sha256(body_a)[:16],
           bfs_head_swap_v1_1_same=sha256(LORAS / "bfs_head_swap_v1.1_krea2.safetensors") == sha256(head_a),
           bfs_body_swap_same=sha256(LORAS / "bfs_body_swap_v1_krea2.safetensors") == sha256(body_a))

    settings = Settings()
    engine = Krea2Engine(settings)
    t0 = time.time()
    engine.load()
    report("load", seconds=round(time.time() - t0, 1), checkpoint=engine.checkpoint_name, matmul=engine.matmul)

    stages = [a for a in sys.argv[1:] if "=" not in a] or ["make", "head", "head_base", "body", "body_base"]
    if "make" in stages:
        a = run(engine, "make_person_a", prompt=PERSON_A, resolution="480", aspect_ratio="3:4", seed=8)
        b = run(engine, "make_person_b", prompt=PERSON_B, resolution="480", aspect_ratio="3:4", seed=9)
    else:
        a = Image.open(OUT / "check_swap_make_person_a.png").convert("RGB")
        b = Image.open(OUT / "check_swap_make_person_b.png").convert("RGB")

    common = dict(resolution="480", aspect_ratio="auto", references=[a, b], reference_mode="scene", seed=42)
    if "head" in stages:
        run(engine, "head_bfs", prompt=HEAD_PROMPT, loras=[("krea2_bfs_v1.1", 1.0)], **common)
    if "head_base" in stages:
        run(engine, "head_base", prompt=HEAD_PROMPT, **common)
    if "body" in stages:
        run(engine, "body_swap", prompt=BODY_PROMPT, loras=[("krea2_bodyswap_v1", 1.0)], **common)
    if "body_base" in stages:
        run(engine, "body_base", prompt=BODY_PROMPT, **common)


if __name__ == "__main__":
    main()
