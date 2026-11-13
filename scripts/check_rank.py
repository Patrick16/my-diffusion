"""GPU check: the edit LoRA at rank 256 vs rank 128 on the same edits (same seed, prompt, reference, size).

    uv run python scripts/check_rank.py [frac=0.97]

The engine picks the rank with engine.edit_lora_rank(w, h, n_refs); here it is replaced by a forced value, so each
case runs twice, once per rank, with everything else identical. Reuses check_edit.py's run(), mad() and tmad().
Cases: fox 848x480, cat 848x480, fox 736x560, fox 1280x720 (rank 256 at 720 runs last: it may OOM).
Pictures: outputs/check_rank_<subject>_<size>_r<rank>.png.
mad = mean absolute pixel difference 0..255; thumb = the same on 32x18 thumbnails (composition)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_edit as ce   # noqa: E402  (also wraps apply_loras to measure the attach peak)

import my_diffusion.engine as engine_module   # noqa: E402
from my_diffusion.config import Settings   # noqa: E402
from my_diffusion.engine import Krea2Engine   # noqa: E402
from my_diffusion.imaging import cover_crop   # noqa: E402

FORCED: dict[str, int] = {"rank": 256}
engine_module.edit_lora_rank = lambda width, height, references: FORCED["rank"]

# (subject, size label, prompt, resolution, aspect)
CASES = [
    ("fox", "848x480", ce.FOX_PROMPT, "480", "16:9"),
    ("cat", "848x480", ce.CAT_PROMPT, "480", "16:9"),
    ("fox", "736x560", ce.FOX_PROMPT, "480", "4:3"),
    ("fox", "1280x720", ce.FOX_PROMPT, "720", "16:9"),
]
# rank 256 first for the 480-tier cases, then all rank 128, then rank 256 at 720 (the one that may fail)
ORDER = [(c, 256) for c in CASES[:3]] + [(c, 128) for c in CASES] + [(CASES[3], 256)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    torch.cuda.set_per_process_memory_fraction(float(ce.arg("frac", "0.97")))
    engine = Krea2Engine(Settings(checkpoint="krea2_turbo_mxfp8.safetensors", matmul="fp8_all"))
    t0 = time.time()
    engine.load()
    print(f"engine loaded in {time.time() - t0:.0f}s: {engine.checkpoint_name}, matmul {engine.matmul}", flush=True)
    refs = {"fox": Image.open(ce.FOX).convert("RGB"), "cat": Image.open(ce.CAT).convert("RGB")}

    pictures: dict[tuple[str, str, int], Image.Image] = {}
    for (subject, size, prompt, resolution, aspect), rank in ORDER:
        FORCED["rank"] = rank
        name = f"{subject}_{size}_r{rank}"
        row = ce.run(engine, name, prompt, [refs[subject]], resolution, aspect)
        if "image" in row:
            # ce.run saves as check_edit_<name>.png; rename to this check's naming
            src = Path(row["png"])
            dst = ce.OUT / f"check_rank_{name}.png"
            dst.unlink(missing_ok=True)
            src.rename(dst)
            pictures[(subject, size, rank)] = row["image"]
            print(f"saved {dst} (engine reported rank {row['edit_lora_rank']})", flush=True)

    for subject, size, *_ in CASES:
        a, b = pictures.get((subject, size, 256)), pictures.get((subject, size, 128))
        res = dict(check="pair", subject=subject, size=size, have_r256=a is not None, have_r128=b is not None)
        ref = refs[subject]
        for rank, img in ((256, a), (128, b)):
            if img is not None:
                frame = cover_crop(ref, *img.size)
                res[f"mad_r{rank}_vs_ref"] = round(ce.mad(img, frame), 2)
                res[f"thumb_r{rank}_vs_ref"] = round(ce.tmad(img, frame), 2)
        if a is not None and b is not None:
            res["mad_r128_vs_r256"] = round(ce.mad(a, b), 2)
            res["thumb_r128_vs_r256"] = round(ce.tmad(a, b), 2)
        print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
