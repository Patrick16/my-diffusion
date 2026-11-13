"""Check of background removal (background.py) on a real picture. No GPU needed.

    uv run python scripts/check_background.py PATH [fill=R,G,B]

Writes outputs/check_background_before.png, check_background_mask.png and check_background_after.png
and prints the share of the picture kept as the subject and the time taken."""

from __future__ import annotations

import sys
import time

import numpy as np
from PIL import Image

from my_diffusion.background import remove_background, subject_mask
from my_diffusion.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs"


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" not in a]
    if not args:
        sys.exit(__doc__)
    fill = next((tuple(int(c) for c in a[5:].split(",")) for a in sys.argv[1:] if a.startswith("fill=")),
                (255, 255, 255))
    image = Image.open(args[0]).convert("RGB")
    OUT.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    mask = subject_mask(image)            # includes loading the session
    t1 = time.perf_counter()
    result = remove_background(image, fill)
    t2 = time.perf_counter()

    image.save(OUT / "check_background_before.png")
    mask.save(OUT / "check_background_mask.png")
    result.save(OUT / "check_background_after.png")
    kept = float((np.asarray(mask) > 127).mean())
    print(f"{image.size[0]}x{image.size[1]}  subject share {kept:.0%}  "
          f"first call {t1 - t0:.2f}s (with loading), second {t2 - t1:.2f}s  -> {OUT}")


if __name__ == "__main__":
    main()
