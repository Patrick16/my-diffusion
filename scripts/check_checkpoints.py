"""Step 5 check: whether all checkpoints from models/checkpoints can be read and produce an image.

For each file: a strict load, one 736x560 generation with the default settings
for its kind, saved to outputs/check_ckpt_<name>.png. Look at the result with your own eyes:
noise or mush means the weight format was read incorrectly.

    uv run python scripts/check_checkpoints.py [file_name ...]"""

from __future__ import annotations

import logging
import shutil
import sys
import time

from my_diffusion.config import Settings
from my_diffusion.engine import Krea2Engine


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = Settings()
    engine = Krea2Engine(settings)
    names = sys.argv[1:] or engine.list_files("checkpoints")

    engine.load()
    for name in names:
        try:
            t0 = time.time()
            if engine.checkpoint_name != name:
                engine.load_checkpoint(name)
            formats = sorted({m.fmt for m in engine.model.modules() if hasattr(m, "fmt")})
            result = engine.generate(
                "a fox walking in fresh snow at golden hour, natural light, detailed fur",
                seed=42, resolution="480", aspect_ratio="4:3",
            )
            out = settings.output_dir / f"check_ckpt_{name.rsplit('.', 1)[0]}.png"
            shutil.move(result["image_path"], out)
            print(f"OK   {name}: layer formats {formats}, kind {engine.kind}, {result['steps']} steps, "
                  f"sampling {result['timings']['sample']}s, total with loading {time.time() - t0:.0f}s -> {out.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
