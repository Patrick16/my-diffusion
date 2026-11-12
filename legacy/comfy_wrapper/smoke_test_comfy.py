"""End-to-end test of the new ComfyUI-managed path: engine.load() starts
the ComfyUI subprocess and the local Qwen3-VL enhancer, engine.generate() runs
a real generation through the ComfyUI API and saves the image to outputs/.

    uv run python scripts/smoke_test_comfy.py"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_diffusion.config import Settings
from my_diffusion.engine import KreaEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    settings = Settings()
    engine = KreaEngine(settings)
    try:
        engine.load()
        result = engine.generate(
            prompt="a fox walking in fresh snow at golden hour, natural light, detailed fur",
            resolution="720",
            aspect_ratio="16:9",
            enhance=False,  # first check the generation path itself, without the extra variable
        )
        print("OK ->", result)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
