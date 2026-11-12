"""Step 1 check: whether we read the compressed weights correctly.

Compares our unpacking of mxfp8 with a reference (computed by the ComfyUI library and
saved to a file), and also loads the whole checkpoint into the model in strict mode.

    uv run python scripts/check_weights.py [reference.pt]"""

from __future__ import annotations

import sys
import time

import torch

from my_diffusion.config import Settings
from my_diffusion.transformer import Krea2Transformer
from my_diffusion.weights import load_into, read_checkpoint


def main() -> None:
    settings = Settings()
    path = settings.checkpoint_path
    t0 = time.time()
    ckpt = read_checkpoint(path)
    print(f"read {path.name}: {len(ckpt.tensors)} tensors, quantized layers {len(ckpt.formats)}, {time.time() - t0:.1f}s")

    model = Krea2Transformer()
    load_into(model, ckpt)
    print("strict load into the model: OK (every tensor found its place)")

    if len(sys.argv) > 1:
        reference = torch.load(sys.argv[1], weights_only=True)  # the file contains only tensors
        modules = dict(model.named_modules())
        for name, ref in reference.items():
            ours = modules[name].dense_weight(torch.float32)
            diff = (ours - ref).abs().max().item()
            print(f"  {name}: max deviation from the ComfyUI reference = {diff:.3e}  (weights std {ref.std().item():.4f})")
            assert diff == 0.0, "unpacking does not match the reference"
        print("mxfp8 unpacking matches ComfyUI bit for bit")


if __name__ == "__main__":
    main()
