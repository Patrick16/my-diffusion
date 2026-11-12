"""Step 4 check: whether all LoRAs from models/loras can be read by our loader.

The model is not loaded onto the GPU and the checkpoint weights are not read - only the
names and sizes of the layers are needed, so the check is fast.

    uv run python scripts/check_loras.py"""

from __future__ import annotations

from collections import Counter

from my_diffusion.config import Settings
from my_diffusion.lora import load_lora
from my_diffusion.transformer import Krea2Transformer


def main() -> None:
    settings = Settings()
    model = Krea2Transformer()   # empty: only the layer structure
    root = settings.lora_dir.resolve()
    ok, failed = [], []
    for path in sorted(root.rglob("*.safetensors")):
        rel = path.relative_to(root).as_posix()
        try:
            lora = load_lora(path, model)
            ranks = Counter(down.shape[0] for down, _, _ in lora.layers.values())
            scales = sorted({round(s, 3) for _, _, s in lora.layers.values()})
            ok.append((rel, len(lora.layers), len(lora.diffs), dict(ranks), scales))
        except Exception as exc:  # noqa: BLE001
            failed.append((rel, f"{type(exc).__name__}: {exc}"))

    print(f"readable: {len(ok)}, not readable: {len(failed)}\n")
    summary = Counter((n_layers, tuple(sorted(ranks.items())), tuple(scales)) for _, n_layers, _, ranks, scales in ok)
    print("summary of the readable files (layers, ranks, alpha/r multipliers) -> number of files:")
    for (n_layers, ranks, scales), count in summary.most_common():
        print(f"  {count:3d} x  layers {n_layers}, ranks {dict(ranks)}, alpha/r {list(scales)}")
    print("\nnot readable:")
    for rel, err in failed:
        print(f"  {rel}\n      {err[:230]}")


if __name__ == "__main__":
    main()
