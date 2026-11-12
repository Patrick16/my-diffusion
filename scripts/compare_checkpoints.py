"""How do the checkpoints differ from each other, layer by layer - and is the difference low-rank?

Why this matters. Fine-tuned checkpoints of one model are usually "the base plus a correction". If that
correction is low-rank (as it is when a LoRA was merged into the weights), a checkpoint could be stored and
loaded as the base plus a small file instead of another 13 GB - switching checkpoints would take seconds
instead of minutes. If the correction is spread over the whole matrix, that idea does not work.

What the script does. For a few sample layers it unpacks the weights of every checkpoint and of the base
one, takes the difference D = W - W_base and looks at its singular values:

    diff/base      how large the difference is:  |D| / |W_base|
    rank for 90%   how many singular directions hold 90% of the difference; a low-rank difference
                   needs a few dozen, a difference spread over the whole matrix needs thousands
    share at r     the part of the difference captured by the r strongest directions
    left at r      what remains after removing those r directions, relative to |W_base|

The checkpoints are stored in different 8-bit formats, and rounding alone makes two files of the *same*
weights differ. So the last block is a control: the base checkpoint re-rounded into the other formats.
That is what "no real difference, only rounding" looks like - compare the real rows against it.

Reads only the sample layers (a few hundred MB per file), so it does not wait for whole checkpoints.

    uv run python scripts/compare_checkpoints.py [--base NAME] [--layers LAYER ...] [--ranks R ...]
"""

from __future__ import annotations

import argparse
import time

import torch

from my_diffusion.config import Settings
from my_diffusion.weights import Checkpoint, QLinear, read_checkpoint

DEFAULT_LAYERS = [
    "blocks.0.attn.wq", "blocks.0.mlp.up",
    "blocks.13.attn.wq", "blocks.13.attn.wo", "blocks.13.mlp.up", "blocks.13.mlp.down",
    "blocks.27.attn.wq", "blocks.27.mlp.up",
]


def dense(ckpt: Checkpoint, layer: str, device: str) -> torch.Tensor:
    """The layer's weights unpacked to float32, by the same code the engine uses."""
    weight = ckpt.tensors[f"{layer}.weight"]
    linear = QLinear(weight.shape[1], weight.shape[0], bias=False)
    linear.set_weight(weight, ckpt.formats.get(layer, "dense"), ckpt.tensors.get(f"{layer}.weight_scale"), None)
    return linear.to(device).dense_weight(torch.float32)


def spectrum(diff: torch.Tensor) -> torch.Tensor:
    """Squared singular values of diff, largest first. Computed as the eigenvalues of the small Gram
    matrix - much cheaper than a full SVD of a 16384 x 6144 matrix, and we only need the energies."""
    gram = diff.T @ diff if diff.shape[0] >= diff.shape[1] else diff @ diff.T
    return torch.linalg.eigvalsh(gram.double()).clamp_(min=0).flip(0)


def describe(diff: torch.Tensor, base_norm: float, ranks: list[int]) -> str:
    energy = spectrum(diff)
    total = energy.sum()
    captured = energy.cumsum(0) / total
    rank90 = int((captured < 0.9).sum().item()) + 1
    relative = (total.sqrt() / base_norm).item()
    shares = " / ".join(f"{captured[r - 1].item() * 100:4.1f}%" for r in ranks)
    left = " / ".join(f"{relative * (1 - captured[r - 1].item()) ** 0.5 * 100:4.2f}%" for r in ranks)
    return f"{relative * 100:8.2f}% | {rank90:5d} of {len(energy):5d} | {shares} | {left}"


def requantized(weight: torch.Tensor, fmt: str) -> torch.Tensor:
    """The same weights rounded into another 8-bit format and unpacked back (for the control block)."""
    if fmt == "int8 per row":
        scale = weight.abs().amax(dim=1, keepdim=True) / 127
        return (weight / scale).round().clamp(-127, 127) * scale
    if fmt == "int8 per block of 32":
        blocks = weight.view(weight.shape[0], -1, 32)
        scale = blocks.abs().amax(dim=-1, keepdim=True) / 127
        return ((blocks / scale).round().clamp(-127, 127) * scale).view_as(weight)
    if fmt == "fp8, one scale":
        scale = weight.abs().max() / 448
        return (weight / scale).to(torch.float8_e4m3fn).float() * scale
    if fmt == "fp8, no scale":
        # weights are small numbers, and fp8 is coarse near zero, so this format rounds the hardest
        return weight.to(torch.float8_e4m3fn).float()
    raise ValueError(fmt)


def main() -> None:
    settings = Settings()
    folder = settings.models_dir / "checkpoints"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=settings.checkpoint, help="the checkpoint the others are compared against")
    ap.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 64, 256, 1024])
    args = ap.parse_args()

    device = settings.device
    names = sorted(p.name for pattern in ("*.safetensors", "*.gguf") for p in folder.glob(pattern))
    others = [n for n in names if n != args.base]
    layers = set(args.layers)
    header = f"{'layer':20s} {'shape':>11s} | diff/base | rank for 90%   | share at r = {', '.join(map(str, args.ranks))} | left at the same r (of base)"

    with torch.inference_mode():
        t0 = time.time()
        base_ckpt = read_checkpoint(folder / args.base, layers=layers)
        base = {layer: dense(base_ckpt, layer, device).cpu() for layer in args.layers}
        print(f"base: {args.base} ({time.time() - t0:.0f}s to read {len(args.layers)} layers)")

        middle = args.layers[len(args.layers) // 2]
        seen = {args.base: base[middle]}
        for name in others:
            t0 = time.time()
            ckpt = read_checkpoint(folder / name, layers=layers)
            formats = sorted({ckpt.formats.get(layer, "dense") + ":" + str(ckpt.tensors[f"{layer}.weight"].dtype).split(".")[-1] for layer in args.layers})
            print(f"\n== {name}  (stored as {', '.join(formats)}; read in {time.time() - t0:.0f}s)")
            print(header)
            for layer in args.layers:
                weight = dense(ckpt, layer, device)
                reference = base[layer].to(device)
                shape = f"{weight.shape[0]}x{weight.shape[1]}"
                print(f"{layer:20s} {shape:>11s} | {describe(weight - reference, reference.norm().item(), args.ranks)}")
                if layer == middle:
                    seen[name] = weight.cpu()
            del ckpt

        print(f"\n== control: {args.base} re-rounded into other formats (rounding only, no real difference)")
        print(header)
        for layer in (args.layers[0], middle):
            reference = base[layer].to(device)
            for fmt in ("int8 per row", "int8 per block of 32", "fp8, one scale", "fp8, no scale"):
                label = f"{layer} -> {fmt}"
                print(f"{label:46s} | {describe(requantized(reference, fmt) - reference, reference.norm().item(), args.ranks)}")

        print(f"\n== every pair, layer {middle}: |A - B| / |B|, in % (rows: A, columns: B)")
        order = [args.base] + others
        print(" " * 30 + "".join(f"{i:>8d}" for i in range(len(order))))
        for i, a in enumerate(order):
            row = "".join(f"{((seen[a].to(device) - seen[b].to(device)).norm() / seen[b].to(device).norm()).item() * 100:8.2f}" for b in order)
            print(f"{i} {a[:27]:27s} {row}")


if __name__ == "__main__":
    main()
