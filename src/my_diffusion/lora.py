"""LoRA: small add-on files that change the behavior of a large network.

What LoRA is. Instead of training the weight matrix W of a layer as a whole (6144x6144 numbers),
two narrow matrices are trained: down (r x input) and up (output x r), where r is the "rank", usually 16-128.
Their product up @ down is a correction to W. The file turns out hundreds of times smaller than the model.

How we apply it. We do not merge the correction into the weights; we compute it alongside (see QLinear):

    y = x @ W.T  +  strength * (x @ down.T) @ up.T

The "strength" is the LoRA weight from the request (1.0 - as trained, 0.5 - half strength, negative -
the opposite effect). The alpha/r correction multiplier stored in the file is also included in it.

The main hassle is names. Different training programs name the same layers differently;
a LoRA folder contains seven variants. Everything is brought to the "native" layer names
(blocks.0.attn.wq), after which the file is checked strictly: no key may remain
unrecognized, and the matrix sizes must match the model's layers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .pinned import pin_tensors
from .weights import QLinear, read_safetensors

logger = logging.getLogger(__name__)

# Wrappers that different tools prepend to the layer name.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "lora_unet_")

# Key suffixes -> tensor role.
_SUFFIXES = {
    ".lora_A.weight": "down", ".lora_down.weight": "down",
    ".lora_B.weight": "up", ".lora_up.weight": "up",
    ".alpha": "alpha",
    ".diff": "diff",
}

# Layer names in diffusers style -> native names.
_DIFFUSERS_SIMPLE = {
    "first": "img_in", "tmlp.0": "time_embed.linear_1", "tmlp.2": "time_embed.linear_2", "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1", "txtmlp.3": "txt_in.linear_2", "last.linear": "final_layer.linear",
}
_DIFFUSERS_LEAF = {
    "attn.wq": ["attn.to_q"], "attn.wk": ["attn.to_k"], "attn.wv": ["attn.to_v"], "attn.gate": ["attn.to_gate"],
    "attn.wo": ["attn.to_out.0", "attn.to_out"], "mlp.gate": ["ff.gate"], "mlp.up": ["ff.up"], "mlp.down": ["ff.down"],
}


def _aliases(native: str) -> list[str]:
    """All known spellings of a layer name."""
    names = [native]
    if native in _DIFFUSERS_SIMPLE:
        names.append(_DIFFUSERS_SIMPLE[native])
    else:
        head = native.replace("txtfusion.", "text_fusion.")
        if head.startswith("blocks."):
            head = "transformer_" + head
        if head != native:
            names.append(head)
        for leaf, variants in _DIFFUSERS_LEAF.items():
            if head.endswith("." + leaf):
                names += [head[: -len(leaf)] + v for v in variants]
    # kohya writes the name with underscores: blocks_0_attn_wq
    return names + [n.replace(".", "_") for n in names]


@dataclass
class LoraFile:
    name: str
    # native layer name -> (down, up, alpha/r)
    layers: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = field(default_factory=dict)
    # native layer name -> full correction to the matrix (".diff" format)
    diffs: dict[str, torch.Tensor] = field(default_factory=dict)
    # only in a copy made by truncate_rank: native layer name -> share of the correction's energy kept (0..1)
    kept_energy: dict[str, float] = field(default_factory=dict)
    # truncate_rank results for this file, by rank: they live exactly as long as the file itself
    truncated: dict[int, LoraFile] = field(default_factory=dict, repr=False)

    @property
    def worst_kept_energy(self) -> float:
        """The smallest kept energy share over the layers; 1.0 if nothing was truncated."""
        return min(self.kept_energy.values(), default=1.0)


def load_lora(path: Path, model: nn.Module) -> LoraFile:
    tensors, _ = read_safetensors(path)
    targets = {name: m for name, m in model.named_modules() if isinstance(m, QLinear)}
    alias_to_native = {alias: native for native in targets for alias in _aliases(native)}

    if any(".lokr_" in k or ".lora_mid" in k or ".hada_" in k for k in tensors):
        raise NotImplementedError(f"{path.name}: this is LoKr/LoHa, not a plain LoRA - this format is not supported yet")

    parts: dict[str, dict[str, torch.Tensor]] = {}
    unknown: list[str] = []
    for key, tensor in tensors.items():
        role = next((r for suffix, r in _SUFFIXES.items() if key.endswith(suffix)), None)
        if role is None:
            unknown.append(key)
            continue
        layer = key[: -len(next(s for s in _SUFFIXES if key.endswith(s)))]
        for prefix in _PREFIXES:
            if layer.startswith(prefix):
                layer = layer[len(prefix):]
                break
        native = alias_to_native.get(layer)
        if native is None:
            unknown.append(key)
            continue
        parts.setdefault(native, {})[role] = tensor

    if unknown:
        hint = "probably a file from another model or of another type" if len(unknown) > len(tensors) // 2 else "some keys were not recognized"
        raise ValueError(f"{path.name}: {hint}; unrecognized keys {len(unknown)} of {len(tensors)}, for example {unknown[:3]}")

    lora = LoraFile(name=path.stem)
    for native, p in parts.items():
        target = targets[native]
        if "diff" in p:
            if tuple(p["diff"].shape) != (target.out_features, target.in_features):
                raise ValueError(f"{path.name}: {native}.diff has shape {tuple(p['diff'].shape)}")
            lora.diffs[native] = p["diff"]
            continue
        if "down" not in p or "up" not in p:
            raise ValueError(f"{path.name}: layer {native} has no down/up pair")
        down, up = p["down"].flatten(1), p["up"].flatten(1)
        rank = down.shape[0]
        if down.shape[1] != target.in_features or tuple(up.shape) != (target.out_features, rank):
            raise ValueError(f"{path.name}: layer {native}: down {tuple(down.shape)}, up {tuple(up.shape)} do not fit {target.in_features}->{target.out_features}")
        scale = float(p["alpha"]) / rank if "alpha" in p else 1.0
        lora.layers[native] = (down, up, scale)
    return lora


def clear_loras(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, QLinear):
            m.adapters.clear()
            m.adapters_home.clear()
            m.deltas.clear()


def apply_loras(
    model: nn.Module, loras: list[tuple[LoraFile, float]], device: str, dtype: torch.dtype = torch.bfloat16,
    stream: bool = False,
) -> None:
    """Attaches a set of LoRAs to the model, replacing the previous set.

        Several LoRAs on one layer are concatenated into one pair of matrices: the downs are stacked
        one under another, the ups side by side, and the strength of each is pre-multiplied into its up. Mathematically
        it is the same sum of corrections, but in one multiplication instead of several.

        Why layer by layer, on the CPU. The pair for a layer is finished on the CPU (down cast to dtype; up taken
        to float32, multiplied by alpha/r * strength, then cast to dtype; the concatenation too), and only that
        finished pair goes to the device. Preparing everything on the GPU first would hold all the downs and
        all the float32 ups there at once until the concatenation - about 2.5x the LoRA size at the peak, ~4 GB
        for a rank-256 LoRA next to a 12.6 GB network. This way the peak is 1x the LoRA. The numbers are the
        same bit for bit: a float32 multiplication and a rounding to bf16 give the same IEEE result on any device.

        stream=True (settings.lora_streaming; only for a CUDA device): the finished pairs of the layers inside
        the model's streamed blocks (QLinear.streamable) are not sent to the GPU at all. They are packed into
        pinned RAM (QLinear.adapters_home), and each block copies its own pairs in right before it runs and
        drops them right after (weights.streamed_adapters). So the LoRA costs the GPU one block's pairs instead
        of the whole file (rank 128: ~30 MB instead of 0.85 GB), at the price of copying the whole LoRA over
        PCIe on every pass of the network. The pairs are the very same tensors as without streaming, so the
        result is identical. The other layers (and the ".diff" corrections) stay on the GPU as before."""
    clear_loras(model)
    targets = {name: m for name, m in model.named_modules() if isinstance(m, QLinear)}
    active = [(lora, strength) for lora, strength in loras if strength != 0]
    stream = stream and torch.device(device).type == "cuda"

    for lora, strength in active:
        for native, diff in lora.diffs.items():
            targets[native].deltas.append((diff.to(device, dtype), strength))

    streamed: list[tuple[QLinear, torch.Tensor, torch.Tensor]] = []
    # every layer that any of the LoRAs touches, in the order of first appearance
    for native in dict.fromkeys(native for lora, _ in active for native in lora.layers):
        downs, ups = [], []
        for lora, strength in active:
            if native in lora.layers:
                down, up, scale = lora.layers[native]
                downs.append(down.to(dtype))
                ups.append((up.float() * (scale * strength)).to(dtype))
        layer = targets[native]
        if stream and layer.streamable:
            streamed.append((layer, torch.cat(downs, dim=0), torch.cat(ups, dim=1)))
        else:
            layer.adapters.append((torch.cat(downs, dim=0).to(device), torch.cat(ups, dim=1).to(device), 1.0))

    if streamed:
        # one packed pinned copy of all the pairs (pin_tensors lays them into power-of-two chunks, see pinned.py)
        pinned = pin_tensors([t for _, down, up in streamed for t in (down, up)])
        for i, (layer, _, _) in enumerate(streamed):
            layer.adapters_home.append((pinned[2 * i], pinned[2 * i + 1], 1.0))


def truncate_rank(lora: LoraFile, rank: int) -> LoraFile:
    """A copy of the LoRA in which every layer has rank at most `rank`: the closest such correction (SVD).

    Why. Without LoRA streaming a LoRA lives on the GPU for as long as it is attached (with streaming it is copied over
    PCIe on every pass, so a smaller rank is also less traffic); the identity-edit LoRA at rank 256 takes 1.7 GB, which
    is what stops a 720 edit from fitting into 16 GB. Rank 128 halves that, and almost nothing is lost, because the
    correction up @ down concentrates in its first directions: on that file the worst layer keeps 96% of the energy.

    How. up @ down is never built as a full matrix (6144 x 16384 numbers for an MLP layer). Both factors are first
    reduced by QR: up = Qu @ Ru, down.T = Qd @ Rd, so up @ down = Qu @ (Ru @ Rd.T) @ Qd.T, and only the small
    r x r core is decomposed by SVD into U S Vh. The `rank` largest singular directions are kept, and sqrt(S) is
    split evenly between the two new factors. The "kept energy" of a layer is sum(S[:rank]^2) / sum(S^2).
    Computed on the CPU in float32; the result has the dtype of the file. Layers whose rank is already at most
    `rank`, and the ".diff" layers, are shared with the original as is.

    Returns the original object if no layer needed truncation. Otherwise the copy is remembered in
    lora.truncated[rank], so the second call costs nothing and returns the same object (the engine compares the
    attached LoRAs by identity); kept_energy / worst_kept_energy of the copy are for the caller's log."""
    if rank < 1:
        raise ValueError(f"rank must be at least 1, got {rank}")
    if all(down.shape[0] <= rank for down, _, _ in lora.layers.values()):
        return lora
    if rank in lora.truncated:
        return lora.truncated[rank]

    t0 = time.time()
    out = LoraFile(name=lora.name, diffs=dict(lora.diffs))
    for native, (down, up, scale) in lora.layers.items():
        if down.shape[0] <= rank:
            out.layers[native] = (down, up, scale)
            continue
        new_down, new_up, kept = _truncate_pair(down, up, rank)
        out.layers[native] = (new_down, new_up, scale)
        out.kept_energy[native] = kept
    lora.truncated[rank] = out
    logger.info("LoRA %s: rank truncated to %d in %d layers, %.1fs, worst kept energy %.4f",
                lora.name, rank, len(out.kept_energy), time.time() - t0, out.worst_kept_energy)
    return out


def _truncate_pair(down: torch.Tensor, up: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """One layer of truncate_rank: (down, up) -> the best pair of rank at most `rank`, and the kept energy share."""
    d, u = down.to("cpu", torch.float32), up.to("cpu", torch.float32)
    qu, ru = torch.linalg.qr(u)
    qd, rd = torch.linalg.qr(d.T)
    U, S, Vh = torch.linalg.svd(ru @ rd.T)
    energy = S.square()
    total = float(energy.sum())
    kept = float(energy[:rank].sum()) / total if total > 0 else 1.0
    root = S[:rank].sqrt()
    new_down = (root[:, None] * (Vh[:rank] @ qd.T)).to(down.dtype)
    new_up = ((qu @ U[:, :rank]) * root).to(up.dtype)
    return new_down, new_up, kept
