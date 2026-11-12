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
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

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
            m.deltas.clear()


def apply_loras(model: nn.Module, loras: list[tuple[LoraFile, float]], device: str, dtype: torch.dtype = torch.bfloat16) -> None:
    """Attaches a set of LoRAs to the model, replacing the previous set.

        Several LoRAs on one layer are concatenated into one pair of matrices: the downs are stacked
        one under another, the ups side by side, and the strength of each is pre-multiplied into its up. Mathematically
        it is the same sum of corrections, but in one multiplication instead of several."""
    clear_loras(model)
    targets = {name: m for name, m in model.named_modules() if isinstance(m, QLinear)}
    stacked: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}

    for lora, strength in loras:
        if strength == 0:
            continue
        for native, (down, up, scale) in lora.layers.items():
            downs, ups = stacked.setdefault(native, ([], []))
            downs.append(down.to(device, dtype))
            ups.append(up.to(device, torch.float32) * (scale * strength))
        for native, diff in lora.diffs.items():
            targets[native].deltas.append((diff.to(device, dtype), strength))

    for native, (downs, ups) in stacked.items():
        targets[native].adapters.append((torch.cat(downs, dim=0), torch.cat(ups, dim=1).to(dtype), 1.0))
