"""Loads the local Qwen3-VL-4B-Instruct from a quanto-quantized checkpoint
(taken from the Wan2GP installation), without network access.

A peculiarity of this particular file (see the discussion in the chat, where we looked at the
safetensors header): it stores only the backbone - tensors with the prefix
"language_model.". This matches the `Qwen3VLModel` class (the architecture is stated exactly
so in config.json). The LM head is architecturally absent there, because
Wan2GP uses this model only as a text encoder.

We also need the LM head - for the prompt enhancer. `tie_word_embeddings: true` in the config helps:
the LM head is mathematically the transposed `embed_tokens.weight`, so no separate tensor is needed for it.
Therefore, instead of `Qwen3VLModel` we build the full `Qwen3VLForConditionalGeneration`
(it has `.model` = the same backbone, and `.lm_head`), load the backbone weights
under the path `model.*`, and set the link `lm_head.weight = model.language_model.embed_tokens.weight`
with the standard `model.tie_weights()` - copy-less, no extra memory is used.

We pass `.model` from the result to Krea2Pipeline as text_encoder; the object itself -
for the enhancer (it has `.generate()`)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from optimum.quanto import requantize
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, Qwen3VLForConditionalGeneration

WEIGHTS_FILENAME = "Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors"


def load_local_qwen3vl(checkpoint_dir: Path, device: str):
    """Returns (model, tokenizer). model - Qwen3VLForConditionalGeneration;
    model.model - the backbone for Krea2Pipeline.text_encoder, and model itself - for the enhancer."""
    weights_path = checkpoint_dir / WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found - check the path to the local Qwen3-VL")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir)

    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    if "quantization_map_base64" not in metadata:
        raise RuntimeError(f"{weights_path} has no quantization_map_base64 in its metadata - the format does not match what is expected")
    quantization_map = json.loads(base64.b64decode(metadata["quantization_map_base64"]))

    # The file stores the backbone without the "model." prefix (weights exactly as in Qwen3VLModel).
    # We build the full Qwen3VLForConditionalGeneration, where the backbone lives in
    # .model - we add this prefix both to the weights and to the quantization map.
    quantization_map = {f"model.{k}": v for k, v in quantization_map.items()}
    state_dict = {f"model.{k}": v for k, v in load_file(str(weights_path)).items()}

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    # requantize itself: (1) replaces the needed Linear layers with quantized modules per the map,
    # (2) moves all parameters from meta to real memory, (3) moves them to the device,
    # (4) loads state_dict (strict=False - the vision tower is absent from the file and
    # keeps garbage values; we never call it, since we work
    # with text only, without images).
    requantize(model, state_dict, quantization_map, device=torch.device(device))

    # the LM head exists only as a module name (created empty inside
    # Qwen3VLForConditionalGeneration) - tie_weights() links it to embed_tokens
    # without copying, since config.tie_word_embeddings=True.
    model.tie_weights()
    model.eval()

    return model, tokenizer
