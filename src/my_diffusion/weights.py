"""Reading Krea 2 checkpoints and the linear layer that stores its weights in compressed form.

Why this is needed
------------------
Krea 2 has about 13 billion parameters. In ordinary bf16 (2 bytes per number) that is ~26 GB -
which does not fit into 16 GB of video memory. Therefore checkpoints are distributed quantized:
1 byte per number (~12.6 GB). We keep the weights on the GPU in exactly this compressed form
and unpack them into bf16 only for the duration of one matrix multiplication, layer by layer.
The unpacked matrix of one layer is at most ~200 MB and is freed at once.

Formats
-------
The same byte in a file can mean different things, and if this is confused, the network
gets garbage weights and outputs noise (without a single error!). Therefore the loader is strict:
an unknown format is an exception, not "let's try something".

  dense            floating-point numbers as they are (bf16/f16/f32, or "bare" fp8
                   without multipliers). Unpacking is just a type cast.
  mxfp8            the value is stored as fp8 (float8_e4m3fn), plus for every 32 adjacent
                   numbers in a row - a common multiplier 2^(e-127), the exponent e being one byte.
                   Result: w = fp8 * 2^(e-127). The multipliers are needed because fp8 has a small
                   range and only 3 bits of mantissa: without adjusting the scale for each
                   block, the losses would be noticeable.
  float8_e4m3fn    fp8 and one multiplier for the whole layer:  w = fp8 * scale.
  int8_tensorwise  integers -128..127 and a multiplier per row:  w = int8 * scale[row].
  q8_0             the GGUF format (comes from llama.cpp): integers -128..127 and a float16 multiplier
                   per every 32 numbers in a row:  w = int8 * scale[block]."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Some checkpoints (saved whole from ComfyUI) keep the weights under this prefix.
_COMFY_PREFIX = "model.diffusion_model."

MXFP8_BLOCK = 32
E8M0_BIAS = 127

_ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def read_safetensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Reads a .safetensors file entirely into RAM. Returns (tensors, metadata).

    The file format is simple: 8 bytes - the header length, then the JSON header (name -> type,
    size, offset), then the raw bytes of all tensors one after another.

    Why not safetensors.load_file: it maps the file into memory (mmap) and reads lazily,
    on first access to the data. Moving to the GPU then pulls the disk in small pieces -
    on this checkpoint it took 4 minutes instead of seconds. Here we read sequentially."""
    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        metadata = header.pop("__metadata__", None) or {}
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            start, end = info["data_offsets"]
            raw = torch.empty(end - start, dtype=torch.uint8)
            if end > start:
                f.seek(base + start)
                f.readinto(raw.numpy())
            tensors[name] = raw.view(_ST_DTYPES[info["dtype"]]).reshape(info["shape"])
    return tensors, metadata


@dataclass
class Checkpoint:
    """The file contents in a single form, regardless of how it was saved."""

    tensors: dict[str, torch.Tensor]
    # layer name (without ".weight") -> quantization format; layers not listed here are dense
    formats: dict[str, str] = field(default_factory=dict)


def _read_gguf(path: Path) -> Checkpoint:
    """GGUF: a container from the llama.cpp world. Each Q8_0 tensor is laid out in blocks of 34 bytes:
    2 bytes - a float16 multiplier, then 32 int8 values. We split these into two parts and
    handle them further the same way as the other formats."""
    import gguf
    import numpy as np

    tensors: dict[str, torch.Tensor] = {}
    formats: dict[str, str] = {}
    for t in gguf.GGUFReader(str(path)).tensors:
        shape = tuple(int(d) for d in reversed(t.shape))   # in GGUF the sizes are stored in reverse order
        kind = t.tensor_type.name
        data = torch.from_numpy(np.array(t.data))          # np.array: a copy from the mapped file into RAM
        if kind in ("F32", "F16", "BF16"):
            tensors[t.name] = data.reshape(shape)
        elif kind == "Q8_0" and len(shape) == 2 and t.name.endswith(".weight"):
            blocks = data.reshape(shape[0], -1, 2 + MXFP8_BLOCK)
            layer = t.name[: -len(".weight")]
            tensors[t.name] = blocks[..., 2:].contiguous().view(torch.int8).reshape(shape)
            tensors[f"{layer}.weight_scale"] = blocks[..., :2].contiguous().view(torch.float16).reshape(shape[0], -1)
            formats[layer] = "q8_0"
        else:
            raise NotImplementedError(f"{path.name}: tensor {t.name} in GGUF format {kind} is not supported yet")
    return Checkpoint(tensors=tensors, formats=formats)


def read_checkpoint(path: str | Path) -> Checkpoint:
    path = Path(path)
    if path.suffix == ".gguf":
        return _read_gguf(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"I can read .safetensors and .gguf, but this is {path.name}")
    tensors, metadata = read_safetensors(path)

    # Bring names to one form: "blocks.0.attn.wq.weight"
    if any(k.startswith(_COMFY_PREFIX) for k in tensors):
        tensors = {(k[len(_COMFY_PREFIX):] if k.startswith(_COMFY_PREFIX) else k): v for k, v in tensors.items()}

    formats: dict[str, str] = {}

    # Variant 1: the quantization description as one JSON in the file header.
    if "_quantization_metadata" in metadata:
        for layer, conf in json.loads(metadata["_quantization_metadata"])["layers"].items():
            formats[layer] = conf["format"]

    # Variant 2: each layer has its own small tensor "<layer>.comfy_quant" with JSON inside.
    for key in [k for k in tensors if k.endswith(".comfy_quant")]:
        conf = json.loads(bytes(tensors.pop(key).tolist()).decode("utf-8"))
        formats[key[: -len(".comfy_quant")]] = conf["format"]

    return Checkpoint(tensors=tensors, formats=formats)


def unswizzle_block_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Returns the mxfp8 block exponents in natural order (row, block number).

    In the file they are rearranged in 128x4 tiles - this is the layout NVIDIA's cuBLAS library expects
    (see "block scaling factors layout" in its documentation). We do not need this order,
    so once at load time we expand it back. A port of from_blocked from comfy_kitchen."""
    row_tiles = -(-rows // 128)
    col_tiles = -(-cols // 4)
    x = blocked.reshape(-1, 32, 16)
    x = x.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(row_tiles, col_tiles, 4, 32, 4)
    x = x.reshape(row_tiles, col_tiles, 128, 4)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(row_tiles * 128, col_tiles * 4)[:rows, :cols].contiguous()


class QLinear(nn.Module):
    """A linear layer y = x @ W.T + b whose W lives in memory in compressed form.

    Plus the attachment point for LoRA (see lora.py). LoRA does not fit into the compressed weights -
    one would have to unpack, add, and compress again, with loss of precision - so it is computed
    alongside as a small addition:  y += strength * (x @ down.T) @ up.T.
    This also allows changing the LoRA set and its weights between requests without reloading the model."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.fmt = "empty"
        # buffers, not parameters: there is no training, and .to(device) moves both
        self.register_buffer("weight", None, persistent=False)
        self.register_buffer("scale", None, persistent=False)      # float8_e4m3fn / int8_tensorwise
        self.register_buffer("block_exp", None, persistent=False)  # mxfp8: uint8 (out, in/32)
        self.register_buffer("block_scale", None, persistent=False)  # q8_0: float16 (out, in/32)
        self.register_buffer("bias", None, persistent=False)
        # LoRA: (down, up, strength) - low-rank addition; (delta, strength) - full correction to the matrix
        self.adapters: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self.deltas: list[tuple[torch.Tensor, float]] = []

    def set_weight(self, weight: torch.Tensor, fmt: str, scale: torch.Tensor | None, bias: torch.Tensor | None) -> None:
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"weight size {tuple(weight.shape)} != expected {(self.out_features, self.in_features)}")
        if self.has_bias != (bias is not None):
            raise ValueError(f"bias: in the model {'yes' if self.has_bias else 'no'}, in the file {'yes' if bias is not None else 'no'}")

        self.scale = self.block_exp = self.block_scale = None
        blocks = (self.out_features, self.in_features // MXFP8_BLOCK)
        if fmt == "q8_0":
            if weight.dtype != torch.int8 or scale is None or tuple(scale.shape) != blocks or self.in_features % MXFP8_BLOCK:
                raise ValueError("q8_0: needs int8 weights and one multiplier per block of 32 numbers")
            self.block_scale = scale
        elif fmt == "dense":
            if not weight.dtype.is_floating_point or scale is not None:
                raise ValueError(f"dense layer: unexpected type {weight.dtype} or an extra weight_scale")
        elif fmt == "mxfp8":
            if weight.dtype != torch.float8_e4m3fn or scale is None:
                raise ValueError("mxfp8: needs fp8 weights and weight_scale")
            if self.in_features % MXFP8_BLOCK:
                raise ValueError("mxfp8: the layer width is not a multiple of 32")
            self.block_exp = unswizzle_block_scales(scale.view(torch.uint8), self.out_features, self.in_features // MXFP8_BLOCK)
        elif fmt == "float8_e4m3fn":
            if weight.dtype != torch.float8_e4m3fn or scale is None or scale.numel() != 1:
                raise ValueError("float8_e4m3fn: needs fp8 weights and one multiplier per layer")
            self.scale = scale.to(torch.float32).reshape(1, 1)
        elif fmt == "int8_tensorwise":
            if weight.dtype != torch.int8 or scale is None or scale.numel() not in (1, self.out_features):
                raise ValueError("int8_tensorwise: needs int8 weights and a multiplier per layer or per row")
            self.scale = scale.to(torch.float32).reshape(-1, 1)
        else:
            raise NotImplementedError(f"quantization format {fmt!r} is not supported yet")

        self.weight = weight
        self.bias = bias
        self.fmt = fmt

    def dense_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Expands the compressed weights into an ordinary matrix of the needed type (on the same device)."""
        w = self.weight.to(dtype)
        block_scale = None
        if self.block_exp is not None:
            # 2^(e-127); for a block of all zeros e=0, the multiplier does not matter there (0 * anything = 0)
            block_scale = torch.exp2(self.block_exp.to(torch.float32) - E8M0_BIAS).to(dtype)
        elif self.block_scale is not None:
            block_scale = self.block_scale.to(dtype)
        if block_scale is not None:
            w = (w.view(self.out_features, -1, MXFP8_BLOCK) * block_scale.unsqueeze(-1)).view(self.out_features, self.in_features)
        if self.scale is not None:
            w = w * self.scale.to(dtype)
        for delta, strength in self.deltas:
            w = w + delta.to(dtype) * strength
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.dense_weight(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        for down, up, strength in self.adapters:
            y = y + F.linear(F.linear(x, down), up) * strength
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.has_bias}, fmt={self.fmt}"


def load_into(model: nn.Module, ckpt: Checkpoint) -> None:
    """Distributes the checkpoint tensors among the model's layers. Strictly: every tensor of the file must
    find its place, and every layer of the model must receive weights - otherwise an exception is raised."""
    tensors = dict(ckpt.tensors)

    for name, module in model.named_modules():
        if not isinstance(module, QLinear):
            continue
        try:
            weight = tensors.pop(f"{name}.weight")
        except KeyError:
            raise KeyError(f"the checkpoint has no weights for layer {name}") from None
        scale = tensors.pop(f"{name}.weight_scale", None)
        bias = tensors.pop(f"{name}.bias", None)
        try:
            module.set_weight(weight, ckpt.formats.get(name, "dense"), scale, bias)
        except Exception as exc:
            raise type(exc)(f"layer {name}: {exc}") from exc

    for name, param in model.named_parameters():
        try:
            value = tensors.pop(name)
        except KeyError:
            raise KeyError(f"the checkpoint has no parameter {name}") from None
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(f"parameter {name}: size {tuple(value.shape)} != {tuple(param.shape)}")
        param.data = value.to(param.dtype)

    if tensors:
        raise KeyError(f"the checkpoint has unused tensors ({len(tensors)}): {sorted(tensors)[:8]}")
