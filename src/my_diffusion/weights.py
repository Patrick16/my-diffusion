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
                   per every 32 numbers in a row:  w = int8 * scale[block].

Two ways to multiply
--------------------
"Exact" (works for any format): expand the layer's matrix into bf16, multiply, discard.
This is two passes over ~200 MB of memory and multiplication in bf16.

"Native fp8" (only mxfp8 and only NVIDIA 50-series GPUs): do not expand at all.
The tensor cores of these cards can multiply mxfp8 by mxfp8 directly - in fact the format
was designed for this. For this, the layer input is also rounded on the fly to mxfp8 (this is cheap: the input
is dozens of times smaller than the weight matrix). The multiplication comes out ~2.5 times faster, with no unpacking.
The price is rounding the input: 3 bits of mantissa, about 2-3% noise per layer. The checkpoint's author marks
layers where such rounding is harmful (the full_precision_matrix_mult flag in the file) - we compute them
the exact way, as ComfyUI does. See QLinear.forward and set_matmul."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pinned import ALIGN, pinned_buffer, plan_chunks

logger = logging.getLogger(__name__)

# Some checkpoints (saved whole from ComfyUI) keep the weights under this prefix.
_COMFY_PREFIX = "model.diffusion_model."

MXFP8_BLOCK = 32
E8M0_BIAS = 127
FP8_MAX_MANTISSA = 0.875   # the largest fp8 (e4m3) number - 448 = 0.875 * 2^9
FP8_MAX_EXPONENT = 9
NATIVE_OUT_MULTIPLE = 16   # cuBLAS requirement for the native fp8 multiplication: the number of layer outputs is a multiple of 16

# How to multiply matrices (see "Two ways to multiply" above):
#   exact    - always the exact way;
#   fp8      - native fp8 multiplication where the checkpoint author allows it (as ComfyUI does);
#   fp8_all  - native fp8 multiplication in all mxfp8 layers, despite the author's marks (for experiments).
MATMUL_MODES = ("exact", "fp8", "fp8_all")

_ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


_READ_CHUNK = 256 * 2**20


def read_safetensors(path: str | Path, pin: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Reads a .safetensors file entirely into RAM. Returns (tensors, metadata).

    The file format is simple: 8 bytes - the header length, then the JSON header (name -> type,
    size, offset), then the raw bytes of all tensors one after another.

    Why not safetensors.load_file: it maps the file into memory (mmap) and reads lazily,
    on first access to the data. Moving to the GPU then pulls the disk in small pieces -
    on this checkpoint it took 4 minutes instead of seconds. Here we read sequentially.

    pin=True: we read not into ordinary memory but straight into pinned memory (what it is and why - see
    pinned.py): the file lands contiguously in several large chunks, and the tensors are windows into them,
    without copying. We do this for large models that travel to the GPU and back."""
    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        metadata = header.pop("__metadata__", None) or {}
        entries = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

        def view(raw: torch.Tensor, info: dict) -> torch.Tensor:
            return raw.view(_ST_DTYPES[info["dtype"]]).reshape(info["shape"])

        def read_plain(info: dict) -> torch.Tensor:
            start, end = info["data_offsets"]
            raw = torch.empty(end - start, dtype=torch.uint8)
            if end > start:
                f.seek(base + start)
                f.readinto(raw.numpy())
            return view(raw, info)

        if not pin:
            for name, info in entries:
                tensors[name] = read_plain(info)
            return tensors, metadata

        # How much space in the file each tensor takes, together with the gap up to the next one:
        # a pinned memory chunk gets a contiguous segment of the file.
        starts = [info["data_offsets"][0] for _, info in entries]
        data_end = max(info["data_offsets"][1] for _, info in entries) if entries else 0
        extents = [nxt - cur for cur, nxt in zip(starts, starts[1:] + [data_end])]
        position = 0
        for chunk_size, count in plan_chunks([e + ALIGN for e in extents]):   # + ALIGN: a margin for aligning the start
            group = entries[position : position + count]
            position += count
            # we round the segment start down: then the offsets in the chunk and in the file match in alignment
            first = starts[position - count] // ALIGN * ALIGN
            last = max(info["data_offsets"][1] for _, info in group)
            chunk = pinned_buffer(chunk_size) if last > first else None
            if chunk is None:
                for name, info in group:
                    tensors[name] = read_plain(info)
                continue
            target = memoryview(chunk.numpy())[: last - first]
            f.seek(base + first)
            done = 0
            while done < len(target):   # in chunks: reading tens of gigabytes in one call is unreliable on Windows
                got = f.readinto(target[done : done + _READ_CHUNK])
                if not got:
                    raise EOFError(f"{path.name}: the file is shorter than its header says")
                done += got
            for name, info in group:
                start, end = info["data_offsets"]
                if end == start:
                    tensors[name] = view(torch.empty(0, dtype=torch.uint8), info)
                elif start % _ST_DTYPES[info["dtype"]].itemsize:
                    # the start is not a multiple of the number size - a window into the shared chunk is impossible, so we take a copy
                    tensors[name] = view(chunk[start - first : end - first].clone(), info)
                else:
                    tensors[name] = view(chunk[start - first : end - first], info)
    return tensors, metadata


@dataclass
class Checkpoint:
    """The file contents in a single form, regardless of how it was saved."""

    tensors: dict[str, torch.Tensor]
    # layer name (without ".weight") -> quantization format; layers not listed here are dense
    formats: dict[str, str] = field(default_factory=dict)
    # layers that the checkpoint author asks to multiply exactly (full_precision_matrix_mult)
    full_precision: set[str] = field(default_factory=set)


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


def read_checkpoint(path: str | Path, pin: bool = False) -> Checkpoint:
    """pin - read straight into pinned memory (see read_safetensors). Does not apply to .gguf:
    there the tensors are rebuilt during reading; Resident.with_home_copy will pin them later."""
    path = Path(path)
    if path.suffix == ".gguf":
        return _read_gguf(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"I can read .safetensors and .gguf, but this is {path.name}")
    tensors, metadata = read_safetensors(path, pin=pin)

    # Bring names to one form: "blocks.0.attn.wq.weight"
    if any(k.startswith(_COMFY_PREFIX) for k in tensors):
        tensors = {(k[len(_COMFY_PREFIX):] if k.startswith(_COMFY_PREFIX) else k): v for k, v in tensors.items()}

    formats: dict[str, str] = {}
    full_precision: set[str] = set()

    def note(layer: str, conf: dict) -> None:
        formats[layer] = conf["format"]
        if conf.get("full_precision_matrix_mult"):
            full_precision.add(layer)

    # Variant 1: the quantization description as one JSON in the file header.
    if "_quantization_metadata" in metadata:
        for layer, conf in json.loads(metadata["_quantization_metadata"])["layers"].items():
            note(layer, conf)

    # Variant 2: each layer has its own small tensor "<layer>.comfy_quant" with JSON inside.
    for key in [k for k in tensors if k.endswith(".comfy_quant")]:
        note(key[: -len(".comfy_quant")], json.loads(bytes(tensors.pop(key).tolist()).decode("utf-8")))

    return Checkpoint(tensors=tensors, formats=formats, full_precision=full_precision)


# --- mxfp8: block exponent order ---
#
# The block exponents are a table (matrix row, block number of 32 numbers). In the file it is
# not stored in order but rearranged in 128x4 tiles: this is the layout NVIDIA's cuBLAS library expects
# (see "block scaling factors layout" in its documentation), which performs the native
# fp8 multiplication. We keep the table as in the file - then the fast path can use it without
# rework, and the exact path restores the natural order before unpacking (a trivial
# operation: the table is 32 times smaller than the weights themselves).

def _swizzled_size(rows: int, cols: int) -> int:
    return 128 * -(-rows // 128) * 4 * -(-cols // 4)


def unswizzle_block_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """cuBLAS order -> natural (row, block number). A port of from_blocked from comfy_kitchen."""
    row_tiles = -(-rows // 128)
    col_tiles = -(-cols // 4)
    x = blocked.reshape(-1, 32, 16)
    x = x.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(row_tiles, col_tiles, 4, 32, 4)
    x = x.reshape(row_tiles, col_tiles, 128, 4)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(row_tiles * 128, col_tiles * 4)[:rows, :cols].contiguous()


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """Natural order (row, block number) -> cuBLAS order. The inverse of unswizzle_block_scales;
    a port of to_blocked from comfy_kitchen. If the sizes are not multiples of 128x4, the table is padded with zeros."""
    rows, cols = scales.shape
    row_tiles = -(-rows // 128)
    col_tiles = -(-cols // 4)
    if (rows, cols) != (row_tiles * 128, col_tiles * 4):
        padded = scales.new_zeros(row_tiles * 128, col_tiles * 4)
        padded[:rows, :cols] = scales
        scales = padded
    x = scales.view(row_tiles, 128, col_tiles, 4).permute(0, 2, 1, 3)
    return x.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).contiguous()


# --- mxfp8: layer input for the native fp8 multiplication ---

@dataclass
class Mxfp8Input:
    """The layer input rounded to mxfp8. The original x is stored alongside: it is needed by the LoRA add-ons and
    by the layers that are computed the exact way."""

    x: torch.Tensor        # as received: (..., in), bf16
    data: torch.Tensor     # (rows, in), fp8
    scales: torch.Tensor   # block exponents in cuBLAS order, type float8_e8m0fnu


def quantize_mxfp8(x: torch.Tensor) -> Mxfp8Input:
    """Rounds the layer input to mxfp8: every 32 adjacent numbers are divided by a common power of two
    so that the largest of them fits into the fp8 range (up to 448), and then rounded to fp8.

    The power of two is found without logarithms, by parsing the number itself: frexp decomposes
    the block maximum as m * 2^e, where 0.5 <= m < 1. Since 448 = 0.875 * 2^9, the block fits
    with the multiplier 2^(e-9) if m <= 0.875, and with 2^(e-8) otherwise. Division by a power of two is exact,
    so the whole loss of precision is only in the rounding to fp8 itself."""
    rows = x.reshape(-1, x.shape[-1])
    count, width = rows.shape
    blocks = rows.reshape(count, width // MXFP8_BLOCK, MXFP8_BLOCK)
    peak = torch.linalg.vector_norm(blocks, ord=float("inf"), dim=-1).float()   # maximum absolute value in the block
    mantissa, exponent = torch.frexp(peak)
    exponent = (exponent - FP8_MAX_EXPONENT + (mantissa > FP8_MAX_MANTISSA)).clamp_(-E8M0_BIAS, E8M0_BIAS)
    data = torch.ldexp(blocks, -exponent.unsqueeze(-1)).view(count, width).to(torch.float8_e4m3fn)
    scales = swizzle_block_scales((exponent + E8M0_BIAS).to(torch.uint8)).view(torch.float8_e8m0fnu)
    return Mxfp8Input(x=x, data=data, scales=scales)


def shared_input(x: torch.Tensor, *layers: "QLinear") -> "torch.Tensor | Mxfp8Input":
    """Several layers often get the same input (wq, wk, wv and gate in attention;
    gate and up in the MLP). If at least one of them goes the native fp8 path, we round the input once
    for all of them, rather than anew in each layer."""
    if any(layer.uses_native for layer in layers):
        return quantize_mxfp8(x)
    return x


_native_checked: dict[str, None] = {}


def require_native_mxfp8(device: str) -> None:
    """Checks in practice that the GPU and torch can do native mxfp8 multiplication: tries a tiny one.
    If not - a loud error right when the mode is enabled, rather than in the middle of a generation."""
    if device in _native_checked:
        return
    try:
        from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
        probe = quantize_mxfp8(torch.ones(NATIVE_OUT_MULTIPLE, MXFP8_BLOCK, device=device, dtype=torch.bfloat16))
        out = scaled_mm(
            probe.data, probe.data.t(), scale_a=probe.scales, scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=probe.scales, scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4, output_dtype=torch.bfloat16,
        )
        if not bool((out.float() == MXFP8_BLOCK).all()):   # all block scales equal to 1.0 (a row of 32 ones per row of 32 ones)
            raise RuntimeError(f"test multiplication gave {out.float().flatten()[0].item()} instead of {MXFP8_BLOCK}")
    except Exception as exc:
        raise RuntimeError(
            "native fp8 multiplication is unavailable (needs torch >= 2.10 and an NVIDIA 50-series GPU); "
            f"set MY_DIFFUSION_MATMUL=exact. Reason: {type(exc).__name__}: {exc}"
        ) from exc
    _native_checked[device] = None


class QLinear(nn.Module):
    """A linear layer y = x @ W.T + b whose W lives in memory in compressed form.

    It can multiply in two ways - exact and native fp8 (see the module description); the mode for
    each layer is chosen by set_matmul.

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
        self.full_precision = False   # the checkpoint author's mark: multiply this layer only exactly
        self.native = False           # selected multiplication mode; set by set_matmul
        # buffers, not parameters: there is no training, and .to(device) moves both
        self.register_buffer("weight", None, persistent=False)
        self.register_buffer("scale", None, persistent=False)      # float8_e4m3fn / int8_tensorwise
        self.register_buffer("block_exp", None, persistent=False)  # mxfp8: uint8, block exponents in cuBLAS order (as in the file)
        self.register_buffer("block_scale", None, persistent=False)  # q8_0: float16 (out, in/32)
        self.register_buffer("bias", None, persistent=False)
        # LoRA: (down, up, strength) - low-rank addition; (delta, strength) - full correction to the matrix
        self.adapters: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self.deltas: list[tuple[torch.Tensor, float]] = []

    def set_weight(
        self, weight: torch.Tensor, fmt: str, scale: torch.Tensor | None, bias: torch.Tensor | None, full_precision: bool = False,
    ) -> None:
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
            if scale.numel() != _swizzled_size(*blocks):
                raise ValueError(f"mxfp8: weight_scale has {scale.numel()} exponents, but the layer needs {_swizzled_size(*blocks)}")
            self.block_exp = scale.view(torch.uint8)
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
        self.full_precision = full_precision
        self.native = False

    @property
    def uses_native(self) -> bool:
        """Whether the next call will go the native fp8 path. A full correction to the matrix (LoRA in
        the .diff format) does not fit into the compressed weights, so a layer with it is computed the exact way."""
        return self.native and not self.deltas

    def dense_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Expands the compressed weights into an ordinary matrix of the needed type (on the same device)."""
        w = self.weight.to(dtype)
        block_scale = None
        if self.block_exp is not None:
            exp = unswizzle_block_scales(self.block_exp, self.out_features, self.in_features // MXFP8_BLOCK)
            # 2^(e-127); for a block of all zeros e=0, the multiplier does not matter there (0 * anything = 0)
            block_scale = torch.exp2(exp.to(torch.float32) - E8M0_BIAS).to(dtype)
        elif self.block_scale is not None:
            block_scale = self.block_scale.to(dtype)
        # Multiply in place (mul_): w here is a fresh copy (the compressed fp8/int8 were just converted
        # to dtype), so nothing to protect, and a second temporary matrix of ~200 MB is not created.
        if block_scale is not None:
            w.view(self.out_features, -1, MXFP8_BLOCK).mul_(block_scale.unsqueeze(-1))
        if self.scale is not None:
            w.mul_(self.scale.to(dtype))
        for delta, strength in self.deltas:
            w = w + delta.to(dtype) * strength
        return w

    def forward(self, x: "torch.Tensor | Mxfp8Input") -> torch.Tensor:
        packed = x if isinstance(x, Mxfp8Input) else None
        if packed is not None:
            x = packed.x
        if self.uses_native:
            y = self._native_linear(packed or quantize_mxfp8(x))
        else:
            y = F.linear(x, self.dense_weight(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        for down, up, strength in self.adapters:
            y = y + F.linear(F.linear(x, down), up) * strength
        return y

    def _native_linear(self, packed: Mxfp8Input) -> torch.Tensor:
        """y = x @ W.T directly on the compressed numbers: both the weights and the input are mxfp8, cuBLAS multiplies them
        on the tensor cores, and the result comes out in bf16 at once. The weight matrix is not expanded in this process."""
        x = packed.x
        y = F.scaled_mm(
            packed.data, self.weight.t(),
            scale_a=packed.scales, scale_recipe_a=F.ScalingType.BlockWise1x32,
            scale_b=self.block_exp.view(torch.float8_e8m0fnu), scale_recipe_b=F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4, swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            output_dtype=x.dtype,
        ).view(*x.shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias.to(x.dtype)

    def extra_repr(self) -> str:
        mult = "native" if self.native else "exact"
        return f"in={self.in_features}, out={self.out_features}, bias={self.has_bias}, fmt={self.fmt}, matmul={mult}"


def set_matmul(model: nn.Module, mode: str, device: str) -> int:
    """Selects the multiplication mode for all layers of the model (see MATMUL_MODES). Returns how many
    layers will go the native fp8 path. The mode can be changed at any time; reloading the weights is not needed."""
    if mode not in MATMUL_MODES:
        raise ValueError(f"unknown multiplication mode {mode!r}; available: {MATMUL_MODES}")
    layers = [m for m in model.modules() if isinstance(m, QLinear)]
    chosen = [
        m for m in layers
        if mode != "exact" and m.fmt == "mxfp8" and m.out_features % NATIVE_OUT_MULTIPLE == 0
        and (mode == "fp8_all" or not m.full_precision)
    ]
    if chosen:
        require_native_mxfp8(device)
    for m in layers:
        m.native = False
    for m in chosen:
        m.native = True
    return len(chosen)


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
            module.set_weight(weight, ckpt.formats.get(name, "dense"), scale, bias, full_precision=name in ckpt.full_precision)
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
