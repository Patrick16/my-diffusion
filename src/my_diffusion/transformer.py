"""The Krea 2 neural network - a "single-stream MMDiT".

What it does: it receives a noisy image (in compressed latent form), the text
and the noise level, and returns the "velocity" - the direction in which the image must be shifted
so that there is less noise. The sampler (sampler.py) calls it several times
in a row and step by step turns pure noise into an image.

Structure:
  1. The latent is cut into 2x2 squares ("patches"); each patch is one token.
  2. The text arrives as 12 layers of Qwen3-VL hidden states per token.
     The txtfusion block folds these 12 layers into one vector per token.
  3. The text and image tokens are concatenated into ONE sequence ("single-stream")
     and pass through 28 identical blocks: attention + MLP.
  4. The noise level is mixed into each block via scale/shift (modulation).
  5. The image tokens are assembled back into the latent.

A port of comfy/ldm/krea2/model.py from ComfyUI without its infrastructure. The module names
match the tensor names in the checkpoint - that is why the weights load directly."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .weights import QLinear, shared_input, streamed_adapters, streamed_weights

# Which kernel computes attention.
#
# F.scaled_dot_product_attention is not one algorithm but several ready-made "kernels", from which
# torch picks by itself. The best known one, FlashAttention, is not included in the torch build for Windows, and by
# default torch takes the "efficient" kernel here. But the build also has a kernel from NVIDIA's
# cuDNN library: torch tries it last, but on this network it is 2-2.4x faster (measured per block:
# 480p 2.6 -> 1.3 ms, 720p 10.8 -> 4.7 ms, 1080p 52.6 -> 21.7 ms). The accuracy is the same: both kernels
# are equally far from a float32 reference; they differ from each other only at the rounding level (~0.1%).
#
#   cudnn   - try the cuDNN kernel first; if it does not accept the input (for example, length 1) -
#             torch itself moves on to the next kernel in the list;
#   default - whatever torch picks (on Windows that is "efficient"): for comparison with ComfyUI.
#
# The cuDNN kernel has a cost: for each new input shape (that is, a new sequence length)
# it is prepared once for ~0.2-0.4 s, after which it works instantly. Hence two rules:
#   - we enable it only in the 28 main blocks, where attention costs tens of milliseconds per call.
#     In the text part (txtfusion) attention takes fractions of a millisecond, and there are two new shapes
#     per prompt length there - the preparation would cost more than it saves;
#   - the preparation is remembered separately in each thread, so the server runs all generations
#     in the same thread (see api.py).
#
#   sage2   - SageAttention 2 (the optional `sageattention` package, see the "sage2" group in pyproject.toml).
#             Unlike the two above it is an APPROXIMATION: queries and keys are quantized to int8, the attention
#             weights and values to fp8, and the products run on the tensor cores. One call is 1.9-2.5x faster
#             than cudnn (480p 1.3 -> 0.65 ms, 1080p 22 -> 9.4 ms, 1080p + a reference 88 -> 40 ms), but its
#             output is ~4% away from a float32 reference where cudnn is 0.4% away, so the picture changes.
#             Never chosen automatically. Sequences shorter than `sage_min_tokens` still go to cudnn: there the
#             kernel saves little time and costs the same accuracy.
ATTENTION_KERNELS = ("cudnn", "default", "sage2")
CUDNN_FIRST = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
_CAN_SET_PRIORITY = "set_priority" in inspect.signature(sdpa_kernel).parameters
_sageattn = None   # sageattention.sageattn, imported on first use (the package is optional)


def _load_sageattn():
    global _sageattn
    if _sageattn is None:
        try:
            from sageattention import sageattn
        except ImportError as e:
            raise RuntimeError(
                "the sage2 attention kernel needs the optional sageattention package (with triton-windows): "
                f"install it with `uv sync --group sage2` or set MY_DIFFUSION_ATTENTION=cudnn ({e})"
            ) from e
        _sageattn = sageattn
    return _sageattn


def set_attention_kernel(blocks: nn.Module, kernel: str, sage_min_tokens: int = 0) -> None:
    """Selects the attention kernel (see ATTENTION_KERNELS) for all attention layers inside blocks.
    For the Krea 2 network, model.blocks is passed here - the main blocks, without the text part.
    sage_min_tokens - only for sage2: sequences shorter than this are computed by cudnn."""
    if kernel not in ATTENTION_KERNELS:
        raise ValueError(f"unknown attention kernel {kernel!r}; available: {ATTENTION_KERNELS}")
    if kernel in ("cudnn", "sage2") and not _CAN_SET_PRIORITY:
        raise RuntimeError(f"the {kernel} attention kernel needs a newer torch (no sdpa_kernel(set_priority=...)); set MY_DIFFUSION_ATTENTION=default")
    if kernel == "sage2":
        _load_sageattn()   # fail here, at load time, not in the middle of the first generation
    for module in blocks.modules():
        if isinstance(module, Attention):
            module.kernel = kernel
            module.sage_min_tokens = sage_min_tokens


@dataclass(frozen=True)
class Krea2Config:
    features: int = 6144      # width of the main stream
    heads: int = 48           # attention heads (128 dimensions each)
    kvheads: int = 12         # GQA: keys/values are computed for 12 heads and repeated x4 - saves weights
    layers: int = 28
    multiplier: int = 4       # how many times wider the MLP is than the stream
    patch: int = 2
    channels: int = 16        # channels in the VAE latent
    tdim: int = 256           # size of the sinusoidal encoding of the noise level
    theta: float = 1000.0     # base frequency of the positional encoding (RoPE)
    txtdim: int = 2560        # width of the hidden states of Qwen3-VL-4B
    txtlayers: int = 12       # how many Qwen layers we take
    txtheads: int = 20
    txtkvheads: int = 20


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(*shape), requires_grad=False)


# Working memory of one block.
#
# The weights take 12.6 GB of a 16 GB card, so what a block needs while it runs decides which sizes fit. Call
# the stream itself - (tokens, 6144) in bf16, 12 KB per token - one "unit": 100 MB at 1080p, 205 MB for a 1080p
# edit with one reference (16.7 thousand tokens). Computed in one piece, three places need many units at once:
#   - the MLP: its hidden layer is 2.67 units wide, and gate, up and their product exist together: ~11 units;
#   - RoPE on the queries: float32 copies of the queries and of both products: ~8 units;
#   - the normalizations, which also work in float32: ~4 units.
# At 16.7 thousand tokens that is 2.3 GB for a moment, and the edit does not fit. But all three work on each
# token separately, so they can go over the sequence in pieces of CHUNK_TOKENS tokens and write the pieces into
# one result, and the working memory of these three no longer grows with the picture (~0.55 GB at most).
# Measured (scripts/check_memory.py): a 1080p edit with one reference, a 720p edit with two and 1440p
# text-to-image fit a 16 GB card this way, in the same time; before they ran out of memory on the first step.
#
# Are the numbers the same? The normalizations and RoPE: yes, bit for bit, they are computed element by element.
# The MLP is matrix products. The compressed (fp8) product gives the same bits whole and in pieces, so
# text-to-image pictures do not change. The product of a LoRA pair is an ordinary bf16 one, and there the GPU
# library rounds a piece differently from the whole: with a LoRA attached (every edit has one) the same seed
# gives another variant of the picture, as far from the old one as after a change of the attention kernel.
# So the MLP goes in pieces only above MLP_WHOLE_TOKENS, a length no request reached before (the longest that
# fitted were ~8.5 thousand tokens: 1080p text-to-image, a 720p edit with one reference): every picture that
# could be made before comes out the same, and only the new sizes are computed in pieces.
CHUNK_TOKENS = 4096
MLP_WHOLE_TOKENS = 10000


def in_chunks(fn, x: torch.Tensor, size: int | None = None, dim: int = -2) -> torch.Tensor:
    """fn(x), computed over slices of x along `dim` (the token axis) of at most `size` (CHUNK_TOKENS by default)
    and written into one result. fn must work on every slice independently and keep `dim` in its place."""
    size = CHUNK_TOKENS if size is None else size
    n = x.shape[dim] if x.dim() >= -dim else 0
    if n <= size:
        return fn(x)
    out = None
    for start in range(0, n, size):
        piece = fn(x.narrow(dim, start, min(size, n - start)))
        if out is None:
            shape = list(piece.shape)
            shape[dim] = n
            out = piece.new_empty(shape)
        out.narrow(dim, start, piece.shape[dim]).copy_(piece)
    return out


class RMSNorm(nn.Module):
    """RMS normalization. A peculiarity of Krea 2: the file stores not the multiplier itself
    but its deviation from one, that is, the real weight = 1 + scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = _param(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.scale.float() + 1.0

        def norm(part: torch.Tensor) -> torch.Tensor:
            # compute in float32: a sum of squares over 6144 numbers loses precision in bf16
            return F.rms_norm(part.float(), (part.shape[-1],), weight=weight, eps=self.eps).to(part.dtype)

        return in_chunks(norm, x)   # the float32 copies only of one piece at a time, see CHUNK_TOKENS


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)


class SwiGLU(nn.Module):
    """MLP with a gate: down(silu(gate(x)) * up(x))."""

    def __init__(self, features: int, multiplier: int, multiple: int = 128):
        super().__init__()
        hidden = int(2 * features / 3) * multiplier
        hidden = multiple * ((hidden + multiple - 1) // multiple)  # round up to a multiple of 128
        self.gate = QLinear(features, hidden, bias=False)
        self.up = QLinear(features, hidden, bias=False)
        self.down = QLinear(hidden, features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def mlp(part: torch.Tensor) -> torch.Tensor:
            part = shared_input(part, self.gate, self.up)   # gate and up share one input: prepare it for the fp8 multiplication once
            return self.down(F.silu(self.gate(part)) * self.up(part))

        if x.shape[-2] <= MLP_WHOLE_TOKENS:
            return mlp(x)
        return in_chunks(mlp, x)   # the wide hidden layer only of one piece at a time, see CHUNK_TOKENS


# --- RoPE: the token position is encoded by rotating coordinate pairs in queries and keys ---

def _rope_axis(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    angles = pos.to(torch.float64).unsqueeze(-1) / (theta ** scale)
    rot = torch.stack([torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)], dim=-1)
    return rot.view(*angles.shape, 2, 2).to(torch.float32)


def build_rope(pos: torch.Tensor, axes: tuple[int, ...], theta: float) -> torch.Tensor:
    """pos: (B, N, 3) - (image index, row, column). Result: (B, 1, N, 64, 2, 2)."""
    return torch.cat([_rope_axis(pos[..., i], axes[i], theta) for i in range(len(axes))], dim=-3).unsqueeze(1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, N, 128). Rotates each pair of adjacent coordinates by its own angle."""

    def rotate(part: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
        pairs = part.float().reshape(*part.shape[:-1], -1, 1, 2)
        out = rot[..., 0] * pairs[..., 0] + rot[..., 1] * pairs[..., 1]
        return out.reshape(part.shape).to(part.dtype)

    n = x.shape[-2]
    if n <= CHUNK_TOKENS:
        return rotate(x, rope)
    out = torch.empty_like(x)   # in pieces: the float32 copies only of one piece at a time, see CHUNK_TOKENS
    for start in range(0, n, CHUNK_TOKENS):
        length = min(CHUNK_TOKENS, n - start)
        out.narrow(-2, start, length).copy_(rotate(x.narrow(-2, start, length), rope.narrow(2, start, length)))
    return out


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads
        self.headdim = dim // heads
        self.wq = QLinear(dim, self.headdim * heads, bias=False)
        self.wk = QLinear(dim, self.headdim * kvheads, bias=False)
        self.wv = QLinear(dim, self.headdim * kvheads, bias=False)
        self.gate = QLinear(dim, dim, bias=False)
        self.qknorm = QKNorm(self.headdim)
        self.wo = QLinear(dim, dim, bias=False)
        self.kernel = "default"   # attention kernel; set by set_attention_kernel
        self.sage_min_tokens = 0  # with the sage2 kernel: shorter sequences are computed by cudnn

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        x = shared_input(x, self.wq, self.wk, self.wv, self.gate)   # one shared input for four layers
        q = self.wq(x).view(b, n, self.heads, self.headdim).transpose(1, 2)
        k = self.wk(x).view(b, n, self.kvheads, self.headdim).transpose(1, 2)
        v = self.wv(x).view(b, n, self.kvheads, self.headdim).transpose(1, 2)
        # One statement per tensor, and `del` below: a tensor that is no longer needed must not wait for the
        # end of the function, each of them is a whole "unit" of working memory (see CHUNK_TOKENS).
        q = self.qknorm.qnorm(q)
        k = self.qknorm.knorm(k)
        if rope is not None:
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if self.kernel == "sage2" and n >= self.sage_min_tokens:
            out = _sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        elif self.kernel in ("cudnn", "sage2"):
            with sdpa_kernel(CUDNN_FIRST, set_priority=True):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        del q, k, v
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.headdim)
        # the attention output is additionally weighted by a sigmoid gate. In place: `out` and the gate are
        # fresh tensors of this call, and each copy of them would be one more unit of working memory.
        gate = self.gate(x).sigmoid_()
        del x
        out = out.mul_(gate)
        del gate
        return self.wo(out)


class TextFusionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int, multiplier: int):
        super().__init__()
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.attn = Attention(dim, heads, kvheads)
        self.mlp = SwiGLU(dim, multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.prenorm(x))
        return x + self.mlp(self.postnorm(x))


class TextFusion(nn.Module):
    """Folds the 12 Qwen layers into one vector per token."""

    def __init__(self, cfg: Krea2Config):
        super().__init__()
        args = (cfg.txtdim, cfg.txtheads, cfg.txtkvheads, cfg.multiplier)
        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(*args) for _ in range(2)])
        self.projector = QLinear(cfg.txtlayers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(*args) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, nlayers, d = x.shape
        # 1) for each token separately: attention ALONG the 12 layers (not along the text)
        x = x.reshape(b * seq, nlayers, d)
        for block in self.layerwise_blocks:
            with streamed_adapters(block, x.device):   # LoRA streaming, see Krea2Transformer.streamed_blocks
                x = block(x)
        # 2) weighted sum of the 12 layers -> one vector per token
        x = x.view(b, seq, nlayers, d).transpose(2, 3)   # (b, seq, d, 12)
        x = self.projector(x).squeeze(-1)                # (b, seq, d)
        # 3) regular attention along the text
        for block in self.refiner_blocks:
            with streamed_adapters(block, x.device):
                x = block(x)
        return x


class Modulation6(nn.Module):
    """Makes 6 quantities from the noise-level vector: scale, shift and gate for attention and for the MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = _param(6 * dim)

    def forward(self, vec: torch.Tensor):
        return (vec + self.lin.to(vec.dtype)).chunk(6, dim=-1)


class SingleStreamBlock(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.mod = Modulation6(cfg.features)
        self.prenorm = RMSNorm(cfg.features)
        self.postnorm = RMSNorm(cfg.features)
        self.attn = Attention(cfg.features, cfg.heads, cfg.kvheads)
        self.mlp = SwiGLU(cfg.features, cfg.multiplier)

    def forward(self, x: torch.Tensor, tvec: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(tvec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, rope)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class Modulation2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = _param(2, dim)

    def forward(self, vec: torch.Tensor):
        out = vec + self.lin.to(vec.dtype).unsqueeze(0)   # (B, 1, dim) + (1, 2, dim)
        return out.chunk(2, dim=1)


class LastLayer(nn.Module):
    def __init__(self, cfg: Krea2Config):
        super().__init__()
        self.norm = RMSNorm(cfg.features)
        self.linear = QLinear(cfg.features, cfg.patch * cfg.patch * cfg.channels, bias=True)
        self.modulation = Modulation2(cfg.features)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(t)
        return self.linear((1 + scale) * self.norm(x) + shift)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0, time_factor: float = 1000.0) -> torch.Tensor:
    """Noise level (a number 0..1) -> a vector of cosines and sines of different frequencies.
    Internally multiplied by 1000: the network was trained on "steps" 0..1000."""
    t = t.float() * time_factor
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Krea2Transformer(nn.Module):
    def __init__(self, cfg: Krea2Config = Krea2Config()):
        super().__init__()
        self.cfg = cfg
        headdim = cfg.features // cfg.heads
        # the 128 head dimensions are split between the axes: image index / row / column
        self.rope_axes = (headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16))

        self.first = QLinear(cfg.channels * cfg.patch ** 2, cfg.features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(cfg) for _ in range(cfg.layers)])
        self.tmlp = nn.Sequential(QLinear(cfg.tdim, cfg.features), nn.GELU(approximate="tanh"), QLinear(cfg.features, cfg.features))
        self.txtfusion = TextFusion(cfg)
        self.txtmlp = nn.Sequential(RMSNorm(cfg.txtdim), QLinear(cfg.txtdim, cfg.features), nn.GELU(approximate="tanh"), QLinear(cfg.features, cfg.features))
        self.last = LastLayer(cfg)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), QLinear(cfg.features, cfg.features * 6))
        for block in self.streamed_blocks():
            for m in block.modules():
                if isinstance(m, QLinear):
                    m.streamable = True

    def streamed_blocks(self) -> list[nn.Module]:
        """The blocks that move their own LoRA pairs to the GPU and back (weights.streamed_adapters): the 28
        main blocks and the 4 text-fusion blocks. Together they hold all but 8 of the 264 layers and almost
        all LoRA weight. The few layers outside them (first, tmlp, txtmlp, projector, last, tproj) are tiny
        and keep their pairs on the GPU, as without streaming (lora.apply_loras)."""
        return [*self.blocks, *self.txtfusion.layerwise_blocks, *self.txtfusion.refiner_blocks]

    def stream_weights(self, blocks: int) -> None:
        """Keeps the weights of the first `blocks` main blocks in RAM instead of the GPU: 0.42 GB of video memory
        each, for ~0.18 s per 8-step generation (weights.streamed_weights). 0 - the whole network on the GPU.

        Only marks the blocks: the mark is read when the network moves to the GPU (memory.Resident.with_home_copy
        with stay_home=...), so change it while the network is in RAM (the engine does: Krea2Engine.stream_weights)."""
        if not 0 <= blocks <= len(self.blocks):
            raise ValueError(f"blocks must be between 0 and {len(self.blocks)}, got {blocks}")
        for i, block in enumerate(self.blocks):
            block.weights_streamed = i < blocks

    @property
    def streamed_weight_blocks(self) -> int:
        return sum(1 for block in self.blocks if getattr(block, "weights_streamed", False))

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def embed_text(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, tokens, 12, 2560) - Qwen hidden states -> (B, tokens, features): the text in the form
        in which the network blocks accept it.

        Moved out of forward, because the result depends only on the prompt - neither on the noise level
        nor on the latent. The sampler computes it once for the whole generation, not anew at each step."""
        return self.txtmlp(self.txtfusion(context.to(self.dtype)))

    def forward(
        self, latent: torch.Tensor, sigma: torch.Tensor, text: torch.Tensor,
        refs: list[tuple[torch.Tensor, tuple[int, int]]] | None = None,
    ) -> torch.Tensor:
        """latent: (B, 16, h, w) - noisy latent;  sigma: (B,) - noise level 0..1;
        text: (B, tokens, features) - the text after embed_text.  Returns the velocity (B, 16, h, w).

        refs - reference pictures for editing: a list of (clean latent, (off_h, off_w)) pairs. The latent
        is (1, 16, h_k, w_k) in the same normalized space as `latent` (what vae.encode returns), with even
        sides; the offset says where its token grid sits inside the generated picture's grid, in tokens.
        The network already reads text and picture as one sequence, so a reference is simply more tokens
        in it: each one is cut into 2x2 patch tokens exactly like the latent and appended after the
        picture's tokens, and in every block the picture looks at it through the ordinary attention.
        The RoPE positions tell the pictures apart: axis 0 is the image index (0 = the picture being
        generated, k = the k-th reference), axes 1-2 are the row and column in the reference's own grid
        plus its offset. This is the layout the edit LoRA was trained with (Wan2GP, ComfyUI).

        The references are clean (not noised) but share the picture's noise-level modulation (one tvec
        for the whole sequence), as in Wan2GP, where editing works this way; ComfyUI's alternative of
        modulating them as if at noise level 0 is not used by default for Krea 2. Their outputs are
        dropped: only the generated picture's tokens are returned. Cost: every reference adds its tokens
        to the attention of all 28 blocks. refs=None (or empty) runs exactly the code without references."""
        cfg = self.cfg
        p = cfg.patch
        b, c, h, w = latent.shape
        gh, gw = h // p, w // p

        # latent -> patch tokens: (B, gh*gw, 16*2*2)
        img = latent.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
        img = self.first(img.to(self.dtype))

        t = self.tmlp(timestep_embedding(sigma, cfg.tdim).unsqueeze(1).to(self.dtype))   # (B, 1, features)
        tvec = self.tproj(t)                                                              # (B, 1, 6*features)

        txt = text.to(self.dtype)
        ntxt = txt.shape[1]

        # positions: text gets all zeros, an image gets (0, row, column)
        pos = torch.zeros(b, ntxt + gh * gw, 3, device=latent.device, dtype=torch.float32)
        grid = pos[:, ntxt:].view(b, gh, gw, 3)
        grid[..., 1] = torch.arange(gh, device=latent.device, dtype=torch.float32)[:, None]
        grid[..., 2] = torch.arange(gw, device=latent.device, dtype=torch.float32)[None, :]
        if refs:   # reference tokens and their positions go after the picture's
            img, pos = self._append_refs(img, pos, refs)
        rope = build_rope(pos, self.rope_axes, cfg.theta)

        x = torch.cat((txt, img), dim=1)
        for block in self.blocks:
            # A block whose weights live in RAM gets them for this call only (stream_weights); the same for
            # its LoRA pairs with LoRA streaming (streamed_blocks).
            with streamed_weights(block, x.device), streamed_adapters(block, x.device):
                x = block(x, tvec, rope)

        if refs:   # drop the reference tokens; only the generated picture is the output
            x = x[:, : ntxt + gh * gw]
        out = self.last(x, t)[:, ntxt:]                                                   # image tokens only
        out = out.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return out

    def _append_refs(
        self, img: torch.Tensor, pos: torch.Tensor, refs: list[tuple[torch.Tensor, tuple[int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Appends the reference tokens after the picture's (see forward).

        img: (B, N, features) - the picture's tokens after `first`; pos: (B, ntxt + N, 3) - the
        positions built so far (text, then picture). Each reference goes through the same patch cut
        and the same `first` layer as the noisy latent, and the k-th one (counting from 1) gets
        positions (k, off_h + row, off_w + column). Returns the longer img and pos."""
        p = self.cfg.patch
        b = img.shape[0]
        tokens, positions = [img], [pos]
        for k, (ref, (off_h, off_w)) in enumerate(refs, start=1):
            ref = ref.to(img.device).expand(b, -1, -1, -1)
            _, c, rh, rw = ref.shape
            if rh % p or rw % p:
                raise ValueError(f"reference latent {rh}x{rw} must have even sides (patch {p})")
            rgh, rgw = rh // p, rw // p
            patches = ref.reshape(b, c, rgh, p, rgw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, rgh * rgw, c * p * p)
            tokens.append(self.first(patches.to(self.dtype)))
            rpos = torch.zeros(b, rgh, rgw, 3, device=img.device, dtype=torch.float32)
            rpos[..., 0] = k
            rpos[..., 1] = off_h + torch.arange(rgh, device=img.device, dtype=torch.float32)[:, None]
            rpos[..., 2] = off_w + torch.arange(rgw, device=img.device, dtype=torch.float32)[None, :]
            positions.append(rpos.view(b, rgh * rgw, 3))
        return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)
