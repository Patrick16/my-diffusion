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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import QLinear


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


class RMSNorm(nn.Module):
    """RMS normalization. A peculiarity of Krea 2: the file stores not the multiplier itself
    but its deviation from one, that is, the real weight = 1 + scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = _param(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in float32: a sum of squares over 6144 numbers loses precision in bf16
        return F.rms_norm(x.float(), (x.shape[-1],), weight=self.scale.float() + 1.0, eps=self.eps).to(x.dtype)


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
        return self.down(F.silu(self.gate(x)) * self.up(x))


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
    pairs = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = rope[..., 0] * pairs[..., 0] + rope[..., 1] * pairs[..., 1]
    return out.reshape(x.shape).to(x.dtype)


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

    def forward(self, x: torch.Tensor, rope: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.wq(x).view(b, n, self.heads, self.headdim).transpose(1, 2)
        k = self.wk(x).view(b, n, self.kvheads, self.headdim).transpose(1, 2)
        v = self.wv(x).view(b, n, self.kvheads, self.headdim).transpose(1, 2)
        q, k = self.qknorm.qnorm(q), self.qknorm.knorm(k)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.headdim)
        # the attention output is additionally weighted by a sigmoid gate
        return self.wo(out * torch.sigmoid(self.gate(x)))


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
            x = block(x)
        # 2) weighted sum of the 12 layers -> one vector per token
        x = x.view(b, seq, nlayers, d).transpose(2, 3)   # (b, seq, d, 12)
        x = self.projector(x).squeeze(-1)                # (b, seq, d)
        # 3) regular attention along the text
        for block in self.refiner_blocks:
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

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def forward(self, latent: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """latent: (B, 16, h, w) - noisy latent;  sigma: (B,) - noise level 0..1;
        context: (B, tokens, 12, 2560) - Qwen hidden states.  Returns the velocity (B, 16, h, w)."""
        cfg = self.cfg
        p = cfg.patch
        b, c, h, w = latent.shape
        gh, gw = h // p, w // p

        # latent -> patch tokens: (B, gh*gw, 16*2*2)
        img = latent.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
        img = self.first(img.to(self.dtype))

        t = self.tmlp(timestep_embedding(sigma, cfg.tdim).unsqueeze(1).to(self.dtype))   # (B, 1, features)
        tvec = self.tproj(t)                                                              # (B, 1, 6*features)

        txt = self.txtmlp(self.txtfusion(context.to(self.dtype)))                         # (B, tokens, features)
        ntxt = txt.shape[1]

        # positions: text gets all zeros, an image gets (0, row, column)
        pos = torch.zeros(b, ntxt + gh * gw, 3, device=latent.device, dtype=torch.float32)
        grid = pos[:, ntxt:].view(b, gh, gw, 3)
        grid[..., 1] = torch.arange(gh, device=latent.device, dtype=torch.float32)[:, None]
        grid[..., 2] = torch.arange(gw, device=latent.device, dtype=torch.float32)[None, :]
        rope = build_rope(pos, self.rope_axes, cfg.theta)

        x = torch.cat((txt, img), dim=1)
        for block in self.blocks:
            x = block(x, tvec, rope)

        out = self.last(x, t)[:, ntxt:]                                                   # image tokens only
        out = out.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return out
