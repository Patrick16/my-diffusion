"""The block's working memory (transformer.CHUNK_TOKENS): the MLP, RoPE and the normalizations go over a long
sequence in pieces. The pieces must give the very same numbers as one pass, so a tiny network with dense weights
on the CPU is run both ways and compared bit for bit. The MLP is cut only above transformer.MLP_WHOLE_TOKENS."""

from __future__ import annotations

import pytest
import torch

from my_diffusion import transformer
from my_diffusion.transformer import Krea2Config, Krea2Transformer, RMSNorm, SwiGLU, apply_rope, build_rope, in_chunks
from my_diffusion.weights import QLinear

CFG = Krea2Config(features=64, heads=4, kvheads=2, layers=2, tdim=32, txtdim=40, txtlayers=3, txtheads=2, txtkvheads=2)


def fill(module: torch.nn.Module) -> None:
    """Random dense weights for every layer and random values for every parameter."""
    for layer in module.modules():
        if isinstance(layer, QLinear):
            bias = torch.randn(layer.out_features) * 0.1 if layer.has_bias else None
            layer.set_weight(torch.randn(layer.out_features, layer.in_features) * 0.1, "dense", None, bias)
    for param in module.parameters():
        param.data = torch.randn_like(param) * 0.1


@pytest.fixture
def model() -> Krea2Transformer:
    torch.manual_seed(0)
    net = Krea2Transformer(CFG)
    fill(net)
    return net


def test_in_chunks_writes_every_slice() -> None:
    x = torch.randn(2, 11, 3)
    seen: list[int] = []

    def double(part: torch.Tensor) -> torch.Tensor:
        seen.append(part.shape[-2])
        return part * 2

    assert torch.equal(in_chunks(double, x, size=4), x * 2)
    assert seen == [4, 4, 3]
    seen.clear()
    assert torch.equal(in_chunks(double, x, size=11), x * 2)   # fits: one call on the tensor itself
    assert seen == [11]
    assert torch.equal(in_chunks(lambda part: part * 2, torch.ones(5), size=2), torch.ones(5) * 2)   # no token axis: one call


def test_a_piece_may_change_the_last_axis() -> None:
    x = torch.randn(1, 9, 4)
    assert torch.equal(in_chunks(lambda part: part.repeat(1, 1, 2), x, size=2), x.repeat(1, 1, 2))


@pytest.mark.parametrize("chunk", [1, 5, 16])
def test_parts_of_the_block_are_the_same_in_pieces(monkeypatch, chunk: int) -> None:
    torch.manual_seed(1)
    norm, head_norm, mlp = RMSNorm(64), RMSNorm(16), SwiGLU(64, 4)
    for module in (norm, head_norm, mlp):
        fill(module)
    # bf16, the network's own type. In float32 the CPU's matrix product rounds a piece of the MLP differently from
    # the whole (by ~1e-6): the normalizations and RoPE are the same in any type, the MLP only up to that rounding.
    x = torch.randn(1, 37, 64).to(torch.bfloat16)
    heads = torch.randn(1, 4, 37, 16).to(torch.bfloat16)
    pos = torch.rand(1, 37, 3) * 20
    rope = build_rope(pos, (4, 6, 6), 1000.0)
    whole = norm(x), head_norm(heads), mlp(x), apply_rope(heads, rope)
    monkeypatch.setattr(transformer, "CHUNK_TOKENS", chunk)
    monkeypatch.setattr(transformer, "MLP_WHOLE_TOKENS", 0)
    pieces = norm(x), head_norm(heads), mlp(x), apply_rope(heads, rope)
    for a, b in zip(whole, pieces):
        assert torch.equal(a, b)


def test_the_network_is_the_same_in_pieces(monkeypatch, model: Krea2Transformer) -> None:
    torch.manual_seed(2)
    latent = torch.randn(1, 16, 8, 12)        # 4 x 6 = 24 picture tokens
    text = torch.randn(1, 5, CFG.features)
    sigma = torch.tensor([0.7])
    refs = [(torch.randn(1, 16, 6, 8), (0, 1))]   # + 12 reference tokens
    with torch.inference_mode():
        whole = model(latent, sigma, text), model(latent, sigma, text, refs=refs)
        monkeypatch.setattr(transformer, "CHUNK_TOKENS", 7)
        monkeypatch.setattr(transformer, "MLP_WHOLE_TOKENS", 0)
        pieces = model(latent, sigma, text), model(latent, sigma, text, refs=refs)
    for a, b in zip(whole, pieces):
        assert a.shape == latent.shape
        assert torch.equal(a, b)


def test_the_mlp_is_cut_only_above_its_own_threshold(monkeypatch) -> None:
    torch.manual_seed(3)
    mlp = SwiGLU(64, 4)
    fill(mlp)
    lengths: list[int] = []
    mlp.down.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[-2]))
    monkeypatch.setattr(transformer, "CHUNK_TOKENS", 4)
    monkeypatch.setattr(transformer, "MLP_WHOLE_TOKENS", 10)
    mlp(torch.randn(1, 10, 64))
    assert lengths == [10]          # at the threshold: one pass, as before the pieces existed
    lengths.clear()
    mlp(torch.randn(1, 11, 64))
    assert lengths == [4, 4, 3]
