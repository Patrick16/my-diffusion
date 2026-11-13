"""Choice of the attention kernel (transformer.set_attention_kernel). A tiny attention layer with dense weights on
the CPU and a fake sage2 function stand in for the real ones, so neither a GPU nor the sageattention package is needed."""

from __future__ import annotations

import builtins

import pytest
import torch

from my_diffusion import transformer
from my_diffusion.config import Settings
from my_diffusion.transformer import ATTENTION_KERNELS, Attention, set_attention_kernel

DIM, HEADS, KVHEADS = 32, 4, 2


def make_attention() -> Attention:
    torch.manual_seed(0)
    attn = Attention(DIM, HEADS, KVHEADS)
    for layer in (attn.wq, attn.wk, attn.wv, attn.gate, attn.wo):
        layer.set_weight(torch.randn(layer.out_features, layer.in_features) * 0.1, "dense", None, None)
    return attn


class FakeSage:
    """Records the calls and answers with exact attention."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def __call__(self, q, k, v, tensor_layout, is_causal):
        assert tensor_layout == "HND" and is_causal is False
        assert q.shape[1] == k.shape[1] == v.shape[1] == HEADS   # GQA is expanded before the kernel
        self.lengths.append(q.shape[2])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


@pytest.fixture
def fake_sage(monkeypatch) -> FakeSage:
    fake = FakeSage()
    monkeypatch.setattr(transformer, "_sageattn", fake)
    return fake


def test_settings_accept_every_kernel() -> None:
    assert ATTENTION_KERNELS == ("cudnn", "default", "sage2")
    for kernel in ATTENTION_KERNELS:
        assert Settings(attention=kernel).attention == kernel
    assert Settings.model_fields["attention"].default == "cudnn"   # sage2 is opt-in
    assert Settings().attention_sage2_min_tokens == 6000
    with pytest.raises(ValueError):
        Settings(attention_sage2_min_tokens=-1)


def test_unknown_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown attention kernel"):
        set_attention_kernel(make_attention(), "flash")


def test_sage2_without_the_package_fails_at_selection(monkeypatch) -> None:
    real_import = builtins.__import__

    def no_sage(name, *args, **kwargs):
        if name == "sageattention":
            raise ImportError("No module named 'sageattention'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(transformer, "_sageattn", None)
    monkeypatch.setattr(builtins, "__import__", no_sage)
    attn = make_attention()
    with pytest.raises(RuntimeError, match="uv sync --group sage2"):
        set_attention_kernel(attn, "sage2")
    assert attn.kernel == "default"   # nothing was switched


def test_sage2_is_called_and_matches_the_other_kernels(fake_sage) -> None:
    attn = make_attention()
    x = torch.randn(1, 10, DIM)
    outputs = {}
    for kernel in ATTENTION_KERNELS:
        set_attention_kernel(attn, kernel)
        outputs[kernel] = attn(x)
    assert fake_sage.lengths == [10]   # only the sage2 run reached it
    assert torch.allclose(outputs["sage2"], outputs["default"], atol=1e-5)
    assert torch.allclose(outputs["cudnn"], outputs["default"], atol=1e-5)


def test_sage2_leaves_short_sequences_to_cudnn(fake_sage) -> None:
    attn = make_attention()
    set_attention_kernel(attn, "sage2", sage_min_tokens=8)
    attn(torch.randn(1, 7, DIM))
    assert fake_sage.lengths == []
    attn(torch.randn(1, 8, DIM))
    assert fake_sage.lengths == [8]
    set_attention_kernel(attn, "cudnn")   # switching back resets the threshold
    assert attn.sage_min_tokens == 0
