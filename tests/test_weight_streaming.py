"""Streamed weights: blocks of the network that stay in RAM and get their weights on the GPU only while they run
(Krea2Transformer.stream_weights, weights.streamed_weights, memory.Resident.with_home_copy with stay_home).

There is no GPU in the tests. torch's "meta" device stands in for it: a tensor moved there keeps its shape and
loses its data, which is enough to see which tensors travel and which stay."""

from __future__ import annotations

import pytest
import torch

from my_diffusion.memory import Resident
from my_diffusion.transformer import Krea2Config, Krea2Transformer
from my_diffusion.weights import QLinear, streamed_weights

CFG = Krea2Config(features=64, heads=4, kvheads=2, layers=3, tdim=32, txtdim=40, txtlayers=3, txtheads=2, txtkvheads=2)
META = torch.device("meta")


@pytest.fixture
def model() -> Krea2Transformer:
    torch.manual_seed(0)
    net = Krea2Transformer(CFG)
    for layer in net.modules():
        if isinstance(layer, QLinear):
            bias = torch.randn(layer.out_features) * 0.1 if layer.has_bias else None
            layer.set_weight(torch.randn(layer.out_features, layer.in_features) * 0.1, "dense", None, bias)
    return net


def tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.data for p in module.parameters()] + list(module.buffers())


def devices(module: torch.nn.Module) -> set[str]:
    return {t.device.type for t in tensors(module)}


def test_stream_weights_marks_the_first_blocks(model: Krea2Transformer) -> None:
    assert model.streamed_weight_blocks == 0
    model.stream_weights(2)
    assert [getattr(block, "weights_streamed", False) for block in model.blocks] == [True, True, False]
    assert model.streamed_weight_blocks == 2
    model.stream_weights(0)
    assert model.streamed_weight_blocks == 0
    with pytest.raises(ValueError):
        model.stream_weights(4)
    with pytest.raises(ValueError):
        model.stream_weights(-1)


def layers_only() -> torch.nn.ModuleList:
    """Three "blocks" made of compressed-weight layers alone: their tensors are buffers, which can sit on the meta
    device (torch refuses to put a parameter's data there). The parameters take the same path on a real GPU."""
    blocks = torch.nn.ModuleList()
    for _ in range(3):
        block = torch.nn.Sequential(QLinear(8, 8), QLinear(8, 4, bias=False))
        for layer in block:
            layer.set_weight(torch.randn(layer.out_features, layer.in_features), "dense", None, torch.randn(layer.out_features) if layer.has_bias else None)
        blocks.append(block)
    return blocks


def test_a_streamed_block_stays_home_when_the_network_moves() -> None:
    blocks = layers_only()
    resident = Resident.with_home_copy(blocks, stay_home=lambda: [b for b in blocks if getattr(b, "weights_streamed", False)])
    blocks[0].weights_streamed = True
    resident.to_gpu("meta")
    assert devices(blocks[0]) == {"cpu"}
    assert devices(blocks[1]) == devices(blocks[2]) == {"meta"}
    resident.to_cpu()
    assert devices(blocks) == {"cpu"}
    blocks[0].weights_streamed = False   # the marks are read on every move
    resident.to_gpu("meta")
    assert devices(blocks) == {"meta"}


def test_weights_travel_only_for_the_duration_of_the_block() -> None:
    block = layers_only()[0]
    home = tensors(block)
    with streamed_weights(block, META):   # not marked: nothing moves
        assert devices(block) == {"cpu"}
    block.weights_streamed = True
    with streamed_weights(block, META):
        assert devices(block) == {"meta"}
    assert all(a is b for a, b in zip(tensors(block), home))   # the very same home tensors are back
    with pytest.raises(RuntimeError, match="boom"):
        with streamed_weights(block, META):
            raise RuntimeError("boom")
    assert all(a is b for a, b in zip(tensors(block), home))   # also after an error inside the block


def test_the_result_does_not_depend_on_streaming(model: Krea2Transformer) -> None:
    torch.manual_seed(1)
    latent, text, sigma = torch.randn(1, 16, 8, 12), torch.randn(1, 5, CFG.features), torch.tensor([0.7])
    with torch.inference_mode():
        whole = model(latent, sigma, text)
        model.stream_weights(3)
        streamed = model(latent, sigma, text)   # on the CPU the weights are already "there": the same pass
    assert torch.equal(whole, streamed)
