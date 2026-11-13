"""The sampler honours cancellation between steps. A fake network stands in for Krea 2, so no GPU is needed."""

from __future__ import annotations

import threading

import pytest
import torch

from my_diffusion import sampler
from my_diffusion.cancellation import GenerationCancelled

SIGMAS = [1.0, 0.75, 0.5, 0.25, 0.0]   # four steps


class FakeNetwork:
    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, cond: torch.Tensor) -> torch.Tensor:
        return cond

    def __call__(self, x: torch.Tensor, s: torch.Tensor, cond: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(x)


def run(network: FakeNetwork, **kwargs) -> torch.Tensor:
    noise = torch.zeros(1, 4, 2, 2)
    return sampler.sample(network, noise, torch.zeros(1), SIGMAS, **kwargs)


def test_without_an_event_all_steps_run():
    network = FakeNetwork()
    run(network)
    assert network.calls == 4


def test_cancel_stops_before_the_next_network_call():
    network = FakeNetwork()
    event = threading.Event()

    def on_step(done: int, total: int) -> None:
        if done == 2:   # asked to stop after two steps: the third must not start
            event.set()

    with pytest.raises(GenerationCancelled):
        run(network, on_step=on_step, cancel_event=event)
    assert network.calls == 2


def test_progress_is_reported_after_every_step():
    seen: list[tuple[int, int]] = []
    run(FakeNetwork(), on_step=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
