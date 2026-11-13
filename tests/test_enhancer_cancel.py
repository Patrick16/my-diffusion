"""The prompt enhancer's stopping criterion: Qwen stops writing once the job is cancelled. No model is needed."""

from __future__ import annotations

import threading

import torch

from my_diffusion.text_encoder import stop_when


def test_stops_only_after_the_flag_is_set():
    event = threading.Event()
    criteria = stop_when(event.is_set)
    tokens = torch.zeros(2, 5, dtype=torch.long)   # a batch of two sequences
    assert not criteria(tokens, torch.zeros(2, 10)).any()
    event.set()
    assert criteria(tokens, torch.zeros(2, 10)).all()
