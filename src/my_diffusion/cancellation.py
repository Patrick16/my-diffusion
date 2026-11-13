"""Cooperative cancellation of a generation.

A running generation cannot be killed from outside: it is a loop of GPU calls in a worker thread, and
stopping a thread in the middle of a CUDA kernel would leave the GPU and the model in an unknown state.
So cancellation is cooperative. The job queue gives the engine a threading.Event; the engine and the
sampler call checkpoint() at safe points (before each network step, between the big stages), and
checkpoint() raises GenerationCancelled when the event is set. Unwinding the exception frees the
activations, the model and the LoRA set stay consistent, and the next job starts from a clean state.

The cost of a cancel is therefore at most one network step (a fraction of a second to a few seconds
at 1080p). A VAE decode is not interrupted: it is short and has no safe place inside it."""

from __future__ import annotations

import threading


class GenerationCancelled(RuntimeError):
    """Raised at a checkpoint when the user cancelled the job. It is not an error of the request."""


def checkpoint(cancel_event: threading.Event | None) -> None:
    """Raise GenerationCancelled if the job was cancelled. Without an event (CLI, scripts) it does nothing."""
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("the generation was cancelled")
