"""Pinned RAM: why it is needed and how to allocate it without waste.

Why. Windows is free to move ordinary memory at any moment or page it out to the swap file,
so the GPU driver cannot read it directly: it first copies it to itself, then
sends it. Pinned memory the system undertakes not to touch, so the driver reads it at once.
A measurement on this machine: ~4 GB/s from ordinary memory versus 26-36 GB/s from pinned - the 12.6 GB
network is loaded onto the GPU in ~0.7 s instead of ~3 s.

Price. Pinned memory cannot be given to other programs and cannot be paged out
while we hold it. Therefore it must be allocated exactly as much as the models weigh.

Trap. PyTorch rounds each pinned chunk up to a power of two: ask for
12.6 GB and it pins 16; ask for 4.1 and it pins 8 (verified by measurement: the process occupied
24.8 GB instead of 17). This cannot be avoided with a setting, so we ask for chunks of exactly
power-of-two sizes ourselves and lay the tensors into them one after another. 12.6 GB turns into chunks of
8 + 4 + 0.5 + ... GB; only the unfilled tail of each chunk is wasted - in total hundreds of
megabytes instead of several gigabytes."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

MIN_CHUNK = 8 * 2**20   # we do not ask for smaller chunks: the tail of small items goes into one chunk of such size
ALIGN = 16              # the start of each tensor within a chunk is a multiple of this - suitable for any number type

_failed = False


def pinned_buffer(size: int) -> torch.Tensor | None:
    """A chunk of pinned memory of size bytes, or None if the system did not give it (then we warn once
    and do not try again: it is possible to work with ordinary memory too, just more slowly)."""
    global _failed
    if _failed:
        return None
    try:
        return torch.empty(size, dtype=torch.uint8, pin_memory=True)
    except RuntimeError as exc:
        _failed = True
        logger.warning("could not pin %.1f GB of RAM (%s) - continuing with ordinary memory, "
                       "the copy to the GPU will be slower", size / 2**30, str(exc).splitlines()[0])
        return None


def _pow2_floor(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def _pow2_ceil(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def plan_chunks(sizes: list[int]) -> list[tuple[int, int]]:
    """Lays out consecutive items (sizes in bytes) into chunks of power-of-two sizes.
    Returns [(chunk size, how many consecutive items were placed into it)].

    Rule: the next chunk is the largest power of two that does not exceed the remainder (so the chunk
    is certain to fill almost entirely); items are put in while they fit. Only the
    unfilled tail of a chunk is wasted - less than one item."""
    plan: list[tuple[int, int]] = []
    remaining = sum(sizes)
    i = 0
    while i < len(sizes):
        size = _pow2_floor(remaining) if remaining >= MIN_CHUNK else MIN_CHUNK
        size = max(size, _pow2_ceil(sizes[i]))   # the first item must fit
        used, j = 0, i
        while j < len(sizes) and used + sizes[j] <= size:
            used += sizes[j]
            j += 1
        plan.append((size, j - i))
        remaining -= used
        i = j
    return plan


def pin_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Copies of tensors in pinned memory, densely packed into power-of-two chunks.
    Tensors that are already pinned and empty are returned as is. If pinning is refused, the remaining
    ones are returned as is too."""
    result = list(tensors)
    todo = [i for i, t in enumerate(tensors) if t.numel() and not t.is_pinned()]
    sizes = [-(-tensors[i].numel() * tensors[i].element_size() // ALIGN) * ALIGN for i in todo]
    position = 0
    for chunk_size, count in plan_chunks(sizes):
        chunk = pinned_buffer(chunk_size)
        if chunk is None:
            break
        offset = 0
        for i, size in zip(todo[position : position + count], sizes[position : position + count]):
            source = tensors[i]
            nbytes = source.numel() * source.element_size()
            target = chunk[offset : offset + nbytes].view(source.dtype).reshape(source.shape)
            target.copy_(source)
            result[i] = target
            offset += size
        position += count
    return result
