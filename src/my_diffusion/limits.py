"""Size limits of reference editing and what to do about memory: the policy, the streaming plan, the estimates.

Everything here is plain arithmetic (no torch, no GPU) so that tests can drive it; the engine supplies the real
numbers (free video memory, the sizes of the frames) and acts on the answers.

The policy (settings.edit_limits, settings.edit_max_pixels). Editing with references and outpainting used to be
refused above a size that fitted the 16 GB card (EDIT_MAX_TIER in engine.py). With streamed weights (task 4a:
see guides/performance.md, "Large sizes") every size up to 1440p runs, only slower, so the refusal became a choice:
  strict - the old behaviour: a size above the tier is refused;
  warn   - such a size is accepted and the job carries a warning with a rough time (the default);
  off    - accepted silently.
edit_max_pixels is a separate hard cap on the output frame of an edit, in every mode, for a server where somebody
else's jobs must not cost more than N.

Memory. The peak of the sampling in GB for T thousand tokens with n blocks streamed (measured on a 16 GB
laptop GPU, within 0.3 GB): 12.7 + 0.09 T - 0.42 (n - 1) for n >= 1, 12.7 + 0.09 T for n = 0. plan_streaming
inverts it: the fewest blocks that make the working set fit the memory that is free."""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- memory model
WEIGHTS_GB = 12.7          # the network, the LoRA pairs and the fixed buffers of a run
GB_PER_KTOKEN = 0.09       # activations per thousand tokens of the sequence
BLOCK_GB = 0.42            # one main block's weights: what streaming it frees
MAIN_BLOCKS = 28
MARGIN_GB = 0.5            # torch holds ~0.5 GB more than it uses, and the card is shared with the desktop
SPARE_BLOCKS = 1           # one block more than the arithmetic asks for: the rule is accurate to ~0.3 GB
EDIT_TEXT_TOKENS = 400     # the prompt and the vision tokens of the reference pictures, when only sizes are known
PATCH = 16                 # pixels per token side (8 per latent cell, 2 cells per patch)

# What strict mode accepts, by the number of references (the sizes that fitted the card without streaming).
STRICT_TIER = {1: "720", 2: "480"}
POLICIES = ("strict", "warn", "off")


class GenerationOutOfMemory(RuntimeError):
    """The GPU ran out of memory during a job. The job fails with this message; the engine is cleaned up and the
    next job runs normally."""


def image_tokens(width: int, height: int) -> int:
    """Tokens of a picture of this size in the network's sequence."""
    return (width // PATCH) * (height // PATCH)


def peak_gb(tokens: int, streamed: int) -> float:
    """The expected peak of the sampling, in GB, for a sequence of `tokens` with `streamed` blocks streamed."""
    return WEIGHTS_GB + GB_PER_KTOKEN * tokens / 1000 - BLOCK_GB * max(streamed - 1, 0)


def plan_streaming(
    tokens: int, available_gb: float, *, current: int = 0, margin_gb: float = MARGIN_GB, spare: int = SPARE_BLOCKS,
    total: int = MAIN_BLOCKS,
) -> int:
    """How many blocks to stream so that a sequence of `tokens` fits `available_gb` (the video memory the run may
    use, network included). The fewest that fit plus `spare`; 0 if it fits without; never fewer than `current`
    (blocks somebody already asked to stream) and never more than `total`. If even all blocks do not fit the
    answer is `total`: the caller goes on and the out-of-memory guard reports it."""
    budget = available_gb - margin_gb
    if peak_gb(tokens, 0) <= budget:
        return min(max(current, 0), total)
    needed = math.ceil((peak_gb(tokens, 1) - budget) / BLOCK_GB) + 1   # peak(n) <= budget  <=>  n >= 1 + excess / BLOCK_GB
    return min(max(needed + spare, current), total)


# --------------------------------------------------------------------------- how long it takes
# Sampling time of a turbo run (8 steps) by sequence length, measured on a 16 GB laptop GPU (guides/performance.md,
# "Large sizes"; that session was ~30% slower than the others, so these are on the safe side). Streaming adds ~1.5 s.
_SECONDS_AT_8_STEPS = [(8180, 27), (14420, 56), (16679, 77), (21895, 113), (24262, 131), (29159, 169)]


def estimate_seconds(tokens: int, steps: int = 8) -> int:
    """About how long the sampling takes, in seconds, rounded to 10: interpolated from the measurements and scaled
    by the steps. Below the first point the time falls linearly to zero, above the last it goes on at the last slope."""
    points = _SECONDS_AT_8_STEPS
    if tokens <= points[0][0]:
        seconds = points[0][1] * tokens / points[0][0]
    else:
        (t0, s0), (t1, s1) = next(((a, b) for a, b in zip(points, points[1:]) if tokens <= b[0]), points[-2:])
        seconds = s0 + (s1 - s0) * (tokens - t0) / (t1 - t0)
    return max(10, round(seconds * steps / 8 / 10) * 10)


def _duration(seconds: int) -> str:
    return f"{seconds} s" if seconds < 90 else f"{seconds / 60:.1f} minutes".replace(".0 ", " ")


# --------------------------------------------------------------------------- the policy
def check_edit_size(
    policy: str, max_pixels: int | None, width: int, height: int, references: int, tiers: dict[int, str] | None = None,
) -> str | None:
    """Applies the policy to an edit (or outpaint) with `references` pictures at width x height.

    Raises ValueError for what is refused: a frame above `max_pixels` in every mode, a size above the strict tier in
    strict mode. Returns None when the size is within the strict tier, otherwise (warn / off) the text of what
    strict mode would have said - the caller turns it into a warning in warn mode and drops it in off mode."""
    from .resolutions import max_pixels as tier_pixels   # not at the top: resolutions builds its tables on import

    if policy not in POLICIES:
        raise ValueError(f"edit_limits must be one of {list(POLICIES)}, got {policy!r}")
    pixels = width * height
    if max_pixels is not None and pixels > max_pixels:
        raise ValueError(
            f"an edit at {width}x{height} ({pixels / 1e6:.1f} megapixels) is above this server's limit of "
            f"{max_pixels / 1e6:.1f} megapixels (edit_max_pixels); use a smaller resolution"
        )
    tier = (tiers or STRICT_TIER)[min(references, 2)]
    if pixels <= tier_pixels(tier):
        return None
    text = f"an edit with {references} reference(s) at {width}x{height} does not fit the GPU; use resolution {tier} or lower"
    if policy == "strict":
        raise ValueError(text)
    return text


def large_edit_warning(width: int, height: int, references: int, tokens: int, steps: int, streamed: int) -> str:
    """The warning of a job accepted only because of the policy: it is large, and about how long it takes."""
    how = f" ({streamed} blocks of the network are streamed from RAM to fit the video memory)" if streamed else ""
    return (
        f"large edit: {width}x{height} with {references} reference(s) is above the sizes of strict mode "
        f"({', '.join(f'{n} -> {t}' for n, t in STRICT_TIER.items())}); about {tokens // 1000} thousand tokens, "
        f"so the sampling takes about {_duration(estimate_seconds(tokens, steps))}{how}"
    )


def free_vram_gb(device: str) -> float | None:
    """Free video memory of `device` in GB, or None when it cannot be told (no CUDA)."""
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return free / 2**30


def is_out_of_memory(exc: BaseException) -> bool:
    """torch's CUDA out-of-memory error, recognised by name so that this module needs no torch at import."""
    return type(exc).__name__ == "OutOfMemoryError" or (isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc))
