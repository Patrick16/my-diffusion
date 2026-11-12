"""Sampler: turns noise into the image latent in several network calls.

Krea 2 is a "rectified flow" model. The idea: there is a straight line between pure noise
(sigma=1) and the finished image (sigma=0):

    x(sigma) = sigma * noise + (1 - sigma) * image

At any point of this line the network predicts the "velocity" v = noise - image, that is, the derivative
of x with respect to sigma. Therefore, to reduce the noise from sigma to sigma_next, one Euler
step is enough:

    x = x + (sigma_next - sigma) * v

The schedule (at which sigmas the steps are taken) is not uniform: it is shifted toward
large sigmas, because the composition is decided at the noisy steps, and the larger the image,
the more steps are worth spending there. The shift is set by the number mu:

    sigma = e^mu / (e^mu + (1/t - 1)),   t runs uniformly from 1 to 0

Note that mu appears in the exponent. mu=1.15 is a shift by a factor of e^1.15 = 3.16,
not by 1.15."""

from __future__ import annotations

import math
from typing import Callable

import torch

# Dependence of mu on the image size - as in Krea's reference code (and in Wan2GP):
# linear from 0.5 at 256 tokens (256x256 px) to 1.15 at 6400 tokens (1280x1280 px).
MU_MIN, MU_MAX = 0.5, 1.15
TOKENS_MIN, TOKENS_MAX = (256 // 16) ** 2, (1280 // 16) ** 2

# ComfyUI uses a constant shift of 1.15 for any size.
MU_COMFY = 1.15


def auto_mu(image_tokens: int) -> float:
    slope = (MU_MAX - MU_MIN) / (TOKENS_MAX - TOKENS_MIN)
    return MU_MIN + slope * (image_tokens - TOKENS_MIN)


def make_sigmas(steps: int, mu: float) -> list[float]:
    """steps+1 values from 1.0 to 0.0."""
    sigmas = []
    for i in range(steps + 1):
        t = 1.0 - i / steps
        sigmas.append(0.0 if t == 0.0 else math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)))
    return sigmas


def make_noise(seed: int, channels: int, height: int, width: int) -> torch.Tensor:
    """Starting noise. We generate it on the CPU in float32 - then the same seed gives
    the same noise on any GPU (and matches ComfyUI)."""
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, channels, height, width), generator=generator, dtype=torch.float32)


def guided_steps(steps: int, interval: tuple[float, float]) -> tuple[int, int]:
    """Which steps get CFG: the half-open range [first, last) of step indices for an interval given
    as fractions of the run. (0.0, 1.0) -> (0, steps); (0.0, 0.6) with 28 steps -> (0, 17)."""
    start, end = interval
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(f"guidance interval must satisfy 0 <= start <= end <= 1, got {interval}")
    return round(start * steps), round(end * steps)


@torch.inference_mode()
def sample(
    model,
    noise: torch.Tensor,
    cond: torch.Tensor,
    sigmas: list[float],
    uncond: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    on_step: Callable[[int, int], None] | None = None,
    guidance_interval: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Returns the latent of the finished image (in the model's normalized space).

    guidance_scale > 1 enables classifier-free guidance: the network is called twice -
    with the prompt and with the negative prompt - and the result is extrapolated from the second
    toward the first. The turbo model is distilled without CFG, so it needs guidance_scale = 1.

    guidance_interval limits CFG to a part of the run, given as fractions of the steps: (0.0, 1.0)
    is every step, (0.0, 0.6) is the first 60% of the steps. Outside the interval the network is
    called once per step instead of twice, so those steps cost half. Why this is reasonable: the
    negative prompt mostly steers the composition, which is settled at the noisy steps; at the
    last steps the two predictions nearly coincide and guidance mainly over-sharpens (see
    Kynkaanniemi et al., "Applying Guidance in a Limited Interval", 2024). It changes the image,
    so it is off unless asked for.

    What is deliberately not here: skipping steps with a cache (reusing the deep blocks' output of the
    previous step when the first block's output changed little). It was measured on this network and
    rejected: the reused part is 11-40% off even between neighbouring steps, and the resulting image
    is covered with a noisy mosaic texture, while a plain run with fewer steps stays clean."""
    x = noise
    use_cfg = guidance_scale != 1.0 and uncond is not None
    steps = len(sigmas) - 1
    first_guided, last_guided = guided_steps(steps, guidance_interval)
    # The text does not change from step to step: we run it through the network's text part once.
    cond = model.embed_text(cond)
    uncond = model.embed_text(uncond) if use_cfg else None
    for i in range(steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        s = torch.full((x.shape[0],), sigma, device=x.device, dtype=torch.float32)
        v = model(x, s, cond).float()
        if use_cfg and first_guided <= i < last_guided:
            v_uncond = model(x, s, uncond).float()
            v = v_uncond + guidance_scale * (v - v_uncond)
        x = x + (sigma_next - sigma) * v
        if on_step is not None:
            on_step(i + 1, steps)
    return x
