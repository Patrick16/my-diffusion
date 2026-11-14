"""Inpaint refinement: a few Langevin iterations at every noise level, before the sampler's step.

The plain inpaint (engine.py, post_step) puts the kept area back after each step and lets the network
draw the rest. The masked area then gets exactly one network call per noise level, which is too little to
agree with what surrounds it: on a fox cut in half the network draws a second head, on a band through its
body a fifth leg. More steps do not help, because every step has the same weakness.

The cure is to stay on a noise level for a while. At a level sigma the network gives a guess of the clean
picture, x0 = x - sigma * v. Langevin dynamics uses that guess to move x around WITHOUT lowering the noise:
x is pulled toward the guess and fresh noise is added, in a proportion that keeps the amount of noise at
sigma. After each move the network looks at the whole picture again, so the masked area and the kept area
get several rounds to settle on one consistent picture before the sampler takes its step.

The method is LanPaint: Zheng, Lan, Wang, "LanPaint: Training-Free Diffusion Inpainting with Asymptotically
Exact and Fast Conditional Sampling", TMLR 2025 (arXiv 2502.03491). This file is an independent
implementation of the equations, not a port of the authors' code; where it differs is said below.

The equations, in the sampler's own variables (x = sigma * noise + (1 - sigma) * picture). With the time
of the iterations called tau, m the share of the original in a latent cell (1 = kept, 0 = generated; the
feathered edge is in between), y the original latent and L the constant GUIDANCE:

    dx = (-a * x + c) dtau + sigma * sqrt(2) dW,     a = 1 + L * m
    c  = (1 - sigma) * (x0 + (1 + L) * m * (y - x0)) + L * m * x

  - generated area (m = 0): dx = ((1 - sigma) * x0 - x) dtau + noise. Left alone it settles at
    "(1 - sigma) * x0 plus noise of size sigma" - a point of this noise level that agrees with the guess.
  - kept area (m = 1): the same, but the target is y + L * (y - x0) instead of x0. The kept area is not
    simply held to the original: it is pushed AWAY from the network's guess by L times their disagreement.
    This is how the original talks back to the network (the paper's bidirectional guidance): where the
    network's idea of the kept area is wrong, the kept area exaggerates the error, and the network's next
    guess of the masked area has to account for it.

One iteration is one network call. The term -a * x is solved exactly (it is stiff in the kept area, where
a = 1 + L), and c is held constant during a move:

    x_new = e * x + (1 - e) / a * c + sigma * sqrt((1 - e^2) / a) * fresh_noise,     e = exp(-a * h)

The first iteration is one such move of length STEP with c taken at the start. Every later one moves half
of STEP with the previous c, asks the network for c there, and finishes as if that middle value had been in
force for the whole move (a midpoint rule: second order in STEP for the price of one call).

Differences from the reference implementation (Wan2GP's shared/inpainting/lanpaint.py):
  - no velocity variable. The reference integrates an underdamped oscillator, but with the friction it sets
    for Krea 2 the velocity forgets itself 225 times within one move, so the result is the overdamped
    equation above; dropping it removes most of the code and all of its special functions;
  - the midpoint correction is the exact response of the equation, not an impulse;
  - the noise comes from a generator seeded by the request, so the same seed gives the same picture
    (the reference draws from the global generator);
  - nothing is done at sigma = 1: there (1 - sigma) = 0, the network's guess has no weight in c, and the
    iterations would only replace noise by other noise at the price of network calls.

Measured on the turbo checkpoint, 8 steps, a fox on snow, by eye (scripts/check_inpaint_iterations.py).
With this implementation, seeds 42-44, a clean result being one without a second head, an extra leg or a
separate picture inside the mask:
    scene                       plain   2 iterations   5 iterations
    hat on the head              1/3        3/3            3/3
    a band through the body      0/3        3/3            2/3
    the right half of the fox    0/3        1/3            1/3     (seeds 45-50 with 2: 5/6; Wan2GP's code: 8/9 on 42-50)
    removing the whole fox       3/3        3/3            0/3     (a separate landscape with a horizon)
The gain is real for completing an object and is not for filling a hole; with 5 and more iterations the
picture inside a large mask goes its own way. Sampling takes 2.5 times longer with 2 iterations and 4.7
times with 5 (8 steps). Samples this small cannot tell this implementation from Wan2GP's apart: they agree
on the scenes and differ on single seeds. So this is off by default and meant for 2 iterations."""

from __future__ import annotations

from typing import Callable

import torch

# L in the equations: how hard the kept area pushes back. 16 is what Wan2GP uses for Krea 2 and what was
# measured; the paper's authors suggest 4-10, and 4 and 8 gave no better pictures here.
GUIDANCE = 16.0

# Length of one iteration in the time of the dynamics (the generated area forgets e^-STEP = 82% -> 18% of
# itself per iteration). Wan2GP's value for Krea 2.
STEP = 0.2

# More iterations were not better in the measurements (10 lost to 5 on two of four scenes) and each one
# costs a network call per step.
MAX_ITERATIONS = 5

# XOR-ed into the seed of the iterations' noise, so that it is not the run's starting noise over again
# (the same seed and shape would give the very same tensor).
NOISE_SALT = 0x1A27_E417


class LangevinInpaint:
    """The sampler's pre_step for an inpaint: __call__(x, sigma, velocity) -> x after the iterations.

    latent0 - the original picture's clean latent (the y of the equations).
    weight  - the inpaint blend weight, 1 = generate, 0 = keep (imaging.mask_to_latent), on x's device.
    seed    - the request's seed; the noise of the iterations is derived from it.

    x must already hold the kept area at the level sigma (the engine's start and post_step see to that)."""

    def __init__(self, latent0: torch.Tensor, weight: torch.Tensor, seed: int, iterations: int,
                 guidance: float = GUIDANCE, step: float = STEP) -> None:
        if not 1 <= iterations <= MAX_ITERATIONS:
            raise ValueError(f"iterations must be from 1 to {MAX_ITERATIONS}, got {iterations}")
        self.latent0 = latent0
        self.iterations = iterations
        self.guidance = guidance
        self.step = step
        self.kept = 1.0 - weight                      # m
        self.rate = 1.0 + guidance * self.kept        # a
        # Generated on the CPU like the starting noise (sampler.make_noise): the same on any GPU.
        self.generator = torch.Generator("cpu").manual_seed(seed ^ NOISE_SALT)

    def _noise(self, like: torch.Tensor) -> torch.Tensor:
        return torch.randn(like.shape, generator=self.generator, dtype=torch.float32).to(like)

    def _force(self, x: torch.Tensor, sigma: float, velocity: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """c of the equations at the point x. One network call."""
        x0 = x - sigma * velocity(x)
        target = x0 + (1.0 + self.guidance) * self.kept * (self.latent0 - x0)
        return (1.0 - sigma) * target + self.guidance * self.kept * x

    def _move(self, x: torch.Tensor, sigma: float, force: torch.Tensor, length: float) -> torch.Tensor:
        """The exact solution of the equation over `length` with c = force held constant."""
        decay = torch.exp(-self.rate * length)
        spread = sigma * torch.sqrt((1.0 - decay * decay) / self.rate)
        return decay * x + (1.0 - decay) / self.rate * force + spread * self._noise(x)

    def __call__(self, x: torch.Tensor, sigma: float, velocity: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if sigma >= 1.0:   # pure noise: the network's guess has no weight yet (see the module description)
            return x
        half = self.step / 2
        # What a change of c in the first half of a move adds by the end of the move.
        decay_half = torch.exp(-self.rate * half)
        carry = decay_half * (1.0 - decay_half) / self.rate
        force = None
        for _ in range(self.iterations):
            if force is None:
                force = self._force(x, sigma, velocity)
                x = self._move(x, sigma, force, self.step)
            else:
                middle = self._move(x, sigma, force, half)
                new_force = self._force(middle, sigma, velocity)
                x = self._move(middle, sigma, new_force, half) + carry * (new_force - force)
                force = new_force
        return x


def network_calls(steps: int, iterations: int, from_pure_noise: bool) -> int:
    """How many times the network runs in an inpaint with `iterations` (without CFG; twice that with it).

    The iterations run before every step but the last, and not at sigma = 1 (the first step of a run that
    starts from pure noise)."""
    refined = max(0, steps - 1 - (1 if from_pure_noise else 0))
    return steps + refined * iterations

