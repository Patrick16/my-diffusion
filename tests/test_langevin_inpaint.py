"""Fast tests (no GPU, no models) of the inpaint refinement: the sampler's pre_step hook, the Langevin
iterations themselves (langevin.py) and the validation in the engine, the API and the CLI.

A fake network stands in for Krea 2. Its prediction is chosen so that the clean-picture guess x0 is a known
constant, which makes the equations' steady state known exactly."""

from __future__ import annotations

import base64
import io
import threading

import pytest
import torch
from PIL import Image

from my_diffusion import langevin, sampler
from my_diffusion.cancellation import GenerationCancelled
from my_diffusion.config import Settings
from my_diffusion.engine import Krea2Engine
from my_diffusion.langevin import MAX_ITERATIONS, LangevinInpaint, network_calls
from my_diffusion.schemas import GenerateRequest, GenerateResponse

PIC = Image.new("RGB", (64, 48), (200, 30, 30))


class ConstantGuess:
    """A network whose guess of the clean picture is `guess` at every point: v = (x - guess) / sigma."""

    def __init__(self, guess: float = 0.0) -> None:
        self.guess = guess
        self.calls = 0

    def embed_text(self, cond: torch.Tensor) -> torch.Tensor:
        return cond

    def __call__(self, x: torch.Tensor, s: torch.Tensor, cond: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls += 1
        return (x - self.guess) / s.view(-1, 1, 1, 1)


def velocity_of(network: ConstantGuess, sigma: float):
    s = torch.full((1,), sigma)
    return lambda x: network(x, s, None)


# ---- the sampler's hook --------------------------------------------------------------------------------------------

def test_without_pre_step_the_loop_is_the_plain_euler_loop():
    sigmas = [1.0, 0.6, 0.3, 0.0]
    noise = torch.randn(1, 4, 6, 6, generator=torch.Generator().manual_seed(1))
    got = sampler.sample(ConstantGuess(0.25), noise, torch.zeros(1), sigmas)
    x = noise
    for sigma, sigma_next in zip(sigmas, sigmas[1:]):
        x = x + (sigma_next - sigma) * ((x - 0.25) / sigma)
    assert torch.equal(got, x)


def test_pre_step_runs_before_every_step_but_the_last():
    seen: list[float] = []

    def pre_step(x, sigma, velocity):
        seen.append(sigma)
        velocity(x)   # one extra network call
        return x

    network = ConstantGuess()
    sampler.sample(network, torch.zeros(1, 4, 2, 2), torch.zeros(1), [0.9, 0.6, 0.3, 0.0], pre_step=pre_step)
    assert seen == [0.9, 0.6]
    assert network.calls == 3 + 2


def test_pre_step_result_is_what_the_step_starts_from():
    # The hook moves x to 1.0 everywhere; with the guess 0 the last step then scales it by sigma_next / sigma.
    out = sampler.sample(
        ConstantGuess(0.0), torch.zeros(1, 1, 2, 2), torch.zeros(1), [0.8, 0.4, 0.0],
        pre_step=lambda x, sigma, velocity: torch.ones_like(x),
    )
    assert torch.allclose(out, torch.zeros(1, 1, 2, 2))   # 1.0 -> 0.5 at sigma 0.4 -> 0.0
    mid = sampler.sample(
        ConstantGuess(0.0), torch.zeros(1, 1, 2, 2), torch.zeros(1), [0.8, 0.4],
        pre_step=lambda x, sigma, velocity: torch.ones_like(x),
    )
    assert torch.allclose(mid, torch.zeros(1, 1, 2, 2))   # a single step is the last one: the hook is not called


def test_cancel_inside_the_iterations_stops_before_the_next_network_call():
    event = threading.Event()
    network = ConstantGuess()

    def pre_step(x, sigma, velocity):
        velocity(x)
        event.set()        # asked to stop in the middle of the iterations
        velocity(x)        # must not reach the network
        return x

    with pytest.raises(GenerationCancelled):
        sampler.sample(network, torch.zeros(1, 4, 2, 2), torch.zeros(1), [0.9, 0.5, 0.0], pre_step=pre_step, cancel_event=event)
    assert network.calls == 1


# ---- the iterations ------------------------------------------------------------------------------------------------

def refiner(weight: torch.Tensor, original: float = 0.0, seed: int = 7, iterations: int = 2, **kw) -> LangevinInpaint:
    return LangevinInpaint(torch.full_like(weight, original), weight, seed, iterations, **kw)


def test_one_network_call_per_iteration_and_none_at_pure_noise():
    weight = torch.ones(1, 1, 4, 4)
    network = ConstantGuess()
    x = torch.randn(1, 1, 4, 4)
    refiner(weight, iterations=3)(x, 0.7, velocity_of(network, 0.7))
    assert network.calls == 3
    same = refiner(weight, iterations=3)(x, 1.0, velocity_of(network, 1.0))
    assert network.calls == 3 and same is x   # sigma = 1: nothing to refine, x untouched


@pytest.mark.parametrize("steps, iterations, pure, expected", [
    (8, 2, True, 8 + 6 * 2), (8, 2, False, 8 + 7 * 2), (8, 5, True, 38), (1, 2, True, 1), (2, 3, True, 2), (2, 3, False, 5),
])
def test_network_calls_matches_a_real_run(steps, iterations, pure, expected):
    assert network_calls(steps, iterations, pure) == expected
    top = 1.0 if pure else 0.9
    sigmas = [top * (1 - i / steps) for i in range(steps + 1)]
    weight = torch.ones(1, 1, 2, 2)
    network = ConstantGuess()
    sampler.sample(network, torch.zeros(1, 1, 2, 2), torch.zeros(1), sigmas, pre_step=refiner(weight, iterations=iterations))
    assert network.calls == expected


def test_same_seed_same_result_other_seed_other_result():
    weight = torch.ones(1, 1, 8, 8)
    weight[..., :4] = 0.0
    x = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(3))

    def run(seed: int) -> torch.Tensor:
        return refiner(weight, seed=seed, iterations=3)(x, 0.6, velocity_of(ConstantGuess(0.5), 0.6))

    assert torch.equal(run(11), run(11))
    assert not torch.equal(run(11), run(12))


def test_noise_is_not_the_starting_noise_of_the_run():
    # The engine seeds both from the request's seed; the salt must keep the two streams apart.
    shape = (1, 16, 8, 8)
    start = sampler.make_noise(42, 16, 8, 8)
    first = refiner(torch.ones(1, 1, 8, 8), seed=42)._noise(torch.empty(shape))
    assert first.shape == start.shape and not torch.allclose(first, start)


def test_generated_area_settles_on_the_guess_at_the_level_of_sigma():
    # x0 is the constant 2.0, so the steady state is exact: mean (1 - sigma) * 2.0, standard deviation sigma.
    sigma = 0.6
    weight = torch.ones(1, 1, 200, 200)
    step = refiner(weight, iterations=MAX_ITERATIONS)
    x = torch.full((1, 1, 200, 200), -3.0)   # far from the steady state on purpose
    for _ in range(12):
        x = step(x, sigma, velocity_of(ConstantGuess(2.0), sigma))
    assert x.mean().item() == pytest.approx((1 - sigma) * 2.0, abs=0.02)
    assert x.std().item() == pytest.approx(sigma, abs=0.02)


def test_generated_area_keeps_a_level_it_is_already_on():
    sigma = 0.5
    weight = torch.ones(1, 1, 200, 200)
    x = sigma * torch.randn(1, 1, 200, 200, generator=torch.Generator().manual_seed(5)) + (1 - sigma) * 1.5
    x = refiner(weight, iterations=2)(x, sigma, velocity_of(ConstantGuess(1.5), sigma))
    assert x.mean().item() == pytest.approx((1 - sigma) * 1.5, abs=0.02)
    assert x.std().item() == pytest.approx(sigma, abs=0.02)


def test_kept_area_stays_around_the_original_when_the_network_agrees():
    # The network's guess equals the original: no disagreement, so the kept area's mean is the original's
    # point of this level and it does not drift.
    sigma, original = 0.6, 1.0
    weight = torch.zeros(1, 1, 200, 200)
    x = sigma * torch.randn(1, 1, 200, 200, generator=torch.Generator().manual_seed(9)) + (1 - sigma) * original
    step = refiner(weight, original=original, iterations=MAX_ITERATIONS)
    for _ in range(6):
        x = step(x, sigma, velocity_of(ConstantGuess(original), sigma))
    assert x.mean().item() == pytest.approx((1 - sigma) * original, abs=0.02)
    assert 0.3 * sigma < x.std().item() < 1.1 * sigma   # noise of the level's order, neither gone nor blown up


def test_kept_area_pushes_away_from_a_wrong_guess():
    # Guess 0, original 1: the kept area's target is 1 + L * (1 - 0), far beyond the original, away from the guess.
    sigma = 0.6
    weight = torch.zeros(1, 1, 50, 50)
    x = torch.full((1, 1, 50, 50), (1 - sigma) * 1.0)
    moved = refiner(weight, original=1.0, iterations=1)(x, sigma, velocity_of(ConstantGuess(0.0), sigma))
    assert moved.mean().item() > x.mean().item() + 0.2


def test_feathered_weight_mixes_the_two_areas_per_cell():
    weight = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 1, 3)
    step = refiner(weight)
    assert torch.allclose(step.kept, torch.tensor([1.0, 0.5, 0.0]).view(1, 1, 1, 3))
    assert torch.allclose(step.rate, torch.tensor([1 + langevin.GUIDANCE, 1 + langevin.GUIDANCE / 2, 1.0]).view(1, 1, 1, 3))


def test_stays_finite_through_a_whole_run():
    weight = torch.ones(1, 1, 16, 16)
    weight[..., 8:] = 0.0
    latent0 = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(2))
    noise = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(4))
    sigmas = sampler.make_sigmas(8, 1.0)

    def post_step(x, sigma_next):
        return weight * x + (1 - weight) * sampler.add_noise(latent0, noise, sigma_next)

    out = sampler.sample(
        ConstantGuess(0.3), noise, torch.zeros(1), sigmas, post_step=post_step,
        pre_step=LangevinInpaint(latent0, weight, 1, MAX_ITERATIONS),
    )
    assert torch.isfinite(out).all()
    assert torch.equal(out[..., 8:], latent0[..., 8:])          # the kept half is the original
    assert torch.allclose(out[..., :8], torch.full((1, 4, 16, 8), 0.3), atol=1e-4)   # the generated half is the guess


@pytest.mark.parametrize("iterations", [0, MAX_ITERATIONS + 1, -1])
def test_rejects_an_iteration_count_out_of_range(iterations):
    with pytest.raises(ValueError, match="iterations"):
        LangevinInpaint(torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2), 1, iterations)


# ---- validation ----------------------------------------------------------------------------------------------------

def _call(**kw):
    args = dict(prompt="p", negative_prompt="", resolution="480", aspect_ratio="16:9", seed=1, steps=1,
                guidance_scale=1.0, shift=None, loras=[], enhance=False, guidance_interval=(0.0, 1.0))
    args.update(kw)
    return Krea2Engine(Settings())._generate(**args)   # not loaded: must fail in validation


@pytest.mark.parametrize("kw", [
    dict(inpaint_iterations=2),                                    # no mask at all
    dict(inpaint_iterations=2, init_image=PIC),                    # image-to-image, still no mask
    dict(inpaint_iterations=MAX_ITERATIONS + 1, init_image=PIC, mask=PIC),
    dict(inpaint_iterations=-1, init_image=PIC, mask=PIC),
])
def test_engine_rejects_iterations_without_a_mask_or_out_of_range(kw):
    with pytest.raises(ValueError, match="inpaint_iterations"):
        _call(**kw)


def test_schema_default_range_and_echo():
    assert GenerateRequest(prompt="x").inpaint_iterations == 0
    assert GenerateRequest(prompt="x", inpaint_iterations=MAX_ITERATIONS).inpaint_iterations == MAX_ITERATIONS
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", inpaint_iterations=MAX_ITERATIONS + 1)
    base = dict(image_path="a", width=1, height=1, seed=1, prompt="p", enhanced_prompt=None, steps=1, guidance_scale=1.0,
                guidance_interval=[0, 1], shift=1.0, checkpoint="c", matmul="exact", loras=[], timings={})
    assert GenerateResponse(**base).inpaint_iterations == 0
    assert GenerateResponse(**base, inpaint_iterations=2).inpaint_iterations == 2


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from my_diffusion.api import app
    return TestClient(app)


def _b64() -> str:
    buf = io.BytesIO()
    PIC.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("with_init_image", [False, True])
def test_api_iterations_without_mask_is_400(client, with_init_image):
    body = {"prompt": "x", "inpaint_iterations": 2, **({"init_image": _b64()} if with_init_image else {})}
    r = client.post("/generate", json=body)
    assert r.status_code == 400 and "inpaint_iterations" in r.json()["detail"]


def test_api_iterations_with_mask_passes_validation(client):
    r = client.post("/generate", json={"prompt": "x", "init_image": _b64(), "mask": _b64(), "inpaint_iterations": 2})
    assert r.status_code == 503   # past validation, stops at "models are loading"


def test_cli_iterations_need_a_mask(monkeypatch, capsys):
    from my_diffusion import cli
    monkeypatch.setattr("sys.argv", ["my-diffusion-generate", "prompt", "--init-image", "x.png", "--inpaint-iterations", "2"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2 and "--inpaint-iterations needs --mask" in capsys.readouterr().err
