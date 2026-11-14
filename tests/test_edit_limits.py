"""The policy for large edits (limits.py, settings edit_limits / edit_max_pixels), the automatic streaming of network
blocks, the warnings in the job, and the out-of-memory guard. No GPU: the memory numbers are fed in, the engine's
heavy parts are stubs."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from my_diffusion import limits
from my_diffusion.config import Settings
from my_diffusion.engine import EDIT_MAX_TIER, Krea2Engine
from my_diffusion.jobs import Scheduler, Slot
from my_diffusion.store import JobStore


# ---- the memory model -----------------------------------------------------------------------------------------
def test_peak_follows_the_measured_rule():
    assert limits.peak_gb(10_000, 0) == pytest.approx(12.7 + 0.9)
    assert limits.peak_gb(29_000, 1) == pytest.approx(12.7 + 0.09 * 29)
    assert limits.peak_gb(29_000, 5) == pytest.approx(12.7 + 0.09 * 29 - 0.42 * 4)


@pytest.mark.parametrize("tokens", [3_000, 8_000, 14_000, 22_000, 29_000])
def test_streaming_plan_is_the_fewest_blocks_that_fit_plus_spare(tokens):
    available = 14.7
    n = limits.plan_streaming(tokens, available)
    budget = available - limits.MARGIN_GB
    if limits.peak_gb(tokens, 0) <= budget:
        assert n == 0
    else:
        assert limits.peak_gb(tokens, n) <= budget                              # it fits
        assert limits.peak_gb(tokens, n - limits.SPARE_BLOCKS - 1) > budget     # one block fewer than needed would not


def test_streaming_plan_matches_the_guide_cases():
    assert limits.plan_streaming(7_559, 14.7) == 0          # edit 720, one reference
    assert 1 <= limits.plan_streaming(21_895, 14.7) <= 6    # two references at 1080: 4 are enough, a spare block more
    assert limits.plan_streaming(29_159, 14.7) <= 8


def test_streaming_plan_keeps_what_was_asked_and_is_capped():
    assert limits.plan_streaming(3_000, 14.7, current=6) == 6
    assert limits.plan_streaming(500_000, 14.7) == limits.MAIN_BLOCKS
    assert limits.plan_streaming(29_000, 3.0, current=0) == limits.MAIN_BLOCKS


# ---- the policy -----------------------------------------------------------------------------------------------
def test_strict_refuses_above_the_tier_with_the_old_message():
    with pytest.raises(ValueError, match=r"an edit with 2 reference\(s\) at 1280x720 does not fit the GPU; use resolution 480 or lower"):
        limits.check_edit_size("strict", None, 1280, 720, 2, EDIT_MAX_TIER)
    assert limits.check_edit_size("strict", None, 1280, 720, 1, EDIT_MAX_TIER) is None


@pytest.mark.parametrize("policy", ["warn", "off"])
def test_warn_and_off_accept_and_report_what_strict_would_say(policy):
    text = limits.check_edit_size(policy, None, 1920, 1088, 2, EDIT_MAX_TIER)
    assert text and "does not fit the GPU" in text


def test_max_pixels_applies_in_every_mode():
    for policy in limits.POLICIES:
        with pytest.raises(ValueError, match="edit_max_pixels"):
            limits.check_edit_size(policy, 1_000_000, 1280, 720 + 16 * 20, 1, EDIT_MAX_TIER)
        assert limits.check_edit_size(policy, 2_000_000, 640, 480, 1, EDIT_MAX_TIER) is None


def test_estimate_is_about_the_measured_times():
    assert limits.estimate_seconds(8_180) == 30       # measured 27 s
    assert limits.estimate_seconds(29_159) == 170     # measured 169 s
    assert limits.estimate_seconds(29_159, steps=16) == 340
    warning = limits.large_edit_warning(1920, 1088, 2, 21_895, 8, 4)
    assert "about" in warning and "4 blocks" in warning and "1920x1088" in warning


def test_the_settings_default_to_warn_without_a_cap():
    settings = Settings(_env_file=None)
    assert settings.edit_limits == "warn" and settings.edit_max_pixels is None
    with pytest.raises(ValueError):
        Settings(edit_limits="sometimes")


# ---- the engine: automatic streaming ---------------------------------------------------------------------------
class FakeNetwork:
    def __init__(self):
        self.streamed_weight_blocks = 0

    def stream_weights(self, blocks):
        self.streamed_weight_blocks = blocks


class StubEngine(Krea2Engine):
    def __init__(self, **settings):
        super().__init__(Settings(device="cpu", **settings))
        self.model = FakeNetwork()
        self.outcomes: list = []   # per call of _generate: an exception to raise, or a result

    @property
    def loaded(self):
        return True

    def _generate(self, *args, **kwargs):
        self._run_size = (2688, 1360, 24_262)
        self.model.stream_weights(9)   # as _plan_streaming would for this job
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def free_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    box = {"gb": 14.7}
    monkeypatch.setattr(limits, "free_vram_gb", lambda device: box["gb"])
    return box


def test_plan_streaming_streams_on_the_engine_and_returns_the_count(free_memory):
    engine = StubEngine()
    assert engine._plan_streaming(7_559) == 0 and engine.model.streamed_weight_blocks == 0
    n = engine._plan_streaming(29_159)
    assert n > 0 and engine.model.streamed_weight_blocks == n


def test_plan_streaming_counts_the_network_already_on_the_gpu(free_memory):
    engine = StubEngine()
    free_memory["gb"] = 2.0   # the network (12.7 GB) sits on the GPU: what is free is what is left over
    engine.gpu.active = "model"
    assert engine._plan_streaming(7_559) == 0
    engine.gpu.active = None
    assert engine._plan_streaming(7_559) > 0


def test_without_cuda_nothing_is_streamed(monkeypatch):
    monkeypatch.setattr(limits, "free_vram_gb", lambda device: None)
    engine = StubEngine()
    assert engine._plan_streaming(50_000) == 0


def test_generate_puts_the_streaming_back_after_the_run(free_memory):
    engine = StubEngine()
    engine.outcomes = [{"image_path": "x.png"}]
    assert engine.generate("p") == {"image_path": "x.png"}
    assert engine.model.streamed_weight_blocks == 0   # the run set 9, the finished run restored the previous 0


# ---- the engine: out of memory -----------------------------------------------------------------------------------
def test_out_of_memory_becomes_a_clear_error_and_the_engine_goes_on(free_memory):
    engine = StubEngine()
    released = []
    engine.gpu.release = lambda: released.append(True)
    engine.outcomes = [torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.00 GiB"), {"image_path": "y.png"}]
    with pytest.raises(limits.GenerationOutOfMemory) as caught:
        engine.generate("p")
    message = str(caught.value)
    assert "ran out of memory at 2688x1360" in message and "24 thousand tokens" in message and "Traceback" not in message
    assert released and engine.model.streamed_weight_blocks == 0
    assert not engine._lock.locked()
    assert engine.generate("p") == {"image_path": "y.png"}   # the next job runs


def test_other_errors_pass_through_unchanged(free_memory):
    engine = StubEngine()
    engine.outcomes = [ValueError("bad size")]
    with pytest.raises(ValueError, match="bad size"):
        engine.generate("p")


# ---- the scheduler: the slot survives an out-of-memory job ---------------------------------------------------------
def test_scheduler_fails_the_oom_job_and_runs_the_next(free_memory, tmp_path):
    engine = StubEngine()
    engine.outcomes = [torch.cuda.OutOfMemoryError("CUDA out of memory"), {"image_path": str(tmp_path / "z.png")}]
    (tmp_path / "z.png").touch()

    def runner(slot, job, cancel, progress):
        return engine.generate(job.params["prompt"])

    async def scenario():
        store = JobStore(tmp_path / "jobs.sqlite3")
        scheduler = Scheduler(store, [Slot("cpu", ThreadPoolExecutor(1))], runner, max_pending=5, max_per_user=5)
        scheduler.start()
        first = scheduler.submit("u", {"prompt": "big"}, {})
        second = scheduler.submit("u", {"prompt": "small"}, {})
        for _ in range(300):
            if scheduler.get(second.id).finished:
                break
            await asyncio.sleep(0.01)
        jobs = scheduler.get(first.id), scheduler.get(second.id)
        await scheduler.stop()
        return jobs

    first, second = asyncio.run(scenario())
    assert first.status == "failed" and "ran out of memory" in first.error and "GenerationOutOfMemory" in first.error
    assert second.status == "done"
