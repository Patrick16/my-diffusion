"""The scheduler (jobs.py) over a real SQLite store. No GPU: the runner is plain Python that behaves like the
engine with respect to checkpoints and progress. What a job does is chosen by params["kind"]:
- "quick": finishes at once;
- "loop": 10 000 steps with a checkpoint each (a long job that can be cancelled);
- "gate": waits until the test opens the gate named params["gate"];
- "fail": raises ValueError."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from my_diffusion.cancellation import GenerationCancelled, checkpoint
from my_diffusion.jobs import JobNotCancellable, JobNotFound, QueueFull, Scheduler, Slot, fair_order
from my_diffusion.store import INTERRUPTED, JobRecord, JobStore


class Harness:
    """Slots, a store, a runner that records what ran where, and gates the test can open."""

    def __init__(self, tmp_path, devices=("cuda",), max_pending=10, max_per_user=10) -> None:
        self.store = JobStore(tmp_path / "jobs.sqlite3")
        self.pools = [ThreadPoolExecutor(max_workers=1) for _ in devices]
        self.slots = [Slot(d, pool) for d, pool in zip(devices, self.pools)]
        self.gates: dict[str, threading.Event] = {}
        self.ran: list[tuple[str, str]] = []          # (job id, slot)
        self.finished: list[str] = []                 # on_finished calls
        self.started = threading.Event()
        self.max_pending, self.max_per_user = max_pending, max_per_user

    def gate(self, name: str) -> threading.Event:
        return self.gates.setdefault(name, threading.Event())

    def runner(self, slot: Slot, job: JobRecord, cancel_event: threading.Event, progress) -> dict:
        self.ran.append((job.id, slot.name))
        kind = job.params["kind"]
        if kind == "fail":
            raise ValueError("bad size")
        if kind == "gate":
            self.gate(job.params["gate"]).wait(5)
        if kind == "loop":
            self.started.set()
            for i in range(10_000):
                checkpoint(cancel_event)
                progress(i + 1, 10_000)
                time.sleep(0.001)
        return {"image_path": f"C:/server/outputs/{job.id}.png", "width": 16, "seed": 1}

    def scheduler(self) -> Scheduler:
        return Scheduler(self.store, self.slots, self.runner, self.max_pending, self.max_per_user,
                         on_finished=self.finished.append)

    def close(self) -> None:
        for gate in self.gates.values():
            gate.set()
        for pool in self.pools:
            pool.shutdown(wait=True)
        self.store.close()


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.005)


def submit(s: Scheduler, owner: str, kind: str, **params) -> JobRecord:
    return s.submit(owner, {"kind": kind, **params}, {})


def status(s: Scheduler, job: JobRecord) -> str:
    return s.get(job.id).status


# ---- the fair order (pure) ------------------------------------------------------------------------------------

def _job(job_id: str, owner: str, created: float) -> JobRecord:
    return JobRecord(job_id, owner, created, None, None, "queued", False, {}, {}, None, None, None, None)


def test_one_owner_is_plain_fifo():
    jobs = [_job("a1", "A", 1), _job("a2", "A", 2), _job("a3", "A", 3)]
    assert [j.id for j in fair_order(jobs, {})] == ["a1", "a2", "a3"]


def test_owners_take_turns():
    jobs = [_job("a1", "A", 1), _job("a2", "A", 2), _job("a3", "A", 3), _job("b1", "B", 4), _job("c1", "C", 5)]
    assert [j.id for j in fair_order(jobs, {})] == ["a1", "b1", "c1", "a2", "a3"]


def test_the_owner_served_longest_ago_goes_first():
    jobs = [_job("a1", "A", 1), _job("b1", "B", 2)]
    assert [j.id for j in fair_order(jobs, {"A": 10.0, "B": 5.0})] == ["b1", "a1"]


# ---- running jobs ---------------------------------------------------------------------------------------------

def test_jobs_run_and_the_result_is_stored_without_the_server_path(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        jobs = [submit(s, "web:admin", "quick") for _ in range(3)]
        await wait_until(lambda: all(status(s, j) == "done" for j in jobs))
        await s.stop()
        return jobs

    jobs = asyncio.run(scenario())
    assert [job_id for job_id, _ in h.ran] == [j.id for j in jobs]
    done = h.store.get(jobs[0].id)
    assert done.image_name == f"{jobs[0].id}.png"
    assert "image_path" not in done.result and done.result["width"] == 16
    assert h.finished == [j.id for j in jobs]       # inputs are cleaned up after every job


def test_a_failed_job_keeps_its_reason_and_the_queue_goes_on(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        bad = submit(s, "u", "fail")
        good = submit(s, "u", "quick")
        await wait_until(lambda: status(s, good) == "done")
        await s.stop()
        return bad

    bad = asyncio.run(scenario())
    job = h.store.get(bad.id)
    assert job.status == "failed" and job.error == "ValueError: bad size"


def test_nothing_starts_while_paused(h):
    async def scenario():
        s = h.scheduler()
        s.start(paused=True)
        job = submit(s, "u", "quick")
        await asyncio.sleep(0.05)
        assert status(s, job) == "queued" and s.position(job.id) == 1
        s.resume()
        await wait_until(lambda: status(s, job) == "done")
        await s.stop()

    asyncio.run(scenario())


def test_users_take_turns_in_practice(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        blocker = submit(s, "X", "gate", gate="g")
        await wait_until(lambda: status(s, blocker) == "running")
        a = [submit(s, "A", "quick") for _ in range(3)]
        b = submit(s, "B", "quick")
        assert [s.position(j.id) for j in [*a, b]] == [1, 3, 4, 2]
        h.gate("g").set()
        await wait_until(lambda: all(status(s, j) == "done" for j in [*a, b]))
        await s.stop()
        return a, b

    a, b = asyncio.run(scenario())
    assert [job_id for job_id, _ in h.ran[1:]] == [a[0].id, b.id, a[1].id, a[2].id]


def test_two_gpus_run_two_jobs_at_once(tmp_path):
    h2 = Harness(tmp_path, devices=("cuda:0", "cuda:1"))
    try:
        async def scenario():
            s = h2.scheduler()
            s.start()
            one = submit(s, "u", "gate", gate="g")
            two = submit(s, "u", "gate", gate="g")
            await wait_until(lambda: status(s, one) == "running" and status(s, two) == "running")
            assert sorted(s.busy_slots()) == ["cuda:0", "cuda:1"]
            h2.gate("g").set()
            await wait_until(lambda: status(s, one) == "done" and status(s, two) == "done")
            await s.stop()

        asyncio.run(scenario())
        assert sorted(slot for _, slot in h2.ran) == ["cuda:0", "cuda:1"]
    finally:
        h2.close()


# ---- limits ---------------------------------------------------------------------------------------------------

def test_the_queue_and_each_user_have_limits(tmp_path):
    h2 = Harness(tmp_path, max_pending=3, max_per_user=2)
    try:
        async def scenario():
            s = h2.scheduler()
            s.start(paused=True)                    # nothing leaves the queue
            submit(s, "A", "quick")
            submit(s, "A", "quick")
            with pytest.raises(QueueFull, match="you already have 2"):
                submit(s, "A", "quick")
            s.submit("web:admin", {"kind": "quick"}, {}, exempt=True)   # admins skip the per-user limit
            with pytest.raises(QueueFull, match="queue is full"):
                s.submit("web:admin", {"kind": "quick"}, {}, exempt=True)
            await s.stop()

        asyncio.run(scenario())
    finally:
        h2.close()


# ---- cancelling -----------------------------------------------------------------------------------------------

def test_a_cancelled_waiting_job_never_runs(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        blocker = submit(s, "u", "gate", gate="g")
        await wait_until(lambda: status(s, blocker) == "running")
        victim = submit(s, "u", "quick")
        cancelled = s.cancel(victim.id)
        assert cancelled.status == "cancelled" and cancelled.cancel_requested
        h.gate("g").set()
        await wait_until(lambda: status(s, blocker) == "done")
        await s.stop()
        return victim

    victim = asyncio.run(scenario())
    assert victim.id not in [job_id for job_id, _ in h.ran]
    assert victim.id in h.finished                  # its pictures are deleted too


def test_a_running_job_stops_at_its_next_checkpoint(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        job = submit(s, "u", "loop")
        await wait_until(lambda: (s.progress(job.id) or (0, 0))[0] >= 3)
        asked = s.cancel(job.id)
        assert asked.status == "running" and asked.cancel_requested   # stops a moment later
        await wait_until(lambda: status(s, job) == "cancelled")
        follow = submit(s, "u", "quick")
        await wait_until(lambda: status(s, follow) == "done")
        await s.stop()

    asyncio.run(scenario())


def test_finished_and_unknown_jobs_cannot_be_cancelled(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        job = submit(s, "u", "quick")
        await wait_until(lambda: status(s, job) == "done")
        with pytest.raises(JobNotCancellable):
            s.cancel(job.id)
        with pytest.raises(JobNotFound):
            s.cancel("nope")
        await s.stop()

    asyncio.run(scenario())


# ---- restarts -------------------------------------------------------------------------------------------------

def test_a_clean_stop_keeps_the_queue_for_the_next_start(h):
    async def first_run():
        s = h.scheduler()
        s.start()
        running = submit(s, "u", "loop")
        waiting = submit(s, "u", "quick")
        await wait_until(h.started.is_set)
        await s.stop()
        return running, waiting

    running, waiting = asyncio.run(first_run())
    assert h.store.get(running.id).status == "queued"   # interrupted by the shutdown: will run again
    assert h.store.get(waiting.id).status == "queued"

    async def second_run():
        s = h.scheduler()
        s.start()
        await wait_until(lambda: status(s, running) == "running")   # the older job starts again first
        s.cancel(running.id)                                        # (cut short, to keep the test fast)
        await wait_until(lambda: status(s, running) == "cancelled")
        await wait_until(lambda: status(s, waiting) == "done")
        await s.stop()

    asyncio.run(second_run())


def test_a_job_the_user_cancelled_does_not_come_back_after_a_stop(h):
    async def scenario():
        s = h.scheduler()
        s.start()
        job = submit(s, "u", "gate", gate="never")       # ignores the cancel flag, like a VAE decode
        await wait_until(lambda: status(s, job) == "running")
        s.cancel(job.id)
        await s.stop()
        return job

    job = asyncio.run(scenario())
    assert h.store.get(job.id).status == "cancelled"


def test_jobs_found_running_at_start_are_failed(h):
    h.store.add("crashed", "u", {"kind": "quick"}, {})
    h.store.start("crashed", "cuda")                      # as if the server had died in the middle

    async def scenario():
        s = h.scheduler()
        s.start()
        await asyncio.sleep(0.05)
        await s.stop()

    asyncio.run(scenario())
    assert h.store.get("crashed").status == "failed" and h.store.get("crashed").error == INTERRUPTED
    assert "crashed" not in [job_id for job_id, _ in h.ran]
    assert h.finished == ["crashed"]


def test_a_dead_dispatcher_is_reported(h, monkeypatch, caplog):
    async def scenario():
        s = h.scheduler()

        async def crash():
            raise RuntimeError("bug in the scheduler")

        monkeypatch.setattr(s, "_dispatch", crash)
        s.start()
        await wait_until(lambda: not s.alive)
        await s.stop()

    with caplog.at_level("ERROR", logger="my_diffusion.jobs"):
        asyncio.run(scenario())
    assert "scheduler stopped" in caplog.text


def test_a_scheduler_needs_a_slot(h):
    with pytest.raises(ValueError):
        Scheduler(h.store, [], h.runner, 1, 1)


def test_checkpoint_raises_only_when_asked():
    checkpoint(None)
    event = threading.Event()
    checkpoint(event)
    event.set()
    with pytest.raises(GenerationCancelled):
        checkpoint(event)
