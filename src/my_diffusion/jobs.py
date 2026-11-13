"""The scheduler: takes waiting jobs from the store and runs them on the free GPU slots.

Slots. A slot is one GPU with its own engine and its own worker thread. A GPU runs one generation at a time:
one generation fills it (up to ~14 GB of VRAM at 1080p), so two jobs on one card would each run at half speed
and risk running out of memory, while gaining nothing. Parallelism therefore comes from more GPUs, one slot per
device (settings.devices). On a single-GPU machine there is one slot and jobs simply run one after another.
The worker thread of a slot is persistent and bound to its device: CUDA's "current device" is per thread, and
the cuDNN attention kernel keeps its setup per thread (see transformer.py), so a fixed thread stays warm.

Fair order. Jobs are not run strictly first-in-first-out across users: that would let one Telegram user who
sends ten requests make everyone else wait for all ten. Instead the queue goes round the users: each turn takes
the oldest job of the user who was served longest ago. Within one user, jobs keep their order. With one user it
is plain FIFO. Admins are users like any other here; they are only exempt from the per-user limit.

Cancelling. A waiting job is cancelled at once in the store and never touches a GPU. A running one gets its
cancel_event set; the engine stops at its next checkpoint (cancellation.py) and the job ends as "cancelled".
A cancel that arrives after the last checkpoint changes nothing: the picture is finished and the job ends "done".

Restarts. On a clean shutdown, running jobs are asked to stop and put back in the queue: they run again after
the restart, as do the waiting ones. After a crash, jobs found "running" are marked failed (see
JobStore.fail_interrupted) instead of being retried, so a job that kills the server cannot do it in a loop.

Threads. All scheduling happens on the event loop thread. What crosses into the worker threads is a
threading.Event (cancel) and a progress callback that only writes a tuple."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path

from .cancellation import GenerationCancelled
from .store import JobRecord, JobStore

logger = logging.getLogger(__name__)

# Failures that are the request's fault (bad size, unknown LoRA ...): logged in one line, without a stack trace.
# Anything else is a server problem and is logged with its traceback.
_REQUEST_ERRORS = (ValueError, FileNotFoundError, NotImplementedError)


class QueueFull(Exception):
    """The queue (or this user's share of it) is at its limit. The API answers 429."""


class JobNotFound(KeyError):
    pass


class JobNotCancellable(Exception):
    """The job has already finished."""


@dataclass
class Slot:
    name: str            # the device, e.g. "cuda:0"; also the key of its engine
    executor: Executor   # one persistent thread bound to the device


# runner(slot, job, cancel_event, progress) -> the engine's result dict. Runs in the slot's thread.
Runner = Callable[[Slot, JobRecord, threading.Event, Callable[[int, int], None]], dict]


@dataclass
class _Running:
    job: JobRecord
    slot: Slot
    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress: tuple[int, int] | None = None
    task: asyncio.Task | None = None

    def report(self, done: int, total: int) -> None:   # called from the slot's thread
        self.progress = (done, total)


def fair_order(queued: list[JobRecord], last_served: dict[str, float]) -> list[JobRecord]:
    """The order in which the waiting jobs will start: round-robin over owners.

    Owners are visited starting with the one served longest ago (never served = first; ties go to the owner
    whose oldest job is older), one job per owner per round, each owner's jobs in their own submit order."""
    per_owner: dict[str, deque[JobRecord]] = defaultdict(deque)
    for job in queued:   # queued is oldest first, so every deque is too
        per_owner[job.owner].append(job)
    owners = sorted(per_owner, key=lambda o: (last_served.get(o, 0.0), per_owner[o][0].created_at))
    order: list[JobRecord] = []
    while owners:
        for owner in list(owners):
            order.append(per_owner[owner].popleft())
            if not per_owner[owner]:
                owners.remove(owner)
    return order


class Scheduler:
    def __init__(
        self,
        store: JobStore,
        slots: list[Slot],
        runner: Runner,
        max_pending: int,
        max_per_user: int,
        on_finished: Callable[[str], None] | None = None,
    ) -> None:
        if not slots:
            raise ValueError("the scheduler needs at least one slot")
        self.store = store
        self.slots = slots
        self._runner = runner
        self._max_pending = max_pending
        self._max_per_user = max_per_user
        self._on_finished = on_finished          # e.g. delete the job's input pictures
        self._running: dict[str, _Running] = {}  # job id -> its run
        self._busy: dict[str, str] = {}          # slot name -> job id
        self._last_served: dict[str, float] = {}
        self._paused = True
        self._wake: asyncio.Event | None = None
        self._dispatcher: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self, paused: bool = False) -> None:
        """Starts dispatching. Call from inside the running event loop. With paused=True jobs are accepted but
        none starts until resume() (the API pauses until the models are loaded)."""
        for job_id in self.store.fail_interrupted():
            logger.warning("job %s was running when the server stopped unexpectedly; marked failed", job_id)
            self._finished(job_id)
        self._paused = paused
        self._wake = asyncio.Event()
        self._dispatcher = asyncio.create_task(self._dispatch(), name="job-scheduler")
        self._dispatcher.add_done_callback(self._on_dispatcher_exit)

    def resume(self) -> None:
        self._paused = False
        self._poke()

    @property
    def paused(self) -> bool:
        return self._paused

    async def stop(self) -> None:
        """Clean shutdown: running jobs are asked to stop and go back to the queue (a job the user already
        cancelled ends as cancelled instead). Waiting jobs stay queued for the next start."""
        self._paused = True
        for run in list(self._running.values()):
            run.cancel_event.set()
            if not self.store.requeue(run.job.id):
                self.store.finish(run.job.id, "cancelled")
                self._finished(run.job.id)
            if run.task is not None:
                run.task.cancel()
        tasks = [run.task for run in self._running.values() if run.task is not None]
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            tasks.append(self._dispatcher)
        for task in tasks:
            # A dispatcher that died from a bug re-raises it here; it was already logged.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._running.clear()
        self._busy.clear()
        self._dispatcher = None

    def _on_dispatcher_exit(self, task: asyncio.Task) -> None:
        # The dispatcher never ends on its own. If a bug killed it, every job would wait forever with nothing
        # in the log, so say it loudly; /health shows it too and new jobs are refused.
        if not task.cancelled() and task.exception() is not None:
            logger.error("the job scheduler stopped; no further jobs will start", exc_info=task.exception())

    @property
    def alive(self) -> bool:
        return self._dispatcher is not None and not self._dispatcher.done()

    # ------------------------------------------------------------------ the API side
    def submit(self, owner: str, params: dict, inputs: dict, *, exempt: bool = False, job_id: str | None = None) -> JobRecord:
        """Adds a job. exempt=True skips the per-user limit (admins). Raises QueueFull."""
        if self.store.count(("queued",)) >= self._max_pending:
            raise QueueFull(f"the queue is full ({self._max_pending} jobs waiting); try again later")
        if not exempt and self.store.count(("queued", "running"), owner) >= self._max_per_user:
            raise QueueFull(f"you already have {self._max_per_user} jobs waiting or running; wait for one to finish or cancel one")
        job = self.store.add(job_id or uuid.uuid4().hex, owner, params, inputs)
        self._poke()
        return job

    def get(self, job_id: str) -> JobRecord:
        job = self.store.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job.finished:
            raise JobNotCancellable(job_id)
        if job.status == "queued" and self.store.cancel_queued(job_id):
            self._finished(job_id)
        elif job_id in self._running:
            self.store.request_cancel(job_id)
            self._running[job_id].cancel_event.set()
        return self.get(job_id)

    def progress(self, job_id: str) -> tuple[int, int] | None:
        run = self._running.get(job_id)
        return None if run is None else run.progress

    def order(self) -> list[JobRecord]:
        return fair_order(self.store.queued(), self._last_served)

    def position(self, job_id: str) -> int | None:
        """1 = the next job to start (on the first slot that frees up). None if the job is not waiting."""
        return next((i for i, job in enumerate(self.order(), start=1) if job.id == job_id), None)

    @property
    def is_idle(self) -> bool:
        return not self._running and self.store.count(("queued",)) == 0

    def busy_slots(self) -> dict[str, str]:
        return dict(self._busy)

    # ------------------------------------------------------------------ dispatching
    def _poke(self) -> None:
        if self._wake is not None:
            self._wake.set()

    async def _dispatch(self) -> None:
        assert self._wake is not None
        while True:
            self._wake.clear()
            if not self._paused:
                self._fill_free_slots()
            await self._wake.wait()

    def _fill_free_slots(self) -> None:
        free = [slot for slot in self.slots if slot.name not in self._busy]
        if not free:
            return
        waiting = iter(self.order())
        for slot in free:
            for job in waiting:
                if self.store.start(job.id, slot.name):
                    self._launch(slot, self.get(job.id))
                    break

    def _launch(self, slot: Slot, job: JobRecord) -> None:
        run = _Running(job=job, slot=slot)
        self._running[job.id] = run
        self._busy[slot.name] = job.id
        self._last_served[job.owner] = time.monotonic()
        run.task = asyncio.create_task(self._run(run), name=f"job-{job.id}")

    async def _run(self, run: _Running) -> None:
        job, slot = run.job, run.slot
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(slot.executor, self._runner, slot, job, run.cancel_event, run.report)
        except GenerationCancelled:
            self.store.finish(job.id, "cancelled")
        except Exception as exc:  # noqa: BLE001 - any failure is the job's outcome; the scheduler must go on
            if isinstance(exc, _REQUEST_ERRORS):
                logger.warning("job %s failed: %s: %s", job.id, type(exc).__name__, exc)
            else:
                logger.error("job %s failed on %s", job.id, slot.name, exc_info=exc)
            # Only the text is kept. The exception itself is not stored: its traceback holds the frames of the
            # failed run, and with them its GPU tensors.
            self.store.finish(job.id, "failed", error=f"{type(exc).__name__}: {exc}")
        else:
            image = Path(result["image_path"]).name
            # The server's own path is not anyone's business: clients get the picture through the API.
            public = {key: value for key, value in result.items() if key != "image_path"}
            self.store.finish(job.id, "done", image_name=image, result=public)
        finally:
            # On a clean shutdown stop() has already put the job back and cleared the tables.
            if self._running.pop(job.id, None) is not None:
                self._busy.pop(slot.name, None)
                if self.get(job.id).finished:
                    self._finished(job.id)
                self._poke()

    def _finished(self, job_id: str) -> None:
        if self._on_finished is not None:
            try:
                self._on_finished(job_id)
            except Exception:  # noqa: BLE001 - cleanup must never break the scheduler
                logger.exception("cleanup after job %s failed", job_id)
