"""Where jobs live: a SQLite table, plus a folder for the pictures sent with a job.

Why a database and not memory: the queue survives a restart, finished jobs keep their history, and jobs can be
listed per user. Why SQLite: one file, no server to run, and full ACID transactions. Its one real limit is a
single writer at a time, and a job queue writes a few short rows per job, so that never matters here. Postgres
becomes worth it when several machines (separate worker services) must share the queue: SQLite must not be
used over a network share. Everything outside this module talks to JobStore only, so swapping the backend
later touches this file and nothing else.

Every state change is a conditional UPDATE ("... WHERE status = 'queued'") and reports whether it happened.
Within one process that is belt and braces (the scheduler changes state from a single thread); with several
worker processes it is what keeps two of them from taking the same job.

Statuses: queued -> running -> done | failed | cancelled, and queued -> cancelled. A cancel of a running job
does not change its status at once: it sets cancel_requested, and the job becomes "cancelled" when the engine
reaches its next checkpoint (see cancellation.py)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
INTERRUPTED = "interrupted: the server stopped while the job was running"

_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE jobs (
    id               TEXT PRIMARY KEY,
    owner            TEXT NOT NULL,              -- "<client>:<user>", see auth.py
    created_at       REAL NOT NULL,              -- unix time
    started_at       REAL,
    finished_at      REAL,
    status           TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    params           TEXT NOT NULL,              -- JSON: the whole request except the pictures
    inputs           TEXT NOT NULL DEFAULT '{}', -- JSON: field -> file name(s) in inputs_dir/<id>/
    slot             TEXT,                       -- the device the job ran on
    image_name       TEXT,                       -- PNG in output_dir, once done
    result           TEXT,                       -- JSON: what the engine reported (size, seed, timings ...)
    error            TEXT
);
CREATE INDEX jobs_by_status ON jobs (status, created_at);
CREATE INDEX jobs_by_owner  ON jobs (owner, created_at);
"""


@dataclass(frozen=True)
class JobRecord:
    id: str
    owner: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    status: str
    cancel_requested: bool
    params: dict[str, Any]
    inputs: dict[str, Any]
    slot: str | None
    image_name: str | None
    result: dict[str, Any] | None
    error: str | None

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"], owner=row["owner"], created_at=row["created_at"], started_at=row["started_at"],
        finished_at=row["finished_at"], status=row["status"], cancel_requested=bool(row["cancel_requested"]),
        params=json.loads(row["params"]), inputs=json.loads(row["inputs"]), slot=row["slot"],
        image_name=row["image_name"], result=None if row["result"] is None else json.loads(row["result"]),
        error=row["error"],
    )


class JobStore:
    """Thread-safe: one connection guarded by a lock. Every call is one short transaction (well under a
    millisecond), so it is called directly from the event loop."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: we open transactions ourselves (BEGIN IMMEDIATE takes the write lock up front).
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            # WAL: readers do not block the writer. synchronous=FULL: a committed job survives a power cut
            # (with NORMAL the last commits may be lost, though the file is never corrupted). Writes are rare,
            # so FULL costs nothing noticeable here.
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = FULL")
            self._db.execute("PRAGMA busy_timeout = 5000")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                self._db.executescript(f"BEGIN; {_SCHEMA} PRAGMA user_version = {_SCHEMA_VERSION}; COMMIT;")
            elif version != _SCHEMA_VERSION:
                raise RuntimeError(f"{path}: job database schema version {version}, this code expects {_SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _query(self, sql: str, args: tuple = ()) -> list[JobRecord]:
        with self._lock:
            return [_record(row) for row in self._db.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------ reading
    def get(self, job_id: str) -> JobRecord | None:
        found = self._query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return found[0] if found else None

    def queued(self) -> list[JobRecord]:
        """Waiting jobs, oldest first. Ties in created_at are common (the Windows clock ticks every ~15 ms), so
        they are broken by rowid, the insertion order - not by the id, which is random."""
        return self._query("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid")

    def running(self) -> list[JobRecord]:
        return self._query("SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at, rowid")

    def list(self, owner: str | None = None, limit: int = 50) -> list[JobRecord]:
        """Newest first; owner=None lists everybody's jobs."""
        if owner is None:
            return self._query("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))
        return self._query("SELECT * FROM jobs WHERE owner = ? ORDER BY created_at DESC, rowid DESC LIMIT ?", (owner, limit))

    def count(self, statuses: tuple[str, ...], owner: str | None = None) -> int:
        marks = ", ".join("?" * len(statuses))
        sql = f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})"
        args: tuple = statuses
        if owner is not None:
            sql += " AND owner = ?"
            args += (owner,)
        with self._lock:
            return self._db.execute(sql, args).fetchone()[0]

    # ------------------------------------------------------------------ writing
    def add(self, job_id: str, owner: str, params: dict, inputs: dict, created_at: float | None = None) -> JobRecord:
        with self._write() as db:
            db.execute(
                "INSERT INTO jobs (id, owner, created_at, status, params, inputs) VALUES (?, ?, ?, 'queued', ?, ?)",
                (job_id, owner, time.time() if created_at is None else created_at, json.dumps(params), json.dumps(inputs)),
            )
        record = self.get(job_id)
        assert record is not None
        return record

    def _update(self, sql: str, args: tuple) -> bool:
        with self._write() as db:
            return db.execute(sql, args).rowcount == 1

    def start(self, job_id: str, slot: str) -> bool:
        """queued -> running. False if the job is no longer waiting (cancelled meanwhile, or taken by another worker)."""
        return self._update(
            "UPDATE jobs SET status = 'running', started_at = ?, slot = ? WHERE id = ? AND status = 'queued'",
            (time.time(), slot, job_id),
        )

    def finish(self, job_id: str, status: str, *, image_name: str | None = None, result: dict | None = None,
               error: str | None = None) -> bool:
        """running -> done / failed / cancelled."""
        if status not in TERMINAL:
            raise ValueError(f"not a final status: {status!r}")
        return self._update(
            "UPDATE jobs SET status = ?, finished_at = ?, image_name = ?, result = ?, error = ? WHERE id = ? AND status = 'running'",
            (status, time.time(), image_name, None if result is None else json.dumps(result), error, job_id),
        )

    def cancel_queued(self, job_id: str) -> bool:
        """queued -> cancelled, at once: the job never reaches a GPU."""
        return self._update(
            "UPDATE jobs SET status = 'cancelled', finished_at = ?, cancel_requested = 1 WHERE id = ? AND status = 'queued'",
            (time.time(), job_id),
        )

    def request_cancel(self, job_id: str) -> bool:
        """Marks a running job as asked to stop. The status changes when the job actually stops."""
        return self._update("UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status = 'running'", (job_id,))

    def requeue(self, job_id: str) -> bool:
        """running -> queued: the server is shutting down cleanly, so the job is run again after the restart."""
        return self._update(
            "UPDATE jobs SET status = 'queued', started_at = NULL, slot = NULL WHERE id = ? AND status = 'running' AND cancel_requested = 0",
            (job_id,),
        )

    def fail_interrupted(self) -> list[str]:
        """Called at startup. A job still "running" means the server died in the middle of it (a crash, a kill,
        out of memory). It is marked failed, not run again: if that job is what killed the server, running it
        again would kill it again, in a loop. Returns the ids."""
        with self._write() as db:
            ids = [row["id"] for row in db.execute("SELECT id FROM jobs WHERE status = 'running'").fetchall()]
            db.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, error = ? WHERE status = 'running'",
                (time.time(), INTERRUPTED),
            )
        return ids


class InputFiles:
    """The pictures of a job, as the client sent them (already checked to be valid images), in inputs_dir/<job id>/.

    Kept as compressed bytes rather than decoded pictures: a waiting job then costs disk, not RAM, and the job
    can still run after a restart. Job ids come from uuid4 (hex only), never from the client, so they are safe
    as folder names."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _folder(self, job_id: str) -> Path:
        if not job_id.isalnum():
            raise ValueError(f"bad job id: {job_id!r}")
        return self.root / job_id

    def save(self, job_id: str, pictures: dict[str, bytes | list[bytes]]) -> dict[str, str | list[str]]:
        """{"init_image": b"...", "references": [b"...", b"..."]} -> the same keys with file names."""
        folder = self._folder(job_id)
        folder.mkdir(parents=True, exist_ok=False)
        names: dict[str, str | list[str]] = {}
        for field, data in pictures.items():
            if isinstance(data, list):
                names[field] = [self._write(folder, f"{field}_{i}", item) for i, item in enumerate(data)]
            else:
                names[field] = self._write(folder, field, data)
        return names

    @staticmethod
    def _write(folder: Path, name: str, data: bytes) -> str:
        (folder / name).write_bytes(data)
        return name

    def load(self, job_id: str, name: str) -> bytes:
        return (self._folder(job_id) / Path(name).name).read_bytes()

    def remove(self, job_id: str) -> None:
        shutil.rmtree(self._folder(job_id), ignore_errors=True)
