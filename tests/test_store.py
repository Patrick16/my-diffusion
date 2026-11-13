"""The SQLite job store and the input picture folder (store.py)."""

from __future__ import annotations

import sqlite3

import pytest

from my_diffusion.store import INTERRUPTED, InputFiles, JobStore


@pytest.fixture
def store(tmp_path):
    db = JobStore(tmp_path / "jobs.sqlite3")
    yield db
    db.close()


def test_job_round_trip(store):
    job = store.add("a1", "web:admin", {"prompt": "fox", "loras": [{"name": "x", "weight": 0.5}]}, {"references": ["references_0"]})
    assert job.status == "queued" and not job.cancel_requested
    again = store.get("a1")
    assert again == job
    assert again.params["loras"][0]["weight"] == 0.5
    assert store.get("missing") is None


def test_transitions_happen_only_from_the_right_status(store):
    store.add("a1", "u", {}, {})
    assert store.start("a1", "cuda:0")
    assert not store.start("a1", "cuda:1")          # a second worker cannot take the same job
    assert not store.cancel_queued("a1")            # no longer waiting
    assert store.request_cancel("a1")
    assert store.finish("a1", "cancelled")
    assert not store.finish("a1", "done")           # a finished job stays finished
    job = store.get("a1")
    assert job.status == "cancelled" and job.cancel_requested and job.slot == "cuda:0"


def test_finish_rejects_a_non_final_status(store):
    store.add("a1", "u", {}, {})
    store.start("a1", "cuda")
    with pytest.raises(ValueError):
        store.finish("a1", "queued")


def test_queued_is_oldest_first_and_counts_by_owner(store):
    store.add("b", "u2", {}, {}, created_at=2.0)
    store.add("a", "u1", {}, {}, created_at=1.0)
    store.add("c", "u1", {}, {}, created_at=3.0)
    store.start("c", "cuda")
    assert [job.id for job in store.queued()] == ["a", "b"]
    assert store.count(("queued",)) == 2
    assert store.count(("queued", "running"), "u1") == 2
    assert [job.id for job in store.list("u1")] == ["c", "a"]        # newest first
    assert len(store.list(limit=2)) == 2


def test_jobs_with_the_same_timestamp_keep_their_submit_order(store):
    # Random ids must not decide the order when the clock did not move between two submits.
    ids = [f"{i:x}{'f' * (8 - i)}" for i in range(8)]     # ids in descending string order
    for job_id in ids:
        store.add(job_id, "u", {}, {}, created_at=1.0)
    assert [job.id for job in store.queued()] == ids
    assert [job.id for job in store.list("u")] == ids[::-1]


def test_jobs_survive_reopening_the_database(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    first = JobStore(path)
    first.add("a1", "u", {"prompt": "x"}, {})
    first.close()
    second = JobStore(path)
    assert second.get("a1").params == {"prompt": "x"}
    second.close()


def test_crashed_running_jobs_are_failed_not_retried(store):
    store.add("run", "u", {}, {})
    store.add("wait", "u", {}, {})
    store.start("run", "cuda")
    assert store.fail_interrupted() == ["run"]
    assert store.get("run").status == "failed" and store.get("run").error == INTERRUPTED
    assert store.get("wait").status == "queued"


def test_requeue_puts_a_running_job_back_unless_it_was_cancelled(store):
    store.add("a", "u", {}, {})
    store.add("b", "u", {}, {})
    store.start("a", "cuda")
    store.start("b", "cuda")
    store.request_cancel("b")
    assert store.requeue("a")
    assert not store.requeue("b")
    job = store.get("a")
    assert job.status == "queued" and job.started_at is None and job.slot is None


def test_status_is_checked_by_the_database(store):
    store.add("x", "u", {}, {})
    with pytest.raises(sqlite3.IntegrityError):   # the CHECK constraint, even for code that bypasses JobStore
        store._db.execute("UPDATE jobs SET status = 'weird' WHERE id = 'x'")


def test_unknown_schema_version_is_refused(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    JobStore(path).close()
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="schema version 99"):
        JobStore(path)


def test_input_files_round_trip_and_cleanup(tmp_path):
    files = InputFiles(tmp_path / "inputs")
    names = files.save("abc123", {"init_image": b"one", "references": [b"two", b"three"]})
    assert names == {"init_image": "init_image", "references": ["references_0", "references_1"]}
    assert files.load("abc123", "references_1") == b"three"
    files.remove("abc123")
    assert not (tmp_path / "inputs" / "abc123").exists()
    files.remove("abc123")   # removing twice is harmless


@pytest.mark.parametrize("job_id", ["../x", "a/b", "", "a.b"])
def test_input_files_refuse_ids_that_are_not_plain(tmp_path, job_id):
    with pytest.raises(ValueError):
        InputFiles(tmp_path).save(job_id, {"mask": b"x"})
