"""GET /result/{id}/events: the job's changes pushed as server-sent events. Uses the stub engine of test_api_jobs
(no GPU): a "slow" prompt runs 500 cancellable steps, anything else finishes at once."""

from __future__ import annotations

import json
import threading
import time

from test_api_jobs import WEB, bot, client, secured  # noqa: F401 - fixtures


def read_events(response) -> list[tuple[str, dict]]:
    """Reads a whole event stream (it ends by itself after the final event) into [(event name, job)]."""
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[len("data: "):])))
    return events


def test_stream_of_a_quick_job_ends_with_done(client):
    job = client.post("/generate", json={"prompt": "a cat"}).json()
    with client.stream("GET", f"/result/{job['job_id']}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = read_events(response)
    names = [name for name, _ in events]
    assert names[-1] == "done" and set(names[:-1]) <= {"update"}
    final = events[-1][1]
    assert final["status"] == "done" and final["image_url"] == f"/result/{job['job_id']}/image"
    assert final == client.get(f"/result/{job['job_id']}").json()   # the same JSON as the polling endpoint


def test_stream_of_a_finished_job_is_one_event(client):
    job = client.post("/generate", json={"prompt": "a cat"}).json()
    client.get(f"/result/{job['job_id']}/events").read()   # wait until it is done
    with client.stream("GET", f"/result/{job['job_id']}/events") as response:
        events = read_events(response)
    assert [name for name, _ in events] == ["done"]


def test_stream_shows_progress_and_a_cancel(client):
    # TestClient hands the body over only when the response is complete, so the cancel comes from another thread
    # (as it would from another client) while the stream is open.
    job = client.post("/generate", json={"prompt": "slow"}).json()

    def cancel_after_some_steps():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (client.get(f"/result/{job['job_id']}").json().get("step") or 0) >= 3:
                break
            time.sleep(0.01)
        client.post(f"/cancel/{job['job_id']}")

    canceller = threading.Thread(target=cancel_after_some_steps)
    canceller.start()
    with client.stream("GET", f"/result/{job['job_id']}/events") as response:
        events = read_events(response)
    canceller.join()
    names = [name for name, _ in events]
    assert names[-1] == "cancelled" and events[-1][1]["status"] == "cancelled"
    steps = [body["step"] for name, body in events if name == "update" and body["step"]]
    assert steps and steps == sorted(steps)   # progress was pushed, in order
    assert any(body["cancel_requested"] for _, body in events)


def test_failed_job_ends_the_stream_with_failed(client):
    job = client.post("/generate", json={"prompt": "fail"}).json()
    with client.stream("GET", f"/result/{job['job_id']}/events") as response:
        events = read_events(response)
    name, body = events[-1]
    assert name == "failed" and "does not fit" in body["error"]


def test_stream_of_someone_elses_job_is_404(secured):
    job = secured.post("/generate", json={"prompt": "a cat"}, headers=bot("1")).json()
    assert secured.get(f"/result/{job['job_id']}/events", headers=bot("2")).status_code == 404
    assert secured.get(f"/result/{job['job_id']}/events", headers=bot("1")).status_code == 200
    assert secured.get(f"/result/{job['job_id']}/events", headers=WEB).status_code == 200   # an admin sees all
    assert secured.get(f"/result/{job['job_id']}/events").status_code == 401
    assert secured.get("/result/nope/events", headers=WEB).status_code == 404
