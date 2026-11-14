"""HTTP tests of the job flow: POST /generate -> GET /result/{id} -> GET /result/{id}/image, POST /cancel/{id},
GET /jobs, and who may see what. The engine is a stub that behaves like the real one with respect to cancellation,
progress and the output file, so no GPU or models are needed. The database, the input pictures and the outputs
go to a temporary folder.

The prompt picks the stub's behaviour: "slow" (500 steps with checkpoints, cancellable), "fail" (ValueError,
as a size that does not fit), anything else finishes at once."""

from __future__ import annotations

import base64
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from my_diffusion import api
from my_diffusion.cancellation import checkpoint
from my_diffusion.engine import Krea2Engine

WEB_KEY = "w" * 32
BOT_KEY = "b" * 32
WEB = {"Authorization": f"Bearer {WEB_KEY}"}


def bot(user: str) -> dict:
    return {"Authorization": f"Bearer {BOT_KEY}", "X-User-Id": user}


def png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 200, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


calls: list[dict] = []


def fake_generate(**kwargs) -> dict:
    calls.append(kwargs)
    prompt = kwargs["prompt"]
    if prompt == "fail":
        raise ValueError("this size does not fit the GPU")
    if prompt == "slow":
        for i in range(500):
            checkpoint(kwargs["cancel_event"])
            kwargs["on_progress"](i + 1, 500)
            time.sleep(0.01)
    path = api.settings.output_dir / kwargs["output_name"]
    Image.new("RGB", (16, 16), (200, 0, 0)).save(path)
    return {
        "image_path": str(path), "width": 16, "height": 16, "seed": 1, "prompt": prompt,
        "enhanced_prompt": None, "steps": 1, "guidance_scale": 1.0, "guidance_interval": [0.0, 1.0],
        "shift": 1.0, "checkpoint": "fake.safetensors", "matmul": "exact", "loras": [], "timings": {},
        "references": len(kwargs["references"]),
    }


def make_client(monkeypatch, tmp_path, keys: bool):
    monkeypatch.setattr(api.settings, "database_path", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(api.settings, "inputs_dir", tmp_path / "inputs")
    monkeypatch.setattr(api.settings, "output_dir", tmp_path / "outputs")
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(api.settings, "api_keys", {"web": WEB_KEY, "telegram": BOT_KEY} if keys else {})
    monkeypatch.setattr(api.settings, "admin_clients", ["web"] if keys else [])
    monkeypatch.setattr(api.settings, "queue_max_per_user", 2)
    monkeypatch.setattr(api.engine, "load", lambda: None)
    monkeypatch.setattr(Krea2Engine, "loaded", property(lambda self: True))
    monkeypatch.setattr(api.engine, "generate", fake_generate)
    monkeypatch.setattr(api, "_load_error", None)
    calls.clear()
    return TestClient(api.app)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Access control off: every request is the local admin."""
    with make_client(monkeypatch, tmp_path, keys=False) as test_client:
        yield test_client


@pytest.fixture
def secured(monkeypatch, tmp_path):
    """Access control on: "web" is an admin client, "telegram" a service acting for X-User-Id."""
    with make_client(monkeypatch, tmp_path, keys=True) as test_client:
        yield test_client


def wait_for(client: TestClient, job_id: str, predicate, headers: dict | None = None, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/result/{job_id}", headers=headers).json()
        if predicate(body):
            return body
        if time.monotonic() > deadline:
            raise AssertionError(f"job did not reach the expected state, last seen: {body}")
        time.sleep(0.01)


# ---- the flow -------------------------------------------------------------------------------------------------

def test_generate_returns_the_job_at_once_and_the_picture_comes_later(client):
    response = client.post("/generate", json={"prompt": "a cat", "seed": 7})
    assert response.status_code == 202
    job = response.json()
    assert job["status"] in ("queued", "running") and job["owner"] == "local:admin"
    assert job["params"]["prompt"] == "a cat" and job["params"]["seed"] == 7

    done = wait_for(client, job["job_id"], lambda b: b["status"] == "done")
    assert done["result"]["width"] == 16
    assert "image_path" not in done["result"]            # the server's path is not exposed
    assert done["image_url"] == f"/result/{job['job_id']}/image"

    image = client.get(done["image_url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(image.content)).size == (16, 16)


def test_picture_is_not_there_before_the_job_is_done(client):
    job_id = client.post("/generate", json={"prompt": "slow"}).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "running")
    assert client.get(f"/result/{job_id}/image").status_code == 409
    client.post(f"/cancel/{job_id}")


def test_input_pictures_reach_the_engine_and_stay_after_the_job(client, tmp_path):
    job_id = client.post("/generate", json={"prompt": "edit", "references": [png_b64()]}).json()["job_id"]
    assert (tmp_path / "inputs" / job_id).is_dir()
    wait_for(client, job_id, lambda b: b["status"] == "done")
    assert calls[-1]["references"][0].size == (32, 24)
    assert "references" not in client.get(f"/result/{job_id}").json()["params"]   # pictures are not echoed back
    assert (tmp_path / "inputs" / job_id).is_dir()   # kept for clients, see test_api_inputs


def test_bad_request_is_400_before_queueing(client):
    assert client.post("/generate", json={"prompt": "x", "mask": png_b64()}).status_code == 400
    assert client.post("/generate", json={"prompt": "x", "references": ["not base64!"]}).status_code == 400
    assert client.get("/jobs").json() == []


def test_engine_side_errors_make_the_job_failed(client):
    job_id = client.post("/generate", json={"prompt": "fail"}).json()["job_id"]
    body = wait_for(client, job_id, lambda b: b["status"] == "failed")
    assert "does not fit" in body["error"]


def test_running_job_can_be_cancelled(client):
    job_id = client.post("/generate", json={"prompt": "slow"}).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "running" and (b["step"] or 0) >= 3)
    asked = client.post(f"/cancel/{job_id}")
    assert asked.status_code == 200 and asked.json()["cancel_requested"]
    body = wait_for(client, job_id, lambda b: b["status"] == "cancelled")
    assert body["step"] is None and body["image_url"] is None


def test_waiting_job_can_be_cancelled_and_leaves_the_line(client):
    slow_id = client.post("/generate", json={"prompt": "slow"}).json()["job_id"]
    wait_for(client, slow_id, lambda b: b["status"] == "running")
    waiting = client.post("/generate", json={"prompt": "a bird"}).json()
    assert waiting["status"] == "queued" and waiting["position"] == 1
    cancelled = client.post(f"/cancel/{waiting['job_id']}").json()
    assert cancelled["status"] == "cancelled" and cancelled["position"] is None
    client.post(f"/cancel/{slow_id}")
    wait_for(client, slow_id, lambda b: b["status"] == "cancelled")


def test_finished_job_cannot_be_cancelled_and_unknown_is_404(client):
    job_id = client.post("/generate", json={"prompt": "a fox"}).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "done")
    assert client.post(f"/cancel/{job_id}").status_code == 409
    assert client.get("/result/nope").status_code == 404
    assert client.post("/cancel/nope").status_code == 404


def test_health_shows_the_slots_and_the_queue(client):
    deadline = time.monotonic() + 5
    while True:   # the queue starts once the load at start has finished on the slot's thread: a moment after startup
        body = client.get("/health").json()
        if not body["queue"]["paused"] or time.monotonic() > deadline:
            break
        time.sleep(0.01)
    assert body["access_control"] is False
    assert body["queue"]["slots"] == [{"device": api.DEVICES[0], "job": None}]
    assert body["queue"]["scheduler_alive"] is True and body["queue"]["paused"] is False


def test_checkpoint_switch_is_refused_while_jobs_are_pending(client):
    slow_id = client.post("/generate", json={"prompt": "slow"}).json()["job_id"]
    wait_for(client, slow_id, lambda b: b["status"] == "running")
    assert client.post("/checkpoints/load", json={"name": "other.safetensors"}).status_code == 409
    client.post(f"/cancel/{slow_id}")


# ---- access ---------------------------------------------------------------------------------------------------

def test_a_key_is_required_when_keys_are_configured(secured):
    assert secured.post("/generate", json={"prompt": "x"}).status_code == 401
    assert secured.post("/generate", json={"prompt": "x"}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert secured.get("/jobs", headers={"Authorization": f"Bearer {BOT_KEY}"}).status_code == 400   # no X-User-Id
    assert secured.get("/health").status_code == 200                                                 # health stays open


def test_users_see_only_their_own_jobs(secured):
    anna = secured.post("/generate", json={"prompt": "anna's"}, headers=bot("111")).json()
    assert anna["owner"] == "telegram:111"
    wait_for(secured, anna["job_id"], lambda b: b["status"] == "done", headers=bot("111"))
    job_id = anna["job_id"]

    # Another Telegram user: the job does not exist for them - not its status, not its picture, not a cancel.
    assert secured.get(f"/result/{job_id}", headers=bot("222")).status_code == 404
    assert secured.get(f"/result/{job_id}/image", headers=bot("222")).status_code == 404
    assert secured.post(f"/cancel/{job_id}", headers=bot("222")).status_code == 404
    assert secured.get("/jobs", headers=bot("222")).json() == []

    # The owner and the admin client see it.
    assert secured.get(f"/result/{job_id}/image", headers=bot("111")).status_code == 200
    assert [j["job_id"] for j in secured.get("/jobs", headers=bot("111")).json()] == [job_id]
    assert secured.get(f"/result/{job_id}", headers=WEB).status_code == 200
    assert [j["job_id"] for j in secured.get("/jobs?owner=telegram:111", headers=WEB).json()] == [job_id]


def test_admin_cancels_anyone_and_users_cannot_list_others(secured):
    job_id = secured.post("/generate", json={"prompt": "slow"}, headers=bot("111")).json()["job_id"]
    wait_for(secured, job_id, lambda b: b["status"] == "running", headers=WEB)
    assert secured.get("/jobs?owner=telegram:111", headers=bot("222")).status_code == 403
    assert secured.post(f"/cancel/{job_id}", headers=WEB).status_code == 200
    wait_for(secured, job_id, lambda b: b["status"] == "cancelled", headers=WEB)


def test_per_user_limit_is_429_but_not_for_admins(secured):
    first = secured.post("/generate", json={"prompt": "slow"}, headers=bot("111")).json()["job_id"]
    assert secured.post("/generate", json={"prompt": "a"}, headers=bot("111")).status_code == 202
    limited = secured.post("/generate", json={"prompt": "b"}, headers=bot("111"))
    assert limited.status_code == 429
    assert secured.post("/generate", json={"prompt": "c"}, headers=bot("222")).status_code == 202   # others are not affected
    for _ in range(3):
        assert secured.post("/generate", json={"prompt": "d"}, headers=WEB).status_code == 202
    secured.post(f"/cancel/{first}", headers=WEB)


def test_only_admins_switch_checkpoints(secured):
    assert secured.post("/checkpoints/load", json={"name": "x"}, headers=bot("111")).status_code == 403
