"""The pictures sent with a job stay available: GET /result/{id}/inputs/{name}, JobInfo.inputs, and the clean-up
of old ones (MY_DIFFUSION_INPUTS_KEEP_DAYS). Uses the stub engine of test_api_jobs (no GPU)."""

from __future__ import annotations

import io
import os
import time

from PIL import Image
from test_api_jobs import WEB, bot, client, png_b64, secured, wait_for  # noqa: F401 - fixtures and helpers

from my_diffusion import api


def submit(client, headers=None, **fields) -> str:
    body = {"prompt": "edit", **fields}
    job_id = client.post("/generate", json=body, headers=headers).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "done", headers)
    return job_id


def test_inputs_are_listed_and_served_after_the_job(client, tmp_path):
    job_id = submit(client, references=[png_b64(), png_b64()])
    info = client.get(f"/result/{job_id}").json()
    assert info["inputs"] == {"ref1": f"/result/{job_id}/inputs/ref1", "ref2": f"/result/{job_id}/inputs/ref2"}
    response = client.get(info["inputs"]["ref2"])
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (32, 24)
    assert sorted(p.name for p in (tmp_path / "inputs" / job_id).iterdir()) == [f"{job_id}_ref1.png", f"{job_id}_ref2.png"]


def test_init_image_and_mask_have_their_own_names(client):
    job_id = submit(client, init_image=png_b64(), mask=png_b64())
    assert set(client.get(f"/result/{job_id}").json()["inputs"]) == {"init", "mask"}
    assert client.get(f"/result/{job_id}/inputs/init").status_code == 200
    assert client.get(f"/result/{job_id}/inputs/mask").status_code == 200


def test_missing_inputs_are_404(client):
    job_id = submit(client, init_image=png_b64())
    assert client.get(f"/result/{job_id}/inputs/ref1").status_code == 404
    assert client.get(f"/result/{job_id}/inputs/..%2F..%2Fjobs.sqlite3").status_code == 404
    assert client.get(f"/result/{job_id}/inputs/init_image").status_code == 404   # only the public names
    plain = submit(client)
    assert client.get(f"/result/{plain}").json()["inputs"] == {}


def test_inputs_follow_the_job_access_rules(secured):
    job_id = submit(secured, bot("1"), references=[png_b64()])
    url = f"/result/{job_id}/inputs/ref1"
    assert secured.get(url, headers=bot("1")).status_code == 200
    assert secured.get(url, headers=WEB).status_code == 200
    assert secured.get(url, headers=bot("2")).status_code == 404
    assert secured.get(url).status_code == 401


def test_inputs_of_a_cancelled_job_are_kept_too(client):
    job_id = client.post("/generate", json={"prompt": "slow", "init_image": png_b64()}).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "running")
    client.post(f"/cancel/{job_id}")
    wait_for(client, job_id, lambda b: b["status"] == "cancelled")
    assert client.get(f"/result/{job_id}/inputs/init").status_code == 200


def test_old_inputs_are_cleaned_but_not_those_of_waiting_jobs(client, tmp_path, monkeypatch):
    old = submit(client, references=[png_b64()])
    recent = submit(client, references=[png_b64()])
    waiting = client.post("/generate", json={"prompt": "slow", "init_image": png_b64()}).json()["job_id"]
    second = client.post("/generate", json={"prompt": "slow", "init_image": png_b64()}).json()["job_id"]   # stays queued
    for job_id in (old, waiting, second):
        os.utime(tmp_path / "inputs" / job_id, (time.time() - 8 * 86400,) * 2)
    (tmp_path / "inputs" / "deadbeef").mkdir()   # no such job: a leftover
    os.utime(tmp_path / "inputs" / "deadbeef", (time.time() - 8 * 86400,) * 2)

    assert api._clean_inputs() == 2   # `old` and the leftover
    assert client.get(f"/result/{old}/inputs/ref1").status_code == 404
    assert client.get(f"/result/{recent}/inputs/ref1").status_code == 200
    assert client.get(f"/result/{waiting}/inputs/init").status_code == 200   # running: needs them
    assert client.get(f"/result/{second}/inputs/init").status_code == 200    # queued: needs them
    client.post(f"/cancel/{second}")
    client.post(f"/cancel/{waiting}")


def test_zero_keep_days_deletes_at_once(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, "inputs_keep_days", 0)
    job_id = submit(client, references=[png_b64()])
    assert not (tmp_path / "inputs" / job_id).exists()
    assert client.get(f"/result/{job_id}/inputs/ref1").status_code == 404
