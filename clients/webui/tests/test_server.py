import json

import pytest
from fastapi.testclient import TestClient

from my_diffusion_webui import server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "SETTINGS_FILE", tmp_path / "settings.json")
    for name in ("API_URL", "API_KEY", "USER_ID"):
        monkeypatch.delenv(f"MY_DIFFUSION_WEBUI_{name}", raising=False)
    return TestClient(server.app)


def test_key_is_saved_but_never_returned(client):
    r = client.put("/ui/connection", json={"api_url": "http://127.0.0.1:9/", "api_key": "secret", "user_id": "me"})
    assert r.json() == {"api_url": "http://127.0.0.1:9", "has_key": True, "user_id": "me"}
    assert "secret" not in client.get("/ui/connection").text
    assert json.loads(server.SETTINGS_FILE.read_text())["api_key"] == "secret"


def test_bad_address_and_user_id_are_refused(client):
    assert client.put("/ui/connection", json={"api_url": "ftp://x"}).status_code == 422
    assert client.put("/ui/connection", json={"user_id": "a b"}).status_code == 422


def test_cross_site_post_is_refused(client):
    r = client.post("/api/generate", json={}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_unreachable_api_is_a_502_with_a_reason(client):
    client.put("/ui/connection", json={"api_url": "http://127.0.0.1:9"})
    r = client.get("/api/health")
    assert r.status_code == 502 and r.json()["unreachable"] is True


def test_static_files_are_revalidated_on_every_load(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_listen_address_is_saved_next_to_the_connection(client):
    client.put("/ui/connection", json={"api_key": "secret", "user_id": "me"})
    r = client.put("/ui/listen", json={"host": "0.0.0.0", "port": 7860})
    assert (r.json()["host"], r.json()["port"]) == ("0.0.0.0", 7860)
    assert server.saved_listen() == ("0.0.0.0", 7860)
    assert client.get("/ui/connection").json()["has_key"] is True   # the other part of the file survived
    assert client.put("/ui/listen", json={"host": "bad host!", "port": 7860}).status_code == 422
    assert client.put("/ui/listen", json={"host": "127.0.0.1", "port": 70000}).status_code == 422
