"""HTTP tests of the config API: GET /config, PATCH /config and GET /fs.

The engine is the stub of test_api_jobs with the loading steps replaced by recorders, so the real
Krea2Engine.apply_settings runs (on the slot's thread, under the engine's lock) but nothing is read from disk and
no GPU is touched. The models folder is a temporary tree of empty files."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest
from test_api_jobs import BOT_KEY, WEB, WEB_KEY, bot, make_client, wait_for

from my_diffusion import api, config, runtime_config
from my_diffusion.config import Settings, inside_folder, load_settings
from my_diffusion.engine import Krea2Engine


class Stage:
    """What a test sees of the server: the folders, the override file and what the engine was asked to do."""

    def __init__(self, tmp_path):
        self.models = tmp_path / "models"
        self.extra = tmp_path / "extra"          # an allowed root outside the models folder
        self.outside = tmp_path / "outside"      # not allowed
        self.override = tmp_path / "data" / "settings.override.json"
        self.calls: list[tuple] = []
        self.fail: set[str] = set()
        self.loaded = True

    def saved(self) -> dict:
        return json.loads(self.override.read_text(encoding="utf-8")) if self.override.exists() else {}

    def step(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if name in self.fail:
            raise RuntimeError(f"{name} failed")


@pytest.fixture
def stage(monkeypatch, tmp_path):
    before = dict(api.settings)   # PATCH /config changes the shared settings in place: put them back afterwards
    s = Stage(tmp_path)
    for folder in ("checkpoints", "loras", "vae", "text_encoders/qwen", "text_encoders/qwen2", "rembg"):
        (s.models / folder).mkdir(parents=True)
    for folder in (s.extra / "checkpoints", s.extra / "loras", s.extra / "empty", s.outside / "secret"):
        folder.mkdir(parents=True)
    for file in (
        "checkpoints/a.safetensors", "checkpoints/b.safetensors", "loras/edit.safetensors", "loras/style.safetensors",
        "vae/vae.safetensors", "vae/vae2.safetensors", "rembg/u2net.onnx",
    ):
        (s.models / file).touch()
    for file in (s.extra / "checkpoints" / "c.safetensors", s.extra / "checkpoints" / "notes.txt", s.extra / "loras" / "other.safetensors"):
        file.touch()
    values = {
        "models_dir": s.models, "checkpoints_dir": None, "loras_dir": None, "fs_roots": [s.extra],
        "checkpoint": "a.safetensors", "vae": "vae.safetensors", "text_encoder": "qwen", "edit_lora": "edit",
        "rembg_model": "rembg/u2net.onnx", "settings_file": s.override, "matmul": "fp8_all", "attention": "cudnn",
        "attention_sage2_min_tokens": 6000, "pin_memory": True, "queue_max_pending": 16, "api_docs": True,
        "devices": None, "max_references": 2,
    }
    for name, value in values.items():
        monkeypatch.setattr(api.settings, name, value)
    monkeypatch.setattr(api, "_overrides", {})
    monkeypatch.setattr(api, "_config_problems", [])
    yield s
    for name, value in before.items():
        setattr(api.settings, name, value)


def stub_engine(monkeypatch, s: Stage) -> None:
    """Call after make_client (it patches the shared settings too): the engine gets its copy of them and
    loading steps that only record."""
    engine = api.engine
    monkeypatch.setattr(engine, "settings", api.settings.model_copy(update={"device": engine.settings.device}))
    for name, value in (
        ("matmul", "fp8_all"), ("_attention", ("cudnn", 6000)), ("checkpoint_name", "a.safetensors"),
        ("checkpoint_file", (s.models / "checkpoints" / "a.safetensors").resolve()), ("_checkpoint_request", "a.safetensors"),
    ):
        monkeypatch.setattr(engine, name, value)
    monkeypatch.setattr(Krea2Engine, "loaded", property(lambda self: s.loaded))

    def load_checkpoint(name: str) -> None:
        path = inside_folder(engine.settings.checkpoint_dir, name)
        s.step("load_checkpoint", name)
        engine.checkpoint_file, engine._checkpoint_request, engine.checkpoint_name = path, name, path.name

    def load(checkpoint=None) -> None:
        s.step("load", checkpoint)
        load_checkpoint(checkpoint or engine.settings.checkpoint)
        s.loaded = True

    def unload() -> None:
        s.step("unload")
        s.loaded = False

    def set_matmul(mode: str) -> None:
        s.step("set_matmul", mode)
        engine.matmul = mode

    def set_attention() -> None:
        s.step("set_attention", engine.settings.attention)
        engine._attention = (engine.settings.attention, engine.settings.attention_sage2_min_tokens)

    for func in (load_checkpoint, load, unload, set_matmul, set_attention):
        monkeypatch.setattr(engine, func.__name__, func)


def started(monkeypatch, tmp_path, s: Stage, keys: bool):
    test_client = make_client(monkeypatch, tmp_path, keys=keys)
    stub_engine(monkeypatch, s)
    with test_client:
        deadline = time.monotonic() + 10
        while api.scheduler.paused:   # the load at start runs on the slot's thread; the queue starts after it
            assert time.monotonic() < deadline, "the models did not load"
            time.sleep(0.01)
        s.calls.clear()   # the tests look at what a change of settings asks for, not at the start
        yield test_client


@pytest.fixture
def client(monkeypatch, tmp_path, stage):
    """Access control off: every request is the local admin."""
    yield from started(monkeypatch, tmp_path, stage, keys=False)


@pytest.fixture
def secured(monkeypatch, tmp_path, stage):
    """Access control on: "web" is an admin client, "telegram" a service."""
    yield from started(monkeypatch, tmp_path, stage, keys=True)


def setting(body: dict, name: str) -> dict:
    return next(item for item in body["settings"] if item["name"] == name)


def patch(client, headers=None, **body):
    return client.patch("/config", json=body, headers=headers)


# ---- reading ----------------------------------------------------------------------------------------------------

def test_every_setting_is_described(client):
    body = client.get("/config").json()
    assert [item["name"] for item in body["settings"]] == list(Settings.model_fields)
    matmul = setting(body, "matmul")
    assert matmul == {
        **matmul, "value": "fp8_all", "default": "fp8_all", "source": "default", "apply": "live", "editable": True,
        "type": "string", "choices": ["exact", "fp8", "fp8_all"], "env": "MY_DIFFUSION_MATMUL", "group": "performance",
    }
    assert matmul["description"]
    assert setting(body, "queue_max_pending") == {**setting(body, "queue_max_pending"), "type": "integer", "minimum": 1}
    assert setting(body, "checkpoint")["apply"] == "reload_models" and setting(body, "devices")["apply"] == "restart"
    assert setting(body, "output_dir")["editable"] is False


def test_model_folders_show_the_folder_really_used(client, stage):
    item = setting(client.get("/config").json(), "checkpoints_dir")
    assert item["type"] == "directory" and item["nullable"] is True and item["value"] is None
    assert item["effective"] == str(stage.models / "checkpoints")


def test_secrets_are_never_returned(secured):
    response = secured.get("/config", headers=WEB)
    assert WEB_KEY not in response.text and BOT_KEY not in response.text
    body = response.json()
    keys = setting(body, "api_keys")
    assert keys["secret"] is True and keys["value"] is None and keys["default"] is None and keys["editable"] is False
    assert body["access_control"] == {"enabled": True, "clients": ["telegram", "web"], "admin_clients": ["web"]}
    assert WEB_KEY not in patch(secured, WEB, set={"matmul": "exact"}).text


def test_the_source_says_where_a_value_comes_from(client, monkeypatch):
    monkeypatch.setenv("MY_DIFFUSION_ATTENTION", "default")
    patch(client, set={"matmul": "exact"})
    body = client.get("/config").json()
    assert setting(body, "matmul")["source"] == "override"
    assert setting(body, "attention")["source"] == "env"
    assert setting(body, "pin_memory")["source"] == "default"


# ---- access -----------------------------------------------------------------------------------------------------

def test_only_admins_read_and_change_the_configuration(secured):
    for call in (
        lambda headers: secured.get("/config", headers=headers),
        lambda headers: patch(secured, headers, set={"matmul": "exact"}),
        lambda headers: secured.get("/fs", headers=headers),
    ):
        assert call(None).status_code == 401
        assert call(bot("111")).status_code == 403
        assert call(WEB).status_code == 200
    assert api.settings.matmul == "exact"   # only the admin's call went through


# ---- live settings ----------------------------------------------------------------------------------------------

def test_a_live_setting_applies_at_once_and_is_saved(client, stage):
    response = patch(client, set={"matmul": "exact"})
    assert response.status_code == 200
    body = response.json()
    assert body == {**body, "applied": ["matmul"], "restart_required": [], "unchanged": [], "reloaded": "none", "dry_run": False}
    assert setting(body["config"], "matmul")["value"] == "exact"
    assert api.settings.matmul == "exact" and api.engine.settings.matmul == "exact" and api.engine.matmul == "exact"
    assert stage.calls == [("set_matmul", "exact")]
    assert stage.saved() == {"matmul": "exact"}


def test_attention_is_switched_without_reloading_the_models(client, stage):
    assert patch(client, set={"attention": "default", "attention_sage2_min_tokens": 0}).json()["reloaded"] == "none"
    assert stage.calls == [("set_attention", "default")]
    assert api.engine.settings.attention_sage2_min_tokens == 0


def test_sage2_without_its_package_is_refused_before_anything_changes(client, stage, monkeypatch):
    monkeypatch.setattr(runtime_config.importlib.util, "find_spec", lambda name: None)
    response = patch(client, set={"attention": "sage2"})
    assert response.status_code == 422 and "uv sync --group sage2" in response.json()["errors"]["attention"]
    assert api.settings.attention == "cudnn" and stage.calls == []


def test_queue_limits_apply_without_touching_the_engine(secured, stage):
    old_engine_settings = api.engine.settings
    assert patch(secured, WEB, set={"queue_max_per_user": 1}).json()["applied"] == ["queue_max_per_user"]
    assert api.engine.settings is old_engine_settings and stage.calls == []
    first = secured.post("/generate", json={"prompt": "slow"}, headers=bot("111")).json()["job_id"]
    assert secured.post("/generate", json={"prompt": "a"}, headers=bot("111")).status_code == 429
    secured.post(f"/cancel/{first}", headers=WEB)


def test_a_limit_read_by_both_the_api_and_the_engine_changes_in_both(client):
    assert patch(client, set={"max_references": 1}).status_code == 200
    assert api.settings.max_references == 1 and api.engine.settings.max_references == 1


def test_a_running_generation_keeps_its_settings_and_the_change_waits_for_it(client, stage):
    job_id = client.post("/generate", json={"prompt": "slow"}).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "running")
    answers = []
    caller = threading.Thread(target=lambda: answers.append(patch(client, set={"matmul": "exact"})))
    caller.start()
    time.sleep(0.3)
    assert caller.is_alive() and api.engine.settings.matmul == "fp8_all" and stage.calls == []
    client.post(f"/cancel/{job_id}")
    caller.join(timeout=10)
    assert answers[0].status_code == 200 and api.engine.settings.matmul == "exact"


def test_dry_run_checks_and_reports_but_changes_nothing(client, stage):
    body = patch(client, set={"matmul": "exact", "checkpoint": "b.safetensors", "api_docs": False}, dry_run=True).json()
    assert body["dry_run"] is True and body["reloaded"] == "checkpoint"
    assert sorted(body["applied"]) == ["checkpoint", "matmul"] and body["restart_required"] == ["api_docs"]
    assert api.settings.matmul == "fp8_all" and stage.calls == [] and stage.saved() == {}
    assert patch(client, set={"matmul": "nope"}, dry_run=True).status_code == 422


def test_a_value_that_is_already_set_is_unchanged_but_still_saved(client, stage):
    body = patch(client, set={"matmul": "fp8_all"}).json()
    assert body["unchanged"] == ["matmul"] and body["applied"] == [] and stage.calls == []
    assert stage.saved() == {"matmul": "fp8_all"}   # pinned: a later change of .env does not move it


# ---- settings that reload the models ----------------------------------------------------------------------------

def test_reloading_is_refused_while_jobs_are_waiting_or_running(client, stage):
    job_id = client.post("/generate", json={"prompt": "slow"}).json()["job_id"]
    wait_for(client, job_id, lambda b: b["status"] == "running")
    response = patch(client, set={"checkpoint": "b.safetensors"})
    assert response.status_code == 409 and "reloads the models" in response.json()["detail"]
    assert api.settings.checkpoint == "a.safetensors" and stage.calls == [] and stage.saved() == {}
    client.post(f"/cancel/{job_id}")
    wait_for(client, job_id, lambda b: b["status"] == "cancelled")

    body = patch(client, set={"checkpoint": "b.safetensors"}).json()
    assert body["applied"] == ["checkpoint"] and body["reloaded"] == "checkpoint"
    assert stage.calls == [("load_checkpoint", "b.safetensors")]
    assert client.get("/health").json()["checkpoint"] == "b.safetensors"


def test_vae_reloads_all_the_models(client, stage):
    assert patch(client, set={"vae": "vae2.safetensors"}).json()["reloaded"] == "all"
    assert [call[0] for call in stage.calls] == ["unload", "load", "load_checkpoint"]


def test_switching_pinning_back_on_warns_that_the_ram_stays_until_a_restart(client, stage):
    off = patch(client, set={"pin_memory": False}).json()
    assert off["reloaded"] == "all" and off["warnings"] == []   # the pinned RAM does go back, see pinned.release_unused
    on = patch(client, reset=["pin_memory"]).json()
    assert on["reloaded"] == "all" and len(on["warnings"]) == 1
    assert "pin_memory" in on["warnings"][0] and "restart" in on["warnings"][0]


def test_models_that_failed_to_load_can_be_repaired_through_the_api(client, stage, monkeypatch):
    stage.loaded = False
    monkeypatch.setattr(api, "_load_error", "FileNotFoundError: no file 'a.safetensors'")
    api.scheduler.pause()
    assert client.get("/health").json()["status"] == "error"

    assert patch(client, set={"checkpoint": "b.safetensors"}).json()["reloaded"] == "all"
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["error"] is None and health["queue"]["paused"] is False


# ---- all or nothing ---------------------------------------------------------------------------------------------

def test_one_bad_value_refuses_the_whole_request(client, stage):
    response = patch(client, set={"matmul": "exact", "queue_max_pending": 0, "attention": "fastest"})
    assert response.status_code == 422
    body = response.json()
    assert set(body["errors"]) == {"queue_max_pending", "attention"}
    assert "greater than or equal to 1" in body["errors"]["queue_max_pending"] and "queue_max_pending" in body["detail"]
    assert api.settings.matmul == "fp8_all" and api.settings.queue_max_pending == 16
    assert stage.calls == [] and stage.saved() == {}


def test_unknown_read_only_and_secret_settings_are_refused(client):
    response = patch(client, set={"no_such": 1, "output_dir": "C:\\x", "api_keys": {"me": "k" * 32}, "fs_roots": ["C:\\"]}, reset=["port"])
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert errors["no_such"] == "no such setting"
    for name in ("output_dir", "api_keys", "fs_roots", "port"):
        assert "cannot be changed through the API" in errors[name]
    assert patch(client, set={"matmul": "exact"}, reset=["matmul"]).status_code == 422
    assert client.patch("/config", json={"matmul": "exact"}).status_code == 422   # not wrapped in "set"


def test_a_change_the_engine_cannot_apply_is_undone(client, stage):
    old_engine_settings = api.engine.settings
    stage.fail = {"set_attention"}
    response = patch(client, set={"matmul": "exact", "attention": "default", "queue_max_pending": 4})
    assert response.status_code == 400 and "set_attention failed" in response.json()["detail"]
    assert api.settings.matmul == "fp8_all" and api.settings.attention == "cudnn" and api.settings.queue_max_pending == 16
    assert api.engine.settings is old_engine_settings and api.engine.matmul == "fp8_all"
    assert stage.saved() == {}
    assert client.get("/health").json()["status"] == "ok"


def test_a_change_that_cannot_be_saved_is_undone(client, stage, monkeypatch):
    def disk_full(path, values):
        raise OSError("disk full")

    old_engine_settings = api.engine.settings
    monkeypatch.setattr(api, "write_overrides", disk_full)
    response = patch(client, set={"matmul": "exact", "queue_max_pending": 4})
    assert response.status_code == 500 and "disk full" in response.json()["detail"]
    assert api.settings.matmul == "fp8_all" and api.settings.queue_max_pending == 16
    assert api.engine.settings is old_engine_settings and api.engine.matmul == "fp8_all"
    assert setting(client.get("/config").json(), "matmul")["source"] == "default"


def test_a_failed_reload_that_cannot_be_undone_is_shown_by_health(client, stage):
    stage.fail = {"load"}
    response = patch(client, set={"vae": "vae2.safetensors"})
    assert response.status_code == 400 and "The models are not loaded now" in response.json()["detail"]
    assert api.settings.vae == "vae.safetensors"
    health = client.get("/health").json()
    assert health["status"] == "error" and "load failed" in health["error"] and health["queue"]["paused"] is True


# ---- settings that need a restart -------------------------------------------------------------------------------

def test_a_restart_setting_is_saved_but_not_applied(client, stage):
    body = patch(client, set={"api_docs": False, "devices": "auto"}).json()
    assert body["applied"] == [] and sorted(body["restart_required"]) == ["api_docs", "devices"]
    assert api.settings.api_docs is True and api.settings.devices is None
    assert stage.saved() == {"api_docs": False, "devices": "auto"}
    item = setting(body["config"], "api_docs")
    assert item == {**item, "value": True, "saved": False, "pending_restart": True, "source": "override"}
    assert {"api_docs", "devices"} <= set(body["config"]["restart_pending"])


def test_devices_are_checked_before_they_are_saved(client, stage):
    assert "empty list" in patch(client, set={"devices": []}).json()["errors"]["devices"]
    assert patch(client, set={"devices": ["banana"]}).status_code == 422
    assert patch(client, set={"device": "cuda:99"}).status_code == 422
    assert stage.saved() == {}


# ---- persistence ------------------------------------------------------------------------------------------------

def test_saved_settings_are_what_the_next_start_reads(client, stage, monkeypatch):
    patch(client, set={"matmul": "exact", "queue_max_pending": 5, "api_docs": False})
    monkeypatch.setenv("MY_DIFFUSION_SETTINGS_FILE", str(stage.override))
    monkeypatch.setenv("MY_DIFFUSION_MATMUL", "fp8")   # the file is stronger
    restarted = load_settings().settings
    assert (restarted.matmul, restarted.queue_max_pending, restarted.api_docs) == ("exact", 5, False)


def test_reset_removes_the_saved_value_and_falls_back_to_the_environment(client, stage, monkeypatch):
    patch(client, set={"matmul": "exact", "queue_max_pending": 5, "api_docs": False})
    monkeypatch.setenv("MY_DIFFUSION_QUEUE_MAX_PENDING", "9")
    body = patch(client, reset=["matmul", "queue_max_pending", "api_docs"]).json()
    assert sorted(body["applied"]) == ["matmul", "queue_max_pending"] and body["unchanged"] == ["api_docs"]
    assert api.settings.matmul == Settings.model_fields["matmul"].default and api.settings.queue_max_pending == 9
    assert stage.saved() == {}
    assert setting(body["config"], "queue_max_pending")["source"] == "env"
    assert setting(body["config"], "api_docs")["pending_restart"] is False


def test_entries_ignored_at_start_are_reported(client, monkeypatch):
    monkeypatch.setattr(api, "_config_problems", ["matmul: Input should be 'exact', 'fp8' or 'fp8_all'; ignored"])
    assert client.get("/config").json()["problems"] == ["matmul: Input should be 'exact', 'fp8' or 'fp8_all'; ignored"]


# ---- model folders ----------------------------------------------------------------------------------------------

def test_the_lora_folder_changes_at_once(client, stage):
    assert client.get("/loras").json() == ["edit.safetensors", "style.safetensors"]
    body = patch(client, set={"loras_dir": str(stage.extra / "loras")}).json()
    assert body["applied"] == ["loras_dir"] and body["reloaded"] == "none" and stage.calls == []
    assert client.get("/loras").json() == ["other.safetensors"]
    assert "edit LoRA is not in the new folder" in body["warnings"][0]   # allowed, but reference editing would fail
    assert setting(body["config"], "loras_dir")["effective"] == str(stage.extra / "loras")
    assert str(stage.extra / "loras") in body["config"]["fs_roots"] or str(stage.extra) in body["config"]["fs_roots"]


def test_the_checkpoint_folder_needs_the_checkpoint_in_it(client, stage):
    folder = str(stage.extra / "checkpoints")
    response = patch(client, set={"checkpoints_dir": folder})
    assert response.status_code == 422
    assert "set `checkpoint` to a file of that folder" in response.json()["errors"]["checkpoints_dir"]

    body = patch(client, set={"checkpoints_dir": folder, "checkpoint": "c.safetensors"}).json()
    assert sorted(body["applied"]) == ["checkpoint", "checkpoints_dir"] and body["reloaded"] == "checkpoint"
    assert client.get("/checkpoints").json()["available"] == ["c.safetensors"]

    back = patch(client, set={"checkpoints_dir": None, "checkpoint": "a.safetensors"}).json()   # null = under models_dir again
    assert back["reloaded"] == "checkpoint" and client.get("/checkpoints").json()["current"] == "a.safetensors"


def test_a_folder_must_exist_be_absolute_and_lie_inside_the_allowed_roots(client, stage):
    def refused(value) -> str:
        response = patch(client, set={"loras_dir": value})
        assert response.status_code == 422, value
        return response.json()["errors"]["loras_dir"]

    assert "outside the folders the server may show" in refused(str(stage.outside))
    assert "outside the folders" in refused(str(stage.extra / ".." / "outside"))      # '..' is resolved first
    assert "outside the folders" in refused(str(stage.outside / "does-not-exist"))    # no hint whether it exists
    assert "no folder" in refused(str(stage.extra / "does-not-exist"))
    assert "no folder" in refused(str(stage.extra / "checkpoints" / "c.safetensors"))  # a file
    assert "must be absolute" in refused("loras")
    assert api.settings.loras_dir is None


def test_model_file_names_cannot_leave_their_folder(client, stage):
    (stage.models / "stray.safetensors").touch()
    for name, value in (("checkpoint", "../stray.safetensors"), ("edit_lora", "..\\stray"), ("vae", str(stage.models / "vae" / "vae2.safetensors"))):
        response = patch(client, set={name: value})
        assert response.status_code == 422 and "no absolute path and no '..'" in response.json()["errors"][name]
    assert "no file" in patch(client, set={"edit_lora": "missing"}).json()["errors"]["edit_lora"]
    assert "no file" in patch(client, set={"checkpoint": "missing.safetensors"}).json()["errors"]["checkpoint"]
    assert "no VAE file" in patch(client, set={"vae": "missing.safetensors"}).json()["errors"]["vae"]
    assert "no text encoder folder" in patch(client, set={"text_encoder": "missing"}).json()["errors"]["text_encoder"]


# ---- GET /fs ----------------------------------------------------------------------------------------------------

def link(source, target) -> bool:
    """A folder link source -> target: a symbolic link, or a junction where Windows refuses symbolic links."""
    try:
        os.symlink(target, source, target_is_directory=True)
        return True
    except OSError:
        if sys.platform != "win32":
            return False
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(source))
        return True
    except OSError:
        return False


def test_without_a_path_the_allowed_roots_are_listed(client, stage):
    body = client.get("/fs").json()
    roots = [str(stage.extra.resolve()), str(stage.models.resolve())]
    assert body["path"] is None and body["roots"] == sorted(roots)
    assert [item["path"] for item in body["dirs"]] == sorted(roots)   # checkpoints/ and loras/ lie inside models/


def test_only_folders_are_listed(client, stage):
    body = client.get("/fs", params={"path": str(stage.extra)}).json()
    assert body["path"] == str(stage.extra.resolve()) and body["parent"] is None   # above it is outside the roots
    assert [item["name"] for item in body["dirs"]] == ["checkpoints", "empty", "loras"]
    assert body["dirs"][0]["path"] == str((stage.extra / "checkpoints").resolve())

    inner = client.get("/fs", params={"path": body["dirs"][0]["path"]}).json()
    assert inner["dirs"] == [] and inner["model_files"] == 1 and inner["parent"] == str(stage.extra.resolve())
    assert inner["truncated"] is False


def test_the_file_system_outside_the_roots_is_closed(client, stage):
    for path in (stage.outside, stage.outside / "secret", stage.extra / ".." / "outside", stage.extra.parent, stage.outside / "nothing-here"):
        response = client.get("/fs", params={"path": str(path)})
        assert response.status_code == 403, path
        assert "MY_DIFFUSION_FS_ROOTS" in response.json()["detail"]
    assert client.get("/fs", params={"path": str(stage.extra / "nothing-here")}).status_code == 404
    assert client.get("/fs", params={"path": "relative/folder"}).status_code == 400
    assert client.get("/fs", params={"path": "bad\0path"}).status_code == 400


def test_links_cannot_lead_out_of_the_roots(client, stage):
    if not link(stage.extra / "way-out", stage.outside) or not link(stage.extra / "shortcut", stage.extra / "loras"):
        pytest.skip("this machine allows neither symbolic links nor junctions")
    names = [item["name"] for item in client.get("/fs", params={"path": str(stage.extra)}).json()["dirs"]]
    assert "way-out" not in names and "shortcut" in names   # a link that stays inside the roots is fine
    assert client.get("/fs", params={"path": str(stage.extra / "way-out")}).status_code == 403
    assert client.get("/fs", params={"path": str(stage.extra / "way-out" / "secret")}).status_code == 403
    response = patch(client, set={"loras_dir": str(stage.extra / "way-out")})
    assert response.status_code == 422 and "outside the folders" in response.json()["errors"]["loras_dir"]


def test_long_listings_are_cut(client, stage, monkeypatch):
    monkeypatch.setattr(runtime_config, "MAX_FOLDERS", 2)
    body = client.get("/fs", params={"path": str(stage.extra)}).json()
    assert len(body["dirs"]) == 2 and body["truncated"] is True


def test_the_baseline_survives_a_broken_environment(client, monkeypatch):
    monkeypatch.setenv("MY_DIFFUSION_QUEUE_MAX_PENDING", "minus one")   # edited after the server started
    body = client.get("/config").json()
    assert any("no longer validates" in problem for problem in body["problems"])
    assert patch(client, set={"matmul": "exact"}).status_code == 200            # setting still works
    assert "cannot be reset" in patch(client, reset=["matmul"]).json()["errors"]["matmul"]


def test_config_module_is_the_only_place_that_names_the_settings():
    # The endpoints get everything from the registry: a setting added to Settings appears without touching the API.
    assert set(api._API_ONLY) <= set(config.Settings.model_fields)
