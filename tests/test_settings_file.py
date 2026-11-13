"""The settings as the config API's registry, and the override file: what is read from it, in which order of
priority, and that a broken file never stops the server from starting. No server and no GPU."""

from __future__ import annotations

import json

import pytest

from my_diffusion.config import (
    APPLY_CLASSES,
    Settings,
    load_settings,
    read_overrides,
    setting_meta,
    write_overrides,
)

KEY = "k" * 32


@pytest.fixture
def override(monkeypatch, tmp_path):
    """An override file of its own, and an environment without other MY_DIFFUSION_* settings."""
    import os

    for name in list(os.environ):
        if name.upper().startswith("MY_DIFFUSION_"):
            monkeypatch.delenv(name)
    path = tmp_path / "data" / "settings.override.json"
    monkeypatch.setenv("MY_DIFFUSION_SETTINGS_FILE", str(path))
    return path


def save(path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


# ---- the registry ---------------------------------------------------------------------------------------------

def test_every_setting_carries_the_metadata_of_the_config_api():
    for name, field in Settings.model_fields.items():
        meta = setting_meta(name)
        assert meta["apply"] in APPLY_CLASSES, name
        assert field.description, f"{name} needs a description for the Settings page"
        assert isinstance(meta["editable"], bool) and isinstance(meta["secret"], bool), name


def test_secrets_and_storage_locations_cannot_be_changed_through_the_api():
    assert setting_meta("api_keys") == {**setting_meta("api_keys"), "secret": True, "editable": False}
    for name in ("output_dir", "database_path", "inputs_dir", "admin_clients", "fs_roots", "settings_file", "host", "port"):
        assert not setting_meta(name)["editable"], name


# ---- the checkpoints and LoRA folders --------------------------------------------------------------------------

def test_model_folders_default_to_the_subfolders_of_models_dir(tmp_path):
    s = Settings(models_dir=tmp_path)
    assert s.checkpoints_dir is None and s.loras_dir is None
    assert s.checkpoint_dir == tmp_path / "checkpoints" and s.lora_dir == tmp_path / "loras"
    assert s.checkpoint_path == tmp_path / "checkpoints" / s.checkpoint


def test_model_folders_can_be_set_separately(tmp_path):
    s = Settings(models_dir=tmp_path, checkpoints_dir=tmp_path / "ckpt", loras_dir=tmp_path / "my-loras")
    assert s.checkpoint_dir == tmp_path / "ckpt" and s.lora_dir == tmp_path / "my-loras"
    assert s.vae_path.parent == tmp_path / "vae"   # the other folders stay under models_dir


# ---- priority: override file > environment > .env > default -----------------------------------------------------

def test_without_the_file_the_environment_applies(override, monkeypatch):
    assert load_settings(_env_file=None).settings.matmul == "fp8_all"   # the default
    monkeypatch.setenv("MY_DIFFUSION_MATMUL", "fp8")
    loaded = load_settings(_env_file=None)
    assert loaded.settings.matmul == "fp8" and loaded.overrides == {} and loaded.problems == []


def test_the_file_is_stronger_than_the_environment_and_dotenv(override, monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("MY_DIFFUSION_MATMUL=exact\nMY_DIFFUSION_QUEUE_MAX_PENDING=5\nMY_DIFFUSION_PIN_MEMORY=false\n", encoding="utf-8")
    monkeypatch.setenv("MY_DIFFUSION_QUEUE_MAX_PENDING", "7")
    monkeypatch.setenv("MY_DIFFUSION_PIN_MEMORY", "true")
    save(override, {"pin_memory": False})

    s = load_settings(_env_file=dotenv).settings
    assert s.matmul == "exact"            # only .env names it
    assert s.queue_max_pending == 7       # the environment beats .env
    assert s.pin_memory is False          # the file beats both
    assert s.attention == "cudnn"         # nobody names it: the default


def test_saved_settings_survive_a_restart(override):
    write_overrides(override, {"matmul": "exact", "queue_max_pending": 32, "checkpoints_dir": "D:\\models\\ckpt"})
    assert not list(override.parent.glob("*.tmp"))   # written to a temporary file and moved into place
    loaded = load_settings(_env_file=None)
    assert loaded.settings.matmul == "exact" and loaded.settings.queue_max_pending == 32
    assert str(loaded.settings.checkpoints_dir) == "D:\\models\\ckpt"
    assert loaded.overrides == {"matmul": "exact", "queue_max_pending": 32, "checkpoints_dir": "D:\\models\\ckpt"}


# ---- a file that is wrong ---------------------------------------------------------------------------------------

def test_secrets_and_read_only_settings_are_not_read_from_the_file(override):
    save(override, {"api_keys": {"evil": KEY}, "admin_clients": ["evil"], "output_dir": "C:\\elsewhere", "matmul": "exact"})
    loaded = load_settings(_env_file=None)
    assert loaded.settings.api_keys == {} and loaded.settings.admin_clients == []
    assert loaded.settings.output_dir == Settings(_env_file=None).output_dir
    assert loaded.settings.matmul == "exact"   # the rest of the file still applies
    assert len(loaded.problems) == 3 and KEY not in " ".join(loaded.problems)


def test_bad_entries_are_dropped_and_reported_and_the_rest_applies(override):
    save(override, {"matmul": "turbo-mode", "queue_max_pending": 0, "no_such_setting": 1, "attention": "default"})
    loaded = load_settings(_env_file=None)
    assert loaded.settings.matmul == "fp8_all" and loaded.settings.queue_max_pending == 16
    assert loaded.settings.attention == "default"
    assert loaded.overrides == {"attention": "default"}
    assert {problem.split(":")[0] for problem in loaded.problems} == {"matmul", "queue_max_pending", "no_such_setting"}


@pytest.mark.parametrize("content", ["{ not json", "[1, 2]", ""])
def test_an_unreadable_file_is_ignored(override, content):
    override.parent.mkdir(parents=True)
    override.write_text(content, encoding="utf-8")
    loaded = load_settings(_env_file=None)
    assert loaded.settings.matmul == "fp8_all" and loaded.overrides == {} and len(loaded.problems) == 1


def test_a_missing_file_is_simply_empty(tmp_path):
    assert read_overrides(tmp_path / "nothing.json") == ({}, [])
