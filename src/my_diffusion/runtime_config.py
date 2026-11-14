"""The config API without HTTP: describing the settings (GET /config), checking a change before anything is
touched (PATCH /config) and showing folders (GET /fs). The endpoints and the applying live in api.py.

Nothing here knows particular settings by name, except the checks that need the file system (model files,
folders, devices): everything else comes from the metadata next to each field in config.py.

Folders. The API lets an admin pick the model folders, so it must show folders of the server - but not the
whole file system to whoever holds an admin key. Everything is therefore confined to the ALLOWED ROOTS:
settings.fs_roots (set in the environment only) plus the model folders the server already reads. The same roots
limit what models_dir / checkpoints_dir / loras_dir may be set to: otherwise pointing loras_dir at any folder
and reading GET /loras would list it anyway. Every path is compared by its real location (resolve(): '..',
symbolic links and junctions are followed first), so a link inside a root cannot lead out of the roots."""

from __future__ import annotations

import importlib.util
import os
import types
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import ValidationError

from .config import Settings, inside_folder, setting_meta, validation_messages

ENV_PREFIX = Settings.model_config["env_prefix"]
MODEL_SUFFIXES = (".safetensors", ".gguf")
MAX_FOLDERS = 2000   # a longer listing is cut (and marked): nobody picks a folder from such a list by hand
_PLAIN_TYPES = {bool: "boolean", int: "integer", float: "number", str: "string", Path: "path"}
_RELOAD_ORDER = ("none", "checkpoint", "all")


def heaviest_reload(plans) -> str:
    """Of several engines' reload plans (Krea2Engine.reload_plan), the one that reloads the most."""
    return max(plans, key=_RELOAD_ORDER.index, default="none")


# ---------------------------------------------------------------------- where the values come from
@dataclass
class Baseline:
    """What the environment gives right now, without the override file: the values a setting falls back to when
    its override is removed, and which fields come from an environment variable or from .env."""

    settings: Settings | None = None   # None: the environment does not validate any more
    env: frozenset[str] = frozenset()
    dotenv: frozenset[str] = frozenset()
    problem: str | None = None


def read_baseline() -> Baseline:
    try:
        both = Settings()
        env_only = Settings(_env_file=None)
    except ValidationError as exc:
        text = "; ".join(f"{name}: {message}" for name, message in validation_messages(exc).items())
        return Baseline(problem=f"the environment or .env no longer validates ({text})")
    # pydantic-settings passes what it found in the sources to the model as arguments, so "set" = found there.
    env = frozenset(env_only.model_fields_set)
    return Baseline(both, env, frozenset(both.model_fields_set) - env)


# ---------------------------------------------------------------------- GET /config
def _type_info(name: str) -> dict:
    """The field's type in the terms of a settings form: type, nullable, choices, minimum / maximum."""
    info = Settings.model_fields[name]
    annotation = info.annotation
    nullable = False
    if get_origin(annotation) in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        nullable = len(members) < len(get_args(annotation))
        annotation = members[0] if len(members) == 1 else None   # a real union is edited as JSON
    described: dict[str, Any] = {"nullable": nullable}
    if get_origin(annotation) is Literal:
        described |= {"type": "string", "choices": list(get_args(annotation))}
    else:
        described["type"] = _PLAIN_TYPES.get(annotation, "json")
    if setting_meta(name).get("kind") == "directory":
        described["type"] = "directory"
    for constraint in info.metadata:
        for attr, key in (("ge", "minimum"), ("le", "maximum"), ("gt", "exclusive_minimum"), ("lt", "exclusive_maximum")):
            if getattr(constraint, attr, None) is not None:
                described[key] = getattr(constraint, attr)
    return described


def describe(settings: Settings, overrides: dict[str, Any], problems: list[str], baseline: Baseline) -> dict:
    """The answer of GET /config: every setting with its value and metadata. Secret values are never included."""
    values = settings.model_dump(mode="json")
    defaults = Settings.model_construct().model_dump(mode="json")
    fallback = values if baseline.settings is None else baseline.settings.model_dump(mode="json")
    described = []
    restart_pending = []
    for name, info in Settings.model_fields.items():
        meta = setting_meta(name)
        if name in overrides:
            source = "override"
        elif name in baseline.env:
            source = "env"
        elif name in baseline.dotenv:
            source = "dotenv"
        else:
            source = "default"
        item: dict[str, Any] = {
            "name": name, "group": meta["group"], "env": ENV_PREFIX + name.upper(), "description": info.description,
            **_type_info(name),
            "apply": meta["apply"], "editable": meta["editable"], "secret": meta["secret"],
            "value": None if meta["secret"] else values[name],
            "default": None if meta["secret"] else defaults[name],
            "source": source,
        }
        if "effective" in meta:
            item["effective"] = str(getattr(settings, meta["effective"]))
        # A restart setting may be saved with another value than the server runs with.
        pending = False
        if meta["apply"] == "restart":
            after_restart = overrides[name] if name in overrides else fallback[name]
            pending = after_restart != values[name]
            if name in overrides and not meta["secret"]:
                item["saved"] = overrides[name]
            if pending:
                restart_pending.append(name)
        item["pending_restart"] = pending
        described.append(item)
    return {
        "settings": described,
        "restart_pending": restart_pending,
        "access_control": {
            "enabled": bool(settings.api_keys), "clients": sorted(settings.api_keys),
            "admin_clients": list(settings.admin_clients),
        },
        "override_file": str(settings.settings_file),
        "fs_roots": [str(root) for root in allowed_roots(settings)],
        "problems": [*problems, *([baseline.problem] if baseline.problem else [])],
    }


# ---------------------------------------------------------------------- PATCH /config: the check
class ChangeRefused(Exception):
    """The requested change does not validate. errors: {setting name: what is wrong}."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{name}: {text}" for name, text in errors.items()))
        self.errors = errors


@dataclass
class Change:
    """A checked change, ready to be applied."""

    running: Settings                    # the settings to run with (restart settings keep their running value)
    overrides: dict[str, Any]            # the new content of the override file
    applied: list[str] = field(default_factory=list)            # take effect now: their running value changes
    restart_required: list[str] = field(default_factory=list)   # saved; the running value differs until a restart
    unchanged: list[str] = field(default_factory=list)          # named in the request, but already had that value
    warnings: list[str] = field(default_factory=list)


def _check_device(name: str) -> str | None:
    import torch   # only here: the rest of this module works without it

    try:
        device = torch.device(name)
    except (RuntimeError, ValueError):
        return f"not a torch device: {name!r}"
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return "CUDA is not available on this server"
        if device.index is not None and device.index >= torch.cuda.device_count():
            return f"no such GPU: {name} (this server has {torch.cuda.device_count()})"
    return None


def _plain_name(value: str) -> str | None:
    path = PurePath(value)
    if not value.strip() or path.is_absolute() or path.drive or path.root or ".." in path.parts:
        return "must be a name inside its folder: no absolute path and no '..'"
    return None


def _check(current: Settings, running: Settings, wanted: Settings, to_set: set[str], named: set[str], roots: list[Path]):
    """The checks that pydantic cannot do: folders, model files, the optional package, devices.
    running - the settings to run with; wanted - the same plus the restart settings as asked. Returns (errors, warnings)."""
    errors: dict[str, str] = {}
    warnings: list[str] = []

    def blame(*names: str) -> str:
        """The setting to report the problem on: the first of `names` that this request touches."""
        return next((name for name in names if name in named), names[0])

    for name in named:
        if setting_meta(name).get("kind") != "directory" or getattr(wanted, name) is None:
            continue
        if getattr(wanted, name) == getattr(current, name):
            continue
        # A value sent by the caller must also lie inside the allowed roots; one coming back from the
        # environment (reset) is the server owner's own configuration.
        problem = check_directory(getattr(wanted, name), roots if name in to_set else None)
        if problem:
            errors[name] = problem
    for name in ("checkpoint", "vae", "text_encoder", "edit_lora", "rembg_model"):
        if name in to_set and (problem := _plain_name(getattr(wanted, name))):
            errors[name] = problem
    if errors:
        return errors, warnings   # the file checks below would only repeat the same problem

    if running.vae_path != current.vae_path and not running.vae_path.is_file():
        errors[blame("vae", "models_dir")] = f"no VAE file {running.vae_path}"
    if running.text_encoder_dir != current.text_encoder_dir and not running.text_encoder_dir.is_dir():
        errors[blame("text_encoder", "models_dir")] = f"no text encoder folder {running.text_encoder_dir}"
    if (running.checkpoint, running.checkpoint_dir) != (current.checkpoint, current.checkpoint_dir):
        try:
            inside_folder(running.checkpoint_dir, running.checkpoint)
        except (ValueError, FileNotFoundError) as exc:
            name = blame("checkpoint", "checkpoints_dir", "models_dir")
            hint = "" if name == "checkpoint" else "; set `checkpoint` to a file of that folder in the same request"
            errors[name] = f"{exc}{hint}"
    if (running.edit_lora, running.lora_dir) != (current.edit_lora, current.lora_dir):
        try:
            inside_folder(running.lora_dir, running.edit_lora)
        except (ValueError, FileNotFoundError) as exc:
            if "edit_lora" in named:
                errors["edit_lora"] = str(exc)
            else:   # a LoRA folder without the edit LoRA is fine for everything but reference editing
                warnings.append(f"the edit LoRA is not in the new folder ({exc}): reference editing will fail until it is there")
    if running.rembg_model_path != current.rembg_model_path and not running.rembg_model_path.is_file():
        if "rembg_model" in named:
            errors["rembg_model"] = f"no file {running.rembg_model_path}"
        else:
            warnings.append(f"no background-removal model at {running.rembg_model_path}: remove_background will fail until it is there")

    def pinned(s: Settings) -> bool:
        return s.pin_memory and s.device != "cpu"

    if pinned(running) and not pinned(current):
        warnings.append("pin_memory: the RAM that held the unpinned weights (about as much as the models weigh) "
                        "stays with the server process until it restarts; restart the server to give it back")
    switched_to_sage2 = running.attention == "sage2" and current.attention != "sage2"
    if switched_to_sage2 and importlib.util.find_spec("sageattention") is None:
        errors["attention"] = "the sage2 kernel needs the optional sageattention package: install it with `uv sync --group sage2`"

    if "device" in to_set and (problem := _check_device(wanted.device)):
        errors["device"] = problem
    if "devices" in to_set and isinstance(wanted.devices, list):
        if not wanted.devices:
            errors["devices"] = "an empty list; use null for the default device"
        elif problems := [p for p in map(_check_device, wanted.devices) if p]:
            errors["devices"] = "; ".join(problems)
    return errors, warnings


def plan_change(
    current: Settings, overrides: dict[str, Any], to_set: dict[str, Any], to_reset: list[str],
    baseline: Settings | None, roots: list[Path],
) -> Change:
    """Checks a change of settings as a whole and says what it would do. Nothing is touched.

    to_set - new values; to_reset - settings whose override is removed, so that they fall back to `baseline`
    (the environment's values). Raises ChangeRefused with a message per setting if anything is wrong."""
    errors: dict[str, str] = {}
    for name in dict.fromkeys([*to_set, *to_reset]):
        if name not in Settings.model_fields:
            errors[name] = "no such setting"
        elif not setting_meta(name)["editable"]:
            errors[name] = f"cannot be changed through the API; set {ENV_PREFIX}{name.upper()} in the environment and restart"
        elif name in to_set and name in to_reset:
            errors[name] = "named in both `set` and `reset`"
    if to_reset and baseline is None:
        errors.update({name: "cannot be reset: the environment or .env no longer validates" for name in to_reset})
    if errors:
        raise ChangeRefused(errors)

    values = {**to_set, **{name: getattr(baseline, name) for name in to_reset}}
    try:
        wanted = Settings(**{**dict(current), **values})   # every field is given, so the environment is not read
    except ValidationError as exc:
        raise ChangeRefused(validation_messages(exc)) from None

    change = Change(running=current, overrides={name: value for name, value in overrides.items() if name not in to_reset})
    if to_set:   # an empty `include` would mean "every field"
        change.overrides.update(wanted.model_dump(mode="json", include=set(to_set)))
    running_values = {}
    for name in values:
        if getattr(wanted, name) == getattr(current, name):
            change.unchanged.append(name)
        elif setting_meta(name)["apply"] == "restart":
            change.restart_required.append(name)
        else:
            change.applied.append(name)
            running_values[name] = getattr(wanted, name)
    change.running = current.model_copy(update=running_values)

    errors, change.warnings = _check(current, change.running, wanted, set(to_set), set(values), roots)
    if errors:
        raise ChangeRefused(errors)
    return change


# ---------------------------------------------------------------------- folders
class FsError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def allowed_roots(settings: Settings) -> list[Path]:
    """The folders the config API may show and accept, as real paths: settings.fs_roots plus the model folders
    in use. A folder that does not exist is left out, and so is one that lies inside another root."""
    roots: list[Path] = []
    for folder in (*settings.fs_roots, settings.models_dir, settings.checkpoint_dir, settings.lora_dir):
        try:
            real = folder.resolve()
            if real.is_dir() and real not in roots:
                roots.append(real)
        except OSError:
            continue
    return sorted(root for root in roots if not any(root != other and root.is_relative_to(other) for other in roots))


def _real_path(path: str | Path) -> Path:
    """An absolute path -> its real location ('..' and links followed). FsError 400 for anything else."""
    text = str(path)
    if not text.strip() or "\0" in text or not Path(text).is_absolute():
        raise FsError(400, "the path must be absolute")
    try:
        return Path(text).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise FsError(400, f"not a usable path: {exc}") from None


def _inside_roots(real: Path, roots: list[Path]) -> bool:
    return any(real == root or real.is_relative_to(root) for root in roots)


def _readable_folder(path: str | Path, roots: list[Path] | None) -> Path:
    """The real path of a folder that exists and can be listed. roots=None skips the roots check."""
    real = _real_path(path)
    # The roots first: outside them the answer is the same whether the folder exists or not.
    if roots is not None and not _inside_roots(real, roots):
        raise FsError(403, f"{real} is outside the folders the server may show; add a folder to {ENV_PREFIX}FS_ROOTS to allow it")
    if not real.is_dir():
        raise FsError(404, f"no folder {real}")
    try:
        with os.scandir(real):
            pass
    except OSError as exc:
        raise FsError(403, f"the folder {real} cannot be read: {exc.strerror or exc}") from None
    return real


def check_directory(path: Path, roots: list[Path] | None) -> str | None:
    """What is wrong with a folder given as a setting, or None: it must be absolute, exist, be readable and,
    when roots are given, lie inside them."""
    try:
        _readable_folder(path, roots)
    except FsError as exc:
        return str(exc)
    return None


def list_directory(path: str | None, roots: list[Path]) -> dict:
    """The answer of GET /fs: the subfolders of `path`, or the roots themselves when no path is given.
    Only folders are listed, and only those whose real location is inside the roots."""
    shown_roots = [str(root) for root in roots]
    if path is None or not path.strip():
        entries = [{"name": str(root), "path": str(root)} for root in roots]
        return {"path": None, "parent": None, "dirs": entries, "model_files": 0, "truncated": False, "roots": shown_roots}
    real = _readable_folder(path, roots)
    folders: list[dict] = []
    model_files = 0
    with os.scandir(real) as entries:
        for entry in entries:
            try:
                if entry.is_dir():
                    target = Path(entry.path).resolve()
                    if _inside_roots(target, roots):   # a link that leads out of the roots is not shown
                        folders.append({"name": entry.name, "path": str(target)})
                elif entry.name.lower().endswith(MODEL_SUFFIXES):
                    model_files += 1
            except OSError:
                continue   # no access to this one entry
    folders.sort(key=lambda item: item["name"].casefold())
    parent = real.parent
    return {
        "path": str(real),
        "parent": str(parent) if parent != real and _inside_roots(parent, roots) else None,
        "dirs": folders[:MAX_FOLDERS],
        "model_files": model_files,
        "truncated": len(folders) > MAX_FOLDERS,
        "roots": shown_roots,
    }
