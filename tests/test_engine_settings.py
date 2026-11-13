"""Krea2Engine.apply_settings: what a change of settings makes the engine do (nothing, set the matmul mode or the
attention kernel again, reload the checkpoint, reload everything), and that a change that fails is undone.
The loading itself is replaced by stubs that record the calls, so no models and no GPU are needed."""

from __future__ import annotations

import pytest

from my_diffusion.config import Settings, inside_folder
from my_diffusion.engine import Krea2Engine


@pytest.fixture
def models(tmp_path):
    for folder in ("checkpoints", "other", "loras", "vae", "text_encoders/qwen", "text_encoders/qwen2"):
        (tmp_path / folder).mkdir(parents=True)
    for file in ("checkpoints/a.safetensors", "checkpoints/b.safetensors", "other/a.safetensors", "vae/vae.safetensors", "vae/vae2.safetensors"):
        (tmp_path / file).touch()
    return tmp_path


def base_settings(models, **changes) -> Settings:
    values = {"models_dir": models, "checkpoint": "a.safetensors", "vae": "vae.safetensors", "text_encoder": "qwen", "device": "cpu"}
    return Settings(**{**values, **changes})


class StubEngine(Krea2Engine):
    """The real apply_settings / reload_plan on top of loading that only records what was asked."""

    def __init__(self, settings: Settings, fail: set[str] = frozenset()):
        super().__init__(settings)
        self.calls: list = []
        self.fail = set(fail)   # names of the steps that raise
        self._models = False

    @property
    def loaded(self) -> bool:
        return self._models

    def _step(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    def load(self, checkpoint=None) -> None:
        self._step("load", checkpoint)
        self.load_checkpoint(checkpoint or self.settings.checkpoint)

    def unload(self) -> None:
        self._step("unload")
        self._models = False
        self.checkpoint_file = None

    def load_checkpoint(self, name: str) -> None:
        path = inside_folder(self.settings.checkpoint_dir, name)   # FileNotFoundError like the real one, before anything is dropped
        self._models = False
        self._step("load_checkpoint", name)
        self._models = True
        self.checkpoint_file, self._checkpoint_request, self.checkpoint_name = path, name, path.name
        self._attention = (self.settings.attention, self.settings.attention_sage2_min_tokens)

    def set_matmul(self, mode: str) -> None:
        self._step("set_matmul", mode)
        self.matmul = mode

    def set_attention(self) -> None:
        self._step("set_attention", self.settings.attention)
        self._attention = (self.settings.attention, self.settings.attention_sage2_min_tokens)


@pytest.fixture
def engine(models) -> StubEngine:
    stub = StubEngine(base_settings(models))
    stub.load()
    stub.calls.clear()
    return stub


def changed(engine: StubEngine, **changes) -> Settings:
    return engine.settings.model_copy(update=changes)


# ---- what a change costs ----------------------------------------------------------------------------------------

def test_plain_values_only_swap_the_settings(engine):
    new = changed(engine, edit_lora_weight=0.5, max_references=1, lora_streaming=False, loras_dir=engine.settings.models_dir / "other")
    assert engine.apply_settings(new) == "none"
    assert engine.settings is new and engine.calls == []


def test_matmul_and_attention_are_set_again_without_a_reload(engine):
    assert engine.apply_settings(changed(engine, matmul="exact")) == "none"
    assert engine.calls == [("set_matmul", "exact")]
    engine.calls.clear()
    assert engine.apply_settings(changed(engine, attention="default")) == "none"
    assert engine.apply_settings(changed(engine, attention_sage2_min_tokens=0)) == "none"
    assert engine.calls == [("set_attention", "default"), ("set_attention", "default")]


def test_another_checkpoint_or_folder_reloads_the_checkpoint_only(engine, models):
    assert engine.apply_settings(changed(engine, checkpoint="b.safetensors", matmul="exact")) == "checkpoint"
    assert engine.calls == [("load_checkpoint", "b.safetensors")] and engine.matmul == "exact"
    engine.calls.clear()
    assert engine.apply_settings(changed(engine, checkpoint="a.safetensors", checkpoints_dir=models / "other")) == "checkpoint"
    assert engine.checkpoint_file == (models / "other" / "a.safetensors").resolve()


def test_a_folder_that_holds_the_very_same_file_reloads_nothing(engine, models):
    same = changed(engine, checkpoints_dir=models / "checkpoints")   # was None: checkpoints/ under models_dir
    assert engine.reload_plan(same) == "none"
    assert engine.apply_settings(same) == "none" and engine.calls == []


@pytest.mark.parametrize("change", [{"vae": "vae2.safetensors"}, {"text_encoder": "qwen2"}])
def test_vae_and_text_encoder_reload_everything(engine, change):
    assert engine.apply_settings(changed(engine, **change)) == "all"
    assert [call[0] for call in engine.calls] == ["unload", "load", "load_checkpoint"]


def test_pinning_reloads_everything_on_a_gpu_only(models):
    on_gpu = StubEngine(base_settings(models, device="cuda"))
    on_gpu.load()
    assert on_gpu.reload_plan(on_gpu.settings.model_copy(update={"pin_memory": False})) == "all"
    on_cpu = StubEngine(base_settings(models))   # nothing is pinned on the CPU
    on_cpu.load()
    assert on_cpu.reload_plan(on_cpu.settings.model_copy(update={"pin_memory": False})) == "none"


def test_a_full_reload_keeps_a_checkpoint_switched_by_hand(engine):
    engine.load_checkpoint("b.safetensors")   # POST /checkpoints/load: the setting still names a.safetensors
    engine.calls.clear()
    assert engine.apply_settings(changed(engine, matmul="exact")) == "none"       # the hand-picked checkpoint stays
    assert engine.checkpoint_name == "b.safetensors"
    assert engine.apply_settings(changed(engine, vae="vae2.safetensors")) == "all"
    assert engine.checkpoint_name == "b.safetensors"
    assert engine.apply_settings(changed(engine, checkpoint="a.safetensors", vae="vae.safetensors")) == "all"
    assert engine.checkpoint_name == "b.safetensors"   # the checkpoint settings did not change: still a.safetensors there


def test_an_engine_without_models_retries_the_load_on_any_change(models):
    stub = StubEngine(base_settings(models, checkpoint="missing.safetensors"))
    with pytest.raises(FileNotFoundError):
        stub.load()   # the load at start failed
    assert not stub.loaded
    assert stub.apply_settings(stub.settings.model_copy(update={"checkpoint": "a.safetensors"})) == "all"
    assert stub.loaded and stub.checkpoint_name == "a.safetensors"


# ---- a change that fails is undone ------------------------------------------------------------------------------

def test_failed_attention_puts_the_matmul_mode_back(engine):
    old = engine.settings
    engine.fail = {"set_attention"}
    with pytest.raises(RuntimeError, match="set_attention failed"):
        engine.apply_settings(changed(engine, matmul="exact", attention="sage2"))
    assert engine.settings is old and engine.matmul == "fp8_all"
    assert engine.calls == [("set_matmul", "exact"), ("set_attention", "sage2"), ("set_matmul", "fp8_all")]


def test_a_refused_checkpoint_leaves_the_loaded_one_alone(engine):
    old = engine.settings
    with pytest.raises(FileNotFoundError):
        engine.apply_settings(changed(engine, checkpoint="missing.safetensors", matmul="exact"))
    assert engine.settings is old and engine.loaded and engine.checkpoint_name == "a.safetensors"
    assert engine.matmul == "fp8_all" and ("load_checkpoint", "a.safetensors") not in engine.calls   # no needless reload


def test_a_failed_full_reload_loads_the_previous_models_again(engine):
    old = engine.settings

    def fail_once(name, *args, real=engine._step):
        if name == "load" and not any(call[0] == "load" for call in engine.calls):
            engine.calls.append((name, *args))
            raise RuntimeError("the new VAE is broken")
        real(name, *args)

    engine._step = fail_once
    with pytest.raises(RuntimeError, match="the new VAE is broken"):
        engine.apply_settings(changed(engine, vae="vae2.safetensors"))
    assert engine.settings is old and engine.loaded and engine.checkpoint_name == "a.safetensors"
    assert [call[0] for call in engine.calls] == ["unload", "load", "unload", "load", "load_checkpoint"]


def test_when_the_way_back_fails_too_the_engine_is_left_without_models(engine):
    engine.fail = {"load"}
    with pytest.raises(RuntimeError, match="load failed"):
        engine.apply_settings(changed(engine, vae="vae2.safetensors"))
    assert not engine.loaded and engine.settings.vae == "vae.safetensors"
