"""Fast tests (no GPU, no models) of the outpaint / remove_background request path:
engine-side validation, request schema, API validation and the response schema."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image
from pydantic import ValidationError

import my_diffusion.background as bg
from my_diffusion.engine import Krea2Engine, normalize_outpaint
from my_diffusion.config import Settings
from my_diffusion.imaging import Margins
from my_diffusion.schemas import GenerateRequest, GenerateResponse, OutpaintMargins

PIC = Image.new("RGB", (64, 48), (200, 30, 30))


# ---- normalize_outpaint ----------------------------------------------------------------------------------------

def test_normalize_off():
    assert normalize_outpaint(None, [], "scene") is None
    assert normalize_outpaint((0, 0, 0, 0), [], "subjects") is None   # all zeros = off, whatever else is wrong


def test_normalize_ok():
    assert normalize_outpaint((0, 0, 0, 25), [PIC], "scene") == Margins(0, 0, 0, 25)


@pytest.mark.parametrize("bad", [(-1, 0, 0, 10), (0, 0, float("nan"), 10), (0, 0, 0, float("inf"))])
def test_normalize_rejects_bad_margins(bad):
    with pytest.raises(ValueError, match="margin"):
        normalize_outpaint(bad, [PIC], "scene")


def test_normalize_needs_references():
    with pytest.raises(ValueError, match="needs references"):
        normalize_outpaint((0, 0, 0, 25), [], "scene")


def test_normalize_needs_scene_mode():
    with pytest.raises(ValueError, match="scene"):
        normalize_outpaint((0, 0, 0, 25), [PIC], "subjects")


# ---- Krea2Engine._generate: errors surface before any GPU work ----------------------------------------------------

@pytest.fixture
def engine():
    return Krea2Engine(Settings())   # not loaded: _generate must fail in validation, before touching a model


def _call(engine, **kw):
    args = dict(prompt="p", negative_prompt="", resolution="480", aspect_ratio="16:9", seed=1, steps=1,
                guidance_scale=1.0, shift=None, loras=[], enhance=False, guidance_interval=(0.0, 1.0))
    args.update(kw)
    return engine._generate(**args)


def test_engine_outpaint_without_references(engine):
    with pytest.raises(ValueError, match="needs references"):
        _call(engine, outpaint=(0, 0, 0, 25))


def test_engine_outpaint_in_subjects_mode(engine):
    with pytest.raises(ValueError, match="scene"):
        _call(engine, references=[PIC], reference_mode="subjects", outpaint=(0, 0, 0, 25))


def test_engine_outpaint_negative(engine):
    with pytest.raises(ValueError, match="margin"):
        _call(engine, references=[PIC], outpaint=(0, 0, -5, 25))


def test_engine_bad_remove_background(engine):
    with pytest.raises(ValueError, match="remove_background"):
        _call(engine, references=[PIC], remove_background="all")


def test_engine_missing_background_model(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(bg, "model_path", lambda: tmp_path / "nope.onnx")
    with pytest.raises(FileNotFoundError, match="u2net"):
        _call(engine, references=[PIC, PIC], remove_background="subjects")


# ---- schema -----------------------------------------------------------------------------------------------------

def test_schema_defaults():
    r = GenerateRequest(prompt="x")
    assert r.outpaint is None and r.remove_background == "none"


def test_schema_outpaint_limits():
    assert GenerateRequest(prompt="x", outpaint={"right": 25}).outpaint == OutpaintMargins(right=25)
    for bad in ({"right": -1}, {"left": 301}):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="x", outpaint=bad)


def test_schema_remove_background_pattern():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", remove_background="all")


def test_response_echoes_fields():
    base = dict(image_path="a", width=1, height=1, seed=1, prompt="p", enhanced_prompt=None, steps=1, guidance_scale=1.0,
                guidance_interval=[0, 1], shift=1.0, checkpoint="c", matmul="exact", loras=[], timings={})
    r = GenerateResponse(**base, outpaint={"right": 25.0}, remove_background="subjects")
    assert r.outpaint.right == 25.0 and r.remove_background == "subjects"
    assert GenerateResponse(**base).outpaint is None


# ---- API validation (400 comes before the "models are loading" 503) --------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from my_diffusion.api import app
    return TestClient(app)   # no `with`: the lifespan (model loading) is not started


def _b64() -> str:
    buf = io.BytesIO()
    PIC.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_api_outpaint_without_references(client):
    r = client.post("/generate", json={"prompt": "x", "outpaint": {"right": 25}})
    assert r.status_code == 400 and "references" in r.json()["detail"]


def test_api_outpaint_subjects_mode(client):
    r = client.post("/generate", json={"prompt": "x", "references": [_b64()], "reference_mode": "subjects", "outpaint": {"right": 25}})
    assert r.status_code == 400 and "scene" in r.json()["detail"]


def test_api_outpaint_negative_is_422(client):
    r = client.post("/generate", json={"prompt": "x", "references": [_b64()], "outpaint": {"right": -1}})
    assert r.status_code == 422


def test_api_zero_outpaint_passes_validation(client):
    # all zeros = off: gets past validation and stops at "models are loading" (503), not at 400
    r = client.post("/generate", json={"prompt": "x", "outpaint": {}})
    assert r.status_code == 503
