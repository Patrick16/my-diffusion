"""Fast tests (no GPU, no models) of init_image / mask combined with references (Wan2GP's control image +
references): the init picture becomes image 1, so the limits and bad combinations are checked up front."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from my_diffusion.config import Settings
from my_diffusion.engine import Krea2Engine

PIC = Image.new("RGB", (64, 48), (200, 30, 30))


@pytest.fixture
def engine():
    return Krea2Engine(Settings())


def _call(engine, **kw):
    args = dict(prompt="p", negative_prompt="", resolution="480", aspect_ratio="16:9", seed=1, steps=1,
                guidance_scale=1.0, shift=None, loras=[], enhance=False, guidance_interval=(0.0, 1.0))
    args.update(kw)
    return engine._generate(**args)


def test_engine_init_counts_as_image_1(engine):
    # max_references is 2: the init picture plus two references would be three pictures
    with pytest.raises(ValueError, match="counts as image 1"):
        _call(engine, init_image=PIC, references=[PIC, PIC])


def test_engine_init_with_references_needs_scene_mode(engine):
    with pytest.raises(ValueError, match="scene"):
        _call(engine, init_image=PIC, references=[PIC], reference_mode="subjects")


def test_engine_init_with_references_refuses_outpaint(engine):
    with pytest.raises(ValueError, match="outpaint cannot be combined"):
        _call(engine, init_image=PIC, references=[PIC], outpaint=(0, 0, 0, 25))


def test_engine_mask_without_init_still_refused(engine):
    with pytest.raises(ValueError, match="needs an init image"):
        _call(engine, mask=PIC, references=[PIC])


def test_engine_init_and_one_reference_gets_past_validation(engine):
    # The combination is valid: the call goes on to the heavy part and fails there (no models loaded),
    # not with a validation message about references / init_image.
    with pytest.raises(Exception) as err:
        _call(engine, init_image=PIC, references=[PIC])
    assert "combined" not in str(err.value) and "counts as image 1" not in str(err.value)


# ---- API --------------------------------------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from my_diffusion.api import app
    return TestClient(app)


def _b64() -> str:
    buf = io.BytesIO()
    PIC.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_api_init_with_too_many_references(client):
    r = client.post("/generate", json={"prompt": "x", "init_image": _b64(), "references": [_b64(), _b64()]})
    assert r.status_code == 400 and "image 1" in r.json()["detail"]


def test_api_init_with_references_subjects_mode(client):
    r = client.post("/generate", json={"prompt": "x", "init_image": _b64(), "references": [_b64()], "reference_mode": "subjects"})
    assert r.status_code == 400 and "scene" in r.json()["detail"]


def test_api_init_with_references_and_outpaint(client):
    r = client.post("/generate", json={"prompt": "x", "init_image": _b64(), "references": [_b64()], "outpaint": {"right": 25}})
    assert r.status_code == 400 and "outpaint" in r.json()["detail"]


def test_api_mask_without_init_with_references(client):
    r = client.post("/generate", json={"prompt": "x", "mask": _b64(), "references": [_b64()]})
    assert r.status_code == 400 and "mask needs init_image" in r.json()["detail"]


def test_api_init_mask_and_reference_pass_validation(client):
    # valid: gets past validation and stops at "models are loading" (503)
    r = client.post("/generate", json={"prompt": "x", "init_image": _b64(), "mask": _b64(), "references": [_b64()]})
    assert r.status_code == 503


# ---- CLI --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("extra, message", [
    (["--init-image", "a.png", "--reference", "x.png", "--reference", "y.png"], "counts as image 1"),
    (["--init-image", "a.png", "--reference", "x.png", "--reference-mode", "subjects"], "reference-mode scene"),
    (["--init-image", "a.png", "--reference", "x.png", "--outpaint", "0", "0", "0", "25"], "cannot be combined with --init-image"),
])
def test_cli_rejects_bad_combinations(monkeypatch, capsys, extra, message):
    from my_diffusion import cli
    monkeypatch.setattr("sys.argv", ["my-diffusion-generate", "prompt", *extra])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2 and message in capsys.readouterr().err
