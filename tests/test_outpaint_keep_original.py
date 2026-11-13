"""Fast tests (no GPU, no models) of outpainting with keep_original: the blend weight's geometry,
the canvas the kept area is held to, and the validation in the engine, the API and the CLI."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from my_diffusion.config import Settings
from my_diffusion.engine import KEEP_BLEND, Krea2Engine
from my_diffusion.imaging import Margins, outpaint_frame, outpaint_keep_weight, place_in_frame
from my_diffusion.schemas import GenerateRequest, GenerateResponse

PIC = Image.new("RGB", (64, 48), (200, 30, 30))


# ---- outpaint_keep_weight ------------------------------------------------------------------------------------------

def _cells(box):
    return tuple(v // 8 for v in box)


def test_weight_size_and_values_right_margin():
    # frame 160x64, box on the left 0..112 (14 cells), margin on the right
    w = outpaint_keep_weight(160, 64, (0, 0, 112, 64), blend=2)
    assert w.shape == (1, 1, 8, 20) and w.dtype == torch.float32
    w = w[0, 0]
    assert torch.all(w[:, 14:] == 1)                    # the margin is generated
    assert torch.all(w[:, :12] == 0)                    # kept exactly: the box shrunk by 2 at its inner (right) edge
    assert torch.allclose(w[:, 13], torch.full((8,), 2 / 3))   # the ramp: edge cell 2/3, then 1/3, then 0
    assert torch.allclose(w[:, 12], torch.full((8,), 1 / 3))


@pytest.mark.parametrize("box", [(32, 16, 96, 48), (0, 16, 96, 48), (32, 0, 128, 64), (16, 8, 32, 24)])
@pytest.mark.parametrize("blend", [0, 1, 2, 3])
def test_weight_zero_exactly_inside_shrunk_box_one_outside(box, blend):
    fw, fh = 160, 80
    w = outpaint_keep_weight(fw, fh, box, blend)[0, 0]
    x0, y0, bw, bh = _cells(box)
    x1, y1 = x0 + bw, y0 + bh
    lat_w, lat_h = fw // 8, fh // 8
    # shrink only at inner edges (not on the frame border)
    sx0 = x0 + (blend if x0 > 0 else 0)
    sy0 = y0 + (blend if y0 > 0 else 0)
    sx1 = x1 - (blend if x1 < lat_w else 0)
    sy1 = y1 - (blend if y1 < lat_h else 0)
    inside = torch.zeros(lat_h, lat_w, dtype=torch.bool)
    inside[y0:y1, x0:x1] = True
    shrunk = torch.zeros_like(inside)
    shrunk[max(sy0, 0):max(sy1, 0), max(sx0, 0):max(sx1, 0)] = True
    assert torch.all(w[~inside] == 1)
    assert torch.all(w[shrunk] == 0)
    band = inside & ~shrunk
    assert torch.all((w[band] > 0) & (w[band] < 1))
    assert w.min() >= 0 and w.max() <= 1


def test_weight_box_touching_frame_border_has_no_band_there():
    # box covers the full height: no ramp at top or bottom
    w = outpaint_keep_weight(128, 64, (32, 0, 96, 64), blend=2)[0, 0]
    assert torch.all(w[:, :4] == 1)
    assert torch.all(w[:, 6:] == 0)   # left edge is inner (ramp at cells 4, 5); right edge is the border


def test_weight_on_a_real_outpaint_frame():
    fw, fh, box = outpaint_frame(736, 560, Margins(10, 20, 15, 25), "480")
    w = outpaint_keep_weight(fw, fh, box, KEEP_BLEND)
    assert w.shape == (1, 1, fh // 8, fw // 8)
    x0, y0, bw, bh = _cells(box)
    b = KEEP_BLEND
    assert torch.all(w[0, 0, y0 + b:y0 + bh - b, x0 + b:x0 + bw - b] == 0)
    assert w[0, 0].sum() > 0


@pytest.mark.parametrize("args", [
    (100, 64, (0, 0, 64, 64), 2),      # frame not on the latent grid
    (128, 64, (4, 0, 64, 64), 2),      # box not on the latent grid
    (128, 64, (96, 0, 64, 64), 2),     # box overflows the frame
    (128, 64, (0, 0, 64, 64), -1),     # negative blend
])
def test_weight_rejects_bad_input(args):
    with pytest.raises(ValueError):
        outpaint_keep_weight(*args)


# ---- place_in_frame ------------------------------------------------------------------------------------------------

def test_place_in_frame_keeps_pixels_and_repeats_edges():
    pic = Image.fromarray(np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3))
    canvas = place_in_frame(pic, 64, 48, (16, 16, 32, 16))
    a = np.asarray(canvas)
    assert canvas.size == (64, 48)
    assert np.array_equal(a[16:32, 16:48], np.asarray(pic))
    assert np.array_equal(a[0, 16:48], np.asarray(pic)[0])        # top margin repeats the top row
    assert np.array_equal(a[20, 48:], np.repeat(np.asarray(pic)[4, -1:], 16, axis=0))


def test_place_in_frame_needs_resized_picture():
    with pytest.raises(ValueError):
        place_in_frame(PIC, 128, 64, (0, 0, 32, 32))


# ---- validation ----------------------------------------------------------------------------------------------------

def _call(**kw):
    args = dict(prompt="p", negative_prompt="", resolution="480", aspect_ratio="16:9", seed=1, steps=1,
                guidance_scale=1.0, shift=None, loras=[], enhance=False, guidance_interval=(0.0, 1.0))
    args.update(kw)
    return Krea2Engine(Settings())._generate(**args)   # not loaded: must fail in validation


@pytest.mark.parametrize("kw", [
    dict(keep_original=True),
    dict(keep_original=True, references=[PIC]),
    dict(keep_original=True, references=[PIC], outpaint=(0, 0, 0, 0)),
])
def test_engine_keep_original_needs_outpaint(kw):
    with pytest.raises(ValueError, match="keep_original"):
        _call(**kw)


def test_schema_default_and_echo():
    assert GenerateRequest(prompt="x").keep_original is False
    base = dict(image_path="a", width=1, height=1, seed=1, prompt="p", enhanced_prompt=None, steps=1, guidance_scale=1.0,
                guidance_interval=[0, 1], shift=1.0, checkpoint="c", matmul="exact", loras=[], timings={})
    assert GenerateResponse(**base).keep_original is False
    assert GenerateResponse(**base, keep_original=True).keep_original is True


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from my_diffusion.api import app
    return TestClient(app)


def _b64() -> str:
    buf = io.BytesIO()
    PIC.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("extra", [{}, {"outpaint": {}}, {"outpaint": {"right": 0}}])
def test_api_keep_original_without_outpaint_is_400(client, extra):
    r = client.post("/generate", json={"prompt": "x", "references": [_b64()], "keep_original": True, **extra})
    assert r.status_code == 400 and "keep_original" in r.json()["detail"]


def test_api_keep_original_with_outpaint_passes_validation(client):
    r = client.post("/generate", json={"prompt": "x", "references": [_b64()], "outpaint": {"right": 25}, "keep_original": True})
    assert r.status_code == 503   # past validation, stops at "models are loading"


@pytest.mark.parametrize("extra", [
    ["--reference", "x.png", "--keep-original"],
    ["--reference", "x.png", "--outpaint", "0", "0", "0", "0", "--keep-original"],
])
def test_cli_keep_original_needs_outpaint(monkeypatch, capsys, extra):
    from my_diffusion import cli
    monkeypatch.setattr("sys.argv", ["my-diffusion-generate", "prompt", *extra])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2 and "--keep-original needs --outpaint" in capsys.readouterr().err
