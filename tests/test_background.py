"""Fast tests of background.py that do not need the u2net model."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import my_diffusion.background as bg
from my_diffusion.config import Settings


@pytest.fixture(autouse=True)
def reset_session(monkeypatch):
    monkeypatch.setattr(bg, "_session", None)
    monkeypatch.setattr(bg, "_session_key", None)


def test_default_model_path():
    assert Settings().rembg_model_path.parts[-2:] == ("rembg", "u2net.onnx")


def test_missing_model_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_DIFFUSION_REMBG_MODEL", str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError, match="u2net.onnx"):
        bg.remove_background(Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("fill", [(1, 2), (1, 2, 3, 4), (0, 0, 256), (-1, 0, 0)])
def test_bad_fill_rejected_before_model_is_touched(fill):
    with pytest.raises(ValueError, match="fill"):
        bg.remove_background(Image.new("RGB", (8, 8)), fill)


def test_composite_uses_fill_colour(monkeypatch):
    # Left half is "subject", right half is "background"; a stub replaces the network.
    mask = Image.fromarray(np.tile(np.array([255] * 4 + [0] * 4, dtype=np.uint8), (8, 1)), "L")
    monkeypatch.setattr(bg, "subject_mask", lambda image, **kw: mask)
    src = Image.new("RGB", (8, 8), (10, 20, 30))
    out = bg.remove_background(src.convert("RGBA"), fill=(200, 100, 50))
    assert out.mode == "RGB" and out.size == (8, 8)
    assert out.getpixel((0, 0)) == (10, 20, 30)
    assert out.getpixel((7, 7)) == (200, 100, 50)
