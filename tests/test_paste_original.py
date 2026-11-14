"""Fast tests (no GPU) of imaging.paste_original: the pixel return of the original after the VAE decode."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from my_diffusion.imaging import paste_original

W, H = 96, 64
ORIG = Image.new("RGB", (W, H), (200, 0, 0))
GEN = Image.new("RGB", (W, H), (0, 0, 200))


def _px(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(int)


def test_kept_area_is_exact_and_margin_untouched():
    box = (32, 0, 64, 64)   # the left edge is inner, the other three lie on the frame border
    out = _px(paste_original(GEN, ORIG, box, overlap=8))
    assert (out[:, 32 + 8:] == (200, 0, 0)).all()      # deep inside: exactly the original, up to the frame border
    assert (out[:, :32] == (0, 0, 200)).all()          # margin: exactly the generated pixels
    assert (out[:, 0:32, 0] == 0).all()


def test_ramp_is_monotonic_across_inner_edge():
    out = _px(paste_original(GEN, ORIG, (32, 0, 64, 64), overlap=8))
    red = out[10, 32:32 + 9, 0]
    assert (np.diff(red) > 0).all() and 0 < red[0] < 200


def test_frame_border_edges_get_no_ramp():
    out = _px(paste_original(GEN, ORIG, (0, 0, 64, 64), overlap=8))
    assert (out[:, :56] == (200, 0, 0)).all()          # starts at the frame border with full alpha, no fade


def test_hard_edge_with_zero_overlap():
    out = _px(paste_original(GEN, ORIG, (32, 16, 32, 32), overlap=0))
    assert (out[16:48, 32:64] == (200, 0, 0)).all()
    assert (out[:16] == (0, 0, 200)).all() and (out[:, :32] == (0, 0, 200)).all()


def test_input_not_modified_and_validation():
    before = GEN.tobytes()
    paste_original(GEN, ORIG, (0, 0, 32, 32))
    assert GEN.tobytes() == before
    with pytest.raises(ValueError):
        paste_original(GEN, Image.new("RGB", (8, 8)), (0, 0, 8, 8))
    with pytest.raises(ValueError):
        paste_original(GEN, ORIG, (80, 0, 32, 32))
    with pytest.raises(ValueError):
        paste_original(GEN, ORIG, (0, 0, 32, 32), overlap=-1)
