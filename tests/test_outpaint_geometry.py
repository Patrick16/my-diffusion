"""Pure geometry of outpainting: no model, no GPU."""

import itertools

import pytest

from my_diffusion.imaging import Margins, outpaint_frame
from my_diffusion.resolutions import RESOLUTION_TIERS, resolve_dimensions, resolve_dimensions_from_size


def _check_invariants(frame_w, frame_h, box):
    left, top, w, h = box
    assert frame_w % 16 == 0 and frame_h % 16 == 0
    assert all(v % 16 == 0 for v in box)
    assert w >= 16 and h >= 16
    assert left >= 0 and top >= 0
    assert left + w <= frame_w and top + h <= frame_h
    assert frame_w >= w and frame_h >= h


def test_size_from_picture_keeps_budget_and_aspect():
    w, h = resolve_dimensions_from_size("720", 1000, 500)
    assert (w, h) == (1296, 656) or abs(w / h - 2) < 0.05
    assert w % 16 == 0 and h % 16 == 0
    assert abs(w * h - 921600) / 921600 < 0.05


def test_size_from_picture_matches_preset():
    assert resolve_dimensions_from_size("720", 1600, 900) == resolve_dimensions("720", "16:9")


def test_size_from_picture_errors():
    with pytest.raises(ValueError):
        resolve_dimensions_from_size("999", 10, 10)
    with pytest.raises(ValueError):
        resolve_dimensions_from_size("720", 0, 10)


def test_zero_margins_no_change():
    fw, fh, box = outpaint_frame(1000, 600, Margins(), "720")
    assert (fw, fh) == tuple(resolve_dimensions_from_size("720", 1000, 600))
    assert box == (0, 0, fw, fh)


def test_symmetric_margins_centre_source():
    fw, fh, box = outpaint_frame(800, 800, Margins(25, 25, 25, 25), "720")
    left, top, w, h = box
    assert left == fw - left - w and top == fh - top - h
    assert left > 0 and top > 0


def test_one_sided_margin():
    fw, fh, (left, top, w, h) = outpaint_frame(800, 800, Margins(right=50), "720")
    assert left == 0 and top == 0 and h == fh
    assert w < fw
    assert abs(w / fw - 1 / 1.5) < 0.03
    fw, fh, (left, top, w, h) = outpaint_frame(800, 800, Margins(top=100), "720")
    assert left == 0 and w == fw and top + h == fh
    assert abs(h / fh - 0.5) < 0.03


def test_quantization_to_16():
    for pct in (3, 7, 13, 33, 61):
        fw, fh, box = outpaint_frame(777, 555, Margins(pct, pct / 2, pct * 2, 0), "480")
        _check_invariants(fw, fh, box)


def test_tiny_source_and_huge_margins():
    for src in ((1, 1), (3, 5), (16, 16), (1, 1000)):
        for m in (Margins(), Margins(1000, 1000, 1000, 1000), Margins(left=5000), Margins(bottom=0.1)):
            fw, fh, box = outpaint_frame(*src, m, "480")
            _check_invariants(fw, fh, box)


def test_grid_of_cases_invariants():
    values = (0, 1, 10, 50, 300)
    for tier in RESOLUTION_TIERS:
        for t, b, l, r in itertools.product(values, repeat=4):
            fw, fh, box = outpaint_frame(640, 480, Margins(t, b, l, r), tier)
            _check_invariants(fw, fh, box)


def test_negative_margins_rejected():
    with pytest.raises(ValueError):
        outpaint_frame(100, 100, Margins(left=-1), "720")
    with pytest.raises(ValueError):
        Margins(top=float("nan")).validate()


def test_bad_source_size_rejected():
    with pytest.raises(ValueError):
        outpaint_frame(0, 100, Margins(), "720")


def test_exact_sizes_never_exceed_the_tier_limit():
    """The engine rejects frames above the tier's largest preset frame; exact-ratio sizes must fit it
    (rounding both sides to the nearest 16 used to overshoot for ~40% of random sources)."""
    import random

    from my_diffusion.imaging import Margins, outpaint_frame
    from my_diffusion.resolutions import max_pixels, resolve_dimensions_from_size

    rng = random.Random(7)
    for _ in range(3000):
        tier = rng.choice(["480", "720"])
        w, h = rng.randint(1, 4000), rng.randint(1, 4000)
        size = resolve_dimensions_from_size(tier, w, h)
        assert size.width % 16 == 0 and size.height % 16 == 0
        assert size.width * size.height <= max_pixels(tier)
        margins = Margins(*(rng.choice([0, 0, rng.uniform(0, 300)]) for _ in range(4)))
        fw, fh, _ = outpaint_frame(w, h, margins, tier)
        assert fw * fh <= max_pixels(tier)


def test_known_overshoot_cases_now_fit():
    from my_diffusion.resolutions import max_pixels, resolve_dimensions_from_size

    size = resolve_dimensions_from_size("720", 1000, 333)   # was 1664x560 = 931840 > 921600
    assert size.width * size.height <= max_pixels("720")
