"""Background removal for reference images (u2net salient-object segmentation).

A plain-onnxruntime port of what rembg does for the u2net model (no rembg, numba, scipy or pymatting):
the picture is resized to 320x320, scaled by its maximum and normalised with the ImageNet mean/std, the
network returns a saliency map, which is min-max normalised, resized back to the picture size and used
as an alpha mask. Alpha matting is not applied. The background is then replaced by a flat colour.

The ONNX session is created lazily on the first call and runs on the CPU by default, so that the GPU
stays free for the diffusion model (u2net is small: about a second per picture)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

from my_diffusion.config import Settings

INPUT_SIZE = (320, 320)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None
_session_key: tuple[Path, tuple[str, ...]] | None = None
_lock = threading.Lock()


def model_path() -> Path:
    return Settings().rembg_model_path


def _get_session(path: Path, providers: tuple[str, ...]):
    """The shared ONNX session; created on first use (and again if the path or providers change)."""
    global _session, _session_key
    with _lock:
        if _session is not None and _session_key == (path, providers):
            return _session
        if not path.is_file():
            raise FileNotFoundError(
                f"Background removal model not found: {path}. Put u2net.onnx there "
                "(scripts/link_models.ps1 links it from the Wan2GP folder) or set MY_DIFFUSION_REMBG_MODEL."
            )
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("Background removal needs onnxruntime: run `uv sync`.") from e
        available = set(ort.get_available_providers())
        chosen = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        _session = ort.InferenceSession(str(path), providers=chosen)
        _session_key = (path, providers)
        return _session


def subject_mask(image: Image.Image, *, path: Path | None = None,
                 providers: tuple[str, ...] = ("CPUExecutionProvider",)) -> Image.Image:
    """Saliency mask of the picture as an "L" image of the same size (255 = subject)."""
    session = _get_session(path or model_path(), providers)
    rgb = image.convert("RGB")
    small = np.asarray(rgb.resize(INPUT_SIZE, Image.Resampling.LANCZOS), dtype=np.float32)
    small = small / max(float(small.max()), 1e-6)
    small = ((small - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    pred = session.run(None, {session.get_inputs()[0].name: small})[0][0, 0]
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / max(hi - lo, 1e-6)
    mask = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8), "L")
    return mask.resize(rgb.size, Image.Resampling.BICUBIC)


def remove_background(image: Image.Image, fill: tuple[int, int, int] = (255, 255, 255), *,
                      path: Path | None = None) -> Image.Image:
    """The subject of the picture on a flat `fill` background; returns an RGB image of the same size.
    path - the u2net model file; None means the default from the environment (model_path())."""
    fill = tuple(int(c) for c in fill)
    if len(fill) != 3 or not all(0 <= c <= 255 for c in fill):
        raise ValueError(f"fill must be an (R, G, B) tuple of 0..255 values, got {fill!r}")
    rgb = image.convert("RGB")
    mask = subject_mask(rgb, path=path)
    return Image.composite(rgb, Image.new("RGB", rgb.size, fill), mask)
