"""Picture preparation for image-to-image, inpaint and reference editing: pure helpers, no model code.

A user-supplied picture rarely has the size and proportions we generate at. Before the VAE can
turn it into a latent, it has to be decoded from bytes, brought to exactly the chosen frame, and
(for a mask) shrunk to the latent grid. Keeping this separate from the model code makes it easy
to test with plain pictures."""

from __future__ import annotations

import hashlib
import io
import math
from typing import NamedTuple

import torch
from PIL import Image, UnidentifiedImageError

from .resolutions import resolve_dimensions_from_size

# One Krea 2 token covers 16x16 image pixels: the VAE shrinks 8x and the network packs 2x2 latent cells.
PATCH_PIXELS = 16


class Margins(NamedTuple):
    """How much to grow the source picture on each side, in percent of the source's own side
    (top/bottom relative to its height, left/right to its width) - the Wan2GP convention."""

    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    def validate(self) -> "Margins":
        """Raise ValueError unless every margin is a finite number >= 0; return self for chaining.
        (A NamedTuple cannot validate in its constructor, so outpaint_frame calls this.)"""
        for name, value in zip(self._fields, self):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"margin {name} must be a finite number >= 0, got {value!r}")
        return self


def _quantize_axis(before: float, total: int, inner_raw: float, quantum: int) -> tuple[int, int]:
    """Snap one axis of the frame to the token grid. Returns (offset, size) of the source.

    Why: the source is placed at a token offset (margin // 16), so both margins must be multiples
    of 16. Each margin is rounded on its own; since `total` is a multiple of 16 too, the source
    size comes out as a multiple of 16 as well. If rounding would leave less than one token for
    the source (tiny source, huge margins), the larger margin is cut back so the source never
    overflows the frame."""
    after = total - before - inner_raw
    before_q = max(0, round(before / quantum) * quantum)
    after_q = max(0, round(after / quantum) * quantum)
    overflow = before_q + after_q + quantum - total
    if overflow > 0:
        cut = math.ceil(overflow / quantum) * quantum
        if after_q >= before_q:
            after_q = max(0, after_q - cut)
        else:
            before_q = max(0, before_q - cut)
    return before_q, total - before_q - after_q


def outpaint_frame(
    src_w: int, src_h: int, margins: Margins, tier: str
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Plan an outpainting canvas: returns (frame_w, frame_h, src_box).

    The "full area" is the source grown by the margins. The frame is that area at the tier's pixel
    budget and its exact aspect ratio (sides multiples of 16), so outpainting costs the same as
    any other generation of the tier. src_box = (left, top, width, height) is where the source
    goes inside the frame: offsets and size are multiples of 16, the source is at least one token
    and always inside the frame. Whatever is outside the box is the area to be painted."""
    margins = Margins(*margins).validate()
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"source size must be positive, got {src_w}x{src_h}")
    top, bottom, left, right = margins
    full_w = src_w * (100 + left + right) / 100
    full_h = src_h * (100 + top + bottom) / 100
    frame_w, frame_h = resolve_dimensions_from_size(tier, full_w, full_h)

    # Margins scale with the frame in proportion to the full area; the source gets the rest.
    box_left, box_w = _quantize_axis(
        frame_w * left / (100 + left + right), frame_w, frame_w * 100 / (100 + left + right), PATCH_PIXELS
    )
    box_top, box_h = _quantize_axis(
        frame_h * top / (100 + top + bottom), frame_h, frame_h * 100 / (100 + top + bottom), PATCH_PIXELS
    )
    return frame_w, frame_h, (box_left, box_top, box_w, box_h)


def load_rgb(data: bytes) -> Image.Image:
    """Decode PNG/JPEG (or any format Pillow knows) bytes into an RGB picture.

    Transparency and palettes are flattened by `convert("RGB")`, because the VAE expects exactly
    three channels. Anything that is not an image raises ValueError, so the caller can turn it
    into a clean "bad request" instead of a stack trace."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"the data is not a valid image (PNG or JPEG expected): {e}") from e
    return image.convert("RGB")


def cover_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale the picture so that it fully covers width x height (aspect ratio kept), then cut the
    centre out, so the result is exactly width x height.

    Why: the result then has the same pixel budget as a text-to-image run at that resolution, so
    time and memory stay predictable (decision D2). Nothing is stretched; the cost is that the
    edges of a picture with a different aspect ratio are cropped away. Use the same function for
    the mask, so that the picture and its mask always share one frame."""
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {width}x{height}")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(width, math.ceil(src_w * scale)), max(height, math.ceil(src_h * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    return resized.crop((left, top, left + width, top + height))


def cap_longest_side(image: Image.Image, max_side: int = 768) -> Image.Image:
    """Shrink the picture (LANCZOS, aspect ratio kept) so that its longer side is at most max_side.
    A picture that already fits is returned as is.

    Why: this is the size cap Wan2GP applies before the Qwen3-VL vision tower. The number of vision
    tokens grows with the area (one token per 32x32 pixels), so the cap keeps a reference at a few
    hundred tokens (at most 576) instead of thousands."""
    if max_side <= 0:
        raise ValueError(f"max_side must be positive, got {max_side}")
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def fit_reference(image: Image.Image, width: int, height: int) -> tuple[Image.Image, tuple[int, int]]:
    """Prepare the SECOND reference picture for a width x height result (Wan2GP's `fit` rule).

    Returns (picture, (off_h, off_w)): the prepared picture and where its token grid sits inside the
    result's token grid, in 16-pixel patches. The rule, exactly as in Wan2GP's
    `_encode_image_to_latents(fit=True)`:
    - scale = min(height / h, width / w), the largest scale at which the picture fits inside the frame;
    - if after that scale both sides fill at least 92% of the frame (the aspect ratios almost match),
      the picture "covers" the frame instead: it is scaled up to cover it, the centre is cut out with
      the frame's aspect ratio, and that is resized to exactly width x height (offset 0, 0);
    - otherwise it is fitted inside: each side is scaled, rounded down to a multiple of 16 (at least 16,
      at most the frame side), and the smaller picture is centred: offset = ((grid_h_frame - grid_h) // 2,
      (grid_w_frame - grid_w) // 2).
    Both branches resize with BICUBIC, as Wan2GP does.

    Why not cover_crop as for the first reference: the second reference is usually a person or an
    object with its own proportions; cropping it to the scene's frame could cut the subject off."""
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {width}x{height}")
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(height / src_h, width / src_w)
    if src_h * scale >= height * 0.92 and src_w * scale >= width * 0.92:
        scale = max(height / src_h, width / src_w)
        crop_h, crop_w = min(src_h, round(height / scale)), min(src_w, round(width / scale))
        top, left = (src_h - crop_h) // 2, (src_w - crop_w) // 2
        image = image.crop((left, top, left + crop_w, top + crop_h))
        fit_h, fit_w = height, width
    else:
        fit_h = min(max(PATCH_PIXELS, int(src_h * scale) // PATCH_PIXELS * PATCH_PIXELS), height)
        fit_w = min(max(PATCH_PIXELS, int(src_w * scale) // PATCH_PIXELS * PATCH_PIXELS), width)
    picture = image.resize((fit_w, fit_h), Image.BICUBIC)
    offset = ((height // PATCH_PIXELS - fit_h // PATCH_PIXELS) // 2, (width // PATCH_PIXELS - fit_w // PATCH_PIXELS) // 2)
    return picture, offset


def image_digest(image: Image.Image) -> str:
    """sha256 over the picture's size and its RGB pixel bytes, as a hex string.

    Used as part of the text-encoding cache key when pictures go into the prompt: the same upload
    prepared for another resolution is a different frame, so it must give a different key, while
    the same pixels always give the same key whatever file format they came from."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _gaussian_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable gaussian blur of a (1, 1, h, w) tensor; sigma = radius / 2, kernel half-width = radius.
    Edges are padded by replication, so a mask touching the border does not fade there."""
    sigma = radius / 2
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    pad = radius
    x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, 1, -1))
    x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="replicate")
    return torch.nn.functional.conv2d(x, kernel.view(1, 1, -1, 1))


def mask_to_latent(mask: Image.Image, lat_h: int, lat_w: int, feather: int = 0) -> torch.Tensor:
    """Turn a user mask into a latent-resolution blend weight (1, 1, lat_h, lat_w), float32, in 0..1.

    Convention: white (255 -> 1.0) = "regenerate this", black (0 -> 0.0) = "keep this".
    The mask is converted to grayscale and shrunk to the latent grid with box (area) averaging, so a
    latent cell that is only partly covered gets a fractional value instead of a hard 0/1 jump.
    If feather > 0, the edge is softened with a gaussian blur whose radius is `feather` latent
    pixels (one latent pixel = 8 image pixels), implemented with torch convolution. A soft edge
    hides the seam between the kept and the regenerated parts.

    The mask is NOT cropped here: it must already be in the same frame as the picture. Pass it
    through `cover_crop(mask, width, height)` first, with the same width and height as the image."""
    if lat_h <= 0 or lat_w <= 0:
        raise ValueError(f"latent size must be positive, got {lat_h}x{lat_w}")
    if feather < 0:
        raise ValueError(f"feather must be >= 0, got {feather}")
    small = mask.convert("L").resize((lat_w, lat_h), Image.BOX)
    weight = torch.frombuffer(bytearray(small.tobytes()), dtype=torch.uint8).view(1, 1, lat_h, lat_w).float() / 255.0
    if feather > 0:
        weight = _gaussian_blur(weight, feather)
    return weight.clamp(0.0, 1.0)
