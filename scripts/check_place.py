"""Spike: placing a shop product into a photo of a room (inpaint + reference), through the running HTTP API.

    uv run python scripts/check_place.py make                 # generate the test rooms and products
    uv run python scripts/check_place.py place [pair...] [variant=A,B,...] [seeds=1,2]
    uv run python scripts/check_place.py sheet                # contact sheets of what is there

Why through the API: the server already holds the models on the GPU, so the script loads nothing and the spike runs
next to the user's server. The script only does the pixel work on the CPU (crop, paste, stitch, u2net background cut)
and sends the generations as ordinary jobs (owner "place-spike", visible in the gallery).

Data lives in outputs/place/: rooms/<name>.png, products/<name>.png, and boxes.json - where each product goes:
    {"pairs": {"<pair>": {"room": "living", "product": "armchair", "box": [x0, y0, x1, y1], "on": "floor"}}}
box is in fractions of the room picture (0..1). Real photos can be dropped in the same folders and listed there.

Variants (what the spike compares; letters are used in file names):
    A  full frame, as the API works today: the room shrunk to the 720 frame, mask = box, strength 1
    B  crop and stitch: a context window around the box is generated at 720 and pasted back into the full-size room
    C  as B, the box in the init picture filled with flat grey (image 1 no longer shows the empty spot)
    D  as B, the cut-out product pasted into the box (collage), strength 0.85: the model harmonises it
    E  as D, strength 0.65
    F  as D, the generic prompt ("the object from image 2"), to see what the product description adds
    G  as D, with a rough shadow drawn under the pasted product, strength 0.75
    H  as G, strength 0.85
    I  as D with a faint ambient-occlusion patch over the whole footprint, strength 0.8, mask grown 22% downwards
    J  as I, the product cut by a flood fill of its white background (not u2net) and its colours bent to the room's light
    K  as J, strength 0.9
    L  as J, the generic prompt: does the product description (a vision caption in the service) matter?
    M  as J without the reference: a plain inpaint of the collage (no edit LoRA, no image 2) - is the reference needed?
Numbers per run: seconds, mad_out (mean |difference| 0..255 outside the box against the room: 0 = untouched)."""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from my_diffusion.background import subject_mask
from my_diffusion.config import PROJECT_ROOT
from my_diffusion.resolutions import nearest_aspect_ratio, resolve_dimensions_from_size

API = "http://127.0.0.1:8000"
HEADERS = {"X-User-Id": "place-spike"}
DATA = PROJECT_ROOT / "outputs" / "place"
ROOMS, PRODUCTS, RUNS = DATA / "rooms", DATA / "products", DATA / "runs"
TIER = "720"

ROOM_PROMPTS = {
    "living": "Smartphone photo of a real Scandinavian living room in daylight: light oak floor, white walls, a grey "
              "sofa against the left wall, a large window on the right with sheer curtains, an empty floor area in the "
              "middle and an empty white wall in the back. Natural soft light, realistic, slightly wide angle, eye level.",
    "bedroom": "Smartphone photo of a real cosy bedroom: a double bed with beige linen on the right, a wooden "
               "nightstand, a light grey carpet, an empty corner on the left next to a window, a bare light wall above "
               "the bed. Afternoon daylight, realistic interior photo, eye level.",
    "evening": "Interior photo of a real living room in the evening: dark green walls, a herringbone parquet floor, "
               "a brown leather sofa, warm light from a table lamp, an empty floor area in front of the sofa on the "
               "right. Warm, dim, realistic interior photo, slight noise.",
}
PRODUCT_PROMPTS = {
    "armchair": "E-commerce product photo of a green velvet armchair with thin brass legs and a curved back, "
                "three-quarter view, isolated on a pure white background, studio light, sharp.",
    "lamp": "E-commerce product photo of a tall black metal floor lamp with a large woven rattan shade, full height, "
            "isolated on a pure white background, studio light, sharp.",
    "plant": "E-commerce product photo of a large monstera plant in a white ribbed ceramic pot, isolated on a pure "
             "white background, studio light, sharp.",
    "table": "E-commerce product photo of a small round side table, light oak top and three thin black metal legs, "
             "three-quarter view, isolated on a pure white background, studio light, sharp.",
    "painting": "E-commerce product photo of a large framed abstract painting, terracotta and beige shapes, thin "
                "black frame, front view, isolated on a pure white background.",
}
# Product descriptions as a vision model would caption them (the spike writes them by hand; the service would ask Qwen).
DESCRIPTIONS = {
    "armchair": "green velvet armchair with thin brass legs",
    "lamp": "black floor lamp with a large woven rattan shade",
    "plant": "monstera plant in a white ribbed ceramic pot",
    "table": "round side table with a light oak top and thin black legs",
    "painting": "framed abstract painting in terracotta and beige",
}
PLACEMENT = {
    "floor": "It stands on the floor, with a natural contact shadow on the floor.",
    "wall": "It hangs on the wall, with a faint natural shadow on the wall.",
    "table": "It stands on the surface below it, with a natural contact shadow.",
}
PROMPT = ("Place the {desc} from image 2 into image 1, inside the marked area. {placement} Keep its exact shape, "
          "proportions, colour, material and details from image 2. Match the perspective, scale and lighting of the "
          "room. A realistic photo; everything else in image 1 stays unchanged.")
GENERIC = "Place the object from image 2 into image 1. Make it look natural and realistic."

VARIANTS = {
    "A": dict(stitch=False, fill="orig", strength=1.0, prompt="desc"),
    "B": dict(stitch=True, fill="orig", strength=1.0, prompt="desc"),
    "C": dict(stitch=True, fill="grey", strength=1.0, prompt="desc"),
    "D": dict(stitch=True, fill="paste", strength=0.85, prompt="desc"),
    "E": dict(stitch=True, fill="paste", strength=0.65, prompt="desc"),
    "F": dict(stitch=True, fill="paste", strength=0.85, prompt="generic"),
    "G": dict(stitch=True, fill="paste+shadow", strength=0.75, prompt="desc"),
    "H": dict(stitch=True, fill="paste+shadow", strength=0.85, prompt="desc"),
    "I": dict(stitch=True, fill="paste+ao", strength=0.8, prompt="desc", down=0.22),
    "J": dict(stitch=True, fill="paste+ao", strength=0.8, prompt="desc", down=0.22, cut="flood", harmonize=0.7),
    "K": dict(stitch=True, fill="paste+ao", strength=0.9, prompt="desc", down=0.22, cut="flood", harmonize=0.7),
    "L": dict(stitch=True, fill="paste+ao", strength=0.8, prompt="generic", down=0.22, cut="flood", harmonize=0.7),
    "M": dict(stitch=True, fill="paste+ao", strength=0.8, prompt="noref", down=0.22, cut="flood", harmonize=0.7,
              noref=True),
}
# Edit mode, no mask: the whole window is generated from noise with the pictures as reference tokens, so the model
# decides light, shadows and reflections itself (the inpaint above skips the noisy steps where that is decided and forces
# everything outside the mask back to the original at every step). Only the area around the product is stitched back.
VARIANTS.update({
    "N": dict(stitch=True, fill="paste+ao", mode="edit", image1="collage", product_ref=True, prompt="harmonize",
              down=0.22, cut="flood", harmonize=0.7),
    "O": dict(stitch=True, fill="marker", mode="edit", image1="marker", product_ref=True, prompt="marker",
              down=0.22, cut="flood"),
    "P": dict(stitch=True, fill="paste+ao", mode="edit", image1="collage", product_ref=False, prompt="harmonize1",
              down=0.22, cut="flood", harmonize=0.7),
})
STANDS = {"floor": "stands on the floor", "wall": "hangs on the wall", "table": "stands on the surface"}
HARMONIZE = ("Image 1 is a photo of a room with the {desc} from image 2 roughly pasted in. Turn it into a real "
             "photograph: the {desc} {stands} at exactly the same place and size, with natural contact shadows, lit by "
             "the light of this room, with its colours and reflections. Keep its exact design from image 2. Keep the rest "
             "of the room exactly unchanged.")
HARMONIZE1 = ("This is a photo of a room with a {desc} roughly pasted in. Turn it into a real photograph: the {desc} "
              "{stands} at exactly the same place and size, with natural contact shadows, lit by the light of this room, "
              "with its colours and reflections. Keep its design. Keep the rest of the room exactly unchanged.")
MARKER = ("Place the {desc} from image 2 inside the red rectangle in image 1 so that it fills the rectangle; it {stands}. "
          "Remove the red rectangle. Natural contact shadows and the light of this room. Keep its exact design from "
          "image 2. Keep the rest of the room exactly unchanged.")

# Without a reference the prompt cannot point at image 2: the plain inpaint only sees the collage.
NOREF = ("A realistic photo of a room with a {desc}. {placement} Natural lighting and shadows that match the room, "
         "photorealistic.")


def arg(name: str, default=None):
    return next((a[len(name) + 1:] for a in sys.argv[1:] if a.startswith(name + "=")), default)


def report(**row) -> None:
    print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)


# ------------------------------------------------------------------ API
def b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_job(client: httpx.Client, body: dict) -> tuple[Image.Image, dict]:
    """Submits a job, waits for it and returns the picture and the job."""
    r = client.post(f"{API}/generate", json=body)
    if r.status_code != 202:
        raise RuntimeError(f"generate: {r.status_code} {r.text[:400]}")
    job_id = r.json()["job_id"]
    while True:
        job = client.get(f"{API}/result/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(1.0)
    if job["status"] != "done":
        raise RuntimeError(f"job {job_id} {job['status']}: {job.get('error')}")
    image = Image.open(io.BytesIO(client.get(f"{API}{job['image_url']}").content)).convert("RGB")
    return image, job


# ------------------------------------------------------------------ make
def make(client: httpx.Client) -> None:
    ROOMS.mkdir(parents=True, exist_ok=True)
    PRODUCTS.mkdir(parents=True, exist_ok=True)
    for name, prompt in ROOM_PROMPTS.items():
        path = ROOMS / f"{name}.png"
        if not path.exists():
            # 1440 at 3:2 is about 2350x1568: bigger than the 720 generation frame, like a phone photo
            image, job = run_job(client, {"prompt": prompt, "resolution": "1440", "aspect_ratio": "3:2", "seed": 7})
            image.save(path)
            report(stage="make", file=str(path), size=image.size, seconds=round(job["finished_at"] - job["started_at"], 1))
    for name, prompt in PRODUCT_PROMPTS.items():
        path = PRODUCTS / f"{name}.png"
        if not path.exists():
            ratio = "3:4" if name in ("lamp", "plant") else "1:1"
            image, job = run_job(client, {"prompt": prompt, "resolution": "720", "aspect_ratio": ratio, "seed": 7})
            image.save(path)
            report(stage="make", file=str(path), size=image.size, seconds=round(job["finished_at"] - job["started_at"], 1))


# ------------------------------------------------------------------ pixel work
def white_background_mask(product: Image.Image) -> Image.Image | None:
    """The product's mask for a shop photo on a white background, or None when the background is not white.

    Why not u2net: it finds the *salient* part, so it cut the painting's paint strokes out of its canvas and frame and
    left a white halo around the plant's leaves. On a studio photo the background is simply the near-white area that
    touches the border, so a flood fill from the border takes exactly it and keeps everything the product encloses."""
    rgb = product.convert("RGB")
    a = np.asarray(rgb, dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if (border.min(axis=1) > 225).mean() < 0.8:
        return None
    filled = rgb.copy()
    sentinel = (255, 0, 255)
    w, h = rgb.size
    step = max(1, min(w, h) // 40)
    seeds = [(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)] + \
            [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)]
    for seed in seeds:
        if min(filled.getpixel(seed)) > 225:
            ImageDraw.floodfill(filled, seed, sentinel, thresh=40)
    f = np.asarray(filled)
    background = (f[..., 0] == 255) & (f[..., 1] == 0) & (f[..., 2] == 255)
    mask = Image.fromarray(np.where(background, 0, 255).astype(np.uint8), "L")
    # Shrink by a pixel and soften: the antialiased edge pixels are half white and would leave a light fringe.
    return mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))


def cut_product(product: Image.Image, method: str = "u2net") -> tuple[Image.Image, Image.Image]:
    """The product cropped to its silhouette with a small margin, and its mask ("L").
    method: "u2net" - the salient-object model; "flood" - the white background flood fill, u2net if it is not white."""
    mask = white_background_mask(product) if method == "flood" else None
    if mask is None:
        mask = subject_mask(product).point(lambda v: 255 if v > 127 else 0)
    box = mask.point(lambda v: 255 if v > 127 else 0).getbbox() or (0, 0, *product.size)
    # A small margin, but none at the bottom: the product must stand exactly on the box's bottom edge.
    pad = round(0.03 * max(product.size))
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(product.width, box[2] + pad), box[3])
    return product.crop(box), mask.crop(box)


def to_px(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    return round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)


def grow_mask(box: tuple[int, int, int, int], size: tuple[int, int], on: str, down: float | None = None) -> Image.Image:
    """The box, grown a little on every side and more downwards for floor objects (room for the contact shadow)."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    down = down if down is not None and on == "floor" else (0.12 if on == "floor" else 0.05)
    g = (max(0, x0 - 0.06 * bw), max(0, y0 - 0.04 * bh), min(size[0], x1 + 0.06 * bw), min(size[1], y1 + down * bh))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle([round(v) for v in g], fill=255)
    return mask


def context_window(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """A window around the box that the 720 frame is generated for: about 2.2x the box, at least 45% of the short
    side, shaped like the room picture (the model sees enough of the room for perspective and light), kept inside it."""
    w, h = size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    aspect = w / h
    side_h = max((y1 - y0) * 2.2, (x1 - x0) * 2.2 / aspect, 0.45 * min(w, h))
    side_h = min(side_h, h, w / aspect)
    side_w = side_h * aspect
    left = min(max(0, cx - side_w / 2), w - side_w)
    top = min(max(0, cy - side_h / 2), h - side_h)
    return round(left), round(top), round(left + side_w), round(top + side_h)


def harmonize(cut: Image.Image, room: Image.Image, window: tuple[int, int, int, int], amount: float = 0.7) -> Image.Image:
    """Bends the studio-lit product towards the room's light before it is pasted: the colour cast (grey-world: the
    window's mean colour against its own grey) and the brightness (the window's mean luminance against a bright,
    evenly lit studio). At strength < 1 the model keeps the pasted colours, so a cold, bright product stays cold and
    bright in a warm dim room unless it is corrected here. amount 0..1 - how far to go."""
    local = np.asarray(room.crop(window), dtype=np.float32).reshape(-1, 3).mean(axis=0)
    cast = local / local.mean()
    level = float(np.clip(local.mean() / 150.0, 0.45, 1.05))
    gain = 1.0 + amount * (cast * level - 1.0)
    out = np.asarray(cut, dtype=np.float32) * gain
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def paste_product(room: Image.Image, cut: Image.Image, cut_mask: Image.Image, box: tuple[int, int, int, int],
                  shadow: str | None = None) -> Image.Image:
    """The collage: the product fitted into the box (its own proportions), standing on the box's bottom edge.

    shadow - a rough shadow drawn under the product before it is pasted, so that the model gets a grounded object to
    refine instead of one floating over the floor (at strength < 1 it barely invents a shadow by itself):
    "floor" - a soft dark ellipse along the bottom edge; "wall" - the silhouette offset down-right and blurred."""
    x0, y0, x1, y1 = box
    scale = min((x1 - x0) / cut.width, (y1 - y0) / cut.height)
    size = (max(1, round(cut.width * scale)), max(1, round(cut.height * scale)))
    left, top = x0 + ((x1 - x0) - size[0]) // 2, y1 - size[1]
    sil = cut_mask.resize(size, Image.LANCZOS)
    out = room.copy()
    if shadow:
        dark = Image.new("L", room.size, 0)
        if shadow == "ao":
            # Ambient occlusion over the whole footprint: in a 3/4 view the feet touch the floor at different
            # depths, so the soft dark area spans the bottom ~15% of the product, faint and wide, not a line.
            rx, ry = 0.45 * size[0], 0.09 * size[1]
            cx, cy = left + size[0] / 2, y1 - 0.07 * size[1]
            ImageDraw.Draw(dark).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=100)
            blur = 0.5 * ry
        elif shadow == "floor":
            rx, ry = 0.52 * size[0], max(4, 0.07 * size[1])
            cx, cy = left + size[0] / 2, y1 - 0.25 * ry
            ImageDraw.Draw(dark).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=150)
            blur = 0.35 * ry + 0.02 * size[0]
        else:
            off = round(0.03 * max(size))
            dark.paste(sil.point(lambda v: 110 if v > 127 else 0), (left + off, top + off))
            blur = 0.025 * max(size)
        dark = dark.filter(ImageFilter.GaussianBlur(blur))
        out = Image.composite(Image.new("RGB", room.size, (0, 0, 0)), out, dark)   # darkening by up to ~60%
    out.paste(cut.resize(size, Image.LANCZOS), (left, top), sil)
    return out


def mad_outside(a: Image.Image, b: Image.Image, mask: Image.Image) -> float:
    keep = np.asarray(mask) < 128
    diff = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean(axis=2)
    return float(diff[keep].mean())


# ------------------------------------------------------------------ place
def place(client: httpx.Client, pair_name: str, pair: dict, variant: str, seed: int) -> None:
    v = VARIANTS[variant]
    room = Image.open(ROOMS / f"{pair['room']}.png").convert("RGB")
    room = ImageOps.exif_transpose(room)
    product = Image.open(PRODUCTS / f"{pair['product']}.png").convert("RGB")
    on = pair.get("on", "floor")
    box = to_px(pair["box"], room.size)
    mask = grow_mask(box, room.size, on, v.get("down"))
    cut, cut_mask = cut_product(product, v.get("cut", "u2net"))
    reference = Image.new("RGB", cut.size, (255, 255, 255))
    reference.paste(cut, (0, 0), cut_mask)       # the product on white, cropped to itself
    if v.get("harmonize"):
        # Only the pasted copy is corrected; the reference keeps the true colours of the product.
        cut = harmonize(cut, room, context_window(box, room.size), v["harmonize"])

    init = room
    if v["fill"] == "grey":
        init = room.copy()
        init.paste((128, 128, 128), box)
    elif v["fill"] == "paste":
        init = paste_product(room, cut, cut_mask, box)
    elif v["fill"] == "paste+shadow":
        init = paste_product(room, cut, cut_mask, box, shadow="wall" if on == "wall" else "floor")
    elif v["fill"] == "paste+ao":
        init = paste_product(room, cut, cut_mask, box, shadow="wall" if on == "wall" else "ao")
    elif v["fill"] == "marker":
        init = room.copy()
        ImageDraw.Draw(init).rectangle(box, outline=(255, 0, 0), width=max(3, round(0.004 * max(room.size))))
    RUNS.mkdir(parents=True, exist_ok=True)
    init.crop(context_window(box, room.size)).save(RUNS / f"{pair_name}_{variant}_init.png")   # what the model got

    window = context_window(box, room.size) if v["stitch"] else (0, 0, *room.size)
    init_w, mask_w = init.crop(window), mask.crop(window)
    # The frame the server will generate for this window ("auto": the window's own proportions at the 720 budget)
    dims = resolve_dimensions_from_size(TIER, init_w.width, init_w.height)
    desc = DESCRIPTIONS.get(pair["product"], "object")
    prompt = {"desc": PROMPT, "generic": GENERIC, "noref": NOREF, "harmonize": HARMONIZE, "harmonize1": HARMONIZE1,
              "marker": MARKER}[v["prompt"]].format(desc=desc, placement=PLACEMENT[on], stands=STANDS[on])
    if v.get("mode") == "edit":
        refs = [b64(init_w)] + ([b64(reference)] if v["product_ref"] else [])
        body = {"prompt": prompt, "resolution": TIER, "seed": seed, "aspect_ratio": "auto", "references": refs}
        # The model regenerates the whole window, so the stitched area is wider than the inpaint mask: the shadows and
        # reflections it adds around the product must come along.
        mask_w = mask_w.filter(ImageFilter.MaxFilter(2 * round(0.04 * max(mask_w.size)) + 1))
    else:
        body = {
            "prompt": prompt, "resolution": TIER, "seed": seed,
            "init_image": b64(init_w), "mask": b64(mask_w), "strength": v["strength"], "feather": 2,
        }
    if v.get("mode") == "edit":
        pass   # body is complete
    elif v.get("noref"):
        # "auto" needs a reference, so the nearest preset; the test rooms are exactly 3:2, so the frame is the same
        body["aspect_ratio"] = nearest_aspect_ratio(init_w.width, init_w.height)
    else:
        body.update(aspect_ratio="auto", references=[b64(reference)])
    image, job = run_job(client, body)

    # Stitch: the generated window back to its size, pasted into the full-size room through the feathered mask.
    gen = image.resize(init_w.size, Image.LANCZOS)
    soft = mask_w.filter(ImageFilter.GaussianBlur(max(2, round(0.006 * max(room.size)))))
    out = room.copy()
    out.paste(Image.composite(gen, room.crop(window), soft), window[:2])
    if not v["stitch"]:
        # Variant A as the API works today: the whole picture is the generated frame, upscaled back to the room size.
        out = image.resize(room.size, Image.LANCZOS)

    RUNS.mkdir(parents=True, exist_ok=True)
    stem = f"{pair_name}_{variant}_s{seed}"
    out.save(RUNS / f"{stem}.png")
    # A close-up for judging the product: the grown box with some context, from the full-size result.
    x0, y0, x1, y1 = mask.getbbox()
    m = round(0.25 * max(x1 - x0, y1 - y0))
    close = (max(0, x0 - m), max(0, y0 - m), min(room.width, x1 + m), min(room.height, y1 + m))
    out.crop(close).save(RUNS / f"{stem}_close.png")
    gen.save(RUNS / f"{stem}_window.png")   # the raw generated window, before the stitch
    # drift: how much the model changed the room outside the stitched area (big = the seam may show, or the scene moved)
    drift = mad_outside(gen, room.crop(window), mask_w)
    report(drift=round(drift, 2), stage="place_drift", pair=pair_name, variant=variant)
    report(stage="place", pair=pair_name, variant=variant, seed=seed, frame=[dims.width, dims.height],
           window=list(window), box=list(box), seconds=round(job["finished_at"] - job["started_at"], 1),
           tokens=job["result"].get("tokens"), streamed=job["result"].get("streamed_blocks"),
           mad_out=round(mad_outside(out, room, mask), 2), file=str(RUNS / f"{stem}.png"))


# ------------------------------------------------------------------ sheet
def sheet() -> None:
    """One sheet per pair: the product, then the close-ups of every variant and seed that exist."""
    pairs = json.loads((DATA / "boxes.json").read_text())["pairs"]
    for name, pair in pairs.items():
        closes = sorted(RUNS.glob(f"{name}_*_close.png"))
        if not closes:
            continue
        tiles = [("product", Image.open(PRODUCTS / f"{pair['product']}.png"))] + \
                [(p.stem.removeprefix(name + "_").removesuffix("_close"), Image.open(p)) for p in closes]
        th = 420
        tiles = [(label, im.resize((round(im.width * th / im.height), th), Image.LANCZOS)) for label, im in tiles]
        width = sum(im.width for _, im in tiles) + 8 * (len(tiles) - 1)
        canvas = Image.new("RGB", (width, th + 28), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, im in tiles:
            canvas.paste(im, (x, 28))
            draw.text((x + 4, 6), label, fill=(0, 0, 0))
            x += im.width + 8
        canvas.save(DATA / f"sheet_{name}.png")
        print(f"sheet {DATA / f'sheet_{name}.png'}")


def main() -> None:
    stages = [a for a in sys.argv[1:] if "=" not in a]
    if not stages:
        raise SystemExit(__doc__)
    with httpx.Client(headers=HEADERS, timeout=60) as client:
        if stages[0] == "make":
            make(client)
        elif stages[0] == "place":
            pairs = json.loads((DATA / "boxes.json").read_text())["pairs"]
            names = stages[1:] or list(pairs)
            variants = (arg("variant") or "A,B,C,D,E").split(",")
            seeds = [int(s) for s in (arg("seeds") or "1").split(",")]
            for name in names:
                for variant in variants:
                    for seed in seeds:
                        try:
                            place(client, name, pairs[name], variant, seed)
                        except Exception as exc:   # one failed run must not stop the matrix
                            report(stage="place", pair=name, variant=variant, seed=seed, error=str(exc)[:300])
        elif stages[0] == "sheet":
            sheet()


if __name__ == "__main__":
    main()
