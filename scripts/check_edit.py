"""GPU check of reference editing (docs: the references recipe, check list H).

    uv run python scripts/check_edit.py [stages...] [probe=PATH]

Stages (default: all, in this order):
    vision  - the vision tower: tensor count, grid and image_pad count, context length (H2)
    h3      - our encode vs the probe's Vision.encode on the raw fox upload (H3; needs probe=PATH to fable_probe.py)
    recipe  - 480 (848x480), one reference: the fox on a bridge (probe p4c) and the cat "keep the subject" (probe p5b);
              the fox run is repeated to check the encoding cache and determinism (H4, H6)
    tiers   - the rank rule at the 480 tier with frames larger than 848x480: 640x640 and 736x560, one reference (H5)
    two     - two references at 480 (rank 128 by the rule) (H5, H7)
    r720    - 720 (1280x720), one reference (rank 128 by the rule) (H5, H6: another size re-encodes)

Fixed: turbo krea2_turbo_mxfp8, fp8_all, seed 42, 8 steps, a memory cap of 0.97 of the GPU (as in the probe), so that
an overrun is an OOM error and not a silent spill into shared memory. Pictures: outputs/check_edit_*.png.
mad = mean absolute pixel difference 0..255; thumb = the same on 32x18 thumbnails (composition)."""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image

import my_diffusion.engine as engine_module
from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine
from my_diffusion.imaging import cap_longest_side, cover_crop

OUT = PROJECT_ROOT / "outputs"
FOX = OUT / "check_ckpt_krea2_turbo_mxfp8.png"     # 736x560
CAT = OUT / "check_enhance_cat.png"                # 832x1104
FOX_PROMPT = "Change the background to a wooden bridge over a river, keep the fox unchanged."
CAT_PROMPT = "Change the background to a wooden bridge over a river, keep the subject unchanged."
TWO_PROMPT = "Replace the fox in image 1 with the cat from image 2, keep the snowy background unchanged."
THUMB = (32, 18)
GIB = 2**30


def mad(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean())


def tmad(a: Image.Image, b: Image.Image) -> float:
    return mad(a.resize(THUMB, Image.BOX), b.resize(THUMB, Image.BOX))


def arg(name: str, default=None):
    return next((a[len(name) + 1:] for a in sys.argv[1:] if a.startswith(name + "=")), default)


# The peak while the LoRA set is attached: the engine resets the peak counter before sampling, so we
# measure apply_loras separately by wrapping the function the engine calls.
LORA_APPLY: dict[str, float] = {}
_apply = engine_module.apply_loras


def _measured_apply(model, loras, device, *a, **kw):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _apply(model, loras, device, *a, **kw)
    torch.cuda.synchronize()
    LORA_APPLY.update(
        seconds=round(time.time() - t0, 2), before_gb=round(before / GIB, 2),
        after_gb=round(torch.cuda.memory_allocated() / GIB, 2), peak_gb=round(torch.cuda.max_memory_allocated() / GIB, 2),
    )


engine_module.apply_loras = _measured_apply


def check_vision(engine: Krea2Engine, fox: Image.Image) -> None:
    qwen = engine.qwen
    engine.gpu.use("qwen")
    frame = cover_crop(fox, 736, 560)   # the fox's prepared frame at 480 4:3 is the fox itself
    t0 = time.time()
    cond = qwen.encode(FOX_PROMPT, images=[frame])
    torch.cuda.synchronize()
    seconds = time.time() - t0
    grid = qwen.image_processor(images=[frame], return_tensors="pt")["image_grid_thw"]
    pads = int(grid[0].prod()) // qwen.model.config.vision_config.spatial_merge_size ** 2
    text = len(qwen.tokenizer(FOX_PROMPT, add_special_tokens=False).input_ids)
    row = dict(
        check="vision", tensors=len(qwen._vision_cold), grid=grid.tolist(), image_pad=pads, prompt_tokens=text,
        context=cond.shape[1], expected_context=pads + 2 + text + 5, shape=list(cond.shape), dtype=str(cond.dtype),
        seconds_incl_tower_load=round(seconds, 2),
        ok=len(qwen._vision_cold) == 315 and grid.tolist() == [[1, 36, 46]] and pads == 414 and cond.shape[1] == 437,
    )
    print("RESULT " + json.dumps(row), flush=True)


def check_h3(engine: Krea2Engine, fox: Image.Image, probe_path: str) -> None:
    spec = importlib.util.spec_from_file_location("fable_probe", probe_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    engine.gpu.use("qwen")
    ours = engine.qwen.encode(FOX_PROMPT, images=[fox]).float()
    vision = probe.Vision(engine.qwen)
    theirs = vision.encode(FOX_PROMPT, [fox]).float()
    del vision
    torch.cuda.empty_cache()
    same_shape = ours.shape == theirs.shape
    diff = (ours - theirs).abs() if same_shape else None
    row = dict(check="h3_vs_probe", ours=list(ours.shape), probe=list(theirs.shape),
               max_abs_diff=float(diff.max()) if same_shape else None, mean_abs_diff=float(diff.mean()) if same_shape else None,
               max_abs_value=float(theirs.abs().max()), ok=same_shape and float(diff.max()) <= 1e-2)
    print("RESULT " + json.dumps(row), flush=True)
    # Hypothesis: the probe built its tower with _from_config(...).to(bfloat16), which also casts the rotary table
    # inv_freq to bf16; the project keeps it in float32, as transformers' from_pretrained does. Re-encode with a bf16 table.
    qwen = engine.qwen
    saved = dict(qwen._vision_built_in)
    qwen._vision_built_in = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in saved.items()}
    qwen._vision_to(qwen.device)
    bf16_table = qwen.encode(FOX_PROMPT, images=[fox]).float()
    qwen._vision_built_in = saved
    qwen._vision_to(qwen.device)
    d2 = (bf16_table - theirs).abs()
    row = dict(check="h3_vs_probe_with_bf16_inv_freq", buffers={k: str(v.dtype) for k, v in saved.items()},
               max_abs_diff=float(d2.max()), mean_abs_diff=float(d2.mean()))
    print("RESULT " + json.dumps(row), flush=True)


def run(engine: Krea2Engine, name: str, prompt: str, refs: list[Image.Image], resolution: str, aspect: str,
        compare: dict[str, Image.Image] | None = None) -> dict:
    """One edit; returns the result row. An OOM is caught and reported with the peak at the failure."""
    LORA_APPLY.clear()
    qwen_before = engine.gpu.active
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()   # num_alloc_retries: how often the allocator hit the cap and flushed its cache
    row = dict(name=name, prompt=prompt, resolution=resolution, aspect=aspect, references=len(refs))
    try:
        r = engine.generate(prompt, seed=42, resolution=resolution, aspect_ratio=aspect, references=refs)
    except torch.OutOfMemoryError as e:
        row.update(oom=True, error=str(e).splitlines()[0][:200],
                   peak_alloc_gb=round(torch.cuda.max_memory_allocated() / GIB, 2),
                   peak_reserved_gb=round(torch.cuda.max_memory_reserved() / GIB, 2), lora_apply=dict(LORA_APPLY))
        print("RESULT " + json.dumps(row), flush=True)
        engine._applied_loras = []   # the attached set is in an unknown state after a failure
        torch.cuda.empty_cache()
        return row
    out = OUT / f"check_edit_{name}.png"
    out.unlink(missing_ok=True)
    shutil.move(r["image_path"], out)
    img = Image.open(out).convert("RGB")
    w, h = img.size
    row.update(
        oom=False, png=str(out), size=[w, h], steps=r["steps"], shift=r["shift"], mode=r["mode"],
        edit_lora_rank=r["edit_lora_rank"], loras=r["loras"], timings=r["timings"], qwen_was_active=qwen_before == "qwen",
        peak_alloc_gb=round(torch.cuda.max_memory_allocated() / GIB, 2),
        peak_reserved_gb=round(torch.cuda.max_memory_reserved() / GIB, 2), lora_apply=dict(LORA_APPLY),
        alloc_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0),
        mad_vs_ref1=round(mad(img, cover_crop(refs[0], w, h)), 2), thumb_vs_ref1=round(tmad(img, cover_crop(refs[0], w, h)), 2),
    )
    for label, other in (compare or {}).items():
        other = other if other.size == img.size else cover_crop(other, w, h)
        row[f"mad_vs_{label}"] = round(mad(img, other), 2)
        row[f"thumb_vs_{label}"] = round(tmad(img, other), 2)
    print("RESULT " + json.dumps(row), flush=True)
    row["image"] = img
    return row


def sheet(rows: list[dict]) -> None:
    pics = [r for r in rows if "image" in r]
    if not pics:
        return
    tw, th = 424, 240
    canvas = Image.new("RGB", (tw * 3, th * ((len(pics) + 2) // 3)), "white")
    for i, r in enumerate(pics):
        thumb = r["image"].copy()
        thumb.thumbnail((tw, th), Image.LANCZOS)
        canvas.paste(thumb, ((i % 3) * tw, (i // 3) * th))
    canvas.save(OUT / "check_edit_sheet.png")
    print("sheet:", OUT / "check_edit_sheet.png", "order:", [r["name"] for r in pics], flush=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    stages = [a for a in sys.argv[1:] if "=" not in a] or ["vision", "h3", "recipe", "tiers", "two", "r720"]
    torch.cuda.set_per_process_memory_fraction(float(arg("frac", "0.97")))
    settings = Settings(checkpoint="krea2_turbo_mxfp8.safetensors", matmul="fp8_all")
    engine = Krea2Engine(settings)
    t0 = time.time()
    engine.load()
    print(f"engine loaded in {time.time() - t0:.0f}s: {engine.checkpoint_name}, kind {engine.kind}, matmul {engine.matmul}", flush=True)
    fox = Image.open(FOX).convert("RGB")
    cat = Image.open(CAT).convert("RGB")
    print(f"fox {fox.size}, cat {cat.size}", flush=True)

    rows: list[dict] = []
    if "vision" in stages:
        check_vision(engine, fox)
    if "h3" in stages and arg("probe"):
        check_h3(engine, fox, arg("probe"))
    if "recipe" in stages:
        p4c = Image.open(OUT / "fable_probe_p4c_vision_ref_lora_480full.png").convert("RGB")
        p5b = Image.open(OUT / "fable_probe_p5b_cat_vision_ref_lora_480full.png").convert("RGB")
        first = run(engine, "p4c_fox_bridge_480", FOX_PROMPT, [fox], "480", "16:9", {"probe_p4c": p4c, "cat": cat})
        rows.append(first)
        again = run(engine, "p4c_fox_bridge_480_repeat", FOX_PROMPT, [fox], "480", "16:9")
        if "image" in first and "image" in again:
            print(f"RESULT {json.dumps(dict(check='repeat_identical', max_abs_diff=int(np.abs(np.asarray(first['image'], dtype=np.int16) - np.asarray(again['image'], dtype=np.int16)).max())))}", flush=True)
        rows.append(run(engine, "p5b_cat_bridge_480", CAT_PROMPT, [cat], "480", "16:9", {"probe_p5b": p5b, "cat": cat, "fox": fox}))
        # cache with the SAME prompt: the same fox frame is a hit, another picture (the cat) is a miss
        for label, picture in (("same_prompt_same_ref", fox), ("same_prompt_other_ref", cat)):
            frame = cap_longest_side(cover_crop(picture, 848, 480), settings.vision_max_side)
            size = len(engine._image_cache)
            t0 = time.time()
            engine._encode_with_images(FOX_PROMPT, [frame])
            print("RESULT " + json.dumps(dict(check=f"cache_{label}", seconds=round(time.time() - t0, 3),
                                              entries_before=size, entries_after=len(engine._image_cache))), flush=True)
    if "tiers" in stages:
        rows.append(run(engine, "tier480_640x640", FOX_PROMPT, [fox], "480", "1:1", {"cat": cat}))
        rows.append(run(engine, "tier480_736x560", FOX_PROMPT, [fox], "480", "4:3", {"cat": cat}))
    if "two" in stages:
        rows.append(run(engine, "two_refs_480", TWO_PROMPT, [fox, cat], "480", "16:9", {"cat": cat}))
    if "r720" in stages:
        rows.append(run(engine, "p4c_fox_bridge_720", FOX_PROMPT, [fox], "720", "16:9", {"cat": cat}))
    sheet(rows)


if __name__ == "__main__":
    main()
