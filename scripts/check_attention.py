"""A/B of the attention kernels on real generations: cudnn (the default) against sage2 and its hybrids.

sage2 (SageAttention 2) is an approximation, so numbers alone do not decide: this script makes the same pictures
with every variant (same prompt, seed, size), times the sampling, and saves them side by side for the eye.

    uv run --group sage2 python scripts/check_attention.py run [--cases ...] [--variants ...] [--seeds 42 43]
    uv run --group sage2 python scripts/check_attention.py summary [--faces DIR]

`run` appends one line per picture to outputs/sage2/results.jsonl and keeps the picture next to it; a picture that
is already there is skipped (--force redoes it), so the cases can be run in several goes. The first run makes the
reference portrait (outputs/sage2/ref_man.png) for the face cases.

`summary` reads those files and prints, per case and variant: sampling time against cudnn, and how far the picture
is from the cudnn one - PSNR, SSIM, sharpness ratio (variance of the Laplacian; below 1 = fine detail was lost).
It also writes the sheets outputs/ab_sage2_<case>_s<seed>.png (the variants left to right, in the printed order).
--faces DIR adds the identity check for the edit cases: DIR holds insightface's buffalo_l models (det_10g.onnx,
w600k_r50.onnx); the face of the reference and of each result is embedded by ArcFace and the cosine similarity
printed (same person: usually above 0.5), plus enlarged face crops in outputs/ab_sage2_<case>_s<seed>_face.png.

Variants:
    cudnn        the default kernel: every other variant is compared with it
    default      torch's own choice ("efficient"): as exact as cudnn, so its distance from cudnn shows how much a
                 picture moves from rounding noise alone
    sage2        sage2 in all 28 blocks on all steps
    edge2/edge3  sage2, but the first and last 2 / 3 blocks stay on cudnn
    late2        sage2, but the first 2 steps stay on cudnn (they fix the composition)
    early2       the opposite: sage2 only on the first 2 steps (shows where the difference comes from)

Fixed: turbo krea2_turbo_mxfp8, fp8_all, 8 steps. The 1440 tier and the edits above the engine's size limits are
enabled only inside this script. The process is capped at the video memory that is free at start, so that an
overrun is an OOM error and not a silent spill into shared memory (which is 8-10 times slower)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import my_diffusion.engine as engine_module
from my_diffusion import resolutions
from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine
from my_diffusion.transformer import set_attention_kernel

OUT = PROJECT_ROOT / "outputs"
WORK = OUT / "sage2"
RESULTS = WORK / "results.jsonl"
REF_MAN = WORK / "ref_man.png"
GIB = 2**30
STEPS = 8
LAYERS = 28

FOX = "a fox walking in fresh snow at golden hour, natural light, detailed fur"
PORTRAIT = ('close-up portrait photo of a 35-year-old woman with freckles and green eyes, curly red hair, wearing a knitted '
            'mustard sweater, standing in a bookshop, a small chalkboard sign behind her reads "OPEN LATE", soft window '
            'light, shallow depth of field, 85mm lens')
MAN = ("studio portrait photo of a 50-year-old man with a short grey beard, round wire glasses and a scar across the left "
       "eyebrow, wearing a dark green wool coat, looking at the camera, plain light grey background, sharp focus")
FOX_EDIT = "Change the background to a wooden bridge over a river, keep the fox unchanged."
MAN_EDIT = "Change the background to a busy night street with neon signs, keep the man unchanged."
TWO_EDIT = "The man from image 2 kneels in the snow next to the fox from image 1, keep the fox and the man's face unchanged."


@dataclass(frozen=True)
class Case:
    prompt: str
    resolution: str
    aspect: str
    refs: tuple[str, ...] = ()   # "fox" | "man"


CASES: dict[str, Case] = {
    "fox_480": Case(FOX, "480", "16:9"),
    "fox_720": Case(FOX, "720", "16:9"),
    "fox_1080": Case(FOX, "1080", "16:9"),
    "fox_1440": Case(FOX, "1440", "16:9"),
    "portrait_480": Case(PORTRAIT, "480", "3:4"),
    "portrait_720": Case(PORTRAIT, "720", "3:4"),
    "portrait_1080": Case(PORTRAIT, "1080", "3:4"),
    "edit_fox_480": Case(FOX_EDIT, "480", "16:9", ("fox",)),
    "edit_fox_720": Case(FOX_EDIT, "720", "16:9", ("fox",)),
    "edit_fox_1080": Case(FOX_EDIT, "1080", "16:9", ("fox",)),
    "face_480": Case(MAN_EDIT, "480", "1:1", ("man",)),
    "face_720": Case(MAN_EDIT, "720", "1:1", ("man",)),
    "face_1080": Case(MAN_EDIT, "1080", "1:1", ("man",)),
    "two_480": Case(TWO_EDIT, "480", "16:9", ("fox", "man")),
    "two_720": Case(TWO_EDIT, "720", "16:9", ("fox", "man")),
}


@dataclass(frozen=True)
class Variant:
    kernel: str                                   # cudnn | default | sage2
    cudnn_blocks: frozenset[int] = frozenset()    # sage2: these blocks stay on cudnn
    sage_steps: range = field(default=range(0, STEPS))   # sage2: only on these steps (0-based), cudnn on the others


def edges(k: int) -> frozenset[int]:
    return frozenset(range(k)) | frozenset(range(LAYERS - k, LAYERS))


VARIANTS: dict[str, Variant] = {
    "cudnn": Variant("cudnn"),
    "default": Variant("default"),
    "sage2": Variant("sage2"),
    "edge2": Variant("sage2", edges(2)),
    "edge3": Variant("sage2", edges(3)),
    "late2": Variant("sage2", sage_steps=range(2, STEPS)),
    "early2": Variant("sage2", sage_steps=range(0, 2)),
}


def apply_variant(model, variant: Variant, step: int) -> None:
    """Set the kernels for the network pass number `step` (0-based)."""
    sage = variant.kernel == "sage2" and step in variant.sage_steps
    for i, block in enumerate(model.blocks):
        if variant.kernel == "default":
            kernel = "default"
        else:
            kernel = "sage2" if sage and i not in variant.cudnn_blocks else "cudnn"
        set_attention_kernel(block, kernel)


def picture_path(case: str, seed: int, variant: str) -> Path:
    return WORK / f"{case}_s{seed}_{variant}.png"


def load_results() -> dict[tuple[str, int, str], dict]:
    rows: dict[tuple[str, int, str], dict] = {}
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["case"], row["seed"], row["variant"]] = row   # a later line replaces an earlier one
    return rows


# ---------------------------------------------------------------------- run
def reference(name: str, engine: Krea2Engine) -> Image.Image:
    if name == "fox":
        path = OUT / "check_ckpt_krea2_turbo_mxfp8.png"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: make it with scripts/check_checkpoints.py (or copy it from the main checkout)")
        return Image.open(path).convert("RGB")
    if not REF_MAN.exists():
        apply_variant(engine.model, VARIANTS["cudnn"], 0)
        r = engine.generate(MAN, seed=7, resolution="720", aspect_ratio="1:1", steps=STEPS)
        Path(r["image_path"]).replace(REF_MAN)
        print(f"reference portrait: {REF_MAN}", flush=True)
    return Image.open(REF_MAN).convert("RGB")


def run_one(engine: Krea2Engine, case_name: str, variant_name: str, seed: int) -> dict:
    case, variant = CASES[case_name], VARIANTS[variant_name]
    refs = [reference(name, engine) for name in case.refs]
    marks: list[float] = []
    lengths: list[int] = []
    peak = [0.0]

    def on_progress(done: int, total: int) -> None:
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        peak[0] = torch.cuda.max_memory_allocated() / GIB   # the sampling peak: the VAE decode comes later
        if done < total:
            apply_variant(engine.model, variant, done)

    hook = engine.model.blocks[0].attn.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    apply_variant(engine.model, variant, 0)
    row = dict(case=case_name, variant=variant_name, seed=seed, resolution=case.resolution, aspect=case.aspect, references=len(refs))
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    try:
        r = engine.generate(case.prompt, seed=seed, resolution=case.resolution, aspect_ratio=case.aspect, steps=STEPS,
                            references=refs, on_progress=on_progress)
    except torch.OutOfMemoryError as e:
        row.update(oom=True, error=str(e).splitlines()[0][:200], steps_done=len(marks), tokens=lengths[0] if lengths else None,
                   peak_gb=round(torch.cuda.max_memory_allocated() / GIB, 2))
        if len(marks) > 2:   # the sampling itself may have finished and the VAE decode failed
            row["step_s"] = [round(b - a, 3) for a, b in zip(marks, marks[1:])]
        engine._applied_loras = []   # the attached set is in an unknown state after a failure
        torch.cuda.empty_cache()
        return row
    finally:
        hook.remove()
    path = picture_path(case_name, seed, variant_name)
    Path(r["image_path"]).replace(path)
    steps = [b - a for a, b in zip(marks, marks[1:])]   # steps 2..8: the first one has no start mark of its own
    row.update(
        oom=False, png=path.name, size=[r["width"], r["height"]], tokens=lengths[0], sample_s=r["timings"]["sample"],
        step_s=[round(s, 3) for s in steps], decode_s=r["timings"]["decode"], peak_gb=round(peak[0], 2),
        alloc_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0), edit_lora_rank=r["edit_lora_rank"],
    )
    return row


def run(args) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    WORK.mkdir(parents=True, exist_ok=True)
    # sizes the engine does not offer (yet): only inside this script
    resolutions.RESOLUTION_TIERS["1440"] = 1440
    engine_module.EDIT_MAX_TIER = {1: "1440", 2: "1440"}
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.97, free / total))
    print(f"video memory: {free / GIB:.1f} GB free of {total / GIB:.1f} GB; the process is capped at the free part", flush=True)

    engine = Krea2Engine(Settings(checkpoint="krea2_turbo_mxfp8.safetensors", matmul="fp8_all"))
    t0 = time.time()
    engine.load()
    print(f"engine loaded in {time.time() - t0:.0f}s", flush=True)
    done = load_results()
    with RESULTS.open("a", encoding="utf-8") as log:
        for case in args.cases:
            for variant in args.variants:
                for seed in args.seeds:
                    key = (case, seed, variant)
                    if not args.force and key in done and (done[key]["oom"] or picture_path(*key).exists()):
                        continue
                    row = run_one(engine, case, variant, seed)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if row["oom"]:
                        print(f"{case:>14} {variant:>8} s{seed}: OOM after {row['steps_done']} steps, peak {row['peak_gb']} GB, tokens {row['tokens']}", flush=True)
                        break   # the other seeds of this variant would fail the same way
                    print(f"{case:>14} {variant:>8} s{seed}: {row['tokens']} tokens, sampling {row['sample_s']:.1f}s "
                          f"(steps 2-8: {sum(row['step_s']):.2f}s), peak {row['peak_gb']} GB, retries {row['alloc_retries']}", flush=True)


# ---------------------------------------------------------------------- metrics
def gray(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32))[None, None]


def psnr(a: Image.Image, b: Image.Image) -> float:
    mse = float(((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def ssim(a: Image.Image, b: Image.Image) -> float:
    """Mean SSIM on luma, the standard 11x11 Gaussian window (sigma 1.5)."""
    x, y = gray(a), gray(b)
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    window = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]
    mx, my = F.conv2d(x, window), F.conv2d(y, window)
    vx, vy, cov = F.conv2d(x * x, window) - mx * mx, F.conv2d(y * y, window) - my * my, F.conv2d(x * y, window) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean())


def sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian: how much fine detail the picture has."""
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
    return float(F.conv2d(gray(image), kernel).var())


class Faces:
    """Face detector (SCRFD det_10g) and ArcFace embedder (w600k_r50) from insightface's buffalo_l, run by onnxruntime
    on the CPU. Only what the identity check needs: the most confident face of a picture and its embedding."""

    TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
    DET_SIZE = 640

    def __init__(self, folder: Path):
        import onnxruntime as ort

        self.det = ort.InferenceSession(str(folder / "det_10g.onnx"), providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(folder / "w600k_r50.onnx"), providers=["CPUExecutionProvider"])

    def detect(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray] | None:
        """(box x1 y1 x2 y2, 5 key points) of the most confident face in picture coordinates, or None."""
        scale = self.DET_SIZE / max(image.size)
        small = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
        canvas = Image.new("RGB", (self.DET_SIZE, self.DET_SIZE))
        canvas.paste(small, (0, 0))
        blob = ((np.asarray(canvas, dtype=np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        best = None
        for i, stride in enumerate((8, 16, 32)):
            scores, boxes, points = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
            side = self.DET_SIZE // stride
            ys, xs = np.mgrid[:side, :side]
            centers = np.repeat(np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride, 2, axis=0).astype(np.float32)   # 2 anchors per cell
            j = int(scores.argmax())
            if scores[j] >= 0.5 and (best is None or scores[j] > best[0]):
                cx, cy = centers[j]
                box = np.array([cx - boxes[j, 0], cy - boxes[j, 1], cx + boxes[j, 2], cy + boxes[j, 3]]) / scale
                best = (scores[j], box, (centers[j] + points[j].reshape(5, 2)) / scale)
        return None if best is None else (best[1], best[2])

    def embed(self, image: Image.Image, points: np.ndarray) -> np.ndarray:
        """ArcFace embedding (unit length) of the face with these key points."""
        # similarity transform points -> TEMPLATE (Umeyama), then its inverse for PIL
        ms, md = points.mean(0), self.TEMPLATE.mean(0)
        s, d = points - ms, self.TEMPLATE - md
        u, sing, vt = np.linalg.svd(d.T @ s / len(points))
        flip = np.diag([1.0, -1.0 if np.linalg.det(u) * np.linalg.det(vt) < 0 else 1.0])
        rot = u @ flip @ vt * ((sing * np.diag(flip)).sum() / s.var(0).sum())
        forward = np.eye(3)
        forward[:2, :2], forward[:2, 2] = rot, md - rot @ ms
        inverse = np.linalg.inv(forward)
        face = image.transform((112, 112), Image.AFFINE, tuple(inverse[:2].reshape(-1)), Image.BICUBIC)
        blob = ((np.asarray(face, dtype=np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        vector = self.rec.run(None, {self.rec.get_inputs()[0].name: blob})[0][0]
        return vector / np.linalg.norm(vector)


def crop_face(image: Image.Image, box: np.ndarray, side: int = 384) -> Image.Image:
    cx, cy, half = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1]) * 0.75
    return image.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half))).resize((side, side), Image.LANCZOS)


def summary(args) -> None:
    rows = load_results()
    faces = Faces(Path(args.faces)) if args.faces else None
    man = Image.open(REF_MAN).convert("RGB") if REF_MAN.exists() else None
    man_vector = None
    if faces and man is not None:
        man_vector = faces.embed(man, faces.detect(man)[1])
    cases = [c for c in CASES if any(k[0] == c for k in rows)]
    seeds = sorted({k[1] for k in rows})
    print("| case | size | tokens | variant | sampling, s | vs cudnn | peak, GB | PSNR, dB | SSIM | sharpness | face: ref / cudnn |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        variants = [v for v in VARIANTS if any(k[0] == case and k[2] == v for k in rows)]
        base_time = None
        for variant in variants:
            mine = [rows[case, s, variant] for s in seeds if (case, s, variant) in rows]
            ok = [r for r in mine if not r["oom"]]
            if not ok:
                r = mine[0]
                print(f"| {case} | | {r['tokens']} | {variant} | OOM after {r['steps_done']} steps | | {r['peak_gb']} | | | | |")
                continue
            # time: steps 2..8 scaled to 8 steps - the first step of a new size carries one-time kernel setup
            seconds = float(np.mean([sum(r["step_s"]) for r in ok])) * STEPS / (STEPS - 1)
            base_time = seconds if variant == "cudnn" else base_time
            quality: list[tuple[float, float, float]] = []
            identity: list[tuple[float, float]] = []
            for r in ok:
                base = rows.get((case, r["seed"], "cudnn"))
                if base is None or base["oom"]:
                    continue
                a, b = Image.open(WORK / r["png"]).convert("RGB"), Image.open(WORK / base["png"]).convert("RGB")
                if variant != "cudnn":
                    quality.append((psnr(a, b), ssim(a, b), sharpness(a) / sharpness(b)))
                if man_vector is not None and "man" in CASES[case].refs:
                    found, found_base = faces.detect(a), faces.detect(b)
                    if found and found_base:
                        vector = faces.embed(a, found[1])
                        identity.append((float(vector @ man_vector), float(vector @ faces.embed(b, found_base[1]))))
            q = np.mean(quality, axis=0) if quality else None
            f = np.mean(identity, axis=0) if identity else None
            first = ok[0]
            print(f"| {case} | {first['size'][0]}x{first['size'][1]} | {first['tokens']} | {variant} | {seconds:.2f} | "
                  f"{'' if base_time is None else f'{(seconds / base_time - 1) * 100:+.0f}%'} | {max(r['peak_gb'] for r in ok):.1f} | "
                  + (" | | | " if q is None else f"{q[0]:.1f} | {q[1]:.3f} | {q[2]:.2f} | ")
                  + ("" if f is None else f"{f[0]:.3f} / {f[1]:.3f}") + " |")
        # sheets: the variants side by side, and the face crops enlarged
        for seed in seeds:
            pictures = [(v, WORK / rows[case, seed, v]["png"]) for v in variants if (case, seed, v) in rows and not rows[case, seed, v]["oom"]]
            if len(pictures) < 2:
                continue
            images = [Image.open(p).convert("RGB") for _, p in pictures]
            sheet = Image.new("RGB", (sum(i.width for i in images), images[0].height))
            x = 0
            for image in images:
                sheet.paste(image, (x, 0))
                x += image.width
            sheet.save(OUT / f"ab_sage2_{case}_s{seed}.png")
            if faces and "man" in CASES[case].refs:
                crops = [crop_face(man, faces.detect(man)[0])]
                for image in images:
                    found = faces.detect(image)
                    crops.append(crop_face(image, found[0]) if found else Image.new("RGB", (384, 384)))
                strip = Image.new("RGB", (384 * len(crops), 384))
                for i, crop in enumerate(crops):
                    strip.paste(crop, (384 * i, 0))
                strip.save(OUT / f"ab_sage2_{case}_s{seed}_face.png")
        print(f"<!-- {case}: sheets outputs/ab_sage2_{case}_s<seed>.png, left to right: {', '.join(variants)} -->")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="generate the pictures and record the timings")
    r.add_argument("--cases", nargs="+", choices=list(CASES), default=["fox_480", "fox_720", "fox_1080"])
    r.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["cudnn", "sage2"])
    r.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    r.add_argument("--force", action="store_true", help="redo pictures that are already there")
    s = sub.add_parser("summary", help="print the tables and write the sheets")
    s.add_argument("--faces", default=None, help="folder with det_10g.onnx and w600k_r50.onnx (insightface buffalo_l)")
    args = ap.parse_args()
    with torch.inference_mode():
        run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
