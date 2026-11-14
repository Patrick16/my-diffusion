"""How much video memory and time the large sizes take: text-to-image, edits with 1-2 references, outpaint.

The engine refuses edits above EDIT_MAX_TIER and has no 1440 tier. This script lifts both limits for itself and
measures what really happens there, stage by stage:

    uv run python scripts/check_memory.py run [--cases ...] [--label NAME] [--stream-blocks N] [--seed 42] [--force]
    uv run python scripts/check_memory.py summary [--labels ...]
    uv run python scripts/check_memory.py profile [--cases ...] [--stream-blocks N]

`run` appends one line per generation to outputs/memory/results.jsonl and keeps the picture next to it
(outputs/memory/<label>_<case>.png). A line that is already there is skipped (--force redoes it), so the cases can
be run in several goes. --label names the state of the code or of the settings that is being measured ("base",
"stream8", ...): `summary` prints the labels side by side and compares each picture with the first label's.
--stream-blocks N keeps the weights of N of the network's 28 blocks in RAM (Krea2Engine.stream_weights): each frees
0.44 GB of video memory and is copied to the GPU for the moment it runs.

What is recorded for each generation:
    tokens       length of the network's sequence: text + picture + references
    base_gb      video memory in use right before sampling: the network, the LoRAs, the inputs
    peak_gb      the largest amount in use during sampling (allocated by tensors)
    reserved_gb  the same, as taken from the driver by torch (allocated + its cache): this is what must fit the card
    sample_s     sampling time; step_s - the steps 2..N one by one (the first step of a new size carries one-time setup)
    encode_gb    peak of the VAE encoding the references; decode_gb / decode_s - peak and time of the VAE decode
    oom          the stage that ran out of memory ("sample", "decode", "encode"), or null

`profile` shows where the working memory of one block goes: it runs two steps of each case and prints, for the parts
of a block in the middle of the network (the two normalizations, attention, the MLP), how much is in use when the
part starts and how much more it takes while it runs, in GB and in "units" (one unit = the block's stream itself,
tokens x 6144 numbers in bf16).

Cases: t2i_<tier>, edit1_<tier> (one reference: the fox), edit2_<tier> (the fox and the man), out1_<tier> (outpaint
of the fox: 25% added on the left and on the right), keep1_<tier> (the same with keep_original).

Fixed: turbo krea2_turbo_mxfp8, 8 steps, the settings of the environment otherwise. The process is capped at the
video memory that is free at start, so that an overrun is an OOM error and not a silent spill into shared memory
(which is 8-10 times slower). Stop the server and other programs that hold video memory before measuring."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import my_diffusion.engine as engine_module
from my_diffusion import resolutions, transformer
from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine

OUT = PROJECT_ROOT / "outputs"
WORK = OUT / "memory"
RESULTS = WORK / "results.jsonl"
REF_FOX = WORK / "ref_fox.png"
REF_MAN = WORK / "ref_man.png"
GIB = 2**30
STEPS = 8

FOX = "a fox walking in fresh snow at golden hour, natural light, detailed fur"
MAN = ("studio portrait photo of a 50-year-old man with a short grey beard, round wire glasses and a scar across the left "
       "eyebrow, wearing a dark green wool coat, looking at the camera, plain light grey background, sharp focus")
FOX_EDIT = "Change the background to a wooden bridge over a river, keep the fox unchanged."
TWO_EDIT = "The man from image 2 kneels in the snow next to the fox from image 1, keep the fox and the man's face unchanged."
OUT_EDIT = "Extend the picture to the left and to the right: more of the snowy field and the forest behind the fox."
SIDES = (0.0, 0.0, 25.0, 25.0)   # outpaint margins: top, bottom, left, right, in percent of the reference's sides


@dataclass(frozen=True)
class Case:
    prompt: str
    resolution: str
    aspect: str = "16:9"
    refs: tuple[str, ...] = ()            # "fox" | "man"
    outpaint: tuple[float, float, float, float] | None = None
    keep_original: bool = False


def _cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for tier in ("480", "720", "1080", "1440"):
        cases[f"t2i_{tier}"] = Case(FOX, tier)
        cases[f"edit1_{tier}"] = Case(FOX_EDIT, tier, refs=("fox",))
        cases[f"edit2_{tier}"] = Case(TWO_EDIT, tier, refs=("fox", "man"))
        cases[f"out1_{tier}"] = Case(OUT_EDIT, tier, refs=("fox",), outpaint=SIDES)
        cases[f"keep1_{tier}"] = Case(OUT_EDIT, tier, refs=("fox",), outpaint=SIDES, keep_original=True)
    return cases


CASES = _cases()
DEFAULT_CASES = [
    "t2i_720", "t2i_1080", "t2i_1440",
    "edit1_720", "edit1_1080", "edit1_1440",
    "edit2_480", "edit2_720", "edit2_1080",
    "out1_720", "out1_1080", "out1_1440",
]


def load_results() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["label"], row["case"]] = row   # a later line replaces an earlier one
    return rows


# ---------------------------------------------------------------------- run
def reference(name: str, engine: Krea2Engine) -> Image.Image:
    """The reference pictures, made once: the fox (736x560) and the man's portrait (960x960)."""
    path, prompt, seed, tier, aspect = {
        "fox": (REF_FOX, FOX, 42, "480", "4:3"),
        "man": (REF_MAN, MAN, 7, "720", "1:1"),
    }[name]
    if not path.exists():
        r = engine.generate(prompt, seed=seed, resolution=tier, aspect_ratio=aspect, steps=STEPS)
        Path(r["image_path"]).replace(path)
        print(f"reference picture: {path}", flush=True)
    return Image.open(path).convert("RGB")


class Stages:
    """Wraps the engine's VAE so that the peak of each stage is measured on its own. The engine itself resets the
    peak counter right before sampling; the sampling peak is read in on_progress, after every step."""

    def __init__(self, engine: Krea2Engine):
        self.vae = engine.vae
        self.encode, self.decode = engine.vae.encode, engine.vae.decode
        self.stage = "prepare"
        self.encode_gb = self.decode_gb = 0.0
        engine.vae.encode = self._encode
        engine.vae.decode = self._decode

    def _measured(self, call, name: str, *args):
        self.stage = name
        torch.cuda.reset_peak_memory_stats()
        try:
            return call(*args)
        finally:
            torch.cuda.synchronize()
            setattr(self, f"{name}_gb", max(getattr(self, f"{name}_gb"), torch.cuda.max_memory_allocated() / GIB))

    def _encode(self, image):
        result = self._measured(self.encode, "encode", image)
        self.stage = "sample"   # the next heavy stage; an error between the two is reported as sampling
        return result

    def _decode(self, latent):
        return self._measured(self.decode, "decode", latent)

    def restore(self) -> None:
        self.vae.encode, self.vae.decode = self.encode, self.decode


def run_one(engine: Krea2Engine, case_name: str, label: str, seed: int) -> dict:
    case = CASES[case_name]
    refs = [reference(name, engine) for name in case.refs]
    marks: list[float] = []
    lengths: list[int] = []
    sampling = dict(peak=0.0, reserved=0.0, base=0.0)

    def on_progress(done: int, total: int) -> None:
        torch.cuda.synchronize()
        marks.append(time.perf_counter())
        sampling["peak"] = torch.cuda.max_memory_allocated() / GIB
        sampling["reserved"] = torch.cuda.max_memory_reserved() / GIB

    def before_block(module, args) -> None:
        if not lengths:   # the first network call: what is on the GPU before any activation
            sampling["base"] = torch.cuda.memory_allocated() / GIB
        lengths.append(args[0].shape[1])

    stages = Stages(engine)
    stages.stage = "sample" if not refs else "prepare"
    hook = engine.model.blocks[0].register_forward_pre_hook(before_block)
    row = dict(label=label, case=case_name, seed=seed, resolution=case.resolution, references=len(refs),
               outpaint=case.outpaint is not None, keep_original=case.keep_original)
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    try:
        r = engine.generate(case.prompt, seed=seed, resolution=case.resolution, aspect_ratio=case.aspect, steps=STEPS,
                            references=refs, outpaint=case.outpaint, keep_original=case.keep_original, on_progress=on_progress)
    except torch.OutOfMemoryError as e:
        row.update(
            oom=stages.stage, error=str(e).splitlines()[0][:200], steps_done=len(marks), tokens=lengths[0] if lengths else None,
            base_gb=round(sampling["base"], 2), peak_gb=round(max(sampling["peak"], torch.cuda.max_memory_allocated() / GIB), 2),
            reserved_gb=round(torch.cuda.max_memory_reserved() / GIB, 2),
            encode_gb=round(stages.encode_gb, 2), decode_gb=round(stages.decode_gb, 2),
        )
        if len(marks) > 2:   # the sampling itself finished and a later stage failed
            row["step_s"] = [round(b - a, 3) for a, b in zip(marks, marks[1:])]
        engine._applied_loras = []   # the attached set is in an unknown state after a failure
        torch.cuda.empty_cache()
        return row
    finally:
        hook.remove()
        stages.restore()
    path = WORK / f"{label}_{case_name}.png"
    Path(r["image_path"]).replace(path)
    row.update(
        oom=None, png=path.name, size=[r["width"], r["height"]], tokens=lengths[0],
        base_gb=round(sampling["base"], 2), peak_gb=round(sampling["peak"], 2), reserved_gb=round(sampling["reserved"], 2),
        sample_s=r["timings"]["sample"], step_s=[round(b - a, 3) for a, b in zip(marks, marks[1:])],
        encode_gb=round(stages.encode_gb, 2), decode_gb=round(stages.decode_gb, 2), decode_s=r["timings"]["decode"],
        total_s=r["timings"]["total"], alloc_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0),
        edit_lora_rank=r["edit_lora_rank"],
    )
    return row


def describe(row: dict) -> str:
    if row["oom"]:
        return (f"OOM in {row['oom']} after {row['steps_done']} steps: {row['tokens']} tokens, base {row['base_gb']} GB, "
                f"peak {row['peak_gb']} GB (reserved {row['reserved_gb']})")
    return (f"{row['size'][0]}x{row['size'][1]}, {row['tokens']} tokens: sampling {row['sample_s']:.1f}s, base {row['base_gb']} GB, "
            f"peak {row['peak_gb']} GB (reserved {row['reserved_gb']}), decode {row['decode_gb']} GB / {row['decode_s']:.1f}s, "
            f"total {row['total_s']:.1f}s")


def start_engine(args) -> Krea2Engine:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    WORK.mkdir(parents=True, exist_ok=True)
    # sizes the engine does not offer (yet): only inside this script
    resolutions.RESOLUTION_TIERS["1440"] = 1440
    engine_module.EDIT_MAX_TIER = {1: "1440", 2: "1440"}
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.97, free / total))
    print(f"video memory: {free / GIB:.1f} GB free of {total / GIB:.1f} GB; the process is capped at the free part", flush=True)

    overrides = {}
    if args.lora_streaming is not None:
        overrides["lora_streaming"] = args.lora_streaming == "on"
    settings = Settings(checkpoint="krea2_turbo_mxfp8.safetensors", **overrides)
    engine = Krea2Engine(settings)
    t0 = time.time()
    engine.load()
    engine.stream_weights(args.stream_blocks)
    if args.chunk_tokens is not None:
        transformer.CHUNK_TOKENS = args.chunk_tokens
    print(f"engine loaded in {time.time() - t0:.0f}s: matmul {settings.matmul}, attention {settings.attention}, "
          f"lora_streaming {settings.lora_streaming}, blocks with streamed weights {args.stream_blocks}", flush=True)
    return engine


def run(args) -> None:
    engine = start_engine(args)
    settings = engine.settings
    done = load_results()
    with RESULTS.open("a", encoding="utf-8") as log:
        for case in args.cases:
            if not args.force and (args.label, case) in done:
                continue
            row = run_one(engine, case, args.label, args.seed)
            row.update(matmul=settings.matmul, attention=settings.attention, lora_streaming=settings.lora_streaming,
                       stream_blocks=args.stream_blocks, chunk_tokens=transformer.CHUNK_TOKENS)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"{args.label:>12} {case:>11}: {describe(row)}", flush=True)


# ---------------------------------------------------------------------- profile
PARTS = ("prenorm", "attn", "postnorm", "mlp")


def profile(args) -> None:
    engine = start_engine(args)
    block = engine.model.blocks[len(engine.model.blocks) // 2]
    print("| case | tokens | unit, MB | part | in use at its start, GB | extra while it runs, GB | extra, units |")
    print("|---|---|---|---|---|---|---|")
    for case_name in args.cases:
        case = CASES[case_name]
        refs = [reference(name, engine) for name in case.refs]
        before: dict[str, float] = {}
        extra: dict[str, float] = {}
        tokens = [0]

        def enter(name: str):
            def hook(module, hook_args) -> None:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before[name] = torch.cuda.memory_allocated()
                if name == "prenorm":
                    tokens[0] = hook_args[0].shape[1]
            return hook

        def leave(name: str):
            def hook(module, hook_args, output) -> None:
                torch.cuda.synchronize()
                # the part's result is counted too: it is new memory the part asked for
                extra[name] = max(extra.get(name, 0.0), torch.cuda.max_memory_allocated() - before[name])
            return hook

        hooks = [getattr(block, name).register_forward_pre_hook(enter(name)) for name in PARTS]
        hooks += [getattr(block, name).register_forward_hook(leave(name)) for name in PARTS]
        torch.cuda.empty_cache()
        try:
            engine.generate(case.prompt, seed=args.seed, resolution=case.resolution, aspect_ratio=case.aspect, steps=2,
                            references=refs, outpaint=case.outpaint, keep_original=case.keep_original)
        except torch.OutOfMemoryError as e:
            print(f"| {case_name} | {tokens[0]} | | OOM: {str(e).splitlines()[0][:120]} | | | |")
            engine._applied_loras = []
            torch.cuda.empty_cache()
        finally:
            for hook in hooks:
                hook.remove()
        unit = tokens[0] * engine.model.cfg.features * 2
        for name in PARTS:
            if name in extra:
                print(f"| {case_name} | {tokens[0]} | {unit / 2**20:.0f} | {name} | {before[name] / GIB:.2f} | {extra[name] / GIB:.2f} | {extra[name] / unit:.1f} |", flush=True)


# ---------------------------------------------------------------------- summary
def difference(a: Path, b: Path) -> str:
    """How far picture a is from picture b: "same" (every pixel equal) or the PSNR in dB."""
    x, y = (np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) for p in (a, b))
    if x.shape != y.shape:
        return "other size"
    mse = float(((x - y) ** 2).mean())
    return "same" if mse == 0 else f"{10 * math.log10(255 ** 2 / mse):.1f} dB"


def summary(args) -> None:
    rows = load_results()
    labels = args.labels or list(dict.fromkeys(label for label, _ in rows))
    print(f"| case | label | size | tokens | base, GB | peak, GB | reserved, GB | sampling, s | decode, GB | decode, s | total, s | vs {labels[0]} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for case in CASES:
        first = rows.get((labels[0], case))
        for label in labels:
            r = rows.get((label, case))
            if r is None:
                continue
            if r["oom"]:
                print(f"| {case} | {label} | | {r['tokens']} | {r['base_gb']} | {r['peak_gb']} | {r['reserved_gb']} | "
                      f"OOM in {r['oom']}, step {r['steps_done'] + 1} | {r['decode_gb'] or ''} | | | |")
                continue
            # steps 2..N scaled to all N: the first step of a new size carries one-time kernel setup
            seconds = sum(r["step_s"]) * STEPS / (STEPS - 1)
            versus = "" if label == labels[0] or first is None or first["oom"] else difference(WORK / r["png"], WORK / first["png"])
            print(f"| {case} | {label} | {r['size'][0]}x{r['size'][1]} | {r['tokens']} | {r['base_gb']} | {r['peak_gb']} | "
                  f"{r['reserved_gb']} | {seconds:.1f} | {r['decode_gb']} | {r['decode_s']:.1f} | {r['total_s']:.1f} | {versus} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="generate the pictures and record memory and time")
    r.add_argument("--cases", nargs="+", choices=list(CASES), default=DEFAULT_CASES)
    r.add_argument("--label", default="base", help="name of the measured state of the code / settings")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--lora-streaming", choices=["on", "off"], default=None, help="override settings.lora_streaming")
    r.add_argument("--stream-blocks", type=int, default=0, help="blocks of the network whose weights stay in RAM (0-28)")
    r.add_argument("--chunk-tokens", type=int, default=None, help="override transformer.CHUNK_TOKENS (a huge value = no pieces)")
    r.add_argument("--force", action="store_true", help="redo the cases that are already recorded")
    s = sub.add_parser("summary", help="print the table")
    s.add_argument("--labels", nargs="+", default=None)
    f = sub.add_parser("profile", help="where the working memory of one block goes")
    f.add_argument("--cases", nargs="+", choices=list(CASES), default=["t2i_1080", "edit1_720"])
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--lora-streaming", choices=["on", "off"], default=None, help="override settings.lora_streaming")
    f.add_argument("--stream-blocks", type=int, default=0, help="blocks of the network whose weights stay in RAM (0-28)")
    f.add_argument("--chunk-tokens", type=int, default=None, help="override transformer.CHUNK_TOKENS (a huge value = no pieces)")
    args = ap.parse_args()
    with torch.inference_mode():
        {"run": run, "summary": summary, "profile": profile}[args.command](args)


if __name__ == "__main__":
    main()
