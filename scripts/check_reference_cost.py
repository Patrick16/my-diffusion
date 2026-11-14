"""Can the reference tokens of an edit be made cheaper? An experiment, not a feature: the variants are patched
into the network only inside this script.

    uv run python scripts/check_reference_cost.py [VARIANTS...] [--faces DIR]

A reference doubles the network's sequence, and the step time grows faster than the length. Two ideas are tried on
two 720 edits (the fox: "change the background, keep the fox"; the man's portrait: "... keep the man"), same prompt,
seed and size for every variant:

    normal      the engine as it is: the reference tokens are part of the sequence in every block and step
    iso_sigma   the references go through the blocks on their own (they see only themselves), anew on every step;
                the picture and the text attend to their keys / values. Shows what isolating them does.
    iso_zero    the same, but the references go through ONCE, modulated as at noise level 0, and their keys /
                values are reused on every step: the "K/V cache"
    half        the reference frame is encoded at half the side (a quarter of the tokens), token positions as they are
    half_pos    the same, with the positions stretched over the whole frame

Prints the sampling time per step and the peak memory; the pictures go to outputs/memory/refexp_<case>_<variant>.png.
--faces DIR (insightface's buffalo_l: det_10g.onnx, w600k_r50.onnx) adds the ArcFace similarity of the man's face
to the reference (same person: usually above 0.5).

Result on a 16 GB RTX 5080 Laptop (turbo, 8 steps, seed 42; also in guides/performance.md): isolating the references loses the subject (the
similarity falls from 0.70 to 0.2-0.3), so their keys / values cannot be cached; a half-size reference keeps it
(0.70) and makes a step 40% shorter; stretched positions are worse (0.55).

The reference pictures (outputs/memory/ref_fox.png, ref_man.png) are made by scripts/check_memory.py."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.attention import sdpa_kernel

from my_diffusion.config import PROJECT_ROOT, Settings
from my_diffusion.engine import Krea2Engine
from my_diffusion.resolutions import resolve_dimensions
from my_diffusion.transformer import CUDNN_FIRST, apply_rope, build_rope, timestep_embedding
from my_diffusion.weights import shared_input, streamed_adapters, streamed_weights

WORK = PROJECT_ROOT / "outputs" / "memory"
GIB = 2**30
FOX_EDIT = "Change the background to a wooden bridge over a river, keep the fox unchanged."
MAN_EDIT = "Change the background to a busy night street with neon signs, keep the man unchanged."
CASES = {
    "fox720": (FOX_EDIT, "720", "16:9", "ref_fox.png"),
    "man720": (MAN_EDIT, "720", "1:1", "ref_man.png"),
}
VARIANTS = ("normal", "iso_sigma", "iso_zero", "half", "half_pos")


def attention(attn, x, rope, capture=None, extra=None):
    """Attention.forward (cudnn kernel) that can hand out its keys / values (capture) or take more of them (extra)."""
    b, n, _ = x.shape
    xin = shared_input(x, attn.wq, attn.wk, attn.wv, attn.gate)
    q = attn.wq(xin).view(b, n, attn.heads, attn.headdim).transpose(1, 2)
    k = attn.wk(xin).view(b, n, attn.kvheads, attn.headdim).transpose(1, 2)
    v = attn.wv(xin).view(b, n, attn.kvheads, attn.headdim).transpose(1, 2)
    q = apply_rope(attn.qknorm.qnorm(q), rope)
    k = apply_rope(attn.qknorm.knorm(k), rope)
    if capture is not None:
        capture.append((k, v))
    if extra is not None:
        k = torch.cat((k, extra[0]), dim=2)
        v = torch.cat((v, extra[1]), dim=2)
    rep = attn.heads // attn.kvheads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    with sdpa_kernel(CUDNN_FIRST, set_priority=True):
        out = F.scaled_dot_product_attention(q, k, v)
    del q, k, v
    out = out.transpose(1, 2).reshape(b, n, attn.heads * attn.headdim)
    return attn.wo(out * torch.sigmoid(attn.gate(xin)))


def run_blocks(model, x, tvec, rope, capture=None, extra=None):
    for i, block in enumerate(model.blocks):
        with streamed_weights(block, x.device), streamed_adapters(block, x.device):
            prescale, preshift, pregate, postscale, postshift, postgate = block.mod(tvec)
            x = x + pregate * attention(block.attn, (1 + prescale) * block.prenorm(x) + preshift, rope,
                                        capture, None if extra is None else extra[i])
            x = x + postgate * block.mlp((1 + postscale) * block.postnorm(x) + postshift)
    return x


def isolated_forward(self, latent, sigma, text, refs=None):
    """Krea2Transformer.forward with the references kept out of the main sequence (iso_sigma / iso_zero)."""
    cfg = self.cfg
    p = cfg.patch
    b, c, h, w = latent.shape
    gh, gw = h // p, w // p
    img = latent.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
    img = self.first(img.to(self.dtype))
    t = self.tmlp(timestep_embedding(sigma, cfg.tdim).unsqueeze(1).to(self.dtype))
    tvec = self.tproj(t)
    txt = text.to(self.dtype)
    ntxt = txt.shape[1]
    pos = torch.zeros(b, ntxt + gh * gw, 3, device=latent.device, dtype=torch.float32)
    grid = pos[:, ntxt:].view(b, gh, gw, 3)
    grid[..., 1] = torch.arange(gh, device=latent.device, dtype=torch.float32)[:, None]
    grid[..., 2] = torch.arange(gw, device=latent.device, dtype=torch.float32)[None, :]

    cache = self.ref_cache   # per block: the references' keys and values
    if self.ref_mode == "iso_sigma" or not cache:
        ref_tokens, ref_pos = self._append_refs(img[:, :0], pos[:, :0], refs)
        tvec_ref = tvec
        if self.ref_mode == "iso_zero":
            tvec_ref = self.tproj(self.tmlp(timestep_embedding(torch.zeros_like(sigma), cfg.tdim).unsqueeze(1).to(self.dtype)))
        cache.clear()
        run_blocks(self, ref_tokens, tvec_ref, build_rope(ref_pos, self.rope_axes, cfg.theta), capture=cache)

    x = run_blocks(self, torch.cat((txt, img), dim=1), tvec, build_rope(pos, self.rope_axes, cfg.theta), extra=cache)
    out = self.last(x, t)[:, ntxt:]
    return out.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def stretched_append_refs(self, img, pos, refs):
    """Krea2Transformer._append_refs with each reference's positions stretched over the output grid (self.out_grid)."""
    p = self.cfg.patch
    b = img.shape[0]
    tokens, positions = [img], [pos]
    gh, gw = self.out_grid
    for k, (ref, _offset) in enumerate(refs, start=1):
        ref = ref.to(img.device).expand(b, -1, -1, -1)
        _, c, rh, rw = ref.shape
        rgh, rgw = rh // p, rw // p
        patches = ref.reshape(b, c, rgh, p, rgw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, rgh * rgw, c * p * p)
        tokens.append(self.first(patches.to(self.dtype)))
        rows = (torch.arange(rgh, device=img.device, dtype=torch.float32) + 0.5) * (gh / rgh) - 0.5
        cols = (torch.arange(rgw, device=img.device, dtype=torch.float32) + 0.5) * (gw / rgw) - 0.5
        rpos = torch.zeros(b, rgh, rgw, 3, device=img.device, dtype=torch.float32)
        rpos[..., 0] = k
        rpos[..., 1] = rows[:, None]
        rpos[..., 2] = cols[None, :]
        positions.append(rpos.view(b, rgh * rgw, 3))
    return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", help=f"any of {list(VARIANTS)}; default: all")
    ap.add_argument("--faces", default=None, help="folder with det_10g.onnx and w600k_r50.onnx (insightface buffalo_l)")
    args = ap.parse_args()
    variants = args.variants or list(VARIANTS)
    if unknown := [v for v in variants if v not in VARIANTS]:
        ap.error(f"unknown variant {unknown}; available: {list(VARIANTS)}")
    faces = man_vector = None
    if args.faces:
        sys.path.insert(0, str(Path(__file__).parent))
        from check_attention import Faces   # the detector and the ArcFace embedder live there

        faces = Faces(Path(args.faces))
        man = Image.open(WORK / "ref_man.png").convert("RGB")
        man_vector = faces.embed(man, faces.detect(man)[1])

    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(0.97, free / total))
    engine = Krea2Engine(Settings(checkpoint="krea2_turbo_mxfp8.safetensors"))
    engine.load()
    model = engine.model
    cls = type(model)
    real_forward, real_append, real_encode = cls.forward, cls._append_refs, engine.vae.encode
    try:
        for case, (prompt, tier, aspect, ref_file) in CASES.items():
            ref = Image.open(WORK / ref_file).convert("RGB")
            width, height = resolve_dimensions(tier, aspect)
            model.out_grid = (height // 16, width // 16)
            for variant in variants:
                cls.forward, cls._append_refs, engine.vae.encode = real_forward, real_append, real_encode
                if variant.startswith("iso"):
                    model.ref_mode, model.ref_cache = variant, []
                    cls.forward = isolated_forward
                if variant.startswith("half"):
                    def encode(image, real=real_encode):
                        small = (max(16, round(image.width / 32) * 16), max(16, round(image.height / 32) * 16))
                        return real(image.resize(small, Image.LANCZOS))
                    engine.vae.encode = encode
                    if variant == "half_pos":
                        cls._append_refs = stretched_append_refs
                marks: list[float] = []

                def on_progress(done, total_steps):
                    torch.cuda.synchronize()
                    marks.append(time.perf_counter())

                torch.cuda.empty_cache()
                try:
                    r = engine.generate(prompt, seed=42, resolution=tier, aspect_ratio=aspect, steps=8, references=[ref], on_progress=on_progress)
                except torch.OutOfMemoryError as e:
                    print(f"{case:>8} {variant:>10}: OOM {str(e).splitlines()[0][:100]}", flush=True)
                    engine._applied_loras = []
                    continue
                finally:
                    model.ref_cache = []
                path = WORK / f"refexp_{case}_{variant}.png"
                Path(r["image_path"]).replace(path)
                steps = [b - a for a, b in zip(marks, marks[1:])]
                line = (f"{case:>8} {variant:>10}: sampling {r['timings']['sample']:.1f}s, {sum(steps) / len(steps):.2f} s/step "
                        f"(steps 2-8), peak {torch.cuda.max_memory_allocated() / GIB:.2f} GB")
                if faces is not None and case.startswith("man"):
                    picture = Image.open(path).convert("RGB")
                    found = faces.detect(picture)
                    line += ", no face found" if not found else f", face similarity to the reference {float(faces.embed(picture, found[1]) @ man_vector):.3f}"
                print(line, flush=True)
    finally:
        cls.forward, cls._append_refs, engine.vae.encode = real_forward, real_append, real_encode


if __name__ == "__main__":
    with torch.inference_mode():
        main()
