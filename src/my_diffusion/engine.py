"""Engine: combines all the parts into one operation "prompt -> image".

Order of processing one request:

    [Qwen on the GPU]     (optional) the enhancer rewrites the prompt
                          prompt and negative prompt -> hidden states
    [network on the GPU]  attach LoRAs, noise -> latent in N steps
    [VAE on the GPU]      latent -> pixels -> PNG

Only one large model lives on the GPU at a time (see memory.py). Qwen results
are cached: repeating the same prompt (for example, with another seed) does not require moving Qwen back to the GPU - the network stays in place and the request starts at once."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch

from . import sampler
from .config import Settings
from .lora import LoraFile, apply_loras, load_lora
from .memory import GpuSlot, Resident
from .resolutions import resolve_dimensions
from .text_encoder import QwenVL
from .transformer import Krea2Transformer
from .vae import LATENT_CHANNELS, VAE_SCALE, LatentDecoder
from .weights import load_into, read_checkpoint

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1

# How the two kinds of Krea 2 checkpoints differ:
#   turbo - distilled: few steps and no CFG (guidance 1.0 = off);
#   raw   - the original model: more steps, needs CFG with a negative prompt.
KIND_DEFAULTS = {
    "turbo": {"steps": 8, "guidance_scale": 1.0},
    "raw": {"steps": 28, "guidance_scale": 4.5},
}

ENHANCER_SYSTEM_PROMPT = """You are a prompt writer for a text-to-image model.
Rewrite the user's idea as one detailed image description in natural English prose:
the subject and what it is doing, appearance, setting and background, framing and camera angle, lighting, colors, mood, style.
Keep every detail the user gave and do not change their intent; only add what is missing.
The user's idea may be in any language; always answer in English.
Never add captions, lettering, subtitles, watermarks or any written text to the image, and never quote the user's idea.
The only exception: if the user explicitly asks for specific words to be written in the image, keep those words verbatim in double quotes.
Write under 120 words. Output only the description: no preamble, no explanations, no lists."""


class Krea2Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gpu = GpuSlot(settings.device)
        self.qwen: QwenVL | None = None
        self.model: Krea2Transformer | None = None
        self.vae: LatentDecoder | None = None
        self.checkpoint_name: str | None = None
        self._text_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lora_cache: OrderedDict[tuple[str, float], LoraFile] = OrderedDict()
        self._lock = threading.Lock()   # one generation at a time: there is one GPU

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        s = self.settings
        t0 = time.time()
        self.qwen = QwenVL(s.text_encoder_dir)
        self.qwen.load()
        self.gpu.register("qwen", Resident.plain(self.qwen))
        logger.info("Qwen in RAM: %.1fs", time.time() - t0)

        t0 = time.time()
        self.vae = LatentDecoder(s.vae_path)
        self.vae.load()
        self.gpu.register("vae", Resident.plain(self.vae))
        logger.info("VAE in RAM: %.1fs", time.time() - t0)

        self.load_checkpoint(s.checkpoint)
        s.output_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, name: str) -> None:
        """Reads the checkpoint from disk into RAM (it will move to the GPU on the first generation)."""
        path = self._inside(self.settings.models_dir / "checkpoints", name)
        t0 = time.time()
        if self.gpu.active == "model":
            self.gpu.release()
        # free RAM from the previous checkpoint before reading the new one
        self.model = None
        self.gpu.residents.pop("model", None)
        self._lora_cache.clear()
        model = Krea2Transformer()
        load_into(model, read_checkpoint(path))
        self.model = model
        self.gpu.register("model", Resident.with_home_copy(model))
        self.checkpoint_name = path.name
        logger.info("checkpoint %s in RAM: %.1fs", path.name, time.time() - t0)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def kind(self) -> str:
        if self.settings.checkpoint_kind != "auto":
            return self.settings.checkpoint_kind
        return "raw" if "raw" in (self.checkpoint_name or "").lower() else "turbo"

    # ------------------------------------------------------------------ text
    def enhance_prompt(self, prompt: str, seed: int | None = None) -> str:
        with self._lock:
            return self._enhance(prompt, seed)

    def _enhance(self, prompt: str, seed: int | None) -> str:
        self.gpu.use("qwen")
        return self.qwen.chat(ENHANCER_SYSTEM_PROMPT, prompt, max_new_tokens=self.settings.enhancer_max_new_tokens, seed=seed)

    def _encode(self, prompt: str) -> torch.Tensor:
        cached = self._text_cache.get(prompt)
        if cached is not None:
            self._text_cache.move_to_end(prompt)
            return cached
        self.gpu.use("qwen")
        cond = self.qwen.encode(prompt).cpu()
        self._text_cache[prompt] = cond
        while len(self._text_cache) > 32:
            self._text_cache.popitem(last=False)
        return cond

    # ------------------------------------------------------------------ LoRA
    def _inside(self, folder: Path, name: str) -> Path:
        """Path to a file inside folder; does not allow escaping it via '..'."""
        folder = folder.resolve()
        path = (folder / name).resolve()
        if not path.is_file():
            # the name is given without an extension; we append it rather than replace: names can contain dots ("style_v1.2.1")
            path = path.with_name(path.name + ".safetensors")
        if not path.is_relative_to(folder):
            raise ValueError(f"path {name!r} is outside {folder}")
        if not path.is_file():
            raise FileNotFoundError(f"no file {name!r} in {folder}")
        return path

    def _lora(self, name: str) -> LoraFile:
        path = self._inside(self.settings.lora_dir, name)
        key = (str(path), path.stat().st_mtime)
        if key not in self._lora_cache:
            self._lora_cache[key] = load_lora(path, self.model)
            while len(self._lora_cache) > 12:
                self._lora_cache.popitem(last=False)
        self._lora_cache.move_to_end(key)
        return self._lora_cache[key]

    def list_files(self, kind: str) -> list[str]:
        folder = (self.settings.models_dir / kind).resolve()
        patterns = ("*.safetensors", "*.gguf") if kind == "checkpoints" else ("*.safetensors",)
        return sorted(p.relative_to(folder).as_posix() for pat in patterns for p in folder.rglob(pat))

    # ------------------------------------------------------------------ generation
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        resolution: str = "720",
        aspect_ratio: str = "16:9",
        seed: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        shift: float | None = None,
        loras: list[tuple[str, float]] | None = None,
        enhance: bool = False,
    ) -> dict:
        if not self.loaded:
            raise RuntimeError("the engine is not loaded yet")
        with self._lock:
            return self._generate(prompt, negative_prompt, resolution, aspect_ratio, seed, steps, guidance_scale, shift, loras or [], enhance)

    def _generate(self, prompt, negative_prompt, resolution, aspect_ratio, seed, steps, guidance_scale, shift, loras, enhance) -> dict:
        s = self.settings
        defaults = KIND_DEFAULTS[self.kind]
        steps = steps or defaults["steps"]
        guidance_scale = defaults["guidance_scale"] if guidance_scale is None else guidance_scale
        seed = random.randint(0, MAX_SEED) if seed is None else seed
        width, height = resolve_dimensions(resolution, aspect_ratio)
        lora_files = [(self._lora(name), weight) for name, weight in loras]   # read in advance: a name error surfaces before the heavy work
        timings: dict[str, float] = {}
        t_all = time.time()

        # 1. text
        t0 = time.time()
        enhanced = self._enhance(prompt, seed) if enhance else None
        timings["enhance"] = time.time() - t0
        t0 = time.time()
        use_cfg = guidance_scale != 1.0
        cond = self._encode(enhanced or prompt)
        uncond = self._encode(negative_prompt) if use_cfg else None
        timings["encode"] = time.time() - t0

        # 2. noise -> latent
        t0 = time.time()
        self.gpu.use("model")
        apply_loras(self.model, lora_files, s.device)
        timings["load_model"] = time.time() - t0

        lat_h, lat_w = height // VAE_SCALE, width // VAE_SCALE
        tokens = (lat_h // 2) * (lat_w // 2)
        mu = sampler.auto_mu(tokens) if shift is None else shift
        sigmas = sampler.make_sigmas(steps, mu)
        noise = sampler.make_noise(seed, LATENT_CHANNELS, lat_h, lat_w).to(s.device)

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        latent = sampler.sample(
            self.model, noise, cond.to(s.device), sigmas,
            uncond=None if uncond is None else uncond.to(s.device), guidance_scale=guidance_scale,
        )
        timings["sample"] = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / 2**30

        # 3. latent -> pixels
        t0 = time.time()
        self.gpu.use("vae")
        image = self.vae.decode(latent)
        timings["decode"] = time.time() - t0

        path = s.output_dir / f"{int(time.time())}_{seed}.png"
        image.save(path)
        timings["total"] = time.time() - t_all
        logger.info(
            "%dx%d, %d steps, mu=%.2f: total %.1fs (text %.1f, network on GPU %.1f, sampling %.1f, VAE %.1f), peak VRAM %.1f GB",
            width, height, steps, mu, timings["total"], timings["enhance"] + timings["encode"], timings["load_model"],
            timings["sample"], timings["decode"], peak_vram,
        )
        return {
            "image_path": str(path),
            "width": width, "height": height, "seed": seed,
            "prompt": prompt, "enhanced_prompt": enhanced,
            "steps": steps, "guidance_scale": guidance_scale, "shift": round(mu, 4),
            "checkpoint": self.checkpoint_name,
            "loras": [{"name": f.name, "weight": w} for f, w in lora_files],
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }
