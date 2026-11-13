"""Engine: combines all the parts into one operation "prompt -> image".

Order of processing one request:

    [Qwen on the GPU]     (optional) the enhancer rewrites the prompt
                          prompt and negative prompt -> hidden states
                          (editing: together with the reference pictures, through the vision tower)
    [VAE on the GPU]      (image-to-image, inpaint, editing) input pictures -> clean latents
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
from PIL import Image

from . import background, imaging, sampler
from .config import Settings
from .imaging import cover_crop
from .lora import LoraFile, apply_loras, load_lora, truncate_rank
from .memory import GpuSlot, Resident
from .resolutions import ASPECT_RATIOS, resolve_dimensions, resolve_dimensions_from_size
from .text_encoder import QwenVL
from .transformer import Krea2Transformer, set_attention_kernel
from .vae import LATENT_CHANNELS, VAE_SCALE, LatentDecoder
from .weights import load_into, read_checkpoint, set_matmul

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1
# XOR-ed into the seed for the starting noise of an edit (see _generate): keeps the edit from reusing the
# noise of a picture generated with the same seed and size.
EDIT_NOISE_SALT = 0x5EED_ED17

# How the two kinds of Krea 2 checkpoints differ:
#   turbo - distilled: few steps and no CFG (guidance 1.0 = off);
#   raw   - the original model: more steps, needs CFG with a negative prompt.
KIND_DEFAULTS = {
    "turbo": {"steps": 8, "guidance_scale": 1.0},
    "raw": {"steps": 28, "guidance_scale": 4.5},
}

# The same for reference editing (generate with references), as in Wan2GP's edit presets. Turbo keeps its
# defaults. Raw takes 20 steps and guidance 3.0: Wan2GP's raw-edit preset says 2, but Wan2GP applies
# guidance + 1 as the real CFG scale, so its 2 is our 3.0.
EDIT_DEFAULTS = {
    "turbo": {"steps": 8, "guidance_scale": 1.0},
    "raw": {"steps": 20, "guidance_scale": 3.0},
}


def _tier_max_pixels(tier: str) -> int:
    """The largest frame (in pixels) of a resolution tier over all aspect ratios: rounding the sides to
    multiples of 16 makes some formats of a tier slightly larger than its 16:9 frame."""
    return max(w * h for w, h in (resolve_dimensions(tier, ratio) for ratio in ASPECT_RATIOS))


# What an edit fits into 16 GB of VRAM (measured with the reference probe on a 16 GB laptop GPU): the
# network, the edit LoRA and the reference tokens all sit on the GPU together.
#   - output size, by the number of references: one reference up to the 720 tier, two only at 480;
#   - the edit LoRA at its full rank 256 (+1.7 GB) only for one reference at the 480 tier. Otherwise it is
#     truncated to rank 128 (+0.85 GB, see lora.truncate_rank); at 720 that was checked to look right.
#   - these LoRA sizes are for settings.lora_streaming=False; with streaming (the default) the LoRA waits in
#     pinned RAM and only one block's pairs are on the GPU at a time (lora.apply_loras).
EDIT_MAX_TIER = {1: "720", 2: "480"}
EDIT_RANK_REDUCED = 128   # the edit LoRA is always used at this rank, see edit_lora_rank

# What the references are (Wan2GP's "KI" and "I"): reference 1 is the main scene, or all are people / objects.
REFERENCE_MODES = ("scene", "subjects")
# Background removal of references (Wan2GP's "remove_background_images_ref"): "subjects" cuts the background out of
# the people / objects (in "scene" mode reference 1 stays untouched, it is the scene).
REMOVE_BACKGROUND_MODES = ("none", "subjects")

IMAGE_CACHE_SIZE = 8  # encodings with pictures are large (~27 MB each: ~440 tokens x 12 layers x 2560), keep fewer


def edit_lora_rank(width: int, height: int, references: int) -> int:
    """Rank of the edit LoRA for every edit: EDIT_RANK_REDUCED (128).

    Measured on the GPU: rank 256 showed no visible gain over 128 on the same frames (the pictures differ
    by 7-20 units on average, with the same identity and the same edit), while at 256 the adapter takes
    1.7 GB of VRAM. That pushes the run to the edge of the 16 GB card, and the excess spills into shared
    memory, which made the same edit 8-10 times slower in a plain CLI process. The size and reference
    count are kept in the signature so the call site stays the same if this rule changes again."""
    return EDIT_RANK_REDUCED

ENHANCER_SYSTEM_PROMPT = """You are a prompt writer for a text-to-image model.
Rewrite the user's idea as one detailed image description in natural English prose:
the subject and what it is doing, appearance, setting and background, framing and camera angle, lighting, colors, mood, style.
Keep every detail the user gave and do not change their intent; only add what is missing.
The user's idea may be in any language; always answer in English.
Never add captions, lettering, subtitles, watermarks or any written text to the image, and never quote the user's idea.
The only exception: if the user explicitly asks for specific words to be written in the image, keep those words verbatim in double quotes.
Write under 120 words. Output only the description: no preamble, no explanations, no lists."""


def normalize_outpaint(outpaint, references: list, reference_mode: str) -> imaging.Margins | None:
    """Validate the outpaint request and return its margins, or None when outpainting is off.

    `outpaint` is None or (top, bottom, left, right) in percent of reference 1's sides. All zeros means
    off, so a client can always send the field. Outpainting grows reference 1 (the scene), so it needs
    references in "scene" mode. Pure: runs before any heavy work."""
    if outpaint is None:
        return None
    margins = imaging.Margins(*outpaint).validate()   # ValueError for negative or non-finite margins
    if not any(margins):
        return None
    if not references:
        raise ValueError("outpaint grows reference 1, so it needs references")
    if reference_mode != "scene":
        raise ValueError("outpaint needs reference_mode 'scene': with 'subjects' no reference is the scene to extend")
    return margins


class Krea2Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gpu = GpuSlot(settings.device)
        self.qwen: QwenVL | None = None
        self.model: Krea2Transformer | None = None
        self.vae: LatentDecoder | None = None
        self.checkpoint_name: str | None = None
        self.matmul: str = settings.matmul   # matrix multiplication mode; changed via set_matmul
        self.native_layers = 0               # how many layers are currently computed via the native fp8 path
        self._text_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        # encodings with pictures (editing): key = (prompt, digests of the prepared frames)
        self._image_cache: OrderedDict[tuple[str, tuple[str, ...]], torch.Tensor] = OrderedDict()
        self._lora_cache: OrderedDict[tuple[str, float], LoraFile] = OrderedDict()
        self._applied_loras: list[tuple[LoraFile, float]] = []   # the set of LoRAs currently attached to the network
        self._applied_streaming = settings.lora_streaming        # whether that set was attached with LoRA streaming
        self._lock = threading.Lock()   # one generation at a time: there is one GPU

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        s = self.settings
        t0 = time.time()
        pin = s.pin_memory and s.device != "cpu"
        self.qwen = QwenVL(s.text_encoder_dir, pin=pin)
        self.qwen.load()
        self.gpu.register("qwen", Resident.plain(self.qwen))
        logger.info("Qwen in RAM: %.1fs", time.time() - t0)

        t0 = time.time()
        self.vae = LatentDecoder(s.vae_path)
        self.vae.load()
        self.gpu.register("vae", Resident.with_home_copy(self.vae.vae, pin=pin))
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
        pin = self.settings.pin_memory and self.settings.device != "cpu"
        model = Krea2Transformer()
        load_into(model, read_checkpoint(path, pin=pin))
        self.model = model
        self._applied_loras = []   # the freshly built network has no LoRAs attached
        self.gpu.register("model", Resident.with_home_copy(model, pin=pin))
        self.checkpoint_name = path.name
        logger.info("checkpoint %s in RAM: %.1fs", path.name, time.time() - t0)
        self.set_matmul(self.matmul)
        set_attention_kernel(model.blocks, self.settings.attention)   # only the main blocks, see transformer.py
        logger.info("attention: kernel %s", self.settings.attention)

    def set_matmul(self, mode: str) -> None:
        """Select the matrix multiplication mode (exact / fp8 / fp8_all, see weights.py). It can be changed
        between requests: the weights are not touched, only the way they are multiplied changes."""
        self.native_layers = set_matmul(self.model, mode, self.settings.device)
        self.matmul = mode
        if mode == "exact":
            logger.info("multiplication: exact")
        elif self.native_layers:
            logger.info("multiplication: %s, the native fp8 path is used by %d layers", mode, self.native_layers)
        else:
            # not an error: the fast mode exists only for mxfp8; other formats support only the exact mode
            logger.info("multiplication: checkpoint %s has no mxfp8 layers, so mode %s does not apply - using exact", self.checkpoint_name, mode)

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

    def _encode_with_images(self, prompt: str, frames: list[Image.Image]) -> torch.Tensor:
        """Prompt + pictures -> Qwen hidden states (editing), with its own small cache.

        The key is the prompt plus a digest of each frame's pixels: the frames are already prepared for
        this request's output size, so the same upload at another resolution is another frame and is
        encoded anew, while repeating the request (for example with another seed) costs nothing."""
        key = (prompt, tuple(imaging.image_digest(frame) for frame in frames))
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        self.gpu.use("qwen")
        cond = self.qwen.encode(prompt, images=frames).cpu()
        self._image_cache[key] = cond
        while len(self._image_cache) > IMAGE_CACHE_SIZE:
            self._image_cache.popitem(last=False)
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

    def _with_edit_lora(
        self, lora_files: list[tuple[LoraFile, float]], width: int, height: int, references: int,
    ) -> tuple[list[tuple[LoraFile, float]], int | None]:
        """The LoRA set of an edit: the request's LoRAs plus the edit LoRA, at a rank that fits the GPU.

        The edit LoRA (settings.edit_lora) is added with settings.edit_lora_weight unless the request
        names it itself; then the request's weight is used, and 0 switches it off. If the rank budget
        (edit_lora_rank) is below the file's rank, the file is replaced by its truncated copy. The copy
        is made once and kept on the file object (LoraFile.truncated), so a repeated request passes the
        very same object and the attached set is not rebuilt; the full file and its copy are different
        objects, so switching between them does rebuild it. Returns the set and the rank actually used
        (None if the edit LoRA is off)."""
        s = self.settings
        edit = self._lora(s.edit_lora)   # the same cached object as when the request names this file
        if not any(f is edit for f, _ in lora_files):
            lora_files = lora_files + [(edit, s.edit_lora_weight)]
        if all(w == 0 for f, w in lora_files if f is edit):
            return lora_files, None
        used = truncate_rank(edit, edit_lora_rank(width, height, references))   # `edit` itself if it fits
        rank = max((down.shape[0] for down, _, _ in used.layers.values()), default=None)
        if used is not edit:
            logger.info(
                "edit LoRA %s: rank %d -> %d for %dx%d with %d reference(s), worst kept energy %.4f",
                edit.name, max(down.shape[0] for down, _, _ in edit.layers.values()), rank, width, height,
                references, used.worst_kept_energy,
            )
        return [(used if f is edit else f, w) for f, w in lora_files], rank

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
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        init_image: Image.Image | None = None,
        strength: float | None = None,
        mask: Image.Image | None = None,
        feather: int = 2,
        references: list[Image.Image] | None = None,
        reference_mode: str = "scene",
        outpaint: tuple[float, float, float, float] | None = None,
        remove_background: str = "none",
    ) -> dict:
        """guidance_interval - the part of the run (as fractions of the steps) where CFG is applied;
        only matters when guidance_scale != 1. See sampler.sample.

        init_image / strength - image-to-image. The picture is cropped to the requested size, encoded
        to a latent and noised up to the starting level of the schedule; the network then removes the
        noise only in the remaining steps. So the composition of the original survives when strength
        is low (0.3 changes little, 1.0 is the same as text-to-image and the picture is ignored in effect).
        strength is ignored without init_image. None means the mode's default: 0.6 for image-to-image,
        1.0 for inpaint.

        mask / feather - inpaint (needs init_image). White in the mask = regenerate, black = keep.
        The mask is cropped to the same frame as the picture and shrunk to the latent grid; feather
        softens its edge by that many latent pixels (1 latent pixel = 8 image pixels), which hides the
        seam. After every step the kept area is put back to the original picture at the current noise
        level, so only the white area is really generated. Default strength for inpaint is 1.0: the
        white area starts from pure noise and is not tied to what was there before; a lower strength
        keeps a hint of the old content inside the mask (useful to recolour or restyle, not to replace).

        references - reference editing: 1 or 2 pictures (settings.max_references) the result is built
        from; the prompt says what to do with them ("put the cat from image 1 on a bridge"). Reference 1 is
        the scene: it is cropped to the output frame exactly like init_image. Reference 2 is a subject or a
        person: it keeps its own proportions and is fitted into the frame (imaging.fit_reference).
        reference_mode (Wan2GP's "KI" / "I"): "scene" - reference 1 is the main scene, the others are people
        or objects; "subjects" - every reference is a person or an object, there is no scene, so all of them
        are fitted with their own proportions and the size comes from resolution + aspect_ratio.
        aspect_ratio="auto" (only with references, only in "scene" mode) takes the exact proportions of
        reference 1 at the pixel budget of `resolution` (sides rounded to 16), not the nearest preset.
        Editing switches on three things that only work together: the prepared pictures go into the Qwen
        prompt through its vision tower (the negative prompt, under CFG, gets the same pictures); their
        clean latents are appended to the network's sequence (see Krea2Transformer.forward); and the edit
        LoRA (settings.edit_lora) is attached, at rank 128 instead of 256 when the full rank would not fit
        the GPU (see _with_edit_lora). Each reference roughly doubles the attention's work per step.
        outpaint - (top, bottom, left, right) in percent of reference 1's height / width (Wan2GP's outpainting
        of an edit). Only in "scene" mode; all zeros = off. The result frame is reference 1 grown by those
        margins at the pixel budget of `resolution` (imaging.outpaint_frame; it replaces aspect_ratio). Reference
        1 is not cropped: it is resized to its place in the frame and given to the network at that token
        offset, and the rest of the frame is generated around it.
        remove_background - "subjects": cut the background (u2net, on the CPU) out of the people / objects before
        they are used: references 2.. in "scene" mode, all references in "subjects" mode; "none" - leave as is.
        Not combinable with init_image or mask."""
        if not self.loaded:
            raise RuntimeError("the engine is not loaded yet")
        with self._lock:
            return self._generate(
                prompt, negative_prompt, resolution, aspect_ratio, seed, steps, guidance_scale, shift, loras or [], enhance,
                tuple(guidance_interval), init_image, strength, mask, feather, list(references or []), reference_mode,
                outpaint, remove_background,
            )

    def _generate(
        self, prompt, negative_prompt, resolution, aspect_ratio, seed, steps, guidance_scale, shift, loras, enhance, guidance_interval,
        init_image=None, strength=None, mask=None, feather=2, references=None, reference_mode="scene",
        outpaint=None, remove_background="none",
    ) -> dict:
        s = self.settings
        if reference_mode not in REFERENCE_MODES:
            raise ValueError(f"reference_mode must be one of {list(REFERENCE_MODES)}, got {reference_mode!r}")
        if remove_background not in REMOVE_BACKGROUND_MODES:
            raise ValueError(f"remove_background must be one of {list(REMOVE_BACKGROUND_MODES)}, got {remove_background!r}")
        references = list(references or [])
        margins = normalize_outpaint(outpaint, references, reference_mode)
        if len(references) > s.max_references:
            raise ValueError(f"at most {s.max_references} reference pictures are supported, got {len(references)}")
        if references and (init_image is not None or mask is not None):
            raise ValueError("references cannot be combined with an init image or a mask: editing builds the picture from the references")
        auto_size = aspect_ratio == "auto"
        if auto_size:
            if not references:
                raise ValueError("aspect_ratio 'auto' takes the proportions of reference 1, so it needs references")
            if reference_mode != "scene":
                raise ValueError("aspect_ratio 'auto' needs reference_mode 'scene': with 'subjects' no picture is the scene to take the size from")
        # Which references lose their background: the people / objects, never the scene.
        first_subject = 1 if reference_mode == "scene" else 0
        cut_background = remove_background == "subjects" and len(references) > first_subject
        # The engine's own settings, not background.model_path(): that one reads a fresh Settings() from the
        # environment and would miss a models_dir given to this engine in code.
        rembg_path = s.rembg_model_path
        if cut_background and not rembg_path.is_file():
            raise FileNotFoundError(   # before the heavy work, not in the middle of the preparation
                f"Background removal model not found: {rembg_path}. Put u2net.onnx there "
                "(scripts/link_models.ps1 links it from the Wan2GP folder) or set MY_DIFFUSION_REMBG_MODEL."
            )
        defaults = (EDIT_DEFAULTS if references else KIND_DEFAULTS)[self.kind]
        steps = steps or defaults["steps"]
        guidance_scale = defaults["guidance_scale"] if guidance_scale is None else guidance_scale
        sampler.guided_steps(steps, guidance_interval)   # validate now: a bad interval surfaces before the heavy work
        seed = random.randint(0, MAX_SEED) if seed is None else seed
        src_box = None
        if margins is not None:
            # The frame is the scene grown by the margins; reference 1 goes into src_box inside it.
            width, height, src_box = imaging.outpaint_frame(*references[0].size, margins, resolution)
        elif auto_size:
            width, height = resolve_dimensions_from_size(resolution, *references[0].size)
        else:
            width, height = resolve_dimensions(resolution, aspect_ratio)
        if references:
            tier = EDIT_MAX_TIER[min(len(references), 2)]
            if width * height > _tier_max_pixels(tier):
                raise ValueError(
                    f"an edit with {len(references)} reference(s) at {width}x{height} does not fit the GPU; use resolution {tier} or lower"
                )
        if mask is not None and init_image is None:
            raise ValueError("a mask needs an init image: inpaint regenerates the white part of the mask in a given picture")
        if init_image is not None:
            if strength is None:
                strength = 1.0 if mask is not None else 0.6
            image_steps = sampler.img2img_steps(steps, strength)   # validate strength before the heavy work
        weight = None
        if mask is not None:
            # The mask goes through the same cover_crop as the picture, so both share one frame, then is
            # shrunk to the latent grid (1 = regenerate, 0 = keep). Cheap and on the CPU: done here, so
            # a bad mask or feather surfaces before the heavy work.
            weight = imaging.mask_to_latent(cover_crop(mask, width, height), height // VAE_SCALE, width // VAE_SCALE, feather)
        # Editing: the reference frames, prepared on the CPU now (cheap). In "scene" mode reference 1 gets
        # the output frame itself; the others (all of them in "subjects" mode) keep their proportions and sit centred in the frame, at an
        # offset in tokens. Every side is a multiple of 16, so each latent cuts into whole 2x2 patches.
        cut_seconds = None
        if cut_background:
            t0 = time.time()
            references = [
                background.remove_background(ref, path=rembg_path) if k >= first_subject else ref
                for k, ref in enumerate(references)
            ]
            cut_seconds = time.time() - t0
            logger.info("background removed from %d reference(s): %.1fs", len(references) - first_subject, cut_seconds)
        frames: list[Image.Image] = []
        offsets: list[tuple[int, int]] = []
        for k, ref in enumerate(references):
            if k == 0 and reference_mode == "scene":
                if src_box is None:
                    frame, offset = cover_crop(ref.convert("RGB"), width, height), (0, 0)
                else:
                    # Outpaint: the scene keeps all its pixels, just resized to its box; the margins are generated.
                    box_left, box_top, box_w, box_h = src_box
                    frame = ref.convert("RGB").resize((box_w, box_h), Image.LANCZOS)
                    offset = (box_top // imaging.PATCH_PIXELS, box_left // imaging.PATCH_PIXELS)
            else:
                frame, offset = imaging.fit_reference(ref, width, height)
            frames.append(frame)
            offsets.append(offset)
            logger.debug("reference %d: %dx%d -> frame %dx%d at token offset %s", k + 1, *ref.size, *frame.size, offset)
        # The vision tower sees the same frames as the VAE, shrunk to its size cap.
        vision_frames = [imaging.cap_longest_side(frame, s.vision_max_side) for frame in frames]
        lora_files = [(self._lora(name), weight) for name, weight in loras]   # read in advance: a name error surfaces before the heavy work
        edit_rank = None
        if references:
            lora_files, edit_rank = self._with_edit_lora(lora_files, width, height, len(references))
        timings: dict[str, float] = {}
        if cut_seconds is not None:
            timings["remove_background"] = cut_seconds   # CPU preparation, before "total" starts counting
        t_all = time.time()

        # 1. text (with the reference pictures when editing)
        t0 = time.time()
        enhanced = self._enhance(prompt, seed) if enhance else None
        timings["enhance"] = time.time() - t0
        t0 = time.time()
        use_cfg = guidance_scale != 1.0
        if references:
            # The unconditional context of an edit is "the same pictures + the negative text".
            cond = self._encode_with_images(enhanced or prompt, vision_frames)
            uncond = self._encode_with_images(negative_prompt, vision_frames) if use_cfg else None
        else:
            cond = self._encode(enhanced or prompt)
            uncond = self._encode(negative_prompt) if use_cfg else None
        timings["encode"] = time.time() - t0

        # 1b. image-to-image: the original picture -> latent (VAE first, the network is not on the GPU yet)
        latent0 = None
        if init_image is not None:
            t0 = time.time()
            self.gpu.use("vae")
            latent0 = self.vae.encode(cover_crop(init_image, width, height))
            timings["encode_image"] = time.time() - t0

        # 1c. editing: each reference frame -> clean latent, also in the VAE stage
        refs = None
        if references:
            t0 = time.time()
            self.gpu.use("vae")
            refs = [(self.vae.encode(frame), offset) for frame, offset in zip(frames, offsets)]
            timings["encode_references"] = time.time() - t0

        # 2. noise -> latent
        t0 = time.time()
        self.gpu.use("model")
        # The same LoRA set with the same weights as in the previous request is already attached and sits
        # on the GPU - we do not move it again. We compare the objects themselves (is), not the names:
        # a file changed on disk is read into a new object (see _lora).
        # A change of settings.lora_streaming also rebuilds the set: the pairs then move between RAM and the GPU.
        same = s.lora_streaming == self._applied_streaming and len(lora_files) == len(self._applied_loras) and all(
            new is old and w_new == w_old for (new, w_new), (old, w_old) in zip(lora_files, self._applied_loras)
        )
        if not same:
            apply_loras(self.model, lora_files, s.device, stream=s.lora_streaming)
            self._applied_loras = list(lora_files)
            self._applied_streaming = s.lora_streaming
        timings["load_model"] = time.time() - t0

        lat_h, lat_w = height // VAE_SCALE, width // VAE_SCALE
        tokens = (lat_h // 2) * (lat_w // 2)
        mu = sampler.auto_mu(tokens) if shift is None else shift
        # mu comes from the FULL image size, also in image-to-image: the schedule is the same one text-to-image
        # would use, and image-to-image only runs its tail.
        # In edit mode the noise uses a salted seed. The sampler's noise depends only on the seed and the shape,
        # so an edit with the same seed and size as the picture that was generated would start from that
        # picture's own noise, and the reference-token path then reproduces the reference (measured: a
        # 736x560 fox at seed 42 came back as a near-copy; seed 43 edited correctly). The salt makes the
        # starting noise differ from the one that made the reference, while staying reproducible.
        noise_seed = seed ^ EDIT_NOISE_SALT if references else seed
        noise = sampler.make_noise(noise_seed, LATENT_CHANNELS, lat_h, lat_w).to(s.device)
        post_step = None
        if latent0 is None:
            sigmas = sampler.make_sigmas(steps, mu)
            start = noise
        else:
            latent0 = latent0.to(noise)
            sigmas = sampler.make_sigmas(steps, mu, skip=steps - image_steps)
            start = sampler.add_noise(latent0, noise, sigmas[0])
        # mu above counts only the generated picture's tokens: the references are context, not output.
        if refs is not None:
            refs = [(latent_k.to(noise), offset) for latent_k, offset in refs]
        if weight is not None:
            weight = weight.to(noise)   # float32 on the device of x; broadcasts over the 16 channels

            def post_step(x: torch.Tensor, sigma_next: float) -> torch.Tensor:
                # Inpaint blend (latent blend, as in RePaint): after each step the kept area is replaced
                # by the original picture noised to the level the new x is at. Both latents are in the
                # same normalized space (vae.encode returns it), so mixing them is meaningful.
                # The noise MUST be the same tensor that built `start`. Then the kept area moves along the
                # very same straight line noise -> picture that it started on, and at every step it is
                # exactly where a run that "knew" the picture would be. With fresh noise each step, the
                # kept area would jump between different lines: the network sees inconsistent noise around
                # the mask, the generated area adapts to noise that later vanishes, and seams and blotches
                # appear along the border. After the last step sigma_next = 0, so `known` is latent0 itself.
                known = sampler.add_noise(latent0, noise, sigma_next)
                return weight * x + (1.0 - weight) * known

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        latent = sampler.sample(
            self.model, start, cond.to(s.device), sigmas,
            uncond=None if uncond is None else uncond.to(s.device), guidance_scale=guidance_scale,
            guidance_interval=guidance_interval, post_step=post_step, refs=refs,
        )
        # No extra composite after the loop: the last post_step already puts latent0 back outside the mask
        # (sigma_next = 0). A second blend would apply the feathered weight twice on the soft edge.
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
            "steps": steps, "guidance_scale": guidance_scale, "guidance_interval": list(guidance_interval), "shift": round(mu, 4),
            "checkpoint": self.checkpoint_name,
            "matmul": self.matmul if self.native_layers else "exact",   # the mode the image was actually computed with
            "loras": [{"name": f.name, "weight": w} for f, w in lora_files],
            "mode": "edit" if references else "text2image" if init_image is None else ("inpaint" if mask is not None else "image2image"),
            "strength": None if init_image is None else strength,
            "feather": None if mask is None else feather,
            "references": len(references),
            "reference_mode": reference_mode if references else None,
            "outpaint": None if margins is None else dict(margins._asdict()),
            "remove_background": remove_background if cut_background else "none",
            "edit_lora_rank": edit_rank,   # None without editing or with the edit LoRA switched off
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }
