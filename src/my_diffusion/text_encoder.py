"""Qwen3-VL-4B in two roles: the text encoder for Krea 2 and the prompt enhancer.

Role 1 - encoder. Krea 2 does not understand text itself: the prompt is first run through
the Qwen3-VL language model, and the image network receives its internal ("hidden")
states. Not just the last layer, but a slice of 12 layers of different depth:
the early layers carry more about words and spelling, the late ones about meaning.

Role 2 - enhancer. The same model with a "head" (lm_head) can append text,
that is, it can expand a short prompt into a detailed description. Since the model is one,
it is loaded once and does not take up extra video memory.

The weights file is taken from Wan2GP: the weights are compressed to int8 (the quanto library format; we do not
use the library itself, the format is simple), and the file contains only the language part without lm_head.
The head is not needed separately: in the config tie_word_embeddings=true, that is, lm_head is the
same matrix as the token embedding table, and transformers links them (tie_weights).

Pictures in the prompt (reference editing). Qwen3-VL also has a vision tower: a separate network that
turns a picture into embeddings the language model reads in place of `<|image_pad|>` placeholder
tokens. Its weights are a separate file (`*_vision_bf16.safetensors`, 0.83 GB) and are loaded only
when a picture is encoded for the first time (load_vision)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Which hidden states we take. hidden_states[0] - embeddings, hidden_states[k] - output of layer k.
TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Krea 2 was trained with this prompt wrapping (the same as Qwen-Image).
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
MAX_PROMPT_TOKENS = 512
VISION_MAX_SIDE = 768   # pictures larger than this (longer side) are shrunk before the vision tower, as in Wan2GP


class QwenVL:
    def __init__(self, model_dir: Path, pin: bool = False):
        self.model_dir = Path(model_dir)
        self.pin = pin   # keep the compressed weights in pinned memory: copying to the GPU is several times faster
        self.model = None
        self.tokenizer = None
        self.visual = None             # the vision tower; None until load_vision()
        self.image_processor = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Reads the weights into RAM in compressed form. They reach the GPU via to()."""
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer, Qwen3VLForConditionalGeneration

        from .weights import read_safetensors

        candidates = sorted(self.model_dir.glob("*_quanto_*int8.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"no *_quanto_*int8.safetensors file in {self.model_dir}")
        weights_path = candidates[0]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        config = AutoConfig.from_pretrained(self.model_dir)

        tensors, metadata = read_safetensors(weights_path, pin=self.pin)
        if metadata.get("quantization_format") != "quanto":
            raise RuntimeError(f"{weights_path.name}: expected the quanto format, but the header says {metadata.get('quantization_format')!r}")

        # quanto format: instead of "X.weight", a compressed layer stores "X.weight._data" (int8) and
        # "X.weight._scale" (a per-row multiplier); the real weight = _data * _scale.
        # Names in the file lack the "model." prefix - we add it.
        cold: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for key, tensor in tensors.items():
            if key.endswith(".weight._data"):
                cold[f"model.{key[: -len('._data')]}"] = (tensor, tensors[key[: -len("_data")] + "_scale"])
            elif key.endswith(("._scale", ".input_scale", ".output_scale")):
                continue   # the multipliers are already accounted for above; input/output_scale are only needed for activation quantization
            else:
                cold[f"model.{key}"] = (tensor, None)

        # An "empty" model: the structure exists, no memory is allocated for the weights (device "meta").
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)
        model.model.visual = None   # the vision part: we don't feed images in, and its weights are not in the file

        # Strict check: only lm_head may be left without weights (it equals the embedding table).
        expected = set(model.state_dict().keys()) - {"lm_head.weight"}
        if expected != set(cold):
            missing, extra = sorted(expected - set(cold)), sorted(set(cold) - expected)
            raise RuntimeError(f"Qwen weights do not match the model: {len(missing)} missing ({missing[:4]}), {len(extra)} extra ({extra[:4]})")

        self._cold = cold
        self.model = model.eval()
        # Auxiliary model tables that are not in the weights file (for example, the frequencies of positional
        # encoding): the model builds them itself on creation. We remember them - see to("cpu").
        self._built_in = {name: buf for name, buf in model.named_buffers() if name not in cold}
        logger.info("Qwen3-VL loaded from %s", weights_path.name)

    def load_vision(self) -> None:
        """Reads the vision tower weights into RAM (once; later calls do nothing). Call after load().

        The tower is built "empty" (device meta, no memory, no random initialization - creating 0.4
        billion random weights on the CPU took 16-36 s in the probe), the file is checked key by key
        against it, and its tensors are kept in RAM like the language part (pinned if self.pin).
        They reach the GPU through to(), together with the language part. If the language part is
        already on a device, the tower is placed there right away.

        The tower is a separate module (self.visual), not a part of self.model: it is plugged into
        the model only for the duration of a forward pass with pictures (see encode). So the language
        model, its key check and its to("meta") round trip stay exactly as without pictures."""
        if self.visual is not None:
            return
        from accelerate import init_empty_weights
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        from .weights import read_safetensors

        candidates = sorted(self.model_dir.glob("*_vision_bf16.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"no *_vision_bf16.safetensors file in {self.model_dir}")
        weights_path = candidates[0]
        tensors, _ = read_safetensors(weights_path, pin=self.pin)
        # Names in the file carry the "visual." prefix (the tower's place inside the full model).
        cold = {key.removeprefix("visual."): tensor for key, tensor in tensors.items()}

        with init_empty_weights():
            visual = Qwen3VLVisionModel(self.model.config.vision_config)
        expected = set(visual.state_dict().keys())
        if expected != set(cold) or len(cold) != len(tensors):
            missing, extra = sorted(expected - set(cold)), sorted(set(cold) - expected)
            raise RuntimeError(f"vision weights do not match the tower: {len(missing)} missing ({missing[:4]}), {len(extra)} extra ({extra[:4]})")

        self._vision_cold = cold
        # Tables the tower builds itself and that are not in the file (the rotary frequencies inv_freq).
        # init_empty_weights leaves buffers real (on the CPU, float32 - as transformers' own from_pretrained
        # keeps them); we remember them, because to("meta") erases them (see to()).
        self._vision_built_in = {name: buf for name, buf in visual.named_buffers() if name not in cold}
        self.visual = visual.eval()
        # The PIL variant: the default (fast) processor named in preprocessor_config.json needs torchvision.
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(self.model_dir)
        logger.info("Qwen3-VL vision tower loaded from %s (%d tensors)", weights_path.name, len(cold))
        if self.device.type != "meta":
            self._vision_to(self.device)

    def _vision_to(self, device: str | torch.device) -> None:
        """Moves the vision tower: "cpu" drops the device copy (the RAM copy stays in self._vision_cold),
        anything else puts the RAM copy on that device. The same scheme as to() for the language part."""
        if str(device) == "cpu":
            self.visual.to("meta")
            for name, buf in self._vision_built_in.items():
                owner, _, leaf = name.rpartition(".")
                self.visual.get_submodule(owner)._buffers[leaf] = buf
            return
        # encode() may load the tower lazily inside inference_mode; the weights should still be ordinary
        # tensors, not "inference tensors", like the ones to() makes outside it.
        with torch.inference_mode(False):
            state = {name: tensor.to(device, non_blocking=True) for name, tensor in self._vision_cold.items()}
            self.visual.load_state_dict(state, strict=True, assign=True)
            # The built-in tables are not in the state: unlike the language part's rotary tables (which
            # transformers moves to the input's device on every call), the vision rotary table is used
            # as is, so it has to be on the same device as the weights.
            for name, buf in self._vision_built_in.items():
                owner, _, leaf = name.rpartition(".")
                self.visual.get_submodule(owner)._buffers[leaf] = buf.to(device)

    def to(self, device: str) -> None:
        """On the GPU the int8 weights are unpacked into ordinary bf16 once, when loaded
        (4 GB becomes 8 GB). If we unpacked on every layer call, as quanto does,
        text generation would be many times slower: unpacking all 4 billion weights
        would repeat on every new token. In RAM only the compressed copy is kept.

        The vision tower, once loaded (load_vision), travels together with the language part (+0.83 GB)."""
        if self.visual is not None:
            self._vision_to(device)
        if str(device) == "cpu":
            self.model.to("meta")   # drop the GPU copy; the original stays in self._cold
            # to("meta") erases everything, including the tables that are not in self._cold. We restore them,
            # otherwise after reloading the model would crash on the very first call
            # ("Cannot copy out of meta tensor") - that is, on the second new prompt in a row.
            for name, buf in self._built_in.items():
                owner, _, leaf = name.rpartition(".")
                self.model.get_submodule(owner)._buffers[leaf] = buf
            return
        state = {}
        for name, (tensor, scale) in self._cold.items():
            weight = tensor.to(device, non_blocking=True)   # non_blocking works from pinned memory
            if scale is not None:
                weight = weight.to(torch.bfloat16) * scale.to(device, torch.bfloat16)
            state[name] = weight
        self.model.load_state_dict(state, strict=False, assign=True)
        self.model.tie_weights()    # lm_head.weight := the embedding table (the same memory)

    @property
    def device(self) -> torch.device:
        return self.model.lm_head.weight.device

    # ------------------------------------------------------------------ role 1: encoder
    @torch.inference_mode()
    def encode(self, prompt: str, images: Sequence[Image.Image] = ()) -> torch.Tensor:
        """Prompt (and optional pictures) -> (1, tokens, 12, 2560): 12 hidden states for each token.

        Without pictures this is the plain text path, unchanged. With pictures (reference editing),
        each picture goes into the prompt right after the system prefix, in order, as
        `<|vision_start|>` + `<|image_pad|>` x N + `<|vision_end|>`, followed directly by the user's
        text - the token layout Wan2GP's Krea 2 edit mode uses. The pictures must already be the
        prepared frames (the same ones the VAE gets); here they are only capped at 768 px on the longer
        side and given to the image processor. Pass the same pictures for the negative prompt: the
        unconditional context of an edit is "pictures + negative text"."""
        tok = self.tokenizer
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids[:MAX_PROMPT_TOKENS]
        prefix_ids = tok(PROMPT_PREFIX, add_special_tokens=False).input_ids
        suffix_ids = tok(PROMPT_SUFFIX, add_special_tokens=False).input_ids
        if images:
            out = self._forward_with_images(prefix_ids, prompt_ids, suffix_ids, images)
        else:
            input_ids = torch.tensor([prefix_ids + prompt_ids + suffix_ids], device=self.device)
            out = self.model.model.language_model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden = torch.stack([out.hidden_states[k] for k in TAP_LAYERS], dim=2)

        # The system prompt is only needed to "set up" Qwen; to the image network we pass
        # the states starting from the user's text (and up to the end, including "assistant\n").
        # With pictures, the states at the picture tokens stay too: they are how the network "sees" them.
        return hidden[:, len(prefix_ids):]

    def _forward_with_images(self, prefix_ids: list[int], prompt_ids: list[int], suffix_ids: list[int], images: Sequence[Image.Image]):
        """One forward pass of Qwen3-VL over "prefix + pictures + text + suffix"; returns the model output.

        How many placeholders a picture needs: the processor resizes it so that each side is a multiple
        of 32 (bicubic) and cuts it into 16x16 patches, grid (t=1, h, w); the tower then merges every
        2x2 patches into one embedding, so N = h * w / 4 = H' * W' / 1024 (a 736x560 picture -> 414).

        What transformers does for us inside Qwen3VLModel.forward: runs the tower, puts its embeddings
        in place of the placeholders, adds the DeepStack features (from tower layers 5, 11, 17) to the
        language model's states at the picture places after its first layers, and computes the 3D rotary
        positions (M-RoPE: a picture token gets row/column positions; text tokens get one running index)
        - for which it needs mm_token_type_ids (1 at picture tokens). No attention mask: one prompt,
        no padding, causal attention."""
        from .imaging import cap_longest_side

        self.load_vision()
        config = self.model.config
        frames = [cap_longest_side(image.convert("RGB"), VISION_MAX_SIDE) for image in images]
        processed = self.image_processor(images=frames, return_tensors="pt")
        grid = processed["image_grid_thw"]
        merge = config.vision_config.spatial_merge_size
        vision_ids: list[int] = []
        for g in grid:
            vision_ids += [config.vision_start_token_id] + [config.image_token_id] * (int(g.prod()) // merge**2) + [config.vision_end_token_id]

        device = self.device
        input_ids = torch.tensor([prefix_ids + vision_ids + prompt_ids + suffix_ids], device=device)
        inner = self.model.model
        inner.visual = self.visual
        try:
            return inner(
                input_ids=input_ids,
                pixel_values=processed["pixel_values"].to(device, self.visual.dtype),
                image_grid_thw=grid.to(device),
                mm_token_type_ids=(input_ids == config.image_token_id).int(),
                output_hidden_states=True,
                use_cache=False,
            )
        finally:
            # Unplug the tower, and drop the rotary offset transformers caches after a pass with pictures:
            # a stale value would shift the positions in the next chat() (the prompt enhancer).
            inner.visual = None
            inner.rope_deltas = None

    # ------------------------------------------------------------------ role 2: text generation
    @torch.inference_mode()
    def chat(self, system: str, user: str, max_new_tokens: int = 300, temperature: float = 0.7, seed: int | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
        )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
