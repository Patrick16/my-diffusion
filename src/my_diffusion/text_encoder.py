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
same matrix as the token embedding table, and transformers links them (tie_weights)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

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


class QwenVL:
    def __init__(self, model_dir: Path, pin: bool = False):
        self.model_dir = Path(model_dir)
        self.pin = pin   # keep the compressed weights in pinned memory: copying to the GPU is several times faster
        self.model = None
        self.tokenizer = None

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

    def to(self, device: str) -> None:
        """On the GPU the int8 weights are unpacked into ordinary bf16 once, when loaded
        (4 GB becomes 8 GB). If we unpacked on every layer call, as quanto does,
        text generation would be many times slower: unpacking all 4 billion weights
        would repeat on every new token. In RAM only the compressed copy is kept."""
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
    def encode(self, prompt: str) -> torch.Tensor:
        """Prompt -> (1, tokens, 12, 2560): 12 hidden states for each token."""
        tok = self.tokenizer
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids[:MAX_PROMPT_TOKENS]
        prefix_ids = tok(PROMPT_PREFIX, add_special_tokens=False).input_ids
        suffix_ids = tok(PROMPT_SUFFIX, add_special_tokens=False).input_ids
        input_ids = torch.tensor([prefix_ids + prompt_ids + suffix_ids], device=self.device)

        out = self.model.model.language_model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden = torch.stack([out.hidden_states[k] for k in TAP_LAYERS], dim=2)

        # The system prompt is only needed to "set up" Qwen; to the image network we pass
        # the states starting from the user's text (and up to the end, including "assistant\n").
        return hidden[:, len(prefix_ids):]

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
