"""Step 2 check: Qwen3-VL loads and works in both roles.

Generating meaningful text is a good test of the weight loading: if the weights had landed
incorrectly, the model would produce an incoherent set of tokens.

    uv run python scripts/check_text_encoder.py"""

from __future__ import annotations

import time

import torch

from my_diffusion.config import Settings
from my_diffusion.text_encoder import PROMPT_PREFIX, QwenVL


def vram() -> str:
    return f"{torch.cuda.memory_allocated() / 2**30:.2f} GB"


def main() -> None:
    settings = Settings()
    qwen = QwenVL(settings.text_encoder_dir)

    t0 = time.time()
    qwen.load()
    print(f"loaded into RAM: {time.time() - t0:.1f}s")
    t0 = time.time()
    qwen.to(settings.device)
    print(f"moved to {settings.device}: {time.time() - t0:.1f}s, VRAM {vram()}")

    n_prefix = len(qwen.tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids)
    print(f"tokens in the service prefix: {n_prefix} (Wan2GP hard-codes 34)")

    prompt = "a fox walking in fresh snow at golden hour, natural light, detailed fur"
    t0 = time.time()
    cond = qwen.encode(prompt)
    print(f"encode: {tuple(cond.shape)} {cond.dtype}, {time.time() - t0:.2f}s, "
          f"mean {cond.float().mean().item():.4f}, std {cond.float().std().item():.3f}, nan={bool(cond.isnan().any())}")
    print(f"  std over 12 layers: {[round(v, 2) for v in cond.float().std(dim=(0, 1, 3)).tolist()]}")

    # The model must survive eviction from the GPU and coming back: this happens on every
    # new prompt after the first generation. The result must match the first one bit for bit.
    qwen.to("cpu")
    qwen.to(settings.device)
    again = qwen.encode(prompt)
    print(f"after eviction and reloading on {settings.device}: encode matches the first one: {torch.equal(cond, again)}")
    assert torch.equal(cond, again), "after returning to the GPU, Qwen encodes differently"

    t0 = time.time()
    answer = qwen.chat("You are a helpful assistant.", "Describe a fox in the snow in one sentence.", max_new_tokens=60, temperature=0.0)
    print(f"chat ({time.time() - t0:.1f}s): {answer}")


if __name__ == "__main__":
    main()
