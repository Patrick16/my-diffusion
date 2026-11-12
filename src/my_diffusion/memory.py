"""Who currently occupies the GPU.

Arithmetic for 16 GB: the Krea 2 network is 12.6 GB, Qwen is 4.1 GB, and the VAE needs several GB of working
buffers during decoding. Together they do not fit, so the rule is simple:
only ONE large model lives on the GPU at any moment; the others stay in RAM.

Switching models must be cheap. A plain module.to("cpu") copies 12.6 GB back
to RAM (~4 s) - that is extra work, since the original is already there. So we keep
a RAM copy permanently (the "home"), send a duplicate to the GPU, and on eviction we simply
discard the duplicate and return the references to the "home". Eviction is instant.

Loading is a copy RAM -> GPU, and its speed depends on which memory the "home" lives in. From ordinary
memory it is ~4 GB/s (the network takes about 3 s), from pinned memory - 26-36 GB/s (the network takes about
0.5 s). Therefore the "home" of all three models is pinned; for what this memory is and what we pay for it - see pinned.py."""

from __future__ import annotations

import logging
import time
from typing import Callable

import torch
import torch.nn as nn

from .pinned import pin_tensors

logger = logging.getLogger(__name__)


class Resident:
    """A model that can be loaded onto the GPU and freed from it."""

    def __init__(self, to_gpu: Callable[[str], None], to_cpu: Callable[[], None]):
        self.to_gpu = to_gpu
        self.to_cpu = to_cpu

    @classmethod
    def plain(cls, module) -> "Resident":
        """A plain .to() transfer - for smaller models, where there is nothing to save."""
        return cls(lambda device: module.to(device), lambda: module.to("cpu"))

    @classmethod
    def with_home_copy(cls, module: nn.Module, pin: bool = False) -> "Resident":
        """Fast variant: the RAM copy stays in place (see the module description).

        pin=True: the "home" must live in pinned memory - transfers to the GPU from it are several times
        faster (see pinned.py). Tensors that are not pinned yet (the checkpoint was read without
        pin, or it is a .gguf) are moved into pinned memory here."""
        home: list[tuple[nn.Module, str, str, torch.Tensor]] = []
        for sub in module.modules():
            for key, param in sub._parameters.items():
                if param is not None:
                    home.append((sub, "param", key, param.data))
            for key, buf in sub._buffers.items():
                if buf is not None:
                    home.append((sub, "buffer", key, buf))
        if pin:
            pinned = pin_tensors([tensor for _, _, _, tensor in home])
            home = [(sub, kind, key, tensor) for (sub, kind, key, _), tensor in zip(home, pinned)]

        def place(tensors) -> None:
            for (sub, kind, key, _), tensor in zip(home, tensors):
                if kind == "param":
                    sub._parameters[key].data = tensor
                else:
                    sub._buffers[key] = tensor

        if pin:
            place([t for _, _, _, t in home])   # the module now points at the pinned copies
        return cls(
            to_gpu=lambda device: place([t.to(device, non_blocking=True) for _, _, _, t in home]),
            to_cpu=lambda: place([t for _, _, _, t in home]),
        )


class GpuSlot:
    def __init__(self, device: str):
        self.device = device
        self.residents: dict[str, Resident] = {}
        self.active: str | None = None

    def register(self, name: str, resident: Resident) -> None:
        self.residents[name] = resident

    def use(self, name: str) -> None:
        """Make model `name` active on the GPU, evicting the previous one."""
        if self.active == name or self.device == "cpu":
            return
        t0 = time.time()
        self.release()
        self.residents[name].to_gpu(self.device)
        torch.cuda.synchronize()
        self.active = name
        logger.info("video card: %s (%.1fs, %.1f GB in use)", name, time.time() - t0, torch.cuda.memory_allocated() / 2**30)

    def release(self) -> None:
        if self.active is not None:
            self.residents[self.active].to_cpu()
            self.active = None
            torch.cuda.empty_cache()
