"""VAE: converts the latent into pixels (decode) and pixels into the latent (encode).

The Krea 2 network works not with pixels but with a "latent" - an image compressed 8 times
per side, with 16 channels instead of 3. That is 12 times fewer numbers, so generation
fits into a reasonable time. The VAE decoder expands the latent back.

Krea 2 uses the same VAE as Qwen-Image and Wan 2.1 (it is a video VAE, so the tensors
have an extra time axis of length 1). We take the ready implementation from diffusers - it is
the only thing we use from there.

Normalization: the network was trained on latents normalized per channel to zero
mean and unit spread. Before decoding this has to be undone:
z = latent * std + mean. A forgotten normalization is a common cause of "mush" in the output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

VAE_SCALE = 8          # how many times the latent is smaller than the image along each side
LATENT_CHANNELS = 16
FP32_MAX_PIXELS = 1_300_000   # above this size we decode in bf16 (see LatentDecoder._dtype_for)


class LatentDecoder:
    def __init__(self, weights_path: Path):
        self.weights_path = Path(weights_path)
        self.vae = None
        self.mean = None
        self.std = None

    def load(self) -> None:
        from diffusers import AutoencoderKLQwenImage

        config_path = self.weights_path.with_name(self.weights_path.stem + "_config.json")
        config = {k: v for k, v in json.loads(config_path.read_text(encoding="utf-8")).items() if not k.startswith("_")}

        vae = AutoencoderKLQwenImage(**config)
        vae.load_state_dict(load_file(str(self.weights_path)), strict=True)
        self.vae = vae.eval().requires_grad_(False)
        self.mean = torch.tensor(config["latents_mean"]).view(1, LATENT_CHANNELS, 1, 1, 1)
        self.std = torch.tensor(config["latents_std"]).view(1, LATENT_CHANNELS, 1, 1, 1)

    def to(self, device: str) -> None:
        self.vae.to(device)

    @staticmethod
    def _dtype_for(pixels: int) -> torch.dtype:
        """The VAE itself weighs little, but during decoding it holds many intermediate
        "images" at full resolution in memory. A measurement at 1920x1088: in float32 it needs ~16 GB
        (it does not fit, Windows starts paging, 53 s); in bf16 - 8.5 GB and 0.9 s, with a difference
        of 0.36 on average out of 255. Therefore: where float32 fits with room to spare, we use it;
        for large sizes we switch to half precision."""
        return torch.float32 if pixels <= FP32_MAX_PIXELS else torch.bfloat16

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> latent (1, 16, H/8, W/8) in the model's normalized space, float32.

        The exact inverse of `decode`: the VAE encoder gives a distribution over latents, we take its
        mode (the most likely value, no random noise, so the same picture always gives the same latent),
        then apply the opposite normalization: (z - mean) / std. This is what lets an existing picture
        enter the sampler as if it were a finished generation (image-to-image, inpaint).
        The image sides must be multiples of 8 (use `imaging.cover_crop` with a resolution from the
        resolution table, which already guarantees this)."""
        width, height = image.size
        if width % VAE_SCALE or height % VAE_SCALE:
            raise ValueError(f"image size {width}x{height} must be a multiple of {VAE_SCALE}")
        device = next(self.vae.parameters()).device
        dtype = self._dtype_for(width * height)
        self.vae.to(dtype)
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        pixels = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1   # (1, 3, H, W), -1..1
        pixels = pixels.to(device, dtype).unsqueeze(2)                                                # time axis: (1, 3, 1, H, W)
        z = self.vae.encode(pixels).latent_dist.mode().float()                                        # (1, 16, 1, h, w)
        z = (z - self.mean.to(device)) / self.std.to(device)
        return z[:, :, 0]

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """latent: (1, 16, h, w) in the model's normalized space -> PIL image."""
        device = next(self.vae.parameters()).device
        dtype = self._dtype_for(latent.shape[-1] * latent.shape[-2] * VAE_SCALE ** 2)
        self.vae.to(dtype)
        z = latent.to(device, torch.float32).unsqueeze(2)                 # added the time axis: (1, 16, 1, h, w)
        z = z * self.std.to(device) + self.mean.to(device)
        pixels = self.vae.decode(z.to(dtype)).sample[:, :, 0].float()     # (1, 3, H, W), values -1..1
        pixels = ((pixels.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return Image.fromarray(pixels[0].permute(1, 2, 0).cpu().numpy())
