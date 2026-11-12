"""Builds the ComfyUI API graph (the same JSON that ComfyUI itself sends when you press
Queue Prompt in the UI) for a simple text-to-image flow on Krea2.

The nodes and their inputs were checked against the source of this same ComfyUI build
(D:\\ai\\ComfyUI-torch2.9-cu130-cp313-v1.2\\ComfyUI\\nodes.py,
comfy_extras\\nodes_sd3.py) - no custom nodes are needed: ComfyUI recognizes Krea2
automatically from the checkpoint contents (see
comfy/supported_models.py: Krea2.unet_config); the basic
UNETLoader/CLIPLoader(type="krea2")/VAELoader/KSampler/... nodes are enough."""

from __future__ import annotations

from .loras import LoraSpec


def build_generate_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    guidance_scale: float,
    checkpoint_name: str,
    clip_name: str,
    vae_name: str,
    loras: list[LoraSpec] | None,
    filename_prefix: str = "my_diffusion",
) -> dict:
    loras = loras or []

    graph: dict[str, dict] = {
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": checkpoint_name, "weight_dtype": "default"},
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "krea2", "device": "default"},
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["clip_loader", 0]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["clip_loader", 0]},
        },
        "empty_latent": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
    }

    # LoRA chain between UNETLoader and KSampler - each next node takes
    # MODEL from the previous node (or directly from unet_loader if there are no LoRAs).
    model_link: list = ["unet_loader", 0]
    for i, lora in enumerate(loras):
        node_id = f"lora_{i}"
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_link, "lora_name": lora.path, "strength_model": lora.weight},
        }
        model_link = [node_id, 0]

    # Turbo/distilled checkpoints (guidance_scale=0.0 in our config)
    # trained without classifier-free guidance - cfg=1.0 in KSampler is equivalent
    # "don't run the negative pass at all" (a step is the pure positive prediction
    # velocity). guidance_scale>0 (raw checkpoint) is passed as cfg unchanged.
    cfg = guidance_scale if guidance_scale > 0 else 1.0

    graph["ksampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_link,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "latent_image": ["empty_latent", 0],
            "denoise": 1.0,
        },
    }
    graph["vae_decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["ksampler", 0], "vae": ["vae_loader", 0]},
    }
    graph["save_image"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["vae_decode", 0], "filename_prefix": filename_prefix},
    }

    return graph
