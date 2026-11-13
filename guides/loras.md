# LoRA

LoRA files go into `models/loras` (subfolders are allowed). In a request:

```json
"loras": [{"name": "KreaAmateur_V2", "weight": 0.8}, {"name": "sliders/zoom_krea2_loraholic", "weight": -1.0}]
```

CLI: `--lora KreaAmateur_V2:0.8 --lora sliders/zoom_krea2_loraholic:-1.0`.

- `name` is the path relative to `models/loras`; the extension can be omitted.
- `weight` is the strength: 1.0 as trained, a negative value reverses the effect.
- The LoRA set and its weights can be changed between requests without reloading the model: a LoRA
  is not merged into the weights, it is computed alongside as a small addition.

To check which files are read and why: `uv run python scripts/check_loras.py`.

Not supported: LoKr/LoHa and ControlNet-like files (the loader says so directly).

The edit LoRA used by reference editing (`krea2_identity_edit_v1_2`) is handled separately; see
[image modes](image-modes.md) and the memory notes in [performance](performance.md).
