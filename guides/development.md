# Development and verification

## Tests

```bash
uv run pytest
```

The tests need no GPU and no models. For the job system: `test_store.py` (the SQLite store), `test_jobs.py`
(the scheduler: fairness, limits, several slots, cancelling, restarts; plain Python stands in for a generation),
`test_auth.py` (keys and owners), `test_api_jobs.py` (the HTTP flow and who sees what, with a stub engine),
`test_sampler_cancel.py` and `test_enhancer_cancel.py` (where a running generation stops). See
[jobs, queue and access](job-queue.md).

## Verification scripts

Every stage of building the engine is verified separately - handy if something breaks after updating
the libraries or on another machine. Scripts that need the GPU and the models are run by hand.

| Script | What it confirms |
|---|---|
| `scripts/check_weights.py` | the checkpoint loads into the model in strict mode |
| `scripts/check_text_encoder.py` | Qwen is loaded correctly: it encodes the prompt and writes coherent text |
| `scripts/smoke_test.py` | the whole path "by hand", linearly, without the engine |
| `scripts/check_loras.py` | which LoRAs are read, which are not, and why |
| `scripts/check_checkpoints.py` | every checkpoint produces an image |
| `scripts/check_outpaint.py` | the three reference features on the GPU (saves pictures to `outputs/`) |
| `scripts/check_memory.py`, `check_vae_sizes.py`, `check_reference_cost.py` | video memory and time of the large sizes (edits, outpaint, 1440p) stage by stage, with streamed weights; the VAE whole and tiled up to 4K; cheaper reference tokens (an experiment) |
| `scripts/check_keep_original.py`, `check_background.py`, `check_edit.py`, `check_rank.py`, `check_vae_roundtrip.py` | keep-original outpainting, background removal, reference editing, LoRA rank, VAE round-trip floor |
| `scripts/check_inpaint_iterations.py` | inpaint with `inpaint_iterations` against the plain inpaint, side-by-side sheets |
| `scripts/profile_step.py` | where the step time goes; with `--real` - a comparison of the multiplication modes on the real checkpoint |
| `scripts/compare_checkpoints.py` | how the checkpoints differ from the base one layer by layer, and whether the difference is low-rank |
| `scripts/link_models.ps1` | creates the `models/` links (see [README](../README.md#models)) |

## Legacy code

`legacy/comfy_wrapper/` holds the earlier ComfyUI-based wrapper. It is not used by the engine.

## Not implemented yet

- LoKr/LoHa LoRA files
- other Qwen models for the prompt enhancer (Qwen3.5-4B-Abliterated, qwen3_8b): links to them in
  `models/text_encoders` are created, but the loader currently supports only Qwen3-VL-4B
