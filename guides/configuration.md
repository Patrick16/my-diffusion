# Configuration

Settings are read from environment variables `MY_DIFFUSION_*` or from a `.env` file in the project
root. Copy `.env.example` to `.env` and adjust it. Every variable is optional: without `.env` the
defaults from `src/my_diffusion/config.py` are used.

Most settings can also be changed while the server runs, through the admin API: see
[Changing settings at run time](#changing-settings-at-run-time). The column **Change** in the tables says
how such a change takes effect: `live` - at once; `reload` - the models are read again; `restart` - at the
next start of the server; `-` - not changeable through the API, only here.

## Paths and devices

| Variable | Default | Change | Meaning |
|---|---|---|---|
| `MY_DIFFUSION_DEVICE` | `cuda` | `restart` | torch device |
| `MY_DIFFUSION_MODELS_DIR` | `./models` | `reload` | folder with links to the models (created by `scripts/link_models.ps1`): `vae/`, `text_encoders/`, `rembg/` and, unless set separately, `checkpoints/` and `loras/` |
| `MY_DIFFUSION_CHECKPOINTS_DIR` | unset | `reload` | folder with the checkpoints; unset = `checkpoints/` under the models folder |
| `MY_DIFFUSION_LORAS_DIR` | unset | `live` | folder with the LoRAs; unset = `loras/` under the models folder |
| `MY_DIFFUSION_OUTPUT_DIR` | `./outputs` | `-` | where generated PNGs are saved. Pictures are found in it by name, so changing it hides the old ones |
| `MY_DIFFUSION_FS_ROOTS` | empty | `-` | JSON list of extra folders the admin API may show and accept as model folders, e.g. `["D:\\ComfyUI\\models"]` (see [Choosing folders](#choosing-folders-get-fs)) |
| `MY_DIFFUSION_SETTINGS_FILE` | `./data/settings.override.json` | `-` | where settings changed through the API are saved |

## Models

| Variable | Default | Change | Meaning |
|---|---|---|---|
| `MY_DIFFUSION_CHECKPOINT` | `krea2_turbo_mxfp8.safetensors` | `reload` | the checkpoint loaded at start: a file in the checkpoints folder (`.safetensors` or `.gguf`) |
| `MY_DIFFUSION_CHECKPOINT_KIND` | `auto` | `live` | `auto`, `turbo` or `raw`; sets the default steps and guidance (see [checkpoints](checkpoints.md)) |
| `MY_DIFFUSION_VAE` | `qwen_vae.safetensors` | `reload` | file in `models/vae` |
| `MY_DIFFUSION_TEXT_ENCODER` | `Qwen3-VL-4B-Instruct` | `reload` | folder in `models/text_encoders` |
| `MY_DIFFUSION_ENHANCER_MAX_NEW_TOKENS` | `220` | `live` | length limit of the prompt enhancer's output |

## Performance

| Variable | Default | Change | Meaning |
|---|---|---|---|
| `MY_DIFFUSION_MATMUL` | `fp8_all` | `live` | `exact`, `fp8` or `fp8_all`; fast modes need mxfp8 checkpoints and an NVIDIA 50-series GPU, otherwise use `exact` |
| `MY_DIFFUSION_ATTENTION` | `cudnn` | `live` | `cudnn`, `default` (the torch choice, for comparison with ComfyUI) or `sage2` (optional approximate kernel, faster on large pictures; needs `uv sync --group sage2`) |
| `MY_DIFFUSION_ATTENTION_SAGE2_MIN_TOKENS` | `6000` | `live` | only with `sage2`: shorter sequences still use `cudnn` (480 and 720 text-to-image stay unchanged); `0` = `sage2` everywhere |
| `MY_DIFFUSION_PIN_MEMORY` | `true` | `reload` | keep weights in pinned RAM (faster model switching, ~19 GB of RAM); set `false` if RAM is short |
| `MY_DIFFUSION_LORA_STREAMING` | `true` | `live` | stream LoRA pairs to the GPU block by block (less VRAM, slightly slower) |

Details and measurements: [performance](performance.md).

## Jobs and access

| Variable | Default | Change | Meaning |
|---|---|---|---|
| `MY_DIFFUSION_DEVICES` | unset | `restart` | GPUs that run jobs: unset = `MY_DIFFUSION_DEVICE` only; `auto` = every CUDA device; or a list `["cuda:0","cuda:1"]`. Each GPU holds its own copy of the models (~19 GB RAM) |
| `MY_DIFFUSION_DATABASE_PATH` | `./data/jobs.sqlite3` | `-` | the job database (SQLite) |
| `MY_DIFFUSION_INPUTS_DIR` | `./data/inputs` | `-` | pictures sent with jobs, kept after the job for `MY_DIFFUSION_INPUTS_KEEP_DAYS` |
| `MY_DIFFUSION_INPUTS_KEEP_DAYS` | `7` | `live` | days to keep a job's input pictures (`0` = delete when the job ends) |
| `MY_DIFFUSION_QUEUE_MAX_PENDING` | `16` | `live` | jobs waiting at once, all users together; beyond that `429` |
| `MY_DIFFUSION_QUEUE_MAX_PER_USER` | `3` | `live` | jobs waiting or running per user; admins are exempt |
| `MY_DIFFUSION_API_KEYS` | empty | `-` | JSON `{"client": "key"}`, keys of 16+ characters. Empty = access control off, everyone is an admin |
| `MY_DIFFUSION_ADMIN_CLIENTS` | empty | `-` | JSON list of the clients whose key gives admin rights |
| `MY_DIFFUSION_API_DOCS` | `true` | `restart` | serve the API documentation (`/docs`, `/scalar`, `/redoc`, `/openapi.json`); `false` removes it |

Details: [jobs, queue and access](job-queue.md).

## Reference editing and background removal

| Variable | Default | Change | Meaning |
|---|---|---|---|
| `MY_DIFFUSION_EDIT_LORA` | `krea2_identity_edit_v1_2` | `live` | edit LoRA in the LoRA folder, added automatically for references |
| `MY_DIFFUSION_EDIT_LORA_WEIGHT` | `1.0` | `live` | its weight; a request can name the LoRA itself, weight 0 switches it off |
| `MY_DIFFUSION_VISION_MAX_SIDE` | `768` | `live` | longer side of a picture before the vision tower |
| `MY_DIFFUSION_MAX_REFERENCES` | `2` | `live` | maximum number of reference pictures |
| `MY_DIFFUSION_REMBG_MODEL` | `rembg/u2net.onnx` | `live` | background-removal model in `models/` |

## Server

| Variable | Default | Change | Meaning |
|---|---|---|---|
| `MY_DIFFUSION_HOST` | `127.0.0.1` | `-` | bind address, for reference: the server takes it from the `uvicorn` command line (`--host`) |
| `MY_DIFFUSION_PORT` | `8000` | `-` | port, for reference: `uvicorn --port` |

## Example `.env`

```bash
MY_DIFFUSION_CHECKPOINT=krea2_turbo_mxfp8.safetensors
MY_DIFFUSION_MATMUL=fp8_all
MY_DIFFUSION_ATTENTION=cudnn
MY_DIFFUSION_PIN_MEMORY=true
```

## Changing settings at run time

An admin client (see [access](job-queue.md#access); without API keys every caller is an admin) can read and
change the settings while the server runs. This is what a Settings page of a web UI talks to.

| Endpoint | What it does |
|---|---|
| `GET /config` | every setting with its value, default, where the value comes from, its type and limits, and how a change takes effect |
| `PATCH /config` | change settings; the change is applied, saved, and survives a restart |
| `GET /fs` | folders on the server, for choosing the model folders |

### How a change takes effect

Every setting has one of three classes (`apply` in `GET /config`, the **Change** column above):

| Class | Meaning |
|---|---|
| `live` | In effect when the call answers; the models are not reloaded. Settings the engine reads (matmul, attention, the LoRA folder, the edit LoRA ...) are switched between two generations: a running generation always ends with the settings it started with, and the call waits for it. Queue limits and the keeping time of input pictures change instantly |
| `reload_models` | The models are read again on every GPU: only the checkpoint for `checkpoint` and `checkpoints_dir`, all three models for `vae`, `text_encoder`, `pin_memory` and `models_dir`. Allowed only while no job waits or runs; otherwise `409` and nothing changes |
| `restart` | Only saved. The server keeps running with the old value until it is restarted; `GET /config` shows `pending_restart: true` and the saved value in `saved` |

Settings marked `-` cannot be changed through the API at all: the API keys and the list of admin clients
(secrets and access rights stay in the environment; the keys are never shown either, only the client names),
the folders where existing pictures and jobs are found (`output_dir`, `inputs_dir`, `database_path`), and the
API's own limits (`fs_roots`, `settings_file`).

### `PATCH /config`

```json
{"set": {"matmul": "exact", "queue_max_pending": 32}, "reset": ["attention"], "dry_run": false}
```

- `set` - new values by setting name (the names of `GET /config`: the variable without `MY_DIFFUSION_`, in
  lower case).
- `reset` - remove the saved value: the setting falls back to the environment, `.env` or the default.
- `dry_run: true` - only check the request and report what it would do.

The request is all or nothing. It is checked as a whole first: types and limits, that folders exist and lie
inside the allowed roots, that the named model files exist, that the `sage2` package is installed, that the
devices exist. Anything wrong answers `422` with a message per setting, and nothing has changed:

```json
{"detail": "the settings were not changed: queue_max_pending: Input should be greater than or equal to 1",
 "errors": {"queue_max_pending": "Input should be greater than or equal to 1"}}
```

If a checked change still cannot be applied (for example `fp8` on a GPU without native fp8), everything is put
back and the answer is `400`.

The answer of a successful call:

```json
{"applied": ["matmul", "queue_max_pending"], "restart_required": [], "unchanged": ["attention"],
 "reloaded": "none", "warnings": [], "dry_run": false, "config": {"settings": ["..."]}}
```

`applied` is in effect now, `restart_required` is saved and waits for a restart, `reloaded` tells what was
read again (`none`, `checkpoint` or `all`), and `config` is the new `GET /config`.

Notes:

- `checkpoint` is the checkpoint loaded at start. Setting it loads that checkpoint now and keeps it for the
  next start; `POST /checkpoints/load` only switches the checkpoint until the next restart.
- A new `checkpoints_dir` must contain the `checkpoint`; name both in one request to move to another folder.
- A new `loras_dir` without the edit LoRA is accepted with a warning: only reference editing needs it.
- If the models failed to load at start (`/health` says `error`, for example a wrong checkpoint name), a
  `PATCH /config` retries the load with the new settings, so the server can be repaired without a restart.

### Where changes are saved

In `data/settings.override.json` (`MY_DIFFUSION_SETTINGS_FILE`), a flat JSON object such as
`{"matmul": "exact", "queue_max_pending": 32}`. Your `.env` is never written. The order of priority:

1. the override file,
2. environment variables,
3. `.env`,
4. the defaults.

The file is the strongest, so that a value in `.env` does not silently undo what was saved on the Settings
page; `reset` (or deleting the key from the file) gives the setting back to the environment. `source` in
`GET /config` says which of the four a value comes from.

Only settings the API may change are read from the file. A broken file does not stop the server: entries that
cannot be used are skipped, logged, and listed in `problems` of `GET /config`.

If the server does not start after a `restart` setting was changed (for example `devices` naming a GPU that
is gone), delete that key from the file, or the whole file.

### Choosing folders: `GET /fs`

`GET /fs` lists the allowed roots; `GET /fs?path=D:\ComfyUI\models` lists the subfolders of that folder:

```json
{"path": "D:\\ComfyUI\\models", "parent": null,
 "dirs": [{"name": "checkpoints", "path": "D:\\ComfyUI\\models\\checkpoints"}],
 "model_files": 0, "truncated": false, "roots": ["D:\\ComfyUI\\models", "E:\\my-diffusion\\models"]}
```

Only folders are listed; `model_files` counts the `.safetensors` / `.gguf` files lying directly in the folder.

The server does not show its whole file system, even to an admin. Browsing, and the folders accepted for
`models_dir`, `checkpoints_dir` and `loras_dir`, are limited to the **allowed roots**:

- the model folders in use (`models_dir` and the current checkpoints and LoRA folders), and
- the folders listed in `MY_DIFFUSION_FS_ROOTS`.

So out of the box only the model folders can be browsed. To choose folders elsewhere, add their parent to
`.env` and restart - a whole drive if you like:

```bash
MY_DIFFUSION_FS_ROOTS=["D:\\ComfyUI\\models", "D:\\AI"]
```

`fs_roots` cannot be changed through the API itself. Paths are compared by their real location: `..`,
symbolic links and junctions are followed first, so a link inside a root cannot lead out of the roots, and
such links are not listed. Outside the roots the answer is `403` whether the folder exists or not.

### Examples (PowerShell)

```powershell
$headers = @{ Authorization = "Bearer <admin key>" }   # leave out without configured keys

# what can be changed, and how
(Invoke-RestMethod http://127.0.0.1:8000/config -Headers $headers).settings |
    Format-Table name, value, apply, editable, source

# change two settings
$body = @{ set = @{ matmul = "exact"; queue_max_pending = 32 } } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/config -Method Patch -Headers $headers -ContentType application/json -Body $body

# move the checkpoints to another folder (it must lie inside the allowed roots)
$body = @{ set = @{ checkpoints_dir = "D:\ComfyUI\models\checkpoints"; checkpoint = "krea2_turbo_mxfp8.safetensors" } } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/config -Method Patch -Headers $headers -ContentType application/json -Body $body
```
