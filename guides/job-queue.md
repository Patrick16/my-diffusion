# Jobs, queue and access

Every generation is a job. `POST /generate` puts it in a queue and answers at once with the job id; a background
scheduler runs the waiting jobs on the GPU; the client polls for the result and fetches the picture. A job can be
cancelled while it waits or while it runs. Jobs are stored in SQLite, so the queue survives a restart.

## The flow

| Step | Request | Answer |
|---|---|---|
| Submit | `POST /generate` (same body as before) | `202` with the job: `job_id`, `status`, `position` |
| Poll | `GET /result/{job_id}` | the job: status, position, progress (`step` / `total_steps`), `result`, `image_url`, `warnings`, `error` |
| Watch | `GET /result/{job_id}/events` | the same job, pushed as server-sent events (see below) |
| Inputs | `GET /result/{job_id}/inputs/{name}` | a picture sent with the job (`init`, `mask`, `ref1`, `ref2`), for 7 days; the names are listed in `inputs` of the job |
| Fetch | `GET /result/{job_id}/image` | the PNG, once the status is `done` (`409` before) |
| Cancel | `POST /cancel/{job_id}` | the job; `409` if it has already finished |
| List | `GET /jobs?limit=50` | your jobs, newest first; admins see everybody's (or `?owner=telegram:123`) |
| Delete | `POST /jobs/delete` with `{"ids": [...]}` (up to 500) | finished jobs are removed from the database with their result and input pictures; always `200` with `deleted`, `active` (still waiting or running: skipped, cancel first) and `not_found` (missing or someone else's) |

A request that is malformed (bad base64, `mask` without `init_image` ...) gets `400` at once and is not queued.
Problems only the engine can see (an unknown LoRA, a size refused by `edit_limits: strict`, a GPU out of memory) make the job `failed`, with
the reason in `error`. A full queue answers `429`.

## Events instead of polling

`GET /result/{job_id}/events` keeps the connection open and pushes the job whenever something changes: its
position in the queue, a sampler step, `cancel_requested`, the end. Polling `GET /result/{id}` still works, and
a simple client may keep using it.

```
event: update
data: {"job_id": "...", "status": "running", "step": 3, "total_steps": 8, ...}

event: done
data: {"job_id": "...", "status": "done", "image_url": "/result/.../image", ...}
```

- The first event is the job as it is now, so a client that connects late (or reconnects) loses nothing.
- The data of every event is the same JSON as `GET /result/{id}`. Events named `update` leave the stream open;
  the last is named after the final status (`done`, `failed`, `cancelled`) and the server then closes it.
- Quick changes merge: a client sees the latest state, not every single step.
- A `: keepalive` comment line comes after every 15 s of silence.
- The same access rules apply as for the status (someone else's job is `404`).
- The browser's `EventSource` cannot send an `Authorization` header. A web client reads the stream with `fetch`
  (`response.body.getReader()`); a program uses any streaming HTTP client (for example `httpx.stream`).
- An open stream holds the server's shutdown until it ends (uvicorn waits for connections). Start the server with
  `--timeout-graceful-shutdown 5` (as in the commands in this guide set) so Ctrl+C closes streams after 5 s.

```bash
curl -N -H "Authorization: Bearer <key>" http://127.0.0.1:8000/result/<job id>/events
```

## Job statuses

```
queued -> running -> done | failed | cancelled
queued -> cancelled               (cancelled while waiting: never touches a GPU)
```

| Status | Meaning |
|---|---|
| `queued` | Waiting. `position` 1 is the next job to start |
| `running` | Being generated. `step` / `total_steps` show the sampler's progress. `cancel_requested: true` means it is stopping |
| `done` | Finished; `result` has the size, seed, timings ...; the picture is at `image_url`. `warnings` (also in `result`) is empty unless the job is unusual: for example a large edit accepted by `edit_limits: warn`, with about how long it takes; `result.streamed_blocks` is how many network blocks were streamed to fit the memory |
| `failed` | `error` says why. Also used for a job that was running when the server crashed |
| `cancelled` | Stopped by a cancel. No picture is kept |

## Cancelling

A waiting job is cancelled at once. A running one cannot be killed from outside (stopping a thread in the middle
of a CUDA kernel would leave the model in an unknown state), so cancellation is cooperative: the job gets
`cancel_requested: true`, and the engine stops at its next safe point: before every network step, after every
token of the prompt enhancer, and between the big stages. The delay is at most one step. The final VAE decode is
short and is not interrupted, so a cancel that arrives during it leaves the job `done`.

## Access

The API is called by client programs, not by end users directly. Each client has its own key, sent as
`Authorization: Bearer <key>`:

- an **admin** client (for example the web UI) sees and cancels every job;
- a **service** client (for example a Telegram bot) acts for an end user it names in the `X-User-Id` header and
  sees only that user's jobs. The server trusts that header because only the bot holds its key.

A job belongs to `<client>:<user>`, for example `telegram:123456789`. For anyone else the job does not exist:
status, picture and cancel all answer `404` (not `403`, so job ids cannot be probed). There is no public folder
of pictures any more; every picture goes through this check. `GET /health` and `GET /resolutions` need no key;
switching the checkpoint needs an admin.

```bash
MY_DIFFUSION_API_KEYS='{"web": "<long random key>", "telegram": "<another long random key>"}'
MY_DIFFUSION_ADMIN_CLIENTS='["web"]'
```

**Without keys, access control is off** and every caller is an admin (`local:admin`, or `local:<X-User-Id>`).
That keeps the one-person setup on `127.0.0.1` working without configuration. Do not expose such a server.

## Fairness and limits

The queue goes round the users instead of being strictly first come, first served: each turn takes the oldest job
of the user who was served longest ago, and one user's jobs keep their own order. So a user who sends ten
requests does not make everyone else wait for all ten. With a single user it is plain first in, first out.

| Variable | Default | Meaning |
|---|---|---|
| `MY_DIFFUSION_QUEUE_MAX_PENDING` | `16` | jobs waiting at once, all users together; beyond that `429` |
| `MY_DIFFUSION_QUEUE_MAX_PER_USER` | `3` | jobs waiting or running per user; admins are exempt |

The limits can be changed while the server runs, by an admin: `PATCH /config` with
`{"set": {"queue_max_per_user": 5}}` (see [configuration](configuration.md#changing-settings-at-run-time)).
Jobs already waiting stay in the queue.

## Several GPUs

A GPU runs one generation at a time: one generation fills it (up to about 14 GB of VRAM at 1080p), so two at once
would each run at half speed and risk running out of memory. More throughput therefore comes from more GPUs. The
scheduler has one **slot** per device, each with its own engine, its own copy of the models and its own worker
thread; free slots take the next jobs in the fair order.

| `MY_DIFFUSION_DEVICES` | Slots |
|---|---|
| unset (default) | one, on `MY_DIFFUSION_DEVICE` |
| `auto` | one per visible CUDA device |
| `["cuda:0", "cuda:1"]` | exactly these |

Each extra slot costs another copy of the weights in RAM (about 19 GB with pinned memory). Every slot uses the
same checkpoint; `POST /checkpoints/load` switches all of them. Multi-GPU operation is written but has not been
run on a multi-GPU machine yet.

Run the server as **one process** (`uvicorn` without `--workers`): every process would load its own models.

## Restarts

- Waiting jobs stay in the database and run after the restart.
- On a clean shutdown (Ctrl+C), running jobs are asked to stop and are put back in the queue, so they run again.
  A job the user had already cancelled ends as `cancelled`.
- After a crash, jobs found `running` are marked `failed` ("interrupted"), not retried: if a job is what killed
  the server, retrying it would kill it again.

## Storage

| Path (setting) | What |
|---|---|
| `data/jobs.sqlite3` (`MY_DIFFUSION_DATABASE_PATH`) | the jobs: owner, times, status, request parameters, result, error |
| `data/inputs/<job id>/` (`MY_DIFFUSION_INPUTS_DIR`) | pictures sent with the job, as `<job id>_init`, `_mask`, `_ref1`, `_ref2` plus the extension of the format. Kept after the job (also a failed or cancelled one) for `MY_DIFFUSION_INPUTS_KEEP_DAYS` days (default 7; `0` = delete as soon as the job ends), then deleted by an hourly clean-up. The pictures of a waiting or running job are never deleted |
| `outputs/<job id>.png` (`MY_DIFFUSION_OUTPUT_DIR`) | finished pictures |

SQLite gives full ACID transactions; the database is in WAL mode with `synchronous=FULL`, so a submitted job
survives a power cut. Finished jobs and result pictures are not deleted automatically yet; input pictures expire after 7 days.

## Examples (PowerShell)

```powershell
$headers = @{ Authorization = "Bearer <key>"; "X-User-Id" = "123456789" }
$body = @{ prompt = "a fox in the snow, golden hour"; resolution = "720" } | ConvertTo-Json
$job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/generate -Headers $headers -ContentType "application/json" -Body $body

Invoke-RestMethod -Uri "http://127.0.0.1:8000/result/$($job.job_id)" -Headers $headers

Invoke-WebRequest -Uri "http://127.0.0.1:8000/result/$($job.job_id)/image" -Headers $headers -OutFile fox.png

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/cancel/$($job.job_id)" -Headers $headers
```

Without keys configured, leave out `$headers`.

## Not covered yet

- No automatic clean-up of old jobs and pictures.
- `/enhance-prompt` is not queued: it runs on the first GPU's thread between jobs.
- Size limits of reference editing and outpainting are a policy now (`MY_DIFFUSION_EDIT_LIMITS`, `MY_DIFFUSION_EDIT_MAX_PIXELS`, see [configuration](configuration.md)): large sizes run by streaming network blocks, a job accepted only because of the policy gets `warnings`, and a CUDA out-of-memory error fails just that job. Left: the limits are the same for every client (no "admins unlimited, services strict": the engine does not know who asked, so it would need the owner passed down with each job); the free-memory measurement is done once per job and trusts the rule of thumb, which was measured on one card; the warning is made when the job runs, not when it is submitted (the size of `aspect_ratio: auto` and outpaint frames is known only to the engine), so a queued job has no warning yet; VAE tiling and cheaper references are not done.
- The scheduler lives inside the API process. Separate worker machines would need a shared database (Postgres)
  instead of the SQLite file.
