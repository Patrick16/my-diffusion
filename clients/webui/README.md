# my-diffusion web UI

A browser interface for the [my-diffusion](../../README.md) HTTP API. It is a separate project: its own
dependencies, its own server, and it talks to the API only over HTTP.

```bash
uv run --project clients/webui my-diffusion-webui        # http://127.0.0.1:8080
```

(or `cd clients/webui && uv run my-diffusion-webui`). The API must be running too; the UI works from any machine
that can reach it. Open **Settings** and enter the API address, the key and the optional `X-User-Id`.

## How it works

```
browser  ──►  web UI server (this project)  ──►  my-diffusion API
              serves the pages, forwards /api/...,
              keeps the API key in a file
```

- The browser never sees the API key: the web UI's server adds it (`webui_settings.json`, ignored by git).
- Progress comes from the API's server-sent events (`/result/{id}/events`), read with `fetch` through the same
  forwarding; the list of jobs is also polled, so a broken stream loses nothing.
- The front end is plain JavaScript with [Alpine.js](https://alpinejs.dev) 3.14.9 (loaded from jsdelivr,
  checked by its hash). There is no build step and no Node.

## Options

| Flag / variable | Default | Meaning |
|---|---|---|
| `--host`, `MY_DIFFUSION_WEBUI_HOST` | saved in Settings, else `127.0.0.1` | Address to listen on. `0.0.0.0` shows the UI to the whole network |
| `--port`, `MY_DIFFUSION_WEBUI_PORT` | saved in Settings, else `8080` | Port |
| `MY_DIFFUSION_WEBUI_API_URL` / `_API_KEY` / `_USER_ID` | `http://127.0.0.1:8000` | First-run connection; Settings overrides them |
| `MY_DIFFUSION_WEBUI_SETTINGS` | `clients/webui/webui_settings.json` | Where the connection is saved |

**Listening on the network:** everyone who can open the page acts with the saved API key (an admin key can change the
server's settings). Do it only on a network you trust, and set API keys on the API first.

## What works

- Connection settings with a test (reachable? key accepted? admin?)
- **Create** in all five modes, with one form that changes by mode:
  - *Text2img*: checkpoint picker (folder tree, filter, switch), prompt with `#` comment lines and *Enhance prompt*, size
    tiers and ratios from `/resolutions`, seed, LoRA picker with weights, steps / CFG / interval / shift
  - *Img2img*: init picture (file, drop, paste or from the gallery), strength, a dashed frame showing the crop to the ratio
  - *Inpaint*: mask editor on the picture (brush, eraser, rectangle, undo / redo, invert, load a mask PNG), feather,
    strength, refinement iterations, one optional extra reference
  - *Edit*: up to `max_references` references, scene / subjects, cut the background, "Auto" ratio from image 1,
    "edit inside image 1 only" (mask on image 1)
  - *Outpaint*: margins in %, live frame preview, keep original pixels, optional second reference
  - the server's edit LoRA is shown as an `auto` row (sent only when changed); a note for large edits
  - the queue with cancel and "Copy to form", live progress, the pictures sent with a job, recent pictures
- **Gallery** from `GET /jobs` (filters, search, load more) and the **detail page** (settings, timings, input pictures,
  reuse settings with their input pictures and mask, new seed, use as init image / reference, download)

The mask is drawn at the picture's own size and sent as a black-and-white PNG; the server crops the picture and the mask
to the chosen ratio together. Input pictures are not remembered between visits (only the form is), but a job's
pictures can be loaded back from the server for 7 days.

- **Settings**: Connection, then the server's settings built from `GET /config` (so a new server setting appears by
  itself), grouped as Folders / Models / Performance / Editing / Queue / Server. Each row says how a change takes effect
  (`live`, `reloads models`, `restart`) and where the value comes from (`override`, `env`, `dotenv`). Typed controls
  (switch, choice, number with local checking, JSON); folders are chosen with a browser over `GET /fs`, the start checkpoint
  and the edit LoRA with the same file tree as on Create. Changes are collected and sent together: *Check only*
  (`dry_run`), *Apply*, *Discard*; a refused value shows the server's message under its row, a row saved in the override
  file has *Reset*. Access (read-only) and a "waiting for restart" banner. The VAE and the background-removal model are
  plain text fields: `GET /fs` lists folders, not files.
- **This web UI**: where it listens (this computer / local network / custom, saved for the next start of the web UI),
  live events or polling, gallery page size, remembering the Create form, and the defaults of that form.

## Tests

```bash
uv run --project clients/webui pytest clients/webui/tests
```
