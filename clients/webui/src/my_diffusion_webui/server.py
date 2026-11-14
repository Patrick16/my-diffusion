"""The web UI's own server: it serves the pages and forwards the browser's calls to the my-diffusion API.

    uv run my-diffusion-webui                       # http://127.0.0.1:8080
    uv run my-diffusion-webui --host 0.0.0.0        # visible in the local network (read the README first)

Why a server and not a page that calls the API directly:
  * the API key stays here, in a file, and is never sent to the browser;
  * the browser does not have to be allowed to call another address (no CORS on the API);
  * the server reads the SSE stream of a job with the key in a header, which the browser's EventSource cannot do.

The browser talks only to this server: /ui/... is the web UI's own settings, /api/... is forwarded to the API as is.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
# clients/webui/webui_settings.json when run from the source tree; MY_DIFFUSION_WEBUI_SETTINGS moves it.
SETTINGS_FILE = Path(os.environ.get("MY_DIFFUSION_WEBUI_SETTINGS") or PACKAGE_DIR.parents[1] / "webui_settings.json")

DEFAULT_API_URL = "http://127.0.0.1:8000"
USER_ID_PATTERN = r"^[A-Za-z0-9_.@-]{1,64}$"   # the same rule as the API's X-User-Id


class Connection(BaseModel):
    api_url: str = DEFAULT_API_URL
    api_key: str = ""
    user_id: str = ""


class ConnectionUpdate(BaseModel):
    """A field left out stays as it is. `api_key` "" removes the key."""

    api_url: str | None = None
    api_key: str | None = None
    user_id: str | None = Field(default=None, pattern=r"^([A-Za-z0-9_.@-]{1,64})?$")


_lock = threading.Lock()


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("the API address must look like http://127.0.0.1:8000")
    return url


def _read_file() -> dict:
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def load_connection() -> Connection:
    """The saved connection; the first run starts from the environment (MY_DIFFUSION_WEBUI_API_URL / _API_KEY / _USER_ID)."""
    base = Connection(
        api_url=os.environ.get("MY_DIFFUSION_WEBUI_API_URL", DEFAULT_API_URL),
        api_key=os.environ.get("MY_DIFFUSION_WEBUI_API_KEY", ""),
        user_id=os.environ.get("MY_DIFFUSION_WEBUI_USER_ID", ""),
    )
    saved = _read_file()
    return base.model_copy(update={k: v for k, v in saved.items() if k in Connection.model_fields and isinstance(v, str)})


def _write_file(update: dict) -> None:
    """The file also holds the listen address: a change of one part keeps the other."""
    SETTINGS_FILE.write_text(json.dumps({**_read_file(), **update}, indent=2), encoding="utf-8")


def save_connection(conn: Connection) -> None:
    _write_file(conn.model_dump())


def public_view(conn: Connection) -> dict:
    """What the browser may see: everything but the key itself."""
    return {"api_url": conn.api_url, "has_key": bool(conn.api_key), "user_id": conn.user_id}


def _api_headers(conn: Connection) -> dict[str, str]:
    headers = {}
    if conn.api_key:
        headers["Authorization"] = f"Bearer {conn.api_key}"
    if conn.user_id:
        headers["X-User-Id"] = conn.user_id
    return headers


# No overall timeout: an event stream may stay open for the whole length of a job.
_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))

app = FastAPI(title="my-diffusion web UI", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """A page of another site must not be able to make this server act with the saved API key (a "cross-site request"):
    a call that changes something is accepted only from a page that this server itself served."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin is not None and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "cross-site requests are refused"}, status_code=403)
    return await call_next(request)


# ---------------------------------------------------------------------- the web UI's own settings
@app.get("/ui/connection")
async def get_connection() -> dict:
    return public_view(load_connection())


@app.put("/ui/connection")
async def put_connection(update: ConnectionUpdate) -> dict:
    with _lock:
        conn = load_connection()
        if update.api_url is not None:
            try:
                conn.api_url = _normalize_url(update.api_url)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
        if update.api_key is not None:
            conn.api_key = update.api_key.strip()
        if update.user_id is not None:
            conn.user_id = update.user_id
        save_connection(conn)
    return public_view(conn)


@app.post("/ui/connection/test")
async def test_connection() -> dict:
    """Asks the API who we are: /health needs no key, /jobs does, so a wrong key shows up here."""
    conn = load_connection()
    result: dict = {"reachable": False, "authorized": None, "health": None, "detail": None}
    try:
        health = await _client.get(f"{conn.api_url}/health", timeout=5.0)
        result["reachable"] = True
        result["health"] = health.json()
        jobs = await _client.get(f"{conn.api_url}/jobs", params={"limit": 1}, headers=_api_headers(conn), timeout=5.0)
        result["authorized"] = jobs.status_code == 200
        if jobs.status_code != 200:
            result["detail"] = _detail(jobs)
        else:
            cfg = await _client.get(f"{conn.api_url}/config", headers=_api_headers(conn), timeout=5.0)
            result["admin"] = cfg.status_code == 200
    except (httpx.HTTPError, ValueError) as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return result


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text[:200]


# ---------------------------------------------------------------------- where this web UI itself listens
DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8080


class Listen(BaseModel):
    host: str = Field(pattern=r"^[A-Za-z0-9.:_-]{1,253}$")
    port: int = Field(ge=1, le=65535)


def saved_listen() -> tuple[str, int]:
    saved = _read_file()
    host, port = saved.get("listen_host"), saved.get("listen_port")
    return (host if isinstance(host, str) else DEFAULT_HOST), (port if isinstance(port, int) else DEFAULT_PORT)


def lan_address() -> str | None:
    """This computer's address in the local network (no packet is sent: connecting a UDP socket only picks the route)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return None


@app.get("/ui/listen")
async def get_listen() -> dict:
    host, port = saved_listen()
    return {"host": host, "port": port, "lan_address": lan_address()}


@app.put("/ui/listen")
async def put_listen(listen: Listen) -> dict:
    """Saved for the next start of the web UI; a running server cannot move to another address."""
    with _lock:
        _write_file({"listen_host": listen.host, "listen_port": listen.port})
    return await get_listen()


# ---------------------------------------------------------------------- the forwarded API
_PASS_REQUEST = {"content-type", "accept"}
_PASS_RESPONSE = {"content-type", "content-disposition", "cache-control", "x-accel-buffering"}
# Pictures never change once written, so the browser may keep them: the gallery would otherwise download every PNG again.
_IMMUTABLE = ("/image", "/inputs/")


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PATCH"])
async def forward(path: str, request: Request) -> Response:
    conn = load_connection()
    headers = {k: v for k, v in request.headers.items() if k.lower() in _PASS_REQUEST}
    headers.update(_api_headers(conn))
    try:
        upstream = await _client.send(
            _client.build_request(
                request.method, f"{conn.api_url}/{path}", params=request.query_params,
                headers=headers, content=await request.body(),
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return JSONResponse({"detail": f"cannot reach the API at {conn.api_url} ({reason})", "unreachable": True}, status_code=502)
    out = {k: v for k, v in upstream.headers.items() if k.lower() in _PASS_RESPONSE}
    if upstream.status_code == 200 and request.method == "GET" and any(marker in path for marker in _IMMUTABLE):
        out["Cache-Control"] = "private, max-age=86400"
    return StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code, headers=out, background=BackgroundTask(upstream.aclose),
    )


# ---------------------------------------------------------------------- the pages
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


class RevalidatingFiles(StaticFiles):
    """The browser asks again on every load (answered with 304 when nothing changed), so an updated page is never stale."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatingFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for the my-diffusion API")
    saved_host, saved_port = saved_listen()
    parser.add_argument("--host", default=os.environ.get("MY_DIFFUSION_WEBUI_HOST", saved_host),
                        help="address to listen on (default: the saved one in Settings, else 127.0.0.1); 127.0.0.1 = this computer only, 0.0.0.0 = the whole network")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MY_DIFFUSION_WEBUI_PORT", saved_port)))
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: listening on {args.host}: everybody who can reach this port acts with the saved API key.")
    print(f"Web UI: http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
