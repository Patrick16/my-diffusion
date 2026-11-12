"""Thin synchronous HTTP client for the ComfyUI API (/prompt, /history, /view).

We do not use a websocket for progress - we poll /history once a second. It is simpler and needs no extra dependency.
For one request at a time (our API already serializes generations via a lock in engine.py), a delay of up to a second does not matter."""

from __future__ import annotations

import logging
import time
import uuid

import httpx

logger = logging.getLogger(__name__)


class ComfyGenerationError(RuntimeError):
    """ComfyUI returned an error while executing the graph (see .node_errors)."""

    def __init__(self, message: str, node_errors: dict | None = None):
        super().__init__(message)
        self.node_errors = node_errors or {}


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def queue_prompt(self, workflow: dict) -> str:
        resp = self._http.post("/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if resp.status_code >= 400:
            raise ComfyGenerationError(f"ComfyUI rejected the graph ({resp.status_code}): {resp.text}")
        data = resp.json()
        if "node_errors" in data and data["node_errors"]:
            raise ComfyGenerationError(f"ComfyUI rejected the graph: {data['node_errors']}", data["node_errors"])
        return data["prompt_id"]

    def wait_for_completion(self, prompt_id: str, poll_interval: float = 1.0, timeout: float = 300.0) -> dict:
        """Waits until prompt_id appears in /history with a result, and returns
        the history record (it contains the outputs of each node)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            resp = self._http.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") is True:
                    return entry
                if status.get("status_str") == "error":
                    raise ComfyGenerationError(f"ComfyUI reported an execution error: {status}")
            time.sleep(poll_interval)
        raise TimeoutError(f"ComfyUI did not finish prompt_id={prompt_id} within {timeout}s")

    def get_image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        resp = self._http.get(
            "/view", params={"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self._http.close()
