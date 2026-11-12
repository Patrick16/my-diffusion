"""Runs ComfyUI as a subprocess.

We do not embed ComfyUI into our process (it is a GPL-3.0 server of hundreds of
modules), and we do not ask the user to start it by hand - we simply start
`python.exe ComfyUI/main.py --port <port>` from the user's existing portable build
when our service starts, and kill the process on exit.

We chose ComfyUI as the engine that performs the generation itself, because our own port of the transformer
(see the history in the chat/git) produced noise for an unexplained reason, while the full ComfyUI generates
correctly on the same checkpoint. We run it under the hood and keep our HTTP API thin."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_python(comfy_root: Path) -> Path:
    """The portable ComfyUI build on Windows puts its own interpreter in
    python_embeded/python.exe next to the ComfyUI folder (not in a venv inside it) -
    see run_nvidia_gpu.bat in the build itself. If this is not a portable build
    (there is no such folder), we use the current sys.executable - then ComfyUI must
    be installed in the same environment where our service runs."""
    embedded = comfy_root / "python_embeded" / "python.exe"
    if embedded.exists():
        return embedded
    return Path(sys.executable)


class ComfyProcessManager:
    def __init__(self, comfy_root: Path, host: str, port: int, startup_timeout: float = 180.0):
        self.comfy_root = comfy_root
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen | None = None

    def is_reachable(self, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def ensure_running(self) -> None:
        """If something already answers on this port (for example, a ComfyUI that
        the user started themselves), we simply use it and do not touch it."""
        if self.is_reachable():
            logger.info("ComfyUI is already responding at %s - using the existing process.", self.base_url)
            return
        self._start()

    def _start(self) -> None:
        main_py = self.comfy_root / "ComfyUI" / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(f"ComfyUI/main.py not found in {self.comfy_root} - check comfy_root in the settings")
        python_exe = _find_python(self.comfy_root)

        cmd = [
            str(python_exe),
            "-s",
            str(main_py),
            "--port",
            str(self.port),
            "--listen",
            self.host,
        ]
        logger.info("Starting ComfyUI: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            cwd=str(self.comfy_root / "ComfyUI"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            if self._process.poll() is not None:
                raise RuntimeError(f"ComfyUI crashed on startup (exit code {self._process.returncode})")
            if self.is_reachable():
                logger.info("ComfyUI came up in %.1fs at %s", time.time() - t0, self.base_url)
                return
            time.sleep(1.0)
        raise TimeoutError(f"ComfyUI did not respond at {self.base_url} within {self.startup_timeout}s")

    def stop(self) -> None:
        if self._process is None:
            return  # we did not start it (or it is no longer running) - do not touch someone else's process
        logger.info("Stopping ComfyUI (pid=%s)...", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("ComfyUI did not exit within 15s - killing it")
            self._process.kill()
            self._process.wait(timeout=15)
        self._process = None
