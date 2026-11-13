"""Shared test setup.

The server reads the settings saved by the config API (data/settings.override.json) when my_diffusion.api is
imported. Tests must not depend on what a developer saved on this machine, so the file is pointed at a path that
does not exist - before any test module imports the API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ["MY_DIFFUSION_SETTINGS_FILE"] = str(Path(tempfile.gettempdir()) / "my-diffusion-tests" / "no-such-override.json")
