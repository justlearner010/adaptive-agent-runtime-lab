"""Minimal .env loader (no external dependency).

Loads KEY=VALUE pairs from a dotenv file into os.environ, without
overriding variables that are already set (real env vars and CLI
overrides take precedence).

Supported syntax: KEY=VALUE lines, blank lines, '#' comments, and
optional single/double quotes around values.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: str | os.PathLike[str] | None = None) -> bool:
    """Load .env from `path` (default: ./.env). Returns True if loaded.

    Silently skips a missing file. Never overrides existing env vars.
    """
    dotenv = Path(path) if path is not None else Path(".env")
    if not dotenv.is_file():
        return False

    loaded = False
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue  # tolerate malformed lines
        key, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:  # existing env vars win
            os.environ[key] = value
            loaded = True
    return loaded
