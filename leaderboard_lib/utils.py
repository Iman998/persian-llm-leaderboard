"""Shared helper utilities for dynamic imports and more."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import os
import tomllib


def _load_module(path: str):
    """Load a Python module from ``path``.

    If ``path`` is within the repository and points to a ``.py`` file,
    it is imported as a regular module using its dotted name. Otherwise
    the file is loaded dynamically using :mod:`importlib` utilities.
    """
    p = Path(path)
    root = Path(__file__).resolve().parent.parent
    try:
        rel = p.resolve().relative_to(root)
    except ValueError:
        rel = None
    if rel is not None and p.suffix == ".py":
        module_name = ".".join(rel.with_suffix("").parts)
        return importlib.import_module(module_name)
    spec = importlib.util.spec_from_file_location("dyn", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def load_api_key() -> str | None:
    """Return the OpenAI API key from environment or ``secrets.toml``.

    The function checks the ``OPENAI_API_KEY`` environment variable first. If
    not present, it looks for ``secrets.toml`` in the repository root and
    returns ``[openai].api_key`` or ``api_key`` from the TOML file.
    """

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    secrets_file = Path(__file__).resolve().parents[1] / "secrets.toml"
    if secrets_file.exists():
        data = tomllib.loads(secrets_file.read_text())
        if isinstance(data, dict):
            if "api_key" in data:
                return data["api_key"]
            openai_section = data.get("openai")
            if isinstance(openai_section, dict) and "api_key" in openai_section:
                return openai_section["api_key"]

    return None


__all__ = ["_load_module", "load_api_key"]
