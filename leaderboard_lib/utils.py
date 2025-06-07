"""Shared helper utilities for dynamic imports and more."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import os
import re
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
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path!r}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def load_api_key(model_name: str | None = None) -> str | None:
    """Return the OpenAI API key for *model_name*.

    The lookup order is:
    1. ``OPENAI_API_KEY_<MODEL>`` environment variable if *model_name* is given
       (converted to an uppercase identifier).
    2. ``OPENAI_API_KEY`` environment variable.
    3. ``secrets.toml`` – ``[model_keys].[MODEL]`` if present.
    4. ``secrets.toml`` – ``[openai].api_key`` or top-level ``api_key``.
    """

    if model_name:
        ident = re.sub(r"[^A-Za-z0-9]", "_", model_name).upper()
        key = os.getenv(f"OPENAI_API_KEY_{ident}")
        if key:
            return key

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    secrets_file = Path(__file__).resolve().parents[1] / "secrets.toml"
    if secrets_file.exists():
        data = tomllib.loads(secrets_file.read_text())
        if isinstance(data, dict):
            model_keys = data.get("model_keys")
            if (
                model_name is not None
                and isinstance(model_keys, dict)
                and model_name in model_keys
            ):
                return model_keys[model_name]
            if "api_key" in data:
                return data["api_key"]
            openai_section = data.get("openai")
            if isinstance(openai_section, dict) and "api_key" in openai_section:
                return openai_section["api_key"]

    return None


__all__ = ["_load_module", "load_api_key"]
