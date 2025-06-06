"""Shared helper utilities for dynamic imports and more."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


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

__all__ = ["_load_module"]
