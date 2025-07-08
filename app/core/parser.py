"""
Filename parsing and directory‑scanning helpers.

Responsible **only** for discovering CSV result files and
mapping them to (dataset, model, suffix) tuples.  UI code never
touches regexes or directories directly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .paths import MODELS_DIR, RESULTS_DIR

__all__ = ["parse_file", "scan_result_maps"]

# --------------------------------------------------------------------- #
# Build the giant regex only once at import time
# --------------------------------------------------------------------- #
model_names = sorted(
    [p.stem for p in MODELS_DIR.glob("*.yaml")], key=len, reverse=True
)
models_alt = "|".join(map(re.escape, model_names))

_RE_NEW = re.compile(rf"^(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$")
_RE_LEGACY = re.compile(
    rf"^(?P<dataset>.+?)_(?P<model>{models_alt})(?:_(?P<suffix>.+?))?\.csv$"
)


def parse_file(p: Path) -> Tuple[str, str, str] | None:
    """
    Extract `(dataset, model, suffix)` from `p.name`.

    Returns `None` *iff* the filename does not conform to the
    expected pattern or represents a temporary sample file.
    """
    # ⚑ New naming scheme – dataset is the directory name
    m = _RE_NEW.match(p.name)
    if m:
        try:
            ds = p.relative_to(RESULTS_DIR).parts[0]
        except ValueError:  # file not under `results/` (unlikely)
            ds = p.parent.parent.name
        mdl, suf = m.group("model", "suffix")
        if suf and suf.isdigit():
            return None
        return ds, mdl, (suf or "")

    # ⚑ Legacy scheme – dataset encoded in filename
    m = _RE_LEGACY.match(p.name)
    if m:
        ds, mdl, suf = m.group("dataset", "model", "suffix")
        if suf and suf.isdigit():
            return None
        return ds, mdl, (suf or "")

    return None


def scan_result_maps():
    """
    Walk ``results/`` and build three look‑up dictionaries aware of
    few‑shot suffixes.

    * ``datasets`` – all dataset names that have **main** CSVs
    * ``main_map[(ds, mdl, shots)]       -> Path``
    * ``raw_map[(ds, mdl, shots)]        -> Path``
    * ``cat_map[(ds, mdl, shots, cat)]   -> Path``
    """

    main_map: Dict[Tuple[str, str, int], Path] = {}
    raw_map: Dict[Tuple[str, str, int], Path] = {}
    cat_map: Dict[Tuple[str, str, int, str], Path] = {}

    shot_re = re.compile(r"^s(\d+)(?:_(.+))?$")

    for p in RESULTS_DIR.rglob("*.csv"):
        parsed = parse_file(p)
        if not parsed:
            continue
        ds, mdl, suf = parsed

        is_raw = False
        if suf.endswith("_raw"):
            is_raw = True
            suf = suf[:-4]

        shots = 0
        cat = ""
        if suf:
            m = shot_re.match(suf)
            if m:
                shots = int(m.group(1))
                cat = m.group(2) or ""
            else:
                cat = suf

        if is_raw:
            raw_map[(ds, mdl, shots)] = p
        elif cat:
            cat_map[(ds, mdl, shots, cat)] = p
        else:
            main_map[(ds, mdl, shots)] = p

    datasets = sorted({k[0] for k in main_map})
    return datasets, main_map, raw_map, cat_map
