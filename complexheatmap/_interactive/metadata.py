"""Metadata envelope construction helpers.

Each drawing site in complexheatmap-py calls :func:`build_metadata` to
attach a dict on a ``grid_py.Grob`` via the ``metadata`` attribute.  The
``_draw.py`` pipeline in grid_py then hands that dict to the renderer —
CairoRenderer ignores it, WebRenderer stores it on the emitted
``GrobNode`` where the JS plugin picks it up.

The helpers are intentionally stateless and tolerant — callers pass
anything close enough (numpy scalars, None, NaN) and we clean up here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .schema import METADATA_SCHEMA_VERSION, VALID_ENTITIES

__all__ = [
    "build_metadata",
    "scrub_none",
    "heatmap_id",
    "slice_tag",
    "push_metadata",
    "active_web_renderer",
]


# ---------------------------------------------------------------------------
# Renderer context helpers
# ---------------------------------------------------------------------------

def active_web_renderer():
    """Return the current ``grid_py.WebRenderer`` instance, or ``None``.

    Metadata attachment is a no-op on non-Web renderers (e.g. Cairo) —
    callers should branch on this.
    """
    from grid_py._state import get_state
    from grid_py.renderer_web import WebRenderer
    renderer = get_state()._renderer
    return renderer if isinstance(renderer, WebRenderer) else None


from contextlib import contextmanager


@contextmanager
def push_metadata(md):
    """Bind *md* as the metadata for every grob drawn inside the block.

    On non-Web renderers the context manager is a no-op.  Nested calls
    stack correctly: inner metadata overrides outer for the duration of
    the inner block, outer is restored on exit.
    """
    renderer = active_web_renderer()
    if renderer is None or md is None:
        yield
        return
    prev = getattr(renderer, "_current_grob_metadata", None)
    renderer.set_grob_metadata(md)
    try:
        yield
    finally:
        if prev is None:
            renderer.clear_grob_metadata()
        else:
            renderer.set_grob_metadata(prev)


def scrub_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ``None`` and NaN entries from a payload dict (non-recursive)."""
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        out[k] = v
    return out


def _coerce_scalar(v: Any) -> Any:
    """Convert numpy scalars / arrays of length 1 to native Python scalars."""
    if v is None:
        return None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        if v.size == 1:
            flat0 = v.flat[0]
            return flat0.item() if isinstance(flat0, np.generic) else flat0
        return [_coerce_scalar(x) for x in v.tolist()]
    return v


def build_metadata(
    entity: str,
    *,
    heatmap: Optional[str] = None,
    slice: Optional[tuple] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
    tooltip_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a metadata envelope suitable for attaching to a grob.

    Parameters
    ----------
    entity : str
        One of the ``ENTITY_*`` constants in :mod:`.schema`.
    heatmap : str, optional
        Name of the owning :class:`~complexheatmap.Heatmap`.
    slice : (int, int), optional
        1-indexed slice position (row_slice_k, col_slice_l) matching R.
    row, col : int, optional
        GLOBAL indices in the original matrix — the cross-panel id.
    payload : dict, optional
        Entity-specific fields (value, names, annotations, etc).
    tooltip_ref : str, optional
        Key into the scene-graph tooltip_templates registry.  Defaults to
        *entity* so the JS runtime picks a template of the same name.
    """
    if entity not in VALID_ENTITIES:
        raise ValueError(f"unknown entity {entity!r}")

    env: Dict[str, Any] = {
        "v": METADATA_SCHEMA_VERSION,
        "entity": entity,
    }
    if heatmap is not None:
        env["heatmap"] = str(heatmap)
    if slice is not None:
        env["slice"] = [int(slice[0]), int(slice[1])]
    if row is not None:
        env["row"] = int(_coerce_scalar(row))
    if col is not None:
        env["col"] = int(_coerce_scalar(col))
    if payload:
        cleaned: Dict[str, Any] = {}
        for k, v in payload.items():
            if v is None:
                continue
            cv = _coerce_scalar(v)
            if isinstance(cv, float) and np.isnan(cv):
                continue
            cleaned[k] = cv
        if cleaned:
            env["payload"] = cleaned
    env["tooltip_ref"] = tooltip_ref or entity
    return env


def heatmap_id(heatmap_obj: Any) -> str:
    """Return the stable id for a :class:`Heatmap` — used as ``heatmap`` key."""
    return str(getattr(heatmap_obj, "name", "") or "")


def slice_tag(k: int, l: int) -> str:
    """Deterministic string form of a (k, l) slice position."""
    return f"{int(k)}_{int(l)}"
