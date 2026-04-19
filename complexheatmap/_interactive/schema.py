"""Metadata schema and :class:`InteractionConfig`.

Canonical reference for the metadata envelope attached to every grob
emitted by complexheatmap-py drawing pipelines.  Mirrored in the JS
runtime (``gridpy-heatmap.js``); field names must stay in sync.

metadata envelope (v=1)::

    {
      "v": 1,
      "entity": ENTITY_CELL | ENTITY_ROW_LABEL | ... ,
      "heatmap": str,              # Heatmap.name this grob belongs to
      "slice": [k, l],             # 1-indexed (row_slice, col_slice), matches R
      "row": int | None,           # ORIGINAL-matrix global index (cross-panel id)
      "col": int | None,
      "payload": {...},            # entity-specific: value/row_name/anno_name/...
      "tooltip_ref": str,          # key into scene_graph tooltip_templates
    }

The ``row``/``col`` keys are deliberately global — two heatmaps sharing
row order (e.g. ``ht1 + ht2``) use the same id-space, making cross-panel
hover highlight natural for the JS runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = [
    "METADATA_SCHEMA_VERSION",
    "ENTITY_CELL", "ENTITY_ROW_LABEL", "ENTITY_COL_LABEL",
    "ENTITY_ANNO_CELL", "ENTITY_DEND_BRANCH", "ENTITY_DEND_LEAF",
    "ENTITY_LEGEND_ITEM", "ENTITY_SLICE_TITLE",
    "InteractionConfig",
    "normalize_interactive",
    "is_interactive_enabled",
]

METADATA_SCHEMA_VERSION: int = 1

# Entity constants — used both as metadata["entity"] values and as keys
# in the tooltip template registry.
ENTITY_CELL = "cell"
ENTITY_ROW_LABEL = "row_label"
ENTITY_COL_LABEL = "col_label"
ENTITY_ANNO_CELL = "anno_cell"
ENTITY_DEND_BRANCH = "dend_branch"
ENTITY_DEND_LEAF = "dend_leaf"
ENTITY_LEGEND_ITEM = "legend_item"
ENTITY_SLICE_TITLE = "slice_title"

VALID_ENTITIES = frozenset({
    ENTITY_CELL, ENTITY_ROW_LABEL, ENTITY_COL_LABEL,
    ENTITY_ANNO_CELL, ENTITY_DEND_BRANCH, ENTITY_DEND_LEAF,
    ENTITY_LEGEND_ITEM, ENTITY_SLICE_TITLE,
})


@dataclass
class InteractionConfig:
    """User-facing configuration for interactive rendering.

    Accepted shapes at :class:`~complexheatmap.Heatmap` / :meth:`HeatmapList.draw`
    call sites via the ``interactive`` kwarg:

        * ``False`` / ``None``  — static-only path (Cairo or plain WebRenderer).
        * ``True``              — all interactions enabled with defaults.
        * ``dict``              — kwargs for :class:`InteractionConfig`.
        * :class:`InteractionConfig` instance — used as-is.

    Attributes are 1:1 with the locked design spec (see
    ``memory/project_interactive_heatmap_plan.md``).
    """

    enabled: bool = True
    tooltip: Any = None  # TooltipTemplate | None (imported lazily to avoid cycle)
    hover_highlight: bool = True
    brush: bool = True
    click_to_pin: bool = True
    search: bool = True
    legend_filter: bool = True
    dend_click: Optional[str] = "r-behavior"      # "r-behavior" | "select-only" | None
    dend_click_subpanel: bool = True
    dend_click_recluster: bool = False
    theme: str = "light"                           # "light" | "dark"
    max_inline_cells: int = 50_000
    max_float32_cells: int = 500_000               # switch to quantile8 above this
    hover_events_enabled: bool = False             # hover-event async stream off by default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hover_highlight": self.hover_highlight,
            "brush": self.brush,
            "click_to_pin": self.click_to_pin,
            "search": self.search,
            "legend_filter": self.legend_filter,
            "dend_click": self.dend_click,
            "dend_click_subpanel": self.dend_click_subpanel,
            "dend_click_recluster": self.dend_click_recluster,
            "theme": self.theme,
            "max_inline_cells": self.max_inline_cells,
            "max_float32_cells": self.max_float32_cells,
            "hover_events_enabled": self.hover_events_enabled,
        }


def normalize_interactive(
    value: Union[None, bool, Dict[str, Any], "InteractionConfig"],
) -> Optional[InteractionConfig]:
    """Coerce the ``interactive`` kwarg into an :class:`InteractionConfig`.

    Returns ``None`` when interactivity should be disabled.
    """
    if value is None or value is False:
        return None
    if isinstance(value, InteractionConfig):
        return value if value.enabled else None
    if value is True:
        return InteractionConfig(enabled=True)
    if isinstance(value, dict):
        valid = set(InteractionConfig.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in value.items() if k in valid}
        unknown = set(value) - valid
        if unknown:
            import warnings
            warnings.warn(
                f"InteractionConfig: unknown keys {sorted(unknown)} — ignoring."
            )
        cfg = InteractionConfig(**kwargs)
        return cfg if cfg.enabled else None
    raise TypeError(
        f"interactive must be bool/dict/InteractionConfig/None, got {type(value).__name__}"
    )


def is_interactive_enabled(
    value: Union[None, bool, Dict[str, Any], InteractionConfig],
) -> bool:
    return normalize_interactive(value) is not None
