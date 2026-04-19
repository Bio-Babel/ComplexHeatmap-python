"""Tooltip template system.

Templates are simple string patterns with ``{field}`` or ``{field:.2f}``
tokens.  They are compiled to a portable form on the Python side — the
raw string is shipped in the scene graph's ``tooltip_templates`` dict
and the JS runtime compiles its own evaluator.

Callable templates (``callable(metadata) -> str``) are also supported.
They run in Python at draw time and pre-bake per-cell HTML into the
scene graph; they are **disabled** when the DataGrid bypass is active
for a heatmap (the path has no per-cell grob to hang pre-baked HTML on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from .schema import (
    ENTITY_CELL, ENTITY_ROW_LABEL, ENTITY_COL_LABEL,
    ENTITY_ANNO_CELL, ENTITY_DEND_BRANCH, ENTITY_DEND_LEAF,
    ENTITY_LEGEND_ITEM, ENTITY_SLICE_TITLE,
)

__all__ = ["TooltipTemplate", "DEFAULT_TEMPLATES"]


TemplateLike = Union[str, Callable[[Dict[str, Any]], str], None]


DEFAULT_TEMPLATES: Dict[str, str] = {
    ENTITY_CELL: "{row_name} × {col_name}\nvalue: {value:.3f}",
    ENTITY_ROW_LABEL: "{row_name}",
    ENTITY_COL_LABEL: "{col_name}",
    ENTITY_ANNO_CELL: "{anno_name}: {value}",
    ENTITY_DEND_BRANCH: "cluster ({n_leaves} leaves)\nheight: {height:.2f}",
    ENTITY_DEND_LEAF: "{name}",
    ENTITY_LEGEND_ITEM: "{label}",
    ENTITY_SLICE_TITLE: "{title} ({n} {axis}s)",
}


@dataclass
class TooltipTemplate:
    """Per-entity tooltip template registry.

    Every attribute is optional; unset entities fall back to
    :data:`DEFAULT_TEMPLATES`.  Set an attribute to ``None`` to disable
    tooltips for that entity.
    """

    cell: TemplateLike = None
    row_label: TemplateLike = None
    col_label: TemplateLike = None
    anno_cell: TemplateLike = None
    dend_branch: TemplateLike = None
    dend_leaf: TemplateLike = None
    legend_item: TemplateLike = None
    slice_title: TemplateLike = None

    # Internal: track explicit overrides (including None)
    _overrides: Dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for k in ("cell", "row_label", "col_label", "anno_cell",
                  "dend_branch", "dend_leaf", "legend_item", "slice_title"):
            val = getattr(self, k)
            if val is not None:
                self._overrides[k] = val

    def resolve(self, entity: str) -> TemplateLike:
        """Return the active template for *entity* (after fallbacks)."""
        v = getattr(self, entity, None)
        if v is not None:
            return v
        if entity in self._overrides:
            return self._overrides[entity]  # may be None (disabled)
        return DEFAULT_TEMPLATES.get(entity)

    def to_registry(self) -> Dict[str, Any]:
        """Compile into a dict keyed by entity for scene-graph embedding.

        Callable values are replaced with a sentinel (``"__callable__"``)
        since we cannot ship Python functions to the browser.  Call sites
        are responsible for pre-baking callable tooltips into metadata
        before the DataGrid bypass strips grob-level metadata.
        """
        out: Dict[str, Any] = {}
        for k in ("cell", "row_label", "col_label", "anno_cell",
                  "dend_branch", "dend_leaf", "legend_item", "slice_title"):
            tmpl = self.resolve(k)
            if tmpl is None:
                continue
            if callable(tmpl):
                # Skip callables — they were pre-baked per-cell by the caller.
                continue
            out[k] = tmpl
        return out


def coerce_template(value: Any) -> TooltipTemplate:
    """Accept None/dict/TooltipTemplate and return a TooltipTemplate."""
    if value is None:
        return TooltipTemplate()
    if isinstance(value, TooltipTemplate):
        return value
    if isinstance(value, dict):
        return TooltipTemplate(**{k: v for k, v in value.items()
                                  if k in TooltipTemplate.__dataclass_fields__})
    raise TypeError(f"tooltip must be dict/TooltipTemplate/None, got {type(value).__name__}")
