"""Interactive-rendering support for complexheatmap-py.

Adds a thin semantic layer on top of grid_py's WebRenderer.  Each grob
emitted during drawing carries a metadata envelope (see :mod:`schema`)
that the ``gridpy-heatmap`` JS plugin consumes to provide tooltips,
hover highlight, dendrogram-click sub-panels, brush selection, and
legend filtering.

Public surface:
    - :class:`~complexheatmap._interactive.schema.InteractionConfig`
    - :class:`~complexheatmap._interactive.templates.TooltipTemplate`
    - :func:`~complexheatmap._interactive.metadata.build_metadata`
    - :func:`~complexheatmap._interactive.metadata.heatmap_id`
    - :class:`~complexheatmap._interactive.widget.HeatmapWidget` (PR 2)
"""

from __future__ import annotations

from .schema import (
    InteractionConfig,
    ENTITY_CELL, ENTITY_ROW_LABEL, ENTITY_COL_LABEL,
    ENTITY_ANNO_CELL, ENTITY_DEND_BRANCH, ENTITY_DEND_LEAF,
    ENTITY_LEGEND_ITEM, ENTITY_SLICE_TITLE,
    METADATA_SCHEMA_VERSION,
    normalize_interactive,
    is_interactive_enabled,
)
from .templates import TooltipTemplate, DEFAULT_TEMPLATES
from .metadata import build_metadata, scrub_none
from .datagrid import encode_datagrid, should_use_datagrid

__all__ = [
    "InteractionConfig",
    "TooltipTemplate",
    "DEFAULT_TEMPLATES",
    "ENTITY_CELL", "ENTITY_ROW_LABEL", "ENTITY_COL_LABEL",
    "ENTITY_ANNO_CELL", "ENTITY_DEND_BRANCH", "ENTITY_DEND_LEAF",
    "ENTITY_LEGEND_ITEM", "ENTITY_SLICE_TITLE",
    "METADATA_SCHEMA_VERSION",
    "build_metadata",
    "scrub_none",
    "encode_datagrid",
    "should_use_datagrid",
    "normalize_interactive",
    "is_interactive_enabled",
]
