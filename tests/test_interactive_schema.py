"""Tests for the interactive-rendering metadata schema + scene graph wiring.

Covers:
    - InteractionConfig normalization.
    - Metadata envelope construction.
    - TooltipTemplate resolution and registry compilation.
    - Heatmap + WebRenderer: DataGrid registration and entity_index build.
    - Cross-panel shared row-id namespace when ``ht1 + ht2`` is drawn.
    - Backward compatibility: ``interactive=False`` / unset → static path unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap import (
    Heatmap, HeatmapList, color_ramp2,
    rowAnnotation, columnAnnotation,
    anno_simple, anno_barplot,
    InteractionConfig, TooltipTemplate,
)
from complexheatmap._interactive import (
    build_metadata,
    ENTITY_CELL, ENTITY_ANNO_CELL, ENTITY_DEND_BRANCH,
    METADATA_SCHEMA_VERSION,
    normalize_interactive,
    encode_datagrid, should_use_datagrid,
)


# ----------------------------------------------------------------------------
# Config normalization
# ----------------------------------------------------------------------------

class TestInteractionConfig:

    def test_none_returns_none(self) -> None:
        assert normalize_interactive(None) is None

    def test_false_returns_none(self) -> None:
        assert normalize_interactive(False) is None

    def test_true_returns_default(self) -> None:
        cfg = normalize_interactive(True)
        assert cfg is not None
        assert cfg.enabled is True
        assert cfg.dend_click == "r-behavior"
        assert cfg.max_inline_cells == 50_000

    def test_dict_kwargs(self) -> None:
        cfg = normalize_interactive({"theme": "dark", "brush": False})
        assert cfg is not None
        assert cfg.theme == "dark"
        assert cfg.brush is False

    def test_dict_ignores_unknown_with_warning(self) -> None:
        with pytest.warns(UserWarning, match="unknown keys"):
            cfg = normalize_interactive({"theme": "dark", "bogus_key": 1})
        assert cfg is not None
        assert cfg.theme == "dark"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            normalize_interactive(42)


# ----------------------------------------------------------------------------
# Metadata envelope
# ----------------------------------------------------------------------------

class TestBuildMetadata:

    def test_cell_envelope(self) -> None:
        md = build_metadata(ENTITY_CELL, heatmap="m1", slice=(1, 2),
                            row=37, col=12,
                            payload={"value": 2.31, "row_name": "TP53"})
        assert md["v"] == METADATA_SCHEMA_VERSION
        assert md["entity"] == ENTITY_CELL
        assert md["heatmap"] == "m1"
        assert md["slice"] == [1, 2]
        assert md["row"] == 37
        assert md["col"] == 12
        assert md["payload"]["value"] == 2.31
        assert md["tooltip_ref"] == ENTITY_CELL

    def test_dend_branch_envelope(self) -> None:
        md = build_metadata(
            ENTITY_DEND_BRANCH, heatmap="m1", slice=(1, 1),
            payload={"axis": "row", "leaves": [1, 2, 3], "n_leaves": 3,
                     "height": 2.5},
        )
        assert md["entity"] == ENTITY_DEND_BRANCH
        assert md["payload"]["leaves"] == [1, 2, 3]

    def test_nan_scrubbed(self) -> None:
        md = build_metadata(ENTITY_CELL, row=1, col=2,
                            payload={"value": float("nan"), "row_name": "r1"})
        assert "value" not in md.get("payload", {})
        assert md["payload"]["row_name"] == "r1"

    def test_numpy_scalar_coerced(self) -> None:
        md = build_metadata(ENTITY_CELL, row=np.int64(7), col=np.int32(3),
                            payload={"value": np.float32(1.5)})
        assert md["row"] == 7
        assert md["col"] == 3
        assert isinstance(md["row"], int)
        assert md["payload"]["value"] == pytest.approx(1.5)

    def test_unknown_entity_raises(self) -> None:
        with pytest.raises(ValueError):
            build_metadata("bogus_entity", row=1)


# ----------------------------------------------------------------------------
# Tooltip templates
# ----------------------------------------------------------------------------

class TestTooltipTemplate:

    def test_defaults_when_empty(self) -> None:
        tmpl = TooltipTemplate()
        reg = tmpl.to_registry()
        assert "cell" in reg
        assert "dend_branch" in reg

    def test_explicit_override(self) -> None:
        tmpl = TooltipTemplate(cell="VAL={value}")
        assert tmpl.resolve("cell") == "VAL={value}"

    def test_callable_skipped_in_registry(self) -> None:
        tmpl = TooltipTemplate(cell=lambda md: "x")
        # Callable present, but not in registry (no JS eval possible)
        assert "cell" not in tmpl.to_registry()


# ----------------------------------------------------------------------------
# DataGrid bypass
# ----------------------------------------------------------------------------

class TestDataGrid:

    def test_should_use_datagrid_small(self) -> None:
        cfg = InteractionConfig(max_inline_cells=50_000)
        assert should_use_datagrid(10, 10, False, cfg) is False
        assert should_use_datagrid(10, 10, True, cfg) is True  # raster forces

    def test_should_use_datagrid_large(self) -> None:
        cfg = InteractionConfig(max_inline_cells=1_000)
        assert should_use_datagrid(100, 100, False, cfg) is True

    def test_encode_float32(self) -> None:
        vals = np.arange(20, dtype=float).reshape(4, 5)
        dg = encode_datagrid(
            "g1",
            values=vals,
            row_ids=[0, 1, 2, 3], col_ids=[0, 1, 2, 3, 4],
            row_names=["a", "b", "c", "d"],
            col_names=["u", "v", "w", "x", "y"],
            viewport_name="vp",
        )
        assert dg.value_shape == [4, 5]
        assert dg.value_dtype == "float32"
        assert dg.values  # non-empty base64

    def test_encode_quantile8_for_large(self) -> None:
        vals = np.random.randn(1000, 600)
        dg = encode_datagrid(
            "g1", values=vals,
            row_ids=list(range(1000)), col_ids=list(range(600)),
            row_names=[str(i) for i in range(1000)],
            col_names=[str(j) for j in range(600)],
            viewport_name="vp",
            max_float32_cells=500_000,
        )
        assert dg.value_dtype == "quantile8"
        assert dg.value_lut is not None
        assert len(dg.value_lut) == 256


# ----------------------------------------------------------------------------
# End-to-end: Heatmap + WebRenderer
# ----------------------------------------------------------------------------

@pytest.fixture
def mat():
    np.random.seed(42)
    return np.random.randn(20, 10)


@pytest.fixture
def color_fn():
    return color_ramp2([-2, 0, 2], ["blue", "white", "red"])


class TestHeatmapInteractive:

    def test_draw_returns_widget_when_interactive(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, interactive=True)
        res = ht.draw(show=False)
        assert type(res).__name__ in ("HeatmapWidget", "HeatmapListWidgetFallback")

    def test_draw_returns_heatmap_list_when_static(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn)
        res = ht.draw(show=False)
        # Static: HeatmapList (None returned via ht_list.draw returning self)
        assert res is None or isinstance(res, HeatmapList)

    def test_scene_graph_has_schema_v11(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, interactive=True)
        ht.draw(show=False)
        rend = grid_py.get_state().get_renderer()
        sd = rend.to_scene_dict()
        assert sd["schema_version"] == "1.1"
        assert "gridpy-heatmap" in sd["interaction_modules"]

    def test_datagrid_registered_per_slice(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, row_km=3, interactive=True)
        ht.draw(show=False)
        rend = grid_py.get_state().get_renderer()
        sd = rend.to_scene_dict()
        # Three row slices → three DataGrids
        ids = sorted(dg["id"] for dg in sd["data_grids"])
        assert len(ids) == 3
        for gid in ids:
            assert gid.startswith("t1_")

    def test_tooltip_templates_registered(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, interactive=True,
                     tooltip=TooltipTemplate(cell="C={value}"))
        ht.draw(show=False)
        rend = grid_py.get_state().get_renderer()
        sd = rend.to_scene_dict()
        assert sd["tooltip_templates"]["cell"] == "C={value}"

    def test_dendrogram_metadata_attached(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, row_km=2, interactive=True)
        ht.draw(show=False)
        rend = grid_py.get_state().get_renderer()
        sd = rend.to_scene_dict()
        # Walk scene graph looking for dend_branch entities
        found = []

        def walk(node):
            if node.get("type") not in ("viewport",):
                data = node.get("data", {})
                if data.get("entity") == "dend_branch":
                    found.append(data)
            for c in node.get("children", []):
                walk(c)
        walk(sd["root"])
        assert len(found) >= 1
        first = found[0]
        assert first["payload"]["axis"] in ("row", "col")
        assert first["payload"]["n_leaves"] > 0

    def test_datagrid_row_ids_match_original_indices(
        self, mat, color_fn
    ) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, row_km=2, interactive=True)
        ht.draw(show=False)
        rend = grid_py.get_state().get_renderer()
        sd = rend.to_scene_dict()
        all_rows = []
        for dg in sd["data_grids"]:
            all_rows.extend(dg["row_ids"])
        assert sorted(all_rows) == list(range(mat.shape[0]))


class TestHeatmapListInteractive:

    def test_cross_panel_shared_row_namespace(self, mat, color_fn) -> None:
        """When ht1 + ht2 is drawn interactively, row_ids in DataGrids
        should reference the SAME original-matrix row indices, so the JS
        runtime can do cross-panel hover highlight without extra mapping.
        """
        m2 = mat[:, :5]  # same rows
        ht1 = Heatmap(mat, name="m1", col=color_fn, interactive=True)
        ht2 = Heatmap(m2, name="m2", col=color_fn)
        (ht1 + ht2).draw(show=False, width=8, height=5)
        rend = grid_py.get_state().get_renderer()
        sd = rend.to_scene_dict()
        grids_by_name = {}
        for dg in sd["data_grids"]:
            grids_by_name.setdefault(dg["id"].split("_")[0], []).append(dg)
        assert "m1" in grids_by_name
        assert "m2" in grids_by_name
        # m1 and m2 share rows 0..19
        rows_m1 = set()
        for dg in grids_by_name["m1"]:
            rows_m1.update(dg["row_ids"])
        rows_m2 = set()
        for dg in grids_by_name["m2"]:
            rows_m2.update(dg["row_ids"])
        assert rows_m1 == rows_m2


class TestBackwardsCompat:
    """Interactive must not break the static path."""

    def test_non_interactive_no_data_grids(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn)
        ht.draw(show=False)
        rend = grid_py.get_state().get_renderer()
        # Cairo renderer has no to_scene_dict
        assert not hasattr(rend, "register_data_grid")

    def test_interactive_false_takes_static_path(self, mat, color_fn) -> None:
        ht = Heatmap(mat, name="t1", col=color_fn, interactive=False)
        res = ht.draw(show=False)
        # Should NOT return a widget
        assert type(res).__name__ not in ("HeatmapWidget", "HeatmapListWidgetFallback")
