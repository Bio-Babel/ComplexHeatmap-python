"""Edge case tests from ALL R test files — focused on patterns most
likely to expose bugs in the Python port.

Sources: test-SingleAnnotation.R, test-ColorMapping-class.R,
test-HeatmapAnnotation.R, test-AnnotationFunction.R,
test-annotation_axis.R, testthat-ColorMapping.R, test-utils.R
"""
from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap.heatmap import Heatmap
from complexheatmap.heatmap_annotation import HeatmapAnnotation, rowAnnotation
from complexheatmap.annotation_functions import (
    anno_simple, anno_barplot, anno_points, anno_lines,
    anno_text, anno_mark, anno_empty, anno_block,
)
from complexheatmap._color import color_ramp2
from complexheatmap._globals import ht_opt, reset_ht_opt
from complexheatmap.color_mapping import ColorMapping
from complexheatmap._utils import is_abs_unit


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_ht_opt()


# ---------------------------------------------------------------------------
# test-SingleAnnotation.R: SingleAnnotation edge cases
# ---------------------------------------------------------------------------

class TestSingleAnnotationEdgeCases:
    """R test-SingleAnnotation.R"""

    def test_discrete_values(self):
        """R: SingleAnnotation(value=c(rep(c('a','b'), 5)))"""
        ha = HeatmapAnnotation(grp=np.array(["a", "b"] * 5))
        Heatmap(np.random.randn(10, 3), name="sa1",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_sa_disc.png", width=4, height=4, dpi=72)

    def test_discrete_with_colors(self):
        """R: SingleAnnotation with col=c(a='red', b='blue')"""
        ha = HeatmapAnnotation(
            grp=np.array(["a", "b"] * 5),
            col={"grp": {"a": "red", "b": "blue"}},
        )
        Heatmap(np.random.randn(10, 3), name="sa2",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_sa_discol.png", width=4, height=4, dpi=72)

    def test_matrix_annotation(self):
        """R: SingleAnnotation(value=cbind(1:10, 10:1))"""
        m = np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)])
        ha = HeatmapAnnotation(mat=m.astype(float))
        Heatmap(np.random.randn(10, 3), name="sa3",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_sa_mat.png", width=4, height=4, dpi=72)


# ---------------------------------------------------------------------------
# testthat-ColorMapping.R: strict color mapping
# ---------------------------------------------------------------------------

class TestColorMappingStrict:
    """R testthat-ColorMapping.R"""

    def test_discrete_maps_correctly(self):
        """R: map_to_colors(cm, 'a') → '#0000FFFF'"""
        cm = ColorMapping(
            colors={"a": "blue", "b": "white", "c": "red"},
            name="disc",
        )
        result = cm.map_to_colors(np.array(["a"]))
        assert result[0].upper().startswith("#0000FF")

    def test_discrete_vector(self):
        cm = ColorMapping(
            colors={"a": "blue", "b": "white", "c": "red"},
            name="disc2",
        )
        result = cm.map_to_colors(np.array(["a", "a", "b", "c"]))
        assert len(result) == 4
        # a→blue, b→white, c→red
        assert result[0].upper().startswith("#0000FF")

    def test_continuous_boundary(self):
        """R: map_to_colors(cm, 0) → '#0000FFFF'"""
        col = color_ramp2([0, 0.5, 1], ["blue", "white", "red"])
        cm = ColorMapping(col_fun=col, name="cont")
        result = cm.map_to_colors(np.array([0.0]))
        assert result[0].upper().startswith("#0000FF")

    def test_continuous_out_of_range(self):
        """R: map_to_colors(cm, 2) → clamped to red"""
        col = color_ramp2([0, 0.5, 1], ["blue", "white", "red"])
        cm = ColorMapping(col_fun=col, name="cont2")
        result = cm.map_to_colors(np.array([2.0]))
        assert result[0].upper().startswith("#FF0000")


# ---------------------------------------------------------------------------
# test-AnnotationFunction.R: NA handling
# ---------------------------------------------------------------------------

class TestAnnotationNAHandling:
    """R test-AnnotationFunction.R — NA values in annotations."""

    def test_anno_points_with_na(self):
        """R: anno_points(c(1:5, NA, 7:10))"""
        vals = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10], dtype=float)
        ha = HeatmapAnnotation(pt=anno_points(vals))
        Heatmap(np.random.randn(10, 3), name="na1",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_na_pts.png", width=4, height=4, dpi=72)

    def test_anno_barplot_with_na(self):
        """R: anno_barplot(c(1:5, NA, 7:10))"""
        vals = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10], dtype=float)
        ha = HeatmapAnnotation(bar=anno_barplot(vals))
        Heatmap(np.random.randn(10, 3), name="na2",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_na_bar.png", width=4, height=4, dpi=72)

    def test_anno_lines_with_na(self):
        """R: anno_lines(c(1:5, NA, 7:10))"""
        vals = np.array([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10], dtype=float)
        ha = HeatmapAnnotation(ln=anno_lines(vals))
        Heatmap(np.random.randn(10, 3), name="na3",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_na_lines.png", width=4, height=4, dpi=72)

    def test_anno_simple_with_na(self):
        """R: anno_simple(c(1:9, NA))"""
        vals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], dtype=float)
        ha = HeatmapAnnotation(s=anno_simple(vals))
        Heatmap(np.random.randn(10, 3), name="na4",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_na_simple.png", width=4, height=4, dpi=72)


# ---------------------------------------------------------------------------
# test-AnnotationFunction.R: anno_barplot edge cases
# ---------------------------------------------------------------------------

class TestAnnoBarplotEdgeCases:
    """R test-AnnotationFunction.R:226-269"""

    def test_matrix_barplot(self):
        """R: anno_barplot(matrix(nc=2, c(1:10, 10:1)))"""
        m = np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)])
        ha = HeatmapAnnotation(bar=anno_barplot(m.astype(float)))
        Heatmap(np.random.randn(10, 3), name="bp1",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_bar_mat.png", width=4, height=5, dpi=72)

    def test_barplot_row(self):
        """R: anno_barplot(1:10, which='row')"""
        ra = rowAnnotation(bar=anno_barplot(
            np.arange(1, 11, dtype=float), which="row"))
        Heatmap(np.random.randn(10, 3), name="bp2").draw(
            show=False, filename="/tmp/rt_bar_row.png", width=5, height=4, dpi=72)

    def test_barplot_with_fill(self):
        """R: anno_barplot(1:10, gp=gpar(fill=1:10))"""
        ha = HeatmapAnnotation(
            bar=anno_barplot(np.arange(1, 11, dtype=float),
                             gp={"fill": [f"#{i*25:02X}0000" for i in range(10)]}))
        Heatmap(np.random.randn(10, 3), name="bp3",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_bar_fill.png", width=4, height=5, dpi=72)


# ---------------------------------------------------------------------------
# test-AnnotationFunction.R: anno_text rotations
# ---------------------------------------------------------------------------

class TestAnnoTextRotation:
    """R test-AnnotationFunction.R:204-223"""

    def test_text_rotated(self):
        labels = [f"label_{i}" for i in range(10)]
        ha = HeatmapAnnotation(
            txt=anno_text(labels, rot=45, just="left"))
        Heatmap(np.random.randn(10, 3), name="tx1",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_text_rot.png", width=4, height=5, dpi=72)

    def test_text_on_row(self):
        labels = [f"r{i}" for i in range(10)]
        ha = HeatmapAnnotation(
            txt=anno_text(labels, which="row"), which="row")
        Heatmap(np.random.randn(10, 3), name="tx2",
                right_annotation=ha).draw(
            show=False, filename="/tmp/rt_text_row.png", width=5, height=4, dpi=72)


# ---------------------------------------------------------------------------
# test-HeatmapAnnotation.R: multiple annotations with gap
# ---------------------------------------------------------------------------

class TestHeatmapAnnotationComplex:
    """R test-HeatmapAnnotation.R:14-51"""

    def test_mixed_anno_types(self):
        """R: foo=1:10, bar=c('a','b'), pt=anno_points(1:10)"""
        ha = HeatmapAnnotation(
            foo=np.arange(1, 11, dtype=float),
            bar=np.array(["a", "b"] * 5),
            pt=anno_points(np.arange(1, 11, dtype=float)),
            annotation_name_side="left",
        )
        Heatmap(np.random.randn(10, 3), name="cx1",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_ha_complex.png", width=4, height=5, dpi=72)

    def test_anno_empty_with_bar(self):
        """R: HeatmapAnnotation(foo=anno_empty(), bar=1:10)"""
        ha = HeatmapAnnotation(
            foo=anno_empty(),
            bar=np.arange(1, 11, dtype=float),
        )
        Heatmap(np.random.randn(10, 3), name="cx2",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_ha_empty.png", width=4, height=5, dpi=72)


# ---------------------------------------------------------------------------
# testhat-utiles.R: is_abs_unit edge cases
# ---------------------------------------------------------------------------

class TestIsAbsUnitFromR:
    """R testhat-utiles.R:38-43"""

    def test_mm_is_absolute(self):
        assert is_abs_unit(grid_py.Unit(1, "mm")) is True

    def test_npc_not_absolute(self):
        assert is_abs_unit(grid_py.Unit(1, "npc")) is False

    def test_sum_with_npc_not_absolute(self):
        """R: is_abs_unit(unit(1,'mm')+unit(1,'npc')) → FALSE"""
        u = grid_py.Unit(1, "mm") + grid_py.Unit(1, "npc")
        assert is_abs_unit(u) is False

    def test_points_is_absolute(self):
        assert is_abs_unit(grid_py.Unit(5, "points")) is True

    def test_inches_is_absolute(self):
        assert is_abs_unit(grid_py.Unit(1, "inches")) is True


# ---------------------------------------------------------------------------
# test-HeatmapList-class.R: legend positions
# ---------------------------------------------------------------------------

class TestHeatmapListLegendPositions:
    """R test-HeatmapList-class.R:42-46"""

    def test_legend_bottom(self):
        mat = np.random.randn(10, 5)
        (Heatmap(mat, name="lb1") + Heatmap(mat, name="lb2")).draw(
            heatmap_legend_side="bottom",
            show=False, filename="/tmp/rt_leg_bot.png", width=7, height=5, dpi=72)

    def test_legend_left(self):
        mat = np.random.randn(10, 5)
        (Heatmap(mat, name="ll1") + Heatmap(mat, name="ll2")).draw(
            heatmap_legend_side="left",
            show=False, filename="/tmp/rt_leg_left.png", width=7, height=5, dpi=72)

    def test_legend_top(self):
        mat = np.random.randn(10, 5)
        (Heatmap(mat, name="lt1") + Heatmap(mat, name="lt2")).draw(
            heatmap_legend_side="top",
            show=False, filename="/tmp/rt_leg_top.png", width=7, height=6, dpi=72)


# ---------------------------------------------------------------------------
# test-HeatmapList-class.R: width parameters
# ---------------------------------------------------------------------------

class TestHeatmapListWidths:
    """R test-HeatmapList-class.R:48-65"""

    def test_first_fixed_width(self):
        mat = np.random.randn(10, 5)
        (Heatmap(mat, name="fw1", width=grid_py.Unit(6, "cm")) +
         Heatmap(mat, name="fw2")).draw(
            show=False, filename="/tmp/rt_fw1.png", width=9, height=5, dpi=72)

    def test_both_fixed_width(self):
        mat = np.random.randn(10, 5)
        (Heatmap(mat, name="fw3", width=grid_py.Unit(6, "cm")) +
         Heatmap(mat, name="fw4", width=grid_py.Unit(6, "cm"))).draw(
            show=False, filename="/tmp/rt_fw2.png", width=9, height=5, dpi=72)

    def test_proportional_width(self):
        """R: Heatmap(mat1, width=2) + Heatmap(mat2, width=1) → 2:1 ratio"""
        mat = np.random.randn(10, 5)
        (Heatmap(mat, name="pw1", width=2) +
         Heatmap(mat, name="pw2", width=1)).draw(
            show=False, filename="/tmp/rt_pw.png", width=8, height=5, dpi=72)


# ---------------------------------------------------------------------------
# test-HeatmapList-class.R:78-85: split synced with km on secondary
# ---------------------------------------------------------------------------

class TestHeatmapListSplitAdvanced:
    """R: ht_list = Heatmap(mat1, row_km=2) + Heatmap(mat2, row_km=3)
    draw(ht_list, main_heatmap='m1') → m2 inherits m1's 2-split."""

    def test_secondary_km_overridden(self):
        np.random.seed(42)
        mat1 = np.random.randn(24, 5)
        mat2 = np.random.randn(24, 5)
        ht = Heatmap(mat1, name="km1", row_km=2) + \
             Heatmap(mat2, name="km2", row_km=3)  # km=3 should be overridden
        ht.make_layout(main_heatmap="km1")
        # m1 has 2 slices; m2 should inherit 2 (not 3)
        assert len(ht.ht_list[0]._row_order_list) == 2
        assert len(ht.ht_list[1]._row_order_list) == 2

    def test_row_annotations_with_km_split(self):
        """R: Heatmap(mat, row_km=2) + rowAnnotation(foo=1:n, bar=anno_points(n:1))"""
        np.random.seed(42)
        mat = np.random.randn(20, 5)
        ht = Heatmap(mat, name="rka", row_km=2) + \
             rowAnnotation(
                 foo=np.arange(1, 21, dtype=float),
                 bar=anno_points(np.arange(20, 0, -1, dtype=float), which="row"),
                 width=grid_py.Unit(4, "cm"),
             )
        ht.draw(show=False, filename="/tmp/rt_km_anno.png",
                width=7, height=5, dpi=72)


# ---------------------------------------------------------------------------
# test-HeatmapAnnotation.R:70-71: row annotation standalone
# ---------------------------------------------------------------------------

class TestRowAnnotationStandalone:
    def test_rowAnnotation_standalone(self):
        """R: rowAnnotation(foo=1:10, bar=anno_points(10:1))"""
        ra = rowAnnotation(
            foo=np.arange(1, 11, dtype=float),
            bar=anno_points(np.arange(10, 0, -1, dtype=float), which="row"),
        )
        assert len(ra.anno_list) == 2
        assert "foo" in ra.anno_list
        assert "bar" in ra.anno_list


# ---------------------------------------------------------------------------
# test-Heatmap-class.R:319: draw with factor level ordering
# ---------------------------------------------------------------------------

class TestFactorLevelOrdering:
    def test_column_split_factor_order(self):
        """R: column_split=factor(x, levels=c('B','A')) reverses group order."""
        import pandas as pd
        mat = np.random.randn(10, 10)
        groups = np.array(["A"] * 5 + ["B"] * 5)
        factor = pd.Categorical(groups, categories=["B", "A"], ordered=True)
        ht = Heatmap(mat, name="fo1", column_split=factor,
                     cluster_column_slices=False)
        ht.make_layout()
        # B should come before A
        labels = ht._column_split_labels
        assert labels[0] == "B"
        assert labels[1] == "A"
