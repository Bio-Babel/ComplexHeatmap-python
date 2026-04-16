"""Tests derived from R's own test suite — verifying critical invariants.

R source: ComplexHeatmap/tests/test-*.R and tests/testthat/*.R
"""
from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap.heatmap import Heatmap
from complexheatmap.heatmap_annotation import HeatmapAnnotation, rowAnnotation
from complexheatmap.annotation_functions import (
    anno_barplot, anno_points, anno_simple, anno_mark,
)
from complexheatmap._color import color_ramp2
from complexheatmap._utils import smart_align, pindex
from complexheatmap._globals import ht_opt, reset_ht_opt
from complexheatmap.color_mapping import ColorMapping


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    reset_ht_opt()


# ---------------------------------------------------------------------------
# R test-utils.R: smartAlign2 invariants
# ---------------------------------------------------------------------------

class TestSmartAlignRInvariants:
    """R test-utils.R:45-58: after alignment, no visual overlaps."""

    def test_no_overlap_after_align(self):
        """R: pos[1:(n-1), 2] <= pos[2:n, 1] for sorted positions."""
        start = np.array([0.04, 0.049, 0.042, 0.055, 0.082,
                          0.127, 0.179, 0.328, 0.571, 0.818])
        end = np.array([0.092, 0.107, 0.138, 0.159, 0.178,
                        0.207, 0.305, 0.463, 0.677, 0.930])
        result = smart_align(start, end, (0, 1))
        # Sort by center
        centers = (result[:, 0] + result[:, 1]) / 2
        od = np.argsort(centers)
        sorted_result = result[od]
        # No overlap: each end <= next start (with tolerance)
        for i in range(len(sorted_result) - 1):
            assert sorted_result[i, 1] <= sorted_result[i + 1, 0] + 0.001, \
                f"Overlap at {i}: {sorted_result[i, 1]} > {sorted_result[i+1, 0]}"

    def test_heights_preserved(self):
        """R: interval heights should be preserved."""
        h1 = np.array([1.0, 3.0, 5.0])
        h2 = np.array([2.0, 4.5, 7.0])
        orig_heights = h2 - h1
        result = smart_align(h1, h2, (0, 10))
        new_heights = result[:, 1] - result[:, 0]
        assert np.allclose(orig_heights, new_heights, atol=0.01)


# ---------------------------------------------------------------------------
# R test-Heatmap-class.R: row_order/column_order invariants
# ---------------------------------------------------------------------------

class TestHeatmapOrderInvariants:
    """R test-Heatmap-class.R:145-155"""

    def test_explicit_row_order(self):
        """R: Heatmap(mat, row_order=od) → order preserved."""
        mat = np.random.randn(10, 5)
        od = list(range(9, -1, -1))  # reverse
        ht = Heatmap(mat, name="ord", row_order=od, cluster_rows=False)
        ht.make_layout()
        actual = list(ht._row_order_list[0])
        assert actual == od

    def test_explicit_column_order(self):
        mat = np.random.randn(10, 5)
        od = [4, 3, 2, 1, 0]
        ht = Heatmap(mat, name="cord", column_order=od, cluster_columns=False)
        ht.make_layout()
        actual = list(ht._column_order_list[0])
        assert actual == od

    def test_row_order_length(self):
        """All rows should be present in order."""
        mat = np.random.randn(20, 5)
        ht = Heatmap(mat, name="olen")
        ht.make_layout()
        all_indices = np.concatenate(ht._row_order_list)
        assert set(all_indices) == set(range(20))

    def test_column_order_length(self):
        mat = np.random.randn(10, 8)
        ht = Heatmap(mat, name="colen")
        ht.make_layout()
        all_indices = np.concatenate(ht._column_order_list)
        assert set(all_indices) == set(range(8))


# ---------------------------------------------------------------------------
# R test-Heatmap-class.R: row_km split
# ---------------------------------------------------------------------------

class TestRowKmSplit:
    def test_row_km_produces_slices(self):
        """R: row_km=k should produce k slices."""
        np.random.seed(42)
        mat = np.random.randn(50, 5)
        ht = Heatmap(mat, name="km", row_km=3)
        ht.make_layout()
        assert len(ht._row_order_list) == 3

    def test_all_rows_covered(self):
        """All rows must be in exactly one slice."""
        np.random.seed(42)
        mat = np.random.randn(50, 5)
        ht = Heatmap(mat, name="km2", row_km=4)
        ht.make_layout()
        all_rows = np.concatenate(ht._row_order_list)
        assert len(all_rows) == 50
        assert len(set(all_rows)) == 50


# ---------------------------------------------------------------------------
# R testthat-HeatmapAnnotation-size.R: annotation default sizes
# ---------------------------------------------------------------------------

class TestAnnotationSizingRInvariants:
    """R testthat-HeatmapAnnotation-size.R"""

    def test_simple_anno_default_5mm(self):
        """R: simple annotation default size = simple_anno_size (5mm)."""
        ha = HeatmapAnnotation(foo=np.array([1, 2, 3]))
        # Total height should be ~5mm (simple_anno_size default)
        h = ha.height
        if isinstance(h, (int, float)):
            assert abs(h - 5.0) < 2  # ~5mm
        elif grid_py.is_unit(h):
            h_mm = float(np.squeeze(
                grid_py.convert_height(h, "mm", valueOnly=True)))
            assert abs(h_mm - 5.0) < 2

    def test_complex_anno_default_10mm(self):
        """R: complex annotation (barplot) default size ≈ 10mm."""
        # anno_barplot default height = _DEFAULT_SIZE = 10mm
        ha = HeatmapAnnotation(bar=anno_barplot(np.array([1, 2, 3])))
        h = ha.height
        if isinstance(h, (int, float)):
            assert abs(h - 10.0) < 3
        elif grid_py.is_unit(h):
            h_mm = float(np.squeeze(
                grid_py.convert_height(h, "mm", valueOnly=True)))
            assert abs(h_mm - 10.0) < 3


# ---------------------------------------------------------------------------
# R testthat-ColorMapping.R: mapping invariants
# ---------------------------------------------------------------------------

class TestColorMappingRInvariants:
    """R testthat-ColorMapping.R"""

    def test_continuous_boundaries(self):
        """R: col_fun(min) = first color, col_fun(max) = last color."""
        col = color_ramp2([0, 0.5, 1], ["blue", "white", "red"])
        assert col(0).upper() == "#0000FF"
        assert col(1).upper() == "#FF0000"

    def test_continuous_clamping(self):
        """R: values outside range clamped to boundary colors."""
        col = color_ramp2([0, 0.5, 1], ["blue", "white", "red"])
        assert col(-1) == col(0)   # clamp low
        assert col(2) == col(1)    # clamp high

    def test_discrete_exact_match(self):
        """R: discrete mapping returns exact color for known level."""
        cm = ColorMapping(
            colors={"a": "#0000FF", "b": "#FFFFFF", "c": "#FF0000"},
            name="disc",
        )
        result = cm.map_to_colors(np.array(["a", "b", "c"]))
        assert result[0].upper().startswith("#0000FF")
        assert result[2].upper().startswith("#FF0000")

    def test_nan_in_continuous(self):
        """R: NA → NA_character_. Python: → transparent."""
        col = color_ramp2([0, 1], ["white", "red"])
        result = col(np.array([0.5, np.nan, 1.0]))
        assert result[1] == "#FFFFFF00"  # transparent for NaN
        assert result[0] != "#FFFFFF00"  # non-NaN is normal


# ---------------------------------------------------------------------------
# R test-HeatmapList-class.R: multi-heatmap invariants
# ---------------------------------------------------------------------------

class TestHeatmapListRInvariants:
    """R test-HeatmapList-class.R"""

    def test_row_split_synced(self):
        """R: when main heatmap has row_split, all heatmaps share same split."""
        np.random.seed(42)
        groups = np.repeat(["A", "B"], 10)
        ht = Heatmap(np.random.randn(20, 5), name="s1",
                     row_split=groups, cluster_rows=False) + \
             Heatmap(np.random.randn(20, 3), name="s2")
        ht.make_layout()
        # Both heatmaps should have same number of slices
        assert len(ht.ht_list[0]._row_order_list) == 2
        assert len(ht.ht_list[1]._row_order_list) == 2

    def test_draw_without_error(self):
        """R: basic combination draw should succeed."""
        ht = Heatmap(np.random.randn(10, 5), name="d1") + \
             Heatmap(np.random.randn(10, 3), name="d2")
        ht.draw(show=False, filename="/tmp/t_ht_list_invar.png",
                width=7, height=5, dpi=72)

    def test_annotation_in_list(self):
        """R: rowAnnotation in HeatmapList."""
        mat = np.random.randn(10, 5)
        ht = Heatmap(mat, name="al") + \
             rowAnnotation(pt=anno_points(np.random.randn(10), which="row"))
        ht.draw(show=False, filename="/tmp/t_anno_list_invar.png",
                width=6, height=4, dpi=72)


# ---------------------------------------------------------------------------
# R: pindex edge cases
# ---------------------------------------------------------------------------

class TestPindexRInvariants:
    def test_length_mismatch_recycle(self):
        """R: if length(i)==1, recycle to length(j)."""
        m = np.arange(12).reshape(3, 4)
        result = pindex(m, np.array([1]), np.array([0, 1, 2, 3]))
        assert np.array_equal(result, m[1, :])

    def test_3d_would_need_check(self):
        """R supports 3D arrays. Python should at least not crash on 2D."""
        m = np.arange(20).reshape(4, 5)
        result = pindex(m, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        assert len(result) == 4


# ---------------------------------------------------------------------------
# grid_py usage: ensure correct viewport context
# ---------------------------------------------------------------------------

class TestGridPyUsagePatterns:
    """Verify that complexheatmap uses grid_py correctly."""

    def test_convert_width_needs_context(self):
        """convert_width for npc/native needs a viewport context."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        npc_unit = grid_py.Unit(0.5, "npc")
        mm_val = grid_py.convert_width(npc_unit, "mm", valueOnly=True)
        assert float(np.squeeze(mm_val)) > 0

    def test_absolute_units_context_independent(self):
        """mm, cm, inches convert correctly without viewport."""
        cm_unit = grid_py.Unit(2, "cm")
        mm_val = grid_py.convert_width(cm_unit, "mm", valueOnly=True)
        assert abs(float(np.squeeze(mm_val)) - 20.0) < 0.1

    def test_text_grob_with_gpar(self):
        """text_grob should respect Gpar fontsize for measurement."""
        from grid_py._primitives import text_grob
        from grid_py._size import width_details
        g10 = text_grob(label="X", x=0.5, y=0.5, gp=grid_py.Gpar(fontsize=10))
        g20 = text_grob(label="X", x=0.5, y=0.5, gp=grid_py.Gpar(fontsize=20))
        w10 = float(np.squeeze(grid_py.convert_width(width_details(g10), "mm", valueOnly=True)))
        w20 = float(np.squeeze(grid_py.convert_width(width_details(g20), "mm", valueOnly=True)))
        assert w20 > w10 * 1.5  # ~2x larger

    def test_viewport_native_conversion(self):
        """Native units should convert based on xscale/yscale."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(
            xscale=(0, 200), yscale=(0, 100)
        ))
        # 100 native x out of 0-200 = 0.5 npc
        x = grid_py.convert_width(grid_py.Unit(100, "native"), "npc", valueOnly=True)
        assert abs(float(np.squeeze(x)) - 0.5) < 0.01
        grid_py.up_viewport()
