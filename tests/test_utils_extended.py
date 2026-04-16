"""Extended tests for _utils.py — designed from R source semantics.

R source reference: ComplexHeatmap/R/utils.R, box_align.R
"""
from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap._utils import (
    pindex,
    subset_gp,
    max_text_width,
    max_text_height,
    is_abs_unit,
    list_to_matrix,
    restore_matrix,
    smart_align,
    default_axis_param,
)


# ---------------------------------------------------------------------------
# max_text_width / max_text_height — R: utils.R:393-439
# ---------------------------------------------------------------------------

class TestMaxTextWidth:
    """R: max_text_width(text, gp, rot) = convertWidth(max(grobWidth(textGrob(rot=rot))), 'mm')"""

    def test_single_string(self):
        w = max_text_width("hello")
        mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        assert mm > 0

    def test_multiple_returns_max(self):
        w = max_text_width(["a", "hello world", "hi"])
        mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        w_short = max_text_width(["a"])
        mm_short = float(np.squeeze(grid_py.convert_width(w_short, "mm", valueOnly=True)))
        assert mm > mm_short

    def test_empty_returns_zero(self):
        w = max_text_width([])
        mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        assert mm == 0.0

    def test_rot90_swaps_dimensions(self):
        """R: grobWidth(textGrob('hello', rot=90)) ≈ grobHeight(textGrob('hello', rot=0))"""
        w0 = max_text_width(["hello"], rot=0)
        w90 = max_text_width(["hello"], rot=90)
        h0 = max_text_height(["hello"], rot=0)
        w0_mm = float(np.squeeze(grid_py.convert_width(w0, "mm", valueOnly=True)))
        w90_mm = float(np.squeeze(grid_py.convert_width(w90, "mm", valueOnly=True)))
        h0_mm = float(np.squeeze(grid_py.convert_height(h0, "mm", valueOnly=True)))
        # w(rot=90) should equal h(rot=0) — text height becomes width
        assert abs(w90_mm - h0_mm) < 0.5

    def test_returns_unit_in_mm(self):
        w = max_text_width(["test"])
        assert grid_py.is_unit(w)
        utype = grid_py.unit_type(w)
        assert utype == "mm" or (isinstance(utype, list) and utype[0] == "mm")

    def test_returns_single_element(self):
        """Should return a single Unit, not multi-element."""
        w = max_text_width(["a", "bb", "ccc"])
        mm = grid_py.convert_width(w, "mm", valueOnly=True)
        assert np.atleast_1d(mm).shape == (1,)


class TestMaxTextHeight:
    def test_single_string(self):
        h = max_text_height("X")
        mm = float(np.squeeze(grid_py.convert_height(h, "mm", valueOnly=True)))
        assert mm > 0

    def test_rot90_equals_width_rot0(self):
        """R: grobHeight(textGrob('hello', rot=90)) ≈ grobWidth(textGrob('hello', rot=0))"""
        h90 = max_text_height(["hello"], rot=90)
        w0 = max_text_width(["hello"], rot=0)
        h90_mm = float(np.squeeze(grid_py.convert_height(h90, "mm", valueOnly=True)))
        w0_mm = float(np.squeeze(grid_py.convert_width(w0, "mm", valueOnly=True)))
        assert abs(h90_mm - w0_mm) < 0.5


# ---------------------------------------------------------------------------
# pindex — R: utils.R:799-821
# ---------------------------------------------------------------------------

class TestPindex:
    def test_diagonal(self):
        """R: pindex(m, 1:4, 1:4) extracts diagonal (1-based in R)."""
        m = np.arange(16).reshape(4, 4)
        result = pindex(m, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        assert np.array_equal(result, np.array([0, 5, 10, 15]))

    def test_recycle_i(self):
        """R: pindex(m, 1, 1:4) recycles i to length of j."""
        m = np.arange(12).reshape(3, 4)
        result = pindex(m, np.array([0]), np.array([0, 1, 2, 3]))
        assert np.array_equal(result, m[0, :])

    def test_recycle_j(self):
        """R: pindex(m, 1:3, 1) recycles j to length of i."""
        m = np.arange(12).reshape(3, 4)
        result = pindex(m, np.array([0, 1, 2]), np.array([0]))
        assert np.array_equal(result, m[:, 0])

    def test_arbitrary_indices(self):
        m = np.arange(20).reshape(4, 5)
        result = pindex(m, np.array([0, 3, 1]), np.array([4, 2, 0]))
        assert np.array_equal(result, np.array([4, 17, 5]))


# ---------------------------------------------------------------------------
# subset_gp — R: utils.R
# ---------------------------------------------------------------------------

class TestSubsetGp:
    def test_vector_entry(self):
        gp = {"col": ["red", "blue", "green"], "lwd": 2}
        result = subset_gp(gp, [0, 2])
        assert result["col"] == ["red", "green"]
        assert result["lwd"] == 2

    def test_single_index(self):
        gp = {"col": ["red", "blue"], "fill": ["#FF0000", "#0000FF"]}
        result = subset_gp(gp, 0)
        assert result["col"] == ["red"]
        assert result["fill"] == ["#FF0000"]

    def test_scalar_preserved(self):
        gp = {"fontsize": 12, "col": ["a", "b"]}
        result = subset_gp(gp, [1])
        assert result["fontsize"] == 12


# ---------------------------------------------------------------------------
# smart_align — R: box_align.R smartAlign2
# ---------------------------------------------------------------------------

class TestSmartAlign:
    def test_no_overlap(self):
        """Non-overlapping intervals should stay unchanged."""
        h1 = np.array([1.0, 3.0, 5.0])
        h2 = np.array([2.0, 4.0, 6.0])
        result = smart_align(h1, h2, (0, 10))
        # Centers should be preserved
        centers = (result[:, 0] + result[:, 1]) / 2
        original_centers = np.array([1.5, 3.5, 5.5])
        assert np.allclose(centers, original_centers, atol=0.1)

    def test_overlapping_intervals(self):
        """Overlapping intervals should be separated."""
        h1 = np.array([1.0, 1.5])
        h2 = np.array([3.0, 3.5])
        result = smart_align(h1, h2, (0, 10))
        # After alignment, no overlap
        assert result[0, 1] <= result[1, 0] or result[1, 1] <= result[0, 0]

    def test_bounds_respected(self):
        """Results should stay within bounds."""
        h1 = np.array([0.0, 0.5, 1.0])
        h2 = np.array([2.0, 2.5, 3.0])
        result = smart_align(h1, h2, (0, 5))
        assert np.all(result[:, 0] >= 0)
        assert np.all(result[:, 1] <= 5)

    def test_empty(self):
        result = smart_align(np.array([]), np.array([]), (0, 10))
        assert result.shape == (0, 2)

    def test_single_interval(self):
        result = smart_align(np.array([2.0]), np.array([4.0]), (0, 10))
        assert result.shape == (1, 2)
        assert abs(result[0, 1] - result[0, 0] - 2.0) < 0.01

    def test_overflow_path(self):
        """When total height exceeds range, R distributes evenly."""
        h1 = np.array([0.0, 0.0])
        h2 = np.array([6.0, 6.0])  # total=12, range=10
        result = smart_align(h1, h2, (0, 10))
        assert result.shape == (2, 2)
        # Heights preserved
        for row in range(2):
            assert abs((result[row, 1] - result[row, 0]) - 6.0) < 0.01


# ---------------------------------------------------------------------------
# is_abs_unit
# ---------------------------------------------------------------------------

class TestIsAbsUnit:
    def test_number(self):
        assert is_abs_unit(5.0) is True
        assert is_abs_unit(3) is True

    def test_unit_mm(self):
        assert is_abs_unit(grid_py.Unit(5, "mm")) is True

    def test_unit_npc(self):
        assert is_abs_unit(grid_py.Unit(0.5, "npc")) is False


# ---------------------------------------------------------------------------
# list_to_matrix — R: Upset.R:324-338
# ---------------------------------------------------------------------------

class TestListToMatrix:
    def test_basic(self):
        lt = {"A": {"x", "y"}, "B": {"y", "z"}}
        mat, rows, cols = list_to_matrix(lt)
        assert set(rows) == {"x", "y", "z"}
        assert set(cols) == {"A", "B"}
        y_idx = rows.index("y")
        assert mat[y_idx, cols.index("A")] == 1
        assert mat[y_idx, cols.index("B")] == 1

    def test_universal_set(self):
        lt = {"A": {"x"}, "B": {"y"}}
        mat, rows, cols = list_to_matrix(lt, universal_set=["x", "y", "z"])
        assert len(rows) == 3
        z_idx = rows.index("z")
        assert mat[z_idx, cols.index("A")] == 0
        assert mat[z_idx, cols.index("B")] == 0

    def test_empty_set(self):
        lt = {"A": set(), "B": {"x"}}
        mat, rows, cols = list_to_matrix(lt)
        assert mat[rows.index("x"), cols.index("A")] == 0
        assert mat[rows.index("x"), cols.index("B")] == 1


# ---------------------------------------------------------------------------
# restore_matrix
# ---------------------------------------------------------------------------

class TestRestoreMatrix:
    def test_basic(self):
        j = np.array([0, 0, 1, 1])
        i = np.array([0, 1, 0, 1])
        x = np.array([10.0, 20.0, 30.0, 40.0])
        y = np.array([100.0, 200.0, 300.0, 400.0])
        result = restore_matrix(j, i, x, y)
        assert result[0, 0] == 100.0
        assert result[1, 1] == 400.0


# ---------------------------------------------------------------------------
# default_axis_param
# ---------------------------------------------------------------------------

class TestDefaultAxisParam:
    def test_column(self):
        ap = default_axis_param("column")
        assert isinstance(ap, dict)
        assert "side" in ap

    def test_row(self):
        ap = default_axis_param("row")
        assert isinstance(ap, dict)
