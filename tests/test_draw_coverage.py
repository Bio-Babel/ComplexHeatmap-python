"""Draw-path coverage tests — exercise rendering pipelines.

These tests ensure that all major draw code paths execute without error,
covering annotation draw callbacks, heatmap component rendering, and
HeatmapList layout paths.
"""
from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap._color import color_ramp2
from complexheatmap.heatmap import Heatmap
from complexheatmap.heatmap_annotation import HeatmapAnnotation, rowAnnotation
from complexheatmap.annotation_functions import (
    anno_simple, anno_barplot, anno_points, anno_lines,
    anno_text, anno_histogram, anno_density, anno_mark,
    anno_block, anno_empty,
)
from complexheatmap._globals import ht_opt, reset_ht_opt


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset ht_opt after each test."""
    yield
    reset_ht_opt()


@pytest.fixture
def small_mat():
    np.random.seed(42)
    return np.random.randn(10, 5)


@pytest.fixture
def medium_mat():
    np.random.seed(42)
    return np.random.randn(30, 8)


# ---------------------------------------------------------------------------
# Single Heatmap draw paths
# ---------------------------------------------------------------------------

class TestHeatmapDraw:
    def test_basic_draw(self, small_mat):
        Heatmap(small_mat, name="d1").draw(
            show=False, filename="/tmp/t_draw1.png", width=4, height=4, dpi=72)

    def test_custom_col(self, small_mat):
        col = color_ramp2([-2, 0, 2], ["blue", "white", "red"])
        Heatmap(small_mat, name="d2", col=col).draw(
            show=False, filename="/tmp/t_draw2.png", width=4, height=4, dpi=72)

    def test_no_cluster(self, small_mat):
        Heatmap(small_mat, name="d3", cluster_rows=False,
                cluster_columns=False).draw(
            show=False, filename="/tmp/t_draw3.png", width=4, height=4, dpi=72)

    def test_row_km(self, small_mat):
        Heatmap(small_mat, name="d4", row_km=2).draw(
            show=False, filename="/tmp/t_draw4.png", width=4, height=4, dpi=72)

    def test_column_split(self, small_mat):
        Heatmap(small_mat, name="d5",
                column_split=np.array(["A", "A", "B", "B", "B"])).draw(
            show=False, filename="/tmp/t_draw5.png", width=4, height=4, dpi=72)

    def test_row_split_factor(self, medium_mat):
        groups = np.repeat(["X", "Y", "Z"], 10)
        Heatmap(medium_mat, name="d6", row_split=groups,
                cluster_rows=False).draw(
            show=False, filename="/tmp/t_draw6.png", width=4, height=5, dpi=72)

    def test_all_sides(self, small_mat):
        """Row/column names/dend on all sides."""
        Heatmap(small_mat, name="d7",
                row_names_side="left", column_names_side="top",
                row_dend_side="right", column_dend_side="bottom",
                column_title="Top Title", column_title_side="top",
                ).draw(
            show=False, filename="/tmp/t_draw7.png", width=5, height=5, dpi=72)

    def test_character_matrix(self):
        mat = np.array([["a", "b"], ["c", "a"], ["b", "c"]])
        Heatmap(mat, name="d8", col={"a": "red", "b": "blue", "c": "green"}).draw(
            show=False, filename="/tmp/t_draw8.png", width=3, height=3, dpi=72)

    def test_labels_override(self, small_mat):
        Heatmap(small_mat, name="d9",
                row_labels=[f"R{i}" for i in range(10)],
                column_labels=[f"C{j}" for j in range(5)]).draw(
            show=False, filename="/tmp/t_draw9.png", width=4, height=4, dpi=72)

    def test_border(self, small_mat):
        Heatmap(small_mat, name="d10", border=True).draw(
            show=False, filename="/tmp/t_draw10.png", width=4, height=4, dpi=72)


# ---------------------------------------------------------------------------
# Annotation draw paths
# ---------------------------------------------------------------------------

class TestAnnotationDraw:
    def test_top_barplot(self, small_mat):
        ha = HeatmapAnnotation(bar=anno_barplot(np.random.randn(5)))
        Heatmap(small_mat, name="a1", top_annotation=ha).draw(
            show=False, filename="/tmp/t_anno1.png", width=4, height=4, dpi=72)

    def test_bottom_annotation(self, small_mat):
        ha = HeatmapAnnotation(val=anno_points(np.random.randn(5)))
        Heatmap(small_mat, name="a2", bottom_annotation=ha).draw(
            show=False, filename="/tmp/t_anno2.png", width=4, height=4, dpi=72)

    def test_left_annotation(self, small_mat):
        ha = HeatmapAnnotation(val=anno_barplot(np.random.randn(10)),
                               which="row")
        Heatmap(small_mat, name="a3", left_annotation=ha).draw(
            show=False, filename="/tmp/t_anno3.png", width=5, height=4, dpi=72)

    def test_right_annotation(self, small_mat):
        ha = HeatmapAnnotation(val=anno_lines(np.random.randn(10)),
                               which="row")
        Heatmap(small_mat, name="a4", right_annotation=ha).draw(
            show=False, filename="/tmp/t_anno4.png", width=5, height=4, dpi=72)

    def test_simple_annotation(self, small_mat):
        ha = HeatmapAnnotation(
            grp=np.array(["A", "A", "B", "B", "B"]),
        )
        Heatmap(small_mat, name="a5", top_annotation=ha).draw(
            show=False, filename="/tmp/t_anno5.png", width=4, height=4, dpi=72)

    def test_text_annotation(self, small_mat):
        ha = HeatmapAnnotation(
            txt=anno_text([f"S{i}" for i in range(5)]),
        )
        Heatmap(small_mat, name="a6", top_annotation=ha).draw(
            show=False, filename="/tmp/t_anno6.png", width=4, height=4, dpi=72)

    def test_histogram_annotation(self, small_mat):
        ha = HeatmapAnnotation(
            hist=anno_histogram(small_mat.T, which="column"),
        )
        Heatmap(small_mat, name="a7", top_annotation=ha).draw(
            show=False, filename="/tmp/t_anno7.png", width=4, height=5, dpi=72)

    def test_density_annotation(self, small_mat):
        ha = HeatmapAnnotation(
            dens=anno_density(small_mat.T, which="column"),
        )
        Heatmap(small_mat, name="a8", top_annotation=ha).draw(
            show=False, filename="/tmp/t_anno8.png", width=4, height=5, dpi=72)

    def test_multiple_annotations(self, small_mat):
        ha = HeatmapAnnotation(
            bar=anno_barplot(np.random.randn(5)),
            grp=np.array(["X", "X", "Y", "Y", "Y"]),
        )
        Heatmap(small_mat, name="a9", top_annotation=ha).draw(
            show=False, filename="/tmp/t_anno9.png", width=4, height=5, dpi=72)


# ---------------------------------------------------------------------------
# HeatmapList draw paths
# ---------------------------------------------------------------------------

class TestHeatmapListDraw:
    def test_two_horizontal(self, small_mat):
        (Heatmap(small_mat, name="l1") +
         Heatmap(small_mat, name="l2")).draw(
            show=False, filename="/tmp/t_list1.png", width=7, height=4, dpi=72)

    def test_with_row_annotation(self, small_mat):
        (Heatmap(small_mat, name="l3") +
         rowAnnotation(pt=anno_points(np.random.randn(10), which="row"),
                       width=grid_py.Unit(1.5, "cm"))).draw(
            show=False, filename="/tmp/t_list2.png", width=6, height=4, dpi=72)

    def test_merge_legends(self, small_mat):
        (Heatmap(small_mat, name="l4") +
         Heatmap(small_mat, name="l5")).draw(
            merge_legends=True,
            show=False, filename="/tmp/t_list3.png", width=7, height=4, dpi=72)

    def test_vertical_two(self, small_mat):
        (Heatmap(small_mat, name="v1") %
         Heatmap(small_mat, name="v2")).draw(
            show=False, filename="/tmp/t_vert1.png", width=4, height=7, dpi=72)

    def test_global_title(self, small_mat):
        ht = Heatmap(small_mat, name="gt1") + Heatmap(small_mat, name="gt2")
        ht.draw(column_title="Global Title", row_title="Row",
                show=False, filename="/tmp/t_gtitle.png", width=7, height=5, dpi=72)

    def test_legend_side_bottom(self, small_mat):
        Heatmap(small_mat, name="lb").draw(
            heatmap_legend_side="bottom",
            show=False, filename="/tmp/t_legbot.png", width=5, height=5, dpi=72)


# ---------------------------------------------------------------------------
# ht_opt integration
# ---------------------------------------------------------------------------

class TestHtOpt:
    def test_set_and_read(self):
        from complexheatmap._globals import ht_opt
        old = ht_opt("DIMNAME_PADDING")
        ht_opt(DIMNAME_PADDING=3)
        assert ht_opt("DIMNAME_PADDING") == 3
        reset_ht_opt()
        assert ht_opt("DIMNAME_PADDING") == old

    def test_title_padding_list(self):
        """R: ht_opt$TITLE_PADDING = unit(c(bottom, top), 'mm')"""
        ht_opt(TITLE_PADDING=[5, 3])
        val = ht_opt("TITLE_PADDING")
        assert val == [5, 3]
        reset_ht_opt()


# ---------------------------------------------------------------------------
# color_ramp2 edge cases
# ---------------------------------------------------------------------------

class TestColorRamp2Extended:
    def test_out_of_range_clamp(self):
        col = color_ramp2([0, 1], ["white", "red"])
        assert col(-1) == col(0)  # clamp to min
        assert col(2) == col(1)   # clamp to max

    def test_na_handling(self):
        """R: col_fun(NA) → NA_character_. Python: → transparent '#FFFFFF00'."""
        col = color_ramp2([0, 1], ["white", "red"])
        result = col(np.nan)
        assert isinstance(result, str)
        assert result == "#FFFFFF00"  # fully transparent

    def test_vector_input(self):
        col = color_ramp2([0, 1], ["white", "red"])
        result = col(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3
        assert all(isinstance(c, str) for c in result)
