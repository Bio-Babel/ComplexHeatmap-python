"""Tests translated from R's ComplexHeatmap test suite.

Each test case maps to a specific R test, with the R source reference
in the docstring. These exercise real usage patterns from the R package.
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


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_ht_opt()


@pytest.fixture
def mat():
    """R test-Heatmap-class.R:6-17 — structured block matrix."""
    np.random.seed(123)
    nr1, nr2, nr3 = 10, 8, 6
    nc1, nc2, nc3 = 6, 8, 10
    m = np.vstack([
        np.hstack([np.random.normal(1, 0.5, (nr1, nc1)),
                    np.random.normal(0, 0.5, (nr1, nc2)),
                    np.random.normal(0.5, 0.5, (nr1, nc3))]),
        np.hstack([np.random.normal(0, 0.5, (nr2, nc1)),
                    np.random.normal(1, 0.5, (nr2, nc2)),
                    np.random.normal(0.5, 0.5, (nr2, nc3))]),
        np.hstack([np.random.normal(0, 0.5, (nr3, nc1)),
                    np.random.normal(0, 0.5, (nr3, nc2)),
                    np.random.normal(1, 0.5, (nr3, nc3))]),
    ])
    return m


# ---------------------------------------------------------------------------
# R test-Heatmap-class.R: basic draw tests
# ---------------------------------------------------------------------------

class TestHeatmapClassBasic:
    """R test-Heatmap-class.R:22-37"""

    def test_basic_draw(self, mat):
        Heatmap(mat, name="t1").draw(
            show=False, filename="/tmp/rt_basic.png", width=5, height=5, dpi=72)

    def test_custom_col(self, mat):
        Heatmap(mat, name="t2",
                col=color_ramp2([-3, 0, 3], ["green", "white", "red"])).draw(
            show=False, filename="/tmp/rt_col.png", width=5, height=5, dpi=72)

    def test_border(self, mat):
        Heatmap(mat, name="t3", border=True).draw(
            show=False, filename="/tmp/rt_border.png", width=5, height=5, dpi=72)


class TestHeatmapTitle:
    """R test-Heatmap-class.R:39-71"""

    def test_row_title(self, mat):
        Heatmap(mat, name="tt1", row_title="blablabla").draw(
            show=False, filename="/tmp/rt_rtitle.png", width=5, height=5, dpi=72)

    def test_row_title_right(self, mat):
        Heatmap(mat, name="tt2", row_title="right title",
                row_title_side="right").draw(
            show=False, filename="/tmp/rt_rtitle_r.png", width=5, height=5, dpi=72)

    def test_row_title_rot0(self, mat):
        Heatmap(mat, name="tt3", row_title="horizontal",
                row_title_rot=0).draw(
            show=False, filename="/tmp/rt_rtitle_rot0.png", width=5, height=5, dpi=72)

    def test_column_title(self, mat):
        Heatmap(mat, name="tt4", column_title="col title").draw(
            show=False, filename="/tmp/rt_ctitle.png", width=5, height=5, dpi=72)

    def test_column_title_bottom(self, mat):
        Heatmap(mat, name="tt5", column_title="bottom",
                column_title_side="bottom").draw(
            show=False, filename="/tmp/rt_ctitle_bot.png", width=5, height=5, dpi=72)

    def test_column_title_rot90(self, mat):
        Heatmap(mat, name="tt6", column_title="rotated",
                column_title_rot=90).draw(
            show=False, filename="/tmp/rt_ctitle_rot.png", width=5, height=5, dpi=72)


class TestHeatmapClustering:
    """R test-Heatmap-class.R:74-140"""

    def test_no_cluster(self, mat):
        Heatmap(mat, name="cl1", cluster_rows=False).draw(
            show=False, filename="/tmp/rt_nocl.png", width=5, height=5, dpi=72)

    def test_pearson_distance(self, mat):
        Heatmap(mat, name="cl2",
                clustering_distance_rows="pearson").draw(
            show=False, filename="/tmp/rt_pearson.png", width=5, height=5, dpi=72)

    def test_single_linkage(self, mat):
        Heatmap(mat, name="cl3",
                clustering_method_rows="single").draw(
            show=False, filename="/tmp/rt_single.png", width=5, height=5, dpi=72)

    def test_dend_right(self, mat):
        Heatmap(mat, name="cl4", row_dend_side="right").draw(
            show=False, filename="/tmp/rt_dend_r.png", width=5, height=5, dpi=72)

    def test_dend_width(self, mat):
        Heatmap(mat, name="cl5",
                row_dend_width=grid_py.Unit(4, "cm")).draw(
            show=False, filename="/tmp/rt_dend_w.png", width=6, height=5, dpi=72)


class TestHeatmapRowColNames:
    """R test-Heatmap-class.R:158-202"""

    def test_no_names_matrix(self, mat):
        """R: Heatmap(unname(mat))"""
        Heatmap(mat, name="nm1").draw(
            show=False, filename="/tmp/rt_noname.png", width=5, height=5, dpi=72)

    def test_hide_row_names(self, mat):
        Heatmap(mat, name="nm2", show_row_names=False).draw(
            show=False, filename="/tmp/rt_hide_rn.png", width=5, height=5, dpi=72)

    def test_row_names_left(self, mat):
        Heatmap(mat, name="nm3", row_names_side="left").draw(
            show=False, filename="/tmp/rt_rn_left.png", width=5, height=5, dpi=72)

    def test_custom_labels(self, mat):
        Heatmap(mat, name="nm4",
                row_labels=[f"r{i}" for i in range(mat.shape[0])]).draw(
            show=False, filename="/tmp/rt_labels.png", width=5, height=5, dpi=72)

    def test_row_names_rot(self, mat):
        Heatmap(mat, name="nm5", row_names_rot=45).draw(
            show=False, filename="/tmp/rt_rn_rot.png", width=5, height=5, dpi=72)

    def test_column_names_top(self, mat):
        Heatmap(mat, name="nm6", column_names_side="top").draw(
            show=False, filename="/tmp/rt_cn_top.png", width=5, height=5, dpi=72)


class TestHeatmapSplit:
    """R test-Heatmap-class.R:219-318"""

    def test_row_km(self, mat):
        Heatmap(mat, name="sp1", row_km=3).draw(
            show=False, filename="/tmp/rt_km.png", width=5, height=5, dpi=72)

    def test_row_split_factor(self, mat):
        groups = np.array(["A"] * 6 + ["B"] * 18)
        Heatmap(mat, name="sp2", row_split=groups).draw(
            show=False, filename="/tmp/rt_split.png", width=5, height=5, dpi=72)

    def test_row_split_with_gap(self, mat):
        groups = np.repeat(["A", "B"], 12)
        Heatmap(mat, name="sp3", row_split=groups,
                row_gap=grid_py.Unit(5, "mm")).draw(
            show=False, filename="/tmp/rt_gap.png", width=5, height=6, dpi=72)

    def test_row_km_with_title(self, mat):
        Heatmap(mat, name="sp4", row_km=3, row_title="foo").draw(
            show=False, filename="/tmp/rt_km_title.png", width=5, height=5, dpi=72)

    def test_row_km_title_rot0(self, mat):
        Heatmap(mat, name="sp5", row_km=3,
                row_title_rot=0).draw(
            show=False, filename="/tmp/rt_km_rot0.png", width=5, height=5, dpi=72)

    def test_column_km(self, mat):
        Heatmap(mat, name="sp6", column_km=2).draw(
            show=False, filename="/tmp/rt_ckm.png", width=5, height=5, dpi=72)

    def test_column_split_with_gap(self, mat):
        groups = np.array(["A"] * 6 + ["B"] * 18)
        Heatmap(mat, name="sp7", column_split=groups,
                column_gap=grid_py.Unit(1, "cm")).draw(
            show=False, filename="/tmp/rt_csplit.png", width=6, height=5, dpi=72)

    def test_row_split_integer(self, mat):
        """R: Heatmap(mat, row_split=2) — integer split = km"""
        Heatmap(mat, name="sp8", row_split=2).draw(
            show=False, filename="/tmp/rt_int_split.png", width=5, height=5, dpi=72)


class TestHeatmapAnnotations:
    """R test-Heatmap-class.R:204-217"""

    def test_top_annotation(self, mat):
        anno = HeatmapAnnotation(
            foo=np.arange(1, mat.shape[1] + 1, dtype=float),
            bar=anno_barplot(np.arange(mat.shape[1], 0, -1, dtype=float)),
        )
        Heatmap(mat, name="an1", top_annotation=anno).draw(
            show=False, filename="/tmp/rt_topanno.png", width=5, height=6, dpi=72)

    def test_bottom_annotation(self, mat):
        anno = HeatmapAnnotation(
            foo=np.arange(1, mat.shape[1] + 1, dtype=float),
        )
        Heatmap(mat, name="an2", bottom_annotation=anno).draw(
            show=False, filename="/tmp/rt_botanno.png", width=5, height=6, dpi=72)

    def test_both_annotations(self, mat):
        anno = HeatmapAnnotation(
            val=np.arange(1, mat.shape[1] + 1, dtype=float),
        )
        Heatmap(mat, name="an3",
                top_annotation=anno, bottom_annotation=anno).draw(
            show=False, filename="/tmp/rt_bothanno.png", width=5, height=6, dpi=72)


# ---------------------------------------------------------------------------
# R test-AnnotationFunction.R: anno_* draw smoke tests
# ---------------------------------------------------------------------------

class TestAnnotationFunctionDraw:
    """R test-AnnotationFunction.R:61-103"""

    def test_anno_simple_vector(self):
        anno = anno_simple(np.arange(1, 11, dtype=float))
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af1",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_asimple.png", width=4, height=4, dpi=72)

    def test_anno_simple_with_na(self):
        """R: anno_simple(c(1:9, NA)) — NA values should not crash."""
        vals = np.arange(1, 11, dtype=float)
        vals[9] = np.nan
        anno = anno_simple(vals)
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af2",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_asimple_na.png", width=4, height=4, dpi=72)

    def test_anno_simple_matrix(self):
        """R: anno_simple(cbind(1:10, 10:1)) — matrix annotation."""
        m = np.column_stack([np.arange(1, 11), np.arange(10, 0, -1)])
        anno = anno_simple(m.astype(float))
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af3",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_asimple_mat.png", width=4, height=4, dpi=72)

    def test_anno_empty(self):
        """R: anno_empty()"""
        anno = anno_empty()
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af4",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_aempty.png", width=4, height=4, dpi=72)

    def test_anno_barplot_basic(self):
        anno = anno_barplot(np.arange(1, 11, dtype=float))
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af5",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_abar.png", width=4, height=5, dpi=72)

    def test_anno_points_basic(self):
        anno = anno_points(np.random.randn(10))
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af6",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_apoints.png", width=4, height=5, dpi=72)

    def test_anno_lines_basic(self):
        anno = anno_lines(np.random.randn(10))
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af7",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_alines.png", width=4, height=5, dpi=72)

    def test_anno_text_basic(self):
        anno = anno_text([f"s{i}" for i in range(10)])
        ha = HeatmapAnnotation(foo=anno)
        Heatmap(np.random.randn(10, 5), name="af8",
                top_annotation=ha).draw(
            show=False, filename="/tmp/rt_atext.png", width=4, height=5, dpi=72)

    def test_anno_block_with_split(self):
        """R: anno_block() works with row_split."""
        mat = np.random.randn(20, 5)
        groups = np.repeat(["A", "B"], 10)
        ha = HeatmapAnnotation(block=anno_block(which="row"), which="row")
        Heatmap(mat, name="af9", row_split=groups,
                left_annotation=ha, cluster_rows=False).draw(
            show=False, filename="/tmp/rt_ablock.png", width=5, height=5, dpi=72)


# ---------------------------------------------------------------------------
# R test-HeatmapList-class.R: combination tests
# ---------------------------------------------------------------------------

class TestHeatmapListFromR:
    """R test-HeatmapList-class.R"""

    def test_two_heatmaps_basic(self, mat):
        (Heatmap(mat, name="hl1") + Heatmap(mat, name="hl2")).draw(
            show=False, filename="/tmp/rt_hl2.png", width=8, height=5, dpi=72)

    def test_heatmap_plus_annotation(self, mat):
        (Heatmap(mat, name="hl3") +
         rowAnnotation(foo=anno_barplot(np.random.randn(mat.shape[0]),
                                       which="row"))).draw(
            show=False, filename="/tmp/rt_hl_anno.png", width=6, height=5, dpi=72)

    def test_three_heatmaps_different_widths(self, mat):
        """R test-HeatmapList-class.R:78-85"""
        (Heatmap(mat[:, :6], name="hl4") +
         Heatmap(mat[:, 6:14], name="hl5") +
         Heatmap(mat[:, 14:], name="hl6")).draw(
            show=False, filename="/tmp/rt_hl3.png", width=9, height=5, dpi=72)

    def test_split_synced_in_list(self, mat):
        """Critical: row_split must sync across all heatmaps."""
        groups = np.array(["A"] * 12 + ["B"] * 12)
        ht_list = Heatmap(mat, name="hs1", row_split=groups,
                          cluster_rows=False) + \
                  Heatmap(mat, name="hs2")
        ht_list.make_layout()
        n1 = len(ht_list.ht_list[0]._row_order_list)
        n2 = len(ht_list.ht_list[1]._row_order_list)
        assert n1 == n2 == 2, f"Split not synced: {n1} vs {n2}"

    def test_split_draw(self, mat):
        groups = np.array(["A"] * 12 + ["B"] * 12)
        (Heatmap(mat, name="hsd1", row_split=groups, cluster_rows=False) +
         Heatmap(mat, name="hsd2")).draw(
            show=False, filename="/tmp/rt_hl_split.png", width=8, height=6, dpi=72)

    def test_km_synced_draw(self, mat):
        """row_km on main → synced to secondary."""
        (Heatmap(mat, name="hkm1", row_km=3) +
         Heatmap(mat, name="hkm2")).draw(
            show=False, filename="/tmp/rt_hl_km.png", width=8, height=6, dpi=72)

    def test_global_titles(self, mat):
        (Heatmap(mat, name="hgt1") + Heatmap(mat, name="hgt2")).draw(
            column_title="Global Col Title", row_title="Global Row Title",
            show=False, filename="/tmp/rt_hl_gt.png", width=8, height=6, dpi=72)

    def test_vertical_list(self, mat):
        (Heatmap(mat, name="hv1") % Heatmap(mat, name="hv2")).draw(
            show=False, filename="/tmp/rt_vert.png", width=5, height=8, dpi=72)
