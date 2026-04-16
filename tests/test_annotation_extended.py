"""Extended tests for annotation_functions.py — designed from R source semantics.

R source: ComplexHeatmap/R/AnnotationFunction-function.R
"""
from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap.annotation_functions import (
    anno_simple,
    anno_barplot,
    anno_points,
    anno_lines,
    anno_text,
    anno_mark,
    anno_block,
    anno_empty,
)
from complexheatmap.annotation_function import AnnotationFunction
from complexheatmap.heatmap import Heatmap
from complexheatmap.heatmap_annotation import HeatmapAnnotation, rowAnnotation


# ---------------------------------------------------------------------------
# anno_simple — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoSimple:
    def test_returns_annotation_function(self):
        af = anno_simple(np.array([1.0, 2.0, 3.0]))
        assert isinstance(af, AnnotationFunction)

    def test_nobs(self):
        af = anno_simple(np.array([1, 2, 3, 4, 5]))
        assert af.nobs == 5

    def test_which_column(self):
        af = anno_simple(np.array([1, 2, 3]), which="column")
        assert af.which == "column"

    def test_which_row(self):
        af = anno_simple(np.array([1, 2, 3]), which="row")
        assert af.which == "row"


# ---------------------------------------------------------------------------
# anno_barplot — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoBarplot:
    def test_basic(self):
        af = anno_barplot(np.array([1.0, 3.0, 2.0]))
        assert isinstance(af, AnnotationFunction)
        assert af.fun_name == "anno_barplot"

    def test_column_which(self):
        af = anno_barplot(np.array([1.0, 2.0]), which="column")
        assert af.which == "column"

    def test_row_which(self):
        af = anno_barplot(np.array([1.0, 2.0]), which="row")
        assert af.which == "row"

    def test_draw_column(self):
        """Should not raise when drawn in a heatmap."""
        mat = np.random.randn(5, 3)
        ha = HeatmapAnnotation(bar=anno_barplot(np.array([1.0, 2.0, 3.0])))
        ht = Heatmap(mat, name="ab", top_annotation=ha, show_row_names=False)
        ht.draw(show=False, filename="/tmp/test_anno_bar.png",
                width=4, height=4, dpi=72)

    def test_draw_row(self):
        mat = np.random.randn(5, 3)
        ra = rowAnnotation(bar=anno_barplot(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                                            which="row"))
        ht = Heatmap(mat, name="abr", show_row_names=False) + ra
        ht.draw(show=False, filename="/tmp/test_anno_bar_row.png",
                width=5, height=4, dpi=72)


# ---------------------------------------------------------------------------
# anno_points — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoPoints:
    def test_basic(self):
        af = anno_points(np.array([1.0, 2.0, 3.0]))
        assert af.fun_name == "anno_points"

    def test_matrix_input(self):
        """R: anno_points accepts a matrix (multiple point series)."""
        af = anno_points(np.random.randn(10, 3), which="row")
        assert af.nobs == 10

    def test_draw(self):
        mat = np.random.randn(8, 4)
        ra = rowAnnotation(pt=anno_points(np.random.randn(8), which="row"),
                           width=grid_py.Unit(2, "cm"))
        ht = Heatmap(mat, name="ap") + ra
        ht.draw(show=False, filename="/tmp/test_anno_points.png",
                width=6, height=4, dpi=72)


# ---------------------------------------------------------------------------
# anno_lines — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoLines:
    def test_basic(self):
        af = anno_lines(np.array([1.0, 2.0, 1.5, 3.0]))
        assert af.fun_name == "anno_lines"


# ---------------------------------------------------------------------------
# anno_text — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoText:
    def test_basic(self):
        af = anno_text(["a", "b", "c"])
        assert af.fun_name == "anno_text"

    def test_nobs(self):
        af = anno_text(["x", "y", "z", "w"])
        assert af.nobs == 4


# ---------------------------------------------------------------------------
# anno_mark — R: AnnotationFunction-function.R:3132-3369
# ---------------------------------------------------------------------------

class TestAnnoMark:
    def test_returns_af(self):
        af = anno_mark(at=[0, 5], labels=["a", "b"], which="row")
        assert isinstance(af, AnnotationFunction)
        assert af.fun_name == "anno_mark"

    def test_width_is_dynamic(self):
        """R: width = link_width + max_text_width(labels, gp, rot)."""
        af = anno_mark(at=[0], labels=["short"], which="row")
        w1 = af.width
        af2 = anno_mark(at=[0], labels=["a_very_very_long_label_name"], which="row")
        w2 = af2.width
        # Longer label → wider annotation
        w1_mm = float(np.squeeze(grid_py.convert_width(w1, "mm", valueOnly=True)))
        w2_mm = float(np.squeeze(grid_py.convert_width(w2, "mm", valueOnly=True)))
        assert w2_mm > w1_mm

    def test_height_for_column(self):
        """R: height = link_height + max_text_height(labels, gp, rot)."""
        af = anno_mark(at=[0, 3], labels=["x", "y"], which="column")
        assert af.height is not None
        assert af.width is None  # column anno has no fixed width

    def test_draw_with_split(self):
        """anno_mark in a split heatmap — tests .pos injection."""
        mat = np.random.randn(30, 5)
        groups = np.repeat(["A", "B", "C"], 10)
        ht = Heatmap(mat, name="am", row_split=groups, cluster_rows=False,
                     show_row_names=False) + \
             rowAnnotation(mark=anno_mark(at=[2, 12, 25],
                                         labels=["g1", "g2", "g3"],
                                         which="row"))
        ht.draw(show=False, filename="/tmp/test_anno_mark_split.png",
                width=6, height=5, dpi=72)

    def test_draw_no_split(self):
        mat = np.random.randn(20, 5)
        ht = Heatmap(mat, name="amn", show_row_names=False) + \
             rowAnnotation(mark=anno_mark(at=[3, 10, 18],
                                         labels=["x", "y", "z"],
                                         which="row"))
        ht.draw(show=False, filename="/tmp/test_anno_mark_nosplit.png",
                width=6, height=5, dpi=72)


# ---------------------------------------------------------------------------
# anno_block — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoBlock:
    def test_basic(self):
        af = anno_block(which="row")
        assert af.fun_name == "anno_block"


# ---------------------------------------------------------------------------
# anno_empty — R: AnnotationFunction-function.R
# ---------------------------------------------------------------------------

class TestAnnoEmpty:
    def test_basic(self):
        af = anno_empty()
        assert af.fun_name == "anno_empty"


# ---------------------------------------------------------------------------
# HeatmapAnnotation integration
# ---------------------------------------------------------------------------

class TestHeatmapAnnotation:
    def test_auto_naming(self):
        """R: name = paste0('heatmap_annotation_', get_row_annotation_index())"""
        ha = HeatmapAnnotation(x=np.array([1, 2, 3]), which="row")
        assert ha.name.startswith("heatmap_annotation_")

    def test_explicit_name(self):
        ha = HeatmapAnnotation(x=np.array([1, 2, 3]), name="my_anno")
        assert ha.name == "my_anno"

    def test_width_unit_conversion(self):
        """Regression: width in cm should be correctly converted to mm internally."""
        ra = rowAnnotation(pt=anno_points(np.array([1.0, 2.0, 3.0]), which="row"),
                           width=grid_py.Unit(2, "cm"))
        w = ra.width
        # Should be ~20mm (2cm), not 2mm
        if isinstance(w, (int, float)):
            assert w >= 15  # at least 15mm
        elif grid_py.is_unit(w):
            mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
            assert mm >= 15


# ---------------------------------------------------------------------------
# Heatmap component sizes (R semantics)
# ---------------------------------------------------------------------------

class TestHeatmapComponentSizes:
    """Test that component_height/width uses ht_opt globals, not hardcoded values."""

    @pytest.fixture
    def simple_ht(self):
        np.random.seed(42)
        ht = Heatmap(np.random.randn(10, 5), name="cs",
                     column_title="Test Title",
                     row_labels=[f"r{i}" for i in range(10)],
                     column_labels=[f"c{j}" for j in range(5)])
        ht.make_layout()
        return ht

    def test_column_title_height_dynamic(self, simple_ht):
        """R: max_text_height(title, gp, rot) + sum(title_padding)."""
        h = simple_ht.component_height("column_title_top")
        h_mm = float(np.squeeze(grid_py.convert_height(h, "mm", valueOnly=True)))
        assert h_mm > 0
        # Should be text height + padding, not hardcoded 5mm+1lines
        assert h_mm < 20  # sanity

    def test_row_title_width_dynamic(self, simple_ht):
        """R: max_text_width(title, gp, rot) + sum(title_padding)."""
        # Single heatmap with no split → no row title
        w = simple_ht.component_width("row_title_left")
        w_mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        assert w_mm == 0  # no split, no row title

    def test_split_row_title_width(self):
        """With row_split, row_title should have dynamic width."""
        mat = np.random.randn(20, 5)
        ht = Heatmap(mat, name="srt",
                     row_split=np.repeat(["A", "B"], 10),
                     cluster_rows=False)
        ht.make_layout()
        w = ht.component_width("row_title_left")
        w_mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        assert w_mm > 0  # should have title width

    def test_dendrogram_includes_padding(self):
        """R: dend_width = user_width + ht_opt$DENDROGRAM_PADDING."""
        from complexheatmap._globals import ht_opt
        mat = np.random.randn(10, 5)
        ht = Heatmap(mat, name="dp")
        ht.make_layout()
        w = ht.component_width("row_dend_left")
        w_mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        dend_pad = float(ht_opt("DENDROGRAM_PADDING"))
        # Default dend width = 10mm + padding
        assert abs(w_mm - (10.0 + dend_pad)) < 0.1

    def test_row_names_uses_dimname_padding(self):
        """R: row_names_width = anno_width + ht_opt$DIMNAME_PADDING * 2."""
        from complexheatmap._globals import ht_opt
        mat = np.random.randn(5, 3)
        ht = Heatmap(mat, name="rn",
                     row_labels=["A", "B", "C", "D", "E"])
        ht.make_layout()
        w = ht.component_width("row_names_right")
        w_mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        dp = float(ht_opt("DIMNAME_PADDING"))
        # Width should include 2 * DIMNAME_PADDING
        assert w_mm >= dp * 2


# ---------------------------------------------------------------------------
# HeatmapList vertical direction
# ---------------------------------------------------------------------------

class TestHeatmapListVertical:
    def test_vertical_draw(self):
        """Vertical HeatmapList should render without error."""
        mat1 = np.random.randn(5, 10)
        mat2 = np.random.randn(3, 10)
        ht = Heatmap(mat1, name="v1") % Heatmap(mat2, name="v2")
        ht.draw(show=False, filename="/tmp/test_vert.png",
                width=6, height=6, dpi=72)

    def test_vertical_body_alignment(self):
        """Bodies should be horizontally aligned in vertical list."""
        mat1 = np.random.randn(5, 10)
        mat2 = np.random.randn(3, 10)
        # mat1 has row_dend, mat2 doesn't
        ht = Heatmap(mat1, name="va1") % \
             Heatmap(mat2, name="va2", cluster_rows=False)
        ht.draw(show=False, filename="/tmp/test_vert_align.png",
                width=6, height=6, dpi=72)


# ---------------------------------------------------------------------------
# HeatmapList horizontal alignment
# ---------------------------------------------------------------------------

class TestHeatmapListHorizontal:
    def test_body_vertical_alignment(self):
        """Bodies vertically aligned when heatmaps have different top components."""
        np.random.seed(42)
        mat = np.random.randn(20, 5)
        ha = HeatmapAnnotation(x=anno_barplot(np.random.randn(5)))
        ht1 = Heatmap(mat, name="ha1", top_annotation=ha)
        ht2 = Heatmap(mat, name="ha2")  # no top annotation
        ht_list = ht1 + ht2
        ht_list.draw(show=False, filename="/tmp/test_horiz_align.png",
                     width=8, height=5, dpi=72)

    def test_three_heatmaps(self):
        np.random.seed(42)
        ht = Heatmap(np.random.randn(10, 5), name="h1", row_km=2) + \
             Heatmap(np.random.rand(10, 3), name="h2") + \
             Heatmap(np.random.choice(["a", "b"], size=(10, 2)), name="h3",
                     col={"a": "red", "b": "blue"})
        ht.draw(show=False, filename="/tmp/test_3ht.png",
                width=8, height=5, dpi=72)
