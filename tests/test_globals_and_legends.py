"""Tests for _globals (ht_opt) and legends — R source semantics.

R source: ComplexHeatmap/R/global.R, grid.Legend.R
"""
from __future__ import annotations

import numpy as np
import pytest
import grid_py

from complexheatmap._globals import ht_opt, reset_ht_opt
from complexheatmap._color import color_ramp2, add_transparency, rand_color
from complexheatmap.legends import Legends, pack_legend
from complexheatmap.color_mapping import ColorMapping
from complexheatmap.heatmap import Heatmap


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_ht_opt()


# ---------------------------------------------------------------------------
# ht_opt — R: global.R
# ---------------------------------------------------------------------------

class TestHtOpt:
    def test_read_all(self):
        """R: ht_opt() returns all options."""
        opts = ht_opt()
        assert isinstance(opts, dict)
        assert "TITLE_PADDING" in opts
        assert "DIMNAME_PADDING" in opts

    def test_read_single(self):
        assert isinstance(ht_opt("TITLE_PADDING"), (int, float))

    def test_set_keyword(self):
        """R: ht_opt$KEY = value"""
        ht_opt(DIMNAME_PADDING=5)
        assert ht_opt("DIMNAME_PADDING") == 5

    def test_reset(self):
        ht_opt(DIMNAME_PADDING=99)
        reset_ht_opt()
        assert ht_opt("DIMNAME_PADDING") != 99

    def test_context_manager(self):
        """R: with(ht_opt(...), ...) — context manager restores."""
        old = ht_opt("DIMNAME_PADDING")
        with ht_opt(DIMNAME_PADDING=42):
            assert ht_opt("DIMNAME_PADDING") == 42
        assert ht_opt("DIMNAME_PADDING") == old

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            ht_opt("NONEXISTENT_KEY")

    def test_all_defaults_exist(self):
        """All R defaults should be present."""
        for key in ["TITLE_PADDING", "DIMNAME_PADDING", "DENDROGRAM_PADDING",
                     "ROW_ANNO_PADDING", "COLUMN_ANNO_PADDING",
                     "HEATMAP_LEGEND_PADDING", "simple_anno_size",
                     "heatmap_row_names_gp", "heatmap_column_names_gp"]:
            val = ht_opt(key)
            assert val is not None, f"ht_opt('{key}') should not be None"


# ---------------------------------------------------------------------------
# color_ramp2 — R: circlize::colorRamp2
# ---------------------------------------------------------------------------

class TestColorRamp2:
    def test_basic_2breaks(self):
        col = color_ramp2([0, 1], ["white", "red"])
        assert col(0).upper() == "#FFFFFF"
        assert col(1).upper() == "#FF0000"

    def test_3breaks_midpoint(self):
        col = color_ramp2([-1, 0, 1], ["blue", "white", "red"])
        assert col(0).upper() == "#FFFFFF"

    def test_clamping(self):
        """R: values outside range are clamped."""
        col = color_ramp2([0, 1], ["white", "black"])
        assert col(-10) == col(0)
        assert col(10) == col(1)

    def test_nan_transparent(self):
        col = color_ramp2([0, 1], ["white", "red"])
        assert col(np.nan) == "#FFFFFF00"

    def test_vector(self):
        col = color_ramp2([0, 1], ["white", "red"])
        result = col(np.array([0.0, 0.5, 1.0]))
        assert isinstance(result, list)
        assert len(result) == 3

    def test_nan_in_vector(self):
        """NaN elements should produce transparent, non-NaN should be normal."""
        col = color_ramp2([0, 1], ["white", "red"])
        result = col(np.array([0.0, np.nan, 1.0]))
        assert result[0].upper() == "#FFFFFF"
        assert result[1] == "#FFFFFF00"
        assert result[2].upper() == "#FF0000"

    def test_metadata(self):
        col = color_ramp2([0, 0.5, 1], ["blue", "white", "red"])
        assert hasattr(col, 'breaks')
        assert hasattr(col, 'colors')
        assert hasattr(col, 'space')

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match="increasing"):
            color_ramp2([1, 0], ["red", "blue"])


# ---------------------------------------------------------------------------
# add_transparency — R: circlize::add_transparency
# ---------------------------------------------------------------------------

class TestAddTransparency:
    def test_basic(self):
        result = add_transparency("red", 0.5)
        assert result.startswith("#")
        assert len(result) == 9  # #RRGGBBAA

    def test_zero_transparent(self):
        """0 = fully opaque → alpha=FF."""
        result = add_transparency("#FF0000", 0.0)
        assert result.upper().endswith("FF")

    def test_full_transparent(self):
        """1 = fully transparent → alpha=00."""
        result = add_transparency("#FF0000", 1.0)
        assert result.upper().endswith("00")

    def test_vector(self):
        result = add_transparency(["red", "blue"], 0.5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# rand_color — R: circlize::rand_color
# ---------------------------------------------------------------------------

class TestRandColor:
    def test_count(self):
        assert len(rand_color(10)) == 10

    def test_hex_format(self):
        for c in rand_color(5):
            assert c.startswith("#")


# ---------------------------------------------------------------------------
# ColorMapping
# ---------------------------------------------------------------------------

class TestColorMapping:
    def test_continuous(self):
        cm = ColorMapping(
            col_fun=color_ramp2([0, 1], ["white", "red"]),
            name="test_cm",
        )
        assert cm.name == "test_cm"

    def test_discrete(self):
        cm = ColorMapping(
            colors={"a": "red", "b": "blue"},
            name="test_disc",
        )
        assert cm.name == "test_disc"


# ---------------------------------------------------------------------------
# Legends — R: grid.Legend.R
# ---------------------------------------------------------------------------

class TestLegends:
    def test_legend_in_draw(self):
        """Legends are created internally during draw()."""
        ht = Heatmap(np.random.randn(5, 3), name="lg_test")
        ht.draw(show=False, filename="/tmp/t_legend.png",
                width=4, height=4, dpi=72)

    def test_two_heatmap_legends(self):
        """Two heatmaps should produce two legends."""
        (Heatmap(np.random.randn(5, 3), name="p1") +
         Heatmap(np.random.randn(5, 3), name="p2")).draw(
            show=False, filename="/tmp/t_2legend.png",
            width=7, height=4, dpi=72)


class TestHeatmapLegend:
    def test_custom_legend_param(self):
        Heatmap(np.random.randn(5, 3), name="clg",
                heatmap_legend_param={"title": "Custom"}).draw(
            show=False, filename="/tmp/t_custom_lgd.png",
            width=4, height=4, dpi=72)

    def test_discrete_legend(self):
        Heatmap(np.array([["a", "b"], ["b", "a"]]), name="dlg",
                col={"a": "red", "b": "blue"}).draw(
            show=False, filename="/tmp/t_disc_lgd.png",
            width=3, height=3, dpi=72)
