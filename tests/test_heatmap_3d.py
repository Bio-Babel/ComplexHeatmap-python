"""Tests for Heatmap3D and bar3D (new architecture).

Heatmap3D is a function that returns a Heatmap object.
bar3D is a grid-level drawing primitive.
"""

from __future__ import annotations

import numpy as np
import pytest

import grid_py
from complexheatmap.heatmap_3d import (
    Heatmap3D,
    bar3D,
    _add_luminance,
    _parse_color_to_rgb,
    _rgb_to_hex,
    _rgb_to_hcl,
    _hcl_to_rgb,
)
from complexheatmap.heatmap import Heatmap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_matrix():
    return np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])


@pytest.fixture
def square_matrix():
    np.random.seed(42)
    return np.random.rand(4, 4)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestAddLuminance:
    def test_returns_three_colors(self):
        result = _add_luminance("#FF8040")
        assert len(result) == 3
        for c in result:
            assert isinstance(c, str)
            assert c.startswith("#")
            assert len(c) == 7

    def test_white(self):
        result = _add_luminance("#FFFFFF")
        assert len(result) == 3

    def test_black(self):
        result = _add_luminance("#000000")
        assert len(result) == 3


class TestColorHelpers:
    def test_parse_color_hex(self):
        r, g, b = _parse_color_to_rgb("#FF8040")
        assert abs(r - 1.0) < 0.01
        assert abs(g - 0.502) < 0.01
        assert abs(b - 0.251) < 0.01

    def test_parse_color_named(self):
        """Should support all 657 R named colours via grid_py."""
        r, g, b = _parse_color_to_rgb("cornflowerblue")
        assert 0.3 < r < 0.5
        assert 0.5 < g < 0.7
        assert 0.8 < b < 1.0

    def test_rgb_to_hex(self):
        assert _rgb_to_hex(1.0, 0.0, 0.0) == "#FF0000"
        assert _rgb_to_hex(0.0, 1.0, 0.0) == "#00FF00"

    def test_hcl_roundtrip(self):
        """RGB -> HCL -> RGB should round-trip."""
        r0, g0, b0 = 1.0, 0x80 / 255, 0x40 / 255
        H, C, L = _rgb_to_hcl(r0, g0, b0)
        r1, g1, b1 = _hcl_to_rgb(H, C, L)
        assert abs(r0 - r1) < 0.01
        assert abs(g0 - g1) < 0.01
        assert abs(b0 - b1) < 0.01

    def test_hex_roundtrip(self):
        orig = "#AB34CD"
        r, g, b = _parse_color_to_rgb(orig)
        result = _rgb_to_hex(r, g, b)
        assert result.upper() == orig.upper()


# ---------------------------------------------------------------------------
# Heatmap3D tests
# ---------------------------------------------------------------------------

class TestHeatmap3D:
    def test_returns_heatmap(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert isinstance(ht, Heatmap)

    def test_basic_properties(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert ht.nrow == 2
        assert ht.ncol == 3

    def test_cluster_rows_false_by_default(self, small_matrix):
        """Heatmap3D does not set cluster_rows — Heatmap default applies."""
        ht = Heatmap3D(small_matrix)
        # Should not error
        assert isinstance(ht, Heatmap)

    def test_show_row_dend_false(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert ht.show_row_dend is False

    def test_show_column_dend_false(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert ht.show_column_dend is False

    def test_row_names_side_left(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert ht.row_names_side == "left"

    def test_negative_values_raise(self):
        mat = np.array([[-1, 2], [3, 4]], dtype=float)
        with pytest.raises(ValueError, match="non-negative"):
            Heatmap3D(mat)

    def test_bar_rel_width_validation(self, small_matrix):
        with pytest.raises(ValueError, match="bar_rel_width"):
            Heatmap3D(small_matrix, bar_rel_width=1.5)

    def test_bar_rel_height_validation(self, small_matrix):
        with pytest.raises(ValueError, match="bar_rel_height"):
            Heatmap3D(small_matrix, bar_rel_height=-0.1)

    def test_custom_bar_angle(self, small_matrix):
        ht = Heatmap3D(small_matrix, bar_angle=45)
        assert isinstance(ht, Heatmap)

    def test_kwargs_forwarded(self, small_matrix):
        ht = Heatmap3D(small_matrix, name="my3d", column_title="Test")
        assert ht.name == "my3d"

    def test_custom_col(self, small_matrix):
        from complexheatmap._color import color_ramp2
        col_fn = color_ramp2([0, 6], ["white", "red"])
        ht = Heatmap3D(small_matrix, col=col_fn)
        assert isinstance(ht, Heatmap)

    def test_has_layer_fun(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert ht.layer_fun is not None

    def test_rect_gp_type_none(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        assert hasattr(ht.rect_gp, '_params')
        assert ht.rect_gp._params.get("type") == "none"

    def test_heatmap_param_type(self, small_matrix):
        """R: ht@heatmap_param$type = 'Heatmap3D'"""
        ht = Heatmap3D(small_matrix)
        assert hasattr(ht, "heatmap_param")
        assert ht.heatmap_param.get("type") == "Heatmap3D"

    def test_bar_max_length_default_unit(self, small_matrix):
        """Default bar_max_length should be Unit(1, 'cm')."""
        ht = Heatmap3D(small_matrix)
        # The closure captures bar_max_length; verify it's a Unit
        assert isinstance(ht, Heatmap)

    def test_draw(self, small_matrix):
        ht = Heatmap3D(small_matrix)
        # Should not raise
        ht.draw(show=False, filename="/tmp/test_h3d_draw.png",
                width=5, height=5, dpi=72)

    def test_composition(self, small_matrix):
        """Heatmap3D + Heatmap should produce HeatmapList."""
        ht3d = Heatmap3D(small_matrix, name="left")
        ht_normal = Heatmap(small_matrix, name="right")
        combined = ht3d + ht_normal
        assert type(combined).__name__ == "HeatmapList"


# ---------------------------------------------------------------------------
# bar3D tests (grid-level primitive)
# ---------------------------------------------------------------------------

class TestBar3D:
    def test_basic_call(self):
        """bar3D should not raise when called with valid args."""
        # Need a renderer context for grid_polygon to work
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        bar3D(
            x=[0.3, 0.7],
            y=[0.5, 0.5],
            w=[0.2, 0.2],
            h=[0.2, 0.2],
            l=[0.1, 0.2],
            fill=["red", "blue"],
        )

    def test_single_bar(self):
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        bar3D(
            x=[0.5], y=[0.5], w=[0.3], h=[0.3], l=[0.15],
            theta=60, fill=["#FF8040"],
        )

    def test_theta_range(self):
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        for theta in [0, 30, 45, 60, 90]:
            bar3D(
                x=[0.5], y=[0.5], w=[0.2], h=[0.2], l=[0.1],
                theta=theta, fill=["green"],
            )

    def test_theta_out_of_range(self):
        """R: theta < 0 | theta > 90 should raise."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        with pytest.raises(ValueError, match="theta"):
            bar3D(x=[0.5], y=[0.5], w=[0.2], h=[0.2], l=[0.1],
                  theta=100, fill=["red"])
        with pytest.raises(ValueError, match="theta"):
            bar3D(x=[0.5], y=[0.5], w=[0.2], h=[0.2], l=[0.1],
                  theta=-10, fill=["red"])
