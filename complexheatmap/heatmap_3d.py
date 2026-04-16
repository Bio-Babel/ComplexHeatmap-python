"""3-D heatmap visualization.

Faithfully ports R's ``ComplexHeatmap::Heatmap3D`` and ``bar3D`` (3d.R).

Architecture (matching R):
- ``bar3D()`` is a **grid-level drawing primitive** that renders 3-D bars
  at given (x, y) positions using oblique projection polygons.
- ``Heatmap3D()`` is a **thin wrapper around** :class:`~.heatmap.Heatmap`
  that suppresses default cell rects (``rect_gp=gpar(type="none")``) and
  uses a ``layer_fun`` to call ``bar3D()`` for each slice.

Because ``Heatmap3D`` returns a real ``Heatmap`` object, it automatically
inherits clustering, annotations, legends, ``+`` / ``%v%`` composition,
and Jupyter ``_repr_png_()`` display.
"""

__all__ = ["Heatmap3D", "bar3D"]

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import grid_py
from grid_py._colour import parse_r_colour


# ---------------------------------------------------------------------------
# Internal helpers — CIE LUV / HCL colour-space conversions
# ---------------------------------------------------------------------------
# These implement the CIE 1976 L*u*v* colour space (D65 illuminant) and its
# polar form (HCL = polarLUV) used by R's ``colorspace`` package, so that
# ``_add_luminance`` can faithfully reproduce R's ``sequential_hcl(n=9)``.
# ---------------------------------------------------------------------------

# D65 reference white
_Xn, _Yn, _Zn = 0.95047, 1.0, 1.08883
_un_prime = 4 * _Xn / (_Xn + 15 * _Yn + 3 * _Zn)
_vn_prime = 9 * _Yn / (_Xn + 15 * _Yn + 3 * _Zn)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _rgb_to_xyz(r: float, g: float, b: float) -> tuple:
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    return X, Y, Z


def _xyz_to_luv(X: float, Y: float, Z: float) -> tuple:
    yr = Y / _Yn
    L = 116 * yr ** (1 / 3) - 16 if yr > (6 / 29) ** 3 else (29 / 3) ** 3 * yr
    denom = X + 15 * Y + 3 * Z
    if denom == 0:
        return L, 0.0, 0.0
    u_prime = 4 * X / denom
    v_prime = 9 * Y / denom
    u = 13 * L * (u_prime - _un_prime)
    v = 13 * L * (v_prime - _vn_prime)
    return L, u, v


def _luv_to_xyz(L: float, u: float, v: float) -> tuple:
    if L <= 0:
        return 0.0, 0.0, 0.0
    Y = _Yn * ((L + 16) / 116) ** 3 if L > 8 else _Yn * L * (3 / 29) ** 3
    if L == 0:
        return 0.0, Y, 0.0
    u_prime = u / (13 * L) + _un_prime
    v_prime = v / (13 * L) + _vn_prime
    if v_prime == 0:
        return 0.0, Y, 0.0
    X = Y * 9 * u_prime / (4 * v_prime)
    Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)
    return X, Y, Z


def _xyz_to_rgb(X: float, Y: float, Z: float) -> tuple:
    rl = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    gl = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    return (
        _linear_to_srgb(max(0.0, min(1.0, rl))),
        _linear_to_srgb(max(0.0, min(1.0, gl))),
        _linear_to_srgb(max(0.0, min(1.0, bl))),
    )


def _rgb_to_hcl(r: float, g: float, b: float) -> tuple:
    """Convert sRGB [0,1] to HCL (polarLUV: H, C, L)."""
    X, Y, Z = _rgb_to_xyz(r, g, b)
    L, u, v = _xyz_to_luv(X, Y, Z)
    C = math.sqrt(u ** 2 + v ** 2)
    H = math.degrees(math.atan2(v, u)) % 360
    return H, C, L


def _hcl_to_rgb(H: float, C: float, L: float) -> tuple:
    """Convert HCL (polarLUV) to sRGB [0,1]."""
    u = C * math.cos(math.radians(H))
    v = C * math.sin(math.radians(H))
    X, Y, Z = _luv_to_xyz(L, u, v)
    return _xyz_to_rgb(X, Y, Z)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) in [0, 1] to ``#RRGGBB``."""
    return (
        f"#{int(round(min(max(r, 0), 1) * 255)):02X}"
        f"{int(round(min(max(g, 0), 1) * 255)):02X}"
        f"{int(round(min(max(b, 0), 1) * 255)):02X}"
    )


def _parse_color_to_rgb(color_spec) -> tuple:
    """Parse any R-compatible colour spec to (r, g, b) in [0, 1].

    Uses ``grid_py._colour.parse_r_colour`` which supports all 657 R named
    colours, hex strings, ``"transparent"``, numeric grey levels, etc.
    """
    rgba = parse_r_colour(color_spec)
    return rgba[0], rgba[1], rgba[2]


def _add_luminance(color_spec) -> List[str]:
    """Generate face-shading colours for a 3-D bar.

    Faithfully reproduces R's ``add_luminance()`` (3d.R:103-117) which
    converts the input colour to HCL (polarLUV) space and then calls
    ``colorspace::sequential_hcl(n=9, h=H, c=C, l=L)`` to produce a
    9-step luminance ramp.  The top face gets colour [1] (lightest),
    the front face gets [2], and the left face gets [3].

    Parameters
    ----------
    color_spec : str
        Any R-compatible colour specification (hex, named colour, etc.).

    Returns
    -------
    list of str
        ``[top_face, front_face, left_face]`` hex colours.
    """
    r, g, b = _parse_color_to_rgb(color_spec)
    H, C, L = _rgb_to_hcl(r, g, b)

    # R: sequential_hcl(n=9, h=H, c=C, l=L)
    # When l is scalar, R uses l = c(L, 90) and c = c(C, 0), power=1.5.
    # R's internal loop: i <- seq(1, 0, length=n)
    #   L_i = l2 - (l2 - l1) * i^power
    #   C_i = c2 + (c1 - c2) * i^power
    # Index [1] (i=1.0): L=l1, C=c1 = ORIGINAL colour (darkest)
    # Index [n] (i=0.0): L=l2, C=c2 = lightest/grey
    n = 9
    l1, l2 = L, 90.0
    c1, c2 = C, 0.0
    power = 1.5

    palette = []
    for i in range(n):
        # Match R's seq(1, 0, length=n): goes from 1.0 down to 0.0
        j = 1.0 - i / (n - 1) if n > 1 else 1.0
        jp = j ** power
        li = l2 - (l2 - l1) * jp
        ci = c2 + (c1 - c2) * jp
        cr, cg, cb = _hcl_to_rgb(H, ci, li)
        palette.append(_rgb_to_hex(cr, cg, cb))

    # R indexes: [1]=top (original colour), [2]=front (lighter), [3]=left
    return [palette[0], palette[1], palette[2]]


def _unit_value(u) -> float:
    """Extract the numeric float from a grid_py.Unit or plain number."""
    if isinstance(u, (int, float)):
        return float(u)
    if hasattr(u, "_values"):
        return float(u._values[0])
    return float(u)


def _gpar_type_none() -> grid_py.Gpar:
    """Create a Gpar with ``type='none'`` to suppress cell-rect drawing.

    Mirrors R's ``gpar(type = "none")``.
    """
    gp = grid_py.Gpar()
    gp._params["type"] = "none"
    return gp


# ---------------------------------------------------------------------------
# bar3D  —  grid-level drawing primitive  (R 3d.R:20-100)
# ---------------------------------------------------------------------------

def bar3D(
    x: Sequence[float],
    y: Sequence[float],
    w: Sequence[float],
    h: Sequence[float],
    l: Sequence[float],
    theta: float = 60,
    default_units: str = "npc",
    fill: Union[str, Sequence[str]] = "white",
    col: str = "black",
) -> None:
    """Draw 3-D bars at given positions.

    Port of R's ``bar3D`` (3d.R:20-100).  Each bar is an oblique
    projection with three visible faces (front, left, top).

    Parameters
    ----------
    x, y : sequence of float
        Centre positions of the bar bases.
    w, h : sequence of float
        Width / height of each bar's base rectangle.
    l : sequence of float
        Length of each bar in the z-direction (projection length).
    theta : float
        Projection angle in degrees (0–90).
    default_units : str
        Grid unit type (default ``"npc"``).
    fill : str or sequence of str
        Fill colour(s) for the bars.
    col : str
        Border colour.
    """
    # --- theta validation (R line 42-44) ----------------------------------
    if not (0 <= theta <= 90):
        raise ValueError("`theta` can only take value between 0 and 90.")

    # --- vectorise (R lines 28-40) ----------------------------------------
    n = max(len(x), len(y), len(w), len(h), len(l))
    x = list(x) if len(x) > 1 else list(x) * n
    y = list(y) if len(y) > 1 else list(y) * n
    w = list(w) if len(w) > 1 else list(w) * n
    h = list(h) if len(h) > 1 else list(h) * n
    l = list(l) if len(l) > 1 else list(l) * n
    if isinstance(fill, str):
        fill = [fill] * n
    elif len(fill) == 1:
        fill = list(fill) * n

    cos_t = math.cos(math.radians(theta))
    sin_t = math.sin(math.radians(theta))

    # Collect all polygon vertices + ids for a single grid.polygon call
    # (matches R lines 60-99)
    all_px: List[float] = []
    all_py: List[float] = []
    all_id: List[int] = []
    all_fill: List[str] = []
    pid = 1

    for i in range(n):
        face_colors = _add_luminance(fill[i])

        # Base corners (R lines 46-49)
        x1 = x[i] - w[i] * 0.5
        x2 = x[i] + w[i] * 0.5
        y1 = y[i] - h[i] * 0.5
        y2 = y[i] + h[i] * 0.5

        # Projected upper corners (R lines 51-57)
        dx = l[i] * cos_t
        dy = l[i] * sin_t
        a1 = x[i] + dx - w[i] * 0.5
        a2 = x[i] + dx + w[i] * 0.5
        b1 = y[i] + dy - h[i] * 0.5
        b2 = y[i] + dy + h[i] * 0.5

        # Front face (R lines 69-82): face_colors[1]
        all_px.extend([x1, a1, a2, x2])
        all_py.extend([y1, b1, b1, y1])
        all_id.extend([pid] * 4)
        all_fill.append(face_colors[1])
        pid += 1

        # Left face (R lines 84-89): face_colors[2]
        all_px.extend([x1, x1, a1, a1])
        all_py.extend([y1, y2, b2, b1])
        all_id.extend([pid] * 4)
        all_fill.append(face_colors[2])
        pid += 1

        # Top face (R lines 91-96): face_colors[0]
        all_px.extend([a1, a1, a2, a2])
        all_py.extend([b1, b2, b2, b1])
        all_id.extend([pid] * 4)
        all_fill.append(face_colors[0])
        pid += 1

    # Single polygon call (R line 99)
    if all_px:
        grid_py.grid_polygon(
            x=all_px,
            y=all_py,
            id=all_id,
            default_units=default_units,
            gp=grid_py.Gpar(fill=all_fill, col=col),
        )


# ---------------------------------------------------------------------------
# Heatmap3D  —  thin wrapper around Heatmap  (R 3d.R:140-176)
# ---------------------------------------------------------------------------

def Heatmap3D(
    matrix: np.ndarray,
    *,
    bar_rel_width: float = 0.6,
    bar_rel_height: float = 0.6,
    bar_max_length: Optional[grid_py.Unit] = None,
    bar_angle: float = 60,
    row_names_side: str = "left",
    show_row_dend: bool = False,
    show_column_dend: bool = False,
    **kwargs: Any,
) -> "Heatmap":
    """Create a 3-D heatmap.

    Thin wrapper around :class:`~.heatmap.Heatmap` that uses ``layer_fun``
    to render each cell as a 3-D bar.  Faithfully ports R's ``Heatmap3D``
    (3d.R:140-176).

    Because this returns a real ``Heatmap`` object, it inherits clustering,
    annotations, legends, ``+`` / ``%v%`` composition, and Jupyter display.

    Parameters
    ----------
    matrix : np.ndarray
        2-D numeric matrix.  Values **must be non-negative**.
    bar_rel_width : float
        Bar width as a fraction of cell width (0–1).  R default: 0.6.
    bar_rel_height : float
        Bar height as a fraction of cell height (0–1).  R default: 0.6.
    bar_max_length : grid_py.Unit, optional
        Maximum bar projection length.  R default: ``unit(1, "cm")``.
        When *None*, defaults to ``grid_py.Unit(1, "cm")``.
    bar_angle : float
        Oblique projection angle in degrees (0–90).  R default: 60.
    row_names_side : str
        Side for row names.  R default: ``"left"``.
    show_row_dend, show_column_dend : bool
        Whether to show dendrograms.  R default: ``False``.
    **kwargs
        All other arguments forwarded to ``Heatmap()``.

    Returns
    -------
    Heatmap
        A fully functional ``Heatmap`` object with 3-D bar rendering.

    Examples
    --------
    >>> import numpy as np
    >>> from complexheatmap import Heatmap3D
    >>> m = np.random.rand(6, 6) * 100
    >>> ht = Heatmap3D(m)
    >>> ht.draw()
    """
    from .heatmap import Heatmap
    from ._utils import pindex

    matrix = np.asarray(matrix, dtype=float)

    # Validate (R line 152-154)
    if np.any(matrix[np.isfinite(matrix)] < 0):
        raise ValueError("The matrix should be non-negative.")
    if not (0 <= bar_rel_width <= 1):
        raise ValueError("`bar_rel_width` must be between 0 and 1.")
    if not (0 <= bar_rel_height <= 1):
        raise ValueError("`bar_rel_height` must be between 0 and 1.")

    # R default: unit(1, "cm")  (3d.R:145)
    if bar_max_length is None:
        bar_max_length = grid_py.Unit(1, "cm")

    max_val = float(np.nanmax(matrix)) if np.any(np.isfinite(matrix)) else 1.0
    if max_val == 0:
        max_val = 1.0

    # --- layer_fun closure (R lines 164-169) ------------------------------
    _matrix = matrix
    _max_val = max_val
    _bar_rel_w = bar_rel_width
    _bar_rel_h = bar_rel_height
    _bar_max_l = bar_max_length
    _bar_angle = bar_angle

    def _layer_fun(j_arr, i_arr, x_arr, y_arr, w_arr, h_arr, fill_arr):
        n = len(j_arr)
        if n == 0:
            return

        # Extract NPC float values
        x_vals = np.array([_unit_value(u) for u in x_arr])
        y_vals = np.array([_unit_value(u) for u in y_arr])
        w_vals = np.array([_unit_value(u) for u in w_arr])
        h_vals = np.array([_unit_value(u) for u in h_arr])

        # Original matrix values at displayed positions (R: pindex)
        v_vals = pindex(_matrix, i_arr, j_arr)

        # Back-to-front draw order (R line 166):
        #   od = rank(order(-as.numeric(y), -as.numeric(x)))
        # R's order(-y, -x) sorts descending y first, then descending x.
        # rank(order(...)) gives the inverse permutation: each element's
        # position in the sorted sequence.  Using od as fancy-index
        # reorders so back cells (large y) come first and front cells
        # (small y) are drawn last (on top).
        #
        # np.lexsort sorts by last key first, so lexsort((x, y)) sorts
        # by y first (ascending), then x.  We negate both to get
        # descending, matching R's order(-y, -x).
        sort_idx = np.lexsort((-x_vals, -y_vals))
        od = np.empty_like(sort_idx)
        od[sort_idx] = np.arange(n)  # rank = inverse permutation

        # Convert bar_max_length to NPC within this viewport.
        # R passes unit(1,"cm") directly and grid resolves mixed-unit
        # arithmetic (NPC + cm) at draw time.  Here we convert the
        # Unit to NPC using grid_py.convert_width (which queries the
        # current viewport context during rendering).
        if isinstance(_bar_max_l, grid_py.Unit):
            bml = float(
                grid_py.convert_width(_bar_max_l, "npc", valueOnly=True)[0]
            )
        else:
            bml = float(_bar_max_l)

        # Reorder all arrays by draw order
        x_o = x_vals[od]
        y_o = y_vals[od]
        w_o = w_vals[od]
        h_o = h_vals[od]
        v_o = v_vals[od]
        f_o = [fill_arr[k] for k in od]

        # 1. Background rects (R line 167)
        grid_py.grid_rect(
            x=grid_py.Unit(list(x_o), "npc"),
            y=grid_py.Unit(list(y_o), "npc"),
            width=grid_py.Unit(list(w_o), "npc"),
            height=grid_py.Unit(list(h_o), "npc"),
            gp=grid_py.Gpar(col="white", fill="#EEEEEE"),
        )

        # 2. 3-D bars (R line 168)
        bar3D(
            x=list(x_o),
            y=list(y_o),
            w=list(w_o * _bar_rel_w),
            h=list(h_o * _bar_rel_h),
            l=list(v_o / _max_val * bml),
            theta=_bar_angle,
            default_units="npc",
            fill=f_o,
        )

    # --- build Heatmap (R lines 163-173) ----------------------------------
    ht = Heatmap(
        matrix,
        rect_gp=_gpar_type_none(),
        layer_fun=_layer_fun,
        show_row_dend=show_row_dend,
        show_column_dend=show_column_dend,
        row_names_side=row_names_side,
        **kwargs,
    )

    # R: ht@heatmap_param$type = "Heatmap3D"  (3d.R:174)
    if not hasattr(ht, "heatmap_param"):
        ht.heatmap_param = {}
    ht.heatmap_param["type"] = "Heatmap3D"

    # Disable viewport clipping so bar projections can extend beyond the
    # body boundary — R's viewport() defaults to clip="inherit" (no clip).
    ht.heatmap_param["clip_body"] = False

    # Auto-adjust global padding so bar projections don't overlap with
    # the column title or legend.  Matches R tutorial:
    #   ht_opt$TITLE_PADDING = unit(c(9, 2), "mm")   # [bottom, top]
    #   ht_opt$HEATMAP_LEGEND_PADDING = unit(5, "mm")
    # We compute the needed padding from bar geometry.
    _bml_val = _unit_value(bar_max_length)
    _overflow_top_mm = _bml_val * math.sin(math.radians(bar_angle)) * 10  # mm
    _overflow_right_mm = _bml_val * math.cos(math.radians(bar_angle)) * 10

    from ._globals import ht_opt
    # TITLE_PADDING as [bottom, top] (mm).  Bottom = gap between title
    # text and body top edge where bar projections overflow.
    _needed_bottom = _overflow_top_mm + 2  # overflow + 2mm gap
    _needed_top = 2.5  # default top padding above title text
    ht_opt(TITLE_PADDING=[_needed_bottom, _needed_top])

    # HEATMAP_LEGEND_PADDING: gap between body right edge and legend
    _needed_legend = _overflow_right_mm + 2
    _cur_lp = ht_opt("HEATMAP_LEGEND_PADDING")
    if isinstance(_cur_lp, (int, float)) and _needed_legend > _cur_lp:
        ht_opt(HEATMAP_LEGEND_PADDING=_needed_legend)

    return ht
