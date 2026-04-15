"""SingleAnnotation class -- a single annotation track.

R source correspondence
-----------------------
``R/SingleAnnotation-class.R`` -- S4 class wrapping an AnnotationFunction
with a name, colour mapping, legend parameters, and display options.

All drawing uses ``grid_py`` (the Python port of R's ``grid`` package).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import grid_py

from .annotation_function import AnnotationFunction
from .annotation_functions import anno_simple
from .color_mapping import ColorMapping
from ._globals import ht_opt

__all__ = [
    "SingleAnnotation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_color_mapping(
    value: np.ndarray,
    col: Union[Dict[str, str], Callable[..., Any], ColorMapping, None],
    name: str,
    na_col: str,
) -> Optional[ColorMapping]:
    """Build a :class:`ColorMapping` from *col* and *value*.

    Port of R ``SingleAnnotation-class.R:484-520`` and ``utils.R:62-125
    default_col()``.

    When *col* is ``None``, R auto-generates colours:
    - Discrete (string/factor): random categorical palette
    - Continuous (numeric): ``colorRamp2(range, c("white", rand_color))``

    Parameters
    ----------
    value : numpy.ndarray
        Data values for the annotation.
    col : dict, callable, ColorMapping, or None
        User-supplied colour specification.
    name : str
        Name used when auto-generating a ``ColorMapping``.
    na_col : str
        Colour for missing / NA values.

    Returns
    -------
    ColorMapping or None
    """
    if isinstance(col, ColorMapping):
        return col
    if isinstance(col, dict):
        return ColorMapping(name=name, colors=col, na_col=na_col)
    if callable(col) and col is not None:
        return ColorMapping(name=name, col_fun=col, na_col=na_col)

    # col is None → auto-generate (R default_col, utils.R:62-125)
    if col is None and value is not None:
        flat = np.asarray(value).ravel()
        # Remove NaN
        valid = flat[~np.isnan(flat)] if np.issubdtype(flat.dtype, np.floating) else flat[flat != None]

        if len(valid) == 0:
            return None

        # Check if discrete (string/object dtype)
        if valid.dtype.kind in ('U', 'S', 'O'):
            # Discrete: auto-generate categorical colours
            from ._color import rand_color
            levels = list(dict.fromkeys(valid))  # unique, order-preserving
            colors = rand_color(len(levels))
            color_map = dict(zip(levels, colors))
            return ColorMapping(name=name, colors=color_map, na_col=na_col)
        elif np.issubdtype(valid.dtype, np.number):
            # Continuous: colorRamp2(range, c("white", rand_color))
            # R: utils.R:120-121
            from ._color import color_ramp2, rand_color
            rc = rand_color(1)[0]
            vmin, vmax = float(valid.min()), float(valid.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            col_fun = color_ramp2([vmin, vmax], ["white", rc])
            return ColorMapping(name=name, col_fun=col_fun, na_col=na_col)

    return None


def _default_anno_size(which: str) -> float:
    """Return the default annotation size in mm from global options."""
    return float(ht_opt("simple_anno_size"))


# ---------------------------------------------------------------------------
# SingleAnnotation
# ---------------------------------------------------------------------------

class SingleAnnotation:
    """Single annotation track for a heatmap.

    Wraps either a simple colour bar (from *value* + *col*) or a custom
    :class:`~complexheatmap.annotation_function.AnnotationFunction`.

    Parameters
    ----------
    name : str
        Annotation name / identifier.
    value : array-like, optional
        Data values for a simple (colour-bar) annotation.
    col : dict or callable or ColorMapping, optional
        Colour mapping.
    fun : AnnotationFunction or callable, optional
        Custom annotation function.
    label : str, optional
        Display label (defaults to *name*).
    na_col : str
        Colour for NA / missing values.
    which : str
        ``"column"`` or ``"row"``.
    show_legend : bool
        Whether to include this annotation in the legend.
    gp : dict, optional
        Graphical parameters forwarded to the drawing function.
    border : bool
        Whether to draw a border around each cell.
    legend_param : dict, optional
        Additional legend customisation parameters.
    show_name : bool
        Whether to display the annotation name alongside the track.
    name_gp : dict, optional
        Text parameters for the annotation name.
    name_side : str, optional
        Side on which to draw the name.
    name_rot : float, optional
        Rotation angle for the annotation name text.
    width : object, optional
        Width of the annotation track.
    height : object, optional
        Height of the annotation track.
    """

    def __init__(
        self,
        name: str,
        value: Optional[Any] = None,
        col: Optional[Union[Dict[str, str], Callable[..., Any], ColorMapping]] = None,
        fun: Optional[Union[AnnotationFunction, Callable[..., Any]]] = None,
        label: Optional[str] = None,
        na_col: str = "grey",
        which: str = "column",
        show_legend: bool = True,
        gp: Optional[Dict[str, Any]] = None,
        border: bool = False,
        legend_param: Optional[Dict[str, Any]] = None,
        show_name: bool = True,
        name_gp: Optional[Dict[str, Any]] = None,
        name_side: Optional[str] = None,
        name_rot: Optional[float] = None,
        width: Optional[Any] = None,
        height: Optional[Any] = None,
    ) -> None:
        if which not in ("column", "row"):
            raise ValueError(f"`which` must be 'column' or 'row', got {which!r}")

        self.name: str = name
        self.label: str = label if label is not None else name
        self.na_col: str = na_col
        self.which: str = which
        self.show_legend: bool = show_legend
        self.gp: Dict[str, Any] = gp if gp is not None else {}
        self.border: bool = border
        self.legend_param: Dict[str, Any] = legend_param if legend_param is not None else {}
        self.show_name: bool = show_name
        self.name_gp: Dict[str, Any] = name_gp if name_gp is not None else {}
        self.name_rot: Optional[float] = name_rot
        self._color_mapping: Optional[ColorMapping] = None
        self._is_anno_matrix: bool = False

        # Resolve name_side defaults
        if name_side is not None:
            self.name_side = name_side
        else:
            self.name_side = "right" if which == "column" else "bottom"

        # ------------------------------------------------------------------
        # Build internal AnnotationFunction
        # ------------------------------------------------------------------
        if fun is not None:
            # Custom annotation function provided
            if isinstance(fun, AnnotationFunction):
                self._anno_fun: AnnotationFunction = fun
                # If the AnnotationFunction was created with a different `which`
                # (e.g., anno_barplot() defaults to which="column" but used in
                # rowAnnotation), swap width/height for the correct orientation.
                af_which = getattr(fun, 'which', None)
                if af_which is not None and af_which != which:
                    self._anno_fun.which = which
                    old_w = self._anno_fun.width
                    old_h = self._anno_fun.height

                    def _is_npc(u):
                        return hasattr(u, '_units') and len(u._units) > 0 and u._units[0] == 'npc'

                    # Swap: column→row means height→width, width→height
                    if which == "row" and not _is_npc(old_h):
                        self._anno_fun.width = old_h   # short axis
                        self._anno_fun.height = grid_py.Unit(1, "npc")  # long axis
                    elif which == "column" and not _is_npc(old_w):
                        self._anno_fun.height = old_w
                        self._anno_fun.width = grid_py.Unit(1, "npc")
            elif callable(fun):
                self._anno_fun = AnnotationFunction(
                    fun=fun,
                    fun_name=name,
                    which=which,
                    n=len(value) if value is not None else None,
                    width=width,
                    height=height,
                )
            else:
                raise TypeError(
                    f"`fun` must be an AnnotationFunction or callable, got {type(fun)!r}"
                )
            self._value: Optional[np.ndarray] = (
                np.asarray(value) if value is not None else None
            )
        elif value is not None:
            # Simple value-based annotation
            self._value = np.asarray(value)
            self._color_mapping = _infer_color_mapping(
                self._value, col, name, na_col
            )
            self._is_anno_matrix = self._value.ndim == 2

            # Extract col for anno_simple from the inferred ColorMapping
            col_arg = col
            if isinstance(col, ColorMapping):
                col_arg = col.color_map if col.is_discrete else col._col_fun
            elif col is None and self._color_mapping is not None:
                # Auto-generated ColorMapping: pass its function/dict to anno_simple
                cm = self._color_mapping
                if cm.is_discrete:
                    col_arg = cm.color_map
                elif cm._col_fun is not None:
                    col_arg = cm._col_fun

            self._anno_fun = anno_simple(
                x=self._value,
                col=col_arg,
                na_col=na_col,
                which=which,
                border=border,
                gp=gp,
                width=width,
                height=height,
            )
        else:
            raise ValueError(
                "Either `value` or `fun` must be provided to SingleAnnotation."
            )

        # Override dimensions if explicitly set
        if width is not None:
            self._anno_fun.width = width
        if height is not None:
            self._anno_fun.height = height

        # Apply default sizes when not set
        if self.which == "column" and self._anno_fun.height is None:
            self._anno_fun.height = _default_anno_size(which)
        if self.which == "row" and self._anno_fun.width is None:
            self._anno_fun.width = _default_anno_size(which)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def color_mapping(self) -> Optional[ColorMapping]:
        """The :class:`ColorMapping` for this annotation, or ``None``."""
        return self._color_mapping

    @property
    def is_anno_matrix(self) -> bool:
        """``True`` when the annotation value is a 2-D matrix."""
        return self._is_anno_matrix

    @property
    def nobs(self) -> Optional[int]:
        """Number of observations."""
        return self._anno_fun.nobs

    @property
    def extended(self) -> tuple:
        """Overflow space (top, right, bottom, left) in mm.

        Port of R ``SingleAnnotation-class.R:440-473``.  Computes how
        much space the annotation name and axis labels need beyond the
        annotation body.
        """
        top = right = bottom = left = 0.0
        name_offset_mm = 1.0  # R default

        if self.show_name and self.name:
            # Estimate text dimensions (R: max_text_width/max_text_height)
            label = self.label if self.label else self.name
            fontsize = 10
            if self.name_gp:
                fontsize = self.name_gp.get("fontsize", 10)
            # Rough estimate: each char ~2mm wide, line height ~3.5mm
            lines = label.split("\n")
            char_w_mm = fontsize * 0.2  # approximate mm per char
            max_line_w = max(len(ln) for ln in lines) * char_w_mm
            text_h_mm = len(lines) * fontsize * 0.35
            text_extent = name_offset_mm  # at minimum, the offset

            if self.which == "column":
                rot = self.name_rot if self.name_rot is not None else 0
                if rot in (0, 180, None):
                    text_extent += max_line_w
                else:
                    text_extent += text_h_mm
                if self.name_side == "left":
                    left = text_extent
                else:
                    right = text_extent
            else:  # row
                rot = self.name_rot if self.name_rot is not None else 90
                if rot in (90, 270):
                    text_extent += max_line_w
                else:
                    text_extent += text_h_mm
                if self.name_side == "bottom":
                    bottom = text_extent
                else:
                    top = text_extent

        return (top, right, bottom, left)

    @property
    def width(self) -> Optional[Any]:
        """Width of the annotation track."""
        return self._anno_fun.width

    @width.setter
    def width(self, value: Optional[Any]) -> None:
        self._anno_fun.width = value

    @property
    def height(self) -> Optional[Any]:
        """Height of the annotation track."""
        return self._anno_fun.height

    @height.setter
    def height(self, value: Optional[Any]) -> None:
        self._anno_fun.height = value

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        index: Union[np.ndarray, Sequence[int]],
        k: int = 1,
        n: int = 1,
    ) -> None:
        """Draw the annotation track + name label.

        Port of R ``SingleAnnotation-class.R:606-713``.

        Parameters
        ----------
        index : array-like of int
            Observation indices (0-based).
        k : int
            Current slice index (1-based).
        n : int
            Total number of slices.
        """
        # ---- Data viewport (R SingleAnnotation-class.R:654-666) ----
        # The annotation function draws in native coordinates; we must
        # set up the viewport with the correct xscale/yscale so that
        # native units map correctly.
        ni = len(index)
        fun_data_scale = getattr(self._anno_fun, "data_scale", None)
        _pushed_vp = False

        if fun_data_scale is not None and ni > 0:
            if self.which == "column":
                xscale = (0.5, ni + 0.5)
                yscale = tuple(fun_data_scale)
            else:  # row
                xscale = tuple(fun_data_scale)
                yscale = (0.5, ni + 0.5)
            grid_py.push_viewport(grid_py.Viewport(
                xscale=xscale,
                yscale=yscale,
                clip=False,
                name=f"sa_data_{self.name}_{k}",
            ))
            _pushed_vp = True

        self._anno_fun.draw(index, k=k, n=n)

        if _pushed_vp:
            grid_py.up_viewport()

        # ---- Annotation name label (R line 671-712) ----
        if not self.show_name:
            return
        if not self.name:
            return

        # For split heatmaps: only draw name on the appropriate slice
        draw_name = True
        if n > 1:
            side = self.name_side
            if self.which == "row":
                draw_name = (k == n and side == "bottom") or (k == 1 and side == "top")
            elif self.which == "column":
                draw_name = (k == 1 and side == "left") or (k == n and side == "right")

        if not draw_name:
            return

        # Name offset: R default is unit(1, "mm")
        offset = grid_py.Unit(1, "mm")
        name_gp = grid_py.Gpar(fontsize=10)
        if hasattr(self, 'name_gp') and self.name_gp:
            name_gp = grid_py.Gpar(**self.name_gp) if isinstance(self.name_gp, dict) else self.name_gp

        # Compute name position (R SingleAnnotation-class.R:268-340)
        if self.which == "column":
            if self.name_side == "right":
                x = grid_py.Unit(1, "npc") + offset
                y = grid_py.Unit(0.5, "npc")
                just = "left"
                rot = 0
            else:  # left
                x = grid_py.Unit(0, "npc") - offset
                y = grid_py.Unit(0.5, "npc")
                just = "right"
                rot = 0
        else:  # row
            if self.name_side == "bottom":
                x = grid_py.Unit(0.5, "npc")
                y = grid_py.Unit(0, "npc") - offset
                just = "right"
                rot = 90
            else:  # top
                x = grid_py.Unit(0.5, "npc")
                y = grid_py.Unit(1, "npc") + offset
                just = "left"
                rot = 90

        grid_py.grid_text(
            label=self.name,
            x=x, y=y, just=just, rot=rot,
            gp=name_gp,
        )

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset(self, indices: Union[np.ndarray, Sequence[int]]) -> "SingleAnnotation":
        """Return a new :class:`SingleAnnotation` with subsetted data.

        Parameters
        ----------
        indices : array-like of int
            0-based observation indices to keep.

        Returns
        -------
        SingleAnnotation
        """
        new = copy.copy(self)
        new._anno_fun = self._anno_fun.subset(indices)
        if self._value is not None:
            idx = np.asarray(indices, dtype=int)
            if self._value.ndim == 2:
                new._value = self._value[idx, :]
            else:
                new._value = self._value[idx]
        return new

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SingleAnnotation(name={self.name!r}, which={self.which!r}, "
            f"nobs={self.nobs})"
        )
