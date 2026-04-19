"""DataGrid bypass encoder.

For a large matrix the heatmap body is drawn as a single raster grob;
inlining per-cell metadata would blow up the scene-graph JSON.  We
instead register a :class:`grid_py.DataGridNode` containing the raw
value array (compressed) + row/col names; the JS side resolves tooltips
by pixel → (row, col) lookup.

Entry points:
    * :func:`should_use_datagrid` — policy decision (cell count, raster)
    * :func:`encode_datagrid` — build the DataGridNode payload
"""

from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["should_use_datagrid", "encode_datagrid"]


def should_use_datagrid(
    nrow: int,
    ncol: int,
    use_raster: bool,
    cfg: Any,
) -> bool:
    """Decide whether a Heatmap body should use the DataGrid bypass.

    Rule (locked decision #2): activate when raster is used OR when
    ``nrow * ncol > cfg.max_inline_cells`` (default 50 000).
    """
    if use_raster:
        return True
    threshold = getattr(cfg, "max_inline_cells", 50_000)
    return int(nrow) * int(ncol) > int(threshold)


def encode_datagrid(
    grid_id: str,
    *,
    values: np.ndarray,
    row_ids: Sequence[int],
    col_ids: Sequence[int],
    row_names: Sequence[str],
    col_names: Sequence[str],
    viewport_name: str,
    annotations: Optional[Dict[str, Any]] = None,
    max_float32_cells: int = 500_000,
) -> "DataGridNode":
    """Pack a matrix into a ``grid_py.DataGridNode``.

    The dtype is ``float32`` by default; when the matrix exceeds
    ``max_float32_cells`` it is quantized to 8-bit codes with a 256-entry
    LUT (locked decision #2).
    """
    from grid_py import DataGridNode  # local import to avoid cycle at pkg load

    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D, got ndim={values.ndim}")
    nrow, ncol = values.shape
    n_cells = nrow * ncol

    if n_cells > int(max_float32_cells):
        codes, lut = _quantile8_encode(values)
        b64 = base64.b64encode(codes.tobytes()).decode("ascii")
        return DataGridNode(
            id=grid_id,
            row_ids=[int(x) for x in row_ids],
            col_ids=[int(x) for x in col_ids],
            row_names=[str(x) for x in row_names],
            col_names=[str(x) for x in col_names],
            values=b64,
            value_shape=[int(nrow), int(ncol)],
            value_dtype="quantile8",
            value_lut=[float(x) for x in lut],
            annotations=annotations or {},
            viewport_name=viewport_name,
        )

    arr32 = values.astype(np.float32, copy=False)
    b64 = base64.b64encode(arr32.tobytes()).decode("ascii")
    return DataGridNode(
        id=grid_id,
        row_ids=[int(x) for x in row_ids],
        col_ids=[int(x) for x in col_ids],
        row_names=[str(x) for x in row_names],
        col_names=[str(x) for x in col_names],
        values=b64,
        value_shape=[int(nrow), int(ncol)],
        value_dtype="float32",
        value_lut=None,
        annotations=annotations or {},
        viewport_name=viewport_name,
    )


def _quantile8_encode(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """256-quantile encoding.  Returns (uint8 codes, 256-entry LUT).

    NaN is mapped to code 0 and the LUT[0] entry is set to NaN.
    """
    flat = values.ravel()
    mask = np.isfinite(flat)
    finite = flat[mask]
    codes = np.zeros(flat.size, dtype=np.uint8)
    lut = np.full(256, np.nan, dtype=np.float64)

    if finite.size == 0:
        return codes.reshape(values.shape), lut

    # Reserve code 0 for NaN.  255 usable codes.
    n_codes = 255
    qs = np.linspace(0.0, 1.0, n_codes + 1)
    breakpoints = np.quantile(finite, qs)
    bins = np.unique(breakpoints)
    if bins.size < 2:
        # All equal — everything to code 1
        codes[mask] = 1
        lut[1] = float(finite[0])
        return codes.reshape(values.shape), lut

    # Digitize returns 1..len(bins) for values inside range; clip to usable codes
    codes_finite = np.digitize(finite, bins[1:-1], right=False) + 1  # 1..n_codes
    codes_finite = np.clip(codes_finite, 1, n_codes).astype(np.uint8)
    codes[mask] = codes_finite

    # LUT value per code = midpoint of its bin
    for c in range(1, n_codes + 1):
        idx = codes_finite == c
        if np.any(idx):
            lut[c] = float(finite[idx].mean())
    return codes.reshape(values.shape), lut
