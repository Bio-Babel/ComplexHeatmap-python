# ComplexHeatmap-python

A Python port of the R [ComplexHeatmap](https://github.com/jokergoo/ComplexHeatmap) package (v2.25.3), built on [grid_py](https://github.com/Bio-Babel/grid_py) — a pure-Python reimplementation of R's grid graphics system.

**No matplotlib dependency.** The entire rendering pipeline uses grid_py's Cairo backend, producing publication-quality output identical to R's grid device.

## Features

- **Heatmap** with row/column clustering, splitting (k-means, factor), dendrograms, and custom color mapping
- **HeatmapList** for combining multiple heatmaps horizontally (`+`) or vertically (`%`), with synchronized row/column ordering and cross-heatmap body alignment
- **20 annotation functions**: `anno_barplot`, `anno_points`, `anno_lines`, `anno_mark`, `anno_text`, `anno_boxplot`, `anno_histogram`, `anno_density`, `anno_block`, `anno_image`, and more
- **HeatmapAnnotation** with flexible sizing, gap control, and annotation name positioning
- **Legends** with continuous color bars and discrete icon legends, packable and positionable on all four sides
- **oncoPrint** for mutation landscape visualization
- **UpSet** plots for set intersection analysis
- **Heatmap3D** for 3D bar-style heatmaps with CIE LUV/HCL color space
- **densityHeatmap** and **frequencyHeatmap** for distribution visualization
- **Global options** via `ht_opt()` matching R's `ht_opt$...` interface
- **Decoration** API (`decorate_heatmap_body`, `decorate_annotation`, etc.) for post-draw customization

## Architecture

```
complexheatmap_py          grid_py (R grid reimplementation)
  Heatmap                    Viewport, GridLayout
  HeatmapList       --->     Unit (mm, cm, npc, native, null)
  HeatmapAnnotation          Gpar, text_grob, rect_grob
  Legends                    CairoRenderer (PNG output)
  oncoPrint, UpSet            width_details, height_details
```

All layout computation uses grid_py's viewport stack, unit arithmetic, and text measurement — the same architecture as R's grid package. No hardcoded pixel values.

## Installation

```bash
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Clustering, KDE |
| pandas | DataFrame support, categorical factors |
| grid_py | Grid graphics system (R grid reimplementation) |
| Pillow | Image loading for `anno_image` |

## Quick Start

```python
import numpy as np
from complexheatmap import Heatmap, color_ramp2

mat = np.random.randn(20, 10)
col = color_ramp2([-2, 0, 2], ["blue", "white", "red"])

ht = Heatmap(mat, name="example", col=col,
             row_km=3, column_title="My Heatmap")
ht.draw()
```

### Combining Heatmaps

```python
import grid_py as gp
from complexheatmap import HeatmapAnnotation, rowAnnotation, anno_barplot

ht1 = Heatmap(mat1, name="expr", col=col_expr)
ht2 = Heatmap(mat2, name="cnv", col=col_cnv)

ht_list = ht1 + ht2 + rowAnnotation(
    bar=anno_barplot(values, which="row"),
    width=gp.Unit(2, "cm"),
)
ht_list.draw(merge_legends=True)
```

## Tutorials

Jupyter notebooks porting the official R ComplexHeatmap tutorials:

| # | Notebook | Topic |
|---|----------|-------|
| 02 | [Single Heatmap](tutorials/02-single_heatmap-v2.ipynb) | Colors, clustering, splitting, annotations, labels |
| 03 | [Heatmap Annotations](tutorials/03-heatmap_annotations.ipynb) | All 20 annotation types, sizing, decoration |
| 04 | [A List of Heatmaps](tutorials/04-a_list_of_heatmaps.ipynb) | Horizontal/vertical combination, row/column sync |
| 05 | [Legends](tutorials/05-legends.ipynb) | Continuous/discrete legends, positioning |
| 06 | [Heatmap Decoration](tutorials/06-heatmap_decoration.ipynb) | Post-draw customization via decorate API |
| 07 | [OncoPrint](tutorials/07-oncoprint.ipynb) | Mutation landscape visualization |
| 08 | [UpSet Plot](tutorials/08-upset-v2.ipynb) | Set intersection analysis |
| 10 | [Integration](tutorials/10-integrate-v2.ipynb) | Combining with other analyses |
| 11 | [Other High-Level Plots](tutorials/11-other-high-level-plots.ipynb) | densityHeatmap, frequencyHeatmap |
| 12 | [3D Heatmap](tutorials/12-3d-heatmap.ipynb) | Heatmap3D with bar projections |
| 13 | [Genome-Level Heatmap](tutorials/13-genome-level-heatmap-v2.ipynb) | Chromosome-split genome visualization |
| 14 | [Examples](tutorials/14-examples-v2.ipynb) | Gene expression, measles, methylation |
| 15 | [Other Tricks](tutorials/15-other-tricks.ipynb) | Utility functions, text measurement, ht_opt |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

