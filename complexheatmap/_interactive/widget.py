"""anywidget-backed interactive heatmap widget (PR 2 integration).

This wraps the WebRenderer's scene graph in an ``anywidget.AnyWidget``
that provides bidirectional Python↔JS state sync:

    - sync traitlets:   ``selected``, ``filter``, ``pinned``, ``color_threshold``
    - async event stream: use ``widget.on_event("dend_click", cb)`` etc.

When anywidget is not installed, :func:`make_widget` falls back to the
iframe / raw-HTML path — callers see ``HeatmapListWidgetFallback`` with
a matching ``_repr_html_`` so Jupyter display still works.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

__all__ = ["HeatmapWidget", "HeatmapListWidgetFallback", "make_widget"]


# Lazy resource loader — duplicates grid_py._load_resource to avoid
# reaching into its private API.
def _load_grid_resource(name: str) -> str:
    import grid_py
    resources_dir = os.path.join(os.path.dirname(grid_py.__file__), "resources")
    path = os.path.join(resources_dir, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Compose the JS that anywidget executes inside the widget's shadow DOM.
def _build_widget_js() -> str:
    d3 = _load_grid_resource("d3.v7.min.js")
    core = _load_grid_resource("gridpy.js")
    heatmap = _load_grid_resource("gridpy-heatmap.js")
    # anywidget expects ES-module semantics: the IIFE-style core uses `var`
    # which becomes module-scoped (not global).  We assign to `globalThis`
    # so both modules and later-loaded plugins can find the runtime, and we
    # also make d3 globally visible since core references `d3.quadtree`
    # etc. by bare global name.
    glue = """
// Expose globals so IIFE-scoped code resolves them the same way it would
// inside a <script>-loaded HTML file.
if (typeof globalThis.d3 === 'undefined' && typeof d3 !== 'undefined') {
  globalThis.d3 = d3;
}
if (typeof globalThis.gridpy === 'undefined' && typeof gridpy !== 'undefined') {
  globalThis.gridpy = gridpy;
}

export function render({ model, el }) {
  el.innerHTML = '';
  const mount = document.createElement('div');
  mount.id = 'gridpy-plot-' + (crypto.randomUUID ? crypto.randomUUID() : Math.random());
  el.appendChild(mount);

  const scene = model.get('scene');
  const options = model.get('options') || {};

  const state = globalThis.gridpy.render(mount, scene, options);

  // --- Sync traitlets -----------------------------------------------------
  // brush selection -> model.selected
  globalThis.gridpy.on(state, 'brush', (detail) => {
    const sel = (detail.selected || []).map(item => item.data || item);
    model.set('selected', sel);
    model.save_changes();
  });

  // generic click events -> model.events (async stream)
  ['cell_click', 'anno_cell_click', 'row_label_click', 'col_label_click',
   'dend_click', 'legend_item_click', 'blank_click', 'search_match',
   'color_threshold'
  ].forEach(name => {
    globalThis.gridpy.on(state, name, (detail) => {
      const events = (model.get('events') || []).slice();
      events.push({ type: name, payload: detail, ts: Date.now() });
      model.set('events', events);
      model.save_changes();
    });
  });

  // filter traitlet (Python -> JS)
  model.on('change:filter', () => {
    const f = model.get('filter') || {};
    globalThis.gridpy.emit(state, 'apply_filter', f);
  });
}
"""
    return d3 + "\n" + core + "\n" + heatmap + "\n" + glue


def _build_widget_css() -> str:
    return (
        _load_grid_resource("gridpy.css")
        + "\n"
        + _load_grid_resource("gridpy-heatmap.css")
    )


# ---------------------------------------------------------------------------
# HeatmapWidget (anywidget path)
# ---------------------------------------------------------------------------

def _try_import_anywidget():
    try:
        import anywidget  # type: ignore
        import traitlets  # type: ignore
    except ImportError:
        return None, None
    return anywidget, traitlets


_anywidget, _traitlets = _try_import_anywidget()

if _anywidget is not None:

    class HeatmapWidget(_anywidget.AnyWidget):  # type: ignore[misc]
        """anywidget wrapper exposing sync traitlets + async event stream.

        Not instantiated directly — use :func:`make_widget`.
        """

        _esm = _build_widget_js()
        _css = _build_widget_css()

        scene = _traitlets.Dict().tag(sync=True)
        options = _traitlets.Dict().tag(sync=True)

        # sync state
        selected = _traitlets.List().tag(sync=True)
        filter = _traitlets.Dict().tag(sync=True)
        pinned = _traitlets.List().tag(sync=True)
        color_threshold = _traitlets.List().tag(sync=True)

        # async event stream (append-only)
        events = _traitlets.List().tag(sync=True)

        def __init__(self, scene: Dict[str, Any], options: Optional[Dict[str, Any]] = None, **kw: Any) -> None:
            super().__init__(scene=scene, options=options or {}, **kw)
            self._event_callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
            self._last_event_idx: int = 0
            self.observe(self._fire_event_callbacks, names="events")

        # --- Async event-stream helpers -----------------------------------
        def on_event(
            self,
            name: str,
            callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        ) -> Any:
            """Register a callback for a named event.

            Returns a no-op decorator when *callback* is ``None`` so the
            method can be used as ``@widget.on_event("dend_click")``.
            """
            if callback is None:
                def _decorator(fn: Callable[[Dict[str, Any]], None]) -> Callable[[Dict[str, Any]], None]:
                    self._event_callbacks.setdefault(name, []).append(fn)
                    return fn
                return _decorator
            self._event_callbacks.setdefault(name, []).append(callback)
            return callback

        def _fire_event_callbacks(self, change: Any) -> None:
            # Drain only events appended since the last notification so that
            # a single append doesn't re-fire handlers for every previous
            # event.  ``_last_event_idx`` is initialised in ``__init__``.
            events = change["new"]
            if not events:
                return
            new_items = events[self._last_event_idx:]
            self._last_event_idx = len(events)
            for ev in new_items:
                for cb in self._event_callbacks.get(ev["type"], []):
                    # User callback is a system boundary — any exception
                    # must not break the trait observer chain.
                    try:
                        cb(ev["payload"])
                    except Exception as exc:  # pragma: no cover
                        import warnings
                        warnings.warn(
                            f"on_event('{ev['type']}') callback raised: {exc}"
                        )

else:

    class HeatmapWidget:  # type: ignore[no-redef]
        """Placeholder when anywidget is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise ImportError(
                "anywidget / traitlets must be installed for HeatmapWidget. "
                "Install with: pip install anywidget traitlets"
            )


# ---------------------------------------------------------------------------
# Fallback — same constructor signature, iframe-based display
# ---------------------------------------------------------------------------

class HeatmapListWidgetFallback:
    """Static-HTML fallback used when anywidget is unavailable."""

    def __init__(self, scene: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> None:
        self._scene = scene
        self._options = options or {}
        self.selected: List[Any] = []
        self.events: List[Dict[str, Any]] = []
        self.filter: Dict[str, Any] = {}
        self.pinned: List[Any] = []
        self.color_threshold: List[float] = []

    def on_event(self, name: str, callback: Optional[Callable] = None) -> Any:
        """Record handler but never fire — fallback mode has no JS→Py bridge."""
        if callback is None:
            return lambda fn: fn
        return callback

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """No-op in fallback mode."""
        return None

    def _repr_html_(self) -> str:
        # Render via iframe like WebRenderer._repr_html_
        html = _render_scene_to_html(self._scene, self._options)
        escaped = html.replace("&", "&amp;").replace('"', "&quot;")
        w = int(self._scene.get("width", 800))
        h = int(self._scene.get("height", 600))
        return (
            f'<iframe srcdoc="{escaped}" '
            f'width="{w}" height="{h}" '
            f'style="border:none;" '
            f'sandbox="allow-scripts"></iframe>'
        )


def _render_scene_to_html(scene: Dict[str, Any], options: Dict[str, Any]) -> str:
    """Inline-D3 full HTML wrapper for a scene dict — used by fallback."""
    core = _load_grid_resource("gridpy.js")
    heatmap = _load_grid_resource("gridpy-heatmap.js")
    css = (
        _load_grid_resource("gridpy.css")
        + "\n"
        + _load_grid_resource("gridpy-heatmap.css")
    )
    d3 = _load_grid_resource("d3.v7.min.js")
    scene_json = json.dumps(scene)
    opts_json = json.dumps(options)
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><style>{css}</style></head>
<body>
<div id='gridpy-plot'></div>
<script>{d3}</script>
<script>{core}</script>
<script>{heatmap}</script>
<script>
gridpy.render(document.getElementById('gridpy-plot'), {scene_json}, {opts_json});
</script>
</body></html>"""


def make_widget(
    scene: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return a HeatmapWidget (anywidget) or HeatmapListWidgetFallback."""
    if _anywidget is not None:
        return HeatmapWidget(scene=scene, options=options or {})
    return HeatmapListWidgetFallback(scene=scene, options=options or {})
