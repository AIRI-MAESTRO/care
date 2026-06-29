"""DAG → box-and-arrow terminal rendering.

When MAGE's ``build_dag`` stage completes it hands CARE a
``DAGStructure`` describing the chain's step graph. The chat stage
trail used to collapse that into a single ``"5 nodes, 4 edges"``
count (see :func:`care.intermediate_artifacts._summarise_dag`). This
module renders the same payload as an inline **box-and-arrow graph**
so the user can actually *see* the chain's shape — including genuine
parallel branches drawn as side-by-side box columns.

Design notes:

- **Pure + payload-shape tolerant.** :func:`render_dag_boxes` takes
  whatever the stage produced — a Pydantic ``DAGStructure`` already
  projected to a dict, a ``{"nodes": [...], "edges": [...]}`` mapping,
  or a list of step dicts that carry their own ``dependencies`` — and
  normalises all of them to one ``(nodes, edges)`` model. No upstream
  import, so a broken MAGE install can't break the render.
- **Layered 2-D layout.** Nodes are assigned a *layer* by longest path
  from a root, so every dependency sits in a strictly earlier layer.
  Each layer is drawn as a centred row of boxes; parallel steps in the
  same layer appear side by side. Edges between adjacent layers are
  routed on a small character canvas with proper box-drawing junctions
  (``┬ ┴ ├ ┤ ┼``) so forks and joins read as forks and joins.
- **Compact mode for wide graphs.** When the full-label layout would
  exceed the width budget (lots of parallel branches), the boxes shrink
  to bare step numbers and a numbered legend underneath spells out what
  each step is. The graph shape stays intact; only the labels move.
- **Skip edges stay honest.** An edge that spans more than one layer
  (e.g. ``1 → 4`` alongside ``1 → 2 → 3 → 4``) would have to cross an
  intervening row of boxes, so instead of drawing through them it is
  surfaced as a ``◀ N`` annotation on the dependent box. Nothing is
  lost — the annotation names the source step.
- **Cycle-safe.** A dependency cycle can't be layered, so the renderer
  falls back to a single flat column with every edge annotated, under
  a ``(cycle detected)`` marker.

The result is a ``list[str]`` of plain lines so the caller can emit
each as a ``⎿``-prefixed stage-trail sub-row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text

__all__ = [
    "render_dag_boxes",
    "render_dag_styled",
    "render_dag_diff",
    "render_dag_mermaid",
    "diff_chains",
    "dag_display_opts",
]

# Box-drawing glyphs, isolated so a future ASCII-only fallback has one
# place to swap.
_TOP_L, _TOP_R = "┌", "┐"
_BOT_L, _BOT_R = "└", "┘"
_H, _V = "─", "│"
_ARROW_DOWN = "▼"
_ARROW_RIGHT = "▶"
_ARROW_BACK = "◀"

_DEFAULT_MAX_WIDTH = 32
# Above this rendered width the full-label layout collapses to the
# compact number-box + legend variant.
_DEFAULT_MAX_GRAPH_WIDTH = 72
_GUTTER = 3  # blank columns between side-by-side boxes in a layer
_LR_HGAP = 4  # columns of edge band between layer columns (left-to-right)
_LR_VGUTTER = 1  # blank rows between stacked boxes in a column (left-to-right)

# Edge-segment direction bits, OR-combined per canvas cell so junctions
# resolve to the right box-drawing glyph regardless of draw order.
_U, _D, _L, _R = 1, 2, 4, 8
_MASK_GLYPH = {
    0: " ",
    _U: _V, _D: _V, _U | _D: _V,
    _L: _H, _R: _H, _L | _R: _H,
    _D | _R: "┌", _D | _L: "┐", _U | _R: "└", _U | _L: "┘",
    _U | _D | _R: "├", _U | _D | _L: "┤",
    _L | _R | _D: "┬", _L | _R | _U: "┴",
    _U | _D | _L | _R: "┼",
}

_STEP_TYPE_LABELS = {
    "llm": "AI",
    "tool": "Tool",
    "mcp": "MCP",
    "code": "Code",
    "python": "Code",
}

# --- Styling (used only by render_dag_styled) ------------------------------
# Rich style strings, kept as theme-agnostic named colours so they read on
# both light and dark terminals without reaching for Textual `$tokens`
# (which aren't valid Rich styles outside a widget's own markup).

# Per-status box tint for the live-run overlay. The box colour then encodes
# how far the chain has progressed, so the graph doubles as a progress view.
_STATUS_STYLE = {
    "pending": "grey50",
    "running": "bold yellow",
    "done": "green",
    "failed": "bold red",
    "skipped": "grey50",
}

# Per-type box tint used when no run-status overlay is supplied — the box
# colour then encodes the step's kind (AI / Tool / MCP / Code).
_TYPE_STYLE = {
    "llm": "cyan",
    "ai": "cyan",
    "tool": "magenta",
    "mcp": "blue",
    "code": "green",
    "python": "green",
}

# Edges, arrowheads and gutters render muted so the coloured boxes carry the
# signal.
_EDGE_STYLE = "grey50"

# Per-change tint for the version diff overlay (old chain → new chain).
_DIFF_STYLE = {
    "added": "bold green",
    "changed": "bold yellow",
    "removed": "bold red",
    "unchanged": "grey50",
}

# Three-stop heat scale (low → high) for the run-metric overlay; a node
# with no recorded metric stays muted.
_HEAT_STYLE = ("green", "yellow", "bold red")

# Boxes outside the selected node's lineage when ``dim_unrelated`` is on —
# dimmer than the edges so the highlighted path reads as the foreground.
_DIM_STYLE = "grey37"

# ASCII fallback: every box-drawing / decorative glyph mapped to a single
# ASCII char so the swap is width-preserving (alignment unchanged). Applied
# as a post-process so the layout engine itself stays Unicode-only.
_ASCII_TABLE = str.maketrans({
    "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    "─": "-", "│": "|",
    "▼": "v", "◀": "<", "▶": ">",
    "…": "~", "·": ".", "—": "-",
})


def dag_display_opts(config: Any) -> dict[str, Any]:
    """Universally-safe DAG render prefs read off a ``CareConfig`` (the
    ASCII glyph swap + bus-lane skip routing). Duck-typed and defensive —
    a ``None`` / partial config reads as all-defaults — so every screen
    can spread ``**dag_display_opts(self.app.config)`` into its
    ``render_dag_*`` call.

    Layout is intentionally excluded: left-to-right only suits the wide,
    horizontally-scrollable DAG modal, so that surface reads
    ``defaults.dag_layout`` itself rather than forcing it on the narrow,
    vertical chat/inspect/run panes.
    """
    defaults = getattr(config, "defaults", None)
    return {
        "ascii_only": bool(getattr(defaults, "dag_ascii", False)),
        "bus_lanes": bool(getattr(defaults, "dag_bus_lanes", False)),
    }


@dataclass
class _Node:
    """One graph node in input order."""

    ref: str  # display reference — step number, id, or position
    label: str  # human label, already including the ``(Type)`` suffix
    deps: list[str] = field(default_factory=list)  # incoming refs
    node_type: str = ""  # normalised raw step type (``"llm"``/``"tool"``/…)


def render_dag_boxes(
    payload: Any,
    *,
    max_width: int = _DEFAULT_MAX_WIDTH,
    max_graph_width: int = _DEFAULT_MAX_GRAPH_WIDTH,
    max_lines: int = 1,
    ascii_only: bool = False,
    layout: str = "tb",
    bus_lanes: bool = False,
) -> list[str]:
    """Render a DAG payload as a box-and-arrow graph.

    When the full-label layout would be wider than ``max_graph_width``
    it collapses to the compact variant: boxes carry only the step
    number and a numbered legend underneath describes each one.

    ``max_lines`` lets full-label boxes wrap their interior across up to
    that many rows instead of truncating with ``…`` (used by the modal,
    which has the room). ``ascii_only`` swaps the box-drawing glyphs for
    plain ASCII (``+ - | v <``) for non-Unicode terminals / clean copy-
    paste — width-preserving so alignment is unchanged.

    ``layout="lr"`` draws the graph left-to-right (layers as columns) —
    a deep linear chain becomes a wide strip instead of a tall column.
    ``bus_lanes`` (top-down only) draws multi-layer "skip" deps as routed
    left-margin channels instead of ``◀ N`` annotations.

    Returns an empty list when the payload carries no recognisable
    nodes so the caller can fall back to its terse count summary.
    """
    nodes = _normalize(payload)
    if not nodes:
        return []

    ordered, has_cycle = _topo_order(nodes)
    if has_cycle:
        return _maybe_ascii(_render_flat(nodes, max_width), ascii_only)

    layer_of = _assign_layers(ordered)
    layers = _order_within_layers(nodes, layer_of)
    skip = {n.ref: _skip_deps(n, layer_of) for n in nodes}

    if layout == "lr":
        inner = _inner_width(
            [_content(n, skip[n.ref]) for n in nodes], max_width,
        )
        return _maybe_ascii(
            _draw_graph_lr(
                layers, inner,
                label_of=lambda n: _content(n, skip[n.ref]),
                max_lines=max_lines,
            ),
            ascii_only,
        )

    skip_edges = (
        [(dep, ref) for ref, deps in skip.items() for dep in deps]
        if bus_lanes else None
    )
    skip_for = (lambda ref: []) if bus_lanes else (lambda ref: skip[ref])

    full_inner = _inner_width(
        [_content(n, skip_for(n.ref)) for n in nodes], max_width,
    )
    if _canvas_width(layers, full_inner) <= max_graph_width:
        return _maybe_ascii(
            _draw_graph(
                layers, full_inner,
                label_of=lambda n: _content(n, skip_for(n.ref)),
                max_lines=max_lines,
                skip_edges=skip_edges,
            ),
            ascii_only,
        )

    # Too wide — shrink boxes to bare step numbers and move the
    # descriptions into a legend beneath the graph.
    ref_inner = _inner_width([n.ref for n in nodes], max_width)
    lines = _draw_graph(
        layers, ref_inner, label_of=lambda n: n.ref, skip_edges=skip_edges,
    )
    lines.append("")
    lines.extend(_legend(nodes, {} if bus_lanes else skip))
    return _maybe_ascii(lines, ascii_only)


def render_dag_styled(
    payload: Any,
    *,
    status_by_ref: dict[str, str] | None = None,
    diff_by_ref: dict[str, str] | None = None,
    metric_by_ref: dict[str, float] | None = None,
    highlight_ref: str | None = None,
    dim_unrelated: bool = False,
    max_width: int = _DEFAULT_MAX_WIDTH,
    max_graph_width: int = _DEFAULT_MAX_GRAPH_WIDTH,
    max_lines: int = 1,
    ascii_only: bool = False,
    layout: str = "tb",
    bus_lanes: bool = False,
    geometry: dict[tuple[int, int], str] | None = None,
) -> list[Text]:
    """Render a DAG payload as a **colour-tinted** box-and-arrow graph.

    Identical layout to :func:`render_dag_boxes` — same layering, edge
    routing, compact fallback and cycle handling — but every box is
    tinted so the graph carries more than shape:

    * When ``status_by_ref`` is supplied (a ``{ref: status}`` map where
      status is ``"pending"`` / ``"running"`` / ``"done"`` / ``"failed"``
      — e.g. live CARL run state keyed by step number) the box colour
      encodes run status, so the graph doubles as a live progress view.
      Nodes absent from the map render as ``"pending"``.
    * Otherwise the box colour encodes the step *type* (AI / Tool / MCP
      / Code), so a static chain still reads at a glance.

    Two further overlays take precedence over status/type when supplied:
    ``diff_by_ref`` (``{ref: "added"|"changed"|"removed"|"unchanged"}`` —
    used by :func:`render_dag_diff`) and ``metric_by_ref`` (``{ref:
    number}`` — a run metric rendered as a low→high heat scale).

    ``highlight_ref`` (a node ref) bold-underlines one box so a caller can
    show "this is the selected step" — used by the DAG modal to link its
    clickable step list to the graph. With ``dim_unrelated`` the boxes
    that are neither an ancestor nor a descendant of the highlighted node
    are muted, so the selected step's data-flow lineage stands out.

    ``max_lines`` / ``ascii_only`` behave as in :func:`render_dag_boxes`.

    Returns a ``list[rich.text.Text]`` (one per line). Returns ``[]`` for
    an unrecognisable payload, mirroring :func:`render_dag_boxes`, so
    callers can fall back to a terse summary line.
    """
    nodes = _normalize(payload)
    if not nodes:
        return []

    by_ref = {n.ref: n for n in nodes}
    related = (
        _related_set(nodes, highlight_ref)
        if dim_unrelated and highlight_ref is not None
        else None
    )
    metric_max = max(metric_by_ref.values(), default=0.0) if metric_by_ref else 0.0

    def style_of(ref: str | None) -> str:
        # Non-box cells (edges / arrowheads / gutters) → muted.
        if not ref:
            return _EDGE_STYLE
        # Base tint precedence: diff > metric > run-status > step-type.
        if diff_by_ref is not None:
            base = _DIFF_STYLE.get(diff_by_ref.get(ref, "unchanged"), "")
        elif metric_by_ref is not None:
            base = _heat_style(metric_by_ref.get(ref), metric_max)
        elif status_by_ref is not None:
            base = _STATUS_STYLE.get(status_by_ref.get(ref, "pending"), "")
        else:
            node = by_ref.get(ref)
            base = _TYPE_STYLE.get(node.node_type, "") if node else ""
        if highlight_ref is not None and ref == highlight_ref:
            # Bold + underline keeps the type/status hue but makes the
            # selected box stand out; both survive the per-span → Textual
            # Content round-trip (a `reverse` attribute does not) and read
            # on any theme.
            return f"bold underline {base}".strip()
        if related is not None and ref not in related:
            # Outside the selected node's lineage → recede.
            return _DIM_STYLE
        return base

    ordered, has_cycle = _topo_order(nodes)
    if has_cycle:
        return _maybe_ascii(
            _render_flat_styled(nodes, max_width, style_of), ascii_only,
        )

    layer_of = _assign_layers(ordered)
    layers = _order_within_layers(nodes, layer_of)
    skip = {n.ref: _skip_deps(n, layer_of) for n in nodes}

    if layout == "lr":
        # Left-to-right: skip deps keep their ◀ annotation (no compact /
        # bus-lane handling in this orientation).
        inner = _inner_width(
            [_content(n, skip[n.ref]) for n in nodes], max_width,
        )
        return _maybe_ascii(
            _draw_graph_lr(
                layers, inner,
                label_of=lambda n: _content(n, skip[n.ref]),
                style_of=style_of,
                max_lines=max_lines,
                geometry=geometry,
            ),
            ascii_only,
        )

    # Top-down. With bus lanes the skip deps are drawn as left-margin
    # channels instead of ◀ annotations, so keep them out of the boxes.
    skip_edges = (
        [(dep, ref) for ref, deps in skip.items() for dep in deps]
        if bus_lanes else None
    )
    skip_for = (lambda ref: []) if bus_lanes else (lambda ref: skip[ref])

    full_inner = _inner_width(
        [_content(n, skip_for(n.ref)) for n in nodes], max_width,
    )
    if _canvas_width(layers, full_inner) <= max_graph_width:
        return _maybe_ascii(
            _draw_graph(
                layers, full_inner,
                label_of=lambda n: _content(n, skip_for(n.ref)),
                style_of=style_of,
                max_lines=max_lines,
                geometry=geometry,
                skip_edges=skip_edges,
            ),
            ascii_only,
        )

    ref_inner = _inner_width([n.ref for n in nodes], max_width)
    lines = _draw_graph(
        layers, ref_inner, label_of=lambda n: n.ref, style_of=style_of,
        geometry=geometry, skip_edges=skip_edges,
    )
    lines.append(Text(""))
    lines.extend(
        _legend_styled(nodes, {} if bus_lanes else skip, style_of)
    )
    return _maybe_ascii(lines, ascii_only)


def render_dag_mermaid(payload: Any, *, layout: str = "tb") -> str:
    """Render a DAG payload as Mermaid ``flowchart`` source, ready to paste
    into Markdown / a PR / docs. Direction follows ``layout`` (``"tb"`` →
    ``TD``, ``"lr"`` → ``LR``). Returns ``""`` for an unrecognisable
    payload."""
    nodes = _normalize(payload)
    if not nodes:
        return ""
    known = {n.ref for n in nodes}
    direction = "LR" if layout == "lr" else "TD"
    lines = [f"flowchart {direction}"]
    for node in nodes:
        label = _mermaid_label(f"{node.ref} · {node.label}")
        lines.append(f'    {_mermaid_id(node.ref)}["{label}"]')
    for node in nodes:
        for dep in node.deps:
            if dep in known:
                lines.append(
                    f"    {_mermaid_id(dep)} --> {_mermaid_id(node.ref)}"
                )
    return "\n".join(lines)


def _mermaid_id(ref: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(ref))
    return f"n{safe}" if safe else "n_"


def _mermaid_label(text: str) -> str:
    # Inside a "…" Mermaid label: swap the only breaking char (") and
    # flatten newlines; box-drawing isn't involved here.
    return text.replace('"', "'").replace("\n", " ")


def _heat_style(value: float | None, vmax: float) -> str:
    """Map a metric value to a low/mid/high heat tint. A node with no
    recorded metric (``None``) or a zero ceiling stays muted."""
    if value is None or vmax <= 0:
        return _DIM_STYLE
    frac = value / vmax
    idx = 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)
    return _HEAT_STYLE[idx]


# ---------------------------------------------------------------------------
# Version diff
# ---------------------------------------------------------------------------

# Fields excluded when deciding whether a step "changed": ``number`` /
# ``index`` reshuffle on every structural edit, and the raw dep numbers are
# compared by referenced-step identity instead (below) so renumbering a
# dependency doesn't read as a change.
_DIFF_IGNORE = {"number", "index", "dependencies", "deps", "depends_on"}


def render_dag_diff(
    old_payload: Any, new_payload: Any, **opts: Any,
) -> list[Text]:
    """Render ``new_payload``'s DAG tinted as a diff against ``old_payload``
    — added steps green, changed steps amber, unchanged muted — with a red
    ``removed:`` list beneath for steps the edit dropped.

    Steps are matched by title (then number) rather than position, so the
    renumbering a revise does isn't mistaken for churn. ``opts`` forward to
    :func:`render_dag_styled` (``ascii_only`` / ``max_width`` / …)."""
    status, removed = diff_chains(old_payload, new_payload)
    lines = render_dag_styled(new_payload, diff_by_ref=status, **opts)
    if removed:
        lines.append(Text(""))
        lines.append(Text("removed:", style=_DIFF_STYLE["removed"]))
        for ref, label in removed:
            lines.append(
                Text(f"  - {ref}  {label}", style=_DIFF_STYLE["removed"])
            )
    return lines


def diff_chains(
    old_payload: Any, new_payload: Any,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Diff two chain payloads by step identity.

    Returns ``(status_by_ref, removed)`` where ``status_by_ref`` maps each
    *new* node's ref to ``"added"`` / ``"changed"`` / ``"unchanged"`` and
    ``removed`` lists ``(ref, label)`` for steps only in the old chain.
    Matching is by title (falling back to number); dependency comparison
    resolves dep numbers to the referenced step's title so a reshuffle
    isn't flagged as a change."""
    old_steps = _diff_steps(old_payload)
    new_steps = _diff_steps(new_payload)
    old_key = {_node_ref(s, i): _step_key(s, i) for i, s in enumerate(old_steps)}
    new_key = {_node_ref(s, i): _step_key(s, i) for i, s in enumerate(new_steps)}

    old_by_key: dict[str, dict] = {}
    for i, step in enumerate(old_steps):
        old_by_key.setdefault(_step_key(step, i), step)

    status: dict[str, str] = {}
    for i, step in enumerate(new_steps):
        key = _step_key(step, i)
        ref = _node_ref(step, i)
        if key not in old_by_key:
            status[ref] = "added"
        else:
            same = _norm_step(old_by_key[key], old_key) == _norm_step(
                step, new_key
            )
            status[ref] = "unchanged" if same else "changed"

    new_keys = {_step_key(s, i) for i, s in enumerate(new_steps)}
    removed = [
        (_node_ref(s, i), _node_label(s, _node_ref(s, i)))
        for i, s in enumerate(old_steps)
        if _step_key(s, i) not in new_keys
    ]
    return status, removed


def _diff_steps(payload: Any) -> list[dict]:
    return [s for s in _node_items(payload) if isinstance(s, dict)]


def _step_key(step: dict, idx: int) -> str:
    """Stable identity for diff matching — title-ish, lower-cased, with a
    positional fallback for untitled steps."""
    for key in ("title", "name", "step_name"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return f"#{idx}"


def _norm_step(step: dict, key_map: dict[str, str]) -> dict:
    """Normalise a step for change detection: drop volatile numbering and
    re-express dependencies as the *identities* of the steps they point
    at, so a renumber alone doesn't count as a change."""
    out = {k: v for k, v in step.items() if k not in _DIFF_IGNORE}
    out["__deps"] = sorted(
        key_map.get(str(d), str(d)) for d in _coerce_deps(step)
    )
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalize(payload: Any) -> list[_Node]:
    """Project an arbitrary DAG payload into ordered :class:`_Node`s.

    Accepts the three shapes CARE sees in practice:

    * ``{"nodes": [...], "edges": [...]}`` — explicit node + edge lists.
    * ``{"steps": [...]}`` / a bare ``[...]`` — step dicts whose
      ``dependencies`` / ``deps`` carry the edges inline.
    * Nodes that are bare scalars (``[1, 2, 3]``) — positional refs
      with no edges.
    """
    items = _node_items(payload)
    if not items:
        return []

    nodes: list[_Node] = []
    # Map every id-ish value a node answers to → its canonical ref so
    # edge endpoints (which may reference number, id, or name) resolve.
    alias_to_ref: dict[str, str] = {}
    for idx, item in enumerate(items):
        ref = _node_ref(item, idx)
        nodes.append(
            _Node(
                ref=ref,
                label=_node_label(item, ref),
                node_type=_raw_type(item),
            )
        )
        for alias in _node_aliases(item, ref):
            alias_to_ref.setdefault(alias, ref)

    by_ref = {n.ref: n for n in nodes}

    # Inline dependencies carried on each node/step.
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        ref = nodes[idx].ref
        for dep in _coerce_deps(item):
            resolved = alias_to_ref.get(str(dep))
            if resolved and resolved != ref:
                _add_dep(by_ref[ref], resolved)

    # Explicit edge list at the payload level.
    for edge in _edge_items(payload):
        frm, to = _edge_endpoints(edge)
        src = alias_to_ref.get(str(frm)) if frm is not None else None
        dst = alias_to_ref.get(str(to)) if to is not None else None
        if src and dst and src != dst:
            _add_dep(by_ref[dst], src)

    return nodes


def _node_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("nodes", "steps", "vertices"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(payload, (list, tuple)):
        return list(payload)
    return []


def _edge_items(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    for key in ("edges", "dependencies", "links"):
        value = payload.get(key)
        if isinstance(value, list):
            # A node-level ``dependencies`` list of ints isn't an edge
            # list — only treat dicts / pairs as explicit edges.
            if all(isinstance(e, (dict, list, tuple)) for e in value):
                return value
    return []


def _node_ref(item: Any, idx: int) -> str:
    if isinstance(item, dict):
        for key in ("number", "id", "step_id", "index"):
            value = item.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
    elif isinstance(item, (str, int)):
        text = str(item).strip()
        if text:
            return text
    return str(idx + 1)


def _node_aliases(item: Any, ref: str) -> list[str]:
    aliases = [ref]
    if isinstance(item, dict):
        for key in ("number", "id", "step_id", "name", "step_name", "label"):
            value = item.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                aliases.append(str(value).strip())
    return aliases


def _node_label(item: Any, ref: str) -> str:
    if not isinstance(item, dict):
        return f"step {ref}"
    title = ""
    for key in ("title", "name", "step_name", "label"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            title = value.strip()
            break
    if not title:
        title = f"step {ref}"
    type_label = _type_label(item)
    if type_label and type_label.lower() != title.lower():
        return f"{title} ({type_label})"
    return title


def _type_label(item: dict) -> str:
    raw = _raw_type(item)
    if not raw:
        return ""
    return _STEP_TYPE_LABELS.get(raw, raw.replace("_", " ").title())


def _raw_type(item: Any) -> str:
    """Normalised raw step type (``"llm"`` / ``"tool"`` / …) — the key the
    type-colour map is keyed on. Empty for non-dict / untyped nodes."""
    if not isinstance(item, dict):
        return ""
    return str(item.get("step_type") or item.get("type") or "").strip().lower()


def _coerce_deps(item: dict) -> list[Any]:
    deps = (
        item.get("dependencies")
        or item.get("deps")
        or item.get("depends_on")
        or ()
    )
    if isinstance(deps, (str, int)):
        return [deps]
    if isinstance(deps, (list, tuple)):
        return list(deps)
    return []


def _edge_endpoints(edge: Any) -> tuple[Any, Any]:
    if isinstance(edge, dict):
        frm = (
            edge.get("from")
            or edge.get("source")
            or edge.get("src")
            or edge.get("u")
        )
        to = (
            edge.get("to")
            or edge.get("target")
            or edge.get("dst")
            or edge.get("v")
        )
        return frm, to
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return edge[0], edge[1]
    return None, None


def _add_dep(node: _Node, ref: str) -> None:
    if ref not in node.deps:
        node.deps.append(ref)


# ---------------------------------------------------------------------------
# Topological ordering + layering
# ---------------------------------------------------------------------------


def _topo_order(nodes: list[_Node]) -> tuple[list[_Node], bool]:
    """Kahn's algorithm, stable by input order. Returns the ordered
    nodes plus a flag that's ``True`` when a cycle left some nodes
    unresolved (those are appended in input order)."""
    by_ref = {n.ref: n for n in nodes}
    indeg = {n.ref: sum(1 for d in n.deps if d in by_ref) for n in nodes}
    children: dict[str, list[str]] = {n.ref: [] for n in nodes}
    for n in nodes:
        for d in n.deps:
            if d in children:
                children[d].append(n.ref)

    ready = [n.ref for n in nodes if indeg[n.ref] == 0]
    ordered: list[_Node] = []
    emitted: set[str] = set()
    while ready:
        ref = ready.pop(0)
        if ref in emitted:
            continue
        emitted.add(ref)
        ordered.append(by_ref[ref])
        for child in children[ref]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)

    has_cycle = len(ordered) < len(nodes)
    if has_cycle:
        for n in nodes:
            if n.ref not in emitted:
                ordered.append(n)
    return ordered, has_cycle


def _assign_layers(ordered: list[_Node]) -> dict[str, int]:
    """Longest-path layering: a node sits one layer below its
    deepest dependency. ``ordered`` is in topological order so every
    dependency's layer is known by the time we reach the node."""
    by_ref = {n.ref for n in ordered}
    layer: dict[str, int] = {}
    for n in ordered:
        deps = [d for d in n.deps if d in by_ref]
        layer[n.ref] = 0 if not deps else 1 + max(layer[d] for d in deps)
    return layer


def _order_within_layers(
    nodes: list[_Node], layer_of: dict[str, int],
) -> list[list[_Node]]:
    """Order nodes within each layer to minimise edge crossings.

    Groups nodes by layer (seeded in input order) then runs alternating
    downward/upward barycenter sweeps — the standard Sugiyama heuristic —
    keeping whichever ordering produced the fewest crossings. Beats a
    single downward pass on fan-in/out graphs, where a node's best column
    depends on the layers both above and below it."""
    n_layers = max(layer_of.values()) + 1
    layers: list[list[_Node]] = [[] for _ in range(n_layers)]
    for n in nodes:
        layers[layer_of[n.ref]].append(n)
    if n_layers < 2:
        return layers

    # A node's successors in the next layer, for the upward sweep.
    present = {n.ref for n in nodes}
    children: dict[str, list[str]] = {n.ref: [] for n in nodes}
    for n in nodes:
        for d in n.deps:
            if d in present:
                children[d].append(n.ref)

    best = [list(row) for row in layers]
    best_cross = _count_crossings(layers)
    # The chains CARE renders are small, so a handful of sweeps converges;
    # we keep the global best ordering seen across them.
    for _ in range(4):
        _sweep_down(layers)
        _sweep_up(layers, children)
        cross = _count_crossings(layers)
        if cross < best_cross:
            best_cross, best = cross, [list(row) for row in layers]
            if best_cross == 0:
                break
    return best


def _sweep_down(layers: list[list[_Node]]) -> None:
    """Re-order each layer (top→bottom) by the barycenter of each node's
    parents in the layer above."""
    for depth in range(1, len(layers)):
        prev_pos = {n.ref: i for i, n in enumerate(layers[depth - 1])}
        layers[depth] = _by_barycenter(
            layers[depth], prev_pos, lambda node: node.deps,
        )


def _sweep_up(layers: list[list[_Node]], children: dict[str, list[str]]) -> None:
    """Re-order each layer (bottom→top) by the barycenter of each node's
    children in the layer below."""
    for depth in range(len(layers) - 2, -1, -1):
        next_pos = {n.ref: i for i, n in enumerate(layers[depth + 1])}
        layers[depth] = _by_barycenter(
            layers[depth], next_pos, lambda node: children[node.ref],
        )


def _by_barycenter(
    row: list[_Node],
    neighbor_pos: dict[str, int],
    neighbors_of: Callable[[_Node], list[str]],
) -> list[_Node]:
    """Stable-sort ``row`` by the mean position of each node's neighbours;
    nodes with no placed neighbour keep their current slot."""
    def key(item: tuple[int, _Node]) -> float:
        i, node = item
        cols = [neighbor_pos[r] for r in neighbors_of(node) if r in neighbor_pos]
        return sum(cols) / len(cols) if cols else float(i)

    return [node for _, node in sorted(enumerate(row), key=key)]


def _count_crossings(layers: list[list[_Node]]) -> int:
    """Total edge crossings across every adjacent-layer gap. Two edges
    cross when their endpoints are in opposite left-right order in the two
    layers; only direct (single-layer) edges count — skip-layer deps are
    drawn as annotations, not lines."""
    total = 0
    for depth in range(1, len(layers)):
        upper = {n.ref: i for i, n in enumerate(layers[depth - 1])}
        edges: list[tuple[int, int]] = []
        for lower_idx, node in enumerate(layers[depth]):
            for dep in node.deps:
                if dep in upper:
                    edges.append((lower_idx, upper[dep]))
        for a in range(len(edges)):
            la, ua = edges[a]
            for b in range(a + 1, len(edges)):
                lb, ub = edges[b]
                if (la < lb and ua > ub) or (la > lb and ua < ub):
                    total += 1
    return total


def _skip_deps(node: _Node, layer_of: dict[str, int]) -> list[str]:
    """Dependencies that span more than one layer — drawn as a
    ``◀ N`` annotation rather than a line, since a line would cross
    the intervening box rows."""
    here = layer_of[node.ref]
    return [
        d for d in node.deps
        if d in layer_of and here - layer_of[d] != 1
    ]


def _related_set(nodes: list[_Node], ref: str) -> set[str]:
    """Refs in ``ref``'s data-flow lineage: its transitive ancestors
    (what it depends on), its transitive descendants (what depends on
    it), and itself. Everything else can be dimmed."""
    by_ref = {n.ref: n for n in nodes}
    if ref not in by_ref:
        return set()
    children: dict[str, list[str]] = {n.ref: [] for n in nodes}
    for n in nodes:
        for d in n.deps:
            if d in children:
                children[d].append(n.ref)

    related = {ref}
    # Ancestors — walk dependency edges upward.
    stack = list(by_ref[ref].deps)
    while stack:
        r = stack.pop()
        if r in by_ref and r not in related:
            related.add(r)
            stack.extend(by_ref[r].deps)
    # Descendants — walk dependent edges downward.
    stack = list(children[ref])
    while stack:
        r = stack.pop()
        if r not in related:
            related.add(r)
            stack.extend(children.get(r, []))
    return related


# ---------------------------------------------------------------------------
# Content sizing
# ---------------------------------------------------------------------------


def _content(node: _Node, skip: list[str]) -> str:
    base = f"{node.ref} · {node.label}"
    if skip:
        return f"{base}  {_ARROW_BACK} {', '.join(skip)}"
    return base


def _inner_width(contents: list[str], max_width: int) -> int:
    inner = max((len(c) for c in contents), default=1)
    return max(1, min(inner, max_width))


def _fit(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    """Greedily word-wrap ``text`` to at most ``max_lines`` lines of
    ``width`` columns. The final line is truncated with an ellipsis when
    content overflows, so nothing escapes the box. ``max_lines == 1``
    degrades to a single :func:`_fit` line (the legacy behaviour)."""
    if max_lines <= 1 or len(text) <= width:
        return [_fit(text, width)]
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    idx = 0
    while idx < len(words) and len(lines) < max_lines - 1:
        word = words[idx]
        cand = word if not cur else f"{cur} {word}"
        if len(cand) <= width:
            cur = cand
            idx += 1
        elif cur:
            lines.append(cur)
            cur = ""
        else:
            # A single word wider than the box — hard-break it.
            lines.append(_fit(word, width))
            idx += 1
    remaining = " ".join(([cur] if cur else []) + words[idx:]).strip()
    lines.append(_fit(remaining, width))
    return lines


# ---------------------------------------------------------------------------
# 2-D layout + canvas drawing
# ---------------------------------------------------------------------------


def _layer_widths(layers: list[list[_Node]], inner: int) -> list[int]:
    box_w = inner + 4  # ┌ + (inner+2) + ┐
    return [
        len(row) * box_w + max(0, len(row) - 1) * _GUTTER for row in layers
    ]


def _canvas_width(layers: list[list[_Node]], inner: int) -> int:
    widths = _layer_widths(layers, inner)
    return max(widths) if widths else 1


def _draw_graph(
    layers: list[list[_Node]],
    inner: int,
    *,
    label_of,
    style_of: Callable[[str | None], str] | None = None,
    max_lines: int = 1,
    skip_edges: list[tuple[str, str]] | None = None,
    geometry: dict[tuple[int, int], str] | None = None,
) -> list[str] | list[Text]:
    """Draw the layered graph onto a character canvas. ``label_of``
    maps a node to the text shown inside its box (the full label in
    normal mode, just the step number in compact mode) — edges and
    layout are identical either way.

    ``max_lines`` caps how many rows a box interior may wrap to; the box
    height is uniform across the graph (the deepest-wrapping label wins).
    ``max_lines == 1`` keeps the legacy single-line, truncate-with-… box.

    When ``style_of`` is given the canvas is rendered to coloured
    ``rich.text.Text`` lines (``style_of(ref)`` resolves each box's
    tint; ``ref`` is ``None`` for non-box cells). Without it the plain
    ``list[str]`` rendering is returned, byte-for-byte unchanged."""
    box_w = inner + 4  # ┌ + (inner+2) + ┐
    center_off = box_w // 2

    # Layer index per node so edge routing knows which links span a
    # single layer (drawn) versus more (already peeled into the label).
    layer_of = {n.ref: depth for depth, row in enumerate(layers) for n in row}

    # Box interiors, wrapped to at most ``max_lines`` rows. Height is
    # uniform across the whole graph (the deepest label wins) so the grid
    # and edge bands stay regular; shorter labels are bottom-padded.
    content_of = {
        n.ref: _wrap(label_of(n), inner, max_lines)
        for row in layers for n in row
    }
    inner_rows = max((len(v) for v in content_of.values()), default=1)
    inner_rows = max(1, inner_rows)
    for ref, body in content_of.items():
        content_of[ref] = body + [""] * (inner_rows - len(body))
    box_h = inner_rows + 2  # top border + content rows + bottom border

    # Vertical placement: ``box_h`` rows per box + a 2-row edge band/gap.
    # Row of each box is independent of any left-margin reservation.
    layer_top = [depth * (box_h + 2) for depth in range(len(layers))]
    top_of = {ref: layer_top[layer_of[ref]] for ref in layer_of}

    # Bus lanes: reserve left-margin columns for skip edges (deps spanning
    # more than one layer) so they route around the boxes instead of
    # crashing through them. No skip edges → no margin, layout unchanged.
    lane_of, n_lanes = _assign_lanes(skip_edges or [], top_of, box_h)
    margin = (n_lanes + 1) if n_lanes else 0

    # Horizontal placement: lay each layer's boxes left-to-right, centre
    # within the widest layer, and shift right past the bus-lane margin.
    layer_w = _layer_widths(layers, inner)
    max_lw = max(layer_w) if layer_w else 1
    canvas_w = margin + max_lw
    canvas_h = (layer_top[-1] + box_h) if layers else 0

    left_of: dict[str, int] = {}
    center_of: dict[str, int] = {}
    for depth, row in enumerate(layers):
        offset = margin + (max_lw - layer_w[depth]) // 2
        for slot, node in enumerate(row):
            left = offset + slot * (box_w + _GUTTER)
            left_of[node.ref] = left
            center_of[node.ref] = left + center_off

    mask = [[0] * canvas_w for _ in range(canvas_h)]
    literal: dict[tuple[int, int], str] = {}

    # Boxes first; arrows/edges live in the gap rows so they never
    # collide with box glyphs. ``box_cells`` maps every cell a box
    # occupies back to its node ref so the styled renderer can tint the
    # box (border + content) without re-deriving geometry.
    box_cells: dict[tuple[int, int], str] = {}
    for node in (n for row in layers for n in row):
        top = top_of[node.ref]
        left = left_of[node.ref]
        _draw_box(literal, top, left, inner, content_of[node.ref])
        for rr in range(top, top + box_h):
            for cc in range(left, left + box_w):
                box_cells[(rr, cc)] = node.ref

    # Hand the cell→ref map back to a caller that wants click hit-testing
    # (the DAG modal maps a mouse offset to the box it lands on).
    if geometry is not None:
        geometry.update(box_cells)

    # Edges: only direct (single-layer) parent→child links are drawn;
    # multi-layer deps were peeled off into ``skip`` annotations.
    for node in (n for row in layers for n in row):
        here = layer_of[node.ref]
        for dep in node.deps:
            if dep not in layer_of or here - layer_of[dep] != 1:
                continue
            bus_row = top_of[dep] + box_h  # just below the parent box
            arrow_row = bus_row + 1
            _route_edge(
                mask, literal, bus_row, arrow_row,
                center_of[dep], center_of[node.ref],
            )

    # Skip edges (multi-layer deps) routed through their reserved margin
    # lanes — down the left channel, then into the child's incoming band.
    for src, dst in skip_edges or []:
        if src not in top_of or dst not in top_of:
            continue
        lane_col = lane_of[(src, dst)]
        exit_row = top_of[src] + box_h
        entry_row = top_of[dst] - 2
        _route_skip_edge(
            mask, literal, lane_col, exit_row, entry_row,
            center_of[src], center_of[dst],
        )

    if style_of is None:
        return _render_canvas(mask, literal, canvas_h, canvas_w)
    return _render_canvas_styled(
        mask, literal, box_cells, canvas_h, canvas_w, style_of,
    )


def _draw_graph_lr(
    layers: list[list[_Node]],
    inner: int,
    *,
    label_of,
    style_of: Callable[[str | None], str] | None = None,
    max_lines: int = 1,
    geometry: dict[tuple[int, int], str] | None = None,
) -> list[str] | list[Text]:
    """Left-to-right twin of :func:`_draw_graph`: layers become columns, a
    layer's nodes stack vertically, and edges flow rightward ending in
    ``▶`` arrowheads. Trades the tall column a deep linear chain makes in
    the top-down layout for a wide strip that scrolls horizontally."""
    box_w = inner + 4
    layer_of = {n.ref: depth for depth, row in enumerate(layers) for n in row}

    # Box interiors (same wrapping / uniform height as the top-down path).
    content_of = {
        n.ref: _wrap(label_of(n), inner, max_lines)
        for row in layers for n in row
    }
    inner_rows = max(1, max((len(v) for v in content_of.values()), default=1))
    for ref, body in content_of.items():
        content_of[ref] = body + [""] * (inner_rows - len(body))
    box_h = inner_rows + 2

    # One column per layer; a ``box_w``-wide box + a ``_LR_HGAP`` edge band.
    layer_left = [depth * (box_w + _LR_HGAP) for depth in range(len(layers))]
    canvas_w = (layer_left[-1] + box_w) if layers else 1

    def col_height(row: list[_Node]) -> int:
        return len(row) * box_h + max(0, len(row) - 1) * _LR_VGUTTER

    canvas_h = max((col_height(row) for row in layers), default=1)

    left_of: dict[str, int] = {}
    top_of: dict[str, int] = {}
    vcenter_of: dict[str, int] = {}
    for depth, row in enumerate(layers):
        offset = (canvas_h - col_height(row)) // 2
        for slot, node in enumerate(row):
            top = offset + slot * (box_h + _LR_VGUTTER)
            left_of[node.ref] = layer_left[depth]
            top_of[node.ref] = top
            vcenter_of[node.ref] = top + box_h // 2

    mask = [[0] * canvas_w for _ in range(canvas_h)]
    literal: dict[tuple[int, int], str] = {}
    box_cells: dict[tuple[int, int], str] = {}
    for node in (n for row in layers for n in row):
        top, left = top_of[node.ref], left_of[node.ref]
        _draw_box(literal, top, left, inner, content_of[node.ref])
        for rr in range(top, top + box_h):
            for cc in range(left, left + box_w):
                box_cells[(rr, cc)] = node.ref
    if geometry is not None:
        geometry.update(box_cells)

    # Edges: only direct (single-column) parent→child links are drawn;
    # multi-layer deps ride the ◀ annotation, as in the top-down path.
    for node in (n for row in layers for n in row):
        here = layer_of[node.ref]
        for dep in node.deps:
            if dep not in layer_of or here - layer_of[dep] != 1:
                continue
            bus_col = left_of[dep] + box_w  # just right of the parent box
            arrow_col = left_of[node.ref] - 1  # just left of the child box
            _route_edge_lr(
                mask, literal, bus_col, arrow_col,
                vcenter_of[dep], vcenter_of[node.ref],
            )

    if style_of is None:
        return _render_canvas(mask, literal, canvas_h, canvas_w)
    return _render_canvas_styled(
        mask, literal, box_cells, canvas_h, canvas_w, style_of,
    )


def _draw_box(
    literal: dict[tuple[int, int], str],
    top: int,
    left: int,
    inner: int,
    content_lines: list[str],
) -> None:
    """Draw a box whose interior is ``content_lines`` (one row each,
    already fitted to ``inner``). Box height is ``len(content_lines) + 2``
    — callers pad the list so every box in a layer is the same height."""
    bar = _H * (inner + 2)
    _put_text(literal, top, left, f"{_TOP_L}{bar}{_TOP_R}")
    for i, line in enumerate(content_lines):
        _put_text(
            literal, top + 1 + i, left, f"{_V} {line.ljust(inner)} {_V}",
        )
    _put_text(
        literal,
        top + 1 + len(content_lines),
        left,
        f"{_BOT_L}{bar}{_BOT_R}",
    )


def _put_text(
    literal: dict[tuple[int, int], str], row: int, col: int, text: str,
) -> None:
    for i, ch in enumerate(text):
        literal[(row, col + i)] = ch


def _route_edge(
    mask: list[list[int]],
    literal: dict[tuple[int, int], str],
    bus_row: int,
    arrow_row: int,
    pcol: int,
    ccol: int,
) -> None:
    """Draw a parent→child edge across the two-row gap band.

    ``bus_row`` carries the vertical drop out of the parent plus any
    horizontal travel to reach the child's column; ``arrow_row`` holds
    the ``▼`` arrowhead directly above the child box.
    """
    if pcol == ccol:
        _add_mask(mask, bus_row, pcol, _U | _D)
    else:
        toward_child = _R if ccol > pcol else _L
        toward_parent = _L if ccol > pcol else _R
        _add_mask(mask, bus_row, pcol, _U | toward_child)
        step = 1 if ccol > pcol else -1
        for col in range(pcol + step, ccol, step):
            _add_mask(mask, bus_row, col, _L | _R)
        _add_mask(mask, bus_row, ccol, _D | toward_parent)
    literal[(arrow_row, ccol)] = _ARROW_DOWN


def _route_edge_lr(
    mask: list[list[int]],
    literal: dict[tuple[int, int], str],
    bus_col: int,
    arrow_col: int,
    prow: int,
    crow: int,
) -> None:
    """Right-flowing twin of :func:`_route_edge`: a parent→child edge that
    leaves the parent's right side, rides a vertical bus in ``bus_col`` to
    the child's row, then runs right to the ``▶`` arrowhead at the child's
    left edge."""
    if prow == crow:
        for col in range(bus_col, arrow_col):
            _add_mask(mask, prow, col, _L | _R)
    else:
        toward_child = _D if crow > prow else _U
        toward_parent = _U if crow > prow else _D
        _add_mask(mask, prow, bus_col, _L | toward_child)
        step = 1 if crow > prow else -1
        for row in range(prow + step, crow, step):
            _add_mask(mask, row, bus_col, _U | _D)
        _add_mask(mask, crow, bus_col, _R | toward_parent)
        for col in range(bus_col + 1, arrow_col):
            _add_mask(mask, crow, col, _L | _R)
    literal[(crow, arrow_col)] = _ARROW_RIGHT


def _assign_lanes(
    skip_edges: list[tuple[str, str]],
    top_of: dict[str, int],
    box_h: int,
) -> tuple[dict[tuple[str, str], int], int]:
    """Pack skip edges into the fewest left-margin lanes (columns) such
    that no two edges sharing a lane overlap vertically. Greedy
    interval-graph colouring by exit row. Returns ``(edge→lane, count)``;
    lane *i* lives in canvas column *i*."""
    spans: list[tuple[int, int, str, str]] = []
    for src, dst in skip_edges:
        if src not in top_of or dst not in top_of:
            continue
        exit_row = top_of[src] + box_h
        entry_row = top_of[dst] - 2
        lo, hi = (exit_row, entry_row) if exit_row <= entry_row else (entry_row, exit_row)
        spans.append((lo, hi, src, dst))
    spans.sort()

    lane_last: list[int] = []  # last occupied row per lane
    assign: dict[tuple[str, str], int] = {}
    for lo, hi, src, dst in spans:
        placed = False
        for li in range(len(lane_last)):
            if lane_last[li] < lo:
                lane_last[li] = hi
                assign[(src, dst)] = li
                placed = True
                break
        if not placed:
            assign[(src, dst)] = len(lane_last)
            lane_last.append(hi)
    return assign, len(lane_last)


def _route_skip_edge(
    mask: list[list[int]],
    literal: dict[tuple[int, int], str],
    lane_col: int,
    exit_row: int,
    entry_row: int,
    scol: int,
    dcol: int,
) -> None:
    """Route a multi-layer dep through its margin lane: out of the parent
    box's bottom, left into ``lane_col``, down the channel, then right into
    the child's incoming gap band with a ``▼`` arrowhead. All segments OR
    into the shared mask, so crossings resolve to proper junctions."""
    # Out of the parent (down into the gap row) and left to the lane.
    _add_mask(mask, exit_row, scol, _U | _L)
    for col in range(lane_col + 1, scol):
        _add_mask(mask, exit_row, col, _L | _R)
    _add_mask(mask, exit_row, lane_col, _R | _D)
    # Down the lane channel.
    for row in range(exit_row + 1, entry_row):
        _add_mask(mask, row, lane_col, _U | _D)
    _add_mask(mask, entry_row, lane_col, _U | _R)
    # Right into the child's column, then drop the arrowhead.
    for col in range(lane_col + 1, dcol):
        _add_mask(mask, entry_row, col, _L | _R)
    _add_mask(mask, entry_row, dcol, _L | _D)
    literal[(entry_row + 1, dcol)] = _ARROW_DOWN


def _add_mask(mask: list[list[int]], row: int, col: int, bits: int) -> None:
    if 0 <= row < len(mask) and 0 <= col < len(mask[0]):
        mask[row][col] |= bits


def _render_canvas(
    mask: list[list[int]],
    literal: dict[tuple[int, int], str],
    height: int,
    width: int,
) -> list[str]:
    lines: list[str] = []
    for r in range(height):
        chars: list[str] = []
        for c in range(width):
            ch = literal.get((r, c))
            if ch is None:
                ch = _MASK_GLYPH.get(mask[r][c], "?")
            chars.append(ch)
        lines.append("".join(chars).rstrip())
    return lines


def _render_canvas_styled(
    mask: list[list[int]],
    literal: dict[tuple[int, int], str],
    box_cells: dict[tuple[int, int], str],
    height: int,
    width: int,
    style_of: Callable[[str | None], str],
) -> list[Text]:
    """Styled twin of :func:`_render_canvas`: the same glyph grid, but
    each cell is appended to a ``Text`` with the style its owning box
    resolves to (non-box cells fall to ``style_of(None)``)."""
    lines: list[Text] = []
    for r in range(height):
        chars: list[str] = []
        for c in range(width):
            ch = literal.get((r, c))
            if ch is None:
                ch = _MASK_GLYPH.get(mask[r][c], "?")
            chars.append(ch)
        raw = "".join(chars).rstrip()
        text = Text()
        for c, ch in enumerate(raw):
            # Spaces carry no foreground, so leave them unstyled — keeps
            # the span list to the glyphs that actually show colour.
            if ch == " ":
                text.append(" ")
                continue
            text.append(ch, style=style_of(box_cells.get((r, c))) or None)
        lines.append(text)
    return lines


def _maybe_ascii(
    lines: list[str] | list[Text], ascii_only: bool,
) -> list[str] | list[Text]:
    """Translate box-drawing glyphs to ASCII when ``ascii_only`` — a
    width-preserving post-process so the layout engine stays Unicode-only
    and the swap can't disturb alignment."""
    if not ascii_only:
        return lines
    out: list = []
    for line in lines:
        out.append(
            _ascii_text(line) if isinstance(line, Text)
            else line.translate(_ASCII_TABLE)
        )
    return out


def _ascii_text(line: Text) -> Text:
    """ASCII-translate a styled line, keeping its spans intact (the glyph
    map is 1:1 in length so every span offset stays valid)."""
    new = line.copy()
    new._text = [seg.translate(_ASCII_TABLE) for seg in new._text]
    return new


def _legend(
    nodes: list[_Node], skip: dict[str, list[str]],
) -> list[str]:
    """Numbered key for compact mode — one ``N — description`` row per
    step, ordered by step number when the refs are numeric. Skip-layer
    dependencies (not drawn as lines) ride along as ``◀ N``."""
    ordered = sorted(
        nodes,
        key=lambda n: (0, int(n.ref)) if n.ref.isdigit() else (1, n.ref),
    )
    rows = ["steps:"]
    for node in ordered:
        row = f"{node.ref} — {node.label}"
        extra = skip.get(node.ref) or []
        if extra:
            row += f"  {_ARROW_BACK} {', '.join(extra)}"
        rows.append(row)
    return rows


def _legend_styled(
    nodes: list[_Node],
    skip: dict[str, list[str]],
    style_of: Callable[[str | None], str],
) -> list[Text]:
    """Styled twin of :func:`_legend`: the leading ref number is tinted
    with the node's colour so a legend entry colour-matches its box; the
    description rides along plain. ``.plain`` equals the plain legend
    row verbatim."""
    ordered = sorted(
        nodes,
        key=lambda n: (0, int(n.ref)) if n.ref.isdigit() else (1, n.ref),
    )
    rows: list[Text] = [Text("steps:")]
    for node in ordered:
        row = Text()
        row.append(node.ref, style=style_of(node.ref) or None)
        suffix = f" — {node.label}"
        extra = skip.get(node.ref) or []
        if extra:
            suffix += f"  {_ARROW_BACK} {', '.join(extra)}"
        row.append(suffix)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Cycle fallback
# ---------------------------------------------------------------------------


def _render_flat(nodes: list[_Node], max_width: int) -> list[str]:
    """Cycle fallback: a single column of boxes with every edge spelt
    out as a ``◀`` annotation (no spine arrows, since a cycle has no
    well-defined order)."""
    contents = [
        _content(n, list(n.deps)) for n in nodes
    ]
    inner = _inner_width(contents, max_width)
    lines: list[str] = []
    for i, content in enumerate(contents):
        if i > 0:
            lines.append("")
        bar = _H * (inner + 2)
        lines.append(f"{_TOP_L}{bar}{_TOP_R}")
        lines.append(f"{_V} {_fit(content, inner).ljust(inner)} {_V}")
        lines.append(f"{_BOT_L}{bar}{_BOT_R}")
    lines.append("(cycle detected — order is approximate)")
    return lines


def _render_flat_styled(
    nodes: list[_Node],
    max_width: int,
    style_of: Callable[[str | None], str],
) -> list[Text]:
    """Styled twin of :func:`_render_flat` for the cycle fallback: one
    tinted box per node (whole box shares the node's colour since a flat
    column never packs two nodes on a row) under a red cycle marker."""
    contents = [_content(n, list(n.deps)) for n in nodes]
    inner = _inner_width(contents, max_width)
    lines: list[Text] = []
    for i, (node, content) in enumerate(zip(nodes, contents)):
        if i > 0:
            lines.append(Text(""))
        style = style_of(node.ref) or None
        bar = _H * (inner + 2)
        lines.append(Text(f"{_TOP_L}{bar}{_TOP_R}", style=style))
        lines.append(
            Text(f"{_V} {_fit(content, inner).ljust(inner)} {_V}", style=style)
        )
        lines.append(Text(f"{_BOT_L}{bar}{_BOT_R}", style=style))
    lines.append(Text("(cycle detected — order is approximate)", style="bold red"))
    return lines
