"""ChainDagModal — full-DAG popup for a freshly-generated chain.

Opened from the **Read full** inline button the chat surface mounts
after a successful generation (see
:meth:`care.screens.chat.ChatScreen._post_chain_actions`).

The modal shows three things at once:

* the chain's **box-and-arrow graph** (the same renderer the inline
  stage trail uses — :func:`care.runtime.dag_view.render_dag_boxes`),
* a **clickable step list** — one button per DAG node, and
* a **CARL-format detail pane** that prints the selected step exactly
  as it appears in the serialised CARL chain (pretty-printed JSON).

Footer buttons hand control back to the chat screen via the dismiss
value: **Save to library** dismisses with ``"save"`` (the caller saves
the chain to Memory) and **Evolve this chain** dismisses with
``"evolve"`` (the caller saves if needed, then submits to the Platform).
Closing — via the **Close** button or ``Escape`` — dismisses with
``None``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Pretty, Static

from care.runtime.i18n import t


def _steps_of(chain_dict: Any) -> list[dict]:
    """Project a chain payload onto its ordered step dicts. Accepts a
    ``{"steps": [...]}`` mapping or a bare list of step dicts."""
    if isinstance(chain_dict, dict):
        steps = chain_dict.get("steps")
        if isinstance(steps, list):
            return [s for s in steps if isinstance(s, dict)]
        return []
    if isinstance(chain_dict, (list, tuple)):
        return [s for s in chain_dict if isinstance(s, dict)]
    return []


class ChainDagModal(ModalScreen["str | None"]):
    """Full-DAG inspector with clickable steps + an evolve hand-off.

    Args:
        chain_dict: The CARL chain payload (``{"steps": [...], …}``).
        display_name: Human title for the modal header.
        chain_id: Memory entity id when the chain is already saved
            (Production). ``None`` in Ad-Hoc — the caller saves first
            before evolving.
    """

    DEFAULT_CSS = """
    ChainDagModal {
        align: center middle;
    }
    ChainDagModal #dag-box {
        width: 90%;
        max-width: 140;
        height: 85%;
        max-height: 44;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ChainDagModal #dag-title {
        text-style: bold;
        padding-bottom: 1;
    }
    ChainDagModal #dag-graph-scroll {
        height: auto;
        max-height: 12;
        border: solid $primary 30%;
        padding: 0 1;
        margin-bottom: 1;
    }
    ChainDagModal #dag-graph {
        width: auto;
    }
    ChainDagModal #dag-body {
        height: 1fr;
    }
    ChainDagModal #dag-steps {
        width: 38%;
        border: solid $primary 30%;
        padding: 0 1;
    }
    ChainDagModal #dag-steps Button {
        width: 100%;
        margin-bottom: 1;
    }
    ChainDagModal #dag-detail {
        width: 1fr;
        margin-left: 1;
        border: solid $primary 30%;
        padding: 0 1;
    }
    ChainDagModal #dag-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    ChainDagModal #dag-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close", show=True),
        Binding("l", "toggle_layout", "Layout ↕/↔", show=True),
        Binding("y", "copy_mermaid", "Copy Mermaid", show=True),
        Binding("k", "nav_parent", "↑ dep", show=False),
        Binding("j", "nav_child", "↓ dependent", show=False),
        Binding("p", "nav_prev", "prev step", show=False),
        Binding("n", "nav_next", "next step", show=False),
    ]

    def __init__(
        self,
        *,
        chain_dict: Any,
        display_name: str = "",
        chain_id: str | None = None,
    ) -> None:
        super().__init__()
        self._chain_dict = chain_dict
        self._display_name = display_name or t("chat.dag.defaultName")
        self._chain_id = chain_id
        self._steps = _steps_of(chain_dict)
        # Index of the step currently mirrored in the detail pane +
        # highlighted in the graph. Clicking a step button moves it.
        self._selected_idx = 0
        # Graph orientation ("tb"/"lr") + whether the lineage dim is on.
        # ``_layout`` is seeded from config in compose() and flipped live
        # by the `l` binding; ``_dim`` turns on once a step is picked.
        self._layout = "tb"
        self._dim = False
        # Cell→node-ref geometry of the last graph render, so a click on a
        # box maps back to its step. Refreshed by `_graph_renderable`.
        self._cell_to_ref: dict[tuple[int, int], str] = {}
        # node-ref → step index, so a box click and the step list resolve
        # to the same detail/highlight target.
        self._ref_to_index = {
            self._ref_for_step(step, idx): idx
            for idx, step in enumerate(self._steps)
        }
        # A chain that already carries an id (Production, or saved
        # earlier from this modal) opens with the Save button locked.
        self._saved = chain_id is not None
        # Set by the caller (ChatScreen) so the Save button can trigger
        # the actual Memory write without the modal knowing how.
        self.save_handler: Callable[[], None] | None = None
        # Step number the user asked to edit (set when dismissing with
        # "edit"); the chat caller reads it to seed a targeted /revise.
        self.edit_step_number: int | None = None

    def compose(self) -> ComposeResult:
        self._layout = self._configured_layout()
        with Vertical(id="dag-box"):
            yield Static(
                t("chat.dag.title", name=self._display_name),
                id="dag-title",
                markup=False,
            )
            with VerticalScroll(id="dag-graph-scroll"):
                yield Static(
                    self._graph_renderable(
                        0 if self._steps else None,
                    ),
                    id="dag-graph",
                    markup=False,
                )
            with Horizontal(id="dag-body"):
                with VerticalScroll(id="dag-steps"):
                    if self._steps:
                        for idx, step in enumerate(self._steps):
                            yield Button(
                                Text(self._step_button_label(step, idx)),
                                id=f"dagstep-{idx}",
                            )
                    else:
                        yield Static(t("chat.dag.noSteps"), markup=False)
                with VerticalScroll(id="dag-detail"):
                    # The selected step is shown via the `Pretty` widget —
                    # the chain dict rendered as a syntax-coloured, indented
                    # data structure rather than a flat JSON dump.
                    yield Pretty(
                        self._detail_object(0),
                        id="dag-detail-text",
                    )
            with Horizontal(id="dag-buttons"):
                yield Button(
                    t("chat.dag.saved") if self._saved
                    else t("chat.dag.save"),
                    id="dag-save",
                    variant="primary",
                    disabled=self._saved,
                )
                yield Button(
                    t("chat.dag.evolve"), id="dag-evolve", variant="success",
                )
                if self._steps:
                    yield Button(t("chat.dag.editStep"), id="dag-edit")
                yield Button(t("chat.dag.close"), id="dag-close")

    def on_mount(self) -> None:
        # Land focus on the first step so keyboard users can tab/arrow
        # through the nodes immediately; the detail pane already shows
        # step 1 from compose().
        if not self._steps:
            return
        try:
            self.query_one("#dagstep-0", Button).focus()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _graph_renderable(self, selected_idx: int | None, *, dim: bool = False):
        """The colour-tinted box-and-arrow graph for the whole chain:
        boxes tinted by step type, the selected step bold-underlined, and
        — once the user picks a step (``dim``) — everything outside that
        step's data-flow lineage muted so the path stands out. Labels wrap
        to two lines (the modal has the room) instead of truncating. Falls
        back to a terse line when the renderer can't read the payload so
        the modal still opens."""
        highlight = None
        if selected_idx is not None and 0 <= selected_idx < len(self._steps):
            highlight = self._ref_for_step(
                self._steps[selected_idx], selected_idx,
            )
        self._cell_to_ref = {}
        try:
            from care.runtime.dag_view import (
                dag_display_opts,
                render_dag_styled,
            )

            lines = render_dag_styled(
                self._chain_dict,
                highlight_ref=highlight,
                dim_unrelated=dim and highlight is not None,
                max_width=44,
                max_graph_width=120,
                max_lines=2,
                layout=self._layout,
                geometry=self._cell_to_ref,
                **dag_display_opts(getattr(self.app, "config", None)),
            )
        except Exception:
            lines = []
        if not lines:
            return Text(t("chat.dag.stepCount", count=len(self._steps)))
        return Text("\n").join(lines)

    def _configured_layout(self) -> str:
        """Initial graph orientation from config (``CARE_DEFAULTS__DAG_LAYOUT``),
        defaulting to top-down. Defensive: missing config → ``"tb"``."""
        try:
            return "lr" if str(self.app.config.defaults.dag_layout) == "lr" else "tb"
        except Exception:
            return "tb"

    def action_toggle_layout(self) -> None:
        """`l` flips the graph between top-down and left-to-right in place,
        preserving the current selection + dim state."""
        self._layout = "lr" if self._layout == "tb" else "tb"
        try:
            self.query_one("#dag-graph", Static).update(
                self._graph_renderable(self._selected_idx, dim=self._dim),
            )
        except Exception:
            pass

    def action_copy_mermaid(self) -> None:
        """`y` copies the chain as Mermaid ``flowchart`` source for pasting
        into Markdown / a PR / docs. Direction follows the current
        orientation toggle."""
        from care.runtime.clipboard import copy_text
        from care.runtime.dag_view import render_dag_mermaid

        src = render_dag_mermaid(self._chain_dict, layout=self._layout)
        if not src:
            return
        ok = copy_text(self.app, src)
        try:
            self.app.notify(
                t("chat.dag.mermaidCopied") if ok else t("chat.dag.copyFailed"),
                severity="information" if ok else "warning",
            )
        except Exception:
            pass

    # --- keyboard topology navigation ---------------------------------

    def action_nav_parent(self) -> None:
        """`k` — move the selection up to a dependency (parent)."""
        self._nav_to(self._parent_index(self._selected_idx))

    def action_nav_child(self) -> None:
        """`j` — move the selection down to a dependent (child)."""
        self._nav_to(self._child_index(self._selected_idx))

    def action_nav_prev(self) -> None:
        """`p` — previous step by position."""
        self._nav_to(self._selected_idx - 1)

    def action_nav_next(self) -> None:
        """`n` — next step by position."""
        self._nav_to(self._selected_idx + 1)

    def _nav_to(self, idx: int | None) -> None:
        if idx is None or not (0 <= idx < len(self._steps)):
            return
        self._select_index(idx)
        try:
            self.query_one(f"#dagstep-{idx}", Button).focus()
        except Exception:
            pass

    def _parent_index(self, idx: int) -> int | None:
        """Index of the first dependency of step ``idx`` (or ``None``)."""
        if not (0 <= idx < len(self._steps)):
            return None
        for dep in self._step_deps(self._steps[idx]):
            j = self._ref_to_index.get(str(dep))
            if j is not None:
                return j
        return None

    def _child_index(self, idx: int) -> int | None:
        """Index of the first step that depends on step ``idx`` (or
        ``None``)."""
        if not (0 <= idx < len(self._steps)):
            return None
        ref = self._ref_for_step(self._steps[idx], idx)
        for j, step in enumerate(self._steps):
            if any(str(d) == ref for d in self._step_deps(step)):
                return j
        return None

    @staticmethod
    def _step_deps(step: dict) -> list:
        deps = (
            step.get("dependencies")
            or step.get("deps")
            or step.get("depends_on")
            or []
        )
        if isinstance(deps, (str, int)):
            return [deps]
        return list(deps) if isinstance(deps, (list, tuple)) else []

    @staticmethod
    def _ref_for_step(step: dict, idx: int) -> str:
        """The node ref the DAG renderer assigns this step — mirrors
        ``care.runtime.dag_view._node_ref`` so a clicked step maps to the
        right box in the graph."""
        for key in ("number", "id", "step_id", "index"):
            value = step.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
        return str(idx + 1)

    def _step_button_label(self, step: dict, idx: int) -> str:
        from care.screens.inspection import _step_label

        number = step.get("number")
        prefix = f"{number}. " if isinstance(number, int) else f"{idx + 1}. "
        return f"{prefix}{_step_label(step, idx)}"

    def _detail_object(self, idx: int):
        """The selected step as a live object for the `Pretty` widget — the
        step dict exactly as it rides inside the chain (Pretty renders it as
        an indented, coloured data structure). Falls back to a placeholder
        string when there's no step to show."""
        if not (0 <= idx < len(self._steps)):
            return t("chat.dag.nothingToShow")
        return self._steps[idx]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "dag-save":
            self._begin_save()
            return
        if bid == "dag-evolve":
            self.dismiss("evolve")
            return
        if bid == "dag-edit":
            self._begin_edit()
            return
        if bid == "dag-close":
            self.dismiss(None)
            return
        if bid.startswith("dagstep-"):
            try:
                idx = int(bid.rsplit("-", 1)[1])
            except ValueError:
                return
            self._select_index(idx)

    def _select_index(self, idx: int) -> None:
        """Select step ``idx``: mirror it into the detail pane and re-tint
        the graph so its box is highlighted and its lineage stands out.
        Shared by the step list and graph-box clicks."""
        if not (0 <= idx < len(self._steps)):
            return
        self._selected_idx = idx
        self._dim = True
        try:
            self.query_one("#dag-detail-text", Pretty).update(
                self._detail_object(idx),
            )
        except Exception:
            pass
        try:
            self.query_one("#dag-graph", Static).update(
                self._graph_renderable(idx, dim=True),
            )
        except Exception:
            pass

    def _select_by_ref(self, ref: str) -> None:
        """Select the step a node ref belongs to (graph-box click path),
        moving keyboard focus onto its step button too."""
        idx = self._ref_to_index.get(ref)
        if idx is None:
            return
        self._select_index(idx)
        try:
            self.query_one(f"#dagstep-{idx}", Button).focus()
        except Exception:
            pass

    def on_click(self, event: events.Click) -> None:
        """Click a box in the graph to select its step — the picture and
        the list become two ways to drive the same selection."""
        ref = self._ref_at_screen(event.screen_x, event.screen_y)
        if ref is not None:
            self._select_by_ref(ref)

    def _ref_at_screen(self, screen_x: int, screen_y: int) -> str | None:
        """Map a screen coordinate to the node ref of the graph box under
        it, via the last render's geometry (accounting for scroll).
        Returns ``None`` off any box. Best-effort: any layout-query
        failure resolves to a miss rather than raising."""
        if not self._cell_to_ref:
            return None
        try:
            region = self.query_one("#dag-graph", Static).region
        except Exception:
            return None
        if region.area == 0 or not region.contains(screen_x, screen_y):
            return None
        off_x = off_y = 0
        try:
            offset = self.query_one(
                "#dag-graph-scroll", VerticalScroll,
            ).scroll_offset
            off_x, off_y = int(offset.x), int(offset.y)
        except Exception:
            pass
        row = screen_y - region.y + off_y
        col = screen_x - region.x + off_x
        return self._cell_to_ref.get((row, col))

    def _begin_edit(self) -> None:
        """Hand the selected step off to ``/revise``: stash its number and
        dismiss with ``"edit"`` so the chat caller can seed a targeted
        revision prompt."""
        if not self._steps:
            return
        step = self._steps[self._selected_idx]
        number = step.get("number")
        self.edit_step_number = (
            number if isinstance(number, int) else self._selected_idx + 1
        )
        self.dismiss("edit")

    def _begin_save(self) -> None:
        """Kick off the save without closing the modal: lock the button
        into a `Saving…` state and hand off to the caller's handler. The
        handler later calls :meth:`mark_saved` / :meth:`mark_save_failed`
        when the async write resolves."""
        if self._saved:
            return
        self._set_save_button(label=t("chat.dag.saving"), disabled=True)
        if self.save_handler is not None:
            self.save_handler()

    def mark_saved(self) -> None:
        """Lock the Save button into its saved state. Safe to call after
        the modal closed — the widget lookup just no-ops."""
        self._saved = True
        self._set_save_button(label=t("chat.dag.saved"), disabled=True)

    def mark_save_failed(self) -> None:
        """Re-enable the Save button so the user can retry after a failed
        write (the error itself is surfaced in the chat transcript)."""
        self._saved = False
        self._set_save_button(label=t("chat.dag.save"), disabled=False)

    def _set_save_button(self, *, label: str, disabled: bool) -> None:
        try:
            btn = self.query_one("#dag-save", Button)
        except Exception:
            return
        btn.label = label
        btn.disabled = disabled

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ChainDagModal"]
