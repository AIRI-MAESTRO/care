"""Tests for the `Read full` → DAG modal feature.

Covers the new :class:`care.screens.chain_dag.ChainDagModal` plus the
chat-surface wiring that mounts the inline `Read full` button after a
successful generation and evolves the chain from inside the modal.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from textual.app import App
from textual.widgets import Button, Pretty, Static

from care.screens.chain_dag import ChainDagModal
from care.screens.chat import ChatScreen


def _pretty_text(widget: Pretty) -> str:
    """Render a `Pretty` widget's data structure to plain text so tests can
    assert on the values it shows (Pretty prints Python-repr, e.g. keys in
    single quotes)."""
    console = Console(width=240)
    with console.capture() as cap:
        console.print(widget.render())
    return cap.get()

CHAIN = {
    "name": "Demo chain",
    "steps": [
        {
            "number": 1,
            "title": "Analyse query",
            "step_type": "llm",
            "aim": "understand the ask",
            "dependencies": [],
        },
        {
            "number": 2,
            "title": "Draft answer",
            "step_type": "llm",
            "aim": "write a first pass",
            "dependencies": [1],
        },
        {
            "number": 3,
            "title": "Run tool",
            "step_type": "tool",
            "dependencies": [2],
            "config": {"tool": "search"},
        },
    ],
}

# A fork (1 → {2, 3}) so a selection has a genuinely unrelated branch to dim.
FORK = {
    "name": "Fork chain",
    "steps": [
        {"number": 1, "title": "Root", "step_type": "llm", "dependencies": []},
        {"number": 2, "title": "Left", "step_type": "tool",
         "dependencies": [1]},
        {"number": 3, "title": "Right", "step_type": "mcp",
         "dependencies": [1]},
    ],
}


@pytest.fixture(autouse=True)
def _isolate_chat_state_files(monkeypatch, tmp_path):
    """Mirror the chat-suite isolation so ChatScreen never writes to the
    dev's real `~/.local/state/care/` sidecars during these tests."""
    monkeypatch.setenv(
        "CARE_CHAT__TUTORIAL_SIDECAR", str(tmp_path / "tutorial.json"),
    )
    monkeypatch.setenv(
        "CARE_CHAT__SESSION_LOG_DIR", str(tmp_path / "sessions"),
    )
    monkeypatch.setenv("CARE_CHAT__THEME_SIDECAR", str(tmp_path / "theme.txt"))
    monkeypatch.setenv("CARE_CHAT__BRANCHES_DIR", str(tmp_path / "branches"))
    monkeypatch.setenv("CARE_CONTEXT__LTM_DIR", str(tmp_path / "ltm"))
    monkeypatch.setattr("care.memory_ltm.save_from_turn", lambda *a, **k: [])
    monkeypatch.setattr("care.memory_ltm.merge_into_memory", lambda *a, **k: [])


class _ChatHost(App):
    def on_mount(self) -> None:
        self.push_screen(ChatScreen())


def _press(widget, button_id: str) -> None:
    """Invoke a widget's `on_button_pressed` for ``button_id`` directly —
    deterministic where `pilot.click` would depend on hit-testing an
    off-screen scroll child."""
    btn = widget.query_one(f"#{button_id}", Button)
    widget.on_button_pressed(Button.Pressed(btn))


def _render_styles(static) -> set[str]:
    """Collect Rich style strings off a Static's rendered content (styles
    resolve to ansi_* / rgb() forms, so callers match by substring)."""
    rendered = static.render()
    out: set[str] = set()
    if getattr(rendered, "style", None):
        out.add(str(rendered.style))
    for span in getattr(rendered, "spans", []):
        if span.style:
            out.add(str(span.style))
    return out


# ---------------------------------------------------------------------------
# ChainDagModal
# ---------------------------------------------------------------------------


class TestChainDagModal:
    @pytest.mark.asyncio
    async def test_renders_graph_steps_and_default_detail(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN, display_name="Demo"))
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ChainDagModal)
            # One button per DAG node.
            step_btns = [
                b for b in modal.query(Button)
                if (b.id or "").startswith("dagstep-")
            ]
            assert len(step_btns) == 3
            # Chrome is localized through the i18n catalog, not hardcoded.
            assert "Chain DAG" in str(modal.query_one("#dag-title", Static).render())
            assert str(modal.query_one("#dag-save", Button).label) == "Save to library"
            assert str(modal.query_one("#dag-evolve", Button).label) == "Evolve this chain"
            assert str(modal.query_one("#dag-close", Button).label) == "Close"
            # Graph box-art mentions the step titles (full-label mode).
            graph = str(modal.query_one("#dag-graph", Static).render())
            assert "Analyse query" in graph
            # Detail pane defaults to step 1, rendered via the Pretty widget.
            detail = _pretty_text(modal.query_one("#dag-detail-text", Pretty))
            assert "number" in detail and "1" in detail
            assert "understand the ask" in detail

    @pytest.mark.asyncio
    async def test_clicking_step_swaps_detail_to_that_step(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN, display_name="Demo"))
            await pilot.pause()
            modal = app.screen_stack[-1]
            _press(modal, "dagstep-2")
            await pilot.pause()
            detail = _pretty_text(modal.query_one("#dag-detail-text", Pretty))
            assert "search" in detail
            assert "understand the ask" not in detail  # swapped off step 1

    @pytest.mark.asyncio
    async def test_graph_is_tinted_by_step_type(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN, display_name="Demo"))
            await pilot.pause()
            modal = app.screen_stack[-1]
            styles = _render_styles(modal.query_one("#dag-graph", Static))
            # llm steps → cyan, the tool step → magenta.
            assert any("cyan" in s for s in styles)
            assert any("magenta" in s for s in styles)

    @pytest.mark.asyncio
    async def test_selecting_step_highlights_its_box_in_graph(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN, display_name="Demo"))
            await pilot.pause()
            modal = app.screen_stack[-1]
            _press(modal, "dagstep-2")
            await pilot.pause()
            styles = _render_styles(modal.query_one("#dag-graph", Static))
            # The selected node's box is emphasised (bold + underline).
            assert any("underline" in s for s in styles)
            assert modal._selected_idx == 2

    @pytest.mark.asyncio
    async def test_fork_branches_coloured_before_selection(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=FORK))
            await pilot.pause()
            modal = app.screen_stack[-1]
            styles = _render_styles(modal.query_one("#dag-graph", Static))
            # The mcp branch shows its blue tint until a step is selected.
            assert any("blue" in s for s in styles)

    @pytest.mark.asyncio
    async def test_selecting_dims_unrelated_branch(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=FORK))
            await pilot.pause()
            modal = app.screen_stack[-1]
            _press(modal, "dagstep-1")  # select "Left" (ref 2)
            await pilot.pause()
            styles = _render_styles(modal.query_one("#dag-graph", Static))
            assert any("underline" in s for s in styles)  # selected box
            # The unrelated mcp branch (node 3) is dimmed → its blue is gone.
            assert not any("blue" in s for s in styles)

    @pytest.mark.asyncio
    async def test_graph_geometry_populated_for_clicks(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            modal = app.screen_stack[-1]
            # Every step's box contributes cells to the click hit-test map.
            assert set(modal._cell_to_ref.values()) >= {"1", "2", "3"}

    @pytest.mark.asyncio
    async def test_select_by_ref_drives_selection(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            modal = app.screen_stack[-1]
            modal._select_by_ref("3")  # the box-click path resolves here
            await pilot.pause()
            assert modal._selected_idx == 2
            detail = _pretty_text(modal.query_one("#dag-detail-text", Pretty))
            assert "search" in detail  # step 3's tool

    @pytest.mark.asyncio
    async def test_click_on_a_box_resolves_to_its_step(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            await pilot.pause()
            modal = app.screen_stack[-1]
            # Pick a cell that belongs to node "2" and translate it back to
            # screen coordinates the way a real click would arrive.
            target = next(
                rc for rc, ref in modal._cell_to_ref.items() if ref == "2"
            )
            row, col = target
            region = modal.query_one("#dag-graph", Static).region
            from textual.containers import VerticalScroll

            off = modal.query_one("#dag-graph-scroll", VerticalScroll).scroll_offset
            sx = region.x + col - int(off.x)
            sy = region.y + row - int(off.y)
            assert modal._ref_at_screen(sx, sy) == "2"

    @pytest.mark.asyncio
    async def test_off_box_click_is_a_noop(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            modal = app.screen_stack[-1]
            # Far outside the graph region → no hit, no selection change.
            assert modal._ref_at_screen(9999, 9999) is None

    @pytest.mark.asyncio
    async def test_layout_toggle_switches_to_left_to_right(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            modal = app.screen_stack[-1]
            assert modal._layout == "tb"
            modal.action_toggle_layout()
            await pilot.pause()
            assert modal._layout == "lr"
            rendered = str(modal.query_one("#dag-graph", Static).render())
            assert "▶" in rendered  # right-flowing arrows in LR
            modal.action_toggle_layout()
            await pilot.pause()
            assert modal._layout == "tb"

    @pytest.mark.asyncio
    async def test_copy_mermaid_writes_flowchart_to_clipboard(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            "care.runtime.clipboard.copy_text",
            lambda app, text: captured.update(text=text) or True,
        )
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            modal = app.screen_stack[-1]
            modal.action_copy_mermaid()
            assert captured.get("text", "").startswith("flowchart TD")
            assert "-->" in captured["text"]

    @pytest.mark.asyncio
    async def test_keyboard_nav_walks_dependencies(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))  # linear 1→2→3
            await pilot.pause()
            modal = app.screen_stack[-1]
            modal._select_index(0)
            modal.action_nav_child()    # 1 → 2
            assert modal._selected_idx == 1
            modal.action_nav_child()    # 2 → 3
            assert modal._selected_idx == 2
            modal.action_nav_parent()   # 3 → 2
            assert modal._selected_idx == 1

    @pytest.mark.asyncio
    async def test_keyboard_nav_next_prev_clamps(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict=CHAIN))
            await pilot.pause()
            modal = app.screen_stack[-1]
            modal._select_index(0)
            modal.action_nav_prev()     # clamp at 0
            assert modal._selected_idx == 0
            modal.action_nav_next()
            assert modal._selected_idx == 1

    @pytest.mark.asyncio
    async def test_edit_step_dismisses_with_edit_and_target(self):
        app = App()
        async with app.run_test() as pilot:
            results: list = []
            app.push_screen(ChainDagModal(chain_dict=CHAIN), results.append)
            await pilot.pause()
            modal = app.screen_stack[-1]
            _press(modal, "dagstep-2")   # select the third step (number 3)
            await pilot.pause()
            _press(modal, "dag-edit")
            await pilot.pause()
            assert results == ["edit"]
            assert modal.edit_step_number == 3

    @pytest.mark.asyncio
    async def test_evolve_button_dismisses_with_evolve(self):
        app = App()
        async with app.run_test() as pilot:
            results: list = []
            app.push_screen(
                ChainDagModal(chain_dict=CHAIN, display_name="Demo"),
                results.append,
            )
            await pilot.pause()
            modal = app.screen_stack[-1]
            _press(modal, "dag-evolve")
            await pilot.pause()
            assert results == ["evolve"]

    @pytest.mark.asyncio
    async def test_save_button_triggers_handler_without_dismissing(self):
        # Save must NOT close the modal — it locks the button and hands
        # off to the caller's handler so the result can be reflected in
        # place.
        app = App()
        async with app.run_test() as pilot:
            results: list = []
            calls = {"n": 0}
            modal = ChainDagModal(chain_dict=CHAIN, display_name="Demo")
            modal.save_handler = lambda: calls.__setitem__("n", calls["n"] + 1)
            app.push_screen(modal, results.append)
            await pilot.pause()
            _press(modal, "dag-save")
            await pilot.pause()
            assert results == []  # still open
            assert calls["n"] == 1
            save_btn = modal.query_one("#dag-save", Button)
            assert save_btn.disabled is True
            assert str(save_btn.label) == "Saving…"

    @pytest.mark.asyncio
    async def test_mark_saved_locks_button_with_saved_label(self):
        app = App()
        async with app.run_test() as pilot:
            modal = ChainDagModal(chain_dict=CHAIN, display_name="Demo")
            app.push_screen(modal)
            await pilot.pause()
            modal.mark_saved()
            await pilot.pause()
            save_btn = modal.query_one("#dag-save", Button)
            assert save_btn.disabled is True
            assert str(save_btn.label) == "Saved to library"

    @pytest.mark.asyncio
    async def test_already_saved_chain_opens_with_locked_button(self):
        # A chain that already has an id (Production, or saved earlier
        # from a prior modal) opens with the Save button locked — even
        # on a brand-new modal instance.
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(
                ChainDagModal(
                    chain_dict=CHAIN, display_name="Demo", chain_id="chain-7",
                ),
            )
            await pilot.pause()
            modal = app.screen_stack[-1]
            save_btn = modal.query_one("#dag-save", Button)
            assert save_btn.disabled is True
            assert str(save_btn.label) == "Saved to library"

    @pytest.mark.asyncio
    async def test_close_button_dismisses_with_none(self):
        app = App()
        async with app.run_test() as pilot:
            results: list = []
            app.push_screen(
                ChainDagModal(chain_dict=CHAIN, display_name="Demo"),
                results.append,
            )
            await pilot.pause()
            modal = app.screen_stack[-1]
            _press(modal, "dag-close")
            await pilot.pause()
            assert results == [None]

    @pytest.mark.asyncio
    async def test_empty_chain_opens_without_crashing(self):
        app = App()
        async with app.run_test() as pilot:
            app.push_screen(ChainDagModal(chain_dict={"steps": []}))
            await pilot.pause()
            modal = app.screen_stack[-1]
            step_btns = [
                b for b in modal.query(Button)
                if (b.id or "").startswith("dagstep-")
            ]
            assert step_btns == []


# ---------------------------------------------------------------------------
# Chat wiring — inline `Read full` button
# ---------------------------------------------------------------------------


class TestReadFullButton:
    @pytest.mark.asyncio
    async def test_post_chain_actions_mounts_button_and_stashes_payload(self):
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            screen._post_chain_actions(CHAIN, display_name="My task")
            await pilot.pause()
            btns = [
                b for b in screen.query(Button)
                if (b.id or "").startswith("chat-readfull-btn-")
            ]
            assert len(btns) == 1
            bid = btns[0].id
            assert bid in screen._chain_action_payloads
            payload = screen._chain_action_payloads[bid]
            assert payload["display_name"] == "My task"
            assert payload["chain_dict"] is CHAIN
            assert payload["chain_id"] is None
            # The button rides in a captioned row so it reads as part of
            # the trace, not a floating control.
            label = screen.query_one(".chat-readfull-label", Static)
            assert "generated successfully" in str(label.render())
            assert btns[0] in screen.query(".chat-readfull-row Button")

    @pytest.mark.asyncio
    async def test_button_label_has_a_visible_row(self):
        # `Button` paints an unremovable `tall` border that claims two
        # rows; a `height: 1` rule would starve the single label row and
        # render a blank button. Lock the label in with a real content
        # row + enough width for the text.
        app = _ChatHost()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            screen._post_chain_actions(CHAIN, display_name="My task")
            await pilot.pause()
            await pilot.pause()
            btn = next(
                b for b in screen.query(Button)
                if (b.id or "").startswith("chat-readfull-btn-")
            )
            assert str(btn.label) == "View chain"
            assert btn.size.height >= 1
            assert btn.size.width >= len("View chain")

    @pytest.mark.asyncio
    async def test_empty_chain_posts_no_button(self):
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            screen._post_chain_actions({}, display_name="X")
            await pilot.pause()
            btns = [
                b for b in screen.query(Button)
                if (b.id or "").startswith("chat-readfull-btn-")
            ]
            assert btns == []

    @pytest.mark.asyncio
    async def test_pressing_read_full_opens_dag_modal(self):
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            screen._post_chain_actions(CHAIN, display_name="My task")
            await pilot.pause()
            bid = next(
                b.id for b in screen.query(Button)
                if (b.id or "").startswith("chat-readfull-btn-")
            )
            screen.on_button_pressed(
                Button.Pressed(screen.query_one(f"#{bid}", Button)),
            )
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ChainDagModal)


# ---------------------------------------------------------------------------
# Evolve hand-off
# ---------------------------------------------------------------------------


class TestEvolveFromDag:
    @pytest.mark.asyncio
    async def test_known_chain_id_opens_evolution_setup(self, monkeypatch):
        """A saved chain's *Evolve this chain* opens the shared evolution
        setup modal (dataset + budget + rubric) pre-bound to its id —
        NOT a direct fire-and-forget kickoff."""
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            captured: dict = {}

            monkeypatch.setattr(
                screen, "_open_evolution_setup",
                lambda chain_id="": captured.update(chain_id=chain_id),
            )
            screen._evolve_from_dag(
                {
                    "chain_dict": CHAIN,
                    "display_name": "Saved one",
                    "chain_id": "chain-42",
                },
            )
            await pilot.pause()
            assert captured["chain_id"] == "chain-42"

    @pytest.mark.asyncio
    async def test_save_to_library_persists_and_backfills_payload(
        self, monkeypatch,
    ):
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]

            seen: dict = {}

            class _Mem:
                def save_chain(self, chain, **kw):
                    assert chain is CHAIN
                    assert kw["tags"] == ["source:chat-dag-save"]
                    seen.update(kw)
                    return "lib-id-3"

            monkeypatch.setattr(app, "memory", _Mem(), raising=False)
            payload = {
                "chain_dict": CHAIN, "display_name": "Fresh", "chain_id": None,
                # Original generation request — must be stamped as the
                # saved chain's query so a later re-run pre-fills the task.
                "task": "tell weather and news",
            }

            class _FakeModal:
                saved = False
                failed = False

                def mark_saved(self):
                    self.saved = True

                def mark_save_failed(self):
                    self.failed = True

            fake_modal = _FakeModal()
            await screen._save_chain_to_library(
                payload=payload,
                chain_dict=CHAIN,
                display_name="Fresh",
                modal=fake_modal,
            )
            # The new id is backfilled so a later evolve reuses it, and
            # the modal's Save button is locked into its saved state.
            assert payload["chain_id"] == "lib-id-3"
            assert fake_modal.saved is True
            assert fake_modal.failed is False
            # The generation request rode through as the saved chain's query.
            assert seen.get("query") == "tell weather and news"

    @pytest.mark.asyncio
    async def test_save_flips_artifact_and_updates_unsaved_pill(
        self, monkeypatch,
    ):
        from care.widgets.header import CareHeader

        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]

            class _Mem:
                def save_chain(self, chain, **kw):
                    return "lib-id-9"

            monkeypatch.setattr(app, "memory", _Mem(), raising=False)
            # Seed a session artifact like a generation would, then
            # confirm the header pill shows it as unsaved.
            art = screen.artifact_store.append_chain(
                chain=CHAIN, title="c", summary="",
            )
            await pilot.pause()
            header = screen.query_one(CareHeader)
            assert header.artifact_pill == "Artifacts (1 unsaved)"
            # Save through the DAG-modal worker carrying the artifact id.
            payload = {
                "chain_dict": CHAIN,
                "display_name": "x",
                "chain_id": None,
                "artifact_id": art.id,
            }
            await screen._save_chain_to_library(
                payload=payload, chain_dict=CHAIN, display_name="x",
            )
            await pilot.pause()
            # Artifact flipped to saved → pill no longer reports unsaved.
            assert screen.artifact_store.unsaved(kind="chain") == []
            assert header.artifact_pill == "Artifacts (1 saved)"

    @pytest.mark.asyncio
    async def test_save_to_library_already_saved_is_noop(self, monkeypatch):
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]
            called = {"n": 0}

            def boom(*a, **k):
                called["n"] += 1

            # An already-saved chain must NOT spawn a save worker.
            monkeypatch.setattr(screen, "run_worker", boom)
            before = len(screen._lines)
            screen._save_to_library(
                {
                    "chain_dict": CHAIN,
                    "display_name": "x",
                    "chain_id": "chain-99",
                },
            )
            await pilot.pause()
            assert called["n"] == 0
            new = screen._lines[before:]
            assert any("chain-99" in ln.text for ln in new)

    @pytest.mark.asyncio
    async def test_unsaved_chain_saves_then_opens_evolution_setup(
        self, monkeypatch,
    ):
        """An unsaved chain is persisted first, then the evolution setup
        modal opens bound to the new id (so the user picks a dataset)."""
        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen_stack[-1]

            class _Mem:
                def save_chain(self, chain, **kw):
                    assert chain is CHAIN
                    return "saved-id-7"

            monkeypatch.setattr(app, "memory", _Mem(), raising=False)
            captured: dict = {}

            monkeypatch.setattr(
                screen, "_open_evolution_setup",
                lambda chain_id="": captured.update(chain_id=chain_id),
            )
            await screen._save_then_open_evolution_setup(
                chain_dict=CHAIN, display_name="Fresh",
            )
            assert captured["chain_id"] == "saved-id-7"
