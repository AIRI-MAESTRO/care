"""Pilot tests for `ArtifactsScreen` (TODO §3 P0).

Drives the screen against a real `SessionArtifactStore` so the
event-listener loop, table refresh, and bindings stay in
lockstep with the production wiring.

Coverage:

* Mount renders the table + populates rows from the store.
* Empty-state placeholder shows when the store has no entries
  + hides once an artifact lands.
* Store events (append / mark_saved / forget) re-paint the
  table without a manual refresh.
* Bindings — `s`, `d`, `c`, `Esc`, `Enter` — fire the right
  side effects (action log + store mutation + toast surface).
* The `/artifacts` slash command and the
  `CareApp.action_palette_open_artifacts` palette opener both
  push `ArtifactsScreen` against the active chat's store.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Pretty, Static, TabbedContent

from care.runtime.session_artifacts import SessionArtifactStore
from care.screens.artifacts import ArtifactsScreen


def _pretty_text(widget: Pretty) -> str:
    """Render a `Pretty` widget's data to plain text for assertions."""
    console = Console(width=240)
    with console.capture() as cap:
        console.print(widget.render())
    return cap.get()


# ---------------------------------------------------------------------------
# Host scaffolding
# ---------------------------------------------------------------------------


class _Host(App):
    """Minimal host that mounts an ArtifactsScreen on boot
    with a real SessionArtifactStore. Tests pre-populate the
    store before pushing if they want non-empty rows."""

    def __init__(self, store: SessionArtifactStore | None = None) -> None:
        super().__init__()
        self.store = store or SessionArtifactStore()
        self.toasts: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(ArtifactsScreen(self.store))

    def push_toast(self, message: str, *, severity: str = "info", ttl=None):  # type: ignore[override]
        self.toasts.append((message, severity))


def _artifacts_screen(app: _Host) -> ArtifactsScreen:
    for screen in app.screen_stack:
        if isinstance(screen, ArtifactsScreen):
            return screen
    raise AssertionError("ArtifactsScreen not on the stack")


def _activate_tab(screen: ArtifactsScreen, kind: str) -> None:
    """Switch the artifact browser to the tab backing ``kind`` so
    `current_artifact` resolves against that tab's table."""
    tabs = screen.query_one("#artifacts-tabs", TabbedContent)
    tabs.active = f"artifacts-tab-{kind}"


# ---------------------------------------------------------------------------
# Mount + render
# ---------------------------------------------------------------------------


class TestMount:
    @pytest.mark.asyncio
    async def test_mount_with_empty_store_shows_placeholder(self):
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            empty = screen.query_one("#artifacts-empty", Static)
            # Visible because the store has zero rows.
            assert empty.display is True
            # _rows is the canonical "what the table shows"
            # state and matches the empty store.
            assert screen._rows == []

    @pytest.mark.asyncio
    async def test_mount_populates_rows_from_store(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={"steps": []}, title="weather", summary="3-step")
        b = store.append_tool_output(tool="grep", output="found")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            # Chain + tool output now live in separate tabs.
            chain_table = screen.query_one(
                "#artifacts-table-chain", DataTable,
            )
            tool_table = screen.query_one(
                "#artifacts-table-tool_output", DataTable,
            )
            assert chain_table.row_count == 1
            assert tool_table.row_count == 1
            assert [r.value for r in chain_table.rows.keys()] == [a.id]
            assert [r.value for r in tool_table.rows.keys()] == [b.id]


# ---------------------------------------------------------------------------
# Tabs (chains / tool_output / stage_payload)
# ---------------------------------------------------------------------------


class TestTabs:
    @pytest.mark.asyncio
    async def test_tab_order_is_chains_tool_stage(self):
        """The browser exposes exactly three tabs in the order
        chains → tool output → stage payloads."""
        from textual.widgets import TabPane

        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            tabs = screen.query_one("#artifacts-tabs", TabbedContent)
            pane_ids = [
                p.id for p in tabs.query(TabPane)
            ]
            assert pane_ids == [
                "artifacts-tab-chain",
                "artifacts-tab-tool_output",
                "artifacts-tab-stage_payload",
            ]

    @pytest.mark.asyncio
    async def test_each_kind_lands_in_its_own_tab(self):
        """A chain, a tool output, and a stage payload route to
        their respective per-kind tables — none bleed across."""
        store = SessionArtifactStore()
        chain = store.append_chain(chain={}, title="c", summary="")
        tool = store.append_tool_output(tool="grep", output="x")
        stage = store.append_stage_payload(
            stage="step_describing", payload={"ok": True},
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            chain_table = screen.query_one(
                "#artifacts-table-chain", DataTable,
            )
            tool_table = screen.query_one(
                "#artifacts-table-tool_output", DataTable,
            )
            stage_table = screen.query_one(
                "#artifacts-table-stage_payload", DataTable,
            )
            assert [r.value for r in chain_table.rows.keys()] == [
                chain.id,
            ]
            assert [r.value for r in tool_table.rows.keys()] == [
                tool.id,
            ]
            assert [r.value for r in stage_table.rows.keys()] == [
                stage.id,
            ]

    @pytest.mark.asyncio
    async def test_current_artifact_follows_active_tab(self):
        """`current_artifact` resolves against the active tab's
        cursor, so switching tabs changes what actions target."""
        store = SessionArtifactStore()
        chain = store.append_chain(chain={}, title="c", summary="")
        tool = store.append_tool_output(tool="grep", output="x")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            # Default tab is chains.
            assert screen.current_artifact is not None
            assert screen.current_artifact.id == chain.id
            _activate_tab(screen, "tool_output")
            await pilot.pause()
            assert screen.current_artifact is not None
            assert screen.current_artifact.id == tool.id


# ---------------------------------------------------------------------------
# Store event → repaint
# ---------------------------------------------------------------------------


class TestStoreEvents:
    @pytest.mark.asyncio
    async def test_append_after_mount_repaints(self):
        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            table = screen.query_one("#artifacts-table-chain", DataTable)
            assert table.row_count == 0
            app.store.append_chain(chain={}, title="late-arrival", summary="")
            await pilot.pause()
            assert table.row_count == 1
            # Empty-state widget hides once a row lands.
            empty = screen.query_one("#artifacts-empty", Static)
            assert empty.display is False

    @pytest.mark.asyncio
    async def test_mark_saved_updates_row_badge(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="t", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            table = screen.query_one("#artifacts-table-chain", DataTable)
            saved_cell = table.get_cell(a.id, "Saved")
            assert saved_cell == ""
            store.mark_saved(a.id, memory_entity_id="ENT-1")
            await pilot.pause()
            saved_cell = table.get_cell(a.id, "Saved")
            assert "★" in saved_cell
            assert "ENT-1" in saved_cell


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


class TestBindings:
    @pytest.mark.asyncio
    async def test_escape_pops_screen(self):
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            depth_before = len(app.screen_stack)
            screen = _artifacts_screen(app)
            screen.action_back()
            await pilot.pause()
            assert len(app.screen_stack) == depth_before - 1

    @pytest.mark.asyncio
    async def test_delete_from_session_drops_artifact(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="t", summary="")
        store.append_chain(chain={}, title="t2", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            table = screen.query_one("#artifacts-table-chain", DataTable)
            assert table.row_count == 2
            # Cursor is on the top row (newest = t2). Move it
            # to `a` to test deleting the older row, then
            # delete.
            from textual.coordinate import Coordinate

            target_row = next(
                i for i, art in enumerate(screen._rows_by_kind["chain"])
                if art.id == a.id
            )
            table.cursor_coordinate = Coordinate(target_row, 0)
            await pilot.pause()
            screen.action_delete_from_session()
            await pilot.pause()
            assert len(store) == 1
            assert a.id not in store
            assert ("delete_from_session", a.id) in screen.action_log

    @pytest.mark.asyncio
    async def test_copy_payload_invokes_clipboard(self, monkeypatch):
        captured: list[str] = []

        def _fake_copy(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(
            "care.runtime.clipboard.copy_text", _fake_copy,
        )
        store = SessionArtifactStore()
        store.append_chain(
            chain={"steps": [{"id": "s1"}]},
            title="c", summary="",
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_copy_payload()
            await pilot.pause()
            assert captured, "copy_text should have been invoked"
            # Payload serialised as pretty JSON for chain rows.
            assert '"steps"' in captured[0]
            assert '"s1"' in captured[0]

    @pytest.mark.asyncio
    async def test_save_without_memory_facade_warns(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        # No `app.memory` slot → save path should toast a
        # warning and leave the store untouched.
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            assert getattr(app, "memory", None) is None
            screen.action_save()
            await pilot.pause()
            assert not store.list_artifacts()[0].saved_to_memory
            assert any(
                "needs a configured Memory facade" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_save_all_button_label_reflects_unsaved_count(self):
        """The footer button shows `Save all unsaved (N)`
        where N counts chain artifacts that are still
        unsaved. Tag/text artifacts don't count — Save-all
        is chain-only."""
        from textual.widgets import Button

        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c1", summary="")
        store.append_chain(chain={}, title="c2", summary="")
        store.append_tool_output(tool="grep", output="x")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            btn = screen.query_one("#artifacts-save-all-btn", Button)
            # 2 chains unsaved + 1 tool output (excluded).
            assert "Save all unsaved (2)" in str(btn.label)
            # Mark one saved → button drops to 1.
            store.mark_saved(a.id, memory_entity_id="E1")
            for _ in range(2):
                await pilot.pause()
            assert "Save all unsaved (1)" in str(btn.label)

    @pytest.mark.asyncio
    async def test_save_all_button_hides_when_nothing_unsaved(self):
        from textual.widgets import Button

        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        store.mark_saved(a.id, memory_entity_id="E")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            btn = screen.query_one("#artifacts-save-all-btn", Button)
            assert btn.has_class("-hidden")

    @pytest.mark.asyncio
    async def test_save_all_saves_every_unsaved_chain(self):
        # §3 P2 — save-all now pushes a TagEditorModal once
        # at the head of the flow. Dismiss with empty Apply
        # to match the legacy "save with no tags" path.
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append(
                    {"chain": chain, "name": name, "tags": tags},
                )
                return f"ENT-{len(captured)}"

        store = SessionArtifactStore()
        a = store.append_chain(chain={"a": 1}, title="c1", summary="")
        b = store.append_chain(chain={"b": 2}, title="c2", summary="")
        c = store.append_chain(chain={"c": 3}, title="c3", summary="")
        store.mark_saved(b.id, memory_entity_id="ENT-prev")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(10):
                await pilot.pause()
        # Only 2 unsaved chains (a, c) — b was pre-saved.
        assert len(captured) == 2
        names_saved = {row["name"] for row in captured}
        assert names_saved == {"c1", "c3"}
        # Per-row toasts include a per-artifact success line +
        # final summary "Saved all 2 unsaved chains."
        assert any(
            "Saved all 2 unsaved chains" in m for m, _ in app.toasts
        )
        # Local store reflects the new entity ids.
        assert store.get(a.id).saved_to_memory is True
        assert store.get(c.id).saved_to_memory is True
        # b's pre-existing entity_id is unchanged.
        assert store.get(b.id).memory_entity_id == "ENT-prev"

    @pytest.mark.asyncio
    async def test_save_all_partial_failure_summary(self):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        class _MemMaybe:
            def __init__(self):
                self.calls = 0
            def save_chain(self, chain, *, name=None, tags=None):
                self.calls += 1
                if self.calls % 2 == 0:
                    raise RuntimeError("503")
                return f"ENT-{self.calls}"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="")
        store.append_chain(chain={}, title="c2", summary="")
        store.append_chain(chain={}, title="c3", summary="")
        app = _Host(store=store)
        app.memory = _MemMaybe()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(10):
                await pilot.pause()
        # 3 attempts, 2 successes (odd calls), 1 failure (even).
        assert app.memory.calls == 3
        # Summary toast surfaces partial-success.
        assert any(
            "Saved 2 of 3" in m for m, _ in app.toasts
        )

    @pytest.mark.asyncio
    async def test_save_all_without_memory_warns(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        # No memory.
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(2):
                await pilot.pause()
        assert any(
            "Save all needs a configured Memory facade" in m
            for m, _ in app.toasts
        )
        assert store.list_artifacts()[0].saved_to_memory is False

    @pytest.mark.asyncio
    async def test_save_all_with_nothing_unsaved_is_noop(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        store.mark_saved(a.id, memory_entity_id="E")
        app = _Host(store=store)
        app.memory = object()  # any object — should be skipped before save_chain
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(2):
                await pilot.pause()
        assert any("Nothing to save" in m for m, _ in app.toasts)

    @pytest.mark.asyncio
    async def test_save_all_button_press_dispatches_action(self):
        """Clicking the footer Button should fire
        `action_save_all_unsaved` so the keyboard and pointer
        paths reach the same code."""
        from textual.widgets import Button

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            btn = screen.query_one("#artifacts-save-all-btn", Button)
            btn.press()
            await pilot.pause()
            assert ("save_all_unsaved", "") in screen.action_log

    @pytest.mark.asyncio
    async def test_save_with_memory_facade_marks_saved(self):
        # §3 P2 — save now pushes a TagEditorModal first.
        # Dismiss it with an empty `Apply` to get the
        # equivalent of the legacy "save with no tags" path.
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append(
                    {"chain": chain, "name": name, "tags": tags},
                )
                return "ENT-NEW"

        store = SessionArtifactStore()
        a = store.append_chain(
            chain={"steps": [{"id": "s1"}]},
            title="weather", summary="",
        )
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save()
            # Pump the modal in.
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            modal.dismiss(TagEditorResult(submitted=True))
            # Worker spawns; drain it.
            for _ in range(6):
                await pilot.pause()
            assert captured == [{
                "chain": {"steps": [{"id": "s1"}]},
                "name": "weather",
                "tags": None,
            }]
            assert store.get(a.id).saved_to_memory is True
            assert store.get(a.id).memory_entity_id == "ENT-NEW"

    @pytest.mark.asyncio
    async def test_inspect_non_chain_warns_and_skips_push(self):
        store = SessionArtifactStore()
        store.append_tool_output(tool="grep", output="x")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            _activate_tab(screen, "tool_output")
            await pilot.pause()
            before = len(app.screen_stack)
            screen.action_inspect()
            await pilot.pause()
            assert len(app.screen_stack) == before
            assert any(
                "Inspect is only available for chain" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_inspect_unsaved_chain_warns(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_inspect()
            await pilot.pause()
            assert any(
                "Inspect needs a saved chain" in m
                for m, _ in app.toasts
            )


# ---------------------------------------------------------------------------
# Tag editor on save-all (§3 P2 follow-up)
# ---------------------------------------------------------------------------


class TestTagEditorOnSaveAll:
    @pytest.mark.asyncio
    async def test_save_all_pushes_tag_editor_first(self):
        from care.screens.tag_editor import TagEditorModal

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append({"name": name, "tags": tags})
                return f"ENT-{len(captured)}"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="")
        store.append_chain(chain={}, title="c2", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            # The modal is on the stack BEFORE save_chain
            # fires.
            assert any(
                isinstance(s, TagEditorModal)
                for s in app.screen_stack
            )
            assert captured == []

    @pytest.mark.asyncio
    async def test_apply_with_tags_forwards_to_every_save(self):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append(
                    {"name": name, "tags": tuple(tags or ())},
                )
                return f"ENT-{len(captured)}"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="")
        store.append_chain(chain={}, title="c2", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(
                submitted=True,
                add_tags=("domain:weather", "batch:2026q2"),
            ))
            for _ in range(10):
                await pilot.pause()
            # Both saves got the same tag tuple.
            assert len(captured) == 2
            for entry in captured:
                assert entry["tags"] == (
                    "domain:weather", "batch:2026q2",
                )

    @pytest.mark.asyncio
    async def test_cancel_aborts_the_save_all(self):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append({"name": name})
                return "ENT-skip"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="")
        store.append_chain(chain={}, title="c2", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=False))
            for _ in range(6):
                await pilot.pause()
            # No saves happened.
            assert captured == []
            assert all(
                not a.saved_to_memory
                for a in store.list_artifacts()
            )
            assert any(
                "Save-all cancelled" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_apply_with_empty_tags_falls_through_to_no_tags(
        self,
    ):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append({"tags": tags})
                return "ENT-X"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            tag_modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(8):
                await pilot.pause()
            # No `tags=` kwarg makes it through (None upstream).
            assert captured == [{"tags": None}]

    @pytest.mark.asyncio
    async def test_save_all_seeds_tag_editor_with_union(self):
        """§3 P3 — when multiple unsaved chains carry tags,
        TagEditorModal opens with the UNION as `initial_tags`
        so the user can tweak existing labels instead of
        retyping them."""
        from care.screens.tag_editor import TagEditorModal

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                return "ENT-X"

        store = SessionArtifactStore()
        store.append_chain(
            chain={}, title="c1", summary="",
            tags=("ml", "urgent"),
        )
        store.append_chain(
            chain={}, title="c2", summary="",
            tags=("production", "ml"),  # dup ml
        )
        store.append_chain(
            chain={}, title="c3", summary="",
            tags=(),  # no tags — should be skipped
        )
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            # `list_artifacts()` defaults to newest-first, so
            # the walk order is c3 (no tags, skipped) → c2
            # (production, ml) → c1 (ml dup dropped, urgent
            # new). Insertion-order-preserving dedupe yields
            # ('production', 'ml', 'urgent').
            assert tag_modal.initial_tags == (
                "production", "ml", "urgent",
            )

    @pytest.mark.asyncio
    async def test_save_all_seed_empty_when_no_tags_on_unsaved(
        self,
    ):
        """Backwards-compat: no chains carry tags → modal
        opens with an empty chip set (matches the pre-§3-P3
        behavior)."""
        from care.screens.tag_editor import TagEditorModal

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                return "ENT-X"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c1", summary="")
        store.append_chain(chain={}, title="c2", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save_all_unsaved()
            for _ in range(4):
                await pilot.pause()
            tag_modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            assert tag_modal.initial_tags == ()


# ---------------------------------------------------------------------------
# Tag editor on save (§3 P2)
# ---------------------------------------------------------------------------


class TestTagEditorOnSave:
    @pytest.mark.asyncio
    async def test_save_pushes_tag_editor_modal_before_persisting(
        self,
    ):
        from care.screens.tag_editor import TagEditorModal

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append(
                    {"name": name, "tags": tags},
                )
                return "ENT-1"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="forecaster", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save()
            # Modal should land before save_chain fires.
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, TagEditorModal)
                for s in app.screen_stack
            )
            # save_chain has NOT been called yet — modal blocks
            # the persistence until user submits.
            assert captured == []

    @pytest.mark.asyncio
    async def test_apply_with_tags_forwards_to_save_chain(self):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append(
                    {"name": name, "tags": tuple(tags or ())},
                )
                return "ENT-tags"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="weather", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            modal.dismiss(TagEditorResult(
                submitted=True,
                add_tags=("domain:weather", "favourite"),
            ))
            for _ in range(6):
                await pilot.pause()
            assert captured == [{
                "name": "weather",
                "tags": ("domain:weather", "favourite"),
            }]

    @pytest.mark.asyncio
    async def test_cancel_aborts_the_save(self):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                captured.append({"name": name})
                return "ENT-skip"

        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            # Cancel: dismiss with submitted=False (or None
            # at all — both should abort).
            modal.dismiss(TagEditorResult(submitted=False))
            for _ in range(4):
                await pilot.pause()
            assert captured == []
            assert not store.get(a.id).saved_to_memory
            assert any(
                "Save cancelled" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_apply_with_empty_tags_saves_without_tags_kwarg(
        self,
    ):
        from care.screens.tag_editor import (
            TagEditorModal,
            TagEditorResult,
        )

        captured: list[dict] = []

        class _Mem:
            def save_chain(self, chain, *, name=None, tags=None):
                # Record the kwarg as it was passed — None when
                # the screen omitted it, list otherwise.
                captured.append({"tags": tags})
                return "ENT-empty"

        store = SessionArtifactStore()
        store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_save()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, TagEditorModal)
            )
            modal.dismiss(TagEditorResult(submitted=True))
            for _ in range(6):
                await pilot.pause()
            # tags kwarg omitted (None) → upstream save_chain
            # gets its own default; the screen doesn't synthesise
            # an empty list.
            assert captured == [{"tags": None}]


# ---------------------------------------------------------------------------
# Diff-two-chains binding (§3 P1)
# ---------------------------------------------------------------------------


class TestDiffSelected:
    @pytest.mark.asyncio
    async def test_diff_with_no_selection_warns(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="a", summary="")
        store.append_chain(chain={}, title="b", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_diff_selected()
            await pilot.pause()
            assert any(
                "needs two selected chains" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_diff_with_single_selection_warns(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="a", summary="")
        store.append_chain(chain={}, title="b", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen._selected_ids.add(a.id)
            screen.action_diff_selected()
            await pilot.pause()
            assert any(
                "needs two selected chains" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_toggle_select_only_accepts_chain_kind(self):
        store = SessionArtifactStore()
        store.append_tool_output(tool="grep", output="hits")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            _activate_tab(screen, "tool_output")
            await pilot.pause()
            screen.action_toggle_select()
            await pilot.pause()
            assert not screen._selected_ids
            assert any(
                "Only chain artifacts can be selected" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_toggle_select_marks_then_unmarks(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={"x": 1}, title="a", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            # Cursor lands on the freshest row by default.
            screen.action_toggle_select()
            assert a.id in screen._selected_ids
            # Toggle again → removed.
            screen.action_toggle_select()
            assert a.id not in screen._selected_ids
            assert (
                screen.action_log.count(("toggle_select", a.id)) == 2
            )

    @pytest.mark.asyncio
    async def test_diff_with_two_selected_pushes_modal(self):
        from care.screens.diff import DiffModal

        store = SessionArtifactStore()
        a = store.append_chain(
            chain={"steps": [{"id": "s1", "tool_id": "x"}]},
            title="left", summary="",
        )
        b = store.append_chain(
            chain={"steps": [{"id": "s1", "tool_id": "y"}]},
            title="right", summary="",
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen._selected_ids = {a.id, b.id}
            screen.action_diff_selected()
            for _ in range(6):
                await pilot.pause()
            assert any(
                isinstance(s, DiffModal)
                for s in app.screen_stack
            )
            # Diff modal should have computed without needing a
            # Memory facade — pre-loaded payload mode.
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, DiffModal)
            )
            assert modal.diff is not None
            assert modal.load_error is None

    @pytest.mark.asyncio
    async def test_diff_with_three_selected_picks_first_two(self):
        from care.screens.diff import DiffModal

        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="a", summary="")
        b = store.append_chain(chain={}, title="b", summary="")
        c = store.append_chain(chain={}, title="c", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen._selected_ids = {a.id, b.id, c.id}
            screen.action_diff_selected()
            for _ in range(6):
                await pilot.pause()
            assert any(
                isinstance(s, DiffModal)
                for s in app.screen_stack
            )


# ---------------------------------------------------------------------------
# Promote-to-stable binding (§3 P1)
# ---------------------------------------------------------------------------


class TestPromoteStable:
    @pytest.mark.asyncio
    async def test_promote_unsaved_artifact_warns(self):
        store = SessionArtifactStore()
        store.append_chain(chain={}, title="not yet saved", summary="")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_promote_stable()
            await pilot.pause()
            assert any(
                "Save the chain first" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_promote_without_memory_facade_warns(self):
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        store.mark_saved(a.id, memory_entity_id="ENT-1")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_promote_stable()
            await pilot.pause()
            assert any(
                "configured Memory facade" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_promote_chain_calls_promote_to_stable(self):
        calls: list[str] = []

        class _Mem:
            def promote_to_stable(self, entity_id):
                calls.append(entity_id)

        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        store.mark_saved(a.id, memory_entity_id="ENT-promote")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_promote_stable()
            for _ in range(6):
                await pilot.pause()
            assert calls == ["ENT-promote"]
            assert ("promote_stable", a.id) in screen.action_log
            assert any(
                "Pinned ENT-promote latest" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_promote_non_chain_kind_warns(self):
        store = SessionArtifactStore()
        store.append_tool_output(tool="grep", output="x")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            _activate_tab(screen, "tool_output")
            await pilot.pause()
            screen.action_promote_stable()
            await pilot.pause()
            assert any(
                "Only chain artifacts" in m for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_promote_when_sdk_missing_method_warns(self):
        # Memory facade is present but doesn't expose promote_to_stable.
        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        store.mark_saved(a.id, memory_entity_id="ENT-x")
        app = _Host(store=store)
        app.memory = object()  # bare; no promote_to_stable attr
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_promote_stable()
            await pilot.pause()
            assert any(
                "doesn't expose promote_to_stable" in m
                for m, _ in app.toasts
            )

    @pytest.mark.asyncio
    async def test_promote_propagates_runtime_error(self):
        class _Mem:
            def promote_to_stable(self, entity_id):
                raise RuntimeError("HTTP 503 upstream")

        store = SessionArtifactStore()
        a = store.append_chain(chain={}, title="c", summary="")
        store.mark_saved(a.id, memory_entity_id="ENT-fail")
        app = _Host(store=store)
        app.memory = _Mem()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            screen.action_promote_stable()
            for _ in range(4):
                await pilot.pause()
            assert any(
                "Promote failed" in m and "HTTP 503" in m
                for m, _ in app.toasts
            )


# ---------------------------------------------------------------------------
# CareMemory promotion API (unit, no Textual)
# ---------------------------------------------------------------------------


class TestCareMemoryPromotion:
    def test_promote_to_stable_prefers_sdk_helper(self) -> None:
        from care.memory import CareMemory

        calls: list[str] = []

        class _Client:
            def promote_to_stable(self, entity_id):
                calls.append(entity_id)

        mem = CareMemory(client=_Client())  # type: ignore[arg-type]
        mem.promote_to_stable("e-1")
        assert calls == ["e-1"]

    def test_promote_to_stable_falls_back_to_promote(self) -> None:
        from care.memory import CareMemory

        calls: list[tuple[str, str, str, str]] = []

        class _Client:
            def promote(
                self,
                entity_id,
                *,
                from_channel,
                to_channel,
                entity_type,
            ):
                calls.append((
                    entity_id,
                    from_channel,
                    to_channel,
                    entity_type,
                ))

        mem = CareMemory(client=_Client())  # type: ignore[arg-type]
        mem.promote_to_stable("e-2")
        assert calls == [("e-2", "latest", "stable", "chain")]

    def test_set_lifecycle_stable_shims_to_channel_promotion(self) -> None:
        from care.memory import CareMemory

        calls: list[str] = []

        class _Client:
            def promote_to_stable(self, entity_id):
                calls.append(entity_id)

        mem = CareMemory(client=_Client())  # type: ignore[arg-type]
        mem.set_lifecycle("e-3", "stable")
        assert calls == ["e-3"]

    def test_raises_not_implemented_when_no_method(self) -> None:
        import pytest as _pytest
        from care.memory import CareMemory

        mem = CareMemory(client=object())  # type: ignore[arg-type]
        with _pytest.raises(NotImplementedError):
            mem.promote_to_stable("e-4")

    def test_set_lifecycle_rejects_draft_and_tested(self) -> None:
        import pytest as _pytest
        from care.memory import CareMemory

        mem = CareMemory(client=object())  # type: ignore[arg-type]
        for lifecycle in ("draft", "tested", "frozen"):
            with _pytest.raises(ValueError, match="vocabulary was retired"):
                mem.set_lifecycle("e-5", lifecycle)

    def test_rejects_empty_entity_id(self) -> None:
        import pytest as _pytest
        from care.memory import CareMemory

        mem = CareMemory(client=object())  # type: ignore[arg-type]
        with _pytest.raises(ValueError, match="entity_id"):
            mem.promote_to_stable("")


# ---------------------------------------------------------------------------
# Detail pane
# ---------------------------------------------------------------------------


class TestDetailPane:
    @pytest.mark.asyncio
    async def test_detail_reflects_cursor(self):
        store = SessionArtifactStore()
        store.append_chain(
            chain={"steps": [{"id": "s1"}]},
            title="weather", summary="3-step",
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            detail = screen.query_one("#artifacts-detail", Static)
            body = str(detail.render())
            assert "weather" in body  # meta header (title)
            assert "3-step" in body   # meta header (summary)
            # The chain dict renders in the Pretty widget below the meta.
            pretty = _pretty_text(screen.query_one("#artifacts-detail-pretty", Pretty))
            assert "steps" in pretty

    @pytest.mark.asyncio
    async def test_detail_handles_payload_with_brackets(self):
        """User-reported (iter 84 follow-up): a stage_payload
        artifact whose ``str()`` rendering contains
        Rich-markup-looking tokens (e.g.
        ``[CARLStepSchema(step_type='tool', number=1, …)]``
        from a `list[<class instance>]` payload) used to
        raise `rich.MarkupError` inside `Static.update()`
        because the default `markup=True` tried to parse the
        brackets as Rich tags. Fix: detail Static constructed
        with `markup=False`.

        The minimal repro is a class instance whose `__str__`
        opens with `[<Name>(<attr>=<unquoted value>, …)` —
        that's the exact shape Rich's markup parser tries to
        interpret + fails on the unquoted integer value."""
        from textual.widgets import Static

        class _CarlStepLike:
            """Mimics the user's `CARLStepSchema` repr —
            class-name + `(attr='val', number=1)` form whose
            `str()` triggers Rich's markup parser inside the
            literal `[...]` Python-list brackets."""
            def __init__(self, *, step_type: str, number: int):
                self.step_type = step_type
                self.number = number
            def __repr__(self) -> str:
                return (
                    f"CARLStepSchema(step_type='{self.step_type}', "
                    f"number={self.number}, "
                    f"step_context_queries=[], "
                    f"tool_input_mapping=None)"
                )

        store = SessionArtifactStore()
        store.append_stage_payload(
            stage="step_describing",
            payload=[
                _CarlStepLike(step_type="tool", number=1),
                _CarlStepLike(step_type="llm", number=2),
            ],
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            _activate_tab(screen, "stage_payload")
            await pilot.pause()
            # Repaint the cursor row — this is the exact code
            # path that crashed in the user's report. Without
            # the markup=False fix this raises MarkupError
            # before the assertion lands.
            screen._refresh_detail()
            await pilot.pause()
            detail = screen.query_one(
                "#artifacts-detail", Static,
            )
            body = str(detail.render())
            assert "stage_payload" in body
            assert "step_describing" in body
            # The class reprs survive into the body.
            assert "CARLStepSchema" in body


# ---------------------------------------------------------------------------
# Detail-pane scroll + JSON/DAG toggle
# ---------------------------------------------------------------------------


class TestDetailScrollAndDagToggle:
    @pytest.mark.asyncio
    async def test_detail_lives_in_a_vertical_scroll(self):
        """The detail Static is wrapped in a VerticalScroll so
        long chain payloads scroll instead of clipping."""
        from textual.containers import VerticalScroll

        store = SessionArtifactStore()
        store.append_chain(
            chain={"steps": [{"name": f"s{i}"} for i in range(40)]},
            title="big", summary="",
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            scroll = screen.query_one(
                "#artifacts-detail-scroll", VerticalScroll,
            )
            detail = screen.query_one("#artifacts-detail", Static)
            # The Static is a descendant of the scroll container.
            assert detail in scroll.query(Static)

    @pytest.mark.asyncio
    async def test_toggle_switches_chain_detail_to_dag(self):
        """`action_toggle_view` flips a chain's detail pane from
        JSON to the ASCII DAG and back, with the button label
        tracking the next view."""
        from textual.widgets import Button

        store = SessionArtifactStore()
        store.append_chain(
            chain={
                "steps": [
                    {"name": "fetch", "type": "tool"},
                    {"name": "summarise", "type": "llm",
                     "depends_on": ["fetch"]},
                ],
            },
            title="weather", summary="",
        )
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            detail = screen.query_one("#artifacts-detail", Static)
            btn = screen.query_one(
                "#artifacts-detail-toggle", Button,
            )
            # Default = JSON; button invites switching to DAG.
            assert screen._detail_view == "json"
            assert not btn.has_class("-hidden")
            assert "Show DAG" in str(btn.label)
            # JSON view: the chain dict renders in the Pretty widget.
            assert "steps" in _pretty_text(
                screen.query_one("#artifacts-detail-pretty", Pretty),
            )
            # Flip → DAG view shows the tree connectors, not JSON.
            screen.action_toggle_view()
            await pilot.pause()
            assert screen._detail_view == "dag"
            assert "Show JSON" in str(btn.label)
            body = str(detail.render())
            assert "fetch" in body
            assert "summarise" in body
            assert "└─" in body or "├─" in body
            assert '"steps"' not in body
            # Flip back → JSON (chain dict back in the Pretty widget).
            screen.action_toggle_view()
            await pilot.pause()
            assert screen._detail_view == "json"
            assert "steps" in _pretty_text(
                screen.query_one("#artifacts-detail-pretty", Pretty),
            )

    @pytest.mark.asyncio
    async def test_toggle_hidden_and_noop_for_non_chain(self):
        """The DAG toggle hides for non-chain artifacts and the
        action only toasts a hint without changing the view."""
        from textual.widgets import Button

        store = SessionArtifactStore()
        store.append_tool_output(tool="grep", output="hits")
        app = _Host(store=store)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _artifacts_screen(app)
            _activate_tab(screen, "tool_output")
            await pilot.pause()
            btn = screen.query_one(
                "#artifacts-detail-toggle", Button,
            )
            assert btn.has_class("-hidden")
            screen.action_toggle_view()
            await pilot.pause()
            assert screen._detail_view == "json"
            assert any(
                "DAG view is only available for chain" in m
                for m, _ in app.toasts
            )


# ---------------------------------------------------------------------------
# Slash command + palette opener
# ---------------------------------------------------------------------------


class TestSlashAndPaletteIntegration:
    @pytest.mark.asyncio
    async def test_slash_artifacts_pushes_real_screen(self):
        # Drive via a real ChatScreen to confirm the slash
        # registry routes to the dedicated screen now (not the
        # iter-1 stub).
        from care.screens.chat import ChatScreen
        from care.widgets.chat_input import ChatInput

        class _ChatHost(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(ChatScreen())

        app = _ChatHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = next(
                s for s in app.screen_stack if isinstance(s, ChatScreen)
            )
            inp = chat.query_one("#chat-input", ChatInput)
            inp.value = "/artifacts"
            await inp.action_submit()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ArtifactsScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_palette_open_artifacts_finds_chat_store(self):
        from care.app import CareApp
        from care.screens.chat import ChatScreen
        from care.screens.welcome import WelcomeScreen

        WelcomeScreen.DEFAULT_SPLASH_SECONDS = 0.0
        app = CareApp(mode="returning")
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            # Push a ChatScreen so the palette opener finds
            # something to read the store from.
            chat = ChatScreen()
            app.push_screen(chat)
            for _ in range(4):
                await pilot.pause()
            app.action_palette_open_artifacts()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, ArtifactsScreen)
                for s in app.screen_stack
            )

    @pytest.mark.asyncio
    async def test_palette_open_artifacts_without_chat_warns(self):
        from care.app import CareApp
        from care.screens.welcome import WelcomeScreen

        WelcomeScreen.DEFAULT_SPLASH_SECONDS = 0.0
        app = CareApp(mode="first_run")
        toasts: list[tuple[str, str]] = []
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            original = app.push_toast

            def _spy(message, *, severity="info", ttl=None):
                toasts.append((message, severity))
                return original(message, severity=severity, ttl=ttl)

            app.push_toast = _spy  # type: ignore[method-assign]
            # No ChatScreen mounted in first_run mode (WelcomeScreen
            # routes to SettingsScreen instead).
            app.action_palette_open_artifacts()
            for _ in range(2):
                await pilot.pause()
        assert any(
            "Open the chat first" in m for m, _ in toasts
        )
