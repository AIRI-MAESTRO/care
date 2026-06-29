"""Pilot tests for InspectionScreen (TODO §1.1 P0.19).

Exercises:
* `on_mount` fetches the chain via `memory.client.get_chain`
  (falls through to `get_chain_dict` / `get_chain_raw`).
* Four panels render: step list, detail, DAG, memory footer.
* Action bar buttons + key bindings dispatch
  :class:`ActionRequested` envelopes.
* `Back` pops the screen.
* Error state renders without crashing.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Static, TextArea

from care.screens.inspection import (
    InspectionPayload,
    InspectionScreen,
    _project_payload,
    format_step_detail,
    render_chain_dag,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _chain_response(*, entity_id="agent-1"):
    return {
        "entity_id": entity_id,
        "entity_type": "chain",
        "version_id": "v1",
        "channel": "latest",
        "etag": "e",
        "favourite": False,
        "meta": {
            "display_name": "Storm Watcher",
            "domain": "weather",
            "tags": ["weather"],
            "name": "storm",
        },
        "content": {
            "steps": [
                {
                    "name": "fetch",
                    "type": "llm",
                    "prompt_template": "Fetch weather for {region}",
                    "deps": [],
                },
                {
                    "name": "summarise",
                    "type": "llm",
                    "prompt_template": "Summarise {fetch}",
                    "deps": ["fetch"],
                },
            ],
            "description": "Watches storms",
        },
    }


class _StubClient:
    def __init__(self, response=None, *, fail: bool = False):
        self._response = response if response is not None else _chain_response()
        self._fail = fail
        self.calls: list[tuple] = []
        self.delete_calls: list[tuple] = []

    def get_chain(self, entity_id, channel):
        self.calls.append((entity_id, channel))
        if self._fail:
            raise RuntimeError("boom")
        return self._response

    def _delete_entity(self, entity_type, entity_id):
        self.delete_calls.append((entity_type, entity_id))
        return True


class _StubMemory:
    def __init__(self, response=None, *, fail: bool = False):
        self.client = _StubClient(response, fail=fail)


class _InspHost(App):
    def __init__(self, *, entity_id="agent-1", memory=None) -> None:
        super().__init__()
        self.memory = memory if memory is not None else _StubMemory()
        self._entity_id = entity_id
        self.actions: list[tuple[str, str]] = []
        self.refresh_calls: int = 0

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(InspectionScreen(self._entity_id))

    def on_inspection_screen_action_requested(
        self, event: InspectionScreen.ActionRequested,
    ) -> None:
        self.actions.append((event.action, event.entity_id))

    def _refresh_library_screens(self) -> None:
        self.refresh_calls += 1


def _screen(app: App) -> InspectionScreen:
    screen = app.screen_stack[-1]
    assert isinstance(screen, InspectionScreen)
    return screen


# ---------------------------------------------------------------------------
# Payload projection
# ---------------------------------------------------------------------------


class TestPayloadProjection:
    def test_empty_response_returns_empty_steps(self):
        p = _project_payload(None, entity_id="x", channel="latest")
        assert p.entity_id == "x"
        assert p.steps == ()

    def test_dict_response_unwraps_content(self):
        p = _project_payload(
            _chain_response(), entity_id="agent-1", channel="latest",
        )
        assert p.display_name == "Storm Watcher"
        assert p.domain == "weather"
        assert len(p.steps) == 2
        assert p.steps[0]["name"] == "fetch"
        assert p.version_id == "v1"

    def test_step_label_falls_back_to_step_n(self):
        p = InspectionPayload(
            entity_id="x", steps=({"type": "llm"},),
        )
        # No title → falls back to `step-1`; `llm` renders as the
        # friendly "AI" kind.
        assert p.step_label(0) == "step-1 (AI)"

    def test_step_label_uses_carl_title_and_type(self):
        p = InspectionPayload(
            entity_id="x",
            steps=({"number": 1, "title": "Analyse", "step_type": "llm"},),
        )
        assert p.step_label(0) == "Analyse (AI)"


class TestStepDetail:
    def test_llm_step_header_and_fields(self):
        step = {
            "number": 2,
            "title": "Synthesise answer",
            "step_type": "llm",
            "dependencies": [1],
            "aim": "Merge the findings",
            "reasoning_questions": "What matters?",
            "llm_config": {"model": "gpt-4o", "temperature": 0.2},
        }
        steps = [
            {"number": 1, "title": "Gather", "step_type": "tool"},
            step,
        ]
        header, body = format_step_detail(step, 1, steps)
        assert header == "2. Synthesise answer  ·  AI"
        # Dependency number resolves to the upstream step's title.
        assert "Depends on: 1 (Gather)" in body
        assert "Aim: Merge the findings" in body
        assert "Reasoning questions: What matters?" in body
        assert "Model: gpt-4o" in body
        assert "Temperature: 0.2" in body

    def test_tool_step_config_unpacked(self):
        step = {
            "number": 1,
            "title": "Fetch page",
            "step_type": "tool",
            "config": {"tool_name": "http_get"},
        }
        header, body = format_step_detail(step, 0, [step])
        assert header == "1. Fetch page  ·  Tool"
        assert "Tool name: http_get" in body

    def test_unknown_shape_falls_back_to_remaining_fields(self):
        step = {"number": 1, "title": "Mystery", "step_type": "llm",
                "custom_field": "value"}
        header, body = format_step_detail(step, 0, [step])
        assert header == "1. Mystery  ·  AI"
        assert "Custom field: value" in body

    def test_reasoning_chain_object_via_to_dict(self):
        # GigaEvoClient.get_chain() returns a ReasoningChain dataclass
        # (no `.content`, no `model_dump`) whose `to_dict()` carries
        # `steps` at the top level. Regression: steps used to project
        # empty because the object shape wasn't handled.
        class _Chain:
            def to_dict(self):
                return {
                    "steps": [
                        {"name": "fetch", "number": 1},
                        {"name": "summarise", "number": 2,
                         "dependencies": [1]},
                    ],
                    "max_workers": 2,
                }

        p = _project_payload(_Chain(), entity_id="c-1", channel="latest")
        assert len(p.steps) == 2
        assert p.steps[0]["name"] == "fetch"


# ---------------------------------------------------------------------------
# Compose + fetch
# ---------------------------------------------------------------------------


class TestCompose:
    @pytest.mark.asyncio
    async def test_panels_render_after_fetch(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            assert screen.state.payload is not None
            assert screen.state.payload.entity_id == "agent-1"
            assert len(screen.state.payload.steps) == 2
            # Step list populated with clickable step buttons (chat-modal
            # style), one per step.
            container = screen.query_one(
                "#inspection-step-list", VerticalScroll,
            )
            step_btns = [
                b for b in container.query(Button)
                if (b.id or "").startswith("inspection-stepbtn-")
            ]
            assert len(step_btns) == 2
            # DAG / footer Statics are mounted.
            assert screen.query_one("#inspection-dag-text", Static) is not None
            assert screen.query_one(
                "#inspection-memory-footer", Static,
            ) is not None

    @pytest.mark.asyncio
    async def test_clicking_step_button_selects_step(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            btn = screen.query_one("#inspection-stepbtn-1", Button)
            screen.on_button_pressed(Button.Pressed(btn))
            await pilot.pause()
            # Selection moved + the button is tinted active…
            assert screen.selected_step == 1
            assert btn.has_class("-active")
            assert not screen.query_one(
                "#inspection-stepbtn-0", Button,
            ).has_class("-active")
            # …and the detail pane reflects the selected step.
            body = screen.query_one("#inspection-detail-body", VerticalScroll)
            rendered = "\n".join(str(c.render()) for c in body.children)
            assert "summarise" in rendered.lower()

    @pytest.mark.asyncio
    async def test_action_bar_renders_five_buttons(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            for bid in (
                "inspection-btn-run",
                "inspection-btn-edit",
                "inspection-btn-evolve",
                "inspection-btn-back",
            ):
                assert screen.query_one(f"#{bid}", Button) is not None
            # The Duplicate button was removed from the action bar.
            from textual.css.query import NoMatches

            with pytest.raises(NoMatches):
                screen.query_one("#inspection-btn-duplicate", Button)


# ---------------------------------------------------------------------------
# Fetch resilience
# ---------------------------------------------------------------------------


class TestFetchResilience:
    @pytest.mark.asyncio
    async def test_no_memory_renders_error_state(self):
        class _NoMemHost(App):
            memory = None

            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.push_screen(InspectionScreen("x"))

        app = _NoMemHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            assert screen.state.loading is False
            assert screen.state.error is not None
            assert screen.state.payload is None

    @pytest.mark.asyncio
    async def test_client_failure_lands_on_state_error(self):
        app = _InspHost(memory=_StubMemory(fail=True))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            assert screen.state.error is not None
            assert "boom" in screen.state.error


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TestActions:
    @pytest.mark.asyncio
    async def test_run_button_dispatches(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.query_one(
                "#inspection-btn-run", Button,
            ).press()
            await pilot.pause()
            await pilot.pause()
            assert ("run", "agent-1") in app.actions

    @pytest.mark.asyncio
    async def test_keyboard_shortcuts_dispatch(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.action_inspect_edit()
            screen.action_inspect_evolve()
            screen.action_inspect_duplicate()
            await pilot.pause()
            kinds = [a for a, _ in app.actions]
            assert "edit" in kinds
            assert "evolve" in kinds
            assert "duplicate" in kinds

    @pytest.mark.asyncio
    async def test_back_pops_screen(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            initial_depth = len(app.screen_stack)
            screen.action_inspect_back()
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen_stack) < initial_depth
            assert ("back", "agent-1") in app.actions


# ---------------------------------------------------------------------------
# DAG visualisation (§4 P0)
# ---------------------------------------------------------------------------


class TestDagRender:
    """Pure unit tests for `render_chain_dag` — no Textual."""

    def _render(self, steps):
        from care.screens.inspection import render_chain_dag

        return render_chain_dag(steps)

    def test_empty_returns_placeholder(self):
        assert self._render([]) == "(empty)"

    def test_linear_chain_renders_nested_tree(self):
        steps = [
            {"id": "a", "type": "llm"},
            {"id": "b", "type": "tool", "deps": ["a"]},
            {"id": "c", "type": "llm", "deps": ["b"]},
        ]
        out = self._render(steps)
        # All three steps appear.
        assert "a (AI)" in out
        assert "b (Tool)" in out
        assert "c (AI)" in out
        # Root sits on the outermost level; children are
        # indented by `   ` per depth — `c` should be three
        # tree connectors deep (root → mid → leaf).
        a_idx = out.index("a (AI)")
        b_idx = out.index("b (Tool)")
        c_idx = out.index("c (AI)")
        # The Y order is preserved.
        assert a_idx < b_idx < c_idx

    def test_carl_numeric_dependencies_resolve_edges(self):
        """Real CARL chains express edges as ``dependencies``
        referencing each step's ``number`` (an int). Those must
        resolve to graph edges, not dangle as unresolved."""
        steps = [
            {"number": 1, "name": "fetch", "type": "tool",
             "dependencies": []},
            {"number": 2, "name": "summarise", "type": "llm",
             "dependencies": [1]},
        ]
        out = self._render(steps)
        assert "fetch (Tool)" in out
        assert "summarise (AI)" in out
        # `summarise` nests under `fetch` (deeper connector), and
        # nothing dangles as unresolved.
        assert "(unresolved dependencies)" not in out
        assert out.index("fetch (Tool)") < out.index("summarise (AI)")

    def test_diamond_fan_in_renders_step_twice(self):
        """Fan-in is intentionally rendered under every parent
        — visual cost is acceptable for CARL's typical
        topologies and keeps the tree readable."""
        steps = [
            {"id": "root", "type": "llm"},
            {"id": "left", "type": "tool", "deps": ["root"]},
            {"id": "right", "type": "tool", "deps": ["root"]},
            {"id": "merge", "type": "llm", "deps": ["left", "right"]},
        ]
        out = self._render(steps)
        # `merge` shows up under both `left` and `right`.
        assert out.count("merge (AI)") == 2

    def test_unresolved_dep_surfaces_warning_block(self):
        steps = [
            {"id": "a", "type": "llm"},
            {"id": "b", "type": "tool", "deps": ["missing"]},
        ]
        out = self._render(steps)
        assert "(unresolved dependencies)" in out
        assert "b (Tool) → missing (?)" in out

    def test_cycle_fallback_renders_flat_list(self):
        # Every step depends on something → no roots → cycle
        # fallback.
        steps = [
            {"id": "a", "type": "llm", "deps": ["b"]},
            {"id": "b", "type": "tool", "deps": ["a"]},
        ]
        out = self._render(steps)
        assert "cycle detected" in out
        assert "a (AI)" in out
        assert "b (Tool)" in out

    def test_self_loop_renders_recurrence_glyph(self):
        steps = [
            {"id": "a", "type": "llm"},
            {"id": "b", "type": "tool", "deps": ["a"]},
            {"id": "a", "type": "llm", "deps": ["b"]},
        ]
        # Last-write-wins for id "a" makes it appear with deps.
        # That collapses to a pure cycle → flat fallback.
        out = self._render(steps)
        # Either the cycle fallback fires, or a `↺` glyph
        # appears — both are acceptable cycle treatments.
        assert ("↺" in out) or ("cycle detected" in out)


class TestDagPaneToggle:
    @pytest.mark.asyncio
    async def test_toggle_hides_then_shows_dag_pane(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            dag = screen.query_one("#inspection-dag")
            assert dag.display is True
            assert screen._dag_visible is True
            screen.action_toggle_dag()
            await pilot.pause()
            assert dag.display is False
            assert screen._dag_visible is False
            screen.action_toggle_dag()
            await pilot.pause()
            assert dag.display is True
            assert screen._dag_visible is True

    @pytest.mark.asyncio
    async def test_dag_pane_contains_step_labels_after_fetch(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            dag_text = screen.query_one(
                "#inspection-dag-text", Static,
            )
            rendered = str(dag_text.render())
            # Step names from the canonical stub fixture.
            assert "fetch" in rendered
            assert "summarise" in rendered


class TestMarkupSafeContent:
    """Chain-derived text with bracket syntax (e.g. `memory[-1]`,
    typed-config reprs) must not be parsed as Rich markup — doing so
    raised a MarkupError that crashed the whole inspection render."""

    @pytest.mark.asyncio
    async def test_step_config_with_brackets_renders(self):
        response = _chain_response()
        response["content"]["steps"][0]["config"] = (
            "EvaluationStepConfig(eval=memory[-1], max_retries=1)"
        )
        response["content"]["steps"][0]["name"] = "score[final]"
        app = _InspHost(memory=_StubMemory(response))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            # Reached a rendered payload without a MarkupError crash.
            assert screen.state.payload is not None
            body = screen.query_one("#inspection-detail-body", VerticalScroll)
            rendered = "\n".join(str(c.render()) for c in body.children)
            assert "memory[-1]" in rendered

    def test_dag_render_with_bracketed_step_name(self):
        # The DAG Static is constructed markup-free; the helper output
        # itself stays raw so brackets survive verbatim.
        out = render_chain_dag(
            [{"name": "score[final]", "deps": []}],
        )
        assert "score[final]" in out


# ---------------------------------------------------------------------------
# Integration pane (§4 P0)
# ---------------------------------------------------------------------------


class TestIntegrationPane:
    @pytest.mark.asyncio
    async def test_pane_renders_id_meta_and_default_python_snippet(
        self,
    ):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            id_pane = screen.query_one(
                "#inspection-integration-id", Static,
            )
            meta_pane = screen.query_one(
                "#inspection-integration-meta", Static,
            )
            snippet_pane = screen.query_one(
                "#inspection-integration-snippet", TextArea,
            )
            assert "chain_id: agent-1" in str(id_pane.render())
            assert "version: v1" in str(meta_pane.render())
            assert "channel: latest" in str(meta_pane.render())
            # Default python snippet (read-only code viewer) carries the id.
            body = snippet_pane.text
            assert "agent-1" in body
            assert "GigaEvoClient" in body
            assert snippet_pane.language == "python"

    @pytest.mark.asyncio
    async def test_t_cycles_language_and_updates_snippet(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            assert screen._integration_lang == "python"
            screen.action_integration_cycle_lang()
            await pilot.pause()
            assert screen._integration_lang == "curl"
            snippet = screen.query_one(
                "#inspection-integration-snippet", TextArea,
            ).text
            assert "curl" in snippet
            screen.action_integration_cycle_lang()
            assert screen._integration_lang == "cli"
            screen.action_integration_cycle_lang()
            assert screen._integration_lang == "python"
            assert ("cycle_lang", "curl") in (
                screen.integration_action_log
            )

    @pytest.mark.asyncio
    async def test_y_copies_chain_id(self, monkeypatch):
        captured: list[str] = []

        def _fake_copy(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(
            "care.runtime.clipboard.copy_text", _fake_copy,
        )
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.action_integration_copy_id()
            await pilot.pause()
            assert captured == ["agent-1"]
            assert ("copy_id", "agent-1") in (
                screen.integration_action_log
            )

    @pytest.mark.asyncio
    async def test_c_copies_active_snippet(self, monkeypatch):
        captured: list[str] = []

        def _fake_copy(text: str) -> None:
            captured.append(text)

        monkeypatch.setattr(
            "care.runtime.clipboard.copy_text", _fake_copy,
        )
        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            # python → curl → cli
            screen.action_integration_cycle_lang()
            screen.action_integration_cycle_lang()
            screen.action_integration_copy_snippet()
            await pilot.pause()
            assert captured
            assert "care run agent-1" in captured[0]

    @pytest.mark.asyncio
    async def test_L_opens_lineage_modal(self):
        from care.screens.lineage import LineageModal

        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.action_integration_open_lineage()
            for _ in range(4):
                await pilot.pause()
            assert any(
                isinstance(s, LineageModal)
                for s in app.screen_stack
            )
            assert ("open_lineage", "agent-1") in (
                screen.integration_action_log
            )

    @pytest.mark.asyncio
    async def test_u_opens_use_it_now_modal_prefilled(self):
        """§3 P1 — `u` opens UseItNowModal with this chain's
        identity (entity_id / version / channel / display
        name)."""
        from care.screens.use_it_now import UseItNowModal

        app = _InspHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.action_integration_use_it_now()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, UseItNowModal)
            )
            # The chain stub fixture writes entity_id
            # `agent-1` + version_id `v1` + channel
            # `latest` + display_name "Storm Watcher".
            assert modal.entity_id == "agent-1"
            assert modal.version == "v1"
            assert modal.channel == "latest"
            assert modal.display_name == "Storm Watcher"
            # Action log records the dispatch.
            assert ("use_it_now", "agent-1") in (
                screen.integration_action_log
            )

    @pytest.mark.asyncio
    async def test_use_it_now_evolve_routes_to_push_evolution_for(
        self,
    ):
        """Dismissing the modal with `evolve_requested=True`
        routes through `app._push_evolution_for` (the same
        plumbing the ArtifactsScreen save flow uses)."""
        from care.screens.use_it_now import (
            UseItNowModal,
            UseItNowResult,
        )

        invocations: list[str] = []

        # Subclass the host to expose the optional opener.
        class _Host2(_InspHost):
            def _push_evolution_for(self, eid: str) -> None:
                invocations.append(eid)

        app = _Host2()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.action_integration_use_it_now()
            for _ in range(4):
                await pilot.pause()
            modal = next(
                s for s in app.screen_stack
                if isinstance(s, UseItNowModal)
            )
            modal.dismiss(UseItNowResult(
                closed=False, evolve_requested=True,
            ))
            for _ in range(4):
                await pilot.pause()
            assert invocations == ["agent-1"]

    @pytest.mark.asyncio
    async def test_curl_snippet_uses_memory_base_url_when_available(
        self,
    ):
        class _ClientWithBaseURL(_StubClient):
            base_url = "https://memory.example.com"

        class _MemWithBaseURL(_StubMemory):
            def __init__(self):
                self.client = _ClientWithBaseURL(
                    _chain_response(),
                )

        app = _InspHost(memory=_MemWithBaseURL())
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = _screen(app)
            screen.action_integration_cycle_lang()  # → curl
            await pilot.pause()
            snippet = screen.query_one(
                "#inspection-integration-snippet", TextArea,
            ).text
            assert "https://memory.example.com" in snippet
            assert "agent-1" in snippet


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReloadOnResume:
    """Returning to Inspection (e.g. after the Edit screen pops on a
    save) must re-fetch so manual edits show instead of stale cache."""

    @pytest.mark.asyncio
    async def test_resume_refetches_after_initial_load(self):
        class _MutableClient(_StubClient):
            def get_chain(self, entity_id, channel):
                self.calls.append((entity_id, channel))
                # Second fetch (the resume) returns edited content.
                if len(self.calls) >= 2:
                    resp = _chain_response()
                    resp["content"]["steps"][0]["name"] = "EDITED"
                    return resp
                return self._response

        class _MutMem:
            def __init__(self):
                self.client = _MutableClient()

        app = _InspHost(memory=_MutMem())
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _screen(app)
            # Initial load landed.
            assert screen.state.payload is not None
            assert screen.state.payload.steps[0]["name"] == "fetch"
            calls_after_mount = len(app.memory.client.calls)
            # Simulate returning to the screen.
            screen.on_screen_resume()
            for _ in range(6):
                await pilot.pause()
            # It re-fetched and now reflects the edit.
            assert len(app.memory.client.calls) > calls_after_mount
            assert screen.state.payload.steps[0]["name"] == "EDITED"

    @pytest.mark.asyncio
    async def test_resume_before_initial_load_does_not_double_fetch(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            screen = _screen(app)
            # Force the "not yet loaded" state and resume — must no-op.
            screen.state.payload = None
            before = len(app.memory.client.calls)
            screen.on_screen_resume()
            await pilot.pause()
            assert len(app.memory.client.calls) == before

    @pytest.mark.asyncio
    async def test_resume_bypasses_read_cache(self):
        """The reload-after-edit must pass force_refresh=True so the
        client's read cache can't serve the stale pre-save content."""

        class _CacheAwareClient(_StubClient):
            def __init__(self):
                super().__init__()
                self.force_flags: list[bool] = []

            def get_chain_dict(
                self, entity_id, channel="latest",
                cache_ttl=None, force_refresh=False,
            ):
                self.calls.append((entity_id, channel))
                self.force_flags.append(force_refresh)
                return self._response

        class _CacheMem:
            def __init__(self):
                self.client = _CacheAwareClient()

        app = _InspHost(memory=_CacheMem())
        async with app.run_test() as pilot:
            for _ in range(6):
                await pilot.pause()
            screen = _screen(app)
            assert screen.state.payload is not None
            # Initial mount load did NOT force a refresh.
            assert app.memory.client.force_flags
            assert app.memory.client.force_flags[0] is False
            screen.on_screen_resume()
            for _ in range(6):
                await pilot.pause()
            # The resume reload bypassed the cache.
            assert app.memory.client.force_flags[-1] is True


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import InspectionPayload as P
        from care.screens import InspectionScreen as S

        assert P is InspectionPayload
        assert S is InspectionScreen


# ---------------------------------------------------------------------------
# Freshness badge (§4 P1)
# ---------------------------------------------------------------------------


class TestFreshnessBadge:
    """`R` rebinding + periodic poll against
    `memory.get_entity` paints a fresh/stale/unknown badge on
    the Integration pane's meta line."""

    def test_R_bound_to_refresh_freshness(self):
        action_by_key = {
            b.key: getattr(b, "action", None)
            for b in InspectionScreen.BINDINGS
        }
        # Textual normalises uppercase letter bindings to the
        # `upper_<letter>` form.
        candidates = {
            v for k, v in action_by_key.items()
            if k in ("R", "upper_r")
        }
        assert "refresh_freshness" in candidates, (
            f"`R` must bind to refresh_freshness; got "
            f"{[k for k, v in action_by_key.items() if v == 'refresh_freshness']}"
        )

    def test_format_freshness_badge_states(self):
        screen = InspectionScreen("ent-1")
        screen.freshness_status = "fresh"
        assert "fresh" in screen._format_freshness_badge()
        assert "●" in screen._format_freshness_badge()
        screen.freshness_status = "stale"
        out = screen._format_freshness_badge()
        assert "stale" in out
        assert "refresh with R" in out
        screen.freshness_status = "unknown"
        assert "unknown" in screen._format_freshness_badge()

    @pytest.mark.asyncio
    async def test_pinned_version_after_load(self):
        """After the bootstrap load lands, the pinned version
        is set from the payload + the badge reads `fresh`."""
        app = _InspHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.pinned_version_id == "v1"
            assert screen.freshness_status == "fresh"

    @pytest.mark.asyncio
    async def test_check_freshness_detects_new_version(self):
        """When the live poll returns a different version_id
        than the pinned baseline, the status flips to stale."""
        class _MemWithGetEntity:
            def __init__(self):
                self.client = _StubClient()
                self._version = "v1"
            def get_entity(
                self, entity_id, *, entity_type, channel="latest",
            ):
                return {
                    "entity_id": entity_id,
                    "version_id": self._version,
                    "channel": channel,
                }

        memory = _MemWithGetEntity()
        app = _InspHost(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.freshness_status == "fresh"
            # Server now ships a newer version.
            memory._version = "v2"
            await screen._check_freshness()
            await pilot.pause()
            assert screen.freshness_status == "stale"
            assert screen.pinned_version_id == "v1"

    @pytest.mark.asyncio
    async def test_check_freshness_same_version_stays_fresh(self):
        class _MemWithGetEntity:
            def __init__(self):
                self.client = _StubClient()
            def get_entity(
                self, entity_id, *, entity_type, channel="latest",
            ):
                return {
                    "entity_id": entity_id,
                    "version_id": "v1",  # unchanged
                    "channel": channel,
                }

        app = _InspHost(memory=_MemWithGetEntity())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            await screen._check_freshness()
            await pilot.pause()
            assert screen.freshness_status == "fresh"

    @pytest.mark.asyncio
    async def test_check_freshness_exception_marks_unknown(self):
        class _MemRaises:
            def __init__(self):
                self.client = _StubClient()
            def get_entity(
                self, entity_id, *, entity_type, channel="latest",
            ):
                raise RuntimeError("503 backend down")

        app = _InspHost(memory=_MemRaises())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            await screen._check_freshness()
            await pilot.pause()
            assert screen.freshness_status == "unknown"
            assert "503 backend down" in screen.freshness_last_error

    @pytest.mark.asyncio
    async def test_check_freshness_no_pinned_version_skips(self):
        class _MemWithGetEntity:
            calls: list = []
            def __init__(self):
                self.client = _StubClient()
            def get_entity(
                self, entity_id, *, entity_type, channel="latest",
            ):
                self.calls.append(entity_id)
                return {"version_id": "vX"}

        memory = _MemWithGetEntity()
        app = _InspHost(memory=memory)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.pinned_version_id = ""  # clear baseline
            await screen._check_freshness()
            await pilot.pause()
            # No baseline → skip the get_entity call entirely.
            assert memory.calls == []

    @pytest.mark.asyncio
    async def test_check_freshness_missing_get_entity_skips(self):
        # Older Memory facade — only `client.get_chain`, no
        # `.get_entity`. The poll silently returns.
        class _OldMem:
            def __init__(self):
                self.client = _StubClient()
            # NO get_entity method.

        app = _InspHost(memory=_OldMem())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            assert screen.freshness_status == "fresh"
            await screen._check_freshness()
            await pilot.pause()
            # Status stays where it was — silent no-op.
            assert screen.freshness_status == "fresh"

    @pytest.mark.asyncio
    async def test_action_refresh_freshness_respawns_load(self):
        """`action_refresh_freshness` fires a fresh `_load()`
        worker which re-pins the baseline. Action log records
        the gesture."""
        class _MutableMem:
            def __init__(self):
                self.client = _StubClient(
                    _chain_response(entity_id="agent-1"),
                )
            def get_entity(
                self, entity_id, *, entity_type, channel="latest",
            ):
                return {"version_id": "v1"}

        app = _InspHost(memory=_MutableMem())
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            initial_calls = len(app.memory.client.calls)
            screen.action_refresh_freshness()
            for _ in range(4):
                await pilot.pause()
            assert (
                len(app.memory.client.calls) > initial_calls
            ), "refresh should respawn the _load() worker"
            assert ("refresh", "") in screen.integration_action_log

    @pytest.mark.asyncio
    async def test_badge_renders_in_meta_line(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            meta = str(screen.query_one(
                "#inspection-integration-meta", Static,
            ).render())
            # Initial state after load: fresh badge present.
            assert "fresh" in meta
            assert "channel:" in meta


# ---------------------------------------------------------------------------
# Delete (soft-delete via confirm modal)
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_button_present(self):
        app = _InspHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            btn = screen.query_one("#inspection-btn-delete", Button)
            assert btn.variant == "error"

    @pytest.mark.asyncio
    async def test_delete_pushes_confirm_modal(self):
        from care.screens.confirm import ConfirmModal

        app = _InspHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            _screen(app).action_inspect_delete()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], ConfirmModal)

    @pytest.mark.asyncio
    async def test_delete_cancel_does_nothing(self):
        from care.screens.confirm import ConfirmModal

        app = _InspHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            _screen(app).action_inspect_delete()
            for _ in range(4):
                await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ConfirmModal)
            modal.dismiss(False)
            for _ in range(4):
                await pilot.pause()
            assert app.memory.client.delete_calls == []
            assert isinstance(app.screen_stack[-1], InspectionScreen)

    @pytest.mark.asyncio
    async def test_delete_confirm_runs_and_pops(self):
        from care.screens.confirm import ConfirmModal

        app = _InspHost()
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            screen = _screen(app)
            screen.action_inspect_delete()
            for _ in range(4):
                await pilot.pause()
            modal = app.screen_stack[-1]
            assert isinstance(modal, ConfirmModal)
            modal.dismiss(True)
            for _ in range(4):
                await pilot.pause()
            assert app.memory.client.delete_calls == [("chain", "agent-1")]
            assert screen.last_delete_outcome.success is True
            assert app.refresh_calls == 1
            assert not isinstance(app.screen_stack[-1], InspectionScreen)
