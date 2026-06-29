"""Pilot tests for RunContextModal (TODO §1.1 P0.21).

Exercises:
* `extract_run_context_draft` seeds the form from a saved chain.
* Field edits route through the `set_*` / `drop_file` / `restore_file`
  mutators.
* `validate_run_context_draft` issues render inline.
* Submit dismisses with `RunContextResult(submitted=True, draft)`;
  cancel dismisses with `submitted=False`.
* The host can hand the result's draft to `apply_overrides` +
  `build_extra_kwargs`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, Static, TextArea

from care.runtime.run_context_draft import (
    RunContextDraft,
    apply_overrides,
    build_extra_kwargs,
)
from care.screens.run_context import (
    RunContextModal,
    RunContextResult,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _saved_chain(*, entity_id: str = "agent-1") -> dict:
    return {
        "entity_id": entity_id,
        "metadata": {
            "care": {
                "task_description": "Run forecast",
                "context_files": [
                    {"path": "/tmp/a.txt", "sha256": "x", "size_bytes": 1},
                    {"path": "/tmp/b.txt", "sha256": "y", "size_bytes": 2},
                ],
                "display_name": "Storm Watcher",
            },
        },
    }


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class _ModalHost(App):
    def __init__(self, chain=None) -> None:
        super().__init__()
        self.chain = chain if chain is not None else _saved_chain()
        self.dismissed: list[RunContextResult] = []

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        def _on_dismiss(result):
            self.dismissed.append(result)

        self.push_screen(RunContextModal(self.chain), _on_dismiss)


def _modal(app: App) -> RunContextModal:
    s = app.screen_stack[-1]
    assert isinstance(s, RunContextModal)
    return s


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


class TestSeed:
    @pytest.mark.asyncio
    async def test_seed_from_chain(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.draft.task_description == "Run forecast"
            assert len(modal.draft.files) == 2
            task_area = modal.query_one("#run-context-task", TextArea)
            assert task_area.text == "Run forecast"

    @pytest.mark.asyncio
    async def test_seed_label_in_title(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.draft.source_name == "Storm Watcher"

    @pytest.mark.asyncio
    async def test_model_placeholder_shows_config_default(self):
        """The model field's placeholder shows the CURRENT default model
        (from config), so 'leave blank' is self-explanatory."""
        from care.config import CareConfig, MageConfig

        class _CfgHost(App):
            def compose(self) -> ComposeResult:
                yield from ()

            def on_mount(self) -> None:
                self.config = CareConfig(mage=MageConfig(model="gpt-4o-mini"))
                self.push_screen(RunContextModal(_saved_chain()))

        app = _CfgHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen_stack[-1]
            ph = modal.query_one("#run-context-model", Input).placeholder
            assert ph == "gpt-4o-mini (leave blank for the config default)"

    @pytest.mark.asyncio
    async def test_model_placeholder_falls_back_without_config(self):
        app = _ModalHost()  # bare host: no app.config
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            ph = modal.query_one("#run-context-model", Input).placeholder
            assert "leave blank for the config default" in ph


# ---------------------------------------------------------------------------
# Field edits
# ---------------------------------------------------------------------------


class TestEdits:
    @pytest.mark.asyncio
    async def test_task_edit_updates_draft(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#run-context-task", TextArea,
            ).load_text("Updated task")
            await pilot.pause()
            assert "Updated task" in modal.draft.task_description
            assert modal.draft.task_edited is True

    @pytest.mark.asyncio
    async def test_model_override_lands_on_draft(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#run-context-model", Input,
            ).value = "gpt-4o"
            await pilot.pause()
            assert modal.draft.model_override == "gpt-4o"

    @pytest.mark.asyncio
    async def test_base_url_and_api_key_land_on_draft(self):
        """The collapsible connection fields fold onto the draft, and
        editing the model doesn't clear them (all three sync together)."""
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#run-context-base-url", Input).value = "https://x/v1"
            await pilot.pause()
            modal.query_one("#run-context-api-key", Input).value = "sk-123"
            await pilot.pause()
            modal.query_one("#run-context-model", Input).value = "gpt-4o"
            await pilot.pause()
            assert modal.draft.base_url_override == "https://x/v1"
            assert modal.draft.api_key_override == "sk-123"
            assert modal.draft.model_override == "gpt-4o"

    @pytest.mark.asyncio
    async def test_connection_fields_collapsed_by_default(self):
        from textual.widgets import Collapsible

        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one(
                "#run-context-connection", Collapsible,
            ).collapsed is True

    @pytest.mark.asyncio
    async def test_streaming_checkbox_toggle(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.draft.streaming_enabled is True
            modal.query_one(
                "#run-context-streaming", Checkbox,
            ).value = False
            await pilot.pause()
            assert modal.draft.streaming_enabled is False

    @pytest.mark.asyncio
    async def test_drop_then_restore_file(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            # Multiple pauses so the file-row mount queue
            # drains.
            for _ in range(4):
                await pilot.pause()
            modal = _modal(app)
            assert len(modal.draft.files) == 2
            file_path = modal.draft.files[0].path
            button_id = modal._file_button_id(file_path)
            drop_btn = modal.query_one(f"#{button_id}", Button)
            drop_btn.press()
            for _ in range(4):
                await pilot.pause()
            assert any(
                f.status == "dropped" for f in modal.draft.files
            )
            # File rows were rebuilt; query again.
            restore_btn = modal.query_one(f"#{button_id}", Button)
            restore_btn.press()
            for _ in range(4):
                await pilot.pause()
            assert not any(
                f.status == "dropped" for f in modal.draft.files
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.asyncio
    async def test_blank_task_blocks_submit(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#run-context-task", TextArea,
            ).load_text("")
            await pilot.pause()
            assert modal.has_blocking_issues is True
            modal.query_one("#run-context-submit", Button).press()
            await pilot.pause()
            await pilot.pause()
            # No dismiss yet — blocking issue.
            assert app.dismissed == []


# ---------------------------------------------------------------------------
# Submit + cancel
# ---------------------------------------------------------------------------


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_dismisses_with_submitted_true(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#run-context-submit", Button).press()
            await pilot.pause()
            await pilot.pause()
            assert len(app.dismissed) == 1
            assert app.dismissed[0].submitted is True
            assert isinstance(app.dismissed[0].draft, RunContextDraft)

    @pytest.mark.asyncio
    async def test_cancel_dismisses_with_submitted_false(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.action_cancel()
            await pilot.pause()
            await pilot.pause()
            assert app.dismissed[0].submitted is False

    @pytest.mark.asyncio
    async def test_submit_label_flips_on_edit(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            submit = modal.query_one("#run-context-submit", Button)
            assert str(submit.label) == "Run"
            modal.query_one(
                "#run-context-task", TextArea,
            ).load_text("Something new")
            await pilot.pause()
            assert str(submit.label) == "Run (modified)"


# ---------------------------------------------------------------------------
# Result hooks into the data layer
# ---------------------------------------------------------------------------


class TestResultIntegration:
    @pytest.mark.asyncio
    async def test_result_draft_drives_build_extra_kwargs(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one(
                "#run-context-task", TextArea,
            ).load_text("New task")
            await pilot.pause()
            modal.query_one("#run-context-submit", Button).press()
            await pilot.pause()
            await pilot.pause()
            result = app.dismissed[0]
            kwargs = build_extra_kwargs(result.draft)
            assert kwargs["outer_context"] == "New task"

    def test_apply_overrides_returns_config_when_no_override(self):
        draft = RunContextDraft(source_entity_id="x", task_description="t")
        result = apply_overrides("placeholder", draft)
        assert result == "placeholder"


# ---------------------------------------------------------------------------
# Attach files (Browse… / @path) + required-missing banner
# ---------------------------------------------------------------------------


def _chain_with_files(files: list[dict], *, task: str = "do it") -> dict:
    return {
        "entity_id": "agent-f",
        "metadata": {
            "care": {
                "task_description": task,
                "context_files": files,
                "display_name": "Filer",
            },
        },
    }


class TestAttachFiles:
    @pytest.mark.asyncio
    async def test_browse_and_attach_widgets_present(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal.query_one("#run-context-attach", Input) is not None
            assert modal.query_one(
                "#run-context-attach-browse", Button,
            ) is not None

    @pytest.mark.asyncio
    async def test_browse_button_opens_file_picker(self):
        from care.screens.file_picker import FilePickerModal

        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            modal.query_one("#run-context-attach-browse", Button).press()
            for _ in range(4):
                await pilot.pause()
            assert isinstance(app.screen_stack[-1], FilePickerModal)

    @pytest.mark.asyncio
    async def test_attach_path_adds_row_and_draft_file(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("x,y")
        app = _ModalHost()
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            before = len(modal.draft.files)
            assert modal._attach_path(str(f)) is True
            for _ in range(4):
                await pilot.pause()
            assert len(modal.draft.files) == before + 1
            assert any(
                cf.path == str(f) and cf.status == "added"
                for cf in modal.draft.files
            )
            # A live file-row button was mounted for it.
            button_id = modal._file_button_id(str(f))
            assert modal.query_one(f"#{button_id}", Button) is not None

    @pytest.mark.asyncio
    async def test_attached_file_reaches_run_via_basename(self, tmp_path: Path):
        """End-to-end: an attached file lands in build_extra_kwargs under
        its basename so the chain's ${input.<basename>} ref resolves."""
        f = tmp_path / "report.txt"
        f.write_text("payload")
        app = _ModalHost(chain=_chain_with_files([]))
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            modal._attach_path(str(f))
            for _ in range(3):
                await pilot.pause()
            files = build_extra_kwargs(modal.draft)["files"]
            assert files["report.txt"] == "payload"
            assert files[str(f)] == "payload"

    @pytest.mark.asyncio
    async def test_attach_nonexistent_returns_false_and_flags(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            assert modal._attach_path("/no/such/file-xyz.txt") is False
            await pilot.pause()
            banner = modal.query_one("#run-context-required", Static)
            assert "/no/such/file-xyz.txt" in str(banner.render())

    @pytest.mark.asyncio
    async def test_input_submit_attaches_and_clears(self, tmp_path: Path):
        f = tmp_path / "in.txt"
        f.write_text("hi")
        app = _ModalHost(chain=_chain_with_files([]))
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            inp = modal.query_one("#run-context-attach", Input)
            inp.value = str(f)
            inp.focus()
            await pilot.pause()
            await pilot.press("enter")
            for _ in range(4):
                await pilot.pause()
            assert any(cf.path == str(f) for cf in modal.draft.files)
            assert inp.value == ""

    @pytest.mark.asyncio
    async def test_required_banner_shows_when_seeded_files_missing(
        self, tmp_path: Path,
    ):
        ghost = tmp_path / "ghost.txt"  # never created
        chain = _chain_with_files(
            [{"path": str(ghost), "sha256": "x", "size_bytes": 1}],
        )
        app = _ModalHost(chain=chain)
        async with app.run_test() as pilot:
            for _ in range(2):
                await pilot.pause()
            modal = _modal(app)
            banner = modal.query_one("#run-context-required", Static)
            assert str(banner.render()).strip() != ""

    @pytest.mark.asyncio
    async def test_required_banner_empty_when_files_present(
        self, tmp_path: Path,
    ):
        a = tmp_path / "a.txt"
        a.write_text("1")
        chain = _chain_with_files(
            [{"path": str(a), "sha256": "x", "size_bytes": 1}],
        )
        app = _ModalHost(chain=chain)
        async with app.run_test() as pilot:
            for _ in range(2):
                await pilot.pause()
            modal = _modal(app)
            banner = modal.query_one("#run-context-required", Static)
            assert str(banner.render()).strip() == ""

    @pytest.mark.asyncio
    async def test_attach_into_empty_draft_swaps_placeholder(
        self, tmp_path: Path,
    ):
        from textual.css.query import NoMatches

        f = tmp_path / "only.txt"
        f.write_text("z")
        app = _ModalHost(chain=_chain_with_files([]))
        async with app.run_test() as pilot:
            for _ in range(2):
                await pilot.pause()
            modal = _modal(app)
            # Empty-state placeholder visible up front.
            assert modal.query_one("#run-context-no-files", Static) is not None
            modal._attach_path(str(f))
            for _ in range(4):
                await pilot.pause()
            # Placeholder gone, real row mounted.
            with pytest.raises(NoMatches):
                modal.query_one("#run-context-no-files", Static)
            button_id = modal._file_button_id(str(f))
            assert modal.query_one(f"#{button_id}", Button) is not None

    @pytest.mark.asyncio
    async def test_attached_file_row_renders_name(self, tmp_path: Path):
        # Regression: a default Button's `border: tall` is 3 cells high; at
        # `.file-row` height 1 it clipped to the top border and the file name
        # vanished. `compact=True` makes the borderless row render its label.
        f = tmp_path / "report.txt"
        f.write_text("x")
        app = _ModalHost(chain=_chain_with_files([]))
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            modal._attach_path(str(f))
            for _ in range(4):
                await pilot.pause()
            btn = modal.query_one(
                f"#{modal._file_button_id(str(f))}", Button,
            )
            assert "report.txt" in str(btn.render())
            assert btn.outer_size.height == 1

    @pytest.mark.asyncio
    async def test_on_file_picked_round_trip(self, tmp_path: Path):
        # Simulates the FilePickerModal dismissing with a selection.
        f = tmp_path / "picked.txt"
        f.write_text("chosen")
        app = _ModalHost(chain=_chain_with_files([]))
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            modal._on_file_picked(f)  # picker returns a Path
            for _ in range(4):
                await pilot.pause()
            assert any(cf.path == str(f) for cf in modal.draft.files)
            button_id = modal._file_button_id(str(f))
            assert modal.query_one(f"#{button_id}", Button) is not None
            # A cancelled picker (None) is a no-op.
            modal._on_file_picked(None)
            assert sum(1 for cf in modal.draft.files) == 1

    @pytest.mark.asyncio
    async def test_drop_added_file_removes_its_row(self, tmp_path: Path):
        from textual.css.query import NoMatches

        f = tmp_path / "tmp.txt"
        f.write_text("x")
        app = _ModalHost(chain=_chain_with_files([]))
        async with app.run_test() as pilot:
            for _ in range(3):
                await pilot.pause()
            modal = _modal(app)
            modal._attach_path(str(f))
            for _ in range(4):
                await pilot.pause()
            button_id = modal._file_button_id(str(f))
            modal.query_one(f"#{button_id}", Button).press()
            for _ in range(4):
                await pilot.pause()
            # An added file dropped vanishes from the draft AND the DOM.
            assert all(cf.path != str(f) for cf in modal.draft.files)
            with pytest.raises(NoMatches):
                modal.query_one(f"#{button_id}", Button)

    @pytest.mark.asyncio
    async def test_banner_clears_after_attaching_missing_file(
        self, tmp_path: Path,
    ):
        # A required file that's missing → banner shows; attach a
        # same-basename replacement → banner clears.
        missing = tmp_path / "needed.txt"  # never created
        chain = _chain_with_files(
            [{"path": str(missing), "sha256": "x", "size_bytes": 1}],
        )
        app = _ModalHost(chain=chain)
        async with app.run_test() as pilot:
            for _ in range(2):
                await pilot.pause()
            modal = _modal(app)
            banner = modal.query_one("#run-context-required", Static)
            assert str(banner.render()).strip() != ""
            # Provide a real file with the same basename.
            real = tmp_path / "sub" / "needed.txt"
            real.parent.mkdir()
            real.write_text("here")
            # Drop the missing one, attach the real one.
            missing_btn = modal._file_button_id(str(missing))
            modal.query_one(f"#{missing_btn}", Button).press()
            for _ in range(3):
                await pilot.pause()
            modal._attach_path(str(real))
            for _ in range(3):
                await pilot.pause()
            assert str(banner.render()).strip() == ""

    @pytest.mark.asyncio
    async def test_doc_needed_banner_for_document_chain(self, tmp_path: Path):
        # A chain with a doc-reading skill step + no attached file → proactive
        # "attach a document" banner; it clears once a file is attached.
        chain = {
            "entity_id": "d",
            "metadata": {"care": {"task_description": "t", "context_files": []}},
            "steps": [{
                "number": 1, "step_type": "agent_skill",
                "title": "Extract", "aim": "read the docx",
                "step_config": {
                    "skill": "github://anthropics/skills/skills/docx@main",
                    "task": "Extract text from the provided DOCX",
                    "input_mapping": {},
                },
            }],
        }
        app = _ModalHost(chain=chain)
        async with app.run_test() as pilot:
            for _ in range(2):
                await pilot.pause()
            modal = _modal(app)
            assert modal._reads_doc is True
            banner = modal.query_one("#run-context-required", Static)
            assert str(banner.render()).strip() != ""
            f = tmp_path / "doc.txt"
            f.write_text("x")
            modal._attach_path(str(f))
            for _ in range(3):
                await pilot.pause()
            assert str(banner.render()).strip() == ""

    @pytest.mark.asyncio
    async def test_escape_closes_modal_with_attach_input_focused(self):
        app = _ModalHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = _modal(app)
            inp = modal.query_one("#run-context-attach", Input)
            inp.value = "some text"
            inp.focus()
            await pilot.pause()
            await pilot.press("escape")
            for _ in range(2):
                await pilot.pause()
            assert app.dismissed and app.dismissed[-1].submitted is False


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_screens_re_exports(self):
        from care.screens import RunContextModal as M
        from care.screens import RunContextResult as R

        assert M is RunContextModal
        assert R is RunContextResult
