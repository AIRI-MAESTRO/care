"""Tests for the ``care`` CLI router (TODO §9 P2 + §8 P1 catalog CLI).

Tests work against the parser + handlers directly so we don't
spawn a subprocess. Output is captured via ``io.StringIO`` —
each handler accepts injectable ``stdout`` / ``stderr`` streams.

The full ``main()`` wrapper is also exercised so the
``pyproject.toml`` entry-point path stays covered, but without
running the TUI (no subcommand → smoke test argv parsing only).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from care import cli as cli_mod
from care.cli import (
    _build_parser,
    _cmd_catalog,
    _cmd_diff,
    _cmd_dataset,
    _cmd_deploy,
    _cmd_deployments,
    _cmd_evolve,
    _cmd_export,
    _cmd_favourite,
    _cmd_forget,
    _cmd_metrics,
    _cmd_notes,
    _cmd_remember,
    _cmd_generate,
    _cmd_promote,
    _cmd_revise,
    _cmd_rollback,
    _cmd_versions,
    _cmd_help,
    _cmd_init,
    _cmd_lineage,
    _cmd_marketplace,
    _cmd_import,
    _cmd_memory_history,
    _cmd_memory_ls,
    _cmd_memory_show,
    _cmd_replay,
    _cmd_run,
    _cmd_search,
    _cmd_validate,
    _read_file_inputs,
    _warn_missing_context_files,
    main,
)


def _run_handler(handler, ns, stdout=None, stderr=None) -> tuple[int, str, str]:
    """Call a handler with fresh string streams; return code + io."""
    out = stdout or io.StringIO()
    err = stderr or io.StringIO()
    code = handler(ns, out, err)
    return code, out.getvalue(), err.getvalue()


def _parse(argv: list[str]):
    """Parse argv and return the namespace ready for handler call."""
    return _build_parser().parse_args(argv)


def _valid_chain_dict() -> dict:
    return {
        "task_description": "demo",
        "steps": [
            {"number": 1, "title": "first", "step_type": "llm", "aim": "hi"},
        ],
    }


def _write_chain(path: Path, chain: dict | None = None) -> Path:
    chain = chain or _valid_chain_dict()
    path.write_text(json.dumps(chain), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# care catalog
# ---------------------------------------------------------------------------


class TestCatalogCommand:
    def test_text_output_empty(self, tmp_path: Path):
        ns = _parse(["catalog"])
        code, out, err = _run_handler(_cmd_catalog, ns)
        assert code == 0
        assert "no entries" in out

    def test_text_output_with_tool(self, tmp_path: Path):
        (tmp_path / "weather.py").write_text('"""Fetch weather"""\n')
        ns = _parse(["catalog", "--tools", str(tmp_path)])
        code, out, err = _run_handler(_cmd_catalog, ns)
        assert code == 0
        assert "# tool" in out
        assert "weather" in out
        assert "Fetch weather" in out

    def test_json_output(self, tmp_path: Path):
        (tmp_path / "weather.py").write_text('"""Fetch weather"""\n')
        ns = _parse(["catalog", "--tools", str(tmp_path), "--json"])
        code, out, err = _run_handler(_cmd_catalog, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["entries"][0]["kind"] == "tool"
        assert payload["entries"][0]["name"] == "weather"
        assert payload["errors"] == []

    def test_kind_filter(self, tmp_path: Path):
        # One tool + one MCP. --kind=tool drops the MCP entry.
        (tmp_path / "t.py").write_text('"""tool"""\n')
        mcp = tmp_path / "mcp.toml"
        mcp.write_text('[servers.x]\ncommand = "y"\n')
        ns = _parse(
            [
                "catalog",
                "--tools",
                str(tmp_path),
                "--mcp-config",
                str(mcp),
                "--kind",
                "tool",
                "--json",
            ]
        )
        code, out, err = _run_handler(_cmd_catalog, ns)
        payload = json.loads(out)
        kinds = {e["kind"] for e in payload["entries"]}
        assert kinds == {"tool"}

    def test_warnings_go_to_stderr(self, tmp_path: Path):
        # File-instead-of-dir → catalog records a warning.
        f = tmp_path / "skills-not-a-dir"
        f.write_text("not a dir\n")
        ns = _parse(["catalog", "--skills", str(f)])
        code, out, err = _run_handler(_cmd_catalog, ns)
        assert code == 0
        assert "catalog warnings" in err
        assert "not a directory" in err


# ---------------------------------------------------------------------------
# care validate
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_valid_chain_exits_zero(self, tmp_path: Path):
        path = _write_chain(tmp_path / "ok.json")
        ns = _parse(["validate", str(path)])
        code, out, err = _run_handler(_cmd_validate, ns)
        assert code == 0
        assert "ok" in out or "preflight skipped" in out

    def test_invalid_chain_exits_one(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{ not valid", encoding="utf-8")
        ns = _parse(["validate", str(path)])
        code, out, err = _run_handler(_cmd_validate, ns)
        assert code == 1
        assert "failed to parse" in out

    def test_missing_file_exits_two(self, tmp_path: Path):
        ns = _parse(["validate", str(tmp_path / "nope.json")])
        code, out, err = _run_handler(_cmd_validate, ns)
        assert code == 2
        assert "read failed" in err

    def test_json_output(self, tmp_path: Path):
        path = _write_chain(tmp_path / "ok.json")
        ns = _parse(["validate", str(path), "--json"])
        code, out, err = _run_handler(_cmd_validate, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["parsed"] is True
        assert "missing_tools" in payload
        assert "required_tools" in payload

    def test_json_output_on_failure_still_emits_payload(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{ broken", encoding="utf-8")
        ns = _parse(["validate", str(path), "--json"])
        code, out, err = _run_handler(_cmd_validate, ns)
        assert code == 1
        payload = json.loads(out)
        assert payload["parsed"] is False
        assert len(payload["parse_errors"]) >= 1


# ---------------------------------------------------------------------------
# care import
# ---------------------------------------------------------------------------


class TestImportCommand:
    def test_dry_run_by_default(self, tmp_path: Path):
        _write_chain(tmp_path / "a.json")
        _write_chain(tmp_path / "b.json")
        ns = _parse(["import", str(tmp_path / "*.json")])
        code, out, err = _run_handler(_cmd_import, ns)
        assert code == 0
        assert "2 validated" in out
        assert "0 imported" in out

    def test_dry_run_failure_returns_one(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        ns = _parse(["import", str(bad)])
        code, out, err = _run_handler(_cmd_import, ns)
        assert code == 1
        assert "1 failed" in out

    def test_apply_saves_to_memory(self, tmp_path: Path):
        _write_chain(tmp_path / "a.json")
        saved: list = []

        class _Mem:
            def save_chain(self, chain, **kw):
                saved.append((chain, kw))
                return "ent-1"

        _install_memory_stub(_Mem())
        try:
            ns = _parse(["import", str(tmp_path / "a.json"), "--apply"])
            code, out, err = _run_handler(_cmd_import, ns)
        finally:
            _restore_memory_stub()
        assert code == 0
        assert "1 imported" in out
        assert len(saved) == 1

    def test_apply_without_memory_configured_returns_two(self, tmp_path: Path):
        _write_chain(tmp_path / "a.json")

        def _boom():
            raise cli_mod.CliMemoryError("no memory configured")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        try:
            ns = _parse(["import", str(tmp_path / "a.json"), "--apply"])
            code, out, err = _run_handler(_cmd_import, ns)
        finally:
            _restore_memory_stub()
        assert code == 2
        assert "no memory configured" in err

    def test_no_matches_succeeds(self, tmp_path: Path):
        ns = _parse(["import", str(tmp_path / "*.no-such.json")])
        code, out, err = _run_handler(_cmd_import, ns)
        assert code == 0
        assert "0 imported, 0 validated, 0 failed" in out


class TestLtmCommands:
    def test_remember_saves(self, monkeypatch):
        import care.memory_ltm as ltm_mod

        monkeypatch.setattr(ltm_mod, "build_ltm", lambda cfg: object())
        monkeypatch.setattr(ltm_mod, "ltm_session_id", lambda cfg: "s")
        monkeypatch.setattr(
            ltm_mod, "recall_digest", lambda *a, **k: ""
        )
        monkeypatch.setattr(
            ltm_mod,
            "remember_text",
            lambda *a, **k: [{"key": "k", "value": "remember v"}],
        )
        ns = _parse(["remember", "the", "user", "likes", "tea"])
        code, out, err = _run_handler(_cmd_remember, ns)
        assert code == 0
        assert "remember v" in out or "k" in out

    def test_remember_ltm_disabled(self, monkeypatch):
        import care.memory_ltm as ltm_mod

        monkeypatch.setattr(ltm_mod, "build_ltm", lambda cfg: None)
        ns = _parse(["remember", "x"])
        code, out, err = _run_handler(_cmd_remember, ns)
        assert code == 2
        assert "disabled" in err

    def test_notes_shows_digest(self, monkeypatch):
        import care.memory_ltm as ltm_mod

        monkeypatch.setattr(ltm_mod, "build_ltm", lambda cfg: object())
        monkeypatch.setattr(ltm_mod, "ltm_session_id", lambda cfg: "s")
        monkeypatch.setattr(
            ltm_mod, "recall_digest", lambda *a, **k: "• likes tea"
        )
        ns = _parse(["notes"])
        code, out, err = _run_handler(_cmd_notes, ns)
        assert code == 0
        assert "likes tea" in out

    def test_notes_disabled(self, monkeypatch):
        import care.memory_ltm as ltm_mod

        monkeypatch.setattr(ltm_mod, "build_ltm", lambda cfg: None)
        ns = _parse(["notes"])
        code, out, err = _run_handler(_cmd_notes, ns)
        assert code == 0
        assert "disabled" in out


class TestDatasetCommands:
    def teardown_method(self):
        _restore_memory_stub()

    def _mem(self, rows=None):
        class _Mem:
            def __init__(self):
                self.saved = []
                self._rows = rows or []

            def list_entities(self, **kwargs):
                return self._rows

            def save_memory_card(self, content, *, name, tags, when_to_use=None):
                self.saved.append({"content": content, "tags": tags})
                return "card-1"

            def get_chain(self, chain_id, **kwargs):
                return {"steps": [{"id": "a"}]}

        return _Mem()

    def test_list(self):
        rows = [
            {
                "entity_id": "m1",
                "tags": ["dataset-entry:c1"],
                "content": {"task": "do thing", "expected": "ok", "status": "pass"},
            }
        ]
        _install_memory_stub(self._mem(rows))
        ns = _parse(["dataset", "list", "c1"])
        code, out, err = _run_handler(_cmd_dataset, ns)
        assert code == 0
        assert "do thing" in out and "[pass]" in out

    def test_add(self):
        mem = self._mem()
        _install_memory_stub(mem)
        ns = _parse(
            ["dataset", "add", "c1", "do x", "--expected", "y", "--rubric", "r"]
        )
        code, out, err = _run_handler(_cmd_dataset, ns)
        assert code == 0
        assert "added dataset entry: card-1" in out
        assert mem.saved and "dataset-entry:c1" in mem.saved[0]["tags"]

    def test_export(self, tmp_path: Path):
        rows = [
            {
                "entity_id": "m1",
                "tags": ["dataset-entry:c1"],
                "content": {"task": "t", "expected": "e", "status": "pending"},
            }
        ]
        _install_memory_stub(self._mem(rows))
        out_path = tmp_path / "ds.jsonl"
        ns = _parse(["dataset", "export", "c1", str(out_path)])
        code, out, err = _run_handler(_cmd_dataset, ns)
        assert code == 0
        assert out_path.exists()
        assert "exported 1 entries" in out

    def test_run_substring_scored(self, monkeypatch):
        rows = [
            {
                "entity_id": "m1",
                "tags": ["dataset-entry:c1"],
                "content": {"task": "say foo", "expected": "foo"},
            },
            {
                "entity_id": "m2",
                "tags": ["dataset-entry:c1"],
                "content": {"task": "say bar", "expected": "zzz"},
            },
        ]
        _install_memory_stub(self._mem(rows))

        class _Result:
            def __init__(self, ans):
                self.final_answer = ans
                self.success = True

        async def _executor(chain_dict, *, task=None, inputs=None):
            return _Result("the foo answer")  # passes entry 1, fails entry 2

        monkeypatch.setattr(cli_mod, "_build_carl_executor", lambda: _executor)
        ns = _parse(["dataset", "run", "c1"])
        code, out, err = _run_handler(_cmd_dataset, ns)
        # 1 of 2 passes → non-zero exit.
        assert code == 1
        assert "score: 1/2 passed" in out


class TestHubCommands:
    def teardown_method(self):
        cli_mod._BUILD_HUB_OVERRIDE = None

    @staticmethod
    def _install_hub(hub):
        cli_mod._BUILD_HUB_OVERRIDE = lambda: hub

    def test_deployments_lists(self):
        from types import SimpleNamespace

        class _Hub:
            async def list_deployments(self):
                return [
                    SimpleNamespace(
                        name="a",
                        url="http://h/agents/a",
                        version="1",
                        runs=3,
                        ready=True,
                        ready_reason="",
                    )
                ]

        self._install_hub(_Hub())
        ns = _parse(["deployments"])
        code, out, err = _run_handler(_cmd_deployments, ns)
        assert code == 0
        assert "a" in out and "runs=3" in out

    def test_deploy_posts_spec(self):
        from types import SimpleNamespace

        captured: dict = {}

        class _Hub:
            async def deploy(self, spec):
                captured.update(spec)
                return SimpleNamespace(name=spec["name"])

            def agent_url(self, name):
                return f"http://h/agents/{name}"

        self._install_hub(_Hub())
        ns = _parse(["deploy", "c1", "--name", "myagent", "--channel", "stable"])
        code, out, err = _run_handler(_cmd_deploy, ns)
        assert code == 0
        assert captured["entity_id"] == "c1"
        assert captured["name"] == "myagent"
        assert captured["channel"] == "stable"
        assert "api_key" in captured
        assert "deployed: myagent" in out

    def test_deploy_hub_down_returns_two(self):
        from care.runtime.agent_hub import HubUnavailableError

        class _Hub:
            async def deploy(self, spec):
                raise HubUnavailableError("down")

        self._install_hub(_Hub())
        ns = _parse(["deploy", "c1"])
        code, out, err = _run_handler(_cmd_deploy, ns)
        assert code == 2
        assert "not running" in err

    def test_metrics_named_agent(self):
        class _Hub:
            async def agent_metrics(self, name):
                return {"runs": 5, "cost_usd": 0.2}

        self._install_hub(_Hub())
        ns = _parse(["metrics", "a"])
        code, out, err = _run_handler(_cmd_metrics, ns)
        assert code == 0
        assert "runs" in out

    def test_metrics_summary_over_all(self):
        from types import SimpleNamespace

        class _Hub:
            async def list_deployments(self):
                return [
                    SimpleNamespace(name="a", runs=2, ready=True),
                    SimpleNamespace(name="b", runs=3, ready=True),
                ]

        self._install_hub(_Hub())
        ns = _parse(["metrics"])
        code, out, err = _run_handler(_cmd_metrics, ns)
        assert code == 0
        assert "total runs: 5" in out


class TestExportCommand:
    def test_export_writes_bundle(self, tmp_path: Path):
        class _Client:
            def get_chain_dict(self, entity_id, channel):
                return {
                    "chain": {"steps": [{"id": "a"}]},
                    "meta": {"name": f"C-{entity_id}"},
                }

        class _Mem:
            client = _Client()

        _install_memory_stub(_Mem())
        try:
            out_path = tmp_path / "bundle.tar.gz"
            ns = _parse(["export", str(out_path), "c1", "c2"])
            code, out, err = _run_handler(_cmd_export, ns)
        finally:
            _restore_memory_stub()
        assert code == 0
        assert "exported 2 chain(s)" in out
        assert out_path.exists()

    def test_export_without_memory_returns_two(self, tmp_path: Path):
        def _boom():
            raise cli_mod.CliMemoryError("no mem configured")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        try:
            ns = _parse(["export", str(tmp_path / "b.tar.gz"), "c1"])
            code, out, err = _run_handler(_cmd_export, ns)
        finally:
            _restore_memory_stub()
        assert code == 2
        assert "no mem configured" in err


class TestReviseCommand:
    def teardown_method(self):
        _restore_mage_stub()
        _restore_memory_stub()

    @staticmethod
    def _gen_with(result):
        class _Gen:
            calls: list = []

            async def edit(
                self,
                instruction,
                *,
                chain=None,
                entity_id=None,
                channel="latest",
                save=False,
                cancel=None,
            ):
                _Gen.calls.append((instruction, entity_id, channel, save))
                return result

        _Gen.calls = []
        return _Gen()

    @staticmethod
    def _result_with_changes():
        from types import SimpleNamespace

        return SimpleNamespace(
            needs_disambiguation=False,
            edits=[
                SimpleNamespace(
                    op="modify", target_step_number=1, rationale="tighten"
                )
            ],
            summary="Tighten step 1",
            before_chain_dict={"steps": [{"id": "a"}]},
            chain_dict={"steps": [{"id": "a", "x": 1}]},
        )

    def test_preview_only_without_yes(self):
        _install_mage_stub(self._gen_with(self._result_with_changes()))
        ns = _parse(["revise", "c1", "tighten step 1"])
        code, out, err = _run_handler(_cmd_revise, ns)
        assert code == 0
        assert "Planned edit" in out
        assert "Preview only" in out

    def test_yes_saves_new_version(self):
        _install_mage_stub(self._gen_with(self._result_with_changes()))
        saved = []

        class _Mem:
            def get_entity(self, eid):
                return {"display_name": "My Chain"}

            def save_chain(self, chain, **kw):
                saved.append((chain, kw))
                return "c1-v2"

        _install_memory_stub(_Mem())
        ns = _parse(["revise", "c1", "tighten step 1", "--yes"])
        code, out, err = _run_handler(_cmd_revise, ns)
        assert code == 0
        assert "saved new version: c1-v2" in out
        assert len(saved) == 1
        assert saved[0][1]["entity_id"] == "c1"
        assert saved[0][1]["name"] == "My Chain"

    def test_disambiguation_returns_one(self):
        from types import SimpleNamespace

        result = SimpleNamespace(
            needs_disambiguation=True,
            candidates=[SimpleNamespace(entity_id="x", name="X")],
        )
        _install_mage_stub(self._gen_with(result))
        ns = _parse(["revise", "c1", "do thing"])
        code, out, err = _run_handler(_cmd_revise, ns)
        assert code == 1
        assert "ambiguous" in err

    def test_no_changes_returns_zero(self):
        from types import SimpleNamespace

        result = SimpleNamespace(
            needs_disambiguation=False,
            edits=[],
            summary="",
            before_chain_dict={"steps": []},
            chain_dict={"steps": []},
        )
        _install_mage_stub(self._gen_with(result))
        ns = _parse(["revise", "c1", "noop"])
        code, out, err = _run_handler(_cmd_revise, ns)
        assert code == 0
        assert "no changes" in out.lower()


class TestVersionChannelCommands:
    def test_versions_lists_with_channel_annotation(self):
        from types import SimpleNamespace

        class _Mem:
            def list_versions(self, eid, *, entity_type="chain", limit=20):
                return [
                    SimpleNamespace(
                        version_number=2,
                        version_id="v-2",
                        created_at="2026-06-01T00:00:00Z",
                        change_summary="evolved",
                    ),
                    SimpleNamespace(
                        version_number=1,
                        version_id="v-1",
                        created_at="2026-05-01T00:00:00Z",
                        change_summary="seed",
                    ),
                ]

            def get_entity(self, eid):
                return {"channels": {"stable": "v-1", "latest": "v-2"}}

        _install_memory_stub(_Mem())
        try:
            ns = _parse(["versions", "c1"])
            code, out, err = _run_handler(_cmd_versions, ns)
        finally:
            _restore_memory_stub()
        assert code == 0
        assert "v2" in out and "v1" in out
        assert "stable" in out and "latest" in out

    def test_rollback_pins_channel(self):
        calls = []

        class _Mem:
            def pin_channel(self, eid, channel, vid, *, entity_type="chain"):
                calls.append((eid, channel, vid, entity_type))
                return {"ok": True}

        _install_memory_stub(_Mem())
        try:
            ns = _parse(["rollback", "c1", "--to", "v-1"])
            code, out, err = _run_handler(_cmd_rollback, ns)
        finally:
            _restore_memory_stub()
        assert code == 0
        assert calls == [("c1", "stable", "v-1", "chain")]

    def test_promote_moves_channel_pointer(self):
        calls = []

        class _Mem:
            def promote(self, eid, *, from_channel, to_channel, entity_type):
                calls.append((eid, from_channel, to_channel, entity_type))
                return {"ok": True}

        _install_memory_stub(_Mem())
        try:
            ns = _parse(["promote", "c1"])
            code, out, err = _run_handler(_cmd_promote, ns)
        finally:
            _restore_memory_stub()
        assert code == 0
        assert calls == [("c1", "latest", "stable", "chain")]

    def test_forget_previews_without_force(self):
        # No memory stub needed — preview must not touch Memory.
        ns = _parse(["forget", "c1"])
        code, out, err = _run_handler(_cmd_forget, ns)
        assert code == 0
        assert "Would delete" in out
        assert "--force" in out

    def test_forget_force_deletes(self):
        calls = []

        class _Mem:
            def delete_entity(self, eid, *, entity_type="chain"):
                calls.append((eid, entity_type))
                return True

        _install_memory_stub(_Mem())
        try:
            ns = _parse(["forget", "c1", "--force"])
            code, out, err = _run_handler(_cmd_forget, ns)
        finally:
            _restore_memory_stub()
        assert code == 0
        assert calls == [("c1", "chain")]
        assert "forgot" in out


# ---------------------------------------------------------------------------
# main() router
# ---------------------------------------------------------------------------


class TestMainRouter:
    def test_help_returns_zero(self, capsys):
        # `care --help` should exit 0 without trying to launch
        # the TUI.
        code = main(["--help"])
        assert code == 0
        captured = capsys.readouterr()
        assert "CARE" in captured.out

    def test_validate_subcommand_via_main(
        self, tmp_path: Path, capsys
    ):
        path = _write_chain(tmp_path / "ok.json")
        code = main(["validate", str(path)])
        assert code == 0
        captured = capsys.readouterr()
        assert "ok" in captured.out or "preflight skipped" in captured.out

    def test_unknown_subcommand_returns_nonzero(self, capsys):
        code = main(["totally-bogus-cmd"])
        assert code != 0

    def test_catalog_subcommand_via_main(
        self, tmp_path: Path, capsys
    ):
        (tmp_path / "t.py").write_text('"""hi"""\n')
        code = main(["catalog", "--tools", str(tmp_path)])
        assert code == 0
        captured = capsys.readouterr()
        assert "# tool" in captured.out

    def test_import_dry_run_via_main(self, tmp_path: Path, capsys):
        _write_chain(tmp_path / "a.json")
        code = main(["import", str(tmp_path / "*.json")])
        assert code == 0
        captured = capsys.readouterr()
        assert "validated" in captured.out

    def test_help_commands_parity_table(self, capsys):
        code = main(["help", "--commands"])
        assert code == 0
        out = capsys.readouterr().out
        assert "parity" in out.lower()
        assert "care revise" in out and "/revise" in out
        assert "care export" in out
        assert "care dataset" in out and "care deploy" in out
        # TUI-only section present.
        assert "/upload" in out


# ---------------------------------------------------------------------------
# care memory ls
# ---------------------------------------------------------------------------


class _StubMemory:
    """Test double for the CareMemory facade — captures kwargs +
    returns the canned listing supplied at construction time."""

    def __init__(
        self,
        *,
        rows=None,
        raise_on_list=False,
        chain=None,
        raise_on_get_chain=False,
        saved_entity_id="ent-saved",
        raise_on_save=False,
        saved_card_entity_id="card-saved",
        raise_on_save_card=False,
    ):
        self._rows = list(rows or [])
        self._raise = raise_on_list
        self._chain = chain
        self._raise_get_chain = raise_on_get_chain
        self._saved_entity_id = saved_entity_id
        self._raise_save = raise_on_save
        self._saved_card_entity_id = saved_card_entity_id
        self._raise_save_card = raise_on_save_card
        self.list_calls: list[dict] = []
        self.get_chain_calls: list[tuple[str, str]] = []
        self.save_chain_calls: list[dict] = []
        self.save_memory_card_calls: list[dict] = []
        self.get_entity_calls: list[dict] = []
        self.entity_payload: dict | None = None
        self._raise_get_entity = False

    def list_entities(self, **kw):
        self.list_calls.append(kw)
        if self._raise:
            raise RuntimeError("memory-down")
        return list(self._rows)

    def get_chain(self, entity_id, *, channel="latest"):
        self.get_chain_calls.append((entity_id, channel))
        if self._raise_get_chain:
            raise RuntimeError("fetch-down")
        if self._chain is None:
            raise RuntimeError(f"unknown chain {entity_id!r}")
        return dict(self._chain)

    def save_chain(self, chain, **kw):
        self.save_chain_calls.append({"chain": chain, **kw})
        if self._raise_save:
            raise RuntimeError("save-down")
        return self._saved_entity_id

    def save_memory_card(self, card, **kw):
        self.save_memory_card_calls.append({"card": card, **kw})
        if self._raise_save_card:
            raise RuntimeError("save-card-down")
        return self._saved_card_entity_id

    def get_entity(self, entity_id, *, entity_type, channel="latest"):
        self.get_entity_calls.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "channel": channel,
        })
        if self._raise_get_entity:
            raise RuntimeError("show-down")
        if self.entity_payload is None:
            raise RuntimeError(f"unknown entity {entity_id!r}")
        return dict(self.entity_payload)

    def find_capability_matches(
        self, query, *, top_k=10, namespace=None, deep=False,
    ):
        # `_StubMemory` doesn't carry a `.client` attribute by
        # default, so `_cmd_marketplace`'s `getattr(memory,
        # "client", None) or memory` falls through to the
        # memory itself — this method then receives the call.
        if not hasattr(self, "_market_calls"):
            self._market_calls = []
        self._market_calls.append({
            "query": query,
            "top_k": top_k,
            "namespace": namespace,
            "deep": deep,
        })
        if getattr(self, "_raise_market", False):
            raise RuntimeError("market-down")
        return list(getattr(self, "market_hits", []) or [])

    def mark_favourite(self, entity_id, *, entity_type, value=True):
        if not hasattr(self, "_fav_calls"):
            self._fav_calls = []
        self._fav_calls.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "value": value,
        })
        if getattr(self, "_raise_favourite", False):
            raise RuntimeError("fav-down")
        return getattr(self, "favourite_response", None) or {
            "entity_id": entity_id,
            "display_name": "Stub Entity",
            "favourite": bool(value),
        }

    def search(self, query, *, entity_type=None, search_type="bm25", top_k=10):
        if not hasattr(self, "_search_calls"):
            self._search_calls = []
        self._search_calls.append({
            "query": query,
            "entity_type": entity_type,
            "search_type": search_type,
            "top_k": top_k,
        })
        if getattr(self, "_raise_search", False):
            raise RuntimeError("search-down")
        return list(getattr(self, "search_hits", []) or [])

    def get_chain_lineage(self, entity_id, *, channel="latest", version_id=None, max_depth=10):
        if not hasattr(self, "_lineage_calls"):
            self._lineage_calls = []
        self._lineage_calls.append({
            "entity_id": entity_id,
            "channel": channel,
            "version_id": version_id,
            "max_depth": max_depth,
        })
        if getattr(self, "_raise_lineage", False):
            raise RuntimeError("lineage-down")
        return getattr(self, "lineage_response", None)


def _install_memory_stub(stub):
    """Plug a stub through the test-injection hook."""
    cli_mod._BUILD_MEMORY_OVERRIDE = lambda: stub


def _restore_memory_stub():
    cli_mod._BUILD_MEMORY_OVERRIDE = None


def _memory_row(
    *,
    entity_id: str,
    name: str = "",
    display_name: str | None = None,
    runs: int = 0,
    favourite: bool = False,
    tags=None,
) -> dict:
    return {
        "entity_id": entity_id,
        "display_name": display_name if display_name is not None else name,
        "meta": {"name": name, "tags": list(tags or [])},
        "run_count": runs,
        "favourite": favourite,
    }


class TestMemoryLsCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_text_output_with_rows(self):
        stub = _StubMemory(rows=[
            _memory_row(
                entity_id="ent-abc-1234567890",
                name="Storm Watcher",
                runs=5,
                favourite=True,
                tags=["weather", "shared"],
            ),
            _memory_row(
                entity_id="ent-def-0987654321",
                name="Quiet Watcher",
            ),
        ])
        _install_memory_stub(stub)
        ns = _parse(["memory", "ls"])
        code, out, err = _run_handler(_cmd_memory_ls, ns)
        assert code == 0
        assert "Storm Watcher" in out
        assert "Quiet Watcher" in out
        assert "★" in out  # favourite marker
        assert "runs=5" in out
        assert "[weather, shared]" in out
        assert err == ""

    def test_text_output_empty(self):
        _install_memory_stub(_StubMemory(rows=[]))
        ns = _parse(["memory", "ls"])
        code, out, err = _run_handler(_cmd_memory_ls, ns)
        assert code == 0
        assert "no entities" in out

    def test_json_output(self):
        stub = _StubMemory(rows=[
            _memory_row(entity_id="ent-1", name="Hello"),
        ])
        _install_memory_stub(stub)
        ns = _parse(["memory", "ls", "--json"])
        code, out, err = _run_handler(_cmd_memory_ls, ns)
        assert code == 0
        payload = json.loads(out)
        assert "entities" in payload
        assert payload["entities"][0]["entity_id"] == "ent-1"

    def test_filters_forwarded_to_memory(self):
        stub = _StubMemory(rows=[])
        _install_memory_stub(stub)
        ns = _parse([
            "memory", "ls",
            "--entity-type", "agent_skill",
            "--limit", "5",
            "--channel", "stable",
            "--namespace", "team-a",
            "--tag", "weather",
            "--tag", "shared",
            "--q", "storm",
            "--favourites-only",
        ])
        code, out, err = _run_handler(_cmd_memory_ls, ns)
        assert code == 0
        assert len(stub.list_calls) == 1
        call = stub.list_calls[0]
        assert call["entity_type"] == "agent_skill"
        assert call["limit"] == 5
        assert call["channel"] == "stable"
        assert call["namespace"] == "team-a"
        assert call["tags"] == ["weather", "shared"]
        assert call["q"] == "storm"
        assert call["favourites_only"] is True

    def test_default_entity_type_is_chain(self):
        stub = _StubMemory(rows=[])
        _install_memory_stub(stub)
        ns = _parse(["memory", "ls"])
        _run_handler(_cmd_memory_ls, ns)
        assert stub.list_calls[0]["entity_type"] == "chain"

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["memory", "ls"])
        code, out, err = _run_handler(_cmd_memory_ls, ns)
        assert code == 2
        assert "config not found" in err
        assert out == ""

    def test_list_failure_returns_two(self):
        _install_memory_stub(_StubMemory(rows=[], raise_on_list=True))
        ns = _parse(["memory", "ls"])
        code, out, err = _run_handler(_cmd_memory_ls, ns)
        assert code == 2
        assert "lookup failed" in err
        assert "memory-down" in err

    def test_memory_subcommand_via_main(self, capsys):
        _install_memory_stub(_StubMemory(rows=[
            _memory_row(entity_id="x", name="Y"),
        ]))
        code = main(["memory", "ls"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        assert "Y" in captured.out


# ---------------------------------------------------------------------------
# care memory history
# ---------------------------------------------------------------------------


def _agent_run_card(
    *,
    entity_id: str,
    run_id: str,
    agent_id: str = "agent-1",
    finished_at: str = "2026-05-19T10:00:00+00:00",
    duration: float = 2.5,
    steps: int = 3,
    tokens: int = 100,
    success: bool = True,
    task: str = "demo task",
) -> dict:
    status_label = "success" if success else "failed"
    metrics: dict = {
        "duration_seconds": duration,
        "step_count": steps,
        "total_tokens": tokens,
        "exit_status": status_label,
    }
    if not success:
        metrics["error_message"] = "boom"
    return {
        "entity_type": "memory_card",
        "entity_id": entity_id,
        "version_id": "v-1",
        "channel": "latest",
        "etag": "etag",
        "meta": {"tags": ["agent_run", f"agent:{agent_id}", f"status:{status_label}"]},
        "content": {
            "category": "agent_run",
            "task_description": task,
            "description": "Run digest",
            "usage": {
                "run_id": run_id,
                "agent_entity_id": agent_id,
                "agent_name": "Demo Agent",
                "finished_at": finished_at,
                "metrics": metrics,
            },
        },
    }


class _StubHistoryClient:
    """Test double for `memory.client` exposing the
    `_list_entities(entity_type, ...)` surface `fetch_run_history`
    calls."""

    def __init__(self, *, rows=None, raise_on_call=False):
        self._rows = list(rows or [])
        self._raise = raise_on_call
        self.calls: list[tuple[str, dict]] = []

    def _list_entities(self, entity_type, **kw):
        self.calls.append((entity_type, dict(kw)))
        if self._raise:
            raise RuntimeError("history-down")
        return list(self._rows)


class _StubHistoryMemory:
    """Wraps a `_StubHistoryClient` under `memory.client` so
    `fetch_run_history` finds the listing surface."""

    def __init__(self, *, client=None):
        self.client = client or _StubHistoryClient()


def _install_history_memory_stub(memory):
    cli_mod._BUILD_MEMORY_OVERRIDE = lambda: memory


class TestMemoryHistoryCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_text_output_with_entries(self):
        client = _StubHistoryClient(rows=[
            _agent_run_card(
                entity_id="card-1",
                run_id="run-001",
                agent_id="agent-1",
            ),
            _agent_run_card(
                entity_id="card-2",
                run_id="run-002",
                agent_id="agent-1",
                success=False,
                finished_at="2026-05-18T08:00:00+00:00",
            ),
        ])
        _install_history_memory_stub(_StubHistoryMemory(client=client))
        ns = _parse(["memory", "history", "agent-1"])
        code, out, err = _run_handler(_cmd_memory_history, ns)
        assert code == 0
        # Summary header
        assert "history: 2 run(s)" in out
        assert "1 ok" in out
        assert "1 failed" in out
        # Both entries rendered (run id prefix shows in
        # format_one_line).
        assert "run-001" in out
        assert "run-002" in out

    def test_text_output_empty_history(self):
        client = _StubHistoryClient(rows=[])
        _install_history_memory_stub(_StubHistoryMemory(client=client))
        ns = _parse(["memory", "history", "agent-1"])
        code, out, err = _run_handler(_cmd_memory_history, ns)
        assert code == 0
        assert "no runs recorded for agent-1" in out

    def test_json_output(self):
        client = _StubHistoryClient(rows=[
            _agent_run_card(
                entity_id="card-1",
                run_id="run-001",
                agent_id="agent-1",
            ),
        ])
        _install_history_memory_stub(_StubHistoryMemory(client=client))
        ns = _parse(["memory", "history", "agent-1", "--json"])
        code, out, err = _run_handler(_cmd_memory_history, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["chain_id"] == "agent-1"
        assert payload["summary"]["total_runs"] == 1
        assert payload["summary"]["success_count"] == 1
        assert payload["entries"][0]["run_id"] == "run-001"
        assert payload["entries"][0]["status"] == "success"

    def test_filters_forwarded(self):
        client = _StubHistoryClient(rows=[])
        _install_history_memory_stub(_StubHistoryMemory(client=client))
        ns = _parse([
            "memory", "history", "agent-1",
            "--limit", "5",
            "--channel", "stable",
            "--namespace", "team-a",
        ])
        code, out, err = _run_handler(_cmd_memory_history, ns)
        assert code == 0
        assert len(client.calls) == 1
        _, kw = client.calls[0]
        assert kw["limit"] == 5
        assert kw["channel"] == "stable"
        assert kw["namespace"] == "team-a"
        # The fetcher always pins these two tags so the server
        # only returns agent-run cards for this chain.
        assert "agent_run" in kw["tags"]
        assert "agent:agent-1" in kw["tags"]

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["memory", "history", "agent-1"])
        code, out, err = _run_handler(_cmd_memory_history, ns)
        assert code == 2
        assert "config not found" in err

    def test_history_fetch_failure_returns_two(self):
        client = _StubHistoryClient(raise_on_call=True)
        _install_history_memory_stub(_StubHistoryMemory(client=client))
        ns = _parse(["memory", "history", "agent-1"])
        code, out, err = _run_handler(_cmd_memory_history, ns)
        assert code == 2
        assert "history-down" in err

    def test_history_via_main(self, capsys):
        client = _StubHistoryClient(rows=[])
        _install_history_memory_stub(_StubHistoryMemory(client=client))
        code = main(["memory", "history", "agent-1"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        assert "no runs recorded" in captured.out

    def test_history_parser_routes(self):
        ns = _parse(["memory", "history", "agent-1"])
        assert ns._handler is _cmd_memory_history
        assert ns.chain_id == "agent-1"


# ---------------------------------------------------------------------------
# care memory show
# ---------------------------------------------------------------------------


def _show_payload(
    *,
    entity_id: str = "ent-1",
    entity_type: str = "chain",
    name: str = "Storm Watcher",
    tags=("weather", "shared"),
    content: dict | None = None,
    version_id: str = "v-42",
) -> dict:
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "version_id": version_id,
        "channel": "latest",
        "etag": "etag",
        "meta": {"name": name, "tags": list(tags)},
        "content": content or {"steps": [{"prompt": "hi"}]},
    }


class TestMemoryShowCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_default_text_output(self):
        stub = _StubMemory()
        stub.entity_payload = _show_payload()
        _install_memory_stub(stub)
        ns = _parse(["memory", "show", "ent-1"])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 0
        assert "entity: ent-1 (chain)" in out
        assert "version: v-42" in out
        assert "name: Storm Watcher" in out
        assert "tags: weather, shared" in out
        assert "content:" in out
        # Content body pretty-printed (one JSON property per line).
        assert "\"prompt\": \"hi\"" in out

    def test_json_output(self):
        stub = _StubMemory()
        stub.entity_payload = _show_payload()
        _install_memory_stub(stub)
        ns = _parse(["memory", "show", "ent-1", "--json"])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["entity_id"] == "ent-1"
        assert payload["version_id"] == "v-42"
        assert payload["meta"]["name"] == "Storm Watcher"
        assert payload["content"]["steps"][0]["prompt"] == "hi"

    def test_content_only(self):
        stub = _StubMemory()
        stub.entity_payload = _show_payload()
        _install_memory_stub(stub)
        ns = _parse(["memory", "show", "ent-1", "--content-only"])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 0
        # Output should parse as the content dict directly.
        body = json.loads(out)
        assert body == {"steps": [{"prompt": "hi"}]}
        # No metadata header lines.
        assert "entity:" not in out
        assert "name:" not in out

    def test_entity_type_forwarded(self):
        stub = _StubMemory()
        stub.entity_payload = _show_payload(
            entity_id="sk-1",
            entity_type="agent_skill",
            name="pdf-extractor",
        )
        _install_memory_stub(stub)
        ns = _parse([
            "memory", "show", "sk-1",
            "--entity-type", "agent_skill",
        ])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 0
        assert stub.get_entity_calls[0]["entity_type"] == "agent_skill"
        assert "agent_skill" in out

    def test_channel_forwarded(self):
        stub = _StubMemory()
        stub.entity_payload = _show_payload()
        _install_memory_stub(stub)
        ns = _parse([
            "memory", "show", "ent-1", "--channel", "stable",
        ])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 0
        assert stub.get_entity_calls[0]["channel"] == "stable"

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["memory", "show", "ent-1"])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 2
        assert "config not found" in err

    def test_fetch_failure_returns_two(self):
        stub = _StubMemory()
        # entity_payload stays None → get_entity raises "unknown entity".
        _install_memory_stub(stub)
        ns = _parse(["memory", "show", "missing-id"])
        code, out, err = _run_handler(_cmd_memory_show, ns)
        assert code == 2
        assert "failed to fetch chain 'missing-id'" in err

    def test_show_via_main(self, capsys):
        stub = _StubMemory()
        stub.entity_payload = _show_payload()
        _install_memory_stub(stub)
        code = main(["memory", "show", "ent-1", "--json"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["entity_id"] == "ent-1"

    def test_show_parser_routes(self):
        ns = _parse(["memory", "show", "ent-1"])
        assert ns._handler is _cmd_memory_show
        assert ns.entity_type == "chain"


# ---------------------------------------------------------------------------
# CareMemory.get_entity direct surface
# ---------------------------------------------------------------------------


class _StubLineageVersion:
    def __init__(
        self,
        *,
        version_id: str,
        version_number: int,
        parents=(),
        depth: int = 0,
        change_summary: str | None = None,
        author: str | None = None,
        created_at=None,
        evolution_meta=None,
    ):
        self.version_id = version_id
        self.version_number = version_number
        self.parents = list(parents)
        self.depth = depth
        self.change_summary = change_summary
        self.author = author
        self.created_at = created_at
        self.evolution_meta = evolution_meta


class _StubLineageResponse:
    def __init__(
        self,
        *,
        entity_id: str,
        root_version_id: str,
        versions,
        max_depth_reached: bool = False,
    ):
        self.entity_id = entity_id
        self.root_version_id = root_version_id
        self.versions = list(versions)
        self.max_depth_reached = max_depth_reached


def _sample_lineage(*, max_depth_reached: bool = False) -> _StubLineageResponse:
    return _StubLineageResponse(
        entity_id="ent-abc",
        root_version_id="v-root",
        versions=[
            _StubLineageVersion(
                version_id="v-root",
                version_number=0,
                parents=(),
                depth=2,
            ),
            _StubLineageVersion(
                version_id="v-002",
                version_number=1,
                parents=("v-root",),
                depth=1,
                author="user-a",
            ),
            _StubLineageVersion(
                version_id="v-003",
                version_number=2,
                parents=("v-002",),
                depth=0,
            ),
        ],
        max_depth_reached=max_depth_reached,
    )


class TestLineageCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_text_output(self):
        stub = _StubMemory()
        stub.lineage_response = _sample_lineage()
        _install_memory_stub(stub)
        ns = _parse(["lineage", "ent-abc"])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 0
        assert "lineage: ent-abc" in out
        assert "root v-root" in out
        assert "3 version(s)" in out
        # Sorted by depth ascending — v-003 (depth 0) renders
        # first in the body section, then v-002, then v-root.
        body_lines = [
            line for line in out.splitlines() if line.startswith("  ")
        ]
        body_text = "\n".join(body_lines)
        assert body_text.index("v-003") < body_text.index("v-002")
        assert body_text.index("v-002") < body_text.index("v-root")
        # Root version has no parents → "root" badge.
        assert "v0 (v-root)" in out
        # Non-root carries parents=.
        assert "parents=v-root" in out

    def test_max_depth_reached_note(self):
        stub = _StubMemory()
        stub.lineage_response = _sample_lineage(max_depth_reached=True)
        _install_memory_stub(stub)
        ns = _parse(["lineage", "ent-abc"])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 0
        assert "max-depth cap reached" in out

    def test_empty_versions(self):
        stub = _StubMemory()
        stub.lineage_response = _StubLineageResponse(
            entity_id="ent-abc",
            root_version_id="",
            versions=[],
        )
        _install_memory_stub(stub)
        ns = _parse(["lineage", "ent-abc"])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 0
        assert "no versions returned" in out

    def test_json_output(self):
        stub = _StubMemory()
        stub.lineage_response = _sample_lineage()
        _install_memory_stub(stub)
        ns = _parse(["lineage", "ent-abc", "--json"])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["entity_id"] == "ent-abc"
        assert payload["root_version_id"] == "v-root"
        assert payload["max_depth_reached"] is False
        assert len(payload["versions"]) == 3
        version_ids = {v["version_id"] for v in payload["versions"]}
        assert version_ids == {"v-root", "v-002", "v-003"}

    def test_filters_forwarded(self):
        stub = _StubMemory()
        stub.lineage_response = _sample_lineage()
        _install_memory_stub(stub)
        ns = _parse([
            "lineage", "ent-abc",
            "--channel", "stable",
            "--version-id", "v-002",
            "--max-depth", "5",
        ])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 0
        assert stub._lineage_calls[0]["channel"] == "stable"
        assert stub._lineage_calls[0]["version_id"] == "v-002"
        assert stub._lineage_calls[0]["max_depth"] == 5

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["lineage", "ent-abc"])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 2
        assert "config not found" in err

    def test_lineage_fetch_failure_returns_two(self):
        stub = _StubMemory()
        stub._raise_lineage = True
        _install_memory_stub(stub)
        ns = _parse(["lineage", "ent-abc"])
        code, out, err = _run_handler(_cmd_lineage, ns)
        assert code == 2
        assert "failed to fetch lineage" in err
        assert "lineage-down" in err

    def test_lineage_via_main(self, capsys):
        stub = _StubMemory()
        stub.lineage_response = _sample_lineage()
        _install_memory_stub(stub)
        code = main(["lineage", "ent-abc", "--json"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["entity_id"] == "ent-abc"

    def test_lineage_parser_routes(self):
        ns = _parse(["lineage", "ent-abc"])
        assert ns._handler is _cmd_lineage
        assert ns.chain_id == "ent-abc"
        assert ns.channel == "latest"
        assert ns.max_depth == 10


# ---------------------------------------------------------------------------
# care search
# ---------------------------------------------------------------------------


def _search_hit(
    *,
    entity_id: str = "ent-1",
    name: str = "Storm Watcher",
    score: float = 0.91,
    entity_type: str = "chain",
) -> dict:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "version_id": "v-1",
        "score": score,
        "name": name,
        "display_name": name,
    }


class TestSearchCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_text_output_with_hits(self):
        stub = _StubMemory()
        stub.search_hits = [
            _search_hit(entity_id="ent-1", name="Storm Watcher", score=0.91),
            _search_hit(entity_id="ent-2", name="Quiet Watcher", score=0.71),
        ]
        _install_memory_stub(stub)
        ns = _parse(["search", "weather"])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 0
        assert "2 hit(s)" in out
        assert "'weather'" in out
        assert "0.910" in out
        assert "Storm Watcher" in out
        assert "Quiet Watcher" in out
        assert "(chain, bm25)" in out

    def test_text_output_empty(self):
        stub = _StubMemory()
        stub.search_hits = []
        _install_memory_stub(stub)
        ns = _parse(["search", "nothing"])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 0
        assert "no hits for 'nothing'" in out

    def test_json_output(self):
        stub = _StubMemory()
        stub.search_hits = [
            _search_hit(entity_id="ent-1", name="Hello", score=0.5),
        ]
        _install_memory_stub(stub)
        ns = _parse(["search", "hello", "--json"])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["query"] == "hello"
        assert payload["entity_type"] == "chain"
        assert payload["search_type"] == "bm25"
        assert len(payload["hits"]) == 1
        assert payload["hits"][0]["entity_id"] == "ent-1"

    def test_filters_forwarded(self):
        stub = _StubMemory()
        stub.search_hits = []
        _install_memory_stub(stub)
        ns = _parse([
            "search", "weather",
            "--entity-type", "agent_skill",
            "--search-type", "hybrid",
            "--top-k", "5",
        ])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 0
        call = stub._search_calls[0]
        assert call["query"] == "weather"
        assert call["entity_type"] == "agent_skill"
        assert call["search_type"] == "hybrid"
        assert call["top_k"] == 5

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["search", "x"])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 2
        assert "config not found" in err

    def test_search_failure_returns_two(self):
        stub = _StubMemory()
        stub._raise_search = True
        _install_memory_stub(stub)
        ns = _parse(["search", "x"])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 2
        assert "lookup failed" in err
        assert "search-down" in err

    def test_score_fallback_on_non_numeric(self):
        stub = _StubMemory()
        # No score field present → text renders as "—".
        stub.search_hits = [
            {"entity_id": "ent-x", "name": "Foo"},
        ]
        _install_memory_stub(stub)
        ns = _parse(["search", "any"])
        code, out, err = _run_handler(_cmd_search, ns)
        assert code == 0
        assert "—  ent-x" in out
        assert "Foo" in out

    def test_search_via_main(self, capsys):
        stub = _StubMemory()
        stub.search_hits = [_search_hit(entity_id="x", name="Y", score=0.5)]
        _install_memory_stub(stub)
        code = main(["search", "x", "--json"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["hits"][0]["entity_id"] == "x"

    def test_search_parser_routes(self):
        ns = _parse(["search", "x"])
        assert ns._handler is _cmd_search
        assert ns.query == "x"
        assert ns.entity_type == "chain"
        assert ns.search_type == "bm25"
        assert ns.top_k == 10


# ---------------------------------------------------------------------------
# care diff
# ---------------------------------------------------------------------------


class _StubDiffClient:
    """Test double for `memory.client` exposing the
    `get_chain_dict(entity_id, channel)` surface
    `fetch_agent_diff` calls."""

    def __init__(self, *, chains=None, raise_for=None):
        # chains: dict[entity_id, chain_dict]
        self._chains = dict(chains or {})
        self._raise_for = raise_for
        self.calls: list[tuple[str, str]] = []

    def get_chain_dict(self, entity_id, channel="latest"):
        self.calls.append((entity_id, channel))
        if self._raise_for == entity_id:
            raise RuntimeError(f"chain-down:{entity_id}")
        return self._chains.get(entity_id)


class _StubDiffMemory:
    def __init__(self, *, client=None):
        self.client = client or _StubDiffClient()


def _install_diff_memory_stub(memory):
    cli_mod._BUILD_MEMORY_OVERRIDE = lambda: memory


def _chain_with_steps(*, steps, name: str = "") -> dict:
    """Build a chain_dict matching what `fetch_agent_diff`
    expects (raw `get_chain_dict` shape)."""
    chain: dict = {"steps": steps}
    if name:
        chain["metadata"] = {"care": {"display_name": name}}
    return chain


def _left_chain() -> dict:
    return _chain_with_steps(
        name="Storm Watcher v1",
        steps=[
            {"number": 1, "name": "fetch", "type": "llm", "prompt": "old"},
            {"number": 2, "name": "summarise", "type": "llm"},
        ],
    )


def _right_chain() -> dict:
    return _chain_with_steps(
        name="Storm Watcher v2",
        steps=[
            # Modified prompt on step 1.
            {"number": 1, "name": "fetch", "type": "llm", "prompt": "new"},
            # Step 2 unchanged.
            {"number": 2, "name": "summarise", "type": "llm"},
            # New step 3.
            {"number": 3, "name": "verify", "type": "llm"},
        ],
    )


class TestDiffCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_text_output(self):
        client = _StubDiffClient(chains={
            "left-id": _left_chain(),
            "right-id": _right_chain(),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse(["diff", "left-id", "right-id"])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 0
        # Header
        assert "diff: left-id ↔ right-id" in out
        # Step counts in summary
        assert "+1" in out  # added (step 3)
        assert "~1" in out  # modified (step 1)
        # Per-step lines
        assert "+ step 3" in out
        assert "~ step 1" in out
        # Unchanged still shown
        assert "· step 2" in out

    def test_text_output_with_labels(self):
        client = _StubDiffClient(chains={
            "left-id": _left_chain(),
            "right-id": _right_chain(),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse([
            "diff", "left-id", "right-id",
            "--left-label", "v1",
            "--right-label", "v2",
        ])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 0
        assert "diff: v1 ↔ v2" in out

    def test_json_output(self):
        client = _StubDiffClient(chains={
            "left-id": _left_chain(),
            "right-id": _right_chain(),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse(["diff", "left-id", "right-id", "--json"])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["left_entity_id"] == "left-id"
        assert payload["right_entity_id"] == "right-id"
        assert payload["counts"]["added"] == 1
        assert payload["counts"]["modified"] == 1
        assert payload["counts"]["unchanged"] == 1
        # Per-step rows present
        kinds = {s["kind"] for s in payload["steps"]}
        assert kinds == {"added", "modified", "unchanged"}

    def test_identical_chains_no_changes(self):
        same = _left_chain()
        client = _StubDiffClient(chains={
            "left-id": same,
            "right-id": dict(same),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse(["diff", "left-id", "right-id"])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 0
        assert "no differences" in out

    def test_channel_forwarded(self):
        client = _StubDiffClient(chains={
            "left-id": _left_chain(),
            "right-id": _right_chain(),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse([
            "diff", "left-id", "right-id",
            "--channel", "stable",
        ])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 0
        # Both fetches use the supplied channel.
        for (_, channel) in client.calls:
            assert channel == "stable"

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["diff", "l", "r"])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 2
        assert "config not found" in err

    def test_missing_chain_returns_two(self):
        # Only `left-id` is in the stub; right-id returns None
        # which the fetcher surfaces as `AgentDiffError`.
        client = _StubDiffClient(chains={
            "left-id": _left_chain(),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse(["diff", "left-id", "right-id"])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 2
        # Either path is fine — AgentDiffError surfaces with the
        # canonical message; lookup_failed path also lands here.
        assert err

    def test_fetch_exception_returns_two(self):
        client = _StubDiffClient(
            chains={"left-id": _left_chain(), "right-id": _right_chain()},
            raise_for="right-id",
        )
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        ns = _parse(["diff", "left-id", "right-id"])
        code, out, err = _run_handler(_cmd_diff, ns)
        assert code == 2
        # The underlying `chain-down:right-id` propagates through.
        assert "chain-down:right-id" in err

    def test_diff_via_main(self, capsys):
        client = _StubDiffClient(chains={
            "left-id": _left_chain(),
            "right-id": _right_chain(),
        })
        _install_diff_memory_stub(_StubDiffMemory(client=client))
        code = main(["diff", "left-id", "right-id", "--json"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["counts"]["added"] == 1

    def test_diff_parser_routes(self):
        ns = _parse(["diff", "a", "b"])
        assert ns._handler is _cmd_diff
        assert ns.left == "a"
        assert ns.right == "b"
        assert ns.channel == "latest"


# ---------------------------------------------------------------------------
# care favourite
# ---------------------------------------------------------------------------


class TestFavouriteCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_default_stars_chain(self):
        stub = _StubMemory()
        stub.favourite_response = {
            "entity_id": "ent-1",
            "display_name": "Storm Watcher",
            "favourite": True,
        }
        _install_memory_stub(stub)
        ns = _parse(["favourite", "ent-1"])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 0
        # Default value (no --off) stars the entity.
        assert stub._fav_calls[0] == {
            "entity_id": "ent-1",
            "entity_type": "chain",
            "value": True,
        }
        assert "★" in out
        assert "starred: ent-1 (chain)" in out
        assert "Storm Watcher" in out

    def test_off_unstars(self):
        stub = _StubMemory()
        stub.favourite_response = {
            "entity_id": "ent-1",
            "display_name": "Storm Watcher",
            "favourite": False,
        }
        _install_memory_stub(stub)
        ns = _parse(["favourite", "ent-1", "--off"])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 0
        assert stub._fav_calls[0]["value"] is False
        assert "★" not in out
        assert "unstarred: ent-1 (chain)" in out

    def test_entity_type_forwarded(self):
        stub = _StubMemory()
        _install_memory_stub(stub)
        ns = _parse([
            "favourite", "sk-1",
            "--entity-type", "agent_skill",
        ])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 0
        assert stub._fav_calls[0]["entity_type"] == "agent_skill"
        assert "(agent_skill)" in out

    def test_json_output(self):
        stub = _StubMemory()
        stub.favourite_response = {
            "entity_id": "ent-1",
            "display_name": "Storm Watcher",
            "favourite": True,
            "run_count": 7,
        }
        _install_memory_stub(stub)
        ns = _parse(["favourite", "ent-1", "--json"])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["entity_id"] == "ent-1"
        assert payload["favourite"] is True
        assert payload["run_count"] == 7

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["favourite", "ent-1"])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 2
        assert "config not found" in err

    def test_mark_failure_returns_two(self):
        stub = _StubMemory()
        stub._raise_favourite = True
        _install_memory_stub(stub)
        ns = _parse(["favourite", "ent-1"])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 2
        assert "failed to star chain 'ent-1'" in err
        assert "fav-down" in err

    def test_mark_failure_off_uses_unstar_verb(self):
        stub = _StubMemory()
        stub._raise_favourite = True
        _install_memory_stub(stub)
        ns = _parse(["favourite", "ent-1", "--off"])
        code, out, err = _run_handler(_cmd_favourite, ns)
        assert code == 2
        assert "failed to unstar chain 'ent-1'" in err

    def test_favourite_via_main(self, capsys):
        stub = _StubMemory()
        stub.favourite_response = {
            "entity_id": "ent-1",
            "display_name": "Y",
            "favourite": True,
        }
        _install_memory_stub(stub)
        code = main(["favourite", "ent-1"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        assert "starred: ent-1" in captured.out

    def test_favourite_parser_routes(self):
        ns = _parse(["favourite", "ent-1"])
        assert ns._handler is _cmd_favourite
        assert ns.entity_id == "ent-1"
        assert ns.entity_type == "chain"
        assert ns.off is False


# ---------------------------------------------------------------------------
# CareMemory.mark_favourite contract
# ---------------------------------------------------------------------------


class TestMemoryMarkFavouriteUnknownType:
    def test_unknown_entity_type_raises(self):
        import pytest

        from care.memory import CareMemory

        class _ClientStub:
            def _mark_favourite(self, *a, **kw):  # pragma: no cover
                raise AssertionError("should not be called")

        memory = CareMemory(client=_ClientStub())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unsupported entity_type"):
            memory.mark_favourite("x", entity_type="bogus")


# ---------------------------------------------------------------------------
# care marketplace
# ---------------------------------------------------------------------------


def _market_hit(
    *,
    entity_id: str,
    name: str,
    score: float = 0.5,
    tags=(),
    matched_via: str = "skill_description",
    description: str = "",
    snippet=None,
) -> dict:
    return {
        "entity_id": entity_id,
        "name": name,
        "description": description,
        "score": score,
        "tags": list(tags),
        "matched_via": matched_via,
        "snippet": snippet,
    }


class TestMarketplaceCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_text_output_with_listings(self):
        stub = _StubMemory()
        stub.market_hits = [
            _market_hit(
                entity_id="sk-1", name="pdf-extract",
                score=0.91, tags=("pdf", "finance"),
            ),
            _market_hit(
                entity_id="sk-2", name="csv-parser",
                score=0.7, tags=("csv",),
                matched_via="skill_instructions",
            ),
        ]
        _install_memory_stub(stub)
        ns = _parse(["marketplace", "extract pdf"])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 0
        assert "2 listing(s) for 'extract pdf'" in out
        assert "0.910" in out
        assert "pdf-extract" in out
        assert "csv-parser" in out
        # ★ goes to skill_description match, blank for instructions
        # match (rendered as a single space, hard to assert via
        # substring; just check the structural ★ shows up at least
        # once for the matching listing).
        assert "★" in out
        assert "[pdf, finance]" in out

    def test_text_output_empty(self):
        stub = _StubMemory()
        stub.market_hits = []
        _install_memory_stub(stub)
        ns = _parse(["marketplace", "nothing"])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 0
        assert "no listings for 'nothing'" in out

    def test_json_output(self):
        stub = _StubMemory()
        stub.market_hits = [
            _market_hit(
                entity_id="sk-1", name="pdf",
                score=0.91, tags=("pdf",),
            ),
        ]
        _install_memory_stub(stub)
        ns = _parse(["marketplace", "pdf", "--json"])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["query"] == "pdf"
        assert payload["listings"][0]["entity_id"] == "sk-1"
        assert payload["listings"][0]["tags"] == ["pdf"]

    def test_filters_forwarded(self):
        stub = _StubMemory()
        stub.market_hits = []
        _install_memory_stub(stub)
        ns = _parse([
            "marketplace", "pdf",
            "--top-k", "5",
            "--min-score", "0.4",
            "--tag", "pdf",
            "--tag", "finance",
            "--namespace", "team-a",
            "--deep",
        ])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 0
        call = stub._market_calls[0]
        assert call["query"] == "pdf"
        assert call["top_k"] == 5
        assert call["namespace"] == "team-a"
        assert call["deep"] is True

    def test_tag_filter_applied_after_backend(self):
        # The CARE-side `tags` filter is all-of. Backend returns
        # both, the filter drops the one missing a required tag.
        stub = _StubMemory()
        stub.market_hits = [
            _market_hit(
                entity_id="sk-1", name="pdf-only",
                tags=("pdf",),
            ),
            _market_hit(
                entity_id="sk-2", name="pdf-finance",
                tags=("pdf", "finance"),
            ),
        ]
        _install_memory_stub(stub)
        ns = _parse([
            "marketplace", "pdf",
            "--tag", "pdf",
            "--tag", "finance",
        ])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 0
        assert "pdf-finance" in out
        # `pdf-only` filtered out (missing the finance tag).
        assert "pdf-only" not in out
        assert "1 listing(s)" in out

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["marketplace", "x"])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 2
        assert "config not found" in err

    def test_backend_failure_returns_two(self):
        stub = _StubMemory()
        stub._raise_market = True
        _install_memory_stub(stub)
        ns = _parse(["marketplace", "x"])
        code, out, err = _run_handler(_cmd_marketplace, ns)
        assert code == 2
        # `search_marketplace` wraps the underlying exception
        # in `MarketplaceError`; the CLI surfaces the wrapper
        # text.
        assert "market-down" in err

    def test_marketplace_via_main(self, capsys):
        stub = _StubMemory()
        stub.market_hits = [
            _market_hit(entity_id="sk-1", name="Y", score=0.5),
        ]
        _install_memory_stub(stub)
        code = main(["marketplace", "x", "--json"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["listings"][0]["entity_id"] == "sk-1"

    def test_marketplace_parser_routes(self):
        ns = _parse(["marketplace", "x"])
        assert ns._handler is _cmd_marketplace
        assert ns.query == "x"
        assert ns.top_k == 10
        assert ns.min_score == 0.0
        assert ns.deep is False


class TestMemoryGetEntityUnknownType:
    def test_unknown_entity_type_raises(self):
        # Pure pytest unit — no CLI involved — to pin the
        # ValueError contract `_cmd_memory_show` relies on (it
        # surfaces the message verbatim via the `failed to fetch`
        # path).
        import pytest

        from care.memory import CareMemory

        class _ClientStub:
            def _get_entity(self, *a, **kw):  # pragma: no cover
                raise AssertionError("should not be called")

        memory = CareMemory(client=_ClientStub())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unsupported entity_type"):
            memory.get_entity("x", entity_type="bogus")


# ---------------------------------------------------------------------------
# care replay
# ---------------------------------------------------------------------------


def _replay_source_dict() -> dict:
    """RunRecord-shaped dict with two steps that `load_replay`
    parses cleanly."""
    return {
        "chain_id": "ent-abc",
        "chain_title": "Storm Watcher",
        "result": {
            "step_results": [
                {
                    "step_number": 1,
                    "step_title": "fetch",
                    "step_type": "llm",
                    "result": "fetched payload",
                    "success": True,
                    "execution_time_s": 0.42,
                },
                {
                    "step_number": 2,
                    "step_title": "summarise",
                    "step_type": "llm",
                    "result": "summary",
                    "success": True,
                    "execution_time_s": 1.1,
                },
            ],
            "total_execution_time": 1.52,
            "final_answer": "all done",
        },
    }


def _write_replay_file(tmp_path: Path, payload: dict | None = None) -> Path:
    payload = payload or _replay_source_dict()
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestReplayCommand:
    def test_walks_every_step_by_default(self, tmp_path: Path):
        path = _write_replay_file(tmp_path)
        ns = _parse(["replay", str(path)])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 0
        assert "Storm Watcher" in out
        assert "fetch" in out
        assert "summarise" in out
        # The header reports the cursor position; with all steps
        # walked we should see at least step 1/2 and step 2/2.
        assert "step 1/2" in out
        assert "step 2/2" in out

    def test_step_renders_single_step(self, tmp_path: Path):
        path = _write_replay_file(tmp_path)
        ns = _parse(["replay", str(path), "--step", "1"])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 0
        # Only the second step's block (cursor=1).
        assert "step 2/2" in out
        assert "summarise" in out
        # The first step header doesn't appear when --step is set.
        assert "step 1/2" not in out

    def test_json_output(self, tmp_path: Path):
        path = _write_replay_file(tmp_path)
        ns = _parse(["replay", str(path), "--json"])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["chain_id"] == "ent-abc"
        assert payload["chain_title"] == "Storm Watcher"
        assert payload["step_count"] == 2
        assert payload["steps"][0]["step_title"] == "fetch"
        assert payload["steps"][1]["step_title"] == "summarise"
        assert payload["final_answer"] == "all done"

    def test_empty_session_text(self, tmp_path: Path):
        path = _write_replay_file(tmp_path, payload={"result": {}})
        ns = _parse(["replay", str(path)])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 0
        assert "no steps in source" in out

    def test_missing_file_returns_two(self, tmp_path: Path):
        ns = _parse(["replay", str(tmp_path / "nope.json")])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 2
        assert "read failed" in err

    def test_malformed_json_returns_two(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        ns = _parse(["replay", str(path)])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 2
        assert "failed to parse" in err

    def test_stdin_source(self, capsys, monkeypatch):
        # The handler reads from `sys.stdin` when source is `-`.
        import io as _io
        monkeypatch.setattr(
            "sys.stdin",
            _io.StringIO(json.dumps(_replay_source_dict())),
        )
        ns = _parse(["replay", "-"])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 0
        assert "Storm Watcher" in out

    def test_step_clamps_to_bounds(self, tmp_path: Path):
        # ReplaySession.seek clamps out-of-bounds requests to the
        # nearest endpoint, so --step 99 falls onto the last step.
        path = _write_replay_file(tmp_path)
        ns = _parse(["replay", str(path), "--step", "99"])
        code, out, err = _run_handler(_cmd_replay, ns)
        assert code == 0
        assert "step 2/2" in out
        assert "summarise" in out

    def test_replay_via_main(self, capsys, tmp_path: Path):
        path = _write_replay_file(tmp_path)
        code = main(["replay", str(path), "--json"])
        assert code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["chain_id"] == "ent-abc"

    def test_replay_parser_routes(self, tmp_path: Path):
        ns = _parse(["replay", str(tmp_path / "any.json")])
        assert ns._handler is _cmd_replay
        assert ns.step is None
        assert ns.json is False


# ---------------------------------------------------------------------------
# care memory subcommand parser surface
# ---------------------------------------------------------------------------


class TestMemoryParserSurface:
    def test_memory_ls_is_routed(self):
        ns = _parse(["memory", "ls"])
        assert ns._handler is _cmd_memory_ls
        assert ns.entity_type == "chain"

    def test_memory_ls_help_routes_to_no_handler(self):
        # `care memory` alone (no sub) — argparse should leave
        # _handler unset so main() returns 0.
        # We can't test --help here (it calls SystemExit) but
        # we can check the bare subcommand path doesn't crash.
        ns = _parse(["memory"])
        assert getattr(ns, "_handler", None) is None


# ---------------------------------------------------------------------------
# care evolve
# ---------------------------------------------------------------------------


class _StubEvolutionRef:
    def __init__(self, *, evolution_id="evo-1", status="running"):
        self.evolution_id = evolution_id
        self.status = status


class _StubPlatform:
    """Test double for the CarePlatform facade.

    Captures every start_evolution call, yields the events
    supplied at construction time from stream_events, and
    records accept calls.
    """

    def __init__(
        self,
        *,
        events=(),
        ref=None,
        raise_on_start=False,
        raise_on_accept=False,
    ):
        self._events = list(events)
        self._ref = ref or _StubEvolutionRef()
        self._raise_start = raise_on_start
        self._raise_accept = raise_on_accept
        self.start_calls: list[dict] = []
        self.accept_calls: list[tuple[str, str]] = []

    def start_evolution(self, **kw):
        self.start_calls.append(dict(kw))
        if self._raise_start:
            raise RuntimeError("submit-down")
        return self._ref

    def stream_events(self, evolution_id):
        for e in self._events:
            yield e

    def accept_individual(self, evolution_id, individual_id):
        self.accept_calls.append((evolution_id, individual_id))
        if self._raise_accept:
            raise RuntimeError("accept-down")
        return {"accepted": True}


def _install_platform_stub(stub):
    cli_mod._BUILD_PLATFORM_OVERRIDE = lambda: stub


def _restore_platform_stub():
    cli_mod._BUILD_PLATFORM_OVERRIDE = None


class TestEvolveCommand:
    def teardown_method(self):
        _restore_platform_stub()
        _restore_memory_stub()

    def test_basic_submit_without_wait(self):
        stub = _StubPlatform()
        _install_platform_stub(stub)
        ns = _parse(["evolve", "agent-1"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 0
        assert len(stub.start_calls) == 1
        assert stub.start_calls[0]["base_chain_id"] == "agent-1"
        assert "evolution evo-1" in out
        # No --wait → no events drained.
        assert stub.accept_calls == []

    def test_flags_forwarded_to_start_evolution(self):
        stub = _StubPlatform()
        _install_platform_stub(stub)
        ns = _parse([
            "evolve", "agent-1",
            "--mode", "per_step",
            "--iterations", "12",
            "--population", "16",
            "--validation-criteria", "prefer brevity",
            "--objective", "accuracy",
            "--objective", "latency",
            "--threshold", "0.85",
        ])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 0
        call = stub.start_calls[0]
        assert call["evolution_mode"] == "per_step"
        assert call["max_iterations"] == 12
        assert call["population_size"] == 16
        assert call["validation_criteria"] == "prefer brevity"
        assert call["objectives"] == ["accuracy", "latency"]
        assert call["validation_threshold"] == 0.85

    def test_wait_drains_events_and_prints_progress(self):
        events = [
            {"event": "generation_started", "data": {"generation": 0}},
            {
                "event": "individual_evaluated",
                "data": {
                    "generation": 0,
                    "individual_id": "ind-1",
                    "fitness": 0.42,
                },
            },
            {
                "event": "best_updated",
                "data": {
                    "generation": 0,
                    "best_individual_id": "ind-1",
                    "fitness": 0.55,
                },
            },
            {"event": "completed", "data": {}},
        ]
        stub = _StubPlatform(events=events)
        _install_platform_stub(stub)
        ns = _parse(["evolve", "agent-1", "--wait"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 0
        assert "[gen 0] started" in out
        assert "evaluated ind-1 fitness=0.420" in out
        assert "best now ind-1 fitness=0.550" in out
        assert "[done] evolution completed" in out
        assert "status=completed" in out

    def test_accept_promotes_best_after_wait(self):
        events = [
            {
                "event": "best_updated",
                "data": {
                    "generation": 0,
                    "best_individual_id": "ind-winner",
                    "fitness": 0.9,
                },
            },
            {"event": "completed", "data": {}},
        ]
        stub = _StubPlatform(events=events)
        _install_platform_stub(stub)
        ns = _parse([
            "evolve", "agent-1", "--wait", "--accept",
        ])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 0
        assert stub.accept_calls == [("evo-1", "ind-winner")]
        assert "accepted: ind-winner" in out

    def test_accept_chain_experiment_threads_memory_without_winner_id(self):
        # Chain experiments stream no per-individual winner id (best_updated
        # carries fitness only) and promote the persisted best_chain_config
        # via Memory CARE-side — so accept must still fire AND pass memory.
        events = [
            {"event": "best_updated", "data": {"generation": 1, "best_fitness": 0.8}},
            {"event": "completed", "data": {}},
        ]

        class _ExpPlatform(_StubPlatform):
            def accept_individual(self, evolution_id, individual_id, *, memory=None):
                self.accept_calls.append((evolution_id, individual_id))
                self.accept_memory = memory
                return {"chain_id": "chain-xyz", "new_version": 4}

        stub = _ExpPlatform(
            events=events, ref=_StubEvolutionRef(evolution_id="exp_abc"),
        )
        _install_platform_stub(stub)
        sentinel_mem = object()
        _install_memory_stub(sentinel_mem)
        ns = _parse(["evolve", "agent-1", "--wait", "--accept"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 0, err
        # Accept attempted despite no individual id; memory threaded through.
        assert stub.accept_calls == [("exp_abc", "")]
        assert stub.accept_memory is sentinel_mem
        assert "accepted: chain-xyz" in out

    def test_accept_without_wait_returns_two(self):
        stub = _StubPlatform()
        _install_platform_stub(stub)
        ns = _parse(["evolve", "agent-1", "--accept"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 2
        assert "requires --wait" in err
        # No platform call attempted.
        assert stub.start_calls == []

    def test_invalid_plan_returns_two(self):
        stub = _StubPlatform()
        _install_platform_stub(stub)
        # iterations=0 fails build_evolution_request validation.
        ns = _parse(["evolve", "agent-1", "--iterations", "0"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 2
        assert "invalid plan" in err
        assert stub.start_calls == []

    def test_empty_chain_id_returns_two(self):
        stub = _StubPlatform()
        _install_platform_stub(stub)
        ns = _parse(["evolve", ""])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 2
        assert "invalid plan" in err

    def test_submit_failure_returns_two(self):
        stub = _StubPlatform(raise_on_start=True)
        _install_platform_stub(stub)
        ns = _parse(["evolve", "agent-1"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 2
        assert "submit failed" in err
        assert "submit-down" in err

    def test_accept_failure_returns_two(self):
        events = [
            {
                "event": "best_updated",
                "data": {
                    "generation": 0,
                    "best_individual_id": "ind-w",
                    "fitness": 0.8,
                },
            },
            {"event": "completed", "data": {}},
        ]
        stub = _StubPlatform(events=events, raise_on_accept=True)
        _install_platform_stub(stub)
        ns = _parse([
            "evolve", "agent-1", "--wait", "--accept",
        ])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 2
        assert "accept failed" in err
        assert "accept-down" in err

    def test_json_output_after_wait(self):
        events = [
            {
                "event": "best_updated",
                "data": {
                    "generation": 2,
                    "best_individual_id": "ind-w",
                    "fitness": 0.77,
                },
            },
            {"event": "completed", "data": {}},
        ]
        stub = _StubPlatform(events=events)
        _install_platform_stub(stub)
        ns = _parse(["evolve", "agent-1", "--wait", "--json"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["evolution_id"] == "evo-1"
        assert payload["status"] == "completed"
        assert payload["best_individual_id"] == "ind-w"
        assert payload["best_fitness"] == 0.77
        assert payload["generation"] == 2

    def test_platform_build_failure_returns_two(self):
        from care.cli import CliPlatformError

        def _boom():
            raise CliPlatformError("config not found")

        cli_mod._BUILD_PLATFORM_OVERRIDE = _boom
        ns = _parse(["evolve", "agent-1"])
        code, out, err = _run_handler(_cmd_evolve, ns)
        assert code == 2
        assert "config not found" in err

    def test_evolve_subcommand_via_main(self, capsys):
        _install_platform_stub(_StubPlatform())
        code = main(["evolve", "agent-1"])
        _restore_platform_stub()
        assert code == 0
        captured = capsys.readouterr()
        assert "evolution evo-1" in captured.out


# ---------------------------------------------------------------------------
# care help
# ---------------------------------------------------------------------------


class TestHelpCommand:
    def test_default_text_output(self):
        ns = _parse(["help"])
        code, out, err = _run_handler(_cmd_help, ns)
        assert code == 0
        # Tutorial section
        assert "# Tutorial" in out
        assert "Welcome to CARE" in out
        # Key bindings section, grouped by category
        assert "# Key bindings" in out
        assert "## global" in out
        assert "## library" in out
        # Documented globals present
        assert "Ctrl+P" in out
        assert "Ctrl+Q" in out
        assert err == ""

    def test_markdown_output(self):
        ns = _parse(["help", "--markdown"])
        code, out, err = _run_handler(_cmd_help, ns)
        assert code == 0
        # Markdown headings
        assert "## Walkthrough" in out
        assert "## Keys" in out
        # Bold step titles
        assert "**1. Welcome to CARE**" in out
        # Backticks around keys
        assert "`Ctrl+Q`" in out
        # Hint blockquote
        assert "> Try:" in out

    def test_category_filter_restricts_bindings(self):
        ns = _parse(["help", "--category", "library"])
        code, out, err = _run_handler(_cmd_help, ns)
        assert code == 0
        # Tutorial unchanged
        assert "# Tutorial" in out
        # Library section present
        assert "## library" in out
        # Other categories filtered out
        assert "## generation" not in out
        assert "## execution" not in out

    def test_screen_filter_restricts_bindings(self):
        ns = _parse(["help", "--screen", "LibraryScreen"])
        code, out, err = _run_handler(_cmd_help, ns)
        assert code == 0
        # LibraryScreen-scoped bindings should appear
        assert "LibraryScreen" in out
        # Global bindings (no `screen` attribute) get filtered out;
        # tutorial steps remain.
        assert "# Tutorial" in out
        # `Ctrl+Q` is a global without a screen attribute → filtered
        assert "Ctrl+Q" not in out

    def test_help_subcommand_via_main(self, capsys):
        code = main(["help"])
        assert code == 0
        captured = capsys.readouterr()
        assert "# Tutorial" in captured.out
        assert "# Key bindings" in captured.out

    def test_markdown_via_main(self, capsys):
        code = main(["help", "--markdown"])
        assert code == 0
        captured = capsys.readouterr()
        assert "## Walkthrough" in captured.out


# ---------------------------------------------------------------------------
# care run
# ---------------------------------------------------------------------------


class _StubReasoningResult:
    def __init__(
        self,
        *,
        success=True,
        steps=2,
        final_answer="hello",
    ):
        self.success = success
        self.step_results = [object()] * steps
        self.final_answer = final_answer


def _make_carl_executor(*, success=True, raise_on_run=False, capture=None):
    """Build a callable matching :func:`_build_carl_executor`'s
    return shape: a no-arg builder returning an async closure.

    ``capture`` (optional list) records each call's
    ``(chain_dict, task, inputs)`` tuple so tests can assert on
    what the CLI forwarded.
    """

    def _builder():
        async def _exec(chain_dict, *, task=None, inputs=None):
            if capture is not None:
                capture.append((chain_dict, task, dict(inputs or {})))
            if raise_on_run:
                raise RuntimeError("run-down")
            return _StubReasoningResult(success=success)

        return _exec

    return _builder


def _install_carl_executor_stub(builder):
    cli_mod._BUILD_CARL_EXECUTOR_OVERRIDE = builder


def _restore_carl_executor_stub():
    cli_mod._BUILD_CARL_EXECUTOR_OVERRIDE = None


def _runnable_chain_dict() -> dict:
    return {
        "task_description": "demo",
        "steps": [
            {
                "number": 1,
                "title": "first",
                "step_type": "llm",
                "aim": "hi",
            },
        ],
    }


class TestRunCommand:
    def teardown_method(self):
        _restore_memory_stub()

    def test_fetch_and_preflight(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        ns = _parse(["run", "ent-abc"])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 0
        assert stub.get_chain_calls == [("ent-abc", "latest")]
        assert "chain: ent-abc" in out
        # validate_chain.format_text() emits ok-text on success
        assert "ok" in out or "preflight skipped" in out

    def test_channel_forwarded(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        ns = _parse(["run", "ent-abc", "--channel", "stable"])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 0
        assert stub.get_chain_calls == [("ent-abc", "stable")]
        assert "channel=stable" in out

    def test_json_output(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        ns = _parse(["run", "ent-abc", "--json"])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["entity_id"] == "ent-abc"
        assert payload["channel"] == "latest"
        assert payload["parsed"] is True

    def test_export_writes_file(self, tmp_path: Path):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        dest = tmp_path / "out.json"
        ns = _parse(["run", "ent-abc", "--export", str(dest)])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 0
        assert dest.exists()
        assert "exported:" in out
        # Round-trip: the file matches the fetched chain.
        body = json.loads(dest.read_text(encoding="utf-8"))
        assert body["task_description"] == "demo"

    def test_export_failure_returns_two(self, tmp_path: Path):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        # No-extension destination + no --export-format → infer
        # failure inside care.export_chain.
        dest = tmp_path / "noext"
        ns = _parse(["run", "ent-abc", "--export", str(dest)])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 2
        assert "export failed" in err

    def test_fetch_failure_returns_two(self):
        stub = _StubMemory(raise_on_get_chain=True)
        _install_memory_stub(stub)
        ns = _parse(["run", "ent-abc"])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 2
        assert "failed to fetch chain" in err
        assert "fetch-down" in err

    def test_memory_build_failure_returns_two(self):
        from care.cli import CliMemoryError

        def _boom():
            raise CliMemoryError("config not found")

        cli_mod._BUILD_MEMORY_OVERRIDE = _boom
        ns = _parse(["run", "ent-abc"])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 2
        assert "config not found" in err

    def test_invalid_chain_returns_one(self):
        # Wrapper-form invalid chain still parses, but emit a
        # chain that fails preflight by missing required `steps`.
        stub = _StubMemory(chain={"task_description": "nope"})
        _install_memory_stub(stub)
        ns = _parse(["run", "ent-abc"])
        code, out, err = _run_handler(_cmd_run, ns)
        # validate_chain reports parse failure → exit 1
        assert code == 1

    def test_execute_succeeds_with_stub_executor(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse(["run", "ent-abc", "--execute"])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        assert "chain: ent-abc" in out
        assert "executed: status=ok" in out

    def test_execute_failure_returns_one(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=False))
        ns = _parse(["run", "ent-abc", "--execute"])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 1
        assert "executed: status=failed" in out

    def test_execute_carl_build_failure_returns_two(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        from care.cli import CliCarlError

        def _boom():
            raise CliCarlError("mmar_carl isn't installed")

        cli_mod._BUILD_CARL_EXECUTOR_OVERRIDE = _boom
        ns = _parse(["run", "ent-abc", "--execute"])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 2
        assert "mmar_carl isn't installed" in err

    def test_execute_runtime_failure_returns_two(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(raise_on_run=True))
        ns = _parse(["run", "ent-abc", "--execute"])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 2
        assert "execution failed" in err
        assert "run-down" in err

    def test_execute_skipped_on_parse_failure(self):
        # An unparseable chain → exit 1 from preflight, executor
        # is never asked to run.
        stub = _StubMemory(chain={"task_description": "nope"})
        _install_memory_stub(stub)
        ran: list = []

        def _builder():
            async def _exec(chain_dict, *, task=None, inputs=None):
                ran.append(chain_dict)
                return None
            return _exec

        cli_mod._BUILD_CARL_EXECUTOR_OVERRIDE = _builder
        ns = _parse(["run", "ent-abc", "--execute"])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 1
        assert ran == []

    def test_execute_task_forwarded(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        capture: list = []
        _install_carl_executor_stub(_make_carl_executor(capture=capture))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--task", "What is the weather in Paris?",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        assert len(capture) == 1
        _, task, inputs = capture[0]
        assert task == "What is the weather in Paris?"
        assert inputs == {}

    def test_execute_inputs_parsed(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        capture: list = []
        _install_carl_executor_stub(_make_carl_executor(capture=capture))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--input", "city=Paris",
            "--input", "units=metric",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        _, _, inputs = capture[0]
        assert inputs == {"city": "Paris", "units": "metric"}

    def test_execute_file_attached_by_basename(self, tmp_path: Path):
        f = tmp_path / "report.txt"
        f.write_text("FILE-BODY")
        _install_memory_stub(_StubMemory(chain=_runnable_chain_dict()))
        capture: list = []
        _install_carl_executor_stub(_make_carl_executor(capture=capture))
        ns = _parse(["run", "ent-abc", "--execute", "--file", str(f)])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        _, _, inputs = capture[0]
        # The file reaches the chain keyed by its basename.
        assert inputs["report.txt"] == "FILE-BODY"

    def test_execute_input_wins_over_file_same_basename(
        self, tmp_path: Path,
    ):
        f = tmp_path / "report.txt"
        f.write_text("FROM-FILE")
        _install_memory_stub(_StubMemory(chain=_runnable_chain_dict()))
        capture: list = []
        _install_carl_executor_stub(_make_carl_executor(capture=capture))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--file", str(f),
            "--input", "report.txt=FROM-INPUT",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        _, _, inputs = capture[0]
        # Explicit --input text wins over a --file with the same basename.
        assert inputs["report.txt"] == "FROM-INPUT"

    def test_execute_input_with_equals_in_value(self):
        # `partition("=")` semantics: only the first `=` splits, so
        # `--input expr=a=b` becomes `{expr: "a=b"}`.
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        capture: list = []
        _install_carl_executor_stub(_make_carl_executor(capture=capture))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--input", "expr=a=b=c",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        _, _, inputs = capture[0]
        assert inputs == {"expr": "a=b=c"}

    def test_execute_input_without_equals_returns_two(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor())
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--input", "just-a-key",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 2
        assert "KEY=VALUE" in err

    def test_execute_input_with_empty_key_returns_two(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor())
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--input", "=value",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 2
        assert "empty key" in err

    def test_save_result_persists_memory_card(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--save-result", "weather-run-2026-05-20",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        assert "saved-result: card-saved" in out
        assert len(stub.save_memory_card_calls) == 1
        call = stub.save_memory_card_calls[0]
        # Canonical tags (matches `record_run_completion` /
        # `fetch_run_history`'s expectations).
        assert "agent_run" in call["tags"]
        assert "agent:ent-abc" in call["tags"]
        assert "status:success" in call["tags"]
        # User-supplied --save-result NAME lands as a `label:<name>` tag.
        assert "label:weather-run-2026-05-20" in call["tags"]
        # Card body uses the canonical `category=agent_run`
        # shape that InspectionScreen's RunHistory tab parses.
        card = call["card"]
        assert card["category"] == "agent_run"
        assert card["usage"]["agent_entity_id"] == "ent-abc"
        assert card["usage"]["agent_name"]  # falls back to chain_id
        assert "metrics" in card["usage"]
        assert "duration_seconds" in card["usage"]["metrics"]

    def test_save_result_carries_task(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--task", "What is the weather in Paris?",
            "--input", "city=Paris",
            "--save-result", "weather-paris",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        card = stub.save_memory_card_calls[0]["card"]
        # `--task` lands on `task_description` (the canonical
        # field the run-history projector reads).
        assert card["task_description"] == "What is the weather in Paris?"

    def test_save_result_uses_chain_display_name_when_available(self):
        chain_with_name = dict(_runnable_chain_dict())
        chain_with_name["metadata"] = {
            "care": {"display_name": "Storm Watcher"},
        }
        stub = _StubMemory(chain=chain_with_name)
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--save-result", "wx-2026-05-20",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        card = stub.save_memory_card_calls[0]["card"]
        assert card["usage"]["agent_name"] == "Storm Watcher"

    def test_save_result_without_execute_returns_two(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        ns = _parse([
            "run", "ent-abc", "--save-result", "without-exec",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        assert code == 2
        assert "requires --execute" in err
        # No execution, no save attempted.
        assert stub.save_memory_card_calls == []

    def test_save_result_skipped_on_run_failure(self):
        # When the executor returns success=False, the card isn't
        # saved (avoids polluting Memory with failed runs).
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=False))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--save-result", "should-not-save",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 1
        assert stub.save_memory_card_calls == []
        assert "saved-result" not in out

    def test_save_result_card_save_failure_returns_two(self):
        stub = _StubMemory(
            chain=_runnable_chain_dict(),
            raise_on_save_card=True,
        )
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--save-result", "will-fail",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 2
        assert "save-result failed" in err
        assert "save-card-down" in err

    def test_save_result_attempts_run_count_bump(self):
        # `record_run_completion` calls `memory.client._record_run`
        # to bump the agent's run_count. We track that call via a
        # bound counter on the stub client so the test pins the
        # contract end-to-end.
        stub = _StubMemory(chain=_runnable_chain_dict())
        record_calls: list = []

        class _StubClient:
            def _record_run(self, entity_type, entity_id, *, run_id):
                record_calls.append((entity_type, entity_id, run_id))

        stub.client = _StubClient()
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse([
            "run", "ent-abc", "--execute",
            "--save-result", "tracked",
        ])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        assert record_calls
        assert record_calls[0][1] == "ent-abc"

    def test_execute_with_json_output(self):
        stub = _StubMemory(chain=_runnable_chain_dict())
        _install_memory_stub(stub)
        _install_carl_executor_stub(_make_carl_executor(success=True))
        ns = _parse(["run", "ent-abc", "--execute", "--json"])
        code, out, err = _run_handler(_cmd_run, ns)
        _restore_carl_executor_stub()
        assert code == 0
        # Two JSON blobs land back-to-back; split on the boundary.
        # The simplest robust check: every documented field appears
        # somewhere in stdout, and "executed" key marks the
        # execution summary.
        assert '"executed": true' in out
        assert '"success": true' in out

    def test_run_subcommand_via_main(self, capsys):
        _install_memory_stub(_StubMemory(chain=_runnable_chain_dict()))
        code = main(["run", "ent-abc"])
        _restore_memory_stub()
        assert code == 0
        captured = capsys.readouterr()
        assert "chain: ent-abc" in captured.out


# ---------------------------------------------------------------------------
# care generate
# ---------------------------------------------------------------------------


class _StubMageResult:
    def __init__(self, *, chain_dict=None, mode="fast"):
        # `is None` (not `or`) — an empty dict is a meaningful
        # value the empty-result test wants to assert on.
        if chain_dict is None:
            chain_dict = {
                "task_description": "demo",
                "steps": [
                    {"number": 1, "step_type": "llm", "aim": "ping"},
                    {"number": 2, "step_type": "llm", "aim": "pong"},
                ],
            }
        self.chain_dict = chain_dict
        self.mode = mode


class _StubMageGenerator:
    """Captures `generate` calls + returns the canned result."""

    def __init__(
        self,
        *,
        result=None,
        raise_on_generate=False,
        empty_chain_dict=False,
    ):
        self._result = result or _StubMageResult()
        self._raise = raise_on_generate
        self._empty = empty_chain_dict
        self.generate_calls: list[str] = []

    async def generate(self, query, **kw):
        self.generate_calls.append(query)
        if self._raise:
            raise RuntimeError("gen-down")
        if self._empty:
            return _StubMageResult(chain_dict={})
        return self._result


def _install_mage_stub(generator):
    cli_mod._BUILD_MAGE_OVERRIDE = lambda mode: generator


def _restore_mage_stub():
    cli_mod._BUILD_MAGE_OVERRIDE = None


class TestGenerateCommand:
    def teardown_method(self):
        _restore_mage_stub()
        _restore_memory_stub()

    def test_basic_generation_summary(self):
        gen = _StubMageGenerator()
        _install_mage_stub(gen)
        ns = _parse(["generate", "weather report"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 0
        assert gen.generate_calls == ["weather report"]
        assert "generated chain: 2 steps" in out
        assert "mode=fast" in out

    def test_auth_error_gets_friendly_hint(self):
        class _AuthErr(Exception):
            status_code = 403

        class _Gen:
            async def generate(self, q):
                raise _AuthErr("Token expired")

        _install_mage_stub(_Gen())
        ns = _parse(["generate", "x"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 2
        assert "expired or invalid" in err
        assert "care init" in err
        # The raw "attempt N/3" noise must NOT be the message.
        assert "attempt" not in err

    def test_mode_passed_through_override(self):
        captured_mode: list = []

        def _build(mode):
            captured_mode.append(mode)
            return _StubMageGenerator()

        cli_mod._BUILD_MAGE_OVERRIDE = _build
        ns = _parse(["generate", "task", "--mode", "deep"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 0
        assert captured_mode == ["deep"]

    def test_json_output(self):
        _install_mage_stub(_StubMageGenerator())
        ns = _parse(["generate", "task", "--json"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["task_description"] == "demo"
        assert len(payload["steps"]) == 2

    def test_save_persists_via_memory(self):
        _install_mage_stub(_StubMageGenerator())
        memory = _StubMemory()
        _install_memory_stub(memory)
        ns = _parse(["generate", "weather report", "--save", "Storm Watcher"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 0
        assert len(memory.save_chain_calls) == 1
        call = memory.save_chain_calls[0]
        assert call["name"] == "Storm Watcher"
        assert call["query"] == "weather report"
        assert "saved: ent-saved" in out

    def test_save_failure_returns_two(self):
        _install_mage_stub(_StubMageGenerator())
        _install_memory_stub(_StubMemory(raise_on_save=True))
        ns = _parse(["generate", "task", "--save", "X"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 2
        assert "save failed" in err
        assert "save-down" in err

    def test_output_writes_file(self, tmp_path: Path):
        _install_mage_stub(_StubMageGenerator())
        dest = tmp_path / "out.json"
        ns = _parse(["generate", "task", "--output", str(dest)])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 0
        assert dest.exists()
        assert "exported:" in out
        body = json.loads(dest.read_text(encoding="utf-8"))
        assert body["task_description"] == "demo"

    def test_output_failure_returns_two(self, tmp_path: Path):
        _install_mage_stub(_StubMageGenerator())
        # No extension + no --output-format → ChainExportError
        dest = tmp_path / "noext"
        ns = _parse(["generate", "task", "--output", str(dest)])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 2
        assert "export failed" in err

    def test_generation_failure_returns_two(self):
        _install_mage_stub(_StubMageGenerator(raise_on_generate=True))
        ns = _parse(["generate", "task"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 2
        assert "generation failed" in err
        assert "gen-down" in err

    def test_empty_chain_dict_returns_two(self):
        _install_mage_stub(_StubMageGenerator(empty_chain_dict=True))
        ns = _parse(["generate", "task"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 2
        assert "empty" in err
        assert "nothing to" in err

    def test_mage_build_failure_returns_two(self):
        from care.cli import CliMageError

        def _boom(mode):
            raise CliMageError("mmar_mage isn't installed")

        cli_mod._BUILD_MAGE_OVERRIDE = _boom
        ns = _parse(["generate", "task"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 2
        assert "mmar_mage isn't installed" in err

    def test_generate_subcommand_via_main(self, capsys):
        _install_mage_stub(_StubMageGenerator())
        code = main(["generate", "demo task"])
        _restore_mage_stub()
        assert code == 0
        captured = capsys.readouterr()
        assert "generated chain" in captured.out

    def test_chain_dict_extracted_via_to_dict_fallback(self):
        class _Chain:
            def to_dict(self):
                return {
                    "task_description": "from to_dict",
                    "steps": [{"number": 1, "step_type": "llm", "aim": "x"}],
                }

        class _ResultWithChain:
            chain_dict = None
            chain = _Chain()
            mode = "deep"

        _install_mage_stub(_StubMageGenerator(result=_ResultWithChain()))
        ns = _parse(["generate", "task", "--json"])
        code, out, err = _run_handler(_cmd_generate, ns)
        assert code == 0
        payload = json.loads(out)
        assert payload["task_description"] == "from to_dict"


# ---------------------------------------------------------------------------
# care init (Phase 6 P2)
# ---------------------------------------------------------------------------


def _init_ns(
    tmp_path: Path,
    **overrides,
):
    """Build an argparse namespace for `care init` with sane
    test defaults so each test only specifies what it cares
    about."""
    defaults = {
        "env_path": str(tmp_path / ".env"),
        "api_key": None,
        "base_url": None,
        "model": None,
        "mode": None,
        "force": False,
        "non_interactive": False,
    }
    defaults.update(overrides)
    return _parse(
        ["init"]
        + (["--env-path", defaults["env_path"]] if defaults["env_path"] else [])
        + (["--api-key", defaults["api_key"]] if defaults["api_key"] is not None else [])
        + (["--base-url", defaults["base_url"]] if defaults["base_url"] is not None else [])
        + (["--model", defaults["model"]] if defaults["model"] is not None else [])
        + (["--mode", defaults["mode"]] if defaults["mode"] is not None else [])
        + (["--force"] if defaults["force"] else [])
        + (["--non-interactive"] if defaults["non_interactive"] else []),
    )


class TestRunTuiInterruptHandling:
    """`_run_tui` swallows ``KeyboardInterrupt`` so Ctrl+C
    during early startup / shutdown exits cleanly with code 130
    instead of dumping a Python traceback on the user's
    terminal. Inside the Textual event loop, Ctrl+C is already
    routed to the app's `global_quit` action — this guard
    catches the gap before/after that loop owns the TTY."""

    def test_keyboard_interrupt_during_startup_exits_130(
        self, monkeypatch, capsys,
    ):
        from care.cli import _run_tui

        def _interrupt():
            raise KeyboardInterrupt

        # Replace the lazy-imported `care.app.run` with a stub
        # that fires the same KeyboardInterrupt the user hits
        # by pressing Ctrl+C before the TUI takes the TTY.
        import care.app as care_app

        monkeypatch.setattr(care_app, "run", _interrupt)
        code = _run_tui()
        captured = capsys.readouterr()
        assert code == 130
        assert "interrupted" in captured.err

    def test_tui_import_failure_returns_2(self, monkeypatch, capsys):
        """Defensive: an import error from `care.app` still
        produces a clean, single-line error and a non-zero
        exit code rather than a traceback."""
        import builtins

        from care import cli as cli_mod

        real_import = builtins.__import__

        def _ban_app(name, *a, **kw):
            if name == "care.app":
                raise ImportError("synthetic: textual missing")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _ban_app)
        code = cli_mod._run_tui()
        captured = capsys.readouterr()
        assert code == 2
        assert "TUI failed to start" in captured.err


class TestInitCommand:
    """Phase 6 P2 — `care init` writes a minimal `.env` so a
    fresh checkout can boot the TUI right away."""

    def test_non_interactive_with_all_flags_writes_env(
        self, tmp_path: Path,
    ):
        ns = _init_ns(
            tmp_path,
            non_interactive=True,
            api_key="sk-test-123",
            base_url="https://api.example.test/v1",
            model="custom/model",
            mode="production",
        )
        code, out, err = _run_handler(_cmd_init, ns)
        assert code == 0
        env_file = tmp_path / ".env"
        assert env_file.exists()
        body = env_file.read_text(encoding="utf-8")
        assert "CARE_MAGE__BASE_URL=https://api.example.test/v1" in body
        assert "CARE_MAGE__API_KEY=sk-test-123" in body
        assert "CARE_MAGE__MODEL=custom/model" in body
        assert "CARE_CHAT__DEFAULT_MODE=production" in body
        assert "✓ Wrote" in out
        # No api-key-missing warning when one was provided.
        assert "left blank" not in out

    def test_non_interactive_without_flags_falls_back_to_defaults(
        self, tmp_path: Path,
    ):
        ns = _init_ns(tmp_path, non_interactive=True)
        code, out, err = _run_handler(_cmd_init, ns)
        assert code == 0
        body = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "CARE_MAGE__BASE_URL=https://openrouter.ai/api/v1" in body
        assert "CARE_MAGE__API_KEY=" in body  # blank
        assert "anthropic/claude-3.5-sonnet" in body
        assert "CARE_CHAT__DEFAULT_MODE=interactive" in body
        # Blank api-key warning lands.
        assert "left blank" in out

    def test_refuses_overwrite_without_force(self, tmp_path: Path):
        target = tmp_path / ".env"
        target.write_text("# pre-existing content\n", encoding="utf-8")
        ns = _init_ns(tmp_path, non_interactive=True)
        code, out, err = _run_handler(_cmd_init, ns)
        assert code == 1
        assert "already exists" in err
        assert "--force" in err
        # File untouched.
        assert (
            target.read_text(encoding="utf-8")
            == "# pre-existing content\n"
        )

    def test_force_overwrites_existing(self, tmp_path: Path):
        target = tmp_path / ".env"
        target.write_text("STALE=1\n", encoding="utf-8")
        ns = _init_ns(
            tmp_path,
            non_interactive=True,
            api_key="sk-overwritten",
            force=True,
        )
        code, out, err = _run_handler(_cmd_init, ns)
        assert code == 0
        body = target.read_text(encoding="utf-8")
        assert "STALE=1" not in body
        assert "CARE_MAGE__API_KEY=sk-overwritten" in body

    def test_interactive_prompts_use_stdin_values(self, tmp_path: Path):
        # Four prompts: base_url, api_key, model, mode — provide
        # each on its own line via a StringIO stdin.
        stdin = io.StringIO(
            "https://my.endpoint/v1\n"
            "sk-from-stdin\n"
            "custom-model-99\n"
            "production\n",
        )
        ns = _init_ns(tmp_path)  # no flags, no --non-interactive
        out = io.StringIO()
        err = io.StringIO()
        code = _cmd_init(ns, out, err, stdin=stdin)
        assert code == 0
        body = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "CARE_MAGE__BASE_URL=https://my.endpoint/v1" in body
        assert "CARE_MAGE__API_KEY=sk-from-stdin" in body
        assert "CARE_MAGE__MODEL=custom-model-99" in body
        assert "CARE_CHAT__DEFAULT_MODE=production" in body
        # Prompt labels surfaced on stdout.
        prompts = out.getvalue()
        assert "MAGE base URL" in prompts
        assert "MAGE API key" in prompts
        assert "Model id" in prompts

    def test_interactive_blank_input_falls_back_to_default(
        self, tmp_path: Path,
    ):
        # Empty lines → take the default for that field.
        stdin = io.StringIO("\n\n\n\n")
        ns = _init_ns(tmp_path)
        code = _cmd_init(ns, io.StringIO(), io.StringIO(), stdin=stdin)
        assert code == 0
        body = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "CARE_MAGE__BASE_URL=https://openrouter.ai/api/v1" in body
        assert "CARE_MAGE__API_KEY=" in body  # blank
        assert "anthropic/claude-3.5-sonnet" in body
        assert "CARE_CHAT__DEFAULT_MODE=interactive" in body

    def test_invalid_mode_value_rejected(self, tmp_path: Path):
        # --mode is a choices=... argparse field, so the parser
        # rejects bad values directly. Confirm SystemExit is
        # raised before our handler is reached.
        try:
            _parse([
                "init", "--non-interactive",
                "--env-path", str(tmp_path / ".env"),
                "--mode", "bogus",
            ])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            assert False, "argparse should reject --mode bogus"
        assert not (tmp_path / ".env").exists()

    def test_generated_env_roundtrips_through_load_env_file(
        self, tmp_path: Path, monkeypatch,
    ):
        """Sanity check: care.dotenv.load_env_file can parse what
        care init writes — otherwise the quick-start chain breaks
        downstream."""
        from care.dotenv import load_env_file

        ns = _init_ns(
            tmp_path,
            non_interactive=True,
            api_key="sk-roundtrip",
            base_url="https://rt.test/v1",
            model="rt/model",
            mode="ad_hoc",  # legacy spelling — normalised to "interactive"
        )
        code, _out, _err = _run_handler(_cmd_init, ns)
        assert code == 0

        # Wipe the relevant keys so load_env_file is the only
        # source.
        for key in (
            "CARE_MAGE__BASE_URL",
            "CARE_MAGE__API_KEY",
            "CARE_MAGE__MODEL",
            "CARE_CHAT__DEFAULT_MODE",
        ):
            monkeypatch.delenv(key, raising=False)

        load_env_file(tmp_path / ".env")
        import os
        assert os.environ["CARE_MAGE__BASE_URL"] == "https://rt.test/v1"
        assert os.environ["CARE_MAGE__API_KEY"] == "sk-roundtrip"
        assert os.environ["CARE_MAGE__MODEL"] == "rt/model"
        assert os.environ["CARE_CHAT__DEFAULT_MODE"] == "interactive"

    def test_write_failure_returns_nonzero(
        self, tmp_path: Path, monkeypatch,
    ):
        """OSError during write surfaces a friendly message and
        a non-zero exit, not a stack trace."""
        target = tmp_path / "nested" / ".env"

        def _raise(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(
            "pathlib.Path.write_text", _raise,
        )
        ns = _init_ns(
            tmp_path,
            env_path=str(target),
            non_interactive=True,
            api_key="sk",
        )
        code, out, err = _run_handler(_cmd_init, ns)
        assert code == 2
        assert "failed to write" in err
        assert "disk full" in err

    def test_secret_default_redacted_in_prompt(self, tmp_path: Path):
        """When --api-key is omitted but the prompt has no
        default to redact (empty), no `***` appears. When the
        helper is called with a non-empty secret default, the
        prompt redacts it."""
        from care.cli import _prompt_with_default

        stdin_empty = io.StringIO("\n")
        out = io.StringIO()
        result = _prompt_with_default(
            stdin=stdin_empty, stdout=out,
            label="API key", default="existing-secret", secret=True,
        )
        assert result == "existing-secret"
        # Secret default is redacted in the visible prompt.
        prompt = out.getvalue()
        assert "***" in prompt
        assert "existing-secret" not in prompt


class TestRunFileInputs:
    """`care run --file` attachment + the missing-context-files warning."""

    def test_parser_accepts_repeated_file(self):
        ns = _parse(["run", "chain-1", "--file", "a.txt", "--file", "b.txt"])
        assert ns.file == ["a.txt", "b.txt"]

    def test_read_file_inputs_keys_by_basename(self, tmp_path: Path):
        f = tmp_path / "report.txt"
        f.write_text("body")
        out = _read_file_inputs([str(f)])
        # Keyed by basename so ${input.report.txt} resolves.
        assert out == {"report.txt": "body"}

    def test_read_file_inputs_binary_safe(self, tmp_path: Path):
        # A binary file must NOT crash the CLI (old read_text raised
        # UnicodeDecodeError); the canonical loader reads it safely.
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\x80")
        out = _read_file_inputs([str(f)])
        assert "blob.bin" in out
        assert isinstance(out["blob.bin"], str)

    def test_read_file_inputs_none_is_empty(self):
        assert _read_file_inputs(None) == {}

    def test_read_file_inputs_rejects_missing(self):
        import pytest

        with pytest.raises(ValueError, match="not a file"):
            _read_file_inputs(["/no/such/file-xyz.txt"])

    def test_warn_lists_missing_context_files(self, tmp_path: Path):
        ghost = tmp_path / "ghost.txt"  # never created
        chain = {
            "metadata": {
                "care": {
                    "task_description": "t",
                    "context_files": [{"path": str(ghost), "sha256": "x"}],
                },
            },
        }
        err = io.StringIO()
        _warn_missing_context_files(chain, {}, err)
        msg = err.getvalue()
        assert "ghost.txt" in msg
        assert "warning" in msg.lower()

    def test_warn_silent_when_file_supplied_via_input(self, tmp_path: Path):
        ghost = tmp_path / "ghost.txt"
        chain = {
            "metadata": {
                "care": {
                    "task_description": "t",
                    "context_files": [{"path": str(ghost), "sha256": "x"}],
                },
            },
        }
        err = io.StringIO()
        # The basename is provided as a runtime input → no warning.
        _warn_missing_context_files(chain, {"ghost.txt": "data"}, err)
        assert err.getvalue() == ""

    def test_warn_silent_when_files_present(self, tmp_path: Path):
        here = tmp_path / "here.txt"
        here.write_text("ok")
        chain = {
            "metadata": {
                "care": {
                    "task_description": "t",
                    "context_files": [{"path": str(here), "sha256": "x"}],
                },
            },
        }
        err = io.StringIO()
        _warn_missing_context_files(chain, {}, err)
        assert err.getvalue() == ""

    def test_warn_silent_without_metadata(self):
        err = io.StringIO()
        _warn_missing_context_files({"steps": []}, {}, err)
        assert err.getvalue() == ""

    def test_skill_bridge_rewrites_and_merges(self, tmp_path: Path):
        from care.cli import _apply_cli_skill_bridge

        f = tmp_path / "r.txt"
        f.write_text("DOC")
        chain_dict = {
            "task_description": "summarise",
            "steps": [{
                "number": 1, "title": "Extract", "step_type": "agent_skill",
                "aim": "read docx",
                "step_config": {
                    "skill": "github://anthropics/skills/skills/docx@main",
                    "task": "Extract text from the provided DOCX",
                    "execution_mode": "llm", "input_mapping": {},
                    "output_key": "t",
                },
            }],
        }
        new_dict, inputs = _apply_cli_skill_bridge(chain_dict, [str(f)], {})
        mapping = new_dict["steps"][0]["step_config"]["input_mapping"]
        assert any(str(v).startswith("$memory.input.") for v in mapping.values())
        assert "DOC" in inputs.values()

    def test_skill_bridge_noop_without_files(self):
        from care.cli import _apply_cli_skill_bridge

        cd = {"steps": []}
        assert _apply_cli_skill_bridge(cd, None, {}) == (cd, {})

    def test_classify_files_flag(self):
        assert _parse(["run", "id", "--file", "a.txt"]).classify_files == (
            "heuristic"
        )
        assert _parse(
            ["run", "id", "--classify-files", "model"],
        ).classify_files == "model"

    @staticmethod
    def _create_docx_chain() -> dict:
        # A CREATE step the keyword heuristic does NOT flag as a file reader.
        return {
            "steps": [{
                "number": 1, "step_type": "agent_skill",
                "title": "Create report", "aim": "create a docx file",
                "step_config": {
                    "skill": "github://anthropics/skills/skills/docx@main",
                    "task": "Create a new Word document",
                    "input_mapping": {},
                },
            }],
        }

    def test_heuristic_default_does_not_flag_create_step(self, tmp_path: Path):
        from care.cli import _apply_cli_skill_bridge

        f = tmp_path / "r.txt"
        f.write_text("DOC")
        cd = self._create_docx_chain()
        new_dict, inputs = _apply_cli_skill_bridge(
            cd, [str(f)], {}, classify="heuristic",
        )
        assert new_dict is cd  # not rewritten
        assert inputs == {}

    def test_model_classify_overrides_heuristic(self, tmp_path: Path):
        import care.cli as cli_mod

        f = tmp_path / "r.txt"
        f.write_text("DOC")

        class _StubApi:
            async def get_response_with_retries(self, prompt, retries):
                return '{"1": true}'

        cli_mod._BUILD_CLASSIFIER_API_OVERRIDE = lambda: _StubApi()
        try:
            new_dict, inputs = cli_mod._apply_cli_skill_bridge(
                self._create_docx_chain(), [str(f)], {}, classify="model",
            )
        finally:
            cli_mod._BUILD_CLASSIFIER_API_OVERRIDE = None
        mapping = new_dict["steps"][0]["step_config"]["input_mapping"]
        assert any(str(v).startswith("$memory.input.") for v in mapping.values())
        assert "DOC" in inputs.values()