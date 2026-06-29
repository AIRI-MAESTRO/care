"""C5 — /versions: version history + channel markers + diff, with the rollback hint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from care.screens.chat import (
    _COMMAND_HANDLERS,
    ChatScreen,
    _format_version_row,
    _parse_versions_args,
)


class TestParse:
    def test_bare_list(self):
        assert _parse_versions_args("chain-1") == ("chain-1", None)

    def test_diff(self):
        ref, pair = _parse_versions_args("my chain diff vid-1 vid-2")
        assert ref == "my chain"
        assert pair == ("vid-1", "vid-2")

    def test_name_with_spaces_no_diff(self):
        assert _parse_versions_args("weather agent") == ("weather agent", None)


def test_format_row_marks_channels_and_score():
    v = SimpleNamespace(
        version_number=3,
        version_id="vid-0003abcd",
        created_at="2026-06-10T12:00:00+00:00",
        evolution_meta={"fitness_score": 0.83},
        change_summary="added a validation step",
    )
    row = _format_version_row(v, {"vid-0003abcd": ["latest", "stable"]})
    assert "v3" in row
    assert "fitness 0.83" in row
    assert "added a validation step" in row
    assert "← latest, stable" in row


def test_versions_registered_and_usage():
    assert "versions" in _COMMAND_HANDLERS
    posted: list[dict[str, Any]] = []
    screen = SimpleNamespace(
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"text": text, "severity": severity}
        ),
        run_worker=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no worker")),
    )
    _COMMAND_HANDLERS["versions"](screen, "")
    assert posted and "Usage:" in posted[0]["text"]


# ----------------------------------------------------------------- worker
def _version(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        version_number=n,
        version_id=f"vid-{n:04d}",
        created_at=f"2026-06-1{n}T00:00:00+00:00",
        evolution_meta={"fitness_score": 0.5 + n / 10},
        change_summary=f"change {n}",
    )


class FakeClient:
    def __init__(
        self,
        *,
        versions: list[Any] | None = None,
        latest_vid: str = "vid-0003",
        stable_vid: str = "vid-0002",
        diff_patch: dict | None = None,
    ) -> None:
        self.versions = versions if versions is not None else [_version(1), _version(2), _version(3)]
        self.latest_vid = latest_vid
        self.stable_vid = stable_vid
        self.diff_patch = diff_patch
        self.diff_calls: list[tuple[str, str, str]] = []

    def get_chain_record(self, entity_id: str, *, channel: str = "latest") -> Any:
        vid = self.latest_vid if channel == "latest" else self.stable_vid
        return SimpleNamespace(
            entity_id=entity_id, version_id=vid, version_number=3,
            meta={"display_name": "Weather"}, content={"name": "Weather"},
        )

    def list_versions(self, entity_id: str, entity_type: str = "chain", limit: int = 20) -> list[Any]:
        return list(self.versions)

    def diff_versions(self, entity_id: str, from_version: str, to_version: str, entity_type: str = "chain") -> Any:
        self.diff_calls.append((entity_id, from_version, to_version))
        return SimpleNamespace(from_version=from_version, to_version=to_version, patch=self.diff_patch or {})


def _fake_self(client: Any) -> SimpleNamespace:
    posted: list[dict[str, Any]] = []
    fake = SimpleNamespace(
        app=SimpleNamespace(memory=SimpleNamespace(client=client)),
        _post_line=lambda role, text, severity=None, **_: posted.append(
            {"role": role, "text": text, "severity": severity}
        ),
        posted=posted,
    )
    fake._fetch_chain_record = ChatScreen._fetch_chain_record.__get__(fake)
    fake._find_chain_by_name = ChatScreen._find_chain_by_name.__get__(fake)
    fake._versions_display_name = ChatScreen._versions_display_name  # staticmethod
    fake._post_version_diff = ChatScreen._post_version_diff.__get__(fake)
    return fake


async def test_lists_versions_newest_first_with_channels():
    self = _fake_self(FakeClient())
    await ChatScreen._run_versions(self, "chain-1")
    text = "\n".join(p["text"] for p in self.posted)
    assert "Versions of Weather" in text
    # newest first
    rows = [p["text"] for p in self.posted if p["text"].startswith("● ")]
    assert rows[0].startswith("● v3")
    assert "← latest" in rows[0]  # v3 is latest
    assert "← stable" in [r for r in rows if r.startswith("● v2")][0]
    assert "/rollback chain-1 --to <version-id>" in text


async def test_no_versions():
    self = _fake_self(FakeClient(versions=[]))
    await ChatScreen._run_versions(self, "chain-1")
    assert any("No versions found" in p["text"] for p in self.posted)


async def test_diff_subcommand():
    self = _fake_self(FakeClient(diff_patch={"op": "replace", "path": "/name"}))
    await ChatScreen._run_versions(self, "chain-1 diff vid-0001 vid-0003")
    assert self.app.memory.client.diff_calls == [("chain-1", "vid-0001", "vid-0003")]
    text = "\n".join(p["text"] for p in self.posted)
    assert "Diff vid-0001" in text
    assert "replace" in text


async def test_diff_no_changes():
    self = _fake_self(FakeClient(diff_patch={}))
    await ChatScreen._run_versions(self, "chain-1 diff vid-0001 vid-0002")
    assert any("No differences" in p["text"] for p in self.posted)


async def test_unresolvable_ref():
    class NoChainClient:
        def get_chain_record(self, *a: Any, **k: Any) -> Any:
            raise KeyError("nope")

    self = _fake_self(NoChainClient())
    await ChatScreen._run_versions(self, "ghost")
    errors = [p for p in self.posted if p["severity"] == "error"]
    assert errors and "Could not resolve" in errors[0]["text"]
