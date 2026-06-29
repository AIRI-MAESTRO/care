"""Tests for ``care.tools.load_tools_into_context`` (TODO §5 P1).

The loader's contract is narrow: build the right glob from the
configured directory and forward the right kwargs to CARL's
:meth:`ReasoningContext.register_tools_from_path`. CARL owns the
actual import + decorator inspection (heavily tested in
``carl-experiments``), so CARE's tests verify the wiring + the
no-op-when-dir-missing behaviour. The context side is a
duck-typed stub — the loader is documented to accept anything
with that method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from care.config import CareConfig, ToolsConfig
from care.tools import LoadedTools, load_tools_into_context


class _StubContext:
    """Captures every call so we can assert on it."""

    def __init__(self, return_value: list[str] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.return_value = return_value or []

    def register_tools_from_path(
        self,
        glob: str,
        *,
        tag_filter: list[str] | None = None,
        name_prefix: str = "",
    ) -> list[str]:
        self.calls.append(
            {
                "glob": glob,
                "tag_filter": tag_filter,
                "name_prefix": name_prefix,
            }
        )
        return list(self.return_value)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_builds_glob_under_directory(self, tmp_path: Path):
        cfg = ToolsConfig(path=tmp_path)
        ctx = _StubContext(return_value=["weather", "search"])
        result = load_tools_into_context(ctx, cfg)
        # Glob is `<dir>/*.py`.
        assert len(ctx.calls) == 1
        assert ctx.calls[0]["glob"].endswith("/*.py")
        assert ctx.calls[0]["glob"].startswith(str(tmp_path))
        # Names round-trip from the context.
        assert result.names == ("weather", "search")
        assert result.directory == tmp_path.resolve()
        assert result.skipped is False

    def test_accepts_full_care_config(self, tmp_path: Path):
        # Loader unwraps `.tools` from a full CareConfig.
        full = CareConfig(tools=ToolsConfig(path=tmp_path))
        ctx = _StubContext(return_value=["x"])
        result = load_tools_into_context(ctx, full)
        assert result.names == ("x",)
        assert len(ctx.calls) == 1

    def test_tag_filter_and_name_prefix_forwarded(self, tmp_path: Path):
        cfg = ToolsConfig(
            path=tmp_path,
            tag_filter=["external", "search"],
            name_prefix="user_",
        )
        ctx = _StubContext()
        load_tools_into_context(ctx, cfg)
        call = ctx.calls[0]
        assert call["tag_filter"] == ["external", "search"]
        assert call["name_prefix"] == "user_"

    def test_default_kwargs_match_carl_defaults(self, tmp_path: Path):
        # No tag_filter / name_prefix → forwarded as None / "".
        cfg = ToolsConfig(path=tmp_path)
        ctx = _StubContext()
        load_tools_into_context(ctx, cfg)
        call = ctx.calls[0]
        assert call["tag_filter"] is None
        assert call["name_prefix"] == ""

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Path field accepts str-likes; the loader handles ~.
        cfg = ToolsConfig(path=Path("~/.config/care/tools"))
        target = tmp_path / ".config" / "care" / "tools"
        target.mkdir(parents=True)
        ctx = _StubContext()
        result = load_tools_into_context(ctx, cfg)
        # The directory the loader scanned is the resolved tilde path.
        assert result.directory == target.resolve()
        assert ctx.calls[0]["glob"].startswith(str(target.resolve()))


# ---------------------------------------------------------------------------
# Missing-directory no-op
# ---------------------------------------------------------------------------


class TestMissingDir:
    def test_missing_dir_returns_skipped_without_calling_context(
        self, tmp_path: Path
    ):
        cfg = ToolsConfig(path=tmp_path / "does-not-exist")
        ctx = _StubContext()
        result = load_tools_into_context(ctx, cfg)
        assert result.skipped is True
        assert result.names == ()
        # Context was never touched — no spurious import attempts.
        assert ctx.calls == []
        # Directory is reported as-resolved so the TUI banner can
        # show what was checked.
        assert result.directory == (tmp_path / "does-not-exist").resolve()

    def test_empty_dir_is_loaded_not_skipped(self, tmp_path: Path):
        """Empty directory ≠ missing directory. An existing empty
        dir means "user has the tools dir set up, just no tools
        yet" — the loader still calls CARL (which returns ``[]``)
        so the user can confirm CARL was reachable."""
        cfg = ToolsConfig(path=tmp_path)
        ctx = _StubContext(return_value=[])
        result = load_tools_into_context(ctx, cfg)
        assert result.skipped is False
        assert result.names == ()
        assert len(ctx.calls) == 1


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_loaded_tools_is_frozen(self, tmp_path: Path):
        cfg = ToolsConfig(path=tmp_path)
        ctx = _StubContext(return_value=["a"])
        result = load_tools_into_context(ctx, cfg)
        assert isinstance(result, LoadedTools)
        with pytest.raises(Exception):
            result.names = ("b",)  # type: ignore[misc]

    def test_names_preserves_carl_ordering(self, tmp_path: Path):
        # CARL returns in discovery order; CARE shouldn't sort.
        cfg = ToolsConfig(path=tmp_path)
        ctx = _StubContext(return_value=["zeta", "alpha", "mu"])
        result = load_tools_into_context(ctx, cfg)
        assert result.names == ("zeta", "alpha", "mu")


# ---------------------------------------------------------------------------
# Integration: ToolsConfig defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_default_path_points_at_care_config_tools(self):
        cfg = ToolsConfig()
        # Default is ``~/.config/care/tools`` — keep this stable so
        # docs / first-run wizards can reference it.
        assert str(cfg.path) == "~/.config/care/tools"

    def test_default_tag_filter_is_none(self):
        # ``None`` means "register all tools" (matches CARL's docs).
        assert ToolsConfig().tag_filter is None

    def test_default_name_prefix_is_empty(self):
        assert ToolsConfig().name_prefix == ""

    def test_care_config_has_tools_section(self):
        full = CareConfig()
        assert isinstance(full.tools, ToolsConfig)
        # Default propagates through the full config.
        assert str(full.tools.path) == "~/.config/care/tools"
