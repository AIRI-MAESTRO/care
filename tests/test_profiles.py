"""Tests for `care.runtime.profiles` (TODO §6 P2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from care.runtime.profiles import (
    ProfileInfo,
    active_profile_name,
    list_profiles,
    profile_path,
    profiles_dir,
)


class TestProfilesDir:
    def test_resolves_under_config_dir(self, tmp_path: Path) -> None:
        assert profiles_dir(config_dir=tmp_path) == (
            tmp_path / "profiles"
        )


class TestActiveProfileName:
    def test_returns_env_value(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CARE_PROFILE", "dev")
        assert active_profile_name() == "dev"

    def test_strips_whitespace(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CARE_PROFILE", "  prod  ")
        assert active_profile_name() == "prod"

    def test_empty_when_unset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CARE_PROFILE", raising=False)
        assert active_profile_name() == ""


class TestProfilePath:
    def test_valid_name_resolves(self, tmp_path: Path) -> None:
        path = profile_path("dev", config_dir=tmp_path)
        assert path == tmp_path / "profiles" / "dev.toml"

    def test_underscore_and_hyphen_accepted(
        self, tmp_path: Path,
    ) -> None:
        path = profile_path(
            "production-east_1", config_dir=tmp_path,
        )
        assert path.name == "production-east_1.toml"

    def test_path_traversal_rejected(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="invalid profile name"):
            profile_path("../etc/passwd", config_dir=tmp_path)

    def test_empty_name_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            profile_path("", config_dir=tmp_path)

    def test_special_chars_rejected(
        self, tmp_path: Path,
    ) -> None:
        for bad in ("dev/x", "dev:x", "dev.x", "dev x"):
            with pytest.raises(ValueError):
                profile_path(bad, config_dir=tmp_path)


class TestListProfiles:
    def test_empty_when_dir_missing(
        self, tmp_path: Path,
    ) -> None:
        assert list_profiles(config_dir=tmp_path) == []

    def test_lists_toml_files_sorted(
        self, tmp_path: Path,
    ) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "zeus.toml").write_text("[mage]\nmodel='x'\n")
        (pdir / "athena.toml").write_text("[mage]\nmodel='y'\n")
        (pdir / "ares.toml").write_text("[mage]\nmodel='z'\n")
        rows = list_profiles(config_dir=tmp_path)
        assert [r.name for r in rows] == ["ares", "athena", "zeus"]
        assert all(isinstance(r, ProfileInfo) for r in rows)

    def test_skips_non_toml(self, tmp_path: Path) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "ok.toml").write_text("[x]")
        (pdir / "ignored.txt").write_text("nope")
        (pdir / "README.md").write_text("nope")
        rows = list_profiles(config_dir=tmp_path)
        assert [r.name for r in rows] == ["ok"]

    def test_skips_unsafe_names(self, tmp_path: Path) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        (pdir / "good.toml").write_text("[x]")
        (pdir / "bad name.toml").write_text("[x]")
        (pdir / "bad.name.toml").write_text("[x]")
        rows = list_profiles(config_dir=tmp_path)
        # `bad.name.toml` has stem "bad.name" which fails
        # the whitelist; `bad name.toml` similarly fails.
        assert [r.name for r in rows] == ["good"]

    def test_carries_size_and_mtime(
        self, tmp_path: Path,
    ) -> None:
        pdir = profiles_dir(config_dir=tmp_path)
        pdir.mkdir(parents=True)
        path = pdir / "demo.toml"
        path.write_text("[mage]\nmodel = 'gpt-4o'\n")
        rows = list_profiles(config_dir=tmp_path)
        assert len(rows) == 1
        info = rows[0]
        assert info.path == path
        assert info.size_bytes == len(path.read_bytes())
        assert info.mtime > 0
