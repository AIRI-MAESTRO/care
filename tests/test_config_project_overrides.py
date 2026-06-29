"""Tests for per-project ``./care.toml`` overrides (TODO §2 P2).

CARE's config precedence is documented as

    defaults < ~/.config/care/config.toml < ./care.toml < $CARE_*

This file pins the project-layer behaviour without touching the
user's real ``~/.config/care/config.toml``: we monkeypatch
``DEFAULT_CONFIG_PATH`` onto ``tmp_path`` so each test gets a
sandboxed user-global file alongside a sandboxed cwd-style
``care.toml``.

Why a separate file from ``test_config.py``: the existing tests
all pass an explicit ``path=``, which bypasses the new project-
file lookup. These tests exercise the **implicit** code path
(``path=None``) where the new layering kicks in.
"""

from __future__ import annotations

from pathlib import Path

import care.config as cfg_mod
from care.config import (
    PROJECT_CONFIG_FILENAME,
    CareConfig,
)


def _write_user(home_root: Path, body: str) -> Path:
    """Write a user-global ``config.toml`` inside the sandboxed home."""
    user_config = home_root / ".config" / "care" / "config.toml"
    user_config.parent.mkdir(parents=True, exist_ok=True)
    user_config.write_text(body, encoding="utf-8")
    return user_config


def _write_project(cwd: Path, body: str) -> Path:
    p = cwd / PROJECT_CONFIG_FILENAME
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Precedence: defaults → user → project → env
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_project_only(self, tmp_path: Path, monkeypatch):
        """No user-global file; project-only values come through."""
        monkeypatch.setattr(
            cfg_mod, "DEFAULT_CONFIG_PATH", tmp_path / "no-user.toml"
        )
        _write_project(
            tmp_path,
            '[mage]\nmode = "fast"\nbase_url = "https://e.example"\n',
        )
        c = CareConfig.load(env={}, cwd=tmp_path)
        assert c.mage.mode == "fast"
        assert c.mage.base_url == "https://e.example"
        # Defaults for everything not in the file.
        assert c.memory.base_url == "http://localhost:8000"

    def test_user_only(self, tmp_path: Path, monkeypatch):
        """No project file; user-global values come through."""
        user_root = tmp_path / "home"
        user_root.mkdir()
        user_cfg = _write_user(user_root, '[mage]\nmode = "fast"\n')
        monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", user_cfg)
        # cwd is `tmp_path` (no project file there).
        c = CareConfig.load(env={}, cwd=tmp_path)
        assert c.mage.mode == "fast"

    def test_project_overrides_user(self, tmp_path: Path, monkeypatch):
        """When both files exist, project wins per key."""
        user_root = tmp_path / "home"
        user_root.mkdir()
        user_cfg = _write_user(
            user_root,
            '[mage]\nmode = "fast"\nbase_url = "https://u.example"\n'
            '[memory]\nbase_url = "http://user:8000"\n',
        )
        monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", user_cfg)
        _write_project(
            tmp_path,
            # Project flips mode + memory.base_url; keeps mage.base_url
            # from user.
            '[mage]\nmode = "deep"\n'
            '[memory]\nbase_url = "http://project:9000"\n',
        )
        c = CareConfig.load(env={}, cwd=tmp_path)
        # Project wins.
        assert c.mage.mode == "deep"
        assert c.memory.base_url == "http://project:9000"
        # User layer preserved for keys project didn't touch.
        assert c.mage.base_url == "https://u.example"

    def test_env_still_wins_over_project(self, tmp_path: Path, monkeypatch):
        """Env > project > user > defaults."""
        user_root = tmp_path / "home"
        user_root.mkdir()
        user_cfg = _write_user(user_root, '[mage]\nmode = "fast"\n')
        monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", user_cfg)
        _write_project(tmp_path, '[mage]\nmode = "deep"\n')
        c = CareConfig.load(
            env={"CARE_MAGE__MODE": "fast"},
            cwd=tmp_path,
        )
        assert c.mage.mode == "fast"

    def test_nested_dict_merge_not_replace(self, tmp_path: Path, monkeypatch):
        """Project-side `[mage]` should not wipe user-side `[mage]`
        keys that project doesn't mention — `_deep_merge` semantics."""
        user_root = tmp_path / "home"
        user_root.mkdir()
        user_cfg = _write_user(
            user_root,
            '[mage]\nmode = "fast"\nbase_url = "https://u.example"\n'
            'api_key = "sk-user"\nenable_web_research = true\n',
        )
        monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", user_cfg)
        _write_project(tmp_path, '[mage]\nmode = "deep"\n')
        c = CareConfig.load(env={}, cwd=tmp_path)
        # Project changed only `mode`.
        assert c.mage.mode == "deep"
        # All other user-side `[mage]` keys survive.
        assert c.mage.base_url == "https://u.example"
        assert c.mage.api_key == "sk-user"
        assert c.mage.enable_web_research is True


# ---------------------------------------------------------------------------
# Cwd behaviour
# ---------------------------------------------------------------------------


class TestCwdBehaviour:
    def test_missing_cwd_file_is_silent(self, tmp_path: Path, monkeypatch):
        """No `care.toml` in cwd ⇒ user-global / defaults apply
        without any error."""
        user_root = tmp_path / "home"
        user_root.mkdir()
        user_cfg = _write_user(user_root, '[mage]\nmode = "fast"\n')
        monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG_PATH", user_cfg)
        c = CareConfig.load(env={}, cwd=tmp_path)
        assert c.mage.mode == "fast"

    def test_cwd_defaults_to_path_cwd(
        self, tmp_path: Path, monkeypatch
    ):
        """Without `cwd=`, the loader uses `Path.cwd()`."""
        monkeypatch.setattr(
            cfg_mod, "DEFAULT_CONFIG_PATH", tmp_path / "no-user.toml"
        )
        _write_project(tmp_path, '[mage]\nmode = "deep"\n')
        monkeypatch.chdir(tmp_path)
        c = CareConfig.load(env={})
        assert c.mage.mode == "deep"

    def test_explicit_path_skips_project_lookup(
        self, tmp_path: Path, monkeypatch
    ):
        """Passing `path=` is treated as the single source of truth —
        the cwd `care.toml` is NOT layered on. Lets
        `care --config foo.toml` produce predictable output."""
        # Project file says "deep"; explicit path says "fast" — explicit wins.
        _write_project(tmp_path, '[mage]\nmode = "deep"\n')
        explicit = tmp_path / "explicit.toml"
        explicit.write_text('[mage]\nmode = "fast"\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        c = CareConfig.load(path=explicit, env={})
        assert c.mage.mode == "fast"


# ---------------------------------------------------------------------------
# Constant exposure
# ---------------------------------------------------------------------------


class TestConstants:
    def test_project_config_filename_is_care_toml(self):
        # Pin so docs / first-run wizard can reference it without
        # importing internals.
        assert PROJECT_CONFIG_FILENAME == "care.toml"
