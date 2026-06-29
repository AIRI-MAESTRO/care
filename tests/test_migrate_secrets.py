"""Tests for `care.config.migrate_literal_secrets` (TODO §1 P1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from care.config import (
    CareConfig,
    SecretMigrationReport,
    migrate_literal_secrets,
)
from care.runtime.keystore import MemoryKeystore


def _write_config(path: Path, body: str) -> None:
    path.write_text(body)


# ---------------------------------------------------------------------------
# SecretMigrationReport
# ---------------------------------------------------------------------------


class TestReport:
    def test_did_migrate_false_when_empty(self):
        assert SecretMigrationReport().did_migrate is False

    def test_did_migrate_true_when_populated(self):
        r = SecretMigrationReport(
            migrated=(
                ("mage", "api_key",
                 "keystore://care/mage.api_key"),
            ),
        )
        assert r.did_migrate is True

    def test_format_text_zero_migrated(self):
        text = SecretMigrationReport().format_text()
        assert "no literal secrets to migrate" in text

    def test_format_text_lists_migrated_and_skipped(self):
        r = SecretMigrationReport(
            migrated=(("mage", "api_key", "keystore://x/y"),),
            skipped=(("memory.api_key", "empty"),),
        )
        text = r.format_text()
        assert "migrated 1 secret" in text
        assert "mage.api_key" in text
        assert "skipped 1 slot" in text
        assert "memory.api_key — empty" in text


# ---------------------------------------------------------------------------
# migrate_literal_secrets
# ---------------------------------------------------------------------------


class TestMigrateLiteralSecrets:
    def test_literal_offloaded_and_url_written_back(
        self, tmp_path: Path,
    ):
        config_path = tmp_path / "care.toml"
        _write_config(config_path, (
            '[mage]\n'
            'api_key = "sk-literal-7"\n'
            'base_url = "https://api.openai.com/v1"\n'
        ))
        config = CareConfig.load(path=config_path, env={})
        # Read-side resolves literals as-is, so the config
        # carries the literal.
        assert config.mage.api_key == "sk-literal-7"

        store = MemoryKeystore()
        report = migrate_literal_secrets(
            config, path=config_path, keystore=store,
        )

        assert report.did_migrate is True
        assert len(report.migrated) == 1
        parent, field, url = report.migrated[0]
        assert (parent, field) == ("mage", "api_key")
        assert url == "keystore://care/mage.api_key"

        # In-memory config now carries the URL.
        assert config.mage.api_key == url

        # TOML file rewritten.
        text = config_path.read_text()
        assert "sk-literal-7" not in text
        assert "keystore://care/mage.api_key" in text

        # Keystore actually holds the secret.
        assert store.fetch("care", "mage.api_key") == "sk-literal-7"

    def test_url_passes_through_without_re_storing(
        self, tmp_path: Path,
    ):
        store = MemoryKeystore()
        store.store("care", "mage.api_key", "pre-existing")
        # Construct a CareConfig directly with the URL string
        # so we exercise the migrate-side idempotence without
        # involving the read-side dereference (which would
        # touch the real OS keystore).
        config = CareConfig.model_validate({
            "mage": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "keystore://care/mage.api_key",
            },
        })
        report = migrate_literal_secrets(
            config, path=tmp_path / "care.toml", keystore=store,
        )
        assert report.did_migrate is False
        # The slot is in `skipped` with the "already a URL"
        # reason.
        assert any(
            slot == "mage.api_key" and "URL" in reason
            for slot, reason in report.skipped
        )
        # Pre-existing keystore value is untouched.
        assert store.fetch("care", "mage.api_key") == "pre-existing"

    def test_empty_value_is_skipped(self, tmp_path: Path):
        config = CareConfig.model_validate({
            "mage": {"base_url": "https://api.openai.com/v1"},
        })
        store = MemoryKeystore()
        report = migrate_literal_secrets(
            config, path=tmp_path / "care.toml", keystore=store,
        )
        # mage.api_key empty → skipped; no other secret slots
        # were set either.
        assert report.did_migrate is False
        slot_names = {slot for slot, _ in report.skipped}
        assert "mage.api_key" in slot_names

    def test_multiple_slots_migrate_independently(
        self, tmp_path: Path,
    ):
        config = CareConfig.model_validate({
            "mage": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-a",
            },
            "memory": {
                "base_url": "https://m.example",
                "api_key": "sk-b",
            },
            "platform": {
                "base_url": "https://p.example",
                "api_key": "sk-c",
            },
        })
        store = MemoryKeystore()
        report = migrate_literal_secrets(
            config, path=tmp_path / "care.toml", keystore=store,
        )
        assert {
            (parent, field) for parent, field, _ in report.migrated
        } == {
            ("mage", "api_key"),
            ("memory", "api_key"),
            ("platform", "api_key"),
        }
        # Each landed in the keystore under the canonical
        # `<parent>.<field>` key.
        assert store.fetch("care", "mage.api_key") == "sk-a"
        assert store.fetch("care", "memory.api_key") == "sk-b"
        assert store.fetch("care", "platform.api_key") == "sk-c"

    def test_keystore_detect_failure_blocks_migration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from care.runtime import keystore as ks_module

        def _boom(**_kw):
            raise RuntimeError("no keychain on this box")

        monkeypatch.setattr(
            ks_module, "detect_keystore", _boom,
        )
        config = CareConfig.model_validate({
            "mage": {
                "base_url": "https://x",
                "api_key": "sk-needs-migration",
            },
        })
        report = migrate_literal_secrets(
            config, path=tmp_path / "care.toml",
        )
        assert report.did_migrate is False
        # The slot lands in `skipped` with a detect-failed
        # reason.
        assert any(
            "detect failed" in reason
            for _, reason in report.skipped
        )
        # In-memory value is left as the literal (we didn't
        # rewrite anything).
        assert config.mage.api_key == "sk-needs-migration"

    def test_disk_write_failure_lands_on_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from care import config as config_mod

        config = CareConfig.model_validate({
            "mage": {
                "base_url": "https://x",
                "api_key": "sk-real",
            },
        })
        store = MemoryKeystore()
        original_save = config_mod.CareConfig.save_to_disk

        def _fake_save(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(
            config_mod.CareConfig, "save_to_disk", _fake_save,
        )
        try:
            report = migrate_literal_secrets(
                config, path=tmp_path / "care.toml",
                keystore=store,
            )
        finally:
            monkeypatch.setattr(
                config_mod.CareConfig,
                "save_to_disk",
                original_save,
            )
        # Keystore was written (the failure happened later)
        assert store.fetch("care", "mage.api_key") == "sk-real"
        # In-memory carries the URL.
        assert config.mage.api_key == (
            "keystore://care/mage.api_key"
        )
        # Migration list still records the success; the
        # disk failure surfaces as a synthetic skip.
        assert report.did_migrate is True
        assert any(
            slot == "__disk__" and "write failed" in reason
            for slot, reason in report.skipped
        )


# ---------------------------------------------------------------------------
# CLI: `care migrate-secrets`
# ---------------------------------------------------------------------------


class TestCliMigrateSecrets:
    def test_no_config_file_returns_zero(self, tmp_path: Path):
        from care.cli import main

        missing = tmp_path / "missing.toml"
        rc = main(
            ["migrate-secrets", "--config", str(missing)],
        )
        # main() uses real sys.stdout / stderr; the
        # missing-path branch returns 0.
        assert rc == 0

    def test_dry_run_does_not_touch_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        from care.cli import main

        config_path = tmp_path / "care.toml"
        before = (
            '[mage]\n'
            'api_key = "sk-dry-1"\n'
            'base_url = "https://api.openai.com/v1"\n'
        )
        config_path.write_text(before)
        rc = main(
            [
                "migrate-secrets",
                "--config", str(config_path),
                "--dry-run",
            ],
        )
        assert rc == 0
        # Dry-run leaves the file as-is.
        assert config_path.read_text() == before

    def test_full_migration_rewrites_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        import os

        from care import dotenv as dotenv_mod
        from care.cli import main
        from care.runtime import keystore as ks_module

        store = MemoryKeystore()
        monkeypatch.setattr(
            ks_module, "detect_keystore",
            lambda **_kw: store,
        )

        # Strip CARE_* env vars + neutralise the project-local
        # `.env` load so the loader uses only the test's TOML
        # — the user's real env / repo `.env` otherwise
        # overrides the test's literal.
        for key in list(os.environ.keys()):
            if key.startswith("CARE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            dotenv_mod, "load_env_file", lambda *_a, **_kw: None,
        )

        config_path = tmp_path / "care.toml"
        config_path.write_text(
            '[mage]\n'
            'api_key = "sk-cli-2"\n'
            'base_url = "https://api.openai.com/v1"\n'
        )
        rc = main(
            ["migrate-secrets", "--config", str(config_path)],
        )
        assert rc == 0
        text = config_path.read_text()
        assert "sk-cli-2" not in text
        assert "keystore://care/mage.api_key" in text
        assert store.fetch("care", "mage.api_key") == "sk-cli-2"
