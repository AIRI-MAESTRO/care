"""Tests for `care.runtime.keystore` (TODO §1 P0).

Covers every public surface:

* URL helpers — round-trip + rejection of malformed inputs.
* In-memory + file backend — full CRD semantics.
* macOS Keychain backend — exercises the `security` invocation
  shape via a stub subprocess runner (no real Keychain access).
* Linux `secret-tool` backend — same stub-runner pattern.
* `detect_keystore` — preference order, fallback to file.
* `resolve_secret` / `store_secret` — the public read/write
  helpers `CareConfig` will use.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass

import pytest

from care.runtime.keystore import (
    DEFAULT_FILE_KEYSTORE_PATH,
    DEFAULT_SERVICE,
    FileKeystore,
    KEYSTORE_URL_SCHEME,
    Keystore,
    KeystoreError,
    LinuxSecretToolKeystore,
    MacOSKeychainKeystore,
    MemoryKeystore,
    detect_keystore,
    is_keystore_url,
    make_keystore_url,
    parse_keystore_url,
    resolve_secret,
    store_secret,
)


# ---------------------------------------------------------------------------
# Fake subprocess runner
# ---------------------------------------------------------------------------


@dataclass
class _FakeProc:
    """Minimal stand-in for `subprocess.CompletedProcess[str]`."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class _Recorder:
    """Pluggable subprocess runner — records every invocation +
    returns canned responses keyed by the first two args
    (binary + verb). Lets the macOS / Linux backend tests
    exercise the real argv shape without touching the host's
    keychain."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self._responses: list[_FakeProc] = []

    def push(self, *procs: _FakeProc) -> None:
        self._responses.extend(procs)

    def run(
        self,
        args: list[str],
        *,
        input: str | None = None,
        capture_output: bool = True,
        check: bool = False,
        timeout: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, input))
        if not self._responses:
            raise AssertionError(
                f"_Recorder ran out of canned responses; "
                f"called with {args!r}"
            )
        proc = self._responses.pop(0)
        # Build a real CompletedProcess so callers that introspect
        # type(proc) still see the right class.
        return subprocess.CompletedProcess(
            args=args,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestURL:
    def test_is_keystore_url(self):
        assert is_keystore_url("keystore://care/mage.api_key")
        assert not is_keystore_url("sk-abc")
        assert not is_keystore_url(None)
        assert not is_keystore_url(42)
        assert not is_keystore_url("")

    def test_round_trip(self):
        url = make_keystore_url("care", "mage.api_key")
        assert url == "keystore://care/mage.api_key"
        assert parse_keystore_url(url) == ("care", "mage.api_key")

    def test_make_rejects_empty_or_slashed(self):
        with pytest.raises(KeystoreError):
            make_keystore_url("", "key")
        with pytest.raises(KeystoreError):
            make_keystore_url("care", "")
        with pytest.raises(KeystoreError):
            make_keystore_url("care", "with/slash")

    def test_parse_rejects_bad_shape(self):
        with pytest.raises(KeystoreError):
            parse_keystore_url("sk-abc")
        with pytest.raises(KeystoreError):
            parse_keystore_url("keystore://care")   # no key half
        with pytest.raises(KeystoreError):
            parse_keystore_url("keystore:///key")   # empty service
        with pytest.raises(KeystoreError):
            parse_keystore_url("keystore://care/")  # empty key

    def test_scheme_constant_matches_make(self):
        assert make_keystore_url("a", "b").startswith(KEYSTORE_URL_SCHEME)


# ---------------------------------------------------------------------------
# MemoryKeystore
# ---------------------------------------------------------------------------


class TestMemoryKeystore:
    def test_store_fetch_delete_roundtrip(self):
        ks = MemoryKeystore()
        assert ks.fetch("care", "k") is None
        ks.store("care", "k", "value-1")
        assert ks.fetch("care", "k") == "value-1"
        ks.store("care", "k", "value-2")
        assert ks.fetch("care", "k") == "value-2"
        ks.delete("care", "k")
        assert ks.fetch("care", "k") is None

    def test_delete_missing_is_idempotent(self):
        ks = MemoryKeystore()
        ks.delete("care", "missing")  # must not raise

    def test_empty_value_rejected(self):
        ks = MemoryKeystore()
        with pytest.raises(KeystoreError):
            ks.store("care", "k", "")

    def test_persistent_flag_is_false(self):
        assert MemoryKeystore.persistent is False
        assert MemoryKeystore.available() is True


# ---------------------------------------------------------------------------
# FileKeystore
# ---------------------------------------------------------------------------


class TestFileKeystore:
    def test_round_trip_creates_file_with_0600(self, tmp_path):
        path = tmp_path / "secrets.json"
        ks = FileKeystore(path=path)
        assert ks.fetch("care", "k") is None
        ks.store("care", "k", "v")
        assert path.exists()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"
        assert ks.fetch("care", "k") == "v"

    def test_per_service_buckets(self, tmp_path):
        path = tmp_path / "s.json"
        ks = FileKeystore(path=path)
        ks.store("svcA", "k", "vA")
        ks.store("svcB", "k", "vB")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["svcA"]["k"] == "vA"
        assert data["svcB"]["k"] == "vB"

    def test_delete_empties_service_bucket(self, tmp_path):
        path = tmp_path / "s.json"
        ks = FileKeystore(path=path)
        ks.store("svc", "k", "v")
        ks.delete("svc", "k")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "svc" not in data

    def test_corrupt_file_surfaces_keystore_error(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("not json", encoding="utf-8")
        ks = FileKeystore(path=path)
        with pytest.raises(KeystoreError):
            ks.fetch("care", "k")

    def test_atomic_write_no_leftover_tmp(self, tmp_path):
        path = tmp_path / "s.json"
        ks = FileKeystore(path=path)
        ks.store("svc", "k", "v")
        # The `.tmp` shadow used during write should not persist.
        assert not list(path.parent.glob("*.tmp"))

    def test_default_path_under_home(self):
        # Smoke — the canonical default lives under ~/.config/care.
        assert ".config/care" in str(DEFAULT_FILE_KEYSTORE_PATH)


# ---------------------------------------------------------------------------
# MacOSKeychainKeystore
# ---------------------------------------------------------------------------


class TestMacOSKeychain:
    def test_store_invokes_security_with_upsert_flags(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=0))
        ks = MacOSKeychainKeystore(runner=rec)
        ks.store("care", "k", "v")
        args, stdin = rec.calls[0]
        assert args[0] == "security"
        assert args[1] == "add-generic-password"
        assert "-U" in args            # upsert
        assert args[args.index("-a") + 1] == "k"
        assert args[args.index("-s") + 1] == "care"
        assert args[args.index("-w") + 1] == "v"
        assert stdin is None

    def test_fetch_returns_stripped_stdout(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=0, stdout="secret-1\n"))
        ks = MacOSKeychainKeystore(runner=rec)
        assert ks.fetch("care", "k") == "secret-1"

    def test_fetch_missing_returns_none(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=44, stderr="not found"))
        ks = MacOSKeychainKeystore(runner=rec)
        assert ks.fetch("care", "k") is None

    def test_fetch_other_failure_raises(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=1, stderr="boom"))
        ks = MacOSKeychainKeystore(runner=rec)
        with pytest.raises(KeystoreError) as ei:
            ks.fetch("care", "k")
        assert "boom" in str(ei.value)

    def test_delete_treats_44_as_success(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=44))
        ks = MacOSKeychainKeystore(runner=rec)
        ks.delete("care", "k")  # no raise

    def test_empty_value_rejected(self):
        rec = _Recorder()
        ks = MacOSKeychainKeystore(runner=rec)
        with pytest.raises(KeystoreError):
            ks.store("care", "k", "")
        assert rec.calls == []


# ---------------------------------------------------------------------------
# LinuxSecretToolKeystore
# ---------------------------------------------------------------------------


class TestLinuxSecretTool:
    def test_store_sends_value_via_stdin(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=0))
        ks = LinuxSecretToolKeystore(runner=rec)
        ks.store("care", "k", "v")
        args, stdin = rec.calls[0]
        assert args[:2] == ["secret-tool", "store"]
        assert "--label" in args
        assert "service" in args and args[args.index("service") + 1] == "care"
        assert "key" in args and args[args.index("key") + 1] == "k"
        assert stdin == "v"  # never via argv

    def test_lookup_miss_returns_none(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=1, stdout=""))
        ks = LinuxSecretToolKeystore(runner=rec)
        assert ks.fetch("care", "k") is None

    def test_lookup_hit_strips_trailing_newline(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=0, stdout="found-secret\n"))
        ks = LinuxSecretToolKeystore(runner=rec)
        assert ks.fetch("care", "k") == "found-secret"

    def test_clear_invokes_secret_tool(self):
        rec = _Recorder()
        rec.push(_FakeProc(returncode=0))
        ks = LinuxSecretToolKeystore(runner=rec)
        ks.delete("care", "k")
        args, _stdin = rec.calls[0]
        assert args[:2] == ["secret-tool", "clear"]


# ---------------------------------------------------------------------------
# detect_keystore
# ---------------------------------------------------------------------------


class _NeverAvailable(Keystore):
    name = "never"
    persistent = True

    @classmethod
    def available(cls) -> bool:
        return False

    def store(self, service, key, value):
        raise AssertionError("never reached")

    def fetch(self, service, key):
        raise AssertionError("never reached")

    def delete(self, service, key):
        raise AssertionError("never reached")


class _AlwaysAvailable(Keystore):
    name = "always"
    persistent = True

    @classmethod
    def available(cls) -> bool:
        return True

    def store(self, service, key, value):
        pass

    def fetch(self, service, key):
        return None

    def delete(self, service, key):
        pass


class TestDetectKeystore:
    def test_prefers_caller_supplied_backend(self):
        ks = detect_keystore(prefer=[_AlwaysAvailable])
        assert isinstance(ks, _AlwaysAvailable)

    def test_skips_unavailable_and_falls_back_to_file(self, tmp_path, monkeypatch):
        # Force the OS-native backends "unavailable" so the test
        # is stable regardless of which host runs it (macOS dev
        # boxes would otherwise pick MacOSKeychainKeystore here).
        monkeypatch.setattr(
            MacOSKeychainKeystore, "available", classmethod(lambda cls: False),
        )
        monkeypatch.setattr(
            LinuxSecretToolKeystore, "available", classmethod(lambda cls: False),
        )
        path = tmp_path / "s.json"
        ks = detect_keystore(prefer=[_NeverAvailable], file_path=path)
        assert isinstance(ks, FileKeystore)
        assert ks.path == path

    def test_default_path_when_no_override(self, monkeypatch):
        # Force OS-native backends "unavailable" by patching the
        # classmethod, then assert the fallback FileKeystore uses
        # the canonical default path.
        monkeypatch.setattr(MacOSKeychainKeystore, "available", classmethod(lambda cls: False))
        monkeypatch.setattr(LinuxSecretToolKeystore, "available", classmethod(lambda cls: False))
        ks = detect_keystore()
        assert isinstance(ks, FileKeystore)
        assert ks.path == DEFAULT_FILE_KEYSTORE_PATH


# ---------------------------------------------------------------------------
# resolve_secret / store_secret
# ---------------------------------------------------------------------------


class TestPublicHelpers:
    def test_resolve_literal_passthrough(self):
        assert resolve_secret("sk-1234") == "sk-1234"

    def test_resolve_none_returns_none(self):
        assert resolve_secret(None) is None

    def test_resolve_dereferences_url(self):
        ks = MemoryKeystore()
        ks.store(DEFAULT_SERVICE, "mage.api_key", "sk-from-store")
        url = make_keystore_url(DEFAULT_SERVICE, "mage.api_key")
        assert resolve_secret(url, keystore=ks) == "sk-from-store"

    def test_resolve_missing_entry_returns_none(self):
        ks = MemoryKeystore()
        url = make_keystore_url(DEFAULT_SERVICE, "absent")
        assert resolve_secret(url, keystore=ks) is None

    def test_resolve_malformed_url_raises(self):
        with pytest.raises(KeystoreError):
            resolve_secret("keystore://no-key-half", keystore=MemoryKeystore())

    def test_store_secret_returns_url_and_persists(self):
        ks = MemoryKeystore()
        url = store_secret("sk-new", key="mage.api_key", keystore=ks)
        assert url == make_keystore_url(DEFAULT_SERVICE, "mage.api_key")
        assert ks.fetch(DEFAULT_SERVICE, "mage.api_key") == "sk-new"

    def test_store_secret_rejects_empty(self):
        ks = MemoryKeystore()
        with pytest.raises(KeystoreError):
            store_secret("", key="k", keystore=ks)


# ---------------------------------------------------------------------------
# Runtime export
# ---------------------------------------------------------------------------


class TestRuntimeExport:
    def test_top_level_exports_resolve(self):
        from care import runtime as r
        assert r.Keystore is Keystore
        assert r.MemoryKeystore is MemoryKeystore
        assert r.FileKeystore is FileKeystore
        assert r.MacOSKeychainKeystore is MacOSKeychainKeystore
        assert r.LinuxSecretToolKeystore is LinuxSecretToolKeystore
        assert r.detect_keystore is detect_keystore
        assert r.resolve_secret is resolve_secret
        assert r.store_secret is store_secret
        assert r.DEFAULT_KEYSTORE_SERVICE == DEFAULT_SERVICE
