"""OS-native secret storage for CARE config (TODO §1 P0).

Today :class:`care.config.MageConfig.api_key` (and the matching
``MemoryConfig.api_key`` / ``PlatformConfig.api_key`` /
``MageConfig.web_search_api_key``) lives plain-text in
``~/.config/care/config.toml``. That's fine for a quick demo
but unsafe for the ``uvx care`` release — anyone with read
access to the user's home directory walks away with API keys.

This module provides a tiny secret-storage abstraction:

* :class:`Keystore` — the protocol every backend implements.
* :class:`MacOSKeychainKeystore` — shells out to ``security`` on
  Darwin. Stores via ``add-generic-password``, reads via
  ``find-generic-password -w``, deletes via
  ``delete-generic-password``.
* :class:`LinuxSecretToolKeystore` — shells out to
  ``secret-tool`` (libsecret CLI) on Linux when it's on PATH.
* :class:`FileKeystore` — last-resort fallback. Plain JSON at
  ``~/.config/care/secrets.json`` with mode 0600. The caller is
  expected to surface a clear warning toast pointing out that
  the secret is on disk in clear text.
* :class:`MemoryKeystore` — in-memory backend used by tests and
  by callers that explicitly opt into a non-persistent store.

URL format on disk: ``keystore://<service>/<key>``. A config
value that matches that shape is dereferenced through the
detected keystore on read; any other value is returned as-is
(literal). The asymmetry — *literal on read, URL on write* —
keeps the migration cheap: existing literals keep working,
fresh writes upgrade themselves.

Threading: every backend's read/write/delete is synchronous and
re-entrant against the underlying OS call. The shell-out
backends call ``subprocess.run`` directly without locking — the
underlying OS keychain handles concurrency.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("care.keystore")


DEFAULT_SERVICE = "care"
"""Default keychain service name. Scopes CARE entries so they
don't collide with other apps using the same backend. Override
per call when an integration needs a different namespace."""

DEFAULT_FILE_KEYSTORE_PATH = Path.home() / ".config" / "care" / "secrets.json"
"""Where the file-fallback backend writes when the user didn't
pass an explicit path. Mode 0600 enforced on every write."""

KEYSTORE_URL_SCHEME = "keystore://"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KeystoreError(RuntimeError):
    """Raised by any backend when the underlying store rejects an
    operation or the backend itself is unusable in the current
    environment (e.g. ``security`` missing on macOS — should
    never happen but cheaper to surface than to crash later)."""


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def is_keystore_url(value: object) -> bool:
    """Cheap detector — does ``value`` look like a keystore
    reference? Falls back to ``False`` for non-strings so
    callers can throw any config value at it."""
    return isinstance(value, str) and value.startswith(KEYSTORE_URL_SCHEME)


def make_keystore_url(service: str, key: str) -> str:
    """Build the canonical reference URL stored in ``config.toml``.

    Service + key are URL-quoted lightly: only ``/`` is forbidden
    inside ``key`` (it'd ambiguate the URL); everything else
    rides through. Empty service/key raises, since both are
    needed for any backend lookup.
    """
    if not service:
        raise KeystoreError("keystore url requires a non-empty service")
    if not key:
        raise KeystoreError("keystore url requires a non-empty key")
    if "/" in key:
        raise KeystoreError(
            f"keystore key cannot contain '/': {key!r}"
        )
    return f"{KEYSTORE_URL_SCHEME}{service}/{key}"


def parse_keystore_url(url: str) -> tuple[str, str]:
    """Inverse of :func:`make_keystore_url`. Raises
    :class:`KeystoreError` when ``url`` isn't a keystore
    reference or is missing the service / key half.
    """
    if not is_keystore_url(url):
        raise KeystoreError(f"not a keystore url: {url!r}")
    rest = url[len(KEYSTORE_URL_SCHEME):]
    if "/" not in rest:
        raise KeystoreError(
            f"keystore url missing key half: {url!r} "
            f"(expected '{KEYSTORE_URL_SCHEME}<service>/<key>')"
        )
    service, _, key = rest.partition("/")
    if not service or not key:
        raise KeystoreError(
            f"keystore url has empty service or key: {url!r}"
        )
    return service, key


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Keystore(ABC):
    """The minimal CRD surface every secret backend implements.

    Backends are state-free objects — construct one, call
    :meth:`store` / :meth:`fetch` / :meth:`delete`. Concrete
    subclasses MAY raise :class:`KeystoreError` on backend-side
    failure; callers wrap in `try` to fall back to the file
    store or surface a toast.
    """

    name: str = "abstract"
    """Human-readable backend label — surfaces in toasts /
    diagnostic output so the user knows *where* the secret
    lives."""

    persistent: bool = True
    """`True` when the backend keeps secrets across process
    restarts. `False` only for :class:`MemoryKeystore`."""

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Cheap probe — is this backend usable on the host?
        Detector for :func:`detect_keystore`."""

    @abstractmethod
    def store(self, service: str, key: str, value: str) -> None:
        """Persist ``value`` under ``service`` + ``key``.
        Overwrites any prior value. Empty ``value`` is rejected
        — callers can :meth:`delete` instead."""

    @abstractmethod
    def fetch(self, service: str, key: str) -> str | None:
        """Read the previously-stored secret. Returns ``None``
        when the entry doesn't exist (NOT an error — the
        config-loader treats a missing secret as "no key
        configured" and surfaces the matching toast)."""

    @abstractmethod
    def delete(self, service: str, key: str) -> None:
        """Remove the entry. No-op when the key wasn't there —
        idempotent so wizard "reset" flows don't have to check
        first."""


# ---------------------------------------------------------------------------
# In-memory backend (tests + opt-in)
# ---------------------------------------------------------------------------


class MemoryKeystore(Keystore):
    """In-memory backend. Used by tests + by callers that
    explicitly want a non-persistent store (e.g. running CARE in
    a CI sandbox where Keychain isn't available)."""

    name = "memory"
    persistent = False

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    @classmethod
    def available(cls) -> bool:
        return True

    def store(self, service: str, key: str, value: str) -> None:
        if not value:
            raise KeystoreError("refusing to store an empty value")
        self._store[(service, key)] = value

    def fetch(self, service: str, key: str) -> str | None:
        return self._store.get((service, key))

    def delete(self, service: str, key: str) -> None:
        self._store.pop((service, key), None)


# ---------------------------------------------------------------------------
# File-fallback backend
# ---------------------------------------------------------------------------


class FileKeystore(Keystore):
    """Last-resort backend — plain JSON file with mode 0600.

    Used when neither the OS-native backend nor a custom
    backend is available. The file lives at
    ``~/.config/care/secrets.json`` by default; the caller can
    override for tests / multi-profile setups.
    """

    name = "file"
    persistent = True

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_FILE_KEYSTORE_PATH

    @classmethod
    def available(cls) -> bool:
        # Always usable — the worst case is a permission error
        # at write time, which we surface as `KeystoreError`.
        return True

    def store(self, service: str, key: str, value: str) -> None:
        if not value:
            raise KeystoreError("refusing to store an empty value")
        data = self._load()
        data.setdefault(service, {})[key] = value
        self._save(data)

    def fetch(self, service: str, key: str) -> str | None:
        data = self._load()
        return data.get(service, {}).get(key)

    def delete(self, service: str, key: str) -> None:
        data = self._load()
        bucket = data.get(service)
        if not bucket or key not in bucket:
            return
        del bucket[key]
        if not bucket:
            del data[service]
        self._save(data)

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KeystoreError(
                f"FileKeystore failed to read {self.path}: {exc}"
            ) from exc

    def _save(self, data: dict[str, dict[str, str]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # `tmp + replace` keeps the secrets file atomic — a
            # crash mid-write doesn't leave a half-written JSON
            # behind.
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
            os.replace(tmp, self.path)
        except OSError as exc:
            raise KeystoreError(
                f"FileKeystore failed to write {self.path}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# macOS Keychain backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Subprocess:
    """Tiny seam so tests can inject a synthetic subprocess
    invoker without monkey-patching the real one. The default
    instance just forwards to :func:`subprocess.run` with the
    arguments the OS-native backends rely on."""

    def run(
        self,
        args: list[str],
        *,
        input: str | None = None,
        capture_output: bool = True,
        check: bool = False,
        timeout: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            input=input,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
        )


_DEFAULT_SUBPROCESS = _Subprocess()


class MacOSKeychainKeystore(Keystore):
    """macOS Keychain backend.

    Uses the ``security`` CLI shipped with the OS. Each entry
    is a generic password keyed by ``service`` + ``key``
    (account, in Keychain parlance).
    """

    name = "macos-keychain"
    persistent = True

    def __init__(self, *, runner: _Subprocess | None = None) -> None:
        self._runner = runner or _DEFAULT_SUBPROCESS

    @classmethod
    def available(cls) -> bool:
        return sys.platform == "darwin" and shutil.which("security") is not None

    def store(self, service: str, key: str, value: str) -> None:
        if not value:
            raise KeystoreError("refusing to store an empty value")
        # `-U` upserts: replaces an existing entry rather than
        # erroring with "duplicate". `-w` passes the password on
        # the command line — fine for CARE's single-user host;
        # for multi-user hardening we'd switch to `-`+stdin.
        proc = self._runner.run(
            [
                "security", "add-generic-password",
                "-U",
                "-a", key,
                "-s", service,
                "-w", value,
            ],
        )
        if proc.returncode != 0:
            raise KeystoreError(
                f"macOS Keychain store failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()}"
            )

    def fetch(self, service: str, key: str) -> str | None:
        proc = self._runner.run(
            [
                "security", "find-generic-password",
                "-a", key,
                "-s", service,
                "-w",
            ],
        )
        if proc.returncode == 44:  # SecKeychainItemNotFound — no entry yet.
            return None
        if proc.returncode != 0:
            raise KeystoreError(
                f"macOS Keychain fetch failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()}"
            )
        # `security` prints the password followed by a newline.
        return (proc.stdout or "").rstrip("\n")

    def delete(self, service: str, key: str) -> None:
        proc = self._runner.run(
            [
                "security", "delete-generic-password",
                "-a", key,
                "-s", service,
            ],
        )
        # 44 = SecKeychainItemNotFound — idempotent delete.
        if proc.returncode not in (0, 44):
            raise KeystoreError(
                f"macOS Keychain delete failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()}"
            )


# ---------------------------------------------------------------------------
# Linux libsecret backend
# ---------------------------------------------------------------------------


class LinuxSecretToolKeystore(Keystore):
    """Linux libsecret backend via the ``secret-tool`` CLI.

    Entries are keyed by two attributes: ``service`` and ``key``,
    matching the macOS namespace shape so the same
    `(service, key)` tuple looks up the same secret regardless
    of host OS.
    """

    name = "linux-secret-tool"
    persistent = True

    def __init__(self, *, runner: _Subprocess | None = None) -> None:
        self._runner = runner or _DEFAULT_SUBPROCESS

    @classmethod
    def available(cls) -> bool:
        return sys.platform.startswith("linux") and shutil.which("secret-tool") is not None

    def store(self, service: str, key: str, value: str) -> None:
        if not value:
            raise KeystoreError("refusing to store an empty value")
        # `secret-tool store` reads the value from stdin so it
        # doesn't appear in `ps`.
        proc = self._runner.run(
            [
                "secret-tool", "store",
                "--label", f"CARE secret: {service}/{key}",
                "service", service,
                "key", key,
            ],
            input=value,
        )
        if proc.returncode != 0:
            raise KeystoreError(
                f"secret-tool store failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()}"
            )

    def fetch(self, service: str, key: str) -> str | None:
        proc = self._runner.run(
            [
                "secret-tool", "lookup",
                "service", service,
                "key", key,
            ],
        )
        # `secret-tool lookup` returns 1 + empty stdout on miss.
        if proc.returncode == 1 and not (proc.stdout or "").strip():
            return None
        if proc.returncode != 0:
            raise KeystoreError(
                f"secret-tool lookup failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()}"
            )
        return (proc.stdout or "").rstrip("\n") or None

    def delete(self, service: str, key: str) -> None:
        proc = self._runner.run(
            [
                "secret-tool", "clear",
                "service", service,
                "key", key,
            ],
        )
        # `secret-tool clear` returns 0 even on miss — already
        # idempotent.
        if proc.returncode != 0:
            raise KeystoreError(
                f"secret-tool clear failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()}"
            )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_keystore(
    *,
    file_path: Path | None = None,
    prefer: list[type[Keystore]] | None = None,
) -> Keystore:
    """Pick the best available backend for the host.

    Order:

    * Caller-provided ``prefer`` list (each class is probed via
      :meth:`Keystore.available`).
    * macOS Keychain — when running on Darwin with ``security``
      on PATH.
    * Linux ``secret-tool`` — when running on Linux with
      ``secret-tool`` on PATH.
    * File fallback — always usable.

    The file fallback path is configurable for tests / multi-
    profile setups via ``file_path``.

    Surfaces a one-line log entry at WARNING when falling back
    to the file backend so the missing-backend signal lands in
    the app log even if the caller forgets to toast.
    """
    candidates: list[type[Keystore]] = list(prefer or [])
    candidates.extend([MacOSKeychainKeystore, LinuxSecretToolKeystore])
    for cls in candidates:
        try:
            if cls.available():
                return cls()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "keystore backend %s probe failed: %s",
                getattr(cls, "name", cls.__name__), exc,
            )
    _log.warning(
        "no OS-native keystore available; falling back to "
        "FileKeystore at %s — secrets will be stored in plain "
        "JSON",
        file_path or DEFAULT_FILE_KEYSTORE_PATH,
    )
    return FileKeystore(path=file_path)


# ---------------------------------------------------------------------------
# Public read / write helpers
# ---------------------------------------------------------------------------


def resolve_secret(
    raw: str | None,
    *,
    keystore: Keystore | None = None,
) -> str | None:
    """Resolve a config value to its concrete secret.

    Three cases:

    * ``raw is None`` — return ``None`` (matches "no secret
      configured").
    * ``raw`` is a keystore URL — dereference through
      ``keystore`` (auto-detected when unset). Returns the
      stored secret or ``None`` when the entry doesn't exist.
    * ``raw`` is any other string — return it verbatim
      (literal value already in ``config.toml``).

    Designed to be the single point of dereference for every
    ``CareConfig.*_api_key`` field. Raises
    :class:`KeystoreError` only when the URL is malformed; a
    missing-entry case is intentionally not an error so a
    user who deletes the keychain entry sees an empty
    config-level api_key + the friendly "not configured"
    toast instead of a crash.
    """
    if raw is None:
        return None
    if not is_keystore_url(raw):
        return raw
    ks = keystore or detect_keystore()
    service, key = parse_keystore_url(raw)
    return ks.fetch(service, key)


def store_secret(
    value: str,
    *,
    key: str,
    service: str = DEFAULT_SERVICE,
    keystore: Keystore | None = None,
) -> str:
    """Persist ``value`` under ``service`` + ``key`` and return
    the canonical keystore URL the caller writes to
    ``config.toml``. Auto-detects the backend when ``keystore``
    is unset.

    Reverse of :func:`resolve_secret`: caller passes the
    literal value the user typed in the wizard, we store it +
    hand back a URL the loader will dereference on next boot.
    """
    if not value:
        raise KeystoreError("refusing to store an empty value")
    ks = keystore or detect_keystore()
    ks.store(service, key, value)
    return make_keystore_url(service, key)


__all__ = [
    "DEFAULT_FILE_KEYSTORE_PATH",
    "DEFAULT_SERVICE",
    "FileKeystore",
    "KEYSTORE_URL_SCHEME",
    "Keystore",
    "KeystoreError",
    "LinuxSecretToolKeystore",
    "MacOSKeychainKeystore",
    "MemoryKeystore",
    "detect_keystore",
    "is_keystore_url",
    "make_keystore_url",
    "parse_keystore_url",
    "resolve_secret",
    "store_secret",
]
