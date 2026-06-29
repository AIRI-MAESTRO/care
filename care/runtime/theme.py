"""Theming + dark/light toggle data layer (TODO §1 P2).

CARE's Textual app reads TCSS variables for its colour palette;
this module ships the data layer that owns:

* The canonical built-in themes (``dark``, ``light``, plus an
  ``auto`` sentinel that resolves to the system's
  ``prefers-color-scheme``).
* A user-preference store at
  ``~/.config/care/theme.json`` so the user's choice survives
  CARE restarts.
* A small registry the future plugins layer can extend with
  custom themes without touching this module.
* A projection helper that turns a :class:`Theme` into the
  ``{variable_name: value}`` dict the Textual app feeds into
  ``self.dark`` / ``self.styles`` reactive bindings.

The Textual toggle UI is gated on TODO §1 P0 multi-screen
workflow, but the resolver + persistence + registry are
bounded data-layer concerns that ship now.

What this module provides:

* :class:`ThemeKind` literal — ``light`` / ``dark`` / ``auto``.
* :class:`Theme` — frozen theme descriptor.
* :class:`ThemePreference` — frozen on-disk preference shape
  (theme_name + schema_version).
* :class:`ThemePreferenceStore` — atomic JSON store mirroring
  the :class:`care.runtime.RunStateStore` contract.
* Module-level :func:`save_theme_preference` / :func:`load_theme_preference`
  convenience wrappers.
* :data:`DEFAULT_THEMES` — built-in registry.
* :func:`register_theme` / :func:`unregister_theme` /
  :func:`list_themes` / :func:`get_theme` — module-level
  registry hooks.
* :func:`resolve_active_theme` — projects a preferred theme
  name + optional system appearance into the concrete
  :class:`Theme` the app should render.
* :func:`theme_to_tcss_vars` — pure projection from theme →
  TCSS variable dict.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


ThemeKind = Literal["light", "dark", "auto"]
"""The three theme kinds CARE exposes. ``auto`` is a sentinel —
the resolver picks ``light`` or ``dark`` based on the host
appearance signal."""


SystemAppearance = Literal["light", "dark"]
"""What the host signals for `prefers-color-scheme` resolution."""


DEFAULT_THEME_PATH = Path("~/.config/care/theme.json").expanduser()
"""XDG-style location for the persisted theme preference. Sits
alongside the existing `config.toml` so the user's preference
file inventory stays small."""


_THEME_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ThemeError(RuntimeError):
    """Raised for theming-related failures: unknown theme name,
    invalid theme registration, mismatched schema_version on
    load. Per-call IO failures on the store degrade silently
    via ``None`` — matches the run_state contract."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """A named CARE theme.

    Frozen so the registry / config flow through Textual
    messages without defensive copies. ``variables`` is a flat
    string→string dict keyed by TCSS variable names (no leading
    ``$``); the projection helper renders the ``$`` prefix at
    consumption time.

    ``kind`` controls the resolver's behaviour:

    * ``light`` / ``dark`` — direct selection; the resolver
      returns this theme regardless of system appearance.
    * ``auto`` — sentinel that delegates to system appearance.
      Custom themes can use ``auto`` to ship their own light/
      dark variants paired by name (see the ``auto`` default).
    """

    name: str
    kind: ThemeKind
    variables: dict[str, str] = field(default_factory=dict)
    description: str = ""
    light_pair: Optional[str] = None
    dark_pair: Optional[str] = None

    @property
    def is_auto(self) -> bool:
        return self.kind == "auto"


@dataclass(frozen=True)
class ThemePreference:
    """User's persisted preference.

    Frozen on-disk shape; the persistence layer round-trips this
    via :class:`ThemePreferenceStore`.
    """

    theme_name: str = "auto"
    schema_version: int = _THEME_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------


_LIGHT_VARS: dict[str, str] = {
    "background": "#ffffff",
    "surface": "#f5f5f5",
    "panel": "#ebebeb",
    "primary": "#0078d4",
    "secondary": "#00bfa5",
    # Brand accent (was `#ff5722` deep orange). Used by user
    # messages, the `>` chat prompt, and any other surface
    # that calls out an active / hot element via `$accent`.
    "accent": "#2ebfae",
    "warning": "#ff8f00",
    "error": "#d32f2f",
    "success": "#2e7d32",
    "foreground": "#1f1f1f",
    "foreground-muted": "#5a5a5a",
    "border": "#d0d0d0",
}


_DARK_VARS: dict[str, str] = {
    "background": "#0f1115",
    "surface": "#181c22",
    "panel": "#1f242c",
    "primary": "#5aa6ff",
    "secondary": "#33d9b2",
    # Brand accent (was `#ff8a65` peach). Same hex as the
    # light theme so the brand colour stays stable when the
    # user flips appearance — the dark surfaces give it a
    # naturally cooler perceived tone.
    "accent": "#2ebfae",
    "warning": "#ffb74d",
    "error": "#ef5350",
    "success": "#66bb6a",
    "foreground": "#e6e6e6",
    "foreground-muted": "#a0a0a0",
    "border": "#2a313a",
}


_LIGHT_THEME = Theme(
    name="light",
    kind="light",
    description="CARE's canonical light palette",
    variables=_LIGHT_VARS,
)


_DARK_THEME = Theme(
    name="dark",
    kind="dark",
    description="CARE's canonical dark palette",
    variables=_DARK_VARS,
)


_AUTO_THEME = Theme(
    name="auto",
    kind="auto",
    description=(
        "Follows the host's `prefers-color-scheme`; falls back "
        "to dark when the signal is unavailable."
    ),
    variables={},
    light_pair="light",
    dark_pair="dark",
)


DEFAULT_THEMES: tuple[Theme, ...] = (_AUTO_THEME, _LIGHT_THEME, _DARK_THEME)
"""Canonical built-in themes. The ``auto`` entry comes first so
listing UIs surface "Follow system" at the top of the picker."""


# Registry — mutable but guarded by a lock so the future
# plugins module can register themes from a background thread.
_REGISTRY: dict[str, Theme] = {t.name: t for t in DEFAULT_THEMES}
_REGISTRY_LOCK = threading.Lock()


def register_theme(theme: Theme) -> None:
    """Register a custom theme by name.

    Raises:
        ThemeError: ``theme.name`` is empty or already
            registered (re-registration is rejected so plugin
            authors see collisions loudly; call
            :func:`unregister_theme` first if you actually want
            to replace one).
    """
    if not theme.name.strip():
        raise ThemeError("theme name cannot be empty")
    with _REGISTRY_LOCK:
        if theme.name in _REGISTRY and theme.name not in {
            t.name for t in DEFAULT_THEMES
        }:
            raise ThemeError(
                f"theme {theme.name!r} is already registered; "
                f"unregister it first to replace"
            )
        if theme.name in {t.name for t in DEFAULT_THEMES}:
            raise ThemeError(
                f"refusing to override built-in theme {theme.name!r}"
            )
        _REGISTRY[theme.name] = theme


def unregister_theme(name: str) -> bool:
    """Drop a custom theme from the registry.

    Refuses to unregister built-ins (no recovery path). Returns
    ``True`` when a theme was actually removed.
    """
    with _REGISTRY_LOCK:
        if name in {t.name for t in DEFAULT_THEMES}:
            raise ThemeError(
                f"refusing to unregister built-in theme {name!r}"
            )
        return _REGISTRY.pop(name, None) is not None


def list_themes() -> tuple[Theme, ...]:
    """Return every registered theme, sorted with the built-in
    ``auto`` / ``light`` / ``dark`` triple first (in that order)
    and any custom themes appended alphabetically."""
    with _REGISTRY_LOCK:
        snapshot = dict(_REGISTRY)
    builtins = [snapshot[t.name] for t in DEFAULT_THEMES if t.name in snapshot]
    builtin_names = {t.name for t in DEFAULT_THEMES}
    custom = sorted(
        (t for t in snapshot.values() if t.name not in builtin_names),
        key=lambda t: t.name.casefold(),
    )
    return tuple(builtins + custom)


def get_theme(name: str) -> Optional[Theme]:
    """Look up a theme by name. ``None`` when unknown."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(name)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_active_theme(
    preferred_name: str,
    *,
    system_appearance: Optional[SystemAppearance] = None,
    fallback: SystemAppearance = "dark",
) -> Theme:
    """Project a preferred theme name into the concrete
    :class:`Theme` the app should render.

    Resolution rules:

    1. If ``preferred_name`` is unknown, fall back to the
       built-in ``auto`` theme.
    2. If the resolved theme is an explicit ``light`` or
       ``dark`` kind, return it directly.
    3. If the resolved theme is ``auto``, pick the partner
       theme named by ``light_pair`` / ``dark_pair`` based on
       ``system_appearance``. ``None`` system appearance →
       ``fallback`` (default ``"dark"``).
    4. If the partner theme name doesn't resolve to a real
       theme, fall back to the built-in ``light`` / ``dark``.

    Args:
        preferred_name: Theme name the user picked.
        system_appearance: The host's ``prefers-color-scheme``
            signal — ``"light"`` or ``"dark"``. ``None`` when
            the platform doesn't expose one (Textual currently
            doesn't); the resolver uses ``fallback`` then.
        fallback: System-appearance fallback. Defaults to
            ``"dark"`` because CARE's primary use case is a
            terminal where light-on-dark is the default.

    Returns:
        Concrete :class:`Theme` to render. Never ``None``.
    """
    theme = get_theme(preferred_name) or get_theme("auto") or _DARK_THEME
    if not theme.is_auto:
        return theme

    appearance = system_appearance or fallback
    partner_name = (
        theme.light_pair if appearance == "light" else theme.dark_pair
    )
    if partner_name:
        partner = get_theme(partner_name)
        if partner is not None:
            return partner
    return _LIGHT_THEME if appearance == "light" else _DARK_THEME


# ---------------------------------------------------------------------------
# TCSS projection
# ---------------------------------------------------------------------------


def theme_to_tcss_vars(theme: Theme) -> dict[str, str]:
    """Project a :class:`Theme` into the ``{$key: value}`` dict
    Textual's stylesheet variable layer consumes.

    Auto themes have no inline variables; the caller should
    have already resolved them via :func:`resolve_active_theme`.
    """
    if theme.is_auto:
        # Defensive: resolved themes should be light/dark, not
        # auto. Surface an empty dict so the caller can detect
        # the misuse and fall back.
        return {}
    return {f"${k}": v for k, v in theme.variables.items()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class ThemePreferenceStore:
    """Atomic JSON store for :class:`ThemePreference`.

    Mirrors the :class:`care.runtime.RunStateStore` contract:

    * Default path :data:`DEFAULT_THEME_PATH`.
    * Atomic writes (tempfile + ``os.replace``).
    * Tolerant ``load()`` — every failure mode returns ``None``
      (file missing, malformed JSON, schema mismatch).
    * Thread-safe via an internal lock.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            self._path = DEFAULT_THEME_PATH
        else:
            self._path = Path(str(path)).expanduser()
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def save(self, preference: ThemePreference) -> Path:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "schema_version": _THEME_SCHEMA_VERSION,
                    "theme_name": preference.theme_name,
                },
                sort_keys=True,
            )
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=".theme-",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
                    fp.write(payload)
                os.replace(tmp_name, self._path)
            except OSError:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
            return self._path

    def load(self) -> Optional[ThemePreference]:
        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
            except OSError:
                return None
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(data, dict):
                return None
            if data.get("schema_version") != _THEME_SCHEMA_VERSION:
                return None
            theme_name = data.get("theme_name")
            if not isinstance(theme_name, str) or not theme_name.strip():
                return None
            return ThemePreference(
                theme_name=theme_name,
                schema_version=_THEME_SCHEMA_VERSION,
            )

    def clear(self) -> bool:
        with self._lock:
            try:
                self._path.unlink()
                return True
            except FileNotFoundError:
                return False


def save_theme_preference(
    preference: ThemePreference,
    *,
    path: Path | str | None = None,
) -> Path:
    """Persist ``preference`` to ``path`` (default
    :data:`DEFAULT_THEME_PATH`). Returns the resolved path."""
    return ThemePreferenceStore(path).save(preference)


def load_theme_preference(
    path: Path | str | None = None,
) -> Optional[ThemePreference]:
    """Load the persisted preference. Returns ``None`` on every
    failure mode — the calling app falls back to the default
    ``auto`` preference."""
    return ThemePreferenceStore(path).load()


__all__ = [
    "DEFAULT_THEMES",
    "DEFAULT_THEME_PATH",
    "SystemAppearance",
    "Theme",
    "ThemeError",
    "ThemeKind",
    "ThemePreference",
    "ThemePreferenceStore",
    "get_theme",
    "list_themes",
    "load_theme_preference",
    "register_theme",
    "resolve_active_theme",
    "save_theme_preference",
    "theme_to_tcss_vars",
    "unregister_theme",
]
