"""Key-based UI localization for the TUI (Russian by default).

CARE's interface is bilingual. Strings live in per-language JSON catalogs
under ``care/runtime/locales/`` and are looked up by a dotted key — the same
shape as JS i18n libraries (i18next & friends)::

    from care.runtime.i18n import t
    self._post_line("system", t("chat.welcome"))
    yield Button(t("settings.action.save"), id="settings-btn-save")

The catalog is nested JSON; keys address it with dots
(``"settings.field.baseUrl"`` → ``settings → field → baseUrl``). Values may
carry ``str.format`` placeholders that are filled from keyword args::

    t("chat.mode.current", label="Chat", mode="ad_hoc")
    # -> "Current mode: Chat (ad_hoc)."

Lookup order: active language → English fallback → the key itself (so a
missing translation surfaces visibly instead of crashing).

The active language is a process-global set once at app boot from
``config.defaults.ui_language`` (see :class:`care.app.CareApp`) and re-read
on every :func:`t` call — so a language change takes effect on the next
render without re-importing.

This is the TUI's language only. The *agent's* answer language is a separate
setting (``config.defaults.language``, forwarded to CARL); see
``care.config.DefaultsConfig``.

To add a string: put the English text in ``locales/en.json`` and the Russian
in ``locales/ru.json`` under the same key, then call ``t("that.key")``. Write
Russian as natural, idiomatic Russian — never a word-for-word calque.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

UILanguage = Literal["ru", "en"]

#: Fallback language used when a key is missing in the active catalog and as
#: the source of truth for the catalog's key set.
_FALLBACK_LANGUAGE: UILanguage = "en"

# Process-global active UI language. Defaults to Russian; overridden at app
# boot via :func:`set_ui_language`. Module-global (not threaded through every
# screen) because the TUI is single-app, single-language at a time.
_ui_language: UILanguage = "ru"


def set_ui_language(language: str | None) -> None:
    """Set the active UI language. Anything other than ``"en"`` (incl.
    ``None`` / unknown) falls back to Russian, the default."""
    global _ui_language
    _ui_language = "en" if str(language or "").strip().lower() == "en" else "ru"


def get_ui_language() -> UILanguage:
    """Return the active UI language (``"ru"`` or ``"en"``)."""
    return _ui_language


def _flatten(node: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested catalog into ``{"a.b.c": "text"}`` pairs."""
    flat: dict[str, str] = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


@lru_cache(maxsize=None)
def _catalog(language: UILanguage) -> dict[str, str]:
    """Load + flatten a language catalog (cached for the process lifetime)."""
    raw = (
        files("care.runtime.locales")
        .joinpath(f"{language}.json")
        .read_text(encoding="utf-8")
    )
    return _flatten(json.loads(raw))


def t(key: str, /, **params: Any) -> str:
    """Translate *key* into the active UI language.

    Falls back to the English catalog when the active language lacks the key,
    then to the raw key so a typo is visible rather than silent. ``params``
    fill ``str.format`` placeholders (``{name}``) in the resolved string; a
    placeholder with no matching arg leaves the template untouched.
    """
    text = _catalog(_ui_language).get(key)
    if text is None and _ui_language != _FALLBACK_LANGUAGE:
        text = _catalog(_FALLBACK_LANGUAGE).get(key)
    if text is None:
        text = key
    if params:
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            return text
    return text
