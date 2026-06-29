"""Shared test fixtures (§8 P3 test-isolation hardening).

Lightweight scaffolding — most heavy-lifting fixtures live
inside the per-suite files (e.g.
`tests/test_screen_chat.py::_isolate_chat_state_files`).
This file only carries the cross-suite isolation hooks where
a shared-by-default behaviour avoids subtle leaks across
test files.
"""

from __future__ import annotations

import pytest

# Test-safety for the UI animations (TODO §Animations). CARE animates chat-line
# entrances, modal/toast reveals, etc. via Textual's native `styles.animate()` +
# CSS `transition:`. With the animation level left at its env default ("full")
# a headless pilot could observe a widget mid-tween (e.g. opacity 0) right after
# mount and flake. Forcing the level to "none" makes every animation + CSS
# transition resolve INSTANTLY to its final value, so tests always see the
# settled state. `App.__init__` reads `constants.TEXTUAL_ANIMATIONS` at call
# time, so patching the module attribute here (before any test builds an app)
# covers every app variant the suite constructs.
import textual.constants as _textual_constants

_textual_constants.TEXTUAL_ANIMATIONS = "none"


@pytest.fixture(autouse=True)
def _isolate_library_view_state(monkeypatch, tmp_path):
    """§8 P3 — redirect the persisted LibraryScreen view-state
    sidecar to a per-test ``tmp_path`` so a stale
    ``~/.local/state/care/library_view.json`` (from interactive
    debugging or a previous CI run) can't leak into pilot
    assertions about the default sort/filter state.

    Caught in iter 63: a leftover ``search: "/"`` on disk
    pre-populated the LibraryScreen's search input before the
    `/` absorber test could fire, making the bug invisible
    until the file was manually cleared. The env override
    (`CARE_VIEW_STATE_PATH`) reads lazily inside
    `LibraryViewStateStore.__init__` so every test cell gets
    its own fresh slate.
    """
    monkeypatch.setenv(
        "CARE_VIEW_STATE_PATH",
        str(tmp_path / "library_view.json"),
    )
    yield


@pytest.fixture(autouse=True)
def _pin_ui_language_english():
    """The TUI now defaults to Russian (`config.defaults.ui_language="ru"`),
    but the bulk of the screen tests assert ENGLISH copy. Pin the UI
    language to English for tests so those assertions stay valid; tests that
    exercise localization set the language explicitly. Default-Russian
    behaviour is covered in ``tests/test_i18n.py``.
    """
    from care.runtime import i18n

    i18n.set_ui_language("en")
    yield
    i18n.set_ui_language("en")
