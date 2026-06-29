"""Tests for the library "Revise (AI)" row-action (B4).

``action_row_revise`` is a standalone action (not a typed ``RowActionKind``):
it resolves the focused row's ``entity_id`` and asks the app to drop to chat
with ``/revise <id> `` seeded. Tested by overriding ``current_row`` / ``app``
on a subclass so no full mount is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from care.screens.library import LibraryScreen


class _LS(LibraryScreen):
    """LibraryScreen with injectable ``current_row`` + ``app`` for unit tests."""

    def __init__(self, row: Any, app_obj: Any) -> None:
        super().__init__()
        self._row = row
        self._app_obj = app_obj

    @property
    def current_row(self) -> Any:  # type: ignore[override]
        return self._row

    @property
    def app(self) -> Any:  # type: ignore[override]
        return self._app_obj


def _fake_app() -> SimpleNamespace:
    calls: list[str] = []
    return SimpleNamespace(_revise_chain_for=calls.append, calls=calls)


def test_revise_binding_registered() -> None:
    matches = [
        b
        for b in LibraryScreen.BINDINGS
        if getattr(b, "key", None) == "R" and getattr(b, "action", None) == "row_revise"
    ]
    assert matches, "expected an `R` → row_revise binding on LibraryScreen"


def test_action_row_revise_hands_off_entity_id() -> None:
    app = _fake_app()
    ls = _LS(SimpleNamespace(entity_id="chain-7"), app)
    ls.action_row_revise()
    assert app.calls == ["chain-7"]


def test_action_row_revise_noop_without_row() -> None:
    app = _fake_app()
    ls = _LS(None, app)
    ls.action_row_revise()  # must not raise
    assert app.calls == []


def test_action_row_revise_noop_without_entity_id() -> None:
    app = _fake_app()
    ls = _LS(SimpleNamespace(entity_id=""), app)
    ls.action_row_revise()
    assert app.calls == []


def test_action_row_revise_tolerates_app_without_helper() -> None:
    ls = _LS(SimpleNamespace(entity_id="chain-7"), SimpleNamespace())  # no _revise_chain_for
    ls.action_row_revise()  # must not raise
