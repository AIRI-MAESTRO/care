"""Header/footer global key-bindings data layer (TODO §1.1 P0).

CARE binds five global key chords:

* ``Ctrl+P`` — open command palette
* ``Ctrl+Q`` — quit the application
* ``Ctrl+S`` — save current artifact to memory
* ``Ctrl+R`` — re-run the current artifact
* ``Esc``   — back / dismiss

The Textual screen wraps these via `Binding(...)` declarations
and the header/footer widgets render the binding hints. Both
surfaces are gated on TODO §1 P0 multi-screen workflow, but the
binding registry + dispatch + header/footer projection land now
as the data layer.

What this module provides:

* :class:`GlobalActionId` literal pinning the five canonical
  action names.
* :class:`BindingScope` literal — ``"always"`` (every screen),
  ``"screen"`` (any non-modal), ``"modal"`` (in-modal only).
* :class:`GlobalBinding` — frozen binding descriptor (key,
  label, action_id, scope, enable-condition string).
* :class:`HeaderModel` — frozen header projection (app title +
  breadcrumb + version + active screen).
* :class:`FooterModel` — frozen footer projection (binding hints
  to render right-aligned).
* :func:`default_global_bindings` — canonical 5-binding tuple
  matching the TODO spec.
* :func:`find_binding_by_key` — case-insensitive key dispatch.
* :func:`find_binding_by_action` — canonical-name lookup.
* :func:`bindings_for_scope` — filter by current scope.
* :func:`build_footer` — pure projection from a screen name +
  binding registry → :class:`FooterModel`.
* :func:`build_header` — pure projection for the page header.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from care.runtime.i18n import t


GlobalActionId = Literal[
    "open_command_palette",
    "quit",
    "save_artifact",
    "rerun_artifact",
    "back",
    "delete_row",
    "back_to_chat",
]
"""The canonical global action ids. The first five are the
TODO §1.1 P0 set; ``delete_row`` and ``back_to_chat`` are
screen-scoped extras the Library injects into its footer
registry so the destructive row action and the return-to-chat
gesture are discoverable (the chords themselves live on
:class:`LibraryScreen`)."""


BindingScope = Literal["always", "screen", "modal"]
"""Where the binding applies:

* ``always`` — every screen + every modal.
* ``screen`` — full-screen views only (not modals).
* ``modal`` — modal screens only (used for `Esc` dismiss).
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GlobalBindingError(RuntimeError):
    """Raised for caller-mistake failures — unknown action id,
    duplicate key on a custom registry. Per-call dispatch
    misses don't raise; :func:`find_binding_by_key` returns
    ``None``."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalBinding:
    """One global key binding.

    Frozen so the registry flows through Textual messages
    without defensive copies. ``key`` is the canonical chord
    label (e.g. ``"Ctrl+P"``, ``"Esc"``); the Textual screen
    converts to its native `Binding(key="ctrl+p", ...)` form via
    :meth:`textual_key` (lowercased + spaces stripped).
    """

    action_id: GlobalActionId
    key: str
    label: str
    scope: BindingScope = "always"
    description: str = ""

    @property
    def textual_key(self) -> str:
        """Lowercase / dashed form Textual's `Binding` expects
        (``"ctrl+p"``, ``"escape"``). Conversion is local to
        this property so the canonical `key` stays human-
        readable in headers / footers."""
        normalised = self.key.casefold().replace(" ", "")
        if normalised == "esc":
            return "escape"
        return normalised

    def applies_to(self, scope: BindingScope) -> bool:
        """``True`` when this binding is active in the given
        scope. ``always`` bindings match every scope; explicit
        ``screen`` / ``modal`` only match themselves.
        """
        if self.scope == "always":
            return True
        return self.scope == scope


@dataclass(frozen=True)
class HeaderModel:
    """Header projection the future Header widget renders.

    Frozen — a fresh instance per screen transition.
    """

    title: str = "MAESTRO"
    breadcrumb: tuple[str, ...] = ()
    version: str = ""
    active_screen: str = ""

    @property
    def breadcrumb_text(self) -> str:
        """``"Library › Inspection › v3"`` formatting."""
        return " › ".join(self.breadcrumb)


@dataclass(frozen=True)
class FooterHint:
    """One key-hint entry in the footer.

    Frozen so the footer can hold snapshots safely.
    """

    key: str
    label: str
    action_id: GlobalActionId


@dataclass(frozen=True)
class FooterModel:
    """Footer projection — a tuple of :class:`FooterHint` rows
    the widget renders right-aligned. ``active_screen``
    informs the conditional visibility logic.
    """

    hints: tuple[FooterHint, ...] = ()
    active_screen: str = ""

    def __len__(self) -> int:
        return len(self.hints)

    def __iter__(self):
        return iter(self.hints)


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def default_global_bindings() -> tuple[GlobalBinding, ...]:
    """Canonical 5-binding registry matching the TODO §1.1 P0
    spec. Returned as a tuple so callers can extend without
    mutating module state.

    Built fresh on every call (rather than a module-level constant)
    so each binding's label + description resolves :func:`t` in the
    active UI language at access time — the footer/header repaint in
    the chosen language on the next screen transition.
    """
    return (
        GlobalBinding(
            action_id="open_command_palette",
            key="Ctrl+P",
            label=t("globalBindings.palette.label"),
            scope="always",
            description=t("globalBindings.palette.description"),
        ),
        GlobalBinding(
            action_id="save_artifact",
            key="Ctrl+S",
            label=t("globalBindings.save.label"),
            scope="screen",
            description=t("globalBindings.save.description"),
        ),
        GlobalBinding(
            action_id="rerun_artifact",
            key="Ctrl+R",
            label=t("globalBindings.rerun.label"),
            scope="screen",
            description=t("globalBindings.rerun.description"),
        ),
        GlobalBinding(
            action_id="back",
            key="Esc",
            label=t("common.back"),
            scope="always",
            description=t("globalBindings.back.description"),
        ),
        GlobalBinding(
            action_id="quit",
            key="Ctrl+Q",
            label=t("globalBindings.quit.label"),
            scope="always",
            description=t("globalBindings.quit.description"),
        ),
    )


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def find_binding_by_key(
    key: str,
    *,
    registry: Optional[Iterable[GlobalBinding]] = None,
    scope: Optional[BindingScope] = None,
) -> Optional[GlobalBinding]:
    """Look up a binding by its ``key`` chord.

    Case-insensitive match + whitespace tolerant so callers
    can pass through whatever the keymap emits. Returns
    ``None`` when no binding matches the key (or matches but is
    scoped out of the current context).
    """
    if not key:
        return None
    bindings = registry if registry is not None else default_global_bindings()
    needle = key.casefold().replace(" ", "")
    for binding in bindings:
        if binding.textual_key != needle:
            continue
        if scope is not None and not binding.applies_to(scope):
            continue
        return binding
    return None


def find_binding_by_action(
    action_id: GlobalActionId,
    *,
    registry: Optional[Iterable[GlobalBinding]] = None,
) -> Optional[GlobalBinding]:
    """Look up a binding by its canonical action id. Useful for
    the header/footer when rendering the key for a specific
    action ("save: Ctrl+S")."""
    bindings = registry if registry is not None else default_global_bindings()
    for binding in bindings:
        if binding.action_id == action_id:
            return binding
    return None


def bindings_for_scope(
    scope: BindingScope,
    *,
    registry: Optional[Iterable[GlobalBinding]] = None,
) -> tuple[GlobalBinding, ...]:
    """Filter the registry by the current scope.

    Returns bindings in their declaration order, dropping any
    whose scope doesn't apply.
    """
    bindings = registry if registry is not None else default_global_bindings()
    return tuple(b for b in bindings if b.applies_to(scope))


def validate_registry(
    registry: Iterable[GlobalBinding],
) -> None:
    """Ensure the registry has no duplicate keys.

    The screen's `Binding` declarations don't tolerate the same
    key wired to two actions — validate upstream so the user
    sees a clear error rather than a Textual-layer crash.
    Raises :class:`GlobalBindingError` on the first duplicate.
    """
    seen: dict[str, str] = {}
    for binding in registry:
        key = binding.textual_key
        if key in seen:
            raise GlobalBindingError(
                f"duplicate binding for key {binding.key!r}: "
                f"{seen[key]!r} and {binding.action_id!r}"
            )
        seen[key] = binding.action_id


# ---------------------------------------------------------------------------
# Header / footer projection
# ---------------------------------------------------------------------------


def build_header(
    *,
    active_screen: str = "",
    breadcrumb: Iterable[str] = (),
    version: str = "",
    title: str = "MAESTRO",
) -> HeaderModel:
    """Build a :class:`HeaderModel` from the current screen
    state. Pure projection — no I/O. Empty breadcrumb is fine
    (top-level screens omit it)."""
    return HeaderModel(
        title=title,
        breadcrumb=tuple(b for b in breadcrumb if b),
        version=version,
        active_screen=active_screen,
    )


def build_footer(
    *,
    active_screen: str = "",
    scope: BindingScope = "screen",
    registry: Optional[Iterable[GlobalBinding]] = None,
) -> FooterModel:
    """Project the active bindings into a :class:`FooterModel`.

    Only bindings that apply to ``scope`` are surfaced. The
    declaration order is preserved (matches the spec: Palette,
    Save, Re-run, Back, Quit).
    """
    visible = bindings_for_scope(scope, registry=registry)
    hints = tuple(
        FooterHint(
            key=b.key,
            label=b.label,
            action_id=b.action_id,
        )
        for b in visible
    )
    return FooterModel(hints=hints, active_screen=active_screen)


# Re-export the unused `field` marker for future dataclass
# extensions.
_ = field


__all__ = [
    "BindingScope",
    "FooterHint",
    "FooterModel",
    "GlobalActionId",
    "GlobalBinding",
    "GlobalBindingError",
    "HeaderModel",
    "bindings_for_scope",
    "build_footer",
    "build_header",
    "default_global_bindings",
    "find_binding_by_action",
    "find_binding_by_key",
    "validate_registry",
]
