"""SandboxTrustScreen — list + revoke approved AgentSkills
(TODO §6 P1).

Surfaces every entry in the persistent
:class:`care.sandbox.SkillTrustStore` (default location
``~/.local/state/care/skill_trust.json``). The user can:

* Scroll the table of trusted skills — name, URI, SHA prefix,
  approval timestamp, trust policy, allowed tools.
* Revoke a row with ``r`` (after a ConfirmModal). Revocation
  removes the entry + persists the store; the next CARL
  execution that hits the same SKILL.md SHA will trigger the
  re-approval prompt.
* Refresh with ``R`` (force-reload from disk in case another
  process / CLI invocation mutated the store).
* ``Esc`` pops the screen.

This is the user-facing audit + revoke surface; the
``always allow`` / ``ask`` / ``block`` 3-state policy from
the spec is split out as a §6 [P1] follow-up since the
current data layer only stores `trusted` (sha-pinned)
records — adding ``ask`` / ``block`` requires extending
:class:`TrustRecord` + bumping the store format version.
"""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static

from care.runtime.i18n import t
from care.sandbox.trust import (
    SkillTrustStore,
    TrustRecord,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screen.sandbox_trust")


_COLUMN_KEYS: tuple[str, ...] = (
    "Name",
    "URI",
    "SHA",
    "Approved",
    "Policy",
    "Tools",
)


def _columns() -> tuple[str, ...]:
    return (
        t("sandboxTrust.colName"),
        t("sandboxTrust.colUri"),
        t("sandboxTrust.colSha"),
        t("sandboxTrust.colApproved"),
        t("sandboxTrust.colPolicy"),
        t("sandboxTrust.colTools"),
    )


class SandboxTrustScreen(Screen):
    """Persistent skill-trust audit screen."""

    DEFAULT_CSS = """
    SandboxTrustScreen {
        layout: vertical;
    }
    SandboxTrustScreen #sandbox-trust-body {
        height: 1fr;
        padding: 0 1;
    }
    SandboxTrustScreen #sandbox-trust-table {
        height: 1fr;
    }
    SandboxTrustScreen #sandbox-trust-empty {
        padding: 1 2;
        color: $text-muted;
    }
    SandboxTrustScreen #sandbox-trust-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("R", "refresh", "Refresh", show=True),
        Binding("r", "revoke", "Revoke", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(
        self,
        *,
        store: SkillTrustStore | None = None,
    ) -> None:
        super().__init__()
        # Tests inject a pre-built store; production calls
        # :meth:`SkillTrustStore.load` lazily on first paint
        # so the screen doesn't touch disk at construction
        # time (matches the rest of the CARE screen contract).
        self._injected_store = store
        self._store: SkillTrustStore | None = store
        self.rows: tuple[TrustRecord, ...] = ()
        self.last_error: str | None = None
        self.action_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Vertical(id="sandbox-trust-body"):
            yield DataTable(id="sandbox-trust-table")
            yield Static(" ", id="sandbox-trust-empty")
        yield Static(" ", id="sandbox-trust-status")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="SandboxTrustScreen",
                breadcrumb=(t("header.breadcrumb.sandbox"), t("header.breadcrumb.trust")),
            )
        except Exception:
            pass
        try:
            table = self.query_one(
                "#sandbox-trust-table", DataTable,
            )
            for label, col_key in zip(_columns(), _COLUMN_KEYS):
                table.add_column(label, key=col_key)
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        self.app.call_after_refresh(self.refresh_rows)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_rows(self) -> None:
        """Reload from the underlying store + repaint.

        Production calls :meth:`SkillTrustStore.load` on
        each refresh so external mutations (CLI / another
        CARE process) surface here. Tests pass a pre-built
        store; we honour the injection by re-reading the
        same instance's :meth:`list_trusted` rather than
        triggering a disk load.
        """
        try:
            if self._injected_store is not None:
                self._store = self._injected_store
            else:
                self._store = SkillTrustStore.load()
            self.rows = tuple(self._store.list_trusted())
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.rows = ()
            self.last_error = f"{type(exc).__name__}: {exc}"
        self._apply_view()

    def action_refresh(self) -> None:
        self.action_log.append(("refresh", ""))
        self.refresh_rows()

    def action_back(self) -> None:
        self.action_log.append(("back", ""))
        try:
            self.app.pop_screen()
        except Exception:
            pass

    def action_revoke(self) -> None:
        """`r` — revoke the focused row's trust record.

        Opens a `ConfirmModal` before persisting so an
        accidental keystroke can't downgrade the security
        posture. On confirm: calls `store.revoke(sha256)`,
        persists, then re-renders the table."""
        record = self.current_record
        if record is None:
            self._toast(
                t("sandboxTrust.highlightFirst"), severity="info",
            )
            return
        self.action_log.append(("revoke", record.sha256))
        try:
            from care.screens.confirm import ConfirmModal
        except Exception:
            # Modal unavailable → fall back to direct revoke
            # rather than blocking the user.
            self._do_revoke(record)
            return

        def _on_confirm(result):
            if not result:
                return
            self._do_revoke(record)

        self.app.push_screen(
            ConfirmModal(
                title=t("sandboxTrust.revokeTitle"),
                body=(
                    f"{record.name}\n"
                    f"{record.uri}\n"
                    f"sha {_format_sha(record.sha256)}\n\n"
                    + t("sandboxTrust.revokeBody")
                ),
                confirm_label=t("sandboxTrust.revoke"),
            ),
            _on_confirm,
        )

    def _do_revoke(self, record: TrustRecord) -> None:
        if self._store is None:
            self._toast(
                t("sandboxTrust.storeNotLoaded"), severity="error",
            )
            return
        try:
            removed = self._store.revoke(record.sha256)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "SandboxTrustScreen revoke sha=%s failed: %s",
                record.sha256, exc, exc_info=False,
            )
            self._toast(
                t("sandboxTrust.revokeFailed", error=exc),
                severity="error",
            )
            return
        if not removed:
            self._toast(
                t("sandboxTrust.alreadyRevoked"), severity="info",
            )
            return
        self._toast(
            t("sandboxTrust.revoked", name=record.name),
            severity="success",
        )
        self.refresh_rows()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_record(self) -> TrustRecord | None:
        if not self.rows:
            return None
        try:
            table = self.query_one(
                "#sandbox-trust-table", DataTable,
            )
        except Exception:
            return None
        idx = table.cursor_row
        if idx is None or idx < 0 or idx >= len(self.rows):
            return None
        return self.rows[idx]

    def _apply_view(self) -> None:
        try:
            table = self.query_one(
                "#sandbox-trust-table", DataTable,
            )
            empty = self.query_one(
                "#sandbox-trust-empty", Static,
            )
            status = self.query_one(
                "#sandbox-trust-status", Static,
            )
        except Exception:
            return
        table.clear()
        for record in self.rows:
            table.add_row(
                record.name or "—",
                _format_uri(record.uri),
                _format_sha(record.sha256),
                _format_when(record.approved_at),
                record.trust_policy or "—",
                _format_tools(record.allowed_tools),
                key=record.sha256,
            )
        is_empty = not self.rows
        empty.display = is_empty and not self.last_error
        if is_empty and not self.last_error:
            empty.update(t("sandboxTrust.empty"))
        else:
            empty.update(" ")
        if self.last_error:
            status.update(f"⚠ {self.last_error}")
        else:
            count = len(self.rows)
            key = (
                "sandboxTrust.statusOne"
                if count == 1
                else "sandboxTrust.statusMany"
            )
            status.update(t(key, n=count))

    def _toast(self, message: str, *, severity: str = "info") -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        _log.info(
            "SandboxTrustScreen toast [%s]: %s",
            severity, message,
        )


# ---------------------------------------------------------------------------
# Pure formatters (testable without Textual)
# ---------------------------------------------------------------------------


def _format_sha(sha: str) -> str:
    if not sha:
        return "—"
    if len(sha) <= 12:
        return sha
    return f"{sha[:8]}…"


def _format_uri(uri: str) -> str:
    if not uri:
        return "—"
    if len(uri) <= 48:
        return uri
    return uri[:45] + "…"


def _format_when(approved_at: Any) -> str:
    if approved_at is None:
        return "—"
    try:
        return approved_at.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"


def _format_tools(tools: tuple[str, ...]) -> str:
    if not tools:
        return "—"
    if len(tools) <= 3:
        return ", ".join(tools)
    return f"{', '.join(tools[:3])} (+{len(tools) - 3})"


__all__ = [
    "SandboxTrustScreen",
]
