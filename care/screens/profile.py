"""ProfileScreen — list + audit credential profiles (§6 P2).

Surfaces every TOML file under
``~/.config/care/profiles/`` plus the currently-active
selection (`CARE_PROFILE` env var). Useful for the
dev / prod / sandbox cred split without re-running
onboarding.

Bindings:

* ``r`` — re-read the profiles directory.
* ``Esc`` — pop the screen.

Switching profile in-session is intentionally left for a
follow-up — the config-precedence stack needs a careful
refactor. This screen lets the user audit which profiles
exist + surfaces the exact `export` command they'd run to
switch on the next boot.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Static

from care.runtime.i18n import t
from care.runtime.profiles import (
    ProfileInfo,
    active_profile_name,
    list_profiles,
)
from care.widgets.footer import CareFooter
from care.widgets.header import CareHeader

_log = logging.getLogger("care.screen.profile")


_COLUMN_KEYS: tuple[str, ...] = (
    "Active",
    "Name",
    "Path",
    "Modified",
    "Size",
)


def _columns() -> tuple[str, ...]:
    return (
        t("profile.colActive"),
        t("profile.colName"),
        t("profile.colPath"),
        t("profile.colModified"),
        t("profile.colSize"),
    )


class ProfileScreen(Screen):
    """Profile audit + selector."""

    DEFAULT_CSS = """
    ProfileScreen {
        layout: vertical;
    }
    ProfileScreen #profile-body {
        height: 1fr;
        padding: 0 1;
    }
    ProfileScreen #profile-table {
        height: 1fr;
    }
    ProfileScreen #profile-empty {
        padding: 1 2;
        color: $text-muted;
    }
    ProfileScreen #profile-hint {
        padding: 0 2;
        color: $text-muted;
    }
    ProfileScreen #profile-status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(
        self, *, config_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._config_dir = config_dir
        self.rows: tuple[ProfileInfo, ...] = ()
        self.active_name: str = ""
        self.last_error: str | None = None
        self.action_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield CareHeader()
        with Vertical(id="profile-body"):
            yield DataTable(id="profile-table")
            yield Static(" ", id="profile-empty")
            yield Static(
                t("profile.hint"),
                id="profile-hint",
            )
        yield Static(" ", id="profile-status")
        yield CareFooter()

    def on_mount(self) -> None:
        try:
            self.query_one(CareHeader).refresh_from_app(
                active_screen="ProfileScreen",
                breadcrumb=(t("header.breadcrumb.profile"),),
            )
        except Exception:
            pass
        try:
            table = self.query_one("#profile-table", DataTable)
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
        try:
            self.rows = tuple(list_profiles(
                config_dir=self._config_dir,
            ))
            self.active_name = active_profile_name()
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

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _apply_view(self) -> None:
        try:
            table = self.query_one("#profile-table", DataTable)
            empty = self.query_one("#profile-empty", Static)
            status = self.query_one("#profile-status", Static)
        except Exception:
            return
        table.clear()
        for row in self.rows:
            is_active = row.name == self.active_name
            table.add_row(
                "✓" if is_active else " ",
                row.name,
                _format_path(row.path),
                _format_when(row.mtime),
                _format_size(row.size_bytes),
                key=row.name,
            )
        is_empty = not self.rows
        empty.display = is_empty and not self.last_error
        if is_empty and not self.last_error:
            empty.update(t("profile.empty"))
        else:
            empty.update(" ")
        if self.last_error:
            status.update(f"⚠ {self.last_error}")
            return
        count_n = len(self.rows)
        count_key = (
            "profile.statusOne"
            if count_n == 1
            else "profile.statusMany"
        )
        count = t(count_key, n=count_n)
        if self.active_name:
            status.update(
                t(
                    "profile.statusActive",
                    count=count,
                    name=self.active_name,
                ),
            )
        else:
            status.update(
                t("profile.statusDefault", count=count),
            )


# ---------------------------------------------------------------------------
# Pure formatters
# ---------------------------------------------------------------------------


def _format_path(path: Path) -> str:
    text = str(path)
    if len(text) <= 56:
        return text
    return "…" + text[-55:]


def _format_when(mtime: float) -> str:
    if not mtime:
        return "—"
    return time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(mtime),
    )


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


__all__ = [
    "ProfileScreen",
]
