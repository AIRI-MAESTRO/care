"""UseItNowModal — post-save chain-id reveal (TODO §3 P0).

Pushed by the save flow after a chain artifact persists to
Memory. The modal closes the "what now?" loop by surfacing
the **chain_id** alongside three copy-ready
integration snippets the user can paste into their own
service.

Snippets:

* ``python``  — gigaevo-client recipe:
  ``client.get_chain(id, channel="latest")`` +
  ``client.run(chain, task=…)``.
* ``curl``    — direct GET against Memory's chain endpoint
  with the auth + base-url envs the user already sets.
* ``cli``     — ``care run <id> --execute --task "…"``
  for shell automation.

Bindings:

* ``y`` — copy the chain_id to clipboard.
* ``c`` — copy the active language's snippet.
* ``t`` — cycle through the three snippet languages.
* ``e`` — open the §4 EvolutionLaunchModal pre-filled
  with this chain_id (graceful no-op when the launch
  modal isn't available).
* ``Esc`` — close.

Triggered by:

* ArtifactsScreen's `s` (save) binding after a successful
  Memory persist.
* SaveReport modal (§3 P1 follow-up — not yet wired).
* InspectionScreen "Integration" pane (§4 P0 — also
  pending).

The modal is pure presentation — does NOT perform any
mutation. The "Evolve this now" affordance pops + asks
the host to open the existing
:class:`EvolutionLaunchModal`; the host wires this up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, TextArea

from care.runtime.i18n import t

_log = logging.getLogger("care.screen.use_it_now")


SnippetLang = Literal["python", "curl", "cli"]
"""Available snippet languages — `t` cycles through these."""


_LANG_ORDER: tuple[SnippetLang, ...] = ("python", "curl", "cli")


def render_integration_snippet(
    lang: SnippetLang,
    *,
    entity_id: str,
    channel: str = "latest",
    memory_base_url: str = "${CARE_MEMORY__BASE_URL}",
) -> str:
    """Pure projection of one integration snippet — the
    UseItNowModal + §4 InspectionScreen Integration pane both
    use this helper so the recipes stay identical regardless
    of where the user reads them."""
    if lang == "python":
        return (
            "import asyncio\n"
            "from gigaevo_client import GigaEvoClient, GigaEvoConfig\n"
            "from mmar_carl import (\n"
            "    DAGExecutor, ReasoningContext, create_openai_client,\n"
            ")\n"
            "\n"
            "client = GigaEvoClient.from_config(GigaEvoConfig.from_env())\n"
            f"chain = client.get_chain(\"{entity_id}\", channel=\"{channel}\")\n"
            "\n"
            "ctx = ReasoningContext(\n"
            "    outer_context=\"…\",  # your task input\n"
            "    api=create_openai_client(),\n"
            "    endpoint_key=\"default\",\n"
            ")\n"
            "result = asyncio.run(\n"
            "    DAGExecutor(max_workers=chain.max_workers)\n"
            "    .execute(chain.steps, ctx)\n"
            ")"
        )
    if lang == "curl":
        return (
            "curl \\\n"
            "  -H \"X-API-Key: "
            "${CARE_MEMORY__API_KEY}\" \\\n"
            f"  \"{memory_base_url}/v1/chains/{entity_id}\""
            f" \\\n"
            f"  --get --data-urlencode \"channel={channel}\""
        )
    if lang == "cli":
        return (
            f"care run {entity_id} \\\n"
            "  --execute \\\n"
            "  --task \"…\""
        )
    raise ValueError(f"unknown snippet lang: {lang!r}")


def cycle_language(current: SnippetLang) -> SnippetLang:
    """Walk `python → curl → cli → python`. Exposed so the
    InspectionScreen's `t` binding can drive the same cycle
    as the modal."""
    idx = _LANG_ORDER.index(current)
    return _LANG_ORDER[(idx + 1) % len(_LANG_ORDER)]


def snippet_language(lang: SnippetLang) -> str:
    """Map a snippet language to the tree-sitter grammar name a read-only
    `TextArea` uses for syntax highlighting. ``curl`` / ``cli`` are shell
    one-liners, so they highlight as ``bash``."""
    return "python" if lang == "python" else "bash"


def lang_indicator(current: SnippetLang) -> str:
    """Render the `[python]  curl   cli` row with the active
    language bracketed."""
    parts = []
    for lang in _LANG_ORDER:
        if lang == current:
            parts.append(f"[{lang}]")
        else:
            parts.append(f" {lang} ")
    return f"snippet:  {''.join(parts)}    (t to cycle)"


@dataclass(frozen=True)
class UseItNowResult:
    """Dismiss envelope.

    The host listens for ``evolve_requested`` to open the
    EvolutionLaunchModal — the modal itself stays
    presentation-only.
    """

    closed: bool = True
    evolve_requested: bool = False


class UseItNowModal(ModalScreen[UseItNowResult]):
    """Compact modal showing the chain_id + integration
    snippets after a successful save.

    Construct with the resolved chain_id + version + channel.
    Optional ``display_name`` is shown in the header title;
    optional ``memory_base_url`` injects into the curl
    snippet so the user can paste it without filling in the
    server address by hand.
    """

    DEFAULT_CSS = """
    UseItNowModal {
        align: center middle;
    }
    UseItNowModal #use-it-now-box {
        width: 90;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    UseItNowModal #use-it-now-title {
        text-style: bold;
        padding-bottom: 1;
    }
    UseItNowModal #use-it-now-meta {
        color: $text-muted;
        margin-bottom: 1;
    }
    UseItNowModal #use-it-now-id {
        color: $accent;
        margin-bottom: 1;
    }
    UseItNowModal #use-it-now-lang-row {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    UseItNowModal #use-it-now-snippet {
        height: 18;
        background: $boost;
        padding: 0 1;
        margin-bottom: 1;
    }
    UseItNowModal #use-it-now-actions {
        height: 3;
        align-horizontal: right;
    }
    UseItNowModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("y", "copy_id", "Copy ID", show=True),
        Binding("c", "copy_snippet", "Copy snippet", show=True),
        Binding("t", "cycle_lang", "Cycle lang", show=True),
        Binding("e", "evolve", "Evolve", show=True),
    ]

    def __init__(
        self,
        *,
        entity_id: str,
        version: str | None = None,
        channel: str = "latest",
        display_name: str = "",
        memory_base_url: str = "",
    ) -> None:
        super().__init__()
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        self.entity_id = entity_id
        self.version = version or "latest"
        self.channel = channel or "latest"
        self.display_name = display_name
        self.memory_base_url = (
            memory_base_url
            or "${CARE_MEMORY__BASE_URL}"
        )
        self.active_lang: SnippetLang = "python"
        # Action log — tests + telemetry read this rather than
        # scraping internal state (mirrors ArtifactsScreen +
        # EvolutionDashboard conventions).
        self.action_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="use-it-now-box"):
            yield Label(self._title_text(), id="use-it-now-title")
            yield Static(self._meta_text(), id="use-it-now-meta")
            yield Static(
                f"chain_id: {self.entity_id}",
                id="use-it-now-id",
            )
            yield Static(
                self._lang_row_text(), id="use-it-now-lang-row",
            )
            yield TextArea(
                self.render_snippet(self.active_lang),
                language=snippet_language(self.active_lang),
                read_only=True,
                id="use-it-now-snippet",
            )
            with Horizontal(id="use-it-now-actions"):
                yield Button(t("common.close"), id="use-it-now-btn-close")
                yield Button(
                    t("useItNow.evolveThisNow"),
                    id="use-it-now-btn-evolve",
                    variant="primary",
                )

    # ------------------------------------------------------------------
    # Snippet rendering
    # ------------------------------------------------------------------

    def render_snippet(self, lang: SnippetLang) -> str:
        """Instance shim around :func:`render_integration_snippet`
        so existing callers keep working. The pure helper is
        the canonical surface — both this modal and the §4
        InspectionScreen Integration pane render from it."""
        return render_integration_snippet(
            lang,
            entity_id=self.entity_id,
            channel=self.channel,
            memory_base_url=self.memory_base_url,
        )

    def _title_text(self) -> str:
        if self.display_name:
            return t("useItNow.savedNamed", name=self.display_name)
        return t("useItNow.savedToMemory")

    def _meta_text(self) -> str:
        parts = [
            t("useItNow.version", version=self.version),
            t("useItNow.channel", channel=self.channel),
        ]
        return "  ·  ".join(parts)

    def _lang_row_text(self) -> str:
        """Render the [python] [curl] [cli] row with the
        active language highlighted in brackets."""
        return lang_indicator(self.active_lang)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_close(self) -> None:
        self.action_log.append(("close", ""))
        self.dismiss(UseItNowResult(closed=True, evolve_requested=False))

    def action_copy_id(self) -> None:
        self.action_log.append(("copy_id", self.entity_id))
        self._copy(self.entity_id, "id")

    def action_copy_snippet(self) -> None:
        self.action_log.append(("copy_snippet", self.active_lang))
        self._copy(
            self.render_snippet(self.active_lang),
            f"{self.active_lang} snippet",
        )

    def action_cycle_lang(self) -> None:
        nxt = cycle_language(self.active_lang)
        self.active_lang = nxt
        self.action_log.append(("cycle_lang", nxt))
        self._refresh_view()

    def action_evolve(self) -> None:
        self.action_log.append(("evolve", self.entity_id))
        self.dismiss(UseItNowResult(closed=False, evolve_requested=True))

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh_view(self) -> None:
        try:
            lang_row = self.query_one(
                "#use-it-now-lang-row", Static,
            )
            snippet = self.query_one(
                "#use-it-now-snippet", TextArea,
            )
        except Exception:
            return
        lang_row.update(self._lang_row_text())
        snippet.language = snippet_language(self.active_lang)
        snippet.text = self.render_snippet(self.active_lang)

    def _copy(self, text: str, label: str) -> None:
        """Copy ``text`` to the clipboard with a friendly toast.

        Routes through `care.runtime.clipboard.copy_text` to
        get the OSC-52 + pbcopy/xclip/wl-copy fallback ladder
        — same path the chat screen uses for `super+c`."""
        try:
            from care.runtime.clipboard import copy_text

            copy_text(text)
        except Exception as exc:  # noqa: BLE001
            self._toast(
                t("useItNow.copyFailed", error=str(exc)), severity="warning",
            )
            return
        self._toast(
            t("useItNow.copied", label=label), severity="info",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "use-it-now-btn-close":
            self.action_close()
        elif event.button.id == "use-it-now-btn-evolve":
            self.action_evolve()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _toast(self, message: str, *, severity: str = "info") -> None:
        push = getattr(self.app, "push_toast", None)
        if callable(push):
            try:
                push(message, severity=severity)
                return
            except Exception:
                pass
        _log.info(
            "UseItNowModal toast [%s]: %s", severity, message,
        )


__all__ = [
    "SnippetLang",
    "UseItNowModal",
    "UseItNowResult",
    "cycle_language",
    "lang_indicator",
    "render_integration_snippet",
]
