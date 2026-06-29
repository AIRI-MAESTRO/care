"""EvolutionLaunchModal — gather params before kicking off
an evolution run (TODO §4 P0).

Pushed by `LibraryScreen` row-action `evolve_with_data` (key
``E``) and by `CareApp._push_evolution_for(...)`. Surfaces a
small form so the user can pin:

* **Dataset** — JSONL file path with ``{input, expected}``
  rows, OR (future) a saved ``memory_card`` of kind
  ``dataset``. The file-pick UI is a plain ``Input`` for
  now — full picker UX ships under §6 ``/datasets``.
* **Budget** — ``max_iterations`` × ``population_size`` and a
  max wall-time (informational; the platform enforces).
* **Rubric** — short text describing what success looks like.
  Forwarded as `validation_criteria` to EvolutionScreen.
* **Constraints** — optional model whitelist + token-cost cap
  (also informational; the platform consumes if it knows
  how to honour).

On submit, the modal posts :class:`LaunchRequested` carrying
a frozen :class:`EvolutionLaunchSpec`. The caller (typically
`CareApp`) builds an ``EvolutionScreen(base_chain_id=…,
**spec.to_screen_kwargs())`` and pushes it. Cancel dismisses
with ``None``.

The shipped fields default to the same values
``EvolutionScreen.__init__`` does so a user pressing Enter
on first launch sees the canonical CARE-tuned defaults
rather than the platform's raw defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from care.runtime.evolution_validation import (
    BINARY_METHODS,
    CONTINUOUS_METRICS,
    DEFAULT_BINARY_METHOD,
    DEFAULT_CONTINUOUS_METRIC,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_VALIDATION_TYPE,
    VALIDATION_TYPES,
)
from care.runtime.i18n import t

# Rough per-evaluation token estimate for the launch budget preview. A
# chain evaluation runs the chain + an LLM mutation/judge; this is a
# deliberately conservative ballpark (clearly labelled "rough est.").
_TOKENS_PER_EVALUATION = 1500

_DEFAULT_MUTATION_MAX_TOKENS = 8192


def estimate_evolution_budget(
    max_iterations: int, population_size: int
) -> tuple[int, int]:
    """Return ``(evaluations, tokens)`` for the launch budget preview.

    ``evaluations = max_iterations × population_size``; tokens is a rough
    heuristic (``evaluations × _TOKENS_PER_EVALUATION``). Negative inputs
    clamp to 0 so a half-typed field doesn't show nonsense."""
    gens = max(0, int(max_iterations))
    pop = max(0, int(population_size))
    evaluations = gens * pop
    return evaluations, evaluations * _TOKENS_PER_EVALUATION


@dataclass(frozen=True)
class EvolutionLaunchSpec:
    """Frozen rendering of the modal's form state.

    Constructed from the modal's `_collect()` helper; the
    consumer (`CareApp`) builds an
    `EvolutionScreen(base_chain_id=..., **spec.to_screen_kwargs())`
    so the screen receives the user's choices.

    Fields are deliberately permissive (str / float / int) so
    a malformed value doesn't crash the modal — the screen's
    pydantic / runtime validation catches the rest.
    """

    base_chain_id: str
    dataset_path: str = ""
    rubric: str = ""
    validation_type: str = DEFAULT_VALIDATION_TYPE
    continuous_metric: str = DEFAULT_CONTINUOUS_METRIC
    binary_method: str = DEFAULT_BINARY_METHOD
    target_column: str = DEFAULT_TARGET_COLUMN
    max_iterations: int = 10
    population_size: int = 8
    max_wall_time_seconds: float | None = None
    model_whitelist: tuple[str, ...] = ()
    token_cost_cap: float | None = None
    mutation_max_tokens: int | None = None

    def to_screen_kwargs(self) -> dict[str, Any]:
        """Project into the kwargs `EvolutionScreen.__init__`
        understands. Empty / None fields drop so screen-side
        defaults win."""
        kw: dict[str, Any] = {
            "max_iterations": self.max_iterations,
            "population_size": self.population_size,
        }
        if self.dataset_path:
            kw["test_data_path"] = self.dataset_path
        if self.rubric:
            kw["validation_criteria"] = self.rubric
        kw["validation_type"] = self.validation_type
        kw["continuous_metric"] = self.continuous_metric
        kw["binary_method"] = self.binary_method
        kw["target_column"] = self.target_column
        if self.mutation_max_tokens is not None:
            kw["mutation_max_tokens"] = self.mutation_max_tokens
        return kw


class LaunchRequested(Message):
    """Posted when the user clicks Launch. Carries the form
    snapshot; consumer reads `spec` + `entity_id`."""

    def __init__(self, spec: EvolutionLaunchSpec) -> None:
        super().__init__()
        self.spec = spec


class EvolutionLaunchModal(ModalScreen[EvolutionLaunchSpec | None]):
    """Modal form gathering evolution kickoff parameters.

    Dismisses with an :class:`EvolutionLaunchSpec` on Launch,
    ``None`` on Cancel / Esc. The caller wraps the push with
    a callback that branches on the dismissal value:

    ```python
    def _on_dismiss(spec: EvolutionLaunchSpec | None) -> None:
        if spec is not None:
            app.push_screen(
                EvolutionScreen(
                    base_chain_id=spec.base_chain_id,
                    **spec.to_screen_kwargs(),
                ),
            )

    app.push_screen(
        EvolutionLaunchModal(base_chain_id=chain_id),
        _on_dismiss,
    )
    ```
    """

    DEFAULT_CSS = """
    EvolutionLaunchModal {
        align: center middle;
    }
    EvolutionLaunchModal #launch-box {
        width: 70%;
        max-width: 80;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $accent;
    }
    EvolutionLaunchModal #launch-title {
        text-style: bold;
        margin-bottom: 1;
    }
    EvolutionLaunchModal #launch-error {
        color: $error;
        text-style: bold;
        /* Collapsed until `action_submit` finds an empty base chain. */
        display: none;
    }
    EvolutionLaunchModal Label {
        margin-top: 1;
    }
    EvolutionLaunchModal #launch-dataset-row {
        height: auto;
    }
    EvolutionLaunchModal #launch-dataset-row #launch-dataset {
        width: 1fr;
    }
    EvolutionLaunchModal #launch-dataset-browse {
        width: auto;
        margin-left: 1;
    }
    EvolutionLaunchModal #launch-buttons {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    EvolutionLaunchModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+enter", "submit", "Launch", show=True),
    ]

    def __init__(self, *, base_chain_id: str = "") -> None:
        super().__init__()
        # `base_chain_id` may be empty: the Inspection / Library Evolve
        # buttons always pass the inspected chain, but `/evolution` from
        # chat opens the modal cold so the user can type / paste the chain
        # to evolve into the editable Base-chain field below.
        self.base_chain_id = base_chain_id or ""
        # Snapshot of the most recently collected spec — tests
        # + telemetry read this when the modal dismisses.
        self.last_spec: EvolutionLaunchSpec | None = None

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def _title_text(self) -> str:
        """Header line: name the pre-bound chain when there is one, else a
        generic "set up evolution" title for the cold (chat) entry."""
        if self.base_chain_id:
            return t("evolutionLaunch.title", id=self.base_chain_id)
        return t("evolutionLaunch.titleGeneric")

    def _default_mutation_max_tokens(self) -> int:
        """Config default for the mutation LLM completion limit."""
        try:
            from care.config import CareConfig

            return int(CareConfig.load().platform.mutation_max_tokens)
        except Exception:
            return _DEFAULT_MUTATION_MAX_TOKENS

    def compose(self) -> ComposeResult:
        with Vertical(id="launch-box"):
            yield Static(
                self._title_text(),
                id="launch-title",
            )
            yield Label(t("evolutionLaunch.baseChain"))
            yield Input(
                value=self.base_chain_id,
                placeholder=t("evolutionLaunch.baseChainPlaceholder"),
                id="launch-chain",
            )
            # Inline validation slot — hidden until the user tries to
            # launch without a base chain. markup=False: it never carries
            # Rich tags and we don't want a stray bracket to crash render.
            yield Static("", id="launch-error", markup=False)
            yield Label(t("evolutionLaunch.dataset"))
            with Horizontal(id="launch-dataset-row"):
                yield Input(
                    placeholder="~/data/eval.jsonl",
                    id="launch-dataset",
                )
                yield Button(
                    t("exportChain.browse"),
                    id="launch-dataset-browse",
                )
            yield Label(t("evolutionLaunch.rubric"))
            yield Input(
                placeholder=t("evolutionLaunch.rubricPlaceholder"),
                id="launch-rubric",
            )
            yield Label(t("evolutionLaunch.validationType"))
            yield Select(
                [(label, label) for label in VALIDATION_TYPES],
                value=DEFAULT_VALIDATION_TYPE,
                id="launch-validation-type",
                allow_blank=False,
            )
            yield Label(t("evolutionLaunch.continuousMetric"), id="launch-metric-label")
            yield Select(
                [(label, label) for label in CONTINUOUS_METRICS],
                value=DEFAULT_CONTINUOUS_METRIC,
                id="launch-continuous-metric",
                allow_blank=False,
            )
            yield Label(
                t("evolutionLaunch.binaryMethod"),
                id="launch-binary-label",
            )
            yield Select(
                [(label, label) for label in BINARY_METHODS],
                value=DEFAULT_BINARY_METHOD,
                id="launch-binary-method",
                allow_blank=False,
            )
            yield Label(t("evolutionLaunch.targetColumn"))
            yield Input(
                value=DEFAULT_TARGET_COLUMN,
                id="launch-target-column",
            )
            yield Label(t("evolutionLaunch.maxIterations"))
            yield Input(
                value="10", id="launch-max-iter",
            )
            yield Label(t("evolutionLaunch.populationSize"))
            yield Input(
                value="8", id="launch-pop",
            )
            # Live budget preview — updates as the user edits iterations /
            # population so they can size a run before launching.
            yield Static("", id="launch-budget")
            yield Label(t("evolutionLaunch.mutationMaxTokens"))
            yield Input(
                value=str(self._default_mutation_max_tokens()),
                id="launch-mutation-max-tokens",
            )
            yield Label(t("evolutionLaunch.maxWallTime"))
            yield Input(
                placeholder="3600", id="launch-walltime",
            )
            yield Label(t("evolutionLaunch.modelWhitelist"))
            yield Input(
                placeholder="gpt-4o-mini, claude-haiku-4-5",
                id="launch-models",
            )
            yield Label(t("evolutionLaunch.tokenCostCap"))
            yield Input(
                placeholder="5.00", id="launch-cost",
            )
            with Horizontal(id="launch-buttons"):
                yield Button(t("common.cancel"), id="launch-cancel")
                yield Button(
                    t("common.launch"), id="launch-submit", variant="primary",
                )

    # ------------------------------------------------------------------
    # Form → spec
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._sync_metric_field_visibility()
        self._render_budget_preview()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "launch-validation-type":
            self._sync_metric_field_visibility()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Recompute the budget preview when iterations / population change."""
        if event.input.id in ("launch-max-iter", "launch-pop"):
            self._render_budget_preview()

    def _render_budget_preview(self) -> None:
        """Repaint the budget Static from the current iteration / population
        inputs. Best-effort — no-ops when the pane isn't mounted."""
        try:
            pane = self.query_one("#launch-budget", Static)
        except Exception:
            return
        evaluations, tokens = estimate_evolution_budget(
            self._read_int("launch-max-iter", 10),
            self._read_int("launch-pop", 8),
        )
        pane.update(
            t(
                "evolutionLaunch.budget",
                evals=f"{evaluations:,}",
                pop=self._read_int("launch-pop", 8),
                gens=self._read_int("launch-max-iter", 10),
                tokens=f"{tokens:,}",
            )
        )

    def _sync_metric_field_visibility(self) -> None:
        """Show continuous vs binary metric controls like Platform Web UI."""
        try:
            vtype = self.query_one("#launch-validation-type", Select).value
        except Exception:
            return
        continuous = vtype != "Binary (0/1)"
        for widget_id in (
            "launch-metric-label",
            "launch-continuous-metric",
        ):
            try:
                self.query_one(f"#{widget_id}").display = continuous
            except Exception:
                pass
        for widget_id in ("launch-binary-label", "launch-binary-method"):
            try:
                self.query_one(f"#{widget_id}").display = not continuous
            except Exception:
                pass

    def _read_select(self, widget_id: str, default: str) -> str:
        try:
            value = self.query_one(f"#{widget_id}", Select).value
            if value is Select.BLANK or not value:
                return default
            return str(value)
        except Exception:
            return default

    def _read_text(self, widget_id: str) -> str:
        try:
            return (
                self.query_one(f"#{widget_id}", Input).value or ""
            ).strip()
        except Exception:
            return ""

    def _read_int(self, widget_id: str, default: int) -> int:
        raw = self._read_text(widget_id)
        if not raw:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _read_float(self, widget_id: str) -> float | None:
        raw = self._read_text(widget_id)
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def collect_spec(self) -> EvolutionLaunchSpec:
        """Build a frozen spec snapshot from the current
        widget values. Used by the submit handler + tests
        that want to assert the parse-and-coerce behaviour
        without driving the dismissal flow."""
        models_raw = self._read_text("launch-models")
        models = tuple(
            tok.strip() for tok in models_raw.split(",") if tok.strip()
        )
        # The editable Base-chain field is the source of truth (it's
        # pre-filled from the constructor for the Inspection / Library
        # flow). Fall back to the constructor value if the field can't be
        # read for any reason.
        base_chain_id = self._read_text("launch-chain") or self.base_chain_id
        spec = EvolutionLaunchSpec(
            base_chain_id=base_chain_id,
            dataset_path=self._read_text("launch-dataset"),
            rubric=self._read_text("launch-rubric"),
            validation_type=self._read_select(
                "launch-validation-type", DEFAULT_VALIDATION_TYPE,
            ),
            continuous_metric=self._read_select(
                "launch-continuous-metric", DEFAULT_CONTINUOUS_METRIC,
            ),
            binary_method=self._read_select(
                "launch-binary-method", DEFAULT_BINARY_METHOD,
            ),
            target_column=self._read_text("launch-target-column")
            or DEFAULT_TARGET_COLUMN,
            max_iterations=self._read_int("launch-max-iter", 10),
            population_size=self._read_int("launch-pop", 8),
            max_wall_time_seconds=self._read_float("launch-walltime"),
            model_whitelist=models,
            token_cost_cap=self._read_float("launch-cost"),
            mutation_max_tokens=self._read_int(
                "launch-mutation-max-tokens",
                self._default_mutation_max_tokens(),
            ),
        )
        self.last_spec = spec
        return spec

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        spec = self.collect_spec()
        if not spec.base_chain_id:
            # Can't launch without a chain to evolve — surface an inline
            # hint and keep the modal open so the user can fill it in.
            self._show_error(t("evolutionLaunch.errorNoChain"))
            try:
                self.query_one("#launch-chain", Input).focus()
            except Exception:
                pass
            return
        self._clear_error()
        # Post a message too so listeners observing the launch
        # gesture without owning the dismissal callback (e.g.
        # telemetry hooks) get a clean signal.
        self.post_message(LaunchRequested(spec))
        self.dismiss(spec)

    def _show_error(self, text: str) -> None:
        try:
            slot = self.query_one("#launch-error", Static)
        except Exception:
            return
        slot.update(text)
        slot.display = True

    def _clear_error(self) -> None:
        try:
            slot = self.query_one("#launch-error", Static)
        except Exception:
            return
        slot.update("")
        slot.display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "launch-cancel":
            self.action_cancel()
        elif bid == "launch-submit":
            self.action_submit()
        elif bid == "launch-dataset-browse":
            self._open_dataset_picker()

    def _open_dataset_picker(self) -> None:
        """Browse for the evaluation dataset file; on selection, write the
        chosen path into the dataset field. Mirrors the Export modal's
        *Browse…* affordance, filtered to JSONL files."""
        from pathlib import Path

        from care.screens.file_picker import FilePickerModal

        current = self._read_text("launch-dataset")
        start: Path | str
        if current:
            start = Path(current).expanduser()
        else:
            start = Path.cwd()

        def _on_pick(picked: Path | None) -> None:
            if picked is None:
                return
            try:
                self.query_one("#launch-dataset", Input).value = str(picked)
            except Exception:
                pass

        self.app.push_screen(
            FilePickerModal(start=start, extensions=(".jsonl",)),
            _on_pick,
        )


__all__ = [
    "EvolutionLaunchModal",
    "EvolutionLaunchSpec",
    "LaunchRequested",
]
