# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What CARE is

CARE — Collaborative Agent Reasoning Ecosystem — is a Textual TUI + headless `care` CLI built on top of a four-module stack:

- **MAGE** (`mmar-mage`, pinned to `../carl-mage` editable) — turns a query into a CARL chain.
- **CARL** (`mmar-carl`, optional `care[carl]` extra) — runs CARL chains; sandbox runtime; preflight.
- **GigaEvo Memory** (`gigaevo-client`, pinned to `../gigaevo-memory/client/python` editable) — entities (chain / agent / agent_skill / memory_card), library, SSE.
- **GigaEvo Platform** — GA over chains, accept-and-promote.

Sibling repos (`../carl-mage`, `../carl-experiments`, `../gigaevo-memory`, `../gigaevo-platform`) are real working trees — the `uv` editable sources in `pyproject.toml` only resolve when those checkouts exist next to this repo. CARE imports **every** upstream lazily (inside the function that needs it) so a minimal install still boots the CLI and a missing extra surfaces as a friendly install hint instead of an `ImportError` at startup.

## Commands

Day-to-day:

```bash
make run                       # launch the TUI (uv sync + uv run care)
make run LOG=1 LOG_LEVEL=DEBUG # also write logs/care-ui-* + logs/care-app-* sidecars
make test                      # uv run pytest
make lint                      # uv run ruff check .
uv sync --extra dev            # install dev deps (pytest, pytest-asyncio, ruff, respx) — needed before `pytest`
```

Tests don't run via `make test` until `uv sync --extra dev` has installed `pytest`. After that:

```bash
uv run --extra dev pytest tests/test_screen_chat.py -q                                  # one file
uv run --extra dev pytest tests/test_screen_chat.py::TestComposition -q                 # one class
uv run --extra dev pytest tests/test_screen_chat.py -k "history" -q                     # by keyword
uv run --extra dev pytest tests/test_screen_chat.py -x -q                               # fail fast
```

`tests/test_screen_chat.py` is ~18k lines and takes ~2½ minutes end-to-end. **Always run a focused slice (`-k` or `::TestClass`) before the full file**, then run the full file once at the end to catch cross-test regressions. The suite owns ~860 cases; nothing else in `tests/` comes close in size.

Sandbox runs of any `Bash` command sometimes can't reach pypi — `uv sync` will succeed only once. After it has, `uv run --extra dev pytest …` works offline.

## High-level architecture

```
care/
├── app.py                 CareApp Textual entry; global bindings (Ctrl+P/Ctrl+B/Ctrl+K/Ctrl+S/Ctrl+R/Ctrl+Q)
├── cli.py                 `care` headless CLI router + every subcommand
├── config.py              Pydantic CareConfig; nested `mage`, `memory`, `platform`, `sandbox`, `tools`, `telemetry`, `defaults`
├── memory.py              CareMemory facade over GigaEvoClient (stamps CareChainMetadata)
├── platform.py            CarePlatform facade for evolution
├── runtime/               Upstream callback → Textual Message adapters (one rule per file)
│   ├── mage_poster.py        MAGE progress → StageStarted/StageCompleted/StageProgress messages
│   ├── carl_streamer.py      CARL run callbacks → StepStarted/StepCompleted/Progress/ChainCompleted
│   ├── executor.py           build_run_context / execute_chain_async wrappers
│   ├── llm_client.py         build_llm_client / build_carl_llm_client (carl variant adds get_response_with_retries + optional token counter)
│   ├── status_bar.py         SessionTokenCounter + StatusBar refresh
│   ├── clipboard.py          copy_text — OSC-52 + pbcopy/xclip/wl-copy fallback (macOS Terminal needs the fallback)
│   ├── theme.py              Theme registry; the `$accent` brand colour lives in runtime/theme.py (`_LIGHT_VARS` / `_DARK_VARS`)
│   ├── cancellation.py       CancellationToken / CancellationGroup
│   └── … (~30 adapters total)
├── screens/               18+ Textual screens (LibraryScreen, GenerationScreen, ExecutionScreen, EvolutionScreen, …)
│   └── chat.py              PRIMARY user surface — ChatScreen owns ~75% of the user-facing behaviour
├── widgets/               Reusable Textual widgets (status_bar, header, footer, …)
│   └── chat_input.py        ChatInput(TextArea) — multi-line wrapping prompt with an Input-compat surface
├── sandbox/               AgentSkill sandbox backends (local / docker / e2b / firejail)
└── assets/                Packaged static assets (airi_logo_8/10/12/16.png + the un-suffixed default)
```

### ChatScreen is the centre of gravity

`care/screens/chat.py` is ~11k lines and is the **only** screen most users see. New features land there first. Key conventions inside it:

- **`_post_line(role, text, *, severity=None, chrome=False, extra_class=None)`** is the single chat-line mounter. `chrome=True` skips the `[HH:MM] role` caption (used for the boot banner, welcome lines, mode-flip hints). `extra_class` adds a per-call CSS hook (e.g. `chat-line-pre-answer` adds bottom padding before the assistant answer). `severity` mirrors `WARNING`/`ERROR` into `care.chat` logging.
- **Roles** (`ChatRole = Literal["user", "assistant", "system", "tool"]`) gate rendering: `assistant` + `system` mount as `Markdown` widgets, `user` + `tool` mount as `Static`. The Markdown widget carries an inherited `padding: 0 2` that gets stripped via `ChatScreen Markdown.chat-line { padding: 0; }`.
- **Stage trail**: MagePoster events post `▶ Friendly Label…` (tool line) → `_stage_started_indexes[stage]` records the widget index → `StageCompleted` mutes the matching line via `_STAGE_DONE_CSS_CLASS`. Sub-rows use `  ⎿ <text>`. The MAGE metadata summary lands as `⎿` tool rows under the final `✓ Describing steps`, not as an assistant line.
- **Modes + pipeline**: `ChatMode = Literal["interactive", "production"]` (the legacy `ad_hoc` value still deserializes — `normalise_mode()` maps it to `interactive`). Both modes run one pipeline `GENERATE → RUN → SAVE → BASELINE → EVOLVE`; each stage has a `StagePolicy` (`auto`/`ask`/`skip`) from the mode's `ModeSpec` in `MODE_SPECS`, overridable via `CARE_CHAT__MODE__<MODE>__<STAGE>` and merged by `resolve_mode_spec`. **Interactive** = `run=ask` (confirm before running — `_confirm_interactive_run`, toggle in /settings), save/baseline/evolve `skip` (save is the chain-action button, not a modal). **Production** = save/baseline/evolve `auto`, driven by `_drive_production_pipeline` (`_resolve_stage` per stage) — byte-identical to the legacy flow (parity-tested); no pre-save RUN gate (the baseline *is* the run). The transient `#chat-pipeline-strip` (above `#chat-mode-row`) renders the live `◆/○/◇/✗` status via `_render_pipeline_strip`; `_resolve_stage` emits a `chat.pipeline.stage` telemetry event per stage.
- **Interactive context**: `_interactive_history` keeps user/assistant turns across generations so follow-ups reference earlier ones (gated on `spec.followup == "reuse"`). `/new`, `/clear`, and mode flips reset it. Capped via `CARE_CHAT__AD_HOC_HISTORY_TURNS` (default 6) and `CARE_CHAT__AD_HOC_HISTORY_CHARS` (default 1200/turn) — env keys keep the legacy `AD_HOC` spelling.
- **Generation retry**: `_generate_with_retry` runs MAGE generation up to `CARE_CHAT__GENERATION_MAX_ATTEMPTS` (default 3) times with exponential backoff. Cancellation re-raises immediately.
- **`/revise` (NL chain edit)**: `_cmd_revise` → `_run_edit` worker drives MAGE's `MAGEGenerator.edit(...)` via `care.generation.run_edit` (`save=False`), previews the plan with `care.runtime.chain_edit_view.render_edit_plan_lines`, then `ConfirmModal` → saves a NEW VERSION via `app.memory.save_chain(entity_id=…)` (so CARE metadata stamping is preserved). Forms: `/revise <id> <change>` (first token that loads as a chain) or bare `/revise <change>` (MAGE resolves the chain, disambiguating on ties). The library's `R` row-action (`action_row_revise` → `app._revise_chain_for`) hands off here by seeding `ChatScreen.seed_input("/revise <id> ")`. Editing is **explicit** — plain prose is intentionally NOT auto-routed into edit mode.
- **Interactive answer synthesis**: when ≥2 successful steps exist, `_synthesise_user_answer` makes one extra LLM call to merge step outputs into a single coherent reply (bracketed by `▶/✓ Synthesising answer` tool lines). Production mode skips this — chains there must be reproducible.
- **`@<path>` file refs**: `_resolve_file_refs` greedily extends across whitespace while each candidate path keeps resolving on disk, so `@../My Notes.md` works without quoting. Also honors `@"…"` / `@'…'` explicit quoting, `.pdf` extraction via pypdf, **office/rich-text document extraction** (`.docx`/`.pptx`/`.xlsx`/`.odt`/`.rtf` → plain text via `care/runtime/document_extract.py`, routed through `_read_document_ref`; legacy `.doc`/`.ppt`/`.xls` surface a "re-save as …" hint), and image base64-embed for vision-capable models. Doc refs share the PDF two-cap pattern (`CARE_CHAT__DOC_REF_MAX_BYTES` 25 MB on-disk / `CARE_CHAT__DOC_TEXT_MAX_CHARS` 200 k extracted).
- **Test scaffold pattern**: tests use a minimal `_Host(App)` that does `push_screen(ChatScreen())` plus `_chat(app)` to extract the screen. Many tests then build their own `_MemHost` / `_ProdHost` variants that pre-populate `app.config` / `app.memory`. Use the existing patterns instead of inventing new ones — the suite's monkeypatch sites (`care.runtime.llm_client.build_llm_client`, `care.runtime.clipboard.copy_text`, MAGE/CARL stubs) are already wired.

### Configuration precedence (low → high)

1. Defaults in `care.config.CareConfig`
2. `~/.config/care/config.toml`
3. `./care.toml`
4. `CARE_*` env vars (nested via `__`: `CARE_MAGE__MODEL`, `CARE_CHAT__DEFAULT_MODE`, `CARE_CHAT__MODE__INTERACTIVE__RUN`, …)

See `.env.example` for the full surface; `care init [--non-interactive]` writes a starter `.env`. First boot without `~/.config/care/config.toml` lands on `SettingsScreen`; returning users go to ChatScreen.

### Themes / brand color

`$accent` (used by user-message tint, the `>` chat prompt, mode-toggle active state, every other screen's hot-element accent) is defined once in `care/runtime/theme.py`'s `_LIGHT_VARS["accent"]` and `_DARK_VARS["accent"]`. To re-brand globally, change those two hex strings — every screen references the design token rather than hard-coding.

### MAGE vs CARL clients

CARE talks to MAGE through the raw `openai.OpenAI` SDK (returned by `runtime/llm_client.build_llm_client`). CARL's step executors call `get_response_with_retries(prompt, retries)` which the raw SDK doesn't expose — use `build_carl_llm_client(config, token_counter=…)` instead. The token-counter variant intercepts `chat.completions.create` responses and folds `response.usage` into a `SessionTokenCounter` so the StatusBar shows real numbers (without it the chat reads `in 0 / out 0` for CARL runs).

## Testing conventions

- **Async by default**: `pyproject.toml` sets `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "function"`. New async tests get the `@pytest.mark.asyncio` decorator automatically applied.
- **`@pytest.fixture(autouse=True) _isolate_chat_state_files`** at the top of `tests/test_screen_chat.py` redirects every per-session sidecar (`CARE_CHAT__TUTORIAL_SIDECAR`, `CARE_CHAT__SESSION_LOG_DIR`, `CARE_CHAT__THEME_SIDECAR`, `CARE_CHAT__BRANCHES_DIR`) to `tmp_path` — extend this fixture (don't bypass it) when adding new on-disk state.
- **`Input` vs `ChatInput`**: tests query the chat prompt via `screen.query_one("#chat-input", ChatInput)`. The widget subclasses `TextArea` but exposes `.value` / `.cursor_position` / `action_submit` / posts `Input.Submitted` for back-compat — use those, not raw TextArea methods.
- **Verifying widget classes after `_post_line`**: in some test scaffolds the mount happens asynchronously and `query_one("#chat-line-N")` raises `NoMatches` before the next `pilot.pause()`. Either await another pause or monkeypatch `_post_line` directly to capture call args (`TestAdHocSynthesis::test_synthesis_done_line_tagged_for_spacing` is a worked example).

## Things to avoid

- **Don't re-instate `pillow` as a top-level dep** — `rich-pixels>=3.0` pins `pillow>=10.0.0` already; declaring it twice is redundant and was deliberately removed.
- **Don't bind `Ctrl+C` at the screen level** — the app-level `Binding("ctrl+c", "global_quit", …, priority=True)` in `care/app.py` owns it for the classic quit chord. The chat surface intentionally claims only `super+c` for copy-selection. A regression test (`TestClipboard::test_chat_does_not_claim_ctrl_c`) locks this contract.
- **Don't post the MAGE metadata summary as an `assistant` line** — it now rides as `⎿`-prefixed tool sub-rows under `✓ Describing steps` so it visually groups with the stage trail. Use `_format_result_summary_rows(result)` (returns `list[str]`) and emit each row as `self._post_line("tool", f"  ⎿ {row}")`. The legacy `_format_result_summary(result) → str` is kept only for the Production-save path.

## Logging

`make run LOG=1` writes two side-by-side sidecars per launch:

- `logs/care-ui-<timestamp>.log` — Textual UI events (compose, mount, dispatch, render) driven by `TEXTUAL_LOG`.
- `logs/care-app-<timestamp>.log` — Python app/client log (every `care.*` module + httpx Memory/Platform + MAGE/CARL workers) driven by `CARE_LOG_FILE` / `CARE_LOG_LEVEL`.

`_post_line` mirrors every chat entry into `care.chat` at INFO (`tool` rides DEBUG, severity-tagged lines ride WARNING/ERROR), so a session transcript survives in the app log without needing extra instrumentation.
