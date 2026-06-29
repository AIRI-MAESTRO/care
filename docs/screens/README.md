# Screens at a glance

Single-page reference for every Textual screen + modal CARE ships.
Use this as the navigation index when implementing a new feature
("which screen owns X?") or when writing a doc page ("where does
this fit in the user journey?").

Per-screen reference pages (canonical bindings, compose tree, test
file, design constraints) live under `docs/screens/<screen>.md` once
each lands — filed as the §8 P1 follow-up. This README stays the
top-level "what exists, what's its slash command, what does it do".

> This file is **auto-generated** by
> `scripts/generate_screens_index.py`. Edit
> `_SCREEN_METADATA` in that script (or the table headings /
> intro paragraphs in `_HEADER` / `_FOOTER`) and re-run
> `python scripts/generate_screens_index.py --write` to update.
> A regression test (`tests/test_screens_index_doc.py
> ::TestAutoGenerator`) enforces lockstep so the doc can't
> drift from the live `care.screens.__all__` /
> `_COMMAND_BLURBS` registries.

## Four canonical screens

The chat-centric refactor (Phases 1–6) collapsed the original
`Query → Generation → Inspection` flow into four user-visible
screens. ChatScreen is the primary entry; the other three are
reached on demand via slash commands + the Production-mode action
toolbar.

| Screen | Slash | Status | Primary purpose |
| -------- | ------- | ------ | ----------------- |
| `ChatScreen` | (boot) | M0 | Natural-language input, mode toggle, slash palette, artifact pill, Production action toolbar |
| `ArtifactsScreen` | `/artifacts` | M0 | Current-chat artifacts (chain / stage / tool / dataset / synth output); save, copy, drop, inspect |
| `LibraryScreen` | `/library` | M0 | Saved chains — sort, filter, tag-pool, recency strip, mean cost, bulk import / export |
| `EvolutionScreen` | `/evolution` | M0 | Run + watch a GA over a chain; Pareto front, fitness curve, scatter plot, cost meter, accept |

## Supporting screens (reached from the four above)

| Screen | Trigger | Status | Primary purpose |
| -------- | --------- | ------ | ----------------- |
| `CatalogScreen` | (Ctrl+K) | M0 | Browse installed capabilities (skills / MCP / tools / cards) |
| `CostDashboardScreen` | `/cost` | M0 | Token + USD spend rollup by provider / chain / session |
| `DemoScreen` | (boot fallback) | M0 | First-run / config-error fallback so users see something |
| `EditAgentScreen` | (Library `e`) | M0 | Inline edit + save-as-new-version + promote-to-stable |
| `EvolutionDashboard` | `/evolution` | M0 | List of active + recent evolution runs; Enter opens EvolutionScreen, `c` compares two |
| `ExecutionScreen` | (Library `r`) | M0 | Live CARL run + token streaming |
| `GenerationScreen` | (legacy) | M0 | Pre-chat live-MAGE-progress surface — superseded by ChatScreen's inline progress lines |
| `HelpScreen` | `/help` | M0 | Tutorial + every binding (filtered by active screen) |
| `InspectionScreen` | (Library `Enter`) | M0 | Saved-chain detail + run history + Integration pane |
| `LogsScreen` | `/logs` | M0 | Tail the rolling app log; `m` toggles a module filter |
| `MarketplaceScreen` | `/marketplace` | M0 | Search shared agent_skill listings on Memory |
| `OnboardingScreen` | (boot, planned) | M1 | uvx first-run wizard — §1 P0 still in flight |
| `ProfileScreen` | `/profile` | M0 | List credential profiles under `~/.config/care/profiles/` |
| `QueryScreen` | (legacy) | M0 | Pre-chat "+ New agent" task form — still reachable, no longer canonical |
| `ReplayScreen` | (Runs `Enter`) | M0 | Step through a saved ReasoningResult |
| `RunsScreen` | `/runs` | M0 | Local run history (`~/.cache/care/runs/`); Enter opens ReplayScreen sidecar |
| `SandboxTrustScreen` | `/sandbox` | M0 | Audit + revoke trusted AgentSkills (SHA-pinned trust store) |
| `SettingsScreen` | `/settings` | M0 | Edit MAGE / Memory / Platform creds + theme + advanced knobs |
| `TaskListDrawer` | (Ctrl+B) | M0 | In-flight workers panel |
| `WelcomeScreen` | (boot) | M0 | Boot splash; routes to ChatScreen (returning users) or SettingsScreen (first-run / missing creds) |

## Modal screens (overlays)

Modals layer on top of screens but never own the primary navigation
target. Most are pushed via row actions, save flows, or the dismiss
of a screen-level interaction.

| Modal | Triggered from | Status | Primary purpose |
| ------- | ---------------- | ------ | ----------------- |
| `CommandPaletteModal` | `Ctrl+P` (any screen) | M0 | Fuzzy palette over commands + saved entities |
| `ConfirmModal` | Destructive actions | M0 | OK / Cancel confirm for destructive actions (bulk delete, accept-winner) |
| `ConflictModal` | Save-with-name-collision | M0 | Resolve a name collision on save |
| `DiffModal` | Library `D`, EvolutionScreen `D` | M0 | Side-by-side compare two chains / individual vs. parent |
| `EvolutionCompareModal` | Dashboard `c` after multi-select | M0 | Side-by-side fitness curves for two evolution runs |
| `EvolutionLaunchModal` | Library `v` / `E` | M0 | Budget / rubric / objectives picker before EvolutionScreen launches |
| `ExportChainModal` | Evolution `x` | M0 | Export a single chain payload to disk (JSON / Python) |
| `ExportModal` | Library `x` | M0 | Export saved-Memory entities into a tarball |
| `HumanInputModal` | CARL human-input step | M0 | Block CARL execution for a human-supplied answer |
| `ImportModal` | Library `i` | M0 | Import a chain bundle (tar.gz) |
| `LineageModal` | Library `l` | M0 | Walk a chain's ancestry DAG |
| `ResumeModal` | `/resume` | M0 | Rehydrate a Production-mode transcript |
| `RunContextModal` | Library / Execution `r` | M0 | Re-run form: task + context-file picker + tags |
| `SaveAgentModal` | Post-generation | M0 | Tag + name a freshly-generated chain before persistence |
| `SaveReport` | After save-all batch | M0 | Post-mortem table of save-all outcomes |
| `TagEditorModal` | Library `T`, Artifacts `s` | M0 | Edit tags (bulk) + optional editable title (§3 P3 save-flow path) |
| `UseItNowModal` | Post-save + accept-winner | M0 | Copy-paste recipe (python / curl / cli) for the saved chain |

## Status legend

- **M0** — Shipped + tested in the v0.1 release path. Owns its
  documented bindings + flows.
- **M1** — In flight or behind a tracked TODO blocker.
- **M2** — Filed but not started; check `TODO.md` for the priority.

## Where to look next

- Per-screen reference: `docs/screens/<screen>.md` (filed as §8 P1
  follow-up — pending).
- Slash command reference: every command in the table above is
  registered in `_COMMAND_HANDLERS` inside
  [`care/screens/chat.py`](../../care/screens/chat.py) with a
  one-line blurb in `_COMMAND_BLURBS`. The `/help` modal in-app
  reads the same registries so it stays in lockstep with this
  doc.
- Architecture overview: [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
  — the four-screen map in §1 is the canonical mental model for
  how the screens compose.
- Full work plan: [`TODO.md`](../../TODO.md) — every screen has
  shipping notes and follow-up tasks.
