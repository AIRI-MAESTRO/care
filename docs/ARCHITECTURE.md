# CARE — Architecture

CARE (Collaborative Agent Reasoning Ecosystem) is the **consumer** at the
top of a four-module stack. It owns the user-facing Textual TUI + the
headless `care` CLI; everything else — chain generation, persistence,
execution, evolution — lives in dedicated upstream modules that CARE
drives through narrow facades.

This document mirrors the high-level diagram in [`TODO.md §0`](../TODO.md)
and walks each layer with concrete entry points so a reader can locate
the right module, function, and upstream contract for any change.

---

## 1. The four-module stack

```text
┌─────────────────────────────────────── CARE TUI / CLI ───────────────────────────────────────┐
│                                                                                              │
│            ┌─────────────────────────────  ChatScreen  ──────────────────────────────┐       │
│            │  Primary user surface — natural-language input, mode toggle             │       │
│            │  (Ad-Hoc | Production), slash command palette, inline answers,         │       │
│            │  artifact-pill header counter, Production-mode action toolbar.         │       │
│            └────────┬───────────────────────┬───────────────────────┬─────────────────┘       │
│                     │                       │                       │                        │
│            ┌────────▼──────────┐  ┌─────────▼──────────┐  ┌─────────▼──────────┐             │
│            │  ArtifactsScreen  │  │   LibraryScreen    │  │  EvolutionScreen   │             │
│            │  (current chat —  │  │  (saved chains —   │  │  (Pareto front +   │             │
│            │   chain/stage/    │  │   sort, filter,    │  │   live SSE +       │             │
│            │   tool/dataset    │  │   tag-pool,        │  │   compare runs +   │             │
│            │   /synth output)  │  │   recency + cost)  │  │   export / accept) │             │
│            └────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘             │
│                     │                       │                       │                        │
│                     ▼                       ▼                       ▼                        │
│            ┌─────────────────┐    ┌──────────────────┐     ┌──────────────────┐              │
│            │   MAGE async    │    │   CARL runner    │     │  GigaEvo Platform│              │
│            │   pipeline      │    │  (execute chain  │     │  (evolution: GA, │              │
│            │   + progress    │    │   + sandbox)     │     │   accept, SSE)   │              │
│            └────────┬────────┘    └─────────┬────────┘     └─────────┬────────┘              │
│                     │                       │                        │                       │
│                     ▼                       ▼                        ▼                       │
│            ┌─────────────────────────────────────────────────────────────────┐               │
│            │   GigaEvoClient (gigaevo-client SDK — Memory + Platform)        │               │
│            └────────────────────────┬──────────────────────────┬─────────────┘               │
│                                     │                          │                             │
│                                     ▼                          ▼                             │
│                            GigaEvo Memory               GigaEvo Platform                     │
│                            (chain / agent /             (runner pool, evolution              │
│                             agent_skill / card)         experiments, SSE progress)           │
│                                                                                              │
│   AgentSkill sandbox  ◄─── triggered from CARL AgentSkillStep                                │
│   (Docker / e2b / firejail / local)                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

The four canonical screens — **Chat → Artifacts | Library | Evolution** —
own ~95% of the user-facing surface. ChatScreen is the primary entry
(natural-language input, mode toggle, slash palette); the other three
are reached via `/artifacts`, `/library`, `/evolution` slash commands
+ the Production-mode action toolbar that auto-surfaces after the
first chain generation. Modal flows (TagEditor, SaveReport,
UseItNowModal, EvolutionLaunchModal, ExportChainModal, etc.) layer on
top of these screens but are never the primary navigation target.

The original delivery plan (`Query → Generation → Inspection` per
Phase 0) routed everything through dedicated screens; the
chat-centric refactor (Phases 1–6) collapsed that into ChatScreen
with the other surfaces reachable on demand. The legacy QueryScreen /
GenerationScreen / InspectionScreen modules still exist for the
non-chat boot paths + drill-down flows but are no longer the
canonical user journey.

| Layer           | Module                                                          | Role                                                                 |
| --------------- | --------------------------------------------------------------- | -------------------------------------------------------------------- |
| **TUI / CLI**   | `care/` (this repo)                                             | Textual app + `care` CLI + lazy upstream facades                     |
| **Generation**  | [`carl-mage`](../../carl-mage) (PyPI: `mmar-mage`)              | MAGE — turns a query into a CARL chain                               |
| **Execution**   | [`carl-experiments`](../../carl-experiments) (`mmar-carl`)      | CARL — runs CARL chains; sandbox runtime; preflight introspection    |
| **Persistence** | [`gigaevo-memory`](../../gigaevo-memory) (`gigaevo-client` SDK) | Entities (chain / agent / agent_skill / memory_card) + library + SSE |
| **Evolution**   | [`gigaevo-platform`](../../gigaevo-platform)                    | GA over chains, individuals, accept-and-promote                      |

CARE never imports the generation / execution / evolution modules at the
top level — every import is **lazy** (inside the function that needs it)
so a slim `care` install still works for the bounded CLI subcommands
and a missing optional dep surfaces as a friendly install hint rather
than an `ImportError` at startup.

The full preparation status of upstream items is in
[`PREPARE.md`](../PREPARE.md); CARE's own work plan is in
[`TODO.md`](../TODO.md).

---

## 2. Module boundaries inside `care/`

```text
care/
├── app.py              # Textual entry point (TUI mount)
├── cli.py              # Headless `care` CLI router + subcommands
├── config.py           # Pydantic CareConfig + every nested section
├── memory.py           # CareMemory facade over GigaEvoClient
├── platform.py         # CarePlatform facade for evolution
├── catalog.py          # CapabilityCatalog discovery (skills / MCP / tools / cards)
├── skills.py           # promote_skill_to_memory — local SKILL.md → entity
├── tools.py            # load_tools_into_context — @carl_tool registry loader
├── capability_priming.py  # Catalog → MAGE CapabilityContext bridge
├── preflight.py        # validate_chain — parse + preflight a chain
├── chain_export.py     # export_chain — .json / .py (via MAGE CodeGenerator)
├── bulk_import.py      # import_chains — batch validate / save
├── runtime/            # Adapters bridging upstream callbacks → Textual messages
│   ├── mage_poster.py        # MAGE progress → MagePoster messages
│   ├── carl_streamer.py      # CARL run callbacks → CarlStreamer messages
│   ├── executor.py           # build_run_context / execute_chain_async
│   ├── library_watcher.py    # SDK watch_entities → typed events
│   ├── llm_client.py         # build_llm_client — OpenAI client from base_url + api_key
│   ├── pricing.py            # estimate_cost — USD estimate from in-tree LLM pricing table
│   ├── task_registry.py      # In-session multi-task tracking
│   ├── run_state.py          # Atomic ~/.local/state/care/run_state.json
│   ├── draft.py              # Auto-save / promote / discard chain drafts
│   ├── cancellation.py       # CancellationToken + CancellationGroup
│   ├── provenance.py         # SkillProvenanceRecorder — SHA-pinned saves
│   ├── run_recorder.py       # record_run_completion → memory_card
│   ├── skill_runtime_adapter.py  # CareSkillRuntime → CARL register_with_carl
│   ├── human_input.py        # HumanInputBroker — pending-question queue + resolver
│   ├── telemetry.py          # Opt-in Langfuse event sink + registry
│   ├── status_bar.py         # SessionTokenCounter + aggregate_status_bar snapshot
│   ├── clipboard.py          # copy_text — OSC-52 + pbcopy / xclip fallback
│   ├── lineage.py            # LineageGraph + fetch_chain_lineage DAG projection
│   ├── run_context_draft.py  # RunContextDraft form model for the re-run modal
│   ├── run_history.py        # RunHistoryEntry + fetch_run_history projection
│   ├── bulk_ops.py           # BulkSelection + bulk favourite/tag/delete drivers
│   ├── edit_draft.py         # EditAgentDraft + save_edit_as_new_version + promote_to_stable
│   ├── library_run.py        # LibraryRunPlan + load_run_plan + execute_library_run
│   ├── save_agent_form.py    # SaveAgentForm + seed/validate/apply for the post-generation modal
│   ├── library_view.py       # LibraryRow + fetch_library_view + LibraryViewState persistence
│   ├── agent_diff.py         # AgentDiff + diff_chains + fetch_agent_diff side-by-side comparison
│   ├── collections.py        # Collection + list_collections + filter_by_collection sidebar projection
│   ├── library_bundle.py     # BundleManifest + export_library_bundle + import_library_bundle tar.gz
│   ├── row_actions.py        # RowAction registry + single-row mutators (favourite / delete / duplicate)
│   ├── empty_state.py        # EmptyState classifier (no_library / no_results / loading / error)
│   ├── command_palette.py    # PaletteIndex + fuzzy_score + search_palette (Ctrl+P navigation)
│   ├── global_bindings.py    # GlobalBinding registry + Header/Footer projection
│   ├── theme.py              # Theme registry + ThemePreferenceStore + auto/dark/light resolver
│   ├── chain_title.py        # suggest_chain_title — LLM-suggested one-line chain name (§3 P2)
│   ├── cost_rollups.py       # Aggregate token / USD spend per chain / per session (§6 P2)
│   ├── doctor.py             # care doctor — environment + dep health report (§7 P2)
│   ├── fitness_plot.py       # render_fitness_plot — plotext / sparkline backends (§5 P0)
│   ├── pareto_plot.py        # render_pareto_scatter — 2D objective × objective (§5 P1)
│   ├── keystore.py           # OS-keychain backed storage for API keys (§1 P1)
│   ├── local_run_history.py  # ~/.cache/care/runs/<date>.jsonl + ChainRunStats aggregator (§4 P2)
│   ├── log_discovery.py      # /logs file-walking + recency / module / level surfacing (§8 P2)
│   ├── profiles.py           # ~/.config/care/profiles/ — named CareConfig snapshots (§6 P2)
│   ├── session_artifacts.py  # SessionArtifactStore — current-chat artifact entries (§3 P0)
│   ├── session_persistence.py  # Atomic JSONL writeback for the artifact store (§3 P1)
│   ├── user_paths.py         # XDG-style CARE_{CONFIG,CACHE,STATE}_DIR + ensure_user_dirs (§1 P0)
│   ├── i18n.py               # t() translation helper + key-based ru/en JSON locale catalogs
│   ├── dag_view.py           # render_dag_boxes — inline box-and-arrow DAG graph for the chat trail
│   ├── programs_chart.py     # render program valid/invalid counts for the evolution Programs tab
│   ├── deck_builder.py       # multi-pane evolution visualization deck assembly
│   ├── chain_edit_view.py    # render_edit_plan_lines — /revise NL chain-edit plan preview
│   ├── agent_hub.py          # hub agent client — /deploy / /deployments / /metrics adapters
│   ├── deploy_gate.py        # deploy readiness gate (channel + preflight checks)
│   ├── promote_gate.py       # promote gate — latest→stable release validation
│   ├── document_extract.py   # office / rich-text → plain text extraction for @-file refs
│   ├── file_loading.py       # canonical "file → chain-ready content" (office/pdf/text/image, capped, binary-safe) used by every attach surface
│   ├── fitness_export.py     # serialise an evolution fitness curve → CSV / JSON (§5)
│   ├── evolution_redis_probe.py    # live gigavolve Redis probes (generation / fitness / program counts) for local stacks
│   ├── evolution_validation.py     # Platform-aligned validation / metric options for chain evolution
│   ├── evolution_chain_templates.py  # sync Platform chain-experiment templates into a live runner problem dir
│   ├── platform_bootstrap.py # one-shot local Platform stack bootstrap on CarePlatform construction
│   ├── platform_llm_sync.py  # push MAESTRO LLM credentials into gigaevo-platform llm_models.yml
│   ├── runner_tools_sync.py  # copy gigaevo-core metric tools into the live runner Docker clone
│   ├── hint_fit.py           # fit hint lines to terminal width — drop / truncate segments
│   └── open_url.py           # cross-platform URL opener (browser launch)
└── sandbox/            # AgentSkill sandbox backends
    ├── backend.py            # SandboxBackend Protocol + RunResult / NetworkPolicy
    ├── local.py              # LocalSandboxBackend (asyncio subprocess)
    ├── network_policy.py     # parse_webfetch_domains + resolve_network_policy
    ├── trust.py              # SkillTrustStore (SHA-pinned trust JSON)
    ├── audit.py              # SandboxAuditLogger (JSON-lines append)
    ├── resources.py          # parse_resources_block + ResourcePolicy
    └── output_mediation.py   # scan_output_dir — 7 heuristic checks
```

### Why two facades (`care.memory`, `care.platform`)

The `gigaevo-client` SDK exposes a generic `GigaEvoClient`. CARE wraps
it because:

1. CARE's call sites pass **task-shape data** (a user query, a list of
   context files, MAGE metadata) — the facade builds the
   `CareChainMetadata` shape the convention requires (see
   [Memory docs §1.4](../../gigaevo-memory/docs/CHAIN_CONTENT_CONVENTIONS.md)).
2. CARE-side concerns like stamping `domain:{value}` tags or returning
   bare `entity_id` strings instead of the SDK's `EntityRef` belong in
   one place, not duplicated across every screen.

The SDK is the source of truth; `CareMemory` / `CarePlatform` translate
its surface into the kwargs CARE actually has on hand.

### Why a separate `care.runtime` subpackage

Every adapter in `care/runtime/` follows one rule: **translate an
upstream callback or stream into Textual `Message` instances that the
running app posts**. This isolates the contract: screens consume
Textual messages, never raw `MAGEProgressCallback` calls. When MAGE's
callback shape changes, only the adapter changes.

---

## 3. Layer-by-layer reference

### 3.1 Generation — MAGE

| What you need                               | Where                                                                                                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Generator class                             | [`mmar_mage.MAGEGenerator`](../../carl-mage/mmar_mage/generator.py)                                                                                          |
| Progress callback contract                  | [`mmar_mage.MAGEProgressCallback`](../../carl-mage/mmar_mage/protocols.py) — adapted by `care.runtime.MagePoster`                                            |
| Capability context shape                    | [`mmar_mage.CapabilityContext`](../../carl-mage/mmar_mage/agents/capability_lookup_agent.py) — built by `care.build_capability_payload(...)` (§4 P2 priming) |
| Code-generation export                      | `mmar_mage.code_generator.CodeGenerator.generate(chain_dict, query, MAGEConfig)` — wrapped by `care.export_chain(format="python")` (§9 P3)                   |
| Replay artifact                             | `MAGEResult.to_care_dict()` — CARE saves this as one `memory_card` per run                                                                                   |
| Per-stage entrypoints (regenerate DAG only) | `MAGEGenerator.analyze_domain` / `plan_steps` / `build_dag` / `describe_steps` / `critique_steps` / `verify_chain` / `refine`                                |

Upstream preparation: [`PREPARE.md §3`](../PREPARE.md). MAGE-side work
plan: [`carl-mage/TODO.md`](../../carl-mage/TODO.md).

### 3.2 Execution — CARL

| What you need                            | Where                                                                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Chain class                              | [`mmar_carl.ReasoningChain`](../../carl-experiments/src/mmar_carl/chain.py)                                                           |
| Context — re-priming from saved metadata | `ReasoningContext.from_chain_inputs(chain, api=...)` — wrapped by `care.runtime.executor.build_run_context`                           |
| Preflight introspection                  | `ReasoningChain.preflight(context) → PreflightReport` — wrapped by `care.validate_chain(...)` (§4 P2)                                 |
| Tool registry                            | `ReasoningContext.register_tools_from_path(...)` — wrapped by `care.load_tools_into_context(...)` (§5 P1)                             |
| Sandbox protocol                         | [`mmar_carl.SkillRuntime`](../../carl-experiments/src/mmar_carl/) — bridged via `care.runtime.skill_runtime_adapter.CareSkillRuntime` |
| Cancellation                             | `ReasoningContext.cancel()` / `is_cancelled()` + `care.runtime.CancellationToken`                                                     |
| Result serialisation                     | `ReasoningResult.{to,from}_{dict,json}` + `RunRecord` Pydantic wrapper                                                                |
| Sub-step events                          | `on_step_event(step_number, event_type, payload)` callback — adapted by `care.runtime.CarlStreamer`                                   |

Upstream preparation: [`PREPARE.md §5`](../PREPARE.md). CARL's
CARE-facing inventory: [`TODO_CARE.md`](../../carl-experiments/TODO_CARE.md).

### 3.3 Persistence — GigaEvo Memory + `gigaevo-client` SDK

| What you need                     | Where                                                                                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| SDK client                        | [`gigaevo_client.GigaEvoClient`](../../gigaevo-memory/client/python/src/gigaevo_client) — held by `CareMemory(client)`                                |
| Save a chain (with CARE metadata) | `CareMemory.save_chain(chain, *, name, query, domain, context_files, …) → entity_id`                                                                  |
| Save an AgentSkill (with SHA pin) | `CareMemory.save_agent_skill(...)` — promotion wrapper: `care.promote_skill_to_memory(skill_path, memory, …)`                                         |
| Library mutators                  | `mark_favourite`, `record_run`, `update_metadata`, `list_chains`/`list_agents`/`list_agent_skills` (sort / favourites / tags / q / cursor)            |
| Capability matching               | `client.find_capability_matches(rough_aim, top_k, deep=True)`                                                                                         |
| Server-wide event firehose        | `client.watch_events(filter=...)` — typed via `care.runtime.LibrarySubscription`                                                                      |
| Chain content convention          | `CareChainMetadata` + `ContextFileRef` — spec in [Memory `docs/CHAIN_CONTENT_CONVENTIONS.md`](../../gigaevo-memory/docs/CHAIN_CONTENT_CONVENTIONS.md) |
| Evolution lineage                 | `client.get_chain_lineage(entity_id) → LineageResponse` + typed `EvolutionMeta` on each entity                                                        |

Upstream preparation: [`PREPARE.md §1` and §2](../PREPARE.md). Memory
server: [`gigaevo-memory/TODO.md`](../../gigaevo-memory/TODO.md).

### 3.4 Evolution — GigaEvo Platform

| What you need                      | Where                                                                                                                                               |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Platform client                    | `gigaevo_client.PlatformClient` — wrapped by `care.platform.CarePlatform`                                                                           |
| Start an evolution                 | `CarePlatform.start_evolution(...) → EvolutionRef`                                                                                                  |
| Stream evolution events            | `client.stream_events(evolution_id)` — `generation_started`, `individual_evaluated`, `best_updated`, `completed`, `accepted`, `failed`, `cancelled` |
| List individuals / Pareto front    | `client.list_individuals(evolution_id, ...)`                                                                                                        |
| Accept an individual               | `client.accept_individual(evolution_id, individual_id)` — promotes to Memory's `stable` channel                                                     |
| Runner sandbox (`RUN_AGENT_SKILL`) | Platform's `runner_api` with Docker `--network none` default + HTTP-CONNECT egress proxy for `skill_declared` mode                                  |

Upstream preparation: [`PREPARE.md §4`](../PREPARE.md). Platform
work plan: [`gigaevo-platform/TODO.md`](../../gigaevo-platform/TODO.md).

### 3.5 Sandbox

CARE ships its own host-side sandbox stack in `care/sandbox/` so chains
can execute locally during development. The Platform's runner sandbox
(§4) is the production path for `RUN_AGENT_SKILL` tasks; the CARE-side
backends cover local-only dev.

| Backend                       | Implementation                                 | Use case                              |
| ----------------------------- | ---------------------------------------------- | ------------------------------------- |
| `LocalSandboxBackend`         | `care/sandbox/local.py` — `asyncio.subprocess` | CARE-internal testing; unsafe in prod |
| `DockerSandboxBackend` (P0)   | not yet shipped                                | Default dev backend                   |
| `E2BSandboxBackend` (P1)      | not yet shipped                                | Cloud-hosted sandbox                  |
| `FirejailSandboxBackend` (P2) | not yet shipped                                | Lightweight Linux-only                |

Shared shape:

- **Protocol**: `SandboxBackend` in `care/sandbox/backend.py`.
- **Network**: `NetworkPolicy = Literal["none", "skill_declared", "open"]`,
  resolved via `resolve_network_policy(skill_md, policy) →
ResolvedNetworkPolicy`. The `skill_declared` mode parses
  `WebFetch(domain:*)` tokens from the SKILL.md manifest.
- **Trust**: SHA-256 pinning via `SkillTrustStore`. Mismatch raises
  `TrustRefusedError` before any subprocess starts.
- **Audit**: append-only JSON-lines via `SandboxAuditLogger`.
- **Resources**: `parse_resources_block` + `ResourcePolicy` with an
  `allow_manifest_upscale` flag to gate skill-requested ratchets.
- **Output mediation**: `scan_output_dir → MediationReport` runs seven
  heuristic checks (executable bit, binary magic, shebang, network
  tokens, symlink escape, large file, empty out).

---

## 4. The canonical user flow

> The legacy `Query → Generation → Save` flow below is preserved
> verbatim as the **integration-test contract** for the whole stack —
> it pins every adapter / facade / SDK call site through a single
> readable sequence. The day-to-day user journey runs through
> ChatScreen instead (see §5 below); the bullets here are the
> long-lived plumbing the chat-centric refactor still relies on.

The TODO's `TODO.md §1.3` canonical user flow is the integration test
for the whole stack:

1. **Generate Agent A** — `QueryScreen` collects the task; `GenerationScreen`
   posts `MagePoster` messages while MAGE drives the pipeline; on
   completion `SaveAgentModal` writes the chain via
   `CareMemory.save_chain(...)` with `CareChainMetadata` populated.
2. **Save** — entity lands in Memory's `latest` channel under the user's
   namespace. The library refreshes via the server-wide `/v1/events`
   SSE firehose (other CARE sessions see it too).
3. **Generate B, C** — same flow, no state coupling between agents.
4. **Return to A from library** — `LibraryScreen` lists every saved
   agent; sort/filter/favourite supported. `Run` opens `ExecutionScreen`;
   `Edit` opens `EditAgentScreen` (saves as a new version of the same
   `entity_id`); `Evolve` opens `EvolutionScreen`.
5. **Re-run A** — `prime_from_saved_chain(entity_id)` reads
   `content.metadata.care` and rebuilds the `ReasoningContext` deterministically.

Every step in this flow is backed by a shipped CARE primitive — see the
shipping notes against each `[DONE]` bullet in [`TODO.md`](../TODO.md).

---

## 5. The chat-surface dual contract

The Textual TUI's primary surface — landed in Phases 1–6 of the
[delivery plan](../TODO.md) — is `care/screens/chat.py` (`ChatScreen`).
It supersedes the screen-based `Query → Generation → Save` flow above
for everyday work: users describe a task in natural language, CARE
picks the right downstream call sequence based on the current **mode**,
and prints the result inline. The legacy screens remain reachable via
slash commands (`/library`, `/run`, `/settings`).

The mode is session-scoped, defaults to **Ad-Hoc**, and is driven by
three equivalent inputs: the toggle above the prompt, the `/mode`
slash command, and `CARE_CHAT__DEFAULT_MODE` at boot. `ChatScreen.mode`
is a Textual `reactive[ChatMode]`; `watch_mode` is the single side-
effect site (header badge sync, system hint line, production-gate
fallback). Every chat line carries the mode active at post-time
(`ChatLine.mode`), so flipping modes later never rewrites earlier
audit lines.

### 5.1 Why two modes

The mode toggle is the answer to "what should happen when the user
sends a prompt — does CARE need to optimise for fast feedback, or for
a reproducible artefact?" Cramming both onto one workflow forces
either (a) every prompt to write to Memory + run evolution (slow,
clutters Memory) or (b) every prompt to evaporate after the answer
prints (no audit trail, no quality measurement). The decision was
locked under [Phase 0 §1](../TODO.md#1--mage-follow-up-contract-option-b):
two explicit modes, each owning a small contract.

| Mode | Persistence | ReAct loop | Dataset card | Evolution |
| ---- | ----------- | ---------- | ------------ | --------- |
| **Ad-Hoc** | none | yes (up to `CARE_CHAT__LOOP_MAX_ITER`, default 5) | no | no |
| **Production** | `CareMemory.save_chain` | no — one terminal run | seeded baseline + `/dataset add/run/export` | `CarePlatform.start_evolution` when wired |

### 5.2 Ad-Hoc data flow

```text
user prompt (chat input)
       │
       ▼
ChatScreen.on_input_submitted ──► _handle_task
       │
       ▼
ChatScreen._run_generation   (worker group "generate")
       │
       │  Loop driver — `_loop_max_iter()` cap; per-iteration
       │  timing + token-delta footer (Phase 2 P2).
       │
       ├──► MAGE generate (build_mage_generator + run_generation)
       │        │  MagePoster bridges stage callbacks → Textual
       │        │  StageStarted / StageProgress / StageCompleted /
       │        │  StageError / StageRetry messages.
       │        ▼
       │    MAGEResult.chain_dict
       │        │
       │        ▼
       │    summarise_mage_result → "assistant" line
       │
       ├──► CARL execute (_execute_chain_ad_hoc)
       │        │  CarlStreamer bridges step callbacks → Textual
       │        │  StepStarted / Progress / StepCompleted /
       │        │  ChainCompleted messages.
       │        ▼
       │    ReasoningResult
       │        │
       │        ▼
       │    summarise_carl_result → "assistant" line
       │
       └──► _extract_continuation(answer)
                │
                ├─ "[CONTINUE]" / "[CONTINUE: <next>]"
                │      → _build_followup_task → next iteration
                │
                └─ no marker → loop terminates
```

Persistence is intentionally absent — every Ad-Hoc run is throwaway.
Failures at any layer surface as `severity`-tagged `system` lines and
log at WARNING/ERROR (`care.chat` logger); the loop's `finally`
resets `_generating` so the next prompt is accepted immediately.

### 5.3 Production data flow

```text
user prompt (chat input)
       │
       ▼
ChatScreen._handle_task ──► ChatScreen._run_generation
       │
       ├──► MAGE generate ──► MAGEResult.chain_dict
       │
       ├──► dedup gate (Phase 3 P2)
       │        task_hash = sha256(task)[:12]
       │        memory.search_hits(tag=f"task-hash:{hash}")
       │           │
       │           ├─ hit → surface existing chain_id + "/run" hint
       │           │        Escape via "[FORCE]" prefix on the task.
       │           │
       │           └─ miss → continue
       │
       ├──► CareMemory.save_chain
       │        tags = [
       │          "source:chat-prod",
       │          f"task-hash:{hash}",
       │          f"mage:{mode}",
       │          f"mage-model:{slug}",
       │        ]
       │        → chain_id, display_name
       │        → "✓ Saved chain `<id>`" assistant line
       │
       ├──► _record_production_baseline (Phase 4 P0)
       │        ├─ _execute_chain_ad_hoc (CARL run, same helper as Ad-Hoc)
       │        │      → ReasoningResult (the baseline answer)
       │        ├─ record_run_completion → memory_card
       │        │      tags = [
       │        │        f"dataset-entry:{chain_id}",
       │        │        "agent_run",
       │        │        f"agent:{chain_id}",
       │        │        f"status:{pass|fail}",
       │        │      ]
       │        └─ "✓ Baseline recorded as `<card_id>`" line
       │
       └──► _kickoff_evolution (Phase 5 P0)
                ├─ app.platform is None → INFO log + skip line
                └─ CarePlatform.start_evolution(
                       base_chain_id=chain_id,
                       tags=["source:chat-prod", f"chain:{chain_id}"],
                   )
                   → EvolutionRef
                   → "🧬 Evolution run `<evo_id>` started" line
```

The ReAct loop is disabled in Production — each task produces exactly
one reproducible chain. Subsequent quality work happens out-of-band:
`/dataset list / add / run / export` grow + score the dataset card
chain; `/evolution <run_id> / watch <run_id>` follow the evolution
run. Production is gated on `app.memory is not None` —
`ChatScreen.watch_mode` auto-falls back to Ad-Hoc with a warning when
Memory isn't wired (the toggle's tooltip documents this behaviour).

### 5.4 Shared adapters & seams

The two modes share every adapter so behavioural drift is impossible:

| Concern | Adapter | Used by |
| ------- | ------- | ------- |
| MAGE stage callbacks → Textual events | `care/runtime/mage_poster.py` (`MagePoster`, `Stage*` messages) | Both modes |
| MAGE per-stage token usage → `SessionTokenCounter` | `MagePoster(token_counter=app.token_counter)` | Both modes (Phase 2 P2 footer reads the delta) |
| CARL step callbacks → Textual events | `care/runtime/carl_streamer.py` (`CarlStreamer`, `Step*`/`Progress`/`ChainCompleted`) | Ad-Hoc loop + Production baseline + `/dataset run` |
| MAGEResult → single line | `care/mage_summary.py:summarise_mage_result` | Both modes |
| CARL ReasoningResult → single line | `care/carl_summary.py:summarise_carl_result` | Both modes |
| Baseline → dataset card | `care/runtime/run_recorder.py:record_run_completion` | Production baseline + `/dataset add` |
| Production session log | lazy `$XDG_STATE_HOME/care/sessions/care-session-<ts>.md` (Phase 6 P2) | Production lines via `_append_to_session_log` |

The single execute call-site (`_execute_chain_ad_hoc`) is reused
across Ad-Hoc iterations, Production baseline, and `/dataset run`
replay — one place to change CARL wiring, three call sites to
benefit.

---

## 6. Configuration & precedence

Four layers, lowest precedence first:

```text
Pydantic defaults
   < ~/.config/care/config.toml
       < ./care.toml   (per-project overrides)
           < CARE_*  env vars
```

`CareConfig.load(*, path=None, env=None, cwd=None)` (in
`care/config.py`) implements the layering. Per-project overrides are
documented in [`TODO.md §2 P2`](../TODO.md); env vars in
[`.env.example`](../.env.example).

Nested sections: `mage`, `memory`, `platform`, `sandbox`, `tools`,
`telemetry`, `defaults`. Each becomes a `[section]` table in TOML and a
`CARE_<SECTION>__<FIELD>` env var.

---

## 7. CLI vs TUI

CARE's `[project.scripts]` entry-point routes via `care.cli:main`:

- `care` (no subcommand) → launches the Textual TUI.
- `care catalog [--json] [--kind ...]` → render the
  `CapabilityCatalog` (see [§8 P1](../TODO.md)).
- `care validate <chain.json>` → parse + preflight a chain.
- `care import <pattern>... [--apply]` → batch validate / save.

The CLI handlers are **stream-injectable** (`_cmd_*(args, stdout,
stderr) -> int`) so tests capture output without monkey-patching
`sys`. Future subcommands (`generate`, `run`, `evolve`, `memory ls`)
plug in without touching the router.

---

## 8. Where to look next

| Question                         | Answer                                                                   |
| -------------------------------- | ------------------------------------------------------------------------ |
| What's the upstream prep status? | [`PREPARE.md`](../PREPARE.md) — ✅ / 🚧 / ⛔ per item                    |
| What's the CARE work plan?       | [`TODO.md`](../TODO.md) — every `[P0]`/`[P1]`/`[P2]`/`[P3]`              |
| Which env vars exist?            | [`.env.example`](../.env.example)                                        |
| How to run / develop CARE?       | [`README.md`](../README.md)                                              |
| MAGE-side details?               | [`../../carl-mage/README.md`](../../carl-mage/README.md)                 |
| CARL execution semantics?        | [`../../carl-experiments/README.md`](../../carl-experiments/README.md)   |
| Memory entities + endpoints?     | [`../../gigaevo-memory/openapi.yaml`](../../gigaevo-memory/openapi.yaml) |
| Platform endpoints?              | [`../../gigaevo-platform/README.md`](../../gigaevo-platform/README.md)   |
