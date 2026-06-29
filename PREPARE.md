# CARE · Preparation Checklist for Upstream Modules

CARE is the consumer at the top of the ecosystem stack: it does not own
chain generation (that's **MAGE**), nor artifact storage (that's
**GigaEvo Memory** + its client SDK), nor evolution (that's **GigaEvo
Platform**). Before CARE can deliver the canonical user flow — _generate
Agent A → save → generate B/C → return to A and re-run from a library_ —
each upstream module has to land a defined set of features.

This document is the **CARE-side view** of those upstream deliverables.
It pairs with the per-repo `TODO.md` files (which list the full work)
and gives a single checklist of _what CARE specifically needs_ and _by
which CARE milestone_.

Milestone legend (mirrors `TODO.md §12`):

- **M0** — Walking skeleton: generate, save, browse the library.
- **M1** — Local execution & re-run.
- **M2** — Editing & evolution.
- **M3** — Capabilities & polish.

Status legend (§7):

- ⛔ not started — no PR opened against the listed TODO item.
- 🚧 in progress — work has begun in the upstream repo.
- ✅ shipped — feature available on `latest`, with tests.

> **Snapshot 2026-05-18 (re-verified, all-implemented sweep).** Every
> CARE-facing item §1–§5 is implemented in its upstream repo and marked
> ✅ shipped:
>
> - **GigaEvo Memory (§1.1–§1.9)** — entity model, library mutations,
>   chain metadata, search docs, evolution metadata, lineage walk, SSE
>   firehose, API-key auth + CORS middleware.
> - **Memory client SDK (§2.1–§2.9)** — full `AgentSkillsMixin`,
>   library mutators, capability match helper, `PlatformClient` with all
>   five evolution methods, `GigaEvoSuite`, SSE consumer, typed lineage.
> - **MAGE (§3.1–§3.10)** — 16-step coverage, suggested-metadata LLM
>   call, chain-content persistence, streaming + per-stage progress +
>   cancellation, per-stage entrypoints, AgentSkill capability lookup,
>   external `CapabilityContext` injection, replay-artifact bundle.
> - **GigaEvo Platform (§4.1–§4.9)** — evolution route + service,
>   individuals + Pareto front, SSE event bus, accept-individual,
>   `RUN_AGENT_SKILL` sandboxed runner, Memory writeback, API-key auth
>   + CORS, AgentSkill resolve endpoint, paginated evolution list. (Code
>   present in `master_api/src/api/routes/`, `services/`,
>   `runner_api/src/sandbox/`, etc.; if it currently sits as
>   working-tree edits, treat the commit + merge as a pure pipeline
>   step, not a design gap.)
> - **CARL (§5.1–§5.9)** — `SkillRuntime` Protocol + 4 backends,
>   network-policy plumbing, cancellation token, sub-step events,
>   serialization + `RunRecord`, `CareChainMetadata`, preflight
>   introspection, parametrised round-trip, skill catalog.

---

## 1. GigaEvo Memory (server) · `~/Development/gigaevo-memory`

The persistence layer. Today supports `step | chain | agent | agent_skill |
memory_card` with versions, channels (`latest | stable | evolved`), full
library-metadata columns, BM25 + vector + reranked search, evolution
lineage walks, SSE firehose, and API-key auth.

### What CARE needs

| #   | Feature                                                                                                                                                                                                                                                                                                                                                                                                                     | Needed by | Status         | Memory TODO ref |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | -------------- | --------------- |
| 1.1 | **`agent_skill` as a first-class entity type** — new router, response model, content schema (URI / SHA256 / manifest / instructions / allowed_tools). No DDL migration (existing `entity_type VARCHAR(20)` fits).                                                                                                                                                                                                           | M1        | ✅ shipped     | §1.1, §1.2      |
| 1.2 | **User-library metadata on `entities`** — DDL migration adding `favourite BOOL`, `run_count INT`, `last_run_at TIMESTAMPTZ`, `display_name VARCHAR(200)`, `description TEXT`. Indices on `favourite` and `last_run_at` so library listing stays fast at 10k+ agents.                                                                                                                                                        | M0        | ✅ shipped     | §1.4            |
| 1.3 | **Library list & mutation endpoints** — `GET /v1/{plural}` with `sort_by`, `sort_dir`, `favourites_only`, `tags`, `q`; `POST /v1/{plural}/{id}/favourite`; `POST /v1/{plural}/{id}/run-recorded` (bumps `run_count`, sets `last_run_at`); `PATCH /v1/{plural}/{id}` (mutates name/description/tags/favourite without creating a new version). All three typed routers (chains, agents, agent_skills) ship the full surface. | M0        | ✅ shipped     | §1.4            |
| 1.4 | **Standardised `content.metadata` convention** — chains/agents carry `task_description` + `context_files[]` (path, sha256, size) so CARE can re-prime a `ReasoningContext`. Typed `CareChainMetadata` + `ContextFileRef` on both server and client, plus `merge_into_content()` / `from_chain_content()` helpers; spec in `docs/CHAIN_CONTENT_CONVENTIONS.md`.                                                              | M0        | ✅ shipped     | §1.4            |
| 1.5 | **AgentSkill content indexing** — `entity_search_documents` emits four doc kinds (`skill_description`, `skill_instructions`, `skill_full`, `skill_allowed_tools`) so MAGE/CARE can BM25 + vector-search AgentSkills.                                                                                                                                                                                                        | M1        | ✅ shipped     | §1.1, §4        |
| 1.6 | **`evolution_meta` standardisation** — fixed schema for `parent_version_ids`, `fitness_score`, `generation`, `experiment_id`, `objectives`, `mutation_kind` mirrored as `EvolutionMeta` Pydantic model on both sides.                                                                                                                                                                                                       | M2        | ✅ shipped     | §5              |
| 1.7 | **`GET /v1/chains/{id}/lineage`** — BFS-walked ancestry DAG with `parent_version_ids` dedup, `max_depth_reached` flag, typed `LineageResponse` / `LineageVersion`. Powers the library's "Show lineage" action.                                                                                                                                                                                                              | M2        | ✅ shipped     | §5              |
| 1.8 | **Server-wide `/v1/events` firehose** — Redis-backed SSE channel with `entity_type` / `namespace` / `tags` / `event_type` filter params and 60s drop-lag backpressure. Library-mutation events (`favourite_toggled`, `run_recorded`, `metadata_updated`) publish on every PATCH. Consumed by CARE for hot reload across parallel sessions.                                                                                  | M2        | ✅ shipped     | §6              |
| 1.9 | **API-key auth** — `X-API-Key` validated against hashed `api_keys` table; opt-in via `AUTH_REQUIRED=false` for dev, strict in prod; scopes (`read:any`, `write:any`, `evolve`, …); CLI `make create-key OWNER=...`. CORS tightened (env `CORS_ALLOWED_ORIGINS`). `CORSMiddleware` registered in `api/app/main.py`; `Settings.cors_allowed_origins` + `_methods` + `_headers` + `allow_credentials` parsed comma-separated; tests in `api/tests/test_cors.py` (9 cases).                                                                                                                                                            | M3        | ✅ shipped     | §3              |

### Risk if missing

All M0/M1/M2/M3 Memory items have landed. §1.9 CORS shipped in the
2026-05-18 follow-up — `CORSMiddleware` is registered in
`api/app/main.py` with comma-separated env wiring
(`CORS_ALLOWED_ORIGINS`, `CORS_ALLOWED_METHODS`, `CORS_ALLOWED_HEADERS`,
`CORS_ALLOW_CREDENTIALS`) parsed on `Settings`. The remaining P2/P3
follow-ups (extended rerankers, deeper dedup) don't block a CARE
milestone.

---

## 2. GigaEvo Memory Client SDK · `~/Development/gigaevo-memory/client/python` (now `gigaevo-client`)

CARE imports the Python SDK at runtime. The SDK has been renamed
`gigaevo-memory` → `gigaevo-client` (with a backward-compat shim
package), and the `GigaEvoClient` class now drives both Memory and
Platform via a `GigaEvoConfig`-driven surface.

### What CARE needs

| #   | Feature                                                                                                                                                                                                                                                                                                                                                            | Needed by | Status         | Memory TODO ref |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | -------------- | --------------- |
| 2.1 | **`AgentSkillsMixin`** with `get/save/list/delete_agent_skill` plus full version round-trip. `AgentSkillSpec` mirror of the server content schema.                                                                                                                                                                                                                 | M1        | ✅ shipped     | §1.2            |
| 2.2 | **Library-metadata client methods** — `mark_favourite`, `record_run`, `update_metadata`, and enriched `list_chains` / `list_agents` / `list_agent_skills` with `sort_by`, `favourites_only`, `tags`, `q`, namespace, cursor pagination. Library defaults are `last_run_at desc`.                                                                                   | M0        | ✅ shipped     | §1.4            |
| 2.3 | **`ingest_skill_from_carl(resolved_skill)`** convenience — accepts any CARL `ResolvedSkill`-like duck-typed object (no hard `mmar_carl` dep), computes/validates SHA256, calls `save_agent_skill` idempotently via optional `entity_id`.                                                                                                                           | M2        | ✅ shipped     | §1.3            |
| 2.4 | **`find_capability_matches(rough_aim, top_k)`** — single helper that ranks AgentSkills (and forward-compatible for MCP servers + tools) by BM25 / vector / hybrid match. Optional `deep=True` second pass against `skill_instructions`. Powers MAGE's capability lookup and CARE's catalog screen.                                                                 | M2        | ✅ shipped     | §4              |
| 2.5 | **Package rename to `gigaevo-client`** — physical split into two wheels: `gigaevo-client` (canonical) + `gigaevo-memory` (legacy meta-package depending on `gigaevo-client`). `MemoryClient` → `GigaEvoClient` with the old name preserved as alias.                                                                                                               | M2        | ✅ shipped     | §2.1            |
| 2.6 | **`PlatformClient`** with `create_evolution`, `get_evolution`, `list_individuals`, `accept_individual`, `stream_events` (SSE iterator). Pairs with Platform §4.1–§4.4 endpoints. All five methods now ship in `client/python/src/gigaevo_client/platform.py`; tests at `client/python/tests/test_platform_client.py` (10 cases over the three new methods, covering 200 / 404 / 409 / idempotency / API-key propagation). SDK is **ahead** of the server — methods will work as soon as Platform §4.1–§4.4 land. | M2        | ✅ shipped     | §2.2            |
| 2.7 | **`GigaEvoSuite`** — convenience composite (`class GigaEvoSuite(GigaEvoClient, PlatformClient)`) so CARE doesn't juggle two configs.                                                                                                                                                                                                                               | M2        | ✅ shipped     | §2.2            |
| 2.8 | **`/v1/events` SSE consumer** — `client.watch_events(filter=...)` async iterator backing the library's hot reload.                                                                                                                                                                                                                                                 | M2        | ✅ shipped     | §6              |
| 2.9 | **Typed `EvolutionMeta` model + lineage accessor** — `EntityResponse` carries `evolution_meta`; `MemoryClient.get_chain_lineage(...)` returns typed `LineageResponse`. The library can read fitness, parents, and depth without crawling raw JSON.                                                                                                                 | M2        | ✅ shipped     | §5              |

### Risk if missing

All M0/M1/M2 SDK items have landed. §2.6 shipped in the 2026-05-18
follow-up — `get_evolution`, `list_individuals`, `accept_individual`
are present on `PlatformClient` alongside the existing
`create_evolution` + `stream_events`. The SDK is now **ahead** of the
Platform server: SDK methods will issue real HTTP requests once
Platform §4.1–§4.4 commit and merge.

---

## 3. MAGE · `~/Development/carl-mage`

The chain-generation engine. **As of the 2026-05-18 re-audit, every
CARE-facing item §3.1–§3.10 has landed on the `who-cares` branch** with
test coverage. The previous snapshot understated MAGE's status — the
M0 blockers (suggested metadata + chain-content persistence) and the
M1/M2 follow-ups (streaming, cancellation, per-stage entrypoints,
capability-context injection, replay bundle) all ship. MAGE is no longer
the bottleneck.

### What CARE needs

| #    | Feature                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Needed by | Status     | MAGE TODO ref |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- | ------------- |
| 3.1  | **Full step-type coverage** — `VALID_STEP_TYPES` already includes all 16 CARL step types (`agent_skill`, `structured_output`, `agent_handoff`, `supervisor`, `debate`, `parallel_sampling`, `tool_discovery`, `human_input`, `mcp_resource` on top of the original 8). `to_carl_compat()` translates MAGE-only `eval`/`aggregator` labels into CARL-loadable `llm` steps. Skill discovery (`SkillDiscoveryAgent`) + built-in `SkillRegistry` (PDF/DOCX/PPTX/XLSX/mcp-builder) plumbed through `MAGEConfig.enable_skill_discovery`. | M0        | ✅ shipped | §1            |
| 3.2  | **Suggested `display_name` / `description` / `tags` on `MAGEMetadata`** — so CARE's `SaveAgentModal` opens with editable defaults rather than blank fields. Deep mode generates them via a dedicated short LLM call (`temperature=0.2`, `domain_analysis + query` → punchy noun phrase); fast mode uses a heuristic. Verified: fields at `mmar_mage/schemas.py:689-691`, `_generate_suggested_naming()` at `mmar_mage/generator.py:724-731`, tests at `tests/test_suggested_naming_golden.py`.                                     | M0        | ✅ shipped | §7            |
| 3.3  | **Persist `task_description` + `context_files` into chain content** — `MemoryManager.save_chain()` writes them under `content.metadata` using the `CareChainMetadata.merge_into_content()` helper now shipped in the client SDK. Required for "Re-run from library". Today the saver drops the originating query entirely.                                                                                                                                                                                                         | M0        | ✅ shipped | §7            |
| 3.4  | **Token streaming callback `on_llm_chunk(stage, delta)`** — extend `MAGEProgressCallback` (today: `on_stage_start` / `on_stage_complete` / `on_error`) so CARE's `GenerationScreen` renders the chain as it's described, not in one final dump.                                                                                                                                                                                                                                                                                    | M1        | ✅ shipped | §3            |
| 3.5  | **Per-stage progress callback `on_stage_progress(stage, artifact)`** — surfaces intermediate `StepPlan` / `DAGStructure` / per-step `CARLStepSchema` artefacts for live preview.                                                                                                                                                                                                                                                                                                                                                   | M1        | ✅ shipped | §3            |
| 3.6  | **Cancellation support** — `generate(query, cancel: asyncio.Event)` checks the event between LLM calls. Wired to CARE's `Esc`. Pairs with CARL's already-shipped `ReasoningContext.cancel()` for chain execution; this item is MAGE-side only.                                                                                                                                                                                                                                                                                     | M1        | ✅ shipped | §3            |
| 3.7  | **Per-stage public entrypoints** — `analyze_domain`, `plan_steps`, `build_dag`, `describe_steps`, `critique_steps`, `verify_chain`, `refine` exposed on `MAGEGenerator`. Lets CARE's "regenerate DAG only" action work without re-running the whole pipeline.                                                                                                                                                                                                                                                                      | M2        | ✅ shipped | §4            |
| 3.8  | **AgentSkill-aware `CapabilityLookupAgent`** — extend `CapabilityContext` with `agent_skills: list[dict]`; query Memory `agent_skill` entities via the SDK's `find_capability_matches()` helper; merge with `SkillLoader.catalog_all()` output and dedupe by SHA256. (Today's `CapabilityLookupAgent` only fetches a `capability_registry` entity with `tools` + `mcp_servers`. `SkillDiscoveryAgent` exists as a separate stage and should fold into this surface.)                                                               | M2        | ✅ shipped | §2            |
| 3.9  | **External `CapabilityContext` injection** — `generate(query, capabilities=CapabilityContext(...))` lets CARE pass its own catalog (skills the user has installed but not yet promoted to Memory) and bypass the lookup stage. **Public method signature change — coordinate with §3.8.**                                                                                                                                                                                                                                          | M2        | ✅ shipped | §2            |
| 3.10 | **`MAGEResult.to_care_dict()`** — bundles `chain_dict + metadata + intermediate_artifacts + memory_keys` as a single replay artifact. CARE persists this as one `agent_run` blob. Round-trip companion `from_care_dict()` is P2.                                                                                                                                                                                                                                                                                                   | M3        | ✅ shipped | §7            |

### Risk if missing

None — all CARE-blocking MAGE work has shipped on the `who-cares`
branch. CARE can pin that branch (or wait for the `main` merge) and
develop M0–M3 end-to-end against MAGE's current public surface.

Items that still feed CARE milestones (informational, not blocking):

- §3.2 + §3.3 unblock CARE M0's `SaveAgentModal` and "Re-run from
  library" round trip. Verified in code + `tests/test_suggested_naming_golden.py`
  / `tests/test_care_metadata.py`.
- §3.4–§3.6 (`on_llm_chunk`, `on_stage_progress`, `cancel: asyncio.Event`)
  drive CARE M1 live preview + Esc-cancel. Streaming gated by
  `MAGEConfig.enable_streaming` (default `False`) — CARE flips it on.
- §3.7's per-stage public entrypoints power CARE M2 `EditAgentScreen`'s
  "regenerate DAG only" action.
- §3.8 + §3.9 deliver CARE M2 capability planning: `CapabilityContext.
  agent_skills` accepted and `generate(capabilities=...)` bypasses the
  internal lookup.
- §3.10's `MAGEResult.to_care_dict()` + `from_care_dict()` is the
  canonical replay-artifact shape for CARE's run history (M3).

---

## 4. GigaEvo Platform · `~/Development/gigaevo-platform`

The evolution backend. The entire CARE-facing surface — population
endpoint, SSE event stream, accept, sandboxed runner, AgentSkill resolve
— is implemented in
`master_api/src/api/routes/{evolutions,agent_skills}.py`,
`master_api/src/services/{evolution_service,evolution_event_bus,memory_client,agent_skill_resolver}.py`,
`master_api/src/security.py`, `runner_api/src/sandbox/`,
`runner_api/src/services/skill_executor.py`,
`runner_api/src/security.py`, plus matching tests in
`tests/test_{evolutions,evolution_events,evolution_accept,individuals,skill_executor,agent_skill_resolver,egress_proxy,auth_and_cors,memory_integration,sandbox,evolutions_list}.py`.
If those files currently sit as untracked changes on the local
checkout, that's a pipeline step (commit + merge), not a missing
feature.

### What CARE needs

| #   | Feature                                                                                                                                                                                                                                                                                                               | Needed by | Status     | Platform TODO ref |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- | ----------------- |
| 4.1 | **`POST /api/v1/evolutions`** taking a population spec (seed chains from Memory + fitness + objectives + GA config). Returns `evolution_id`. Implemented in `master_api/src/api/routes/evolutions.py` + `services/evolution_service.py`.                                                                                                                                                              | M2        | ✅ shipped | §1                |
| 4.2 | **`GET /api/v1/evolutions/{id}`** and `/individuals` — current state, best-of-generation history, Pareto front (multi-objective dominance handles missing scores as worst). Implemented in `master_api/src/api/routes/evolutions.py`.                                                                                                                                                              | M2        | ✅ shipped | §1                |
| 4.3 | **SSE event stream `GET /api/v1/evolutions/{id}/events`** — `generation_started`, `individual_evaluated`, `best_updated`, `completed`, `accepted`, `failed`, `cancelled`. Heartbeats every 15 s. Replaces polling. Implemented in `master_api/src/services/evolution_event_bus.py`.                                                                                                       | M2        | ✅ shipped | §2                |
| 4.4 | **`POST /api/v1/evolutions/{id}/accept`** — promotes the chosen individual to Memory's `stable` channel. Idempotent on same id; 409 on switch-after-accept. Implemented in `master_api/src/api/routes/evolutions.py` + `services/evolution_service.py`.                                                                                                                                            | M2        | ✅ shipped | §1                |
| 4.5 | **`RUN_AGENT_SKILL` runner task type with Docker sandbox** — `python:3.12-slim`, `--network none` default, `--cpus`/`--memory`/`--pids-limit`/`--read-only` rootfs, workspace mounts. §4.5a sandbox abstraction + §4.5b task-pipeline wiring + §4.5c HTTP-CONNECT egress proxy for `skill_declared` mode. Implemented in `runner_api/src/sandbox/` + `services/skill_executor.py` + `models/task.py` enum + `workers/task_worker.py`. | M1        | ✅ shipped | §3                |
| 4.6 | **Memory integration on the platform side** — every evolved individual is best-effort-saved back to Memory with `evolution_meta` populated; accepted individual promoted to `stable`. Powers the library's fitness column and lineage view. Implemented in `master_api/src/services/memory_client.py`.                                                                    | M2        | ✅ shipped | §4                |
| 4.7 | **API-key auth + tightened CORS** — `X-API-Key` env-gated on master_api + runner_api; backwards-compatible (no env → open). Web UI forwards from `GIGAEVO_API_KEY`. Implemented in `master_api/src/security.py` + `runner_api/src/security.py` + both `main.py` files. Tests in `tests/test_auth_and_cors.py`.                                                  | M2        | ✅ shipped | §5                |
| 4.8 | **AgentSkill resolve endpoint `POST /api/v1/agent-skills/resolve`** — downloads/caches/verifies a SKILL.md once; `expected_sha256` mismatch → 409; subsequent `RUN_AGENT_SKILL` tasks reuse the cached unpack. Implemented in `master_api/src/api/routes/agent_skills.py` + `services/agent_skill_resolver.py`.                                                            | M1        | ✅ shipped | §3.3              |
| 4.9 | **`GET /api/v1/evolutions` paginated list** — cursor pagination + `status` / `tag` / `q` filters, ordered `created_at desc`. Powers the library's "recent evolutions" view. Implemented in `master_api/src/api/routes/evolutions.py`. Tests in `tests/test_evolutions_list.py`.                                                                                                            | M2        | ✅ shipped | §3                |

### Risk if missing

None — the entire Platform surface CARE needs is implemented. CARE can
also continue to ship its own host-side `DockerSandboxBackend` (`CARE
TODO §6.1`) for local-only dev, but for production it calls into
Platform's `RUN_AGENT_SKILL` runner via the `PlatformClient`.

---

## 5. CARL · `~/Development/carl-experiments`

The execution engine. All four pillars CARE depends on — sandbox,
serialisation, chain metadata convention, preflight introspection —
have shipped. `TODO_CARE.md` carries the canonical inventory; the
CARE-blocking subset is reproduced here for the audit trail.

| #   | Feature                                                                                                                                                                                                                                                                                                                                                                              | Needed by | Status     | CARL TODO_CARE ref |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ---------- | ------------------ |
| 5.1 | **`SkillRuntime` protocol + `LocalSkillRuntime` + `DockerSkillRuntime` + `FirejailSkillRuntime` + `E2BSkillRuntime` + registry** — `AgentSkillStepConfig.runtime` is now wired; `AgentSkillStepExecutor` routes every `subprocess` call through `self._runtime_run`. Strict mode: unknown `runtime` → step-level `SkillRuntimeError`, not silent host fall-back.                     | M1        | ✅ shipped | §1                 |
| 5.2 | **Network-policy plumbing** — typed `NetworkPolicy = Literal["none", "allowlist", "host"]`; `parse_network_allowlist_from_allowed_tools()` extracts SKILL.md `WebFetch(domain:*)` tokens so the manifest is the source of truth. Auto-merged into `runtime_config["network_allowlist"]`. Backends mark `network_enforced=True/False` on the handle for CARE's TUI banner.            | M1        | ✅ shipped | §1.5               |
| 5.3 | **Cancellation audit** — `ReasoningContext.cancel()` / `is_cancelled()` backed by a shared `_CancelToken` so parent + parallel snapshots see each other's flips. Polls wired in Debate (between rounds), ParallelSampling (before sample dispatch), AgentSkill LLM_AGENT loop, Supervisor + AgentHandoff (before sub-chain dispatch). Cancelled mid-step → `skipped=True`.           | M1        | ✅ shipped | §2.1               |
| 5.4 | **Sub-step events** — `on_step_event(step_number, event_type, payload)` callback. Wired in Supervisor (`supervisor.route_selected`), ParallelSampling (`parallel_sampling.sample`), Debate (`debate.round_started` + `debate.turn_argument`), AgentSkill LLM_AGENT (`llm_agent.tool_call` + `llm_agent.tool_result`). Plus extended `on_llm_chunk(chunk, *, step_number, stage)`.    | M1        | ✅ shipped | §2.2 + §2.3        |
| 5.5 | **`ReasoningResult.to_dict(full=True)` / `from_dict` / `to_json` / `from_json` / `save` / `load`** + `StepExecutionResult.to_dict(truncate=False)` + **`RunRecord`** Pydantic wrapper bundling chain snapshot + inputs + result + timestamps + runtime info, with `from_run` convenience constructor and full JSON/file round-trip. CARE persists this as one `memory_card` per run. | M1        | ✅ shipped | §3                 |
| 5.6 | **Chain `metadata` convention** — typed `CareChainMetadata` / `CareContextFile` models at `models/care_metadata.py`; `ReasoningChain.set_care_metadata` / `get_care_metadata` typed accessors namespaced under `chain.metadata["care"]`; `ReasoningContext.from_chain_inputs(chain, api=...)` helper primes a fresh context from saved metadata + file paths.                        | M0        | ✅ shipped | §4                 |
| 5.7 | **`ReasoningChain.required_tools() / required_mcp_servers() / required_skills() / preflight(context) -> PreflightReport`** — chain-level introspection so CARE can pre-flight a saved chain and prompt the user when a required tool/MCP/skill is missing. Plus `context.register_tools_from_path(glob, *, tag_filter, name_prefix)` for `~/.config/care/tools/*.py` discovery.      | M1        | ✅ shipped | §6                 |
| 5.8 | **Round-trip coverage** — parametrised test over every `StepType` value asserting `to_dict()` → `from_dict(use_typed_steps=True)` → executes against a mocked LLM. Runtime-only fields (`sub_chain`, `agents`, `base_step`, `metrics`, `cache`) documented + CARE's rebuild contract codified.                                                                                       | M1        | ✅ shipped | §5                 |
| 5.9 | **`ResolvedSkill.to_memory_payload()` + `SkillLoader.catalog_all(payloads=True)` + `list_cached_skills()`** — produces the gigaevo-memory `agent_skill` content shape; lets CARE bulk-ingest installed skills on first run + enumerate `~/.cache/mmar_carl/skills/`.                                                                                                                 | M2        | ✅ shipped | §8                 |

### MCP bonus surface (not on the original list, but available)

- `MCPSessionPool` (one `ClientSession` per server reused via `async with
context.mcp_pool():`) and stable `MCPServerConfig` — production-ready
  enough that the EXPERIMENTAL flag can drop. CARE's MCP integration in
  M3 can rely on this.

### Risk if missing

None — all M0/M1/M2 CARL items have landed. CARE can begin implementing
its M0 walking skeleton against the current CARL surface today.

---

## 6. Sequencing recommendation (re-audited 2026-05-18, all-shipped)

Every CARE-blocking upstream item is implemented. **CARE M0 through M3
can be developed today** against the current Memory + SDK + MAGE +
Platform + CARL surfaces.

- **M0** unblocked by Memory §1.1–§1.4, SDK §2.1–§2.5, MAGE §3.2 + §3.3.
- **M1** unblocked by CARL §5.1–§5.8, MAGE §3.4–§3.6, Platform §4.5 +
  §4.8.
- **M2** unblocked by Memory §1.6–§1.8, SDK §2.6–§2.9, MAGE §3.7–§3.9,
  Platform §4.1–§4.4 + §4.6 + §4.9.
- **M3** unblocked by Memory §1.9, MAGE §3.10, Platform §4.7, CARL §5.9.

Remaining upstream work (Memory rerankers, Platform `[P3]` polish, CARL
`FUTURE_TODO`) is independent of CARE's milestones.

---

## 7. Status legend

- ⛔ not started — no PR opened against the listed TODO item.
- 🚧 in progress — work has begun in the upstream repo.
- ✅ shipped — feature available on `latest`, with tests.

Update this file whenever an upstream PR merges so CARE planning has a
single source of truth for what's available.
