"""CARE-side Platform facade (TODO §7 P1).

`CarePlatform` wraps :class:`gigaevo_client.PlatformClient` with the
narrow surface CARE actually uses — same role :class:`CareMemory`
plays for the Memory side. It centralises CARE-specific concerns
the generic SDK shouldn't carry:

- Build a ``CareEvolutionSpec`` from the chosen base chain +
  user-friendly evolution settings (so the EvolutionScreen doesn't
  have to know the Platform's spec dict shape).
- Stamp CARE-side correlation tags (``source:care``,
  ``base_chain:{id}``) onto every spec for cross-system audit.
- Surface a typed :class:`EvolutionRef` per submit so the screen
  can subscribe to `stream_events` immediately without re-parsing
  the create response.
- Provide a tiny `EvolutionEventStream` async-iterator wrapper that
  re-yields the SDK's synchronous events into a Textual
  ``Worker``-friendly shape.

The facade exposes both: typed methods (`start_evolution`, …) and
``platform`` as an escape-hatch property returning the underlying
SDK client.
"""

from __future__ import annotations

import logging

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from gigaevo_client import GigaEvoConfig, PlatformClient

from care.config import CareConfig, resolve_platform_api_base_url

_log = logging.getLogger("care.platform")

CARE_SOURCE_TAG = "source:care"
"""Stamped on every evolution spec CARE submits so Platform-side
operators can tell CARE-driven runs apart from other consumers."""


@dataclass(frozen=True)
class EvolutionRef:
    """Lightweight handle returned by :meth:`CarePlatform.start_evolution`.

    The Platform's create-response is a free-form dict; CARE pulls
    out the canonical fields it actually needs so call-sites
    (especially the EvolutionScreen state model) don't drift over
    field names.
    """

    evolution_id: str
    base_chain_id: str
    status: str = "queued"
    extras: dict[str, Any] = field(default_factory=dict)


class CarePlatform:
    """Narrow CARE-facing facade over gigaevo-platform.

    Use :meth:`from_config` for the normal CARE startup path. The
    bare constructor is for tests + dependency injection where the
    caller wants to plug in a custom :class:`PlatformClient`
    (e.g. one pointed at a respx-mocked URL).
    """

    def __init__(
        self,
        client: PlatformClient,
        *,
        mutation_model_id: str = "care-mutation",
        validation_model_id: str = "care-validation",
    ):
        self._client = client
        # Logical ids CARE sends in the experiment spec. The
        # Platform's ``llm_models.yml`` (seeded by care-install
        # wizard) maps these to concrete URL / model / key sets.
        self._mutation_model_id = mutation_model_id
        self._validation_model_id = validation_model_id

    @staticmethod
    def _sync_llm_registry(
        config: CareConfig,
        *,
        strict: bool = False,
        mutation_max_tokens: int | None = None,
        validation_max_tokens: int | None = None,
    ) -> None:
        """Refresh ``llm_models.yml`` so Platform evolution uses MAESTRO creds."""
        from care.runtime.platform_llm_sync import (
            sync_platform_llm_registry,
            try_sync_platform_llm_registry,
        )

        sync_kwargs = {
            "mutation_max_tokens": mutation_max_tokens,
            "validation_max_tokens": validation_max_tokens,
        }
        if strict:
            sync_platform_llm_registry(config, **sync_kwargs)
            return
        try_sync_platform_llm_registry(config, **sync_kwargs)

    @classmethod
    def from_config(cls, config: CareConfig) -> "CarePlatform":
        """Construct from a :class:`CareConfig`.

        Reads ``config.platform.base_url`` / ``api_key`` /
        ``timeout`` and builds a :class:`GigaEvoConfig` the SDK
        consumes directly.
        """
        sdk_cfg = GigaEvoConfig(
            platform_base_url=resolve_platform_api_base_url(config.platform),
            api_key=config.platform.api_key,
            timeout=config.platform.timeout,
        )
        # Default to the CARE-managed Platform-side allowlist ids
        # that care-install's wizard seeds into ``llm_models.yml``.
        # ``config.platform.mutation_model`` is the upstream provider
        # model NAME (e.g. "mistralai/mistral-medium-3-5") plumbed
        # into the wizard's YAML, NOT the Platform-side id, so we
        # don't reuse it here. Users with legacy setups (no wizard)
        # can override via ``CARE_PLATFORM__MUTATION_MODEL_ID``.
        mutation_id = (
            getattr(config.platform, "mutation_model_id", None) or "care-mutation"
        )
        validation_id = (
            getattr(config.platform, "validation_model_id", None) or "care-validation"
        )
        plat = cls(
            PlatformClient.from_config(sdk_cfg),
            mutation_model_id=mutation_id,
            validation_model_id=validation_id,
        )
        plat._sync_llm_registry(config)
        from care.runtime.platform_bootstrap import bootstrap_local_platform_once

        bootstrap_local_platform_once(config)
        return plat

    @property
    def client(self) -> PlatformClient:
        """Escape hatch for callers that need the raw SDK surface."""
        return self._client

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Hit a Platform health endpoint.

        gigaevo-client 0.3 probes ``GET /api/v1/status``, but many
        master-api builds only expose ``/health`` and
        ``/api/v1/status/health``. Try those fallbacks before giving up.
        """
        last_exc: Exception | None = None
        for path in ("/health", "/api/v1/status/health", "/api/v1/status"):
            try:
                return self._client._get(path)  # noqa: SLF001 — narrow probe shim
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return self._client.health()

    # ------------------------------------------------------------------
    # Evolution lifecycle
    # ------------------------------------------------------------------

    def start_evolution(
        self,
        *,
        base_chain_id: str,
        evolution_mode: str = "full_chain",
        max_iterations: int = 10,
        population_size: int = 8,
        validation_criteria: str | None = None,
        test_data_path: str | None = None,
        tags: list[str] | None = None,
        objectives: list[str] | None = None,
        extras: dict[str, Any] | None = None,
        base_chain_content: dict[str, Any] | None = None,
        target_column: str = "expected",
        validation_type: str | None = None,
        continuous_metric: str | None = None,
        binary_method: str | None = None,
        regexp_pattern: str = "",
        name: str | None = None,
        mutation_max_tokens: int | None = None,
        validation_max_tokens: int | None = None,
    ) -> EvolutionRef:
        """Submit a new evolution; return a typed :class:`EvolutionRef`.

        When ``base_chain_content`` is provided (recommended), CARE
        submits via the **chain-experiment** route
        (``POST /api/v1/experiments/chains``) — the only path the live
        Platform scheduler actually drains. The chain JSON is inlined
        in ``base_chain_config`` per the platform's
        ``ChainExperimentCreate`` schema; ``test_data_path`` is
        treated as a local file (uploaded via
        ``/api/v1/experiments/upload``) when it isn't already a
        storage path returned by a prior upload.

        When ``base_chain_content`` is ``None`` we fall back to the
        legacy ``POST /api/v1/evolutions`` endpoint — kept so old
        callers that don't fetch the chain still work, even though
        no dispatcher will pick those records up on the current
        platform build.

        Args:
            base_chain_id: Memory entity id of the seed chain.
            base_chain_content: Full CARL chain dict (with ``steps``
                + ``version``). Fetch it via
                :meth:`care.memory.CareMemory.get_chain` and pass it
                in; CARE doesn't store a Memory reference here.
            evolution_mode: ``"full_chain"`` or ``"single_step"``.
            max_iterations: GA generations.
            population_size: Individuals per generation. Forwarded
                via ``extras`` to keep the platform's
                ``ChainExperimentCreate`` schema strict.
            validation_criteria: Free-form natural-language prompt
                for the judge. Forwarded to the chain-experiment
                spec as a ``ChainValidationCriteria`` payload with
                ``validation_type="Continuous (0..1)"`` so the
                judge runs in scoring mode.
            test_data_path: Local file (``.jsonl`` or ``.csv``)
                uploaded automatically; or a remote storage path
                returned by a prior ``/experiments/upload``.
            target_column: CSV column the chain output is scored
                against. Defaults to ``"expected"`` (matches
                CARE's recommended JSONL schema).
            name: Human-readable experiment name. Defaults to
                ``f"CARE evolve {base_chain_id[:8]}"``.

        Returns:
            :class:`EvolutionRef` with the canonical fields the
            EvolutionScreen needs (id, base_chain_id, status). The
            full create-response is preserved on ``ref.extras``.
        """
        merged_tags = _merge_care_source_tag(tags)

        from care.config import CareConfig

        self._sync_llm_registry(
            CareConfig.load(),
            strict=True,
            mutation_max_tokens=mutation_max_tokens,
            validation_max_tokens=validation_max_tokens,
        )

        if base_chain_content is not None:
            from care.runtime.platform_chain_adapter import (
                prepare_chain_for_platform_evolution,
            )
            from care.runtime.platform_chain_gate import (
                gate_chain_for_platform_evolution,
            )
            from care.tool_synthesis import (
                bundled_tools_for_chain,
                bundled_tools_to_python_code,
            )

            cfg = CareConfig.load()
            bundled = bundled_tools_for_chain(base_chain_content, cfg)
            bundled_names = frozenset(t["name"] for t in bundled)
            python_code = bundled_tools_to_python_code(bundled)
            if bundled:
                _log.info(
                    "bundling %d synthesized tool(s) for Platform custom_tools.py: %s",
                    len(bundled),
                    ", ".join(sorted(bundled_names)),
                )

            prepared = prepare_chain_for_platform_evolution(
                base_chain_content,
                target_column=target_column,
                bundled_tool_names=bundled_names,
            )
            if prepared.notes:
                _log.info(
                    "Platform chain adapter (%s): %s",
                    "adapted" if prepared.adapted else "noop",
                    "; ".join(prepared.notes[:6]),
                )
            gate_issues = gate_chain_for_platform_evolution(
                prepared.chain,
                target_column=target_column,
                bundled_tool_names=bundled_names,
            )
            if gate_issues:
                detail = "\n".join(f"- {issue}" for issue in gate_issues[:8])
                if len(gate_issues) > 8:
                    detail += f"\n- … and {len(gate_issues) - 8} more"
                raise ValueError(
                    "Seed chain is not compatible with Platform runner "
                    "(even after CARE auto-adaptation):\n"
                    f"{detail}"
                )
            return self._start_chain_experiment(
                base_chain_id=base_chain_id,
                base_chain_content=prepared.chain,
                evolution_mode=evolution_mode,
                max_iterations=max_iterations,
                population_size=population_size,
                validation_criteria=validation_criteria,
                test_data_path=test_data_path,
                target_column=target_column,
                tags=merged_tags,
                name=name,
                extras=extras,
                validation_type=validation_type,
                continuous_metric=continuous_metric,
                binary_method=binary_method,
                regexp_pattern=regexp_pattern,
                python_code=python_code or None,
            )

        spec: dict[str, Any] = {
            "base_chain_id": base_chain_id,
            "evolution_mode": evolution_mode,
            "max_iterations": max_iterations,
            "population_size": population_size,
            "tags": merged_tags,
        }
        if validation_criteria is not None:
            spec["validation_criteria"] = validation_criteria
        if test_data_path is not None:
            spec["test_data_path"] = test_data_path
        if objectives is not None:
            spec["objectives"] = list(objectives)
        if extras:
            spec.update(extras)

        response = self._client.create_evolution(spec)
        return EvolutionRef(
            evolution_id=str(
                response.get("evolution_id") or response.get("id") or ""
            ),
            base_chain_id=base_chain_id,
            status=str(response.get("status") or "queued"),
            extras=dict(response),
        )

    # ------------------------------------------------------------------
    # Chain-experiment route (the live dispatcher's actual entry point)
    # ------------------------------------------------------------------

    def _start_chain_experiment(
        self,
        *,
        base_chain_id: str,
        base_chain_content: dict[str, Any],
        evolution_mode: str,
        max_iterations: int,
        population_size: int,
        validation_criteria: str | None,
        test_data_path: str | None,
        target_column: str,
        tags: list[str],
        name: str | None,
        extras: dict[str, Any] | None,
        validation_type: str | None = None,
        continuous_metric: str | None = None,
        binary_method: str | None = None,
        regexp_pattern: str = "",
        python_code: str | None = None,
    ) -> EvolutionRef:
        """Submit via ``POST /api/v1/experiments/chains``.

        Side effects: if ``test_data_path`` is a local file, upload
        it via ``/api/v1/experiments/upload`` first and use the
        returned storage path as ``data_path``.
        """
        chain_payload = _ensure_chain_version(base_chain_content)
        data_path = self._resolve_data_path(test_data_path)
        experiment_name = name or f"CARE evolve {base_chain_id[:8]}"

        # The Platform validates ``llm_model`` against its
        # ``llm_models.yml`` allowlist. CARE's wizard seeds two
        # CARE-managed entries (``care-mutation``,
        # ``care-validation``) wired to the user's chosen
        # provider; we send those by default. ``extras`` can
        # still override per-call.
        llm_model = (extras or {}).get("llm_model") or self._mutation_model_id

        # Stash the user's free-form rubric in the description so
        # it survives the round-trip even though the Platform's
        # schema has nowhere to use it as an actual judge prompt
        # (see ``_build_chain_validation_criteria`` for why).
        description_lines = [
            f"CARE-driven chain evolution. base_chain_id={base_chain_id}."
        ]
        if validation_criteria:
            description_lines.append("")
            description_lines.append("Validation rubric (user intent):")
            description_lines.append(validation_criteria)
        spec: dict[str, Any] = {
            "name": experiment_name,
            "description": "\n".join(description_lines),
            "data_path": data_path or "",
            "target_column": target_column,
            "base_chain_config": json.dumps(chain_payload),
            "validation_criteria": _build_chain_validation_criteria(
                validation_type=validation_type,
                continuous_metric=continuous_metric,
                binary_method=binary_method,
                regexp_pattern=regexp_pattern,
            ),
            "max_iterations": max_iterations,
            "evolution_mode": evolution_mode,
            "llm_model": llm_model,
        }
        if python_code and python_code.strip():
            spec["python_code"] = python_code.strip()
        if extras:
            # Population size + objectives don't fit the strict
            # ChainExperimentCreate schema — drop them silently
            # rather than 422 the user. CARE-side dashboards
            # still see them via the tags below if needed.
            for k, v in extras.items():
                if k in {"population_size", "objectives", "tags", "llm_model"}:
                    continue
                spec[k] = v

        response = self._client.create_chain_experiment(spec)
        experiment_id = str(
            response.get("id")
            or response.get("experiment_id")
            or ""
        )
        # Chain-experiment create only persists the record in
        # ``prepared`` state — the queue scheduler scans for
        # ``queued`` rows, so without an explicit start the run
        # sits forever. Kick it off here. Surface any 409/503 as
        # the original create response's status (don't fail the
        # whole submit: the experiment exists and can be started
        # later via the dashboard).
        if experiment_id:
            try:
                self._client.start_experiment(experiment_id)
            except Exception:
                pass
            try:
                from care.runtime.evolution_chain_templates import (
                    schedule_chain_template_sync,
                )

                schedule_chain_template_sync(
                    experiment_id,
                    validation_type=validation_type,
                    continuous_metric=continuous_metric,
                    binary_method=binary_method,
                    target_column=target_column,
                    regexp_pattern=regexp_pattern,
                )
            except Exception:
                pass
        return EvolutionRef(
            evolution_id=experiment_id,
            base_chain_id=base_chain_id,
            status=str(response.get("status") or "queued"),
            extras={
                **dict(response),
                "_care_route": "experiments/chains",
                "_care_tags": tags,
                "_care_population_size": population_size,
            },
        )

    def _resolve_data_path(self, test_data_path: str | None) -> str | None:
        """Convert a local path into a Platform storage path via
        ``/api/v1/experiments/upload``.

        Heuristic: any path that exists on disk is treated as
        local and uploaded; anything else is assumed to already be
        a Platform-side storage path (returned by a prior upload).
        JSONL files are converted to CSV on the fly so the
        Platform's CSV-only uploader accepts them.
        """
        if test_data_path is None:
            return None
        local = Path(test_data_path)
        if not local.exists():
            return test_data_path
        if local.suffix.lower() == ".jsonl":
            csv_bytes = _jsonl_to_csv_bytes(local)
            filename = local.with_suffix(".csv").name
            return self.upload_dataset_bytes(
                csv_bytes, filename=filename,
            )
        return self.upload_dataset(local)

    def upload_dataset(self, local_path: Path | str) -> str:
        """Upload a local CSV file via ``/api/v1/experiments/upload``.
        Returns the storage path the Platform records under the
        ``data_path`` column."""
        path = Path(local_path)
        return self.upload_dataset_bytes(
            path.read_bytes(), filename=path.name,
        )

    def upload_dataset_bytes(self, payload: bytes, *, filename: str) -> str:
        """Lower-level multipart upload — used by
        :meth:`upload_dataset` and by the JSONL → CSV path. Returns
        the Platform storage path."""
        http = getattr(self._client, "_http", None)
        if http is None:
            raise RuntimeError(
                "PlatformClient is missing an `_http` httpx client — "
                "can't upload dataset.",
            )
        resp = http.post(
            "/api/v1/experiments/upload",
            files={"file": (filename, payload, "text/csv")},
        )
        resp.raise_for_status()
        data = resp.json()
        path = data.get("data_path") or data.get("path")
        if not path:
            raise RuntimeError(
                f"Platform upload returned no data_path: {data!r}",
            )
        return str(path)

    # Statuses that mean an evolution is actively running right now (vs
    # queued/terminal) — used by the chat footer's "N running" indicator.
    _ACTIVE_EVOLUTION_STATUSES = frozenset(
        {"running", "dispatching", "initializing", "preparing"}
    )

    def running_evolution_count(self) -> int:
        """How many evolutions/experiments are actively running right now.

        Counts client-side across the merged run inbox (legacy evolutions +
        chain experiments) so it works regardless of which path produced
        them. Returns 0 on any error — the footer indicator must never
        raise into the UI."""
        try:
            envelope = self.list_evolutions(limit=200)
        except Exception:
            return 0
        items = (
            envelope.get("items")
            or envelope.get("evolutions")
            or envelope.get("results")
            or envelope.get("data")
            or []
        )
        count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").lower()
            if status in self._ACTIVE_EVOLUTION_STATUSES:
                count += 1
        return count

    def get_evolution(self, evolution_id: str) -> dict[str, Any]:
        """Return the current evolution/experiment state.

        Routes by id prefix:

        * ``exp_*`` — chain-experiment, fetched via the experiments
          API (``GET /api/v1/experiments/{id}`` merged with
          ``/status`` + ``/results`` so the dashboard sees fitness
          + Pareto data in the same shape as legacy evolutions).
        * Anything else — legacy ``/api/v1/evolutions/{id}``.
        """
        if evolution_id.startswith("exp_"):
            return self._get_experiment_as_evolution(evolution_id)
        return self._client.get_evolution(evolution_id)

    def list_evolutions(
        self,
        status: str | None = None,
        *,
        tag: str | None = None,
        q: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return the combined run inbox (legacy evolutions +
        chain experiments).

        Chain experiments are the path the live Platform actually
        drains (legacy ``/api/v1/evolutions`` has no dispatcher),
        so without merging them in the dashboard would show zero
        runs even though the user has runs in flight.

        Optional filters (``status``, ``tag``, ``q``, ``cursor``,
        ``limit``) are honored for legacy evolutions; chain
        experiments are filtered client-side after the merge
        because the Platform's ``/api/v1/experiments`` endpoint
        doesn't accept these params yet.
        """
        legacy_envelope: dict[str, Any] = {}
        try:
            legacy_envelope = self._client.list_evolutions(
                status=status,
                tag=tag,
                q=q,
                cursor=cursor,
                limit=limit,
            )
        except Exception:
            # Legacy endpoint may be missing on minimal setups;
            # don't let it block the experiments listing.
            legacy_envelope = {"items": []}

        legacy_items = list(legacy_envelope.get("items") or [])
        raw_experiments = self._fetch_experiment_list()

        experiments_norm = [
            self._experiment_to_evolution_row(e)
            for e in raw_experiments
            if isinstance(e, dict)
        ]
        # Client-side filter so callers that pass ``status=...``
        # still see a consistent inbox (Platform-side filter is
        # absent on the experiments route).
        if status:
            wanted = status.lower()
            experiments_norm = [
                r for r in experiments_norm
                if (r.get("status") or "").lower() == wanted
            ]
        if tag:
            experiments_norm = [
                r for r in experiments_norm
                if tag in (r.get("tags") or ())
            ]

        merged = experiments_norm + legacy_items
        # Newest first so the dashboard shows in-flight runs at
        # the top. ``created_at`` may be missing on bare envelopes
        # — fall back to the empty string so sort is stable.
        merged.sort(
            key=lambda r: str(r.get("created_at") or ""),
            reverse=True,
        )
        return {
            **legacy_envelope,
            "items": merged,
        }

    def list_individuals(self, evolution_id: str) -> list[dict[str, Any]]:
        """Return the current population/Pareto front.

        For chain experiments synthesises rows from
        ``GET /api/v1/experiments/{id}/results`` because the
        Platform's experiments resource has no
        ``/individuals`` endpoint."""
        if evolution_id.startswith("exp_"):
            try:
                results = self._client.get_results(evolution_id)
            except Exception:
                return []
            return _individuals_from_results(results)
        return self._client.list_individuals(evolution_id)

    def accept_individual(
        self,
        evolution_id: str,
        individual_id: str,
        *,
        memory: Any | None = None,
        chain_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Promote the winning chain to Memory's ``stable`` channel.

        Two code paths:

        * Legacy ``evo_*`` — POST to the Platform's
          ``/api/v1/evolutions/{id}/accept`` endpoint, which
          handles the Memory write server-side.
        * Chain experiment ``exp_*`` — the Platform has no
          accept endpoint here, so CARE does the promotion
          itself: pull ``best_chain_config`` from
          ``/api/v1/experiments/{id}/results``, resolve the
          original ``base_chain_id`` from the experiment's
          description (CARE stamps it at submit time), and
          save a new chain version under that entity pinned
          to the ``stable`` channel via the supplied
          :class:`care.memory.CareMemory` instance.

        Args:
            evolution_id: Run id from :meth:`start_evolution`.
            individual_id: For legacy evolutions, the specific
                individual to pin. Ignored for chain
                experiments — they expose a single
                ``best_chain_config`` only.
            memory: :class:`CareMemory` instance for the
                chain-experiment path. Required when
                ``evolution_id`` starts with ``exp_``; the
                screen passes ``self.app.memory``.

        Returns:
            Response dict carrying ``chain_id`` (the Memory
            entity id) and ``new_version`` (the Memory version
            number of the freshly pinned ``stable`` row) so the
            screen can render the standard "Accepted" toast.

        Raises:
            ValueError: When required inputs are missing
                (chain experiment + no memory facade, or
                ``best_chain_config`` not yet written by the
                runner).
        """
        if evolution_id.startswith("exp_"):
            return self._promote_chain_experiment_winner(
                evolution_id, memory=memory, chain_override=chain_override,
            )
        return self._client.accept_individual(evolution_id, individual_id)

    def _promote_chain_experiment_winner(
        self,
        experiment_id: str,
        *,
        memory: Any | None,
        chain_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """CARE-side accept for chain experiments.

        Pulls the winning chain off the experiment's
        ``/results`` payload and writes it back to Memory as a
        new version of the seed chain, pinned to the
        ``stable`` channel.

        Public surface lives on :meth:`accept_individual` —
        keep this method internal so callers can't accidentally
        skip the routing.
        """
        if memory is None:
            raise ValueError(
                "Chain-experiment accept needs a CareMemory facade "
                "to write the winning chain back. Pass "
                "``memory=self.app.memory`` to ``accept_individual``."
            )
        try:
            results = self._client.get_results(experiment_id)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Couldn't fetch experiment results for "
                f"{experiment_id}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(results, dict):
            raise ValueError(
                f"Platform returned non-dict results for "
                f"{experiment_id} — got {type(results).__name__}.",
            )
        # Promote the user's SELECTED frontier individual when it carries a
        # real chain (the Pareto table now exposes per-generation chains), so
        # the selection isn't silently ignored. Fall back to the overall
        # ``best_chain_config`` when no usable selection was passed.
        if isinstance(chain_override, dict) and chain_override.get("steps"):
            best_chain = chain_override
        else:
            best_chain = results.get("best_chain_config")
        if not isinstance(best_chain, dict) or not best_chain.get("steps"):
            raise ValueError(
                "No best_chain_config available yet — the runner "
                "hasn't written one to MinIO. Wait until the "
                "experiment reports a real generation and try "
                "again.",
            )
        try:
            experiment = self._client.get_experiment(experiment_id)
        except Exception:
            experiment = {}
        base_chain_id = _extract_base_chain_id_from_experiment(experiment)
        if not base_chain_id:
            raise ValueError(
                f"Couldn't locate the seed chain id on experiment "
                f"{experiment_id}. CARE stamps ``base_chain_id=…`` "
                "into the experiment description at submit time; "
                "this run was probably submitted by a different "
                "client. Promote manually via "
                "``CareMemory.save_chain(content, name=…, "
                "channel='stable')``.",
            )
        # Memory's save_chain creates a new version under the
        # same entity_id, channels it to ``stable`` so library
        # callers reading ``stable`` see the promoted winner.
        ref = memory.save_chain(
            best_chain,
            name=str(experiment.get("name") or "CARE evolve winner"),
            entity_id=base_chain_id,
            channel="stable",
        )
        # Project into the legacy accept-response shape so the
        # screen's toast renderer doesn't need to know which
        # path it came from.
        return {
            "chain_id": getattr(ref, "entity_id", base_chain_id),
            "new_version": getattr(ref, "version", None),
            "previous_version": None,
            "channel": "stable",
            "_care_route": "experiments/chains",
            "_raw": {"results": results, "experiment": experiment},
        }

    def cancel(self, evolution_id: str) -> dict[str, Any]:
        """Cancel a running evolution/experiment server-side.

        Routes by id prefix: ``exp_*`` → POST
        ``/api/v1/experiments/{id}/stop``; otherwise the legacy
        ``/api/v1/evolutions/{id}/cancel``."""
        if evolution_id.startswith("exp_"):
            return self._client.stop_experiment(evolution_id)
        return self._client.cancel_evolution(evolution_id)

    def pause(self, evolution_id: str) -> dict[str, Any]:
        """Pause an evolution when the platform supports lifecycle control."""
        return self._client.pause_evolution(evolution_id)

    def resume(self, evolution_id: str) -> dict[str, Any]:
        """Resume a paused evolution when the platform supports lifecycle control."""
        return self._client.resume_evolution(evolution_id)

    def stream_events(self, evolution_id: str) -> Iterator[dict[str, Any]]:
        """Stream events for an evolution/experiment as parsed dicts.

        Three code paths:

        * Legacy ``/api/v1/evolutions/{id}`` ids — delegate to the
          SDK's SSE generator.
        * Chain-experiment ids (``exp_*``) — prefer the Platform's live
          SSE (``GET /api/v1/experiments/{id}/events``, §P4.2) so we
          stream instead of polling + docker-exec. If that endpoint is
          missing (older Platform → 404) or unreachable, fall back to the
          client-side poll loop so CARE keeps working everywhere.
        """
        if evolution_id.startswith("exp_"):
            return self._stream_experiment_events_with_fallback(evolution_id)
        return self._client.stream_events(evolution_id)

    def _stream_experiment_events_with_fallback(
        self, experiment_id: str
    ) -> Iterator[dict[str, Any]]:
        """Try the Platform's experiment SSE; on the first-frame failure
        (endpoint absent / unreachable) fall back to the poll loop."""
        import os

        streamer = getattr(self._client, "stream_experiment_events", None)
        sse_enabled = os.environ.get("CARE_PLATFORM__EXPERIMENT_SSE", "1") != "0"
        if callable(streamer) and sse_enabled:
            try:
                iterator = streamer(experiment_id)
                # Pull the first frame eagerly so a 404 / connection error
                # surfaces here (the SSE generator runs raise_for_status on
                # connect) and we can cleanly fall back to polling.
                first = next(iterator)
            except StopIteration:
                return
            except Exception:
                # Endpoint missing / unreachable — fall through to polling.
                pass
            else:
                yield first
                yield from iterator
                return
        yield from self._poll_experiment_events(experiment_id)

    def _fetch_experiment_list(self) -> list[dict[str, Any]]:
        """Call ``GET /api/v1/experiments/`` (note the trailing
        slash — FastAPI 307s the no-slash variant, and the SDK's
        httpx client doesn't follow redirects). Fall back to the
        SDK's typed method when our direct httpx attempt fails."""
        try:
            http = self._client._http  # type: ignore[attr-defined]
            resp = http.get("/api/v1/experiments/")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception as exc:
            _log.warning(
                "GET /api/v1/experiments/ failed: %s",
                exc,
            )
        try:
            raw = self._client.list_experiments()
            if isinstance(raw, list):
                return [d for d in raw if isinstance(d, dict)]
        except Exception:
            pass
        return []

    def _get_experiment_as_evolution(self, experiment_id: str) -> dict[str, Any]:
        """Fetch chain-experiment state and project it into the
        legacy evolution payload shape the EvolutionScreen +
        Dashboard already consume."""
        try:
            experiment = self._client.get_experiment(experiment_id)
        except Exception:
            experiment = {}
        try:
            from care.runtime.evolution_chain_templates import (
                maybe_sync_chain_templates,
            )

            maybe_sync_chain_templates(
                experiment_id,
                experiment=experiment if isinstance(experiment, dict) else None,
            )
        except Exception:
            pass
        try:
            status_payload = self._client.get_status(experiment_id)
        except Exception:
            status_payload = {}
        try:
            results = self._client.get_results(experiment_id)
        except Exception:
            results = {}

        out: dict[str, Any] = {
            "evolution_id": experiment_id,
            "id": experiment_id,
            "status": (status_payload.get("status") or experiment.get("status") or "unknown"),
            "name": experiment.get("name"),
            "created_at": experiment.get("created_at"),
            # Surface ``started_at`` (Platform-reported wall clock
            # the runner actually began execution at) so the
            # observe-mode EvolutionScreen can show real elapsed
            # time instead of monotonic-since-mount.
            "started_at": (
                experiment.get("started_at")
                or status_payload.get("started_at")
                or experiment.get("created_at")
            ),
            "updated_at": status_payload.get("updated_at") or experiment.get("updated_at"),
            "error_message": status_payload.get("error_message"),
            "runner_id": status_payload.get("runner_id"),
            # The free-form rubric CARE stamped into the description at
            # submit so observe-mode can show "what's being optimised".
            "validation_rubric": _extract_validation_rubric(
                experiment.get("description")
            ),
            "max_iterations": _max_iterations_from_experiment(experiment),
            "individuals": _individuals_from_results(results),
            "generation": _resolve_display_generation(
                _generation_from_results(results),
                (
                    _probe_live_generation(experiment_id)
                    if experiment_id.startswith("exp_")
                    else None
                ),
                max_iterations=_max_iterations_from_experiment(experiment),
            ),
            "best_fitness": _best_fitness_from_results(results),
            "pareto_front": _pareto_front_from_results(results),
            "_care_route": "experiments/chains",
            "_raw": {
                "experiment": experiment,
                "status": status_payload,
                "results": results,
            },
        }
        return out

    @staticmethod
    def _experiment_to_evolution_row(exp: dict[str, Any]) -> dict[str, Any]:
        """Project a list-experiments row into the legacy
        list-evolutions row shape so the Dashboard can render
        chain experiments alongside evolutions without per-row
        type checks.

        Three fields the Platform's bare list payload doesn't
        carry but the dashboard needs are derived here:

        * ``base_chain_id`` — parsed out of the experiment's
          description stamp (CARE writes
          ``base_chain_id=<uuid>`` there at submit time).
        * ``generation`` / ``best_fitness`` — pulled from the
          embedded ``best_result`` blob the Platform writes
          alongside each row. Older runs without this field
          fall back to ``None`` so the dashboard prints "—"
          for them.
        * ``started_at`` — mirrors ``created_at`` because the
          dashboard's "Started" column reads ``started_at``;
          the list payload only has ``created_at``.
        """
        best_result = exp.get("best_result") or {}
        metrics = exp.get("metrics") or {}
        generation = (
            best_result.get("generation")
            if isinstance(best_result, dict) else None
        )
        if generation is None:
            generation = metrics.get("generation") if isinstance(metrics, dict) else None
        best_fitness = (
            best_result.get("fitness")
            if isinstance(best_result, dict) else None
        )
        if best_fitness is None and isinstance(metrics, dict):
            best_fitness = (
                metrics.get("best_fitness")
                or metrics.get("fitness")
            )
        return {
            "evolution_id": exp.get("id") or exp.get("experiment_id") or "",
            "id": exp.get("id") or exp.get("experiment_id") or "",
            "name": exp.get("name") or "",
            "status": exp.get("status") or "unknown",
            "base_chain_id": _extract_base_chain_id_from_experiment(exp) or "",
            "generation": generation,
            "best_fitness": best_fitness,
            "created_at": exp.get("created_at") or "",
            "started_at": exp.get("started_at") or exp.get("created_at") or "",
            "updated_at": exp.get("updated_at") or "",
            "completed_at": exp.get("completed_at") or "",
            "tags": tuple(exp.get("tags") or ()),
            "_care_route": "experiments/chains",
        }

    def _poll_experiment_events(
        self,
        experiment_id: str,
        *,
        interval: float = 2.0,
        max_wait: float = 60 * 60,
    ) -> Iterator[dict[str, Any]]:
        """Poll ``/api/v1/experiments/{id}/status`` and synthesise
        events compatible with the CLI's ``stream_events`` consumer.

        Yields:

        * ``{event: "status", data: <raw status payload>}`` whenever
          the upstream ``status`` field changes.
        * ``{event: "completed"|"failed"|"cancelled", data: ...}``
          when the experiment reaches a terminal state.
        """
        import time

        deadline = time.monotonic() + max_wait
        last_status: str | None = None
        last_generation: int = -1
        last_best_fitness: float | None = None
        last_tokens: float = 0.0
        max_iterations: int | None = None
        TERMINAL = {"completed", "failed", "cancelled", "error"}
        while time.monotonic() < deadline:
            if (
                max_iterations is None
                and experiment_id.startswith("exp_")
            ):
                try:
                    experiment = self._client.get_experiment(experiment_id)
                except Exception:
                    experiment = None
                if isinstance(experiment, dict):
                    max_iterations = _max_iterations_from_experiment(
                        experiment,
                    )
            try:
                payload = self._client.get_status(experiment_id)
            except Exception:
                time.sleep(interval)
                continue
            if not isinstance(payload, dict):
                time.sleep(interval)
                continue
            status = str(payload.get("status") or "").lower()
            probed_gen = (
                _probe_live_generation(experiment_id)
                if experiment_id.startswith("exp_")
                else None
            )
            display_gen = _resolve_display_generation(
                None, probed_gen, max_iterations=max_iterations,
            )
            if status and status != last_status:
                status_data = dict(payload)
                if probed_gen is not None:
                    status_data["generation"] = display_gen
                yield {"event": "status", "data": status_data}
                last_status = status
                if display_gen > last_generation:
                    yield {
                        "event": "generation_started",
                        "data": {
                            "generation": display_gen,
                            "experiment_id": experiment_id,
                        },
                    }
                    last_generation = display_gen
            elif display_gen > last_generation:
                yield {
                    "event": "status",
                    "data": {
                        **payload,
                        "generation": display_gen,
                    },
                }
                yield {
                    "event": "generation_started",
                    "data": {
                        "generation": display_gen,
                        "experiment_id": experiment_id,
                    },
                }
                last_generation = display_gen

            probed_fitness = (
                _probe_live_best_fitness(experiment_id)
                if experiment_id.startswith("exp_")
                else None
            )
            if (
                probed_fitness is not None
                and (
                    last_best_fitness is None
                    or probed_fitness > last_best_fitness + 1e-12
                )
            ):
                yield {
                    "event": "best_updated",
                    "data": {
                        "best_fitness": probed_fitness,
                        "generation": display_gen,
                        "experiment_id": experiment_id,
                    },
                }
                last_best_fitness = probed_fitness

            # Also poll the richer ``/results`` payload — on EVERY poll,
            # including the terminal one, so the final (richest, post-persist)
            # generation isn't dropped when a run completes between polls. The
            # endpoint 404s before the runner writes its first report — that's
            # fine, we just keep polling status until results show up.
            results_history_present = False
            results_programs_present = False
            try:
                results = self._client.get_results(experiment_id)
            except Exception:
                results = None
            if isinstance(results, dict):
                platform_gen = _generation_from_results(results)
                gen = _resolve_display_generation(
                    platform_gen,
                    probed_gen,
                    max_iterations=max_iterations,
                )
                best = _best_fitness_from_results(results)
                if probed_fitness is not None:
                    best = (
                        probed_fitness
                        if best is None
                        else max(best, probed_fitness)
                    )
                metrics = (
                    results.get("metrics")
                    if isinstance(results.get("metrics"), dict)
                    else {}
                )
                current_fitness = metrics.get("current_fitness")
                programs_valid = metrics.get("programs_valid")
                programs_invalid = metrics.get("programs_invalid")
                history = metrics.get("fitness_history")
                frontier_programs = metrics.get("frontier_programs")
                if isinstance(programs_valid, int) or isinstance(
                    programs_invalid, int
                ):
                    results_programs_present = True
                # Surface the full series whenever it's
                # present so the screen's fitness plot can
                # render a real line — without this only
                # the latest "best_updated" point would
                # feed the tracker, leaving the plot blank
                # for runs whose best stays flat.
                if isinstance(history, list) and history:
                    results_history_present = True
                    history_source = "platform"
                    if _platform_fitness_history_looks_bogus(
                        history, max_iterations=max_iterations,
                    ):
                        rebuilt = _fitness_history_best_per_generation(
                            experiment_id,
                        )
                        if rebuilt:
                            history = rebuilt
                            history_source = "redis_probe"
                    yield {
                        "event": "fitness_history_snapshot",
                        "data": {
                            "history": history,
                            "experiment_id": experiment_id,
                            "source": history_source,
                        },
                    }
                # Surface the per-generation frontier
                # programs so the Versions tab can show
                # real chain content + mutation rationale
                # instead of "(not exposed)" placeholders.
                if isinstance(frontier_programs, list) and frontier_programs:
                    yield {
                        "event": "frontier_programs_snapshot",
                        "data": {
                            "frontier": frontier_programs,
                            "experiment_id": experiment_id,
                        },
                    }
                if gen > last_generation:
                    yield {
                        "event": "generation_started",
                        "data": {
                            "generation": gen,
                            "experiment_id": experiment_id,
                            "results": results,
                            "current_fitness": current_fitness,
                            "programs_valid": programs_valid,
                            "programs_invalid": programs_invalid,
                        },
                    }
                    last_generation = gen
                if (
                    best is not None
                    and (
                        last_best_fitness is None
                        or best > last_best_fitness + 1e-12
                    )
                ):
                    yield {
                        "event": "best_updated",
                        "data": {
                            "best_fitness": best,
                            "generation": gen,
                            "experiment_id": experiment_id,
                            "individuals": _individuals_from_results(results),
                            "pareto_front": _pareto_front_from_results(results),
                            "current_fitness": current_fitness,
                            "programs_valid": programs_valid,
                            "programs_invalid": programs_invalid,
                        },
                    }
                    last_best_fitness = best

                # Cumulative token spend → emit the delta as a cost_tick
                # so the EvolutionScreen's cost meter fills (CARE's
                # _accumulate_cost is additive, so cumulative would
                # double-count).
                total_tokens = metrics.get("total_tokens")
                if (
                    isinstance(total_tokens, (int, float))
                    and not isinstance(total_tokens, bool)
                    and total_tokens > last_tokens
                ):
                    yield {
                        "event": "cost_tick",
                        "data": {
                            "total_tokens": total_tokens - last_tokens,
                            "experiment_id": experiment_id,
                        },
                    }
                    last_tokens = total_tokens

            # Redis fallback for local stacks: when the Platform's
            # ``/results`` carries no live metrics yet, read gigavolve
            # Redis directly so the fitness curve + Programs chart
            # aren't empty. ``probe_fitness_history`` was previously
            # dead code — this is the only caller. Gated to ``exp_*``
            # chain experiments (the only ids the probe understands),
            # skipped when ``/results`` already supplied the data, and
            # skipped on terminal status (the runner container — which the
            # probe shells into — is gone once the run finishes).
            if status not in TERMINAL and experiment_id.startswith("exp_"):
                if not results_history_present:
                    probed_history = _probe_live_fitness_history(
                        experiment_id
                    )
                    if probed_history:
                        yield {
                            "event": "fitness_history_snapshot",
                            "data": {
                                "history": probed_history,
                                "experiment_id": experiment_id,
                                "source": "redis_probe",
                            },
                        }
                if not results_programs_present:
                    pv, pi = _probe_live_programs_counts(experiment_id)
                    if pv is not None or pi is not None:
                        yield {
                            "event": "programs_snapshot",
                            "data": {
                                "programs_valid": pv if pv is not None else -1,
                                "programs_invalid": pi if pi is not None else -1,
                                "experiment_id": experiment_id,
                                "source": "redis_probe",
                            },
                        }

            if status in TERMINAL:
                kind = "completed" if status == "completed" else (
                    "cancelled" if status == "cancelled" else "failed"
                )
                yield {"event": kind, "data": payload}
                return
            time.sleep(interval)


def _extract_validation_rubric(description: Any) -> str | None:
    """Pull the free-form rubric back out of an experiment description.

    Mirror of the stamp written in ``start_evolution``: the block after
    ``Validation rubric (user intent):``. Returns ``None`` when the
    description is missing or carries no rubric block."""
    if not isinstance(description, str) or not description:
        return None
    marker = "Validation rubric (user intent):"
    idx = description.find(marker)
    if idx == -1:
        return None
    rubric = description[idx + len(marker):].strip()
    return rubric or None


def _resolve_display_generation(
    platform_gen: int | None,
    probed_gen: int | None,
    *,
    max_iterations: int | None = None,
) -> int:
    """Return a 0-based GA generation suitable for ``gen: N / max`` UI.

    Older Platform master-api builds label Redis scheduler sequence
    numbers (metric list ``s``) as ``generation``, which grows far
    past ``max_iterations``.  Lineage-based Redis probes are authoritative
    when available.
    """
    pg = 0 if platform_gen is None else int(platform_gen)
    platform_bogus = (
        max_iterations is not None
        and max_iterations > 0
        and pg > max_iterations
    )
    if probed_gen is not None and platform_bogus:
        gen = int(probed_gen)
        if max_iterations is not None and max_iterations > 0:
            return min(gen, max_iterations)
        return gen
    if probed_gen is not None and int(probed_gen) > pg:
        pg = int(probed_gen)
    if platform_bogus and max_iterations is not None and max_iterations > 0:
        return min(pg, max_iterations)
    return pg


def _platform_fitness_history_looks_bogus(
    history: list[dict[str, Any]],
    *,
    max_iterations: int | None,
) -> bool:
    """True when Platform history uses scheduler ticks as generation ids."""
    if not history or not max_iterations or max_iterations <= 0:
        return False
    for row in history:
        if not isinstance(row, dict):
            continue
        gen = row.get("generation")
        if isinstance(gen, (int, float)) and int(gen) > max_iterations:
            return True
    return False


def _fitness_history_best_per_generation(
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Collapse lineage probe rows to one improving point per GA generation."""
    buckets: dict[int, float] = {}
    worst = float("-inf")
    for row in _probe_live_fitness_history(experiment_id):
        if not isinstance(row, dict):
            continue
        gen = row.get("generation")
        bf = row.get("best_fitness")
        if not isinstance(gen, int) or not isinstance(bf, (int, float)):
            continue
        if float(bf) <= -999:
            continue
        prev = buckets.get(gen, worst)
        buckets[gen] = max(prev, float(bf))
    return [
        {
            "generation": g,
            "best_fitness": v,
            "current_fitness": v,
        }
        for g, v in sorted(buckets.items())
        if v > worst
    ]


def _probe_live_generation(experiment_id: str) -> int | None:
    """Best-effort GA generation for ``exp_*`` chain experiments."""
    from care.runtime.evolution_redis_probe import probe_ga_generation

    return probe_ga_generation(experiment_id)


def _probe_live_best_fitness(experiment_id: str) -> float | None:
    """Best valid fitness from gigavolve Redis when ``/results`` is empty."""
    from care.runtime.evolution_redis_probe import probe_best_fitness

    return probe_best_fitness(experiment_id)


def _probe_live_fitness_history(experiment_id: str) -> list[dict[str, Any]]:
    """Per-generation fitness series from gigavolve Redis when ``/results``
    carries no ``fitness_history`` — fills the EvolutionScreen plot on
    local stacks instead of leaving it blank."""
    from care.runtime.evolution_redis_probe import probe_fitness_history

    return probe_fitness_history(experiment_id)


def _probe_live_programs_counts(
    experiment_id: str,
) -> tuple[int | None, int | None]:
    """Latest valid/invalid program counts from gigavolve Redis when
    ``/results`` doesn't report them — fills the Programs chart."""
    from care.runtime.evolution_redis_probe import probe_programs_counts

    return probe_programs_counts(experiment_id)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ensure_chain_version(chain: dict[str, Any]) -> dict[str, Any]:
    """Stamp ``version`` if missing. MAGE-generated chains don't
    set one; the Platform rejects them with 400 ``missing required
    field 'version'``. We default to ``"1.0"`` so submissions work
    without making the caller think about the field.
    """
    if "version" in chain and chain["version"]:
        return chain
    out = dict(chain)
    out["version"] = "1.0"
    return out


_BASE_CHAIN_ID_RE = re.compile(
    r"base_chain_id\s*=\s*([0-9a-fA-F-]{8,})",
)


def _extract_base_chain_id_from_experiment(
    experiment: dict[str, Any] | None,
) -> str | None:
    """Pull the seed chain entity id out of a chain experiment.

    CARE stamps ``base_chain_id=<entity-id>`` into the experiment's
    description at submit time so the round-trip survives even
    when no other side channel is available. This regexes it
    back out. Returns ``None`` when the experiment carries no
    description or the stamp is missing (run was submitted by
    something other than CARE).
    """
    if not isinstance(experiment, dict):
        return None
    candidates: list[str] = []
    desc = experiment.get("description")
    if isinstance(desc, str):
        candidates.append(desc)
    cfg = experiment.get("config")
    if isinstance(cfg, dict):
        nested_desc = cfg.get("description")
        if isinstance(nested_desc, str):
            candidates.append(nested_desc)
    for text in candidates:
        m = _BASE_CHAIN_ID_RE.search(text)
        if m:
            return m.group(1)
    return None


def _individuals_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the per-individual list out of a chain-experiment
    ``/results`` payload.

    Platform's results payload is intentionally schema-light —
    the runner writes whatever its evolution-report serializer
    produced. We accept several shapes so we don't break if the
    runner version drifts:

    * ``{"individuals": [...]}`` — explicit list, ideal.
    * ``{"population": [...]}`` — older runner convention.
    * ``{"best": [...]}`` — only the hall-of-fame is available
      mid-run; treat each as an individual row.
    * Anything else — empty list (UI shows "no individuals yet"
      instead of crashing).
    """
    if not isinstance(results, dict):
        return []
    for key in ("individuals", "population", "best", "pareto", "pareto_front"):
        candidate = results.get(key)
        if isinstance(candidate, list):
            return [c for c in candidate if isinstance(c, dict)]
    return []


def _max_iterations_from_experiment(experiment: dict[str, Any]) -> int | None:
    """Read configured GA generation limit from experiment config."""
    if not isinstance(experiment, dict):
        return None
    config = experiment.get("config")
    if not isinstance(config, dict):
        return None
    raw = config.get("max_iterations")
    if isinstance(raw, (int, float)) and raw >= 1:
        return int(raw)
    return None


def _generation_from_results(results: dict[str, Any]) -> int:
    """Best-effort current generation number from a ``/results`` payload.

    Looks at both top-level and ``metrics`` nesting because the
    Platform now exposes the live runner generation via the
    enriched ``metrics`` dict that master_api merges in from
    gigavolve Redis.
    """
    if not isinstance(results, dict):
        return 0
    metrics = results.get("metrics") if isinstance(results.get("metrics"), dict) else {}
    for source in (results, metrics):
        for key in ("generation", "current_generation", "gen"):
            v = source.get(key)
            if isinstance(v, (int, float)) and v >= 0:
                return int(v)
    return 0


def _best_fitness_from_results(results: dict[str, Any]) -> float | None:
    """Best fitness so far. ``None`` when results carry no
    numeric fitness yet (e.g. runner still in initialization
    or every evaluation has scored 0 with ``v: null`` in
    the runner's metrics list)."""
    if not isinstance(results, dict):
        return None
    metrics = results.get("metrics") if isinstance(results.get("metrics"), dict) else {}
    for source in (results, metrics):
        for key in ("best_fitness", "best_score", "best_objective"):
            v = source.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    # Fall back to scanning individuals.
    for ind in _individuals_from_results(results):
        f = ind.get("fitness") or ind.get("score")
        if isinstance(f, (int, float)):
            return float(f)
    return None


def _pareto_front_from_results(
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pareto-front individuals from a ``/results`` payload.

    Multi-objective runs write a dedicated ``pareto_front`` /
    ``pareto`` field; single-objective runs only have a single
    best, so we synthesise a one-entry "front" from
    ``best_fitness`` / ``best_individual_id`` to keep the
    EvolutionScreen's Pareto pane non-empty.
    """
    if not isinstance(results, dict):
        return []
    for key in ("pareto_front", "pareto", "frontier"):
        v = results.get(key)
        if isinstance(v, list):
            return [c for c in v if isinstance(c, dict)]
    best_id = results.get("best_individual_id")
    best_fit = _best_fitness_from_results(results)
    if best_id and best_fit is not None:
        return [{"id": best_id, "fitness": best_fit}]
    return []


def _build_chain_validation_criteria(
    *,
    validation_type: str | None = None,
    continuous_metric: str | None = None,
    binary_method: str | None = None,
    regexp_pattern: str = "",
) -> dict[str, Any]:
    """Project MAESTRO metric choices into Platform ``ChainValidationCriteria``."""
    from care.runtime.evolution_validation import build_chain_validation_criteria

    return build_chain_validation_criteria(
        validation_type=validation_type,
        continuous_metric=continuous_metric,
        binary_method=binary_method,
        regexp_pattern=regexp_pattern,
    )


def _jsonl_to_csv_bytes(path: Path) -> bytes:
    """Convert a JSONL file (one ``{input, expected}`` object per
    line) into CSV bytes the Platform's uploader accepts. Extra
    keys are preserved so callers can ship multi-column datasets
    without inventing a new format.
    """
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rows.append(obj)
        for k in obj:
            if k not in seen:
                seen.add(k)
                columns.append(k)
    if not rows:
        raise ValueError(f"JSONL is empty or unparseable: {path}")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in columns})
    return buf.getvalue().encode("utf-8")


def _merge_care_source_tag(tags: list[str] | None) -> list[str]:
    """Always stamp ``source:care`` so Platform-side dashboards
    can split CARE traffic from other consumers. Dedup against
    user-supplied tags."""
    out: list[str] = [CARE_SOURCE_TAG]
    for tag in tags or []:
        if tag and tag not in out:
            out.append(tag)
    return out


__all__ = [
    "CARE_SOURCE_TAG",
    "CarePlatform",
    "EvolutionRef",
]
