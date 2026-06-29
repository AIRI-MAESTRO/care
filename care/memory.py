"""CARE-side Memory facade (TODO §3 P0).

`CareMemory` wraps :class:`gigaevo_client.GigaEvoClient` with the
narrow surface CARE actually uses, and centralises CARE-specific
concerns that don't belong in the generic SDK:

- Build :class:`gigaevo_client.CareChainMetadata` from the original
  user query + context-file list, so "Re-run from library" can
  re-prime ``ReasoningContext`` deterministically (§5).
- Stamp a ``domain:{value}`` tag on every saved chain when MAGE
  reports a domain, so library filters work without parsing
  content.
- Return bare ``entity_id`` strings instead of the SDK's
  :class:`EntityRef` so call-sites stay symmetric across entity
  types (mirrors the spec in ``TODO.md §3``).

Construction goes through :meth:`CareMemory.from_config` in normal
operation; the bare constructor is exposed for tests and advanced
callers that already hold a configured ``GigaEvoClient``.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

from gigaevo_client import (
    AgentSkillSpec,
    CareChainMetadata,
    ContextFileRef,
    GigaEvoClient,
    GigaEvoConfig,
)
from gigaevo_client.search_types import SearchType

from care.config import CareConfig


@lru_cache(maxsize=1)
def _gigaevo_save_chain_param_names() -> frozenset[str]:
    """SDK params for ``GigaEvoClient.save_chain`` (varies by wheel version)."""
    return frozenset(inspect.signature(GigaEvoClient.save_chain).parameters)


class CareMemory:
    """Narrow CARE-facing facade over GigaEvo Memory.

    Methods accept the kwargs CARE actually has on hand (originating
    query, domain, MAGE metadata, etc.) and translate them into the
    SDK's lower-level shape. Every save returns the new
    ``entity_id`` so call-sites can store it directly on the CARE
    library row.
    """

    def __init__(self, client: GigaEvoClient):
        """Wrap an already-constructed :class:`GigaEvoClient`.

        Use :meth:`from_config` for the normal CARE startup path —
        this constructor is for tests and dependency injection.
        """
        self._client = client

    @classmethod
    def from_config(cls, config: CareConfig) -> "CareMemory":
        """Construct a CareMemory from a :class:`CareConfig`.

        Reads ``config.memory.base_url`` / ``api_key`` / ``timeout``
        and builds a :class:`GigaEvoConfig` that the SDK consumes
        directly.
        """
        sdk_cfg = GigaEvoConfig(
            memory_base_url=config.memory.base_url,
            api_key=config.memory.api_key,
            timeout=config.memory.timeout,
        )
        client = GigaEvoClient.from_config(sdk_cfg)
        return cls(client)

    @property
    def client(self) -> GigaEvoClient:
        """Escape hatch for callers that need the raw SDK surface."""
        return self._client

    # ------------------------------------------------------------------
    # Chain / agent / step saves
    # ------------------------------------------------------------------

    def save_chain(
        self,
        chain: Any,
        *,
        name: str,
        query: str | None = None,
        domain: str | None = None,
        context_files: list[ContextFileRef] | list[dict[str, Any]] | None = None,
        mage_metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        author: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
        parent_version_id: str | None = None,
        change_summary: str | None = None,
    ) -> str:
        """Save a chain with CARE-style metadata; return ``entity_id``.

        Builds a :class:`CareChainMetadata` from ``query`` +
        ``context_files`` (+ optional ``mage_metadata``) and merges
        it into ``chain.content_json["metadata"]["care"]`` before
        handing off to the SDK.

        Args:
            chain: A CARL ``ReasoningChain`` or raw content dict.
            name: Human-readable library name.
            query: Original user query — saved verbatim so re-runs
                can replay it.
            domain: MAGE-reported domain hint; stamped as a
                ``domain:{value}`` tag when supplied.
            context_files: Optional list of files referenced by the
                originating user task. Accepts either ``ContextFileRef``
                instances or duck-typed dicts.
            mage_metadata: Optional ``MAGEMetadata`` dump (mode,
                stages_completed, etc.) — useful for evolution
                tracking.
            tags: Extra tags. ``domain:{value}`` is prepended when
                ``domain`` is supplied (and not already present).
            when_to_use, author, entity_id, channel: Forwarded to
                :meth:`GigaEvoClient.save_chain`.
            parent_version_id: When updating an existing chain, pin the
                new version's lineage parent to this version (not only the
                channel head). Lets users branch from an older revision.
            change_summary: Short human note stored on the new version.

        Returns:
            The new (or updated) entity's ``entity_id`` as a string.
        """
        resolved_tags = list(tags or [])
        if domain:
            tag = f"domain:{domain}"
            if tag not in resolved_tags:
                resolved_tags.insert(0, tag)

        meta = self._build_care_metadata(
            query=query,
            context_files=context_files,
            mage_metadata=mage_metadata,
            display_name=name,
            description=query,
            tags=tags,
        )
        prepared_chain = self._apply_chain_metadata(chain, meta)
        # Memory's chain validator rejects chains without a top-level
        # ``version`` field (400 "missing required field 'version'").
        # MAGE-generated chains don't stamp one, and the SDK's
        # ``chain_to_content`` (``ReasoningChain.to_dict()``) doesn't emit one
        # either — so BOTH inputs need it. We can't patch the pinned
        # gigaevo-client wheel, so normalise here: serialise a CARL object to
        # its content dict (preserving the metadata just merged in), then
        # default ``version`` on the dict. Passing a dict also means the SDK
        # forwards it verbatim instead of re-serialising.
        if not isinstance(prepared_chain, dict):
            to_dict = getattr(prepared_chain, "to_dict", None)
            if callable(to_dict):
                try:
                    prepared_chain = to_dict()
                except Exception:  # noqa: BLE001 — fall back to the object
                    pass
        if isinstance(prepared_chain, dict) and not prepared_chain.get("version"):
            prepared_chain["version"] = "1.0"

        client_kwargs: dict[str, Any] = {
            "name": name,
            "tags": resolved_tags or None,
            "when_to_use": when_to_use,
            "author": author,
            "entity_id": entity_id,
            "channel": channel,
        }
        sdk_params = _gigaevo_save_chain_param_names()
        if parent_version_id is not None and "parent_version_id" in sdk_params:
            client_kwargs["parent_version_id"] = parent_version_id
        if change_summary is not None and "change_summary" in sdk_params:
            client_kwargs["change_summary"] = change_summary

        ref = self._client.save_chain(prepared_chain, **client_kwargs)
        return ref.entity_id

    def promote_to_stable(self, entity_id: str) -> None:
        """Pin the entity's latest Memory version to stable.

        Memory uses version channels, not CARE review lifecycle
        labels. This method is the canonical CARE promote
        surface and delegates to the SDK's ``promote_to_stable``
        helper when present, falling back to the older generic
        ``promote`` channel method.
        """
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        fn = getattr(self._client, "promote_to_stable", None)
        if callable(fn):
            fn(entity_id)
            return
        promote = getattr(self._client, "promote", None)
        if callable(promote):
            try:
                promote(
                    entity_id,
                    from_channel="latest",
                    to_channel="stable",
                    entity_type="chain",
                )
            except TypeError:
                promote(
                    entity_id,
                    from_channel="latest",
                    to_channel="stable",
                )
            return
        raise NotImplementedError(
            "Memory SDK doesn't expose promote_to_stable / "
            "promote for latest -> stable channel promotion."
        )

    def list_versions(
        self,
        entity_id: str,
        *,
        entity_type: str = "chain",
        limit: int = 20,
    ) -> list[Any]:
        """Version history for an entity (newest first per SDK).

        Thin wrapper over the SDK so the CLI's ``care versions`` shares
        the TUI's data layer."""
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        return list(
            self._client.list_versions(
                entity_id, entity_type=entity_type, limit=limit
            )
        )

    def pin_channel(
        self,
        entity_id: str,
        channel: str,
        version_id: str,
        *,
        entity_type: str = "chain",
    ) -> dict[str, Any]:
        """Repoint ``channel`` at ``version_id`` (rollback / pin).

        Nothing is deleted — channels are pointers, so this is reversible.
        Backs the CLI's ``care rollback``."""
        if not (entity_id and channel and version_id):
            raise ValueError("entity_id, channel and version_id are required")
        return self._client.pin_channel(
            entity_id, channel, version_id, entity_type=entity_type
        )

    def promote(
        self,
        entity_id: str,
        *,
        from_channel: str = "latest",
        to_channel: str = "stable",
        entity_type: str = "chain",
    ) -> dict[str, Any]:
        """Copy one channel pointer to another (e.g. latest → stable).

        Backs the CLI's ``care promote`` (the direct channel move; the
        TUI's gated ``/promote`` adds an interactive baseline/eval gate
        on top)."""
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        return self._client.promote(
            entity_id,
            from_channel=from_channel,
            to_channel=to_channel,
            entity_type=entity_type,
        )

    def delete_entity(
        self,
        entity_id: str,
        *,
        entity_type: str = "chain",
    ) -> bool:
        """Soft-delete an entity via the SDK's per-type delete method.

        Backs the CLI's ``care forget``. Dispatches by ``entity_type``
        because the SDK exposes ``delete_chain`` / ``delete_agent`` /
        … rather than a single generic delete."""
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        method_name = {
            "chain": "delete_chain",
            "step": "delete_step",
            "agent": "delete_agent",
            "agent_skill": "delete_agent_skill",
            "memory_card": "delete_memory_card",
        }.get(entity_type)
        if method_name is None:
            raise ValueError(f"unknown entity_type: {entity_type!r}")
        fn = getattr(self._client, method_name, None)
        if not callable(fn):
            raise NotImplementedError(
                f"Memory SDK doesn't expose {method_name}()"
            )
        return bool(fn(entity_id))

    def set_lifecycle(self, entity_id: str, lifecycle: str) -> None:
        """Compatibility shim for the retired CARE lifecycle API.

        Only ``"stable"`` remains supported, and it maps to
        Memory channel promotion. ``"draft"`` and ``"tested"``
        are CARE-local review labels, not Memory API concepts.
        """
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        if lifecycle != "stable":
            raise ValueError(
                "Memory lifecycle vocabulary was retired; only "
                f"'stable' channel promotion is supported (got {lifecycle!r})"
            )
        self.promote_to_stable(entity_id)

    def save_step_template(
        self,
        step: Any,
        *,
        name: str,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        author: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
    ) -> str:
        """Save a reusable step template; return ``entity_id``."""
        ref = self._client.save_step(
            step,
            name=name,
            tags=tags,
            when_to_use=when_to_use,
            author=author,
            entity_id=entity_id,
            channel=channel,
        )
        return ref.entity_id

    def save_agent(
        self,
        agent_spec: Any,
        *,
        name: str,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        author: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
    ) -> str:
        """Save a generated agent (chain + role + skills composite);
        return ``entity_id``."""
        ref = self._client.save_agent(
            agent_spec,
            name=name,
            tags=tags,
            when_to_use=when_to_use,
            author=author,
            entity_id=entity_id,
            channel=channel,
        )
        return ref.entity_id

    def save_agent_skill(
        self,
        *,
        skill_uri: str,
        manifest: dict[str, Any],
        sha256: str,
        instructions: str = "",
        allowed_tools: list[str] | None = None,
        tarball_url: str | None = None,
        tarball_sha256: str | None = None,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        author: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
    ) -> str:
        """Save an AgentSkill provenance record; return ``entity_id``.

        ``skill_uri`` is the canonical source location
        (``github://owner/repo/skills/foo`` or ``local://...``).
        ``manifest`` is the parsed SKILL.md frontmatter and
        ``sha256`` is the digest of SKILL.md itself (CARE's
        trust-pinning key). ``name`` and ``description`` default to
        the manifest's own ``name`` / ``description`` when omitted —
        explicit overrides win.
        """
        resolved_name = name or manifest.get("name") or skill_uri.rsplit("/", 1)[-1]
        resolved_description = description or manifest.get("description") or ""
        skill = AgentSkillSpec(
            name=resolved_name,
            description=resolved_description,
            uri=skill_uri,
            sha256=sha256,
            manifest=manifest,
            instructions=instructions,
            allowed_tools=list(allowed_tools or manifest.get("allowed-tools") or []),
            tags=list(tags or manifest.get("tags") or []),
            tarball_url=tarball_url,
            tarball_sha256=tarball_sha256,
        )
        ref = self._client.save_agent_skill(
            skill,
            name=resolved_name,
            tags=tags,
            when_to_use=when_to_use,
            author=author,
            entity_id=entity_id,
            channel=channel,
        )
        return ref.entity_id

    def save_memory_card(
        self,
        card_spec: Any,
        *,
        name: str,
        tags: list[str] | None = None,
        when_to_use: str | None = None,
        author: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
    ) -> str:
        """Save a memory_card (run digest, lesson learned, capability
        note); return ``entity_id``."""
        ref = self._client.save_memory_card(
            card_spec,
            name=name,
            tags=tags,
            when_to_use=when_to_use,
            author=author,
            entity_id=entity_id,
            channel=channel,
        )
        return ref.entity_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        entity_type: str | None = None,
        search_type: str = "bm25",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run a memory search; return a list of hit dicts.

        Calls the SDK's generic ``search_hits`` (which returns
        :class:`SearchHit` objects across all entity types) rather
        than ``search`` (which is specialised to memory cards).
        Hits are converted to plain dicts so call-sites don't depend
        on the SDK's response model.

        Args:
            query: User query string.
            entity_type: Optional filter (``chain``, ``agent``,
                ``agent_skill``, ``memory_card``, ``step``). Defaults
                to ``"chain"`` — the most common CARE-side use case.
            search_type: ``"bm25"`` (default), ``"vector"``, or
                ``"hybrid"``.
            top_k: Maximum number of hits to return.
        """
        hits = self._client.search_hits(
            query=query,
            search_type=SearchType(search_type),
            top_k=top_k,
            entity_type=entity_type or "chain",
        )
        return [h.model_dump(mode="json") for h in hits]

    def health_check(self) -> dict[str, Any]:
        """Hit the Memory ``/health`` endpoint and return its JSON
        body. CARE's first-run wizard uses this to validate
        connectivity before persisting the config."""
        return self._client.health_check()

    def mark_favourite(
        self,
        entity_id: str,
        *,
        entity_type: str,
        value: bool = True,
    ) -> dict[str, Any]:
        """Toggle the favourite flag on a library entity.

        Mirrors the SDK's per-kind `mark_*_favourite` helpers but
        dispatches on ``entity_type`` so the CLI can drive any
        kind through one entry-point.

        Args:
            entity_id: Library entity_id.
            entity_type: ``"chain"``, ``"agent"``, ``"agent_skill"``,
                or ``"memory_card"``.
            value: ``True`` (default) stars the entity, ``False``
                unstars it.

        Returns:
            The updated entity as a dict (the SDK's response model
            dumped via ``model_dump``).
        """
        if entity_type not in ("chain", "agent", "agent_skill", "memory_card"):
            raise ValueError(
                f"unsupported entity_type {entity_type!r}; expected "
                "'chain', 'agent', 'agent_skill', or 'memory_card'"
            )
        response = self._client._mark_favourite(
            entity_type, entity_id, value=bool(value),
        )
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response) if isinstance(response, dict) else {}

    def get_chain_lineage(
        self,
        entity_id: str,
        *,
        channel: str = "latest",
        version_id: str | None = None,
        max_depth: int = 10,
    ) -> Any:
        """Walk a chain's ancestry DAG.

        Thin wrapper around the SDK's
        :meth:`gigaevo_client.GigaEvoClient.get_chain_lineage` —
        returns a :class:`gigaevo_client.LineageResponse` carrying
        the BFS-ordered ancestor versions plus the
        ``max_depth_reached`` flag.

        Args:
            entity_id: Chain entity to walk lineage for.
            channel: Start from the version pinned to this channel
                (default ``"latest"``). Ignored when
                ``version_id`` is supplied.
            version_id: Walk from a specific historical version
                instead of the channel head.
            max_depth: Cap on BFS depth (1-100). The server
                clamps to the same range; the response's
                ``max_depth_reached`` flag tells the client whether
                more ancestors exist beyond the cap.
        """
        return self._client.get_chain_lineage(
            entity_id,
            channel=channel,
            version_id=version_id,
            max_depth=max_depth,
        )

    def get_entity(
        self,
        entity_id: str,
        *,
        entity_type: str,
        channel: str = "latest",
    ) -> dict[str, Any]:
        """Fetch the full entity payload by id + type.

        Returns the SDK's underlying ``_get_entity`` response —
        a dict with ``entity_id``, ``version_id``, ``channel``,
        ``etag``, ``meta``, and ``content`` keys. Used by the
        `care memory show` CLI which needs both metadata + body
        in a single hop.

        Unknown entity types raise ``ValueError`` (matches the
        pattern :meth:`list_entities` uses).
        """
        if entity_type not in ("chain", "agent", "agent_skill", "memory_card", "step"):
            raise ValueError(
                f"unsupported entity_type {entity_type!r}; expected "
                "'chain', 'agent', 'agent_skill', 'memory_card', or 'step'"
            )
        return self._client._get_entity(entity_type, entity_id, channel)

    def get_chain(
        self,
        entity_id: str,
        *,
        channel: str = "latest",
    ) -> dict[str, Any]:
        """Fetch a chain's content as a raw dict.

        Thin wrapper around
        :meth:`gigaevo_client.GigaEvoClient.get_chain_dict` —
        returns the chain content (steps + metadata) ready to feed
        into :func:`care.validate_chain` / :func:`care.export_chain`
        or to pass to a future runtime executor.

        Args:
            entity_id: Memory entity_id of the chain.
            channel: Version channel to read (default ``latest``).
        """
        return self._client.get_chain_dict(entity_id, channel=channel)

    def list_entities(
        self,
        *,
        entity_type: str,
        limit: int = 50,
        offset: int = 0,
        channel: str = "latest",
        namespace: str | None = None,
        tags: list[str] | None = None,
        q: str | None = None,
        favourites_only: bool | None = None,
        sort_by: str | None = None,
        sort_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generic listing across the four supported entity types.

        Routes to the right SDK method based on ``entity_type`` and
        returns plain dicts instead of ``EntityResponse`` so call
        sites (the ``care memory ls`` CLI, the LibraryScreen
        prefetch) don't depend on the SDK's response model.

        Args:
            entity_type: ``"chain"`` / ``"agent"`` / ``"agent_skill"`` /
                ``"memory_card"``. Unknown raises ``ValueError``.
            limit / offset / channel: pagination + version channel.
            namespace / tags / q / favourites_only / sort_by / sort_dir:
                Forwarded to the SDK's listing surface where supported.
                Ignored for ``memory_card`` (the SDK's
                ``list_memory_cards`` doesn't accept them).
        """
        if entity_type == "chain":
            rows = self._client.list_chains(
                limit=limit,
                offset=offset,
                channel=channel,
                namespace=namespace,
                tags=tags,
                q=q,
                favourites_only=favourites_only,
                sort_by=sort_by,
                sort_dir=sort_dir,
            )
        elif entity_type == "agent":
            rows = self._client.list_agents(
                limit=limit,
                offset=offset,
                channel=channel,
                namespace=namespace,
                tags=tags,
                q=q,
                favourites_only=favourites_only,
                sort_by=sort_by,
                sort_dir=sort_dir,
            )
        elif entity_type == "agent_skill":
            rows = self._client.list_agent_skills(
                limit=limit,
                offset=offset,
                channel=channel,
                namespace=namespace,
                tags=tags,
                q=q,
                favourites_only=favourites_only,
                sort_by=sort_by,
                sort_dir=sort_dir,
            )
        elif entity_type == "memory_card":
            rows = self._client.list_memory_cards(
                limit=limit,
                offset=offset,
                channel=channel,
            )
        else:
            raise ValueError(
                f"unsupported entity_type {entity_type!r}; expected "
                "'chain', 'agent', 'agent_skill' or 'memory_card'"
            )
        return [
            row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row)
            for row in rows or []
        ]

    def find_entity_by_name(
        self,
        *,
        name: str,
        entity_type: str,
        namespace: str | None = None,
        channel: str = "latest",
        limit: int = 50,
    ) -> dict[str, Any] | None:
        """Look up an entity by exact display name + entity type.

        Used by :func:`care.detect_conflict` as the
        ``find_entity_by_name`` duck-typed lookup it documents.
        Returns ``{"entity_id": str, "content": dict}`` for the
        first listing row whose ``display_name`` (or
        ``content.metadata.care.display_name`` /
        ``meta["name"]`` fallbacks) equals ``name`` exactly, or
        ``None`` if no match is found.

        Entity-type dispatch:
            * ``"chain"`` → ``list_chains(q=name, namespace=..., channel=...)``
            * ``"agent_skill"`` → ``list_agent_skills(q=name, namespace=..., channel=...)``
            * ``"memory_card"`` → ``list_memory_cards(channel=...)`` (no ``q``
              filter on the SDK side, falls back to a full scan up to
              ``limit``).

        Args:
            name: Display name to match exactly.
            entity_type: ``"chain"``, ``"agent_skill"`` or
                ``"memory_card"``.
            namespace: Optional namespace scope. Ignored for
                ``memory_card`` (SDK doesn't expose it on that
                listing).
            channel: Memory version channel to scan.
            limit: Maximum listing rows to walk before giving up.
        """
        if entity_type == "chain":
            rows = self._client.list_chains(
                limit=limit,
                channel=channel,
                q=name,
                namespace=namespace,
            )
        elif entity_type == "agent_skill":
            rows = self._client.list_agent_skills(
                limit=limit,
                channel=channel,
                q=name,
                namespace=namespace,
            )
        elif entity_type == "memory_card":
            rows = self._client.list_memory_cards(
                limit=limit,
                channel=channel,
            )
        else:
            raise ValueError(
                f"unsupported entity_type {entity_type!r}; expected "
                "'chain', 'agent_skill' or 'memory_card'"
            )

        for row in rows or []:
            if self._row_name(row) == name:
                return {
                    "entity_id": getattr(row, "entity_id", "") or "",
                    "content": getattr(row, "content", {}) or {},
                }
        return None

    @staticmethod
    def _row_name(row: Any) -> str | None:
        """Best-effort display-name accessor for an EntityResponse-like row.

        Checks ``display_name``, then ``content.metadata.care.display_name``
        (CARE convention), then ``meta["name"]`` (server-side row label
        the save method pins on creation).
        """
        display = getattr(row, "display_name", None)
        if isinstance(display, str) and display:
            return display
        content = getattr(row, "content", None)
        if isinstance(content, dict):
            care_meta = content.get("metadata", {}).get("care", {})
            if isinstance(care_meta, dict):
                care_name = care_meta.get("display_name")
                if isinstance(care_name, str) and care_name:
                    return care_name
        meta = getattr(row, "meta", None)
        if isinstance(meta, dict):
            raw = meta.get("name")
            if isinstance(raw, str) and raw:
                return raw
        return None

    # ------------------------------------------------------------------
    # Library hot-reload (TODO §3 P1)
    # ------------------------------------------------------------------

    def watch_library(
        self,
        callback,
        *,
        namespace: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        tags: list[str] | None = None,
        event_type: str | None = None,
    ):
        """Subscribe to library mutations as typed
        :class:`care.runtime.LibraryEvent` instances.

        Thin wrapper around :func:`care.runtime.watch_library`
        that defaults the SDK client to the one this facade holds.
        Returns a :class:`care.runtime.LibrarySubscription` — call
        ``.stop()`` or use as a context manager when done.
        """
        # Late import to avoid a circular reference at module load.
        from care.runtime.library_watcher import watch_library

        return watch_library(
            self._client,
            callback,
            namespace=namespace,
            entity_type=entity_type,
            entity_id=entity_id,
            tags=tags,
            event_type=event_type,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_care_metadata(
        *,
        query: str | None,
        context_files: list[ContextFileRef] | list[dict[str, Any]] | None,
        mage_metadata: dict[str, Any] | None,
        display_name: str | None,
        description: str | None,
        tags: list[str] | None,
    ) -> CareChainMetadata:
        normalised_files: list[ContextFileRef] = []
        for entry in context_files or []:
            if isinstance(entry, ContextFileRef):
                normalised_files.append(entry)
            elif isinstance(entry, dict):
                normalised_files.append(ContextFileRef.model_validate(entry))
            else:
                raise TypeError(
                    "context_files must contain ContextFileRef or dict entries; "
                    f"got {type(entry).__name__}"
                )
        return CareChainMetadata(
            task_description=query,
            context_files=normalised_files,
            mage_metadata=mage_metadata,
            display_name=display_name,
            description=description,
            tags=list(tags or []),
        )

    @staticmethod
    def _apply_chain_metadata(chain: Any, meta: CareChainMetadata) -> Any:
        """Return ``chain`` with CARE metadata merged in.

        For a dict, mutate-then-return is safe. For a CARL
        ``ReasoningChain``, mutate its ``.metadata`` directly so the
        SDK's downstream ``chain_to_content`` picks it up. CARE
        avoids importing ``mmar_carl`` here to keep startup light
        and instead duck-types: anything with a settable
        ``metadata`` attribute is treated as a CARL chain.
        """
        if isinstance(chain, dict):
            return meta.merge_into_content(chain)
        existing = getattr(chain, "metadata", None)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(meta.model_dump(exclude_none=True, exclude_defaults=False))
            chain.metadata = merged
        return chain


__all__ = ["CareMemory"]
