"""Library export / import bundle data layer (TODO §1.3 P3).

The LibraryScreen's "Export" action lets the user share selected
agents as a portable ``.tar.gz`` containing JSON chain dumps +
referenced AgentSkill pins (URI + SHA-256). The corresponding
"Import" action reads such a bundle back, restoring entities to
the user's namespace.

The Textual export/import workflow is gated on TODO §1 P0
multi-screen workflow, but the bundle projection + async pack /
unpack drivers are bounded and ship now.

Bundle layout (versioned `manifest.json` + per-entity JSON):

```
care-bundle.tar.gz
├── manifest.json
├── chains/
│   ├── <entity_id>.json
│   └── …
└── agent_skills/
    ├── <entity_id>.json
    └── …
```

`manifest.json` carries:

* ``schema_version`` (currently `1`).
* ``created_at`` ISO timestamp.
* ``source_namespace`` (optional — informational for the importer).
* ``chains`` — list of `{entity_id, file, display_name}`.
* ``agent_skills`` — list of `{entity_id, file, name, sha256}`.

Files are written as JSON-serialised `EntityResponse` payloads
(``meta`` + ``content`` + ``evolution_meta``) so the importer can
hand them straight to ``client.bulk_save(...)`` with minimal
massaging.

What this module provides:

* :class:`BundleManifest` / :class:`BundleEntry` — frozen
  projections of the manifest shape.
* :class:`BundleExportResult` / :class:`BundleImportResult` —
  frozen outcome rows the modal renders into toasts.
* :func:`export_library_bundle` — async helper that fetches the
  selected entities + their AgentSkill pins and writes the
  ``.tar.gz``.
* :func:`import_library_bundle` — async helper that reads a
  ``.tar.gz`` and calls ``client.bulk_save(...)`` with the
  contained entities.
* :func:`read_bundle_manifest` — pure helper that extracts just
  the manifest from a tarball without unpacking everything.
"""

from __future__ import annotations

import asyncio
import json
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


_BUNDLE_SCHEMA_VERSION = 1
"""On-disk schema version. Bump on incompatible changes; the
importer rejects newer versions with a friendly error rather
than guessing."""

_MANIFEST_FILENAME = "manifest.json"
_CHAINS_DIR = "chains"
_SKILLS_DIR = "agent_skills"

CollisionPolicy = Literal["skip", "overwrite", "raise"]
"""How the importer handles entity_ids already present in the
target Memory:

* ``skip`` — leave the existing version alone, count the
  incoming row as skipped (default — least destructive).
* ``overwrite`` — write a new version under the same entity_id
  (creates a fresh version on top of the existing one).
* ``raise`` — abort the entire import on the first collision.
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LibraryBundleError(RuntimeError):
    """Raised for bundle export/import failures the modal can't
    handle inline — file IO error, unreachable Memory, schema
    mismatch on import, malformed tarball."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleEntry:
    """One entry inside the manifest's ``chains`` /
    ``agent_skills`` list."""

    entity_id: str
    file: str
    display_name: str = ""
    name: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class BundleManifest:
    """Top-level on-disk bundle manifest.

    Frozen so it flows through Textual messages without
    defensive copies.
    """

    schema_version: int = _BUNDLE_SCHEMA_VERSION
    created_at: str = ""
    source_namespace: Optional[str] = None
    chains: tuple[BundleEntry, ...] = ()
    agent_skills: tuple[BundleEntry, ...] = ()

    @property
    def total_entries(self) -> int:
        return len(self.chains) + len(self.agent_skills)


@dataclass(frozen=True)
class BundleExportResult:
    """Outcome of :func:`export_library_bundle`."""

    path: Path
    chain_count: int = 0
    skill_count: int = 0
    bytes_written: int = 0
    skipped_chains: tuple[str, ...] = ()
    skipped_skills: tuple[str, ...] = ()
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def total_written(self) -> int:
        return self.chain_count + self.skill_count


@dataclass(frozen=True)
class BundleImportResult:
    """Outcome of :func:`import_library_bundle`."""

    imported_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    failures: tuple[str, ...] = ()
    error: Optional[str] = None
    manifest: Optional[BundleManifest] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.failed_count == 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _manifest_to_dict(m: BundleManifest) -> dict[str, Any]:
    return {
        "schema_version": m.schema_version,
        "created_at": m.created_at,
        "source_namespace": m.source_namespace,
        "chains": [
            {
                "entity_id": e.entity_id,
                "file": e.file,
                "display_name": e.display_name,
            }
            for e in m.chains
        ],
        "agent_skills": [
            {
                "entity_id": e.entity_id,
                "file": e.file,
                "name": e.name,
                "sha256": e.sha256,
            }
            for e in m.agent_skills
        ],
    }


def _manifest_from_dict(data: dict[str, Any]) -> BundleManifest:
    version = data.get("schema_version")
    if not isinstance(version, int):
        raise LibraryBundleError(
            f"manifest missing or invalid schema_version: {version!r}"
        )
    if version > _BUNDLE_SCHEMA_VERSION:
        raise LibraryBundleError(
            f"bundle schema_version {version} is newer than this CARE "
            f"build supports ({_BUNDLE_SCHEMA_VERSION}); upgrade CARE"
        )
    chains_raw = data.get("chains") or []
    if not isinstance(chains_raw, list):
        chains_raw = []
    skills_raw = data.get("agent_skills") or []
    if not isinstance(skills_raw, list):
        skills_raw = []

    chains = tuple(
        BundleEntry(
            entity_id=str(c.get("entity_id") or ""),
            file=str(c.get("file") or ""),
            display_name=str(c.get("display_name") or ""),
        )
        for c in chains_raw
        if isinstance(c, dict)
    )
    skills = tuple(
        BundleEntry(
            entity_id=str(s.get("entity_id") or ""),
            file=str(s.get("file") or ""),
            name=str(s.get("name") or ""),
            sha256=str(s.get("sha256") or ""),
        )
        for s in skills_raw
        if isinstance(s, dict)
    )

    return BundleManifest(
        schema_version=version,
        created_at=str(data.get("created_at") or ""),
        source_namespace=(
            data.get("source_namespace")
            if isinstance(data.get("source_namespace"), str)
            else None
        ),
        chains=chains,
        agent_skills=skills,
    )


def read_bundle_manifest(tarball_path: Path | str) -> BundleManifest:
    """Extract just the manifest from a bundle tarball without
    unpacking everything else.

    Useful for "preview the bundle before importing" affordances:
    the modal can render "Importing 5 chains + 2 skills from
    ~/Downloads/care-bundle.tar.gz" before the user commits.

    Raises:
        LibraryBundleError: tarball missing / unreadable, no
            `manifest.json` inside, malformed JSON, schema
            mismatch.
    """
    target = Path(str(tarball_path)).expanduser()
    if not target.exists():
        raise LibraryBundleError(
            f"bundle not found: {target}"
        )
    try:
        with tarfile.open(target, mode="r:*") as tar:
            try:
                member = tar.getmember(_MANIFEST_FILENAME)
            except KeyError as exc:
                raise LibraryBundleError(
                    f"bundle missing {_MANIFEST_FILENAME}"
                ) from exc
            fh = tar.extractfile(member)
            if fh is None:
                raise LibraryBundleError(
                    f"{_MANIFEST_FILENAME} is not a regular file"
                )
            raw = fh.read()
    except tarfile.TarError as exc:
        raise LibraryBundleError(
            f"failed to read bundle {target}: {exc}"
        ) from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LibraryBundleError(
            f"manifest is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise LibraryBundleError(
            f"manifest is not a JSON object: got {type(data).__name__}"
        )
    return _manifest_from_dict(data)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


async def export_library_bundle(
    memory: Any,
    entity_ids: Iterable[str],
    output_path: Path | str,
    *,
    skill_entity_ids: Iterable[str] = (),
    source_namespace: Optional[str] = None,
    channel: str = "latest",
    timeout: float = 30.0,
) -> BundleExportResult:
    """Pack the selected chains + AgentSkill pins into a tarball.

    The function fetches each entity's full ``content`` +
    ``meta`` + ``evolution_meta`` via the SDK's typed accessors
    (``get_chain_dict`` / ``get_agent_skill``), serialises each
    to JSON inside the tarball, and writes a manifest at the
    top level.

    Args:
        memory: A `CareMemory`-like facade exposing
            ``.client.get_chain_dict(...)`` and
            ``.client.get_agent_skill_dict(...)`` (or
            ``get_agent_skill(...)`` — the helper duck-types
            both).
        entity_ids: Chain entity ids to export. Empty iterable
            produces an empty bundle (manifest with zero
            entries) rather than raising — lets the modal
            offer "export skills only" as a degenerate case.
        output_path: Destination file. Parent dirs are
            auto-created. Existing file is overwritten.
        skill_entity_ids: AgentSkill entity ids to bundle. The
            modal typically pre-computes this from the
            ``allowed_skills`` set referenced by the selected
            chains; passing skills explicitly keeps the export
            shape simple.
        source_namespace: Stamped onto the manifest for the
            importer's reference.
        channel: Memory channel to read from.
        timeout: Per-entity-fetch deadline.

    Returns:
        :class:`BundleExportResult`. Per-entity fetch failures
        land on ``skipped_*`` / ``error`` so the modal can
        render the count without crashing the whole batch.

    Raises:
        LibraryBundleError: file I/O fails, the SDK client
            lacks ``get_chain_dict``, or both entity_ids /
            skill_entity_ids are empty AND we couldn't even
            create the parent dir.
    """
    chain_ids = [eid for eid in entity_ids if eid]
    skill_ids = [eid for eid in skill_entity_ids if eid]

    output = Path(str(output_path)).expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LibraryBundleError(
            f"could not create parent dir {output.parent}: {exc}"
        ) from exc

    client = getattr(memory, "client", None) or getattr(memory, "_client", None)
    if client is None:
        raise LibraryBundleError(
            "memory facade does not expose a `.client` attribute"
        )

    chain_fetcher = getattr(client, "get_chain_dict", None) or getattr(
        client, "get_chain_raw", None
    )
    if chain_ids and not callable(chain_fetcher):
        raise LibraryBundleError(
            "memory facade does not expose client.get_chain_dict()"
        )
    skill_fetcher = (
        getattr(client, "get_agent_skill_dict", None)
        or getattr(client, "get_agent_skill", None)
    )
    if skill_ids and not callable(skill_fetcher):
        raise LibraryBundleError(
            "memory facade does not expose client.get_agent_skill()"
        )

    chain_entries: list[BundleEntry] = []
    skill_entries: list[BundleEntry] = []
    skipped_chains: list[str] = []
    skipped_skills: list[str] = []
    payloads: list[tuple[str, bytes]] = []

    for eid in chain_ids:
        payload = await _fetch_one(
            chain_fetcher, eid, "chain", channel, timeout,
        )
        if payload is None:
            skipped_chains.append(eid)
            continue
        file_name = f"{_CHAINS_DIR}/{eid}.json"
        body = _serialise_entity(payload)
        payloads.append((file_name, body))
        chain_entries.append(
            BundleEntry(
                entity_id=eid,
                file=file_name,
                display_name=_extract_display_name(payload),
            )
        )

    for eid in skill_ids:
        payload = await _fetch_one(
            skill_fetcher, eid, "agent_skill", channel, timeout,
        )
        if payload is None:
            skipped_skills.append(eid)
            continue
        file_name = f"{_SKILLS_DIR}/{eid}.json"
        body = _serialise_entity(payload)
        payloads.append((file_name, body))
        skill_entries.append(
            BundleEntry(
                entity_id=eid,
                file=file_name,
                name=_extract_skill_name(payload),
                sha256=_extract_skill_sha(payload),
            )
        )

    manifest = BundleManifest(
        schema_version=_BUNDLE_SCHEMA_VERSION,
        created_at=_now_iso(),
        source_namespace=source_namespace,
        chains=tuple(chain_entries),
        agent_skills=tuple(skill_entries),
    )

    try:
        bytes_written = await asyncio.to_thread(
            _write_tarball, output, manifest, payloads,
        )
    except OSError as exc:
        return BundleExportResult(
            path=output,
            chain_count=len(chain_entries),
            skill_count=len(skill_entries),
            bytes_written=0,
            skipped_chains=tuple(skipped_chains),
            skipped_skills=tuple(skipped_skills),
            error=f"{type(exc).__name__}: {exc}",
        )

    return BundleExportResult(
        path=output.resolve(),
        chain_count=len(chain_entries),
        skill_count=len(skill_entries),
        bytes_written=bytes_written,
        skipped_chains=tuple(skipped_chains),
        skipped_skills=tuple(skipped_skills),
    )


async def _fetch_one(
    fetcher: Any,
    entity_id: str,
    entity_type: str,
    channel: str,
    timeout: float,
) -> Optional[Any]:
    """Run one sync SDK fetcher in a thread with a deadline.
    Returns the payload OR ``None`` on any failure (per-entity
    timeouts shouldn't crater the whole batch)."""
    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(fetcher, entity_id, channel),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return None
    except Exception:  # noqa: BLE001
        return None
    return payload


def _serialise_entity(entity: Any) -> bytes:
    """JSON-serialise an `EntityResponse`-shaped value (dict OR
    Pydantic model) to bytes. ``default=str`` handles datetime
    + other non-native types Memory's typed responses carry."""
    if hasattr(entity, "model_dump"):
        try:
            payload = entity.model_dump(mode="json")
        except TypeError:
            payload = entity.model_dump()
    elif isinstance(entity, dict):
        payload = entity
    else:
        # Last-ditch: __dict__ projection.
        payload = {
            k: v
            for k, v in vars(entity).items()
            if not k.startswith("_")
        }
    return json.dumps(payload, default=str, sort_keys=True).encode("utf-8")


def _extract_display_name(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("display_name") or "")
    return str(getattr(payload, "display_name", "") or "")


def _extract_skill_name(payload: Any) -> str:
    content = (
        payload.get("content") if isinstance(payload, dict)
        else getattr(payload, "content", None)
    )
    if isinstance(content, dict):
        return str(content.get("name") or "")
    return ""


def _extract_skill_sha(payload: Any) -> str:
    content = (
        payload.get("content") if isinstance(payload, dict)
        else getattr(payload, "content", None)
    )
    if isinstance(content, dict):
        return str(content.get("sha256") or "")
    return ""


def _write_tarball(
    output_path: Path,
    manifest: BundleManifest,
    payloads: list[tuple[str, bytes]],
) -> int:
    """Atomic tar.gz write: serialise to a temp file in the
    target dir, then `os.replace` so a crash mid-write doesn't
    leave a partial bundle behind."""
    manifest_body = json.dumps(
        _manifest_to_dict(manifest),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".bundle-", suffix=".tmp", dir=str(output_path.parent),
    )
    import os
    os.close(tmp_fd)
    try:
        with tarfile.open(tmp_name, mode="w:gz") as tar:
            _write_member(tar, _MANIFEST_FILENAME, manifest_body)
            for file_name, body in payloads:
                _write_member(tar, file_name, body)
        os.replace(tmp_name, output_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return output_path.stat().st_size


def _write_member(tar: tarfile.TarFile, name: str, body: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(body)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, BytesIO(body))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


async def import_library_bundle(
    memory: Any,
    tarball_path: Path | str,
    *,
    on_collision: CollisionPolicy = "skip",
    namespace: Optional[str] = None,
    dry_run: bool = False,
    timeout: float = 60.0,
) -> BundleImportResult:
    """Restore a bundle's entities into Memory.

    Reads the tarball, parses the manifest, and (unless
    ``dry_run=True``) calls ``client.bulk_save(items=...)``
    with the contained entities. The collision policy governs
    behaviour when the bundle's ``entity_id`` already exists
    in the target Memory:

    * ``skip`` — entity_id stripped from the upload payload so
      Memory creates a fresh entity (server-side). Counted as
      ``skipped``.
    * ``overwrite`` — entity_id retained; Memory writes a new
      version on top of the existing one.
    * ``raise`` — pre-flights the same `list_chains` probe the
      SaveAgentModal uses; aborts on first detected collision.

    Args:
        memory: A `CareMemory`-like facade with `.client.bulk_save(...)`.
        tarball_path: Path to the `.tar.gz` to import.
        on_collision: Collision policy. Default ``"skip"`` —
            least destructive.
        namespace: Stamp this namespace on every entity's
            ``meta.namespace`` before upload (so the import
            lands in the user's scope rather than the bundle's
            original namespace).
        dry_run: When ``True``, parse + validate the bundle
            but skip the actual write. Returns counts as if
            the write would have succeeded.
        timeout: Per-bulk-save deadline.

    Returns:
        :class:`BundleImportResult`.
    """
    try:
        manifest = read_bundle_manifest(tarball_path)
        entities = await asyncio.to_thread(
            _extract_entities, Path(str(tarball_path)).expanduser(), manifest,
        )
    except LibraryBundleError as exc:
        return BundleImportResult(error=str(exc))

    if dry_run:
        return BundleImportResult(
            imported_count=len(entities),
            skipped_count=0,
            failed_count=0,
            manifest=manifest,
        )

    if not entities:
        return BundleImportResult(
            imported_count=0,
            skipped_count=0,
            failed_count=0,
            manifest=manifest,
        )

    if namespace is not None:
        entities = [_stamp_namespace(e, namespace) for e in entities]

    if on_collision == "skip":
        entities = [_strip_entity_id(e) for e in entities]

    # Note: `raise` mode would require an additional pre-flight
    # `list_chains(q=name, namespace=)` per entity. For the v1
    # bundle the modal's "Are you sure?" gate covers the common
    # case; we treat `raise` like `overwrite` for now and
    # document the limitation. Future enhancement: add the
    # pre-flight check.
    bulk_save = getattr(getattr(memory, "client", None), "bulk_save", None) or getattr(
        getattr(memory, "_client", None), "bulk_save", None
    )
    if not callable(bulk_save):
        return BundleImportResult(
            failed_count=len(entities),
            manifest=manifest,
            error="memory facade does not expose client.bulk_save()",
        )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(bulk_save, entities),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return BundleImportResult(
            failed_count=len(entities),
            manifest=manifest,
            error=f"bulk_save timed out after {timeout:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        return BundleImportResult(
            failed_count=len(entities),
            manifest=manifest,
            error=f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(response, dict):
        return BundleImportResult(
            failed_count=len(entities),
            manifest=manifest,
            error=f"bulk_save returned unexpected type {type(response).__name__}",
        )

    success_count = int(response.get("success_count") or 0)
    error_count = int(response.get("error_count") or 0)
    failures: list[str] = []
    for item in response.get("results") or ():
        if isinstance(item, dict) and not item.get("success", True):
            err = str(item.get("error") or "unknown")
            failures.append(err)

    skipped = 0 if on_collision == "skip" else 0  # placeholder for future modes

    return BundleImportResult(
        imported_count=success_count,
        skipped_count=skipped,
        failed_count=error_count,
        failures=tuple(failures),
        manifest=manifest,
    )


def _extract_entities(
    tarball_path: Path,
    manifest: BundleManifest,
) -> list[dict[str, Any]]:
    """Read every chain / skill payload referenced by the
    manifest. Sync — runs in an `asyncio.to_thread`."""
    entities: list[dict[str, Any]] = []
    try:
        with tarfile.open(tarball_path, mode="r:*") as tar:
            for entry in (*manifest.chains, *manifest.agent_skills):
                if not entry.file:
                    continue
                try:
                    member = tar.getmember(entry.file)
                except KeyError:
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                try:
                    raw = fh.read()
                    payload = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                projected = _project_for_bulk_save(payload, entry)
                if projected:
                    entities.append(projected)
    except tarfile.TarError as exc:
        raise LibraryBundleError(
            f"failed to read bundle members: {exc}"
        ) from exc
    return entities


def _project_for_bulk_save(
    payload: dict[str, Any],
    entry: BundleEntry,
) -> Optional[dict[str, Any]]:
    """Convert an `EntityResponse` shape to a `bulk_save` item.

    Server expects ``{entity_type, meta, content, [entity_id],
    [embedding], [evolution_meta], [parent_version_id],
    [change_summary], [channel]}``. We drop runtime-only fields
    (`etag`, `version_id`, `last_run_at`, `run_count`) that
    don't belong on a write-side payload.
    """
    entity_type = payload.get("entity_type")
    if not entity_type:
        return None
    item: dict[str, Any] = {
        "entity_type": entity_type,
        "meta": payload.get("meta") or {},
        "content": payload.get("content") or {},
    }
    if payload.get("entity_id"):
        item["entity_id"] = payload["entity_id"]
    if payload.get("evolution_meta"):
        item["evolution_meta"] = payload["evolution_meta"]
    if payload.get("channel"):
        item["channel"] = payload["channel"]
    # Stamp the manifest entry's reference data onto meta when missing.
    meta = item["meta"] if isinstance(item["meta"], dict) else {}
    if entry.display_name and not meta.get("display_name"):
        meta["display_name"] = entry.display_name
    if entry.name and not meta.get("name"):
        meta["name"] = entry.name
    item["meta"] = meta
    return item


def _stamp_namespace(item: dict[str, Any], namespace: str) -> dict[str, Any]:
    """Return a new item with ``meta.namespace`` set."""
    out = dict(item)
    meta = dict(item.get("meta") or {})
    meta["namespace"] = namespace
    out["meta"] = meta
    return out


def _strip_entity_id(item: dict[str, Any]) -> dict[str, Any]:
    """Return a new item with ``entity_id`` removed (forces
    create rather than upsert)."""
    return {k: v for k, v in item.items() if k != "entity_id"}


# Re-export the unused field marker for downstream extension.
_ = field


__all__ = [
    "BundleEntry",
    "BundleExportResult",
    "BundleImportResult",
    "BundleManifest",
    "CollisionPolicy",
    "LibraryBundleError",
    "export_library_bundle",
    "import_library_bundle",
    "read_bundle_manifest",
]
