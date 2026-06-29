"""Tests for the library bundle export/import data layer (TODO §1.3 P3).

The export/import workflow is gated on §1 P0 multi-screen
workflow; this suite pins the contract those modals will bind to.
"""

from __future__ import annotations

import asyncio
import json
import tarfile
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from care.runtime.library_bundle import (
    BundleEntry,
    BundleExportResult,
    BundleImportResult,
    BundleManifest,
    LibraryBundleError,
    export_library_bundle,
    import_library_bundle,
    read_bundle_manifest,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _chain_payload(entity_id: str, display_name: str = "Agent") -> dict:
    return {
        "entity_type": "chain",
        "entity_id": entity_id,
        "version_id": "v-1",
        "channel": "latest",
        "etag": "etag",
        "favourite": False,
        "run_count": 0,
        "last_run_at": None,
        "display_name": display_name,
        "description": "desc",
        "meta": {"name": "agent-name", "tags": ["domain:weather"]},
        "content": {"steps": [{"number": 1, "title": "Step"}]},
        "evolution_meta": None,
    }


def _skill_payload(entity_id: str, name: str = "pdf-extract") -> dict:
    return {
        "entity_type": "agent_skill",
        "entity_id": entity_id,
        "version_id": "v-1",
        "channel": "latest",
        "etag": "etag",
        "meta": {"name": name},
        "content": {
            "name": name,
            "description": "Extract PDF",
            "uri": f"github://anthropic/skill-{name}@v1",
            "sha256": "a" * 64,
        },
    }


class _StubClient:
    def __init__(
        self,
        *,
        chains: dict | None = None,
        skills: dict | None = None,
        chain_exc: Exception | None = None,
        bulk_response: dict | None = None,
        bulk_exc: Exception | None = None,
        bulk_delay: float = 0.0,
    ):
        self.chain_calls: list[dict] = []
        self.skill_calls: list[dict] = []
        self.bulk_calls: list[list] = []
        self._chains = chains or {}
        self._skills = skills or {}
        self._chain_exc = chain_exc
        self._bulk_response = bulk_response or {
            "results": [], "success_count": 0, "error_count": 0,
        }
        self._bulk_exc = bulk_exc
        self._bulk_delay = bulk_delay

    def get_chain_dict(self, entity_id, channel="latest", **_):
        self.chain_calls.append({"entity_id": entity_id, "channel": channel})
        if self._chain_exc:
            raise self._chain_exc
        return self._chains.get(entity_id)

    def get_agent_skill_dict(self, entity_id, channel="latest", **_):
        self.skill_calls.append({"entity_id": entity_id, "channel": channel})
        return self._skills.get(entity_id)

    def bulk_save(self, items, *, stop_on_error=False):
        self.bulk_calls.append(list(items))
        if self._bulk_delay:
            time.sleep(self._bulk_delay)
        if self._bulk_exc:
            raise self._bulk_exc
        # By default echo a success result for each item.
        if not self._bulk_response.get("results"):
            return {
                "results": [
                    {"index": i, "success": True, "entity_ref": {"entity_id": "new"}}
                    for i in range(len(items))
                ],
                "success_count": len(items),
                "error_count": 0,
            }
        return self._bulk_response


class _StubMemory:
    def __init__(self, client):
        self.client = client


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------


class TestModelShape:
    def test_manifest_frozen(self):
        m = BundleManifest()
        with pytest.raises(FrozenInstanceError):
            m.schema_version = 99  # type: ignore[misc]

    def test_entry_frozen(self):
        e = BundleEntry(entity_id="x", file="y")
        with pytest.raises(FrozenInstanceError):
            e.entity_id = "z"  # type: ignore[misc]

    def test_export_result_frozen(self):
        r = BundleExportResult(path=Path("/tmp/x"))
        with pytest.raises(FrozenInstanceError):
            r.chain_count = 1  # type: ignore[misc]

    def test_import_result_frozen(self):
        r = BundleImportResult()
        with pytest.raises(FrozenInstanceError):
            r.imported_count = 1  # type: ignore[misc]

    def test_export_success_predicate(self):
        ok = BundleExportResult(path=Path("/tmp/x"), chain_count=2)
        assert ok.success
        assert ok.total_written == 2
        failed = BundleExportResult(path=Path("/tmp/x"), error="boom")
        assert failed.success is False

    def test_import_success_predicate(self):
        ok = BundleImportResult(imported_count=3)
        assert ok.success
        failed = BundleImportResult(failed_count=1)
        assert failed.success is False
        errored = BundleImportResult(error="boom")
        assert errored.success is False

    def test_manifest_total_entries(self):
        m = BundleManifest(
            chains=(BundleEntry(entity_id="c", file="x.json"),),
            agent_skills=(
                BundleEntry(entity_id="s", file="y.json"),
                BundleEntry(entity_id="s2", file="z.json"),
            ),
        )
        assert m.total_entries == 3


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportLibraryBundle:
    def test_writes_tarball_with_manifest_and_chains(self, tmp_path: Path):
        client = _StubClient(
            chains={
                "c-1": _chain_payload("c-1", "Alpha"),
                "c-2": _chain_payload("c-2", "Beta"),
            }
        )
        memory = _StubMemory(client)
        output = tmp_path / "bundle.tar.gz"

        result = asyncio.run(
            export_library_bundle(memory, ["c-1", "c-2"], output)
        )
        assert result.success
        assert result.chain_count == 2
        assert result.skill_count == 0
        assert result.bytes_written > 0
        assert output.exists()

        # Tarball contents.
        with tarfile.open(output, mode="r:gz") as tar:
            names = tar.getnames()
            assert "manifest.json" in names
            assert "chains/c-1.json" in names
            assert "chains/c-2.json" in names

    def test_manifest_is_readable(self, tmp_path: Path):
        client = _StubClient(chains={"c-1": _chain_payload("c-1", "Alpha")})
        memory = _StubMemory(client)
        output = tmp_path / "bundle.tar.gz"

        asyncio.run(
            export_library_bundle(memory, ["c-1"], output, source_namespace="alice")
        )
        manifest = read_bundle_manifest(output)
        assert manifest.schema_version == 1
        assert manifest.source_namespace == "alice"
        assert len(manifest.chains) == 1
        assert manifest.chains[0].entity_id == "c-1"
        assert manifest.chains[0].display_name == "Alpha"
        assert manifest.created_at  # non-empty ISO

    def test_includes_skills(self, tmp_path: Path):
        client = _StubClient(
            chains={"c-1": _chain_payload("c-1")},
            skills={"s-1": _skill_payload("s-1", "pdf-extract")},
        )
        memory = _StubMemory(client)
        output = tmp_path / "bundle.tar.gz"
        result = asyncio.run(
            export_library_bundle(
                memory, ["c-1"], output,
                skill_entity_ids=["s-1"],
            )
        )
        assert result.skill_count == 1
        manifest = read_bundle_manifest(output)
        assert len(manifest.agent_skills) == 1
        skill_entry = manifest.agent_skills[0]
        assert skill_entry.entity_id == "s-1"
        assert skill_entry.name == "pdf-extract"
        assert skill_entry.sha256 == "a" * 64

    def test_missing_chain_skipped_not_aborted(self, tmp_path: Path):
        client = _StubClient(chains={"c-1": _chain_payload("c-1")})
        memory = _StubMemory(client)
        output = tmp_path / "bundle.tar.gz"
        result = asyncio.run(
            export_library_bundle(memory, ["c-1", "c-missing"], output)
        )
        # 1 chain written, the missing one skipped (not raised).
        assert result.chain_count == 1
        assert "c-missing" in result.skipped_chains

    def test_chain_exception_treated_as_skip(self, tmp_path: Path):
        # `get_chain_dict` raises on one row → that row skipped,
        # the others still land.
        client = _StubClient(
            chains={"c-1": _chain_payload("c-1")},
            chain_exc=RuntimeError("503"),
        )
        memory = _StubMemory(client)
        output = tmp_path / "bundle.tar.gz"
        result = asyncio.run(
            export_library_bundle(memory, ["c-1"], output)
        )
        # Both raise → 0 chains, 1 skipped.
        assert result.chain_count == 0
        assert result.skipped_chains == ("c-1",)

    def test_empty_entity_ids_writes_empty_bundle(self, tmp_path: Path):
        memory = _StubMemory(_StubClient())
        output = tmp_path / "empty.tar.gz"
        result = asyncio.run(export_library_bundle(memory, [], output))
        assert result.success
        assert result.chain_count == 0
        assert result.skill_count == 0
        # Manifest exists + has zero entries.
        manifest = read_bundle_manifest(output)
        assert manifest.total_entries == 0

    def test_missing_client_attr_raises(self, tmp_path: Path):
        with pytest.raises(LibraryBundleError, match="`.client`"):
            asyncio.run(
                export_library_bundle(object(), ["c-1"], tmp_path / "x.tar.gz")
            )

    def test_missing_get_chain_dict_raises(self, tmp_path: Path):
        class _EmptyClient:
            pass

        memory = _StubMemory(_EmptyClient())
        with pytest.raises(LibraryBundleError, match="get_chain_dict"):
            asyncio.run(
                export_library_bundle(
                    memory, ["c-1"], tmp_path / "x.tar.gz",
                )
            )

    def test_missing_get_agent_skill_raises_only_when_skills_requested(
        self, tmp_path: Path,
    ):
        # Chain works, no skills requested → fine.
        client = _StubClient(chains={"c-1": _chain_payload("c-1")})
        # Drop the skill method.
        del client.__dict__  # no instance dict
        # Actually just patch by setting class fallback unreachable.
        # Better: build a custom client without get_agent_skill_dict.

        class _ChainOnlyClient:
            def __init__(self):
                self.chain_calls: list[dict] = []

            def get_chain_dict(self, entity_id, channel="latest", **_):
                self.chain_calls.append({"entity_id": entity_id})
                return _chain_payload(entity_id)

        memory = _StubMemory(_ChainOnlyClient())
        # No skills requested → works.
        result = asyncio.run(
            export_library_bundle(
                memory, ["c-1"], tmp_path / "ok.tar.gz",
            )
        )
        assert result.success
        # Skills requested → raises.
        with pytest.raises(LibraryBundleError, match="get_agent_skill"):
            asyncio.run(
                export_library_bundle(
                    memory, ["c-1"], tmp_path / "x.tar.gz",
                    skill_entity_ids=["s-1"],
                )
            )

    def test_atomic_write_no_leftovers(self, tmp_path: Path):
        client = _StubClient(chains={"c-1": _chain_payload("c-1")})
        memory = _StubMemory(client)
        output = tmp_path / "bundle.tar.gz"
        for _ in range(3):
            asyncio.run(
                export_library_bundle(memory, ["c-1"], output)
            )
        # No `.bundle-*` leftovers.
        leftovers = list(tmp_path.glob(".bundle-*"))
        assert leftovers == []

    def test_creates_parent_dir(self, tmp_path: Path):
        client = _StubClient(chains={"c-1": _chain_payload("c-1")})
        memory = _StubMemory(client)
        output = tmp_path / "nested" / "deeper" / "bundle.tar.gz"
        result = asyncio.run(
            export_library_bundle(memory, ["c-1"], output)
        )
        assert result.success
        assert output.exists()


# ---------------------------------------------------------------------------
# read_bundle_manifest
# ---------------------------------------------------------------------------


class TestReadBundleManifest:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(LibraryBundleError, match="not found"):
            read_bundle_manifest(tmp_path / "nope.tar.gz")

    def test_missing_manifest_inside_raises(self, tmp_path: Path):
        out = tmp_path / "empty.tar.gz"
        with tarfile.open(out, mode="w:gz") as tar:
            # Add a stub file without the manifest.
            info = tarfile.TarInfo(name="random.txt")
            body = b"x"
            info.size = len(body)
            from io import BytesIO

            tar.addfile(info, BytesIO(body))
        with pytest.raises(LibraryBundleError, match="missing manifest"):
            read_bundle_manifest(out)

    def test_malformed_json_raises(self, tmp_path: Path):
        out = tmp_path / "bad.tar.gz"
        with tarfile.open(out, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="manifest.json")
            body = b"not json"
            info.size = len(body)
            from io import BytesIO

            tar.addfile(info, BytesIO(body))
        with pytest.raises(LibraryBundleError, match="JSON"):
            read_bundle_manifest(out)

    def test_unsupported_schema_version_raises(self, tmp_path: Path):
        out = tmp_path / "newer.tar.gz"
        with tarfile.open(out, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="manifest.json")
            body = json.dumps({"schema_version": 999}).encode("utf-8")
            info.size = len(body)
            from io import BytesIO

            tar.addfile(info, BytesIO(body))
        with pytest.raises(LibraryBundleError, match="newer"):
            read_bundle_manifest(out)

    def test_missing_schema_version_raises(self, tmp_path: Path):
        out = tmp_path / "noversion.tar.gz"
        with tarfile.open(out, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="manifest.json")
            body = json.dumps({"chains": []}).encode("utf-8")
            info.size = len(body)
            from io import BytesIO

            tar.addfile(info, BytesIO(body))
        with pytest.raises(LibraryBundleError, match="schema_version"):
            read_bundle_manifest(out)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class TestImportLibraryBundle:
    def _make_bundle(
        self, tmp_path: Path, *, chains: list[dict] | None = None,
    ) -> Path:
        client = _StubClient(
            chains={c["entity_id"]: c for c in (chains or [])}
        )
        memory = _StubMemory(client)
        output = tmp_path / "src.tar.gz"
        asyncio.run(
            export_library_bundle(
                memory, [c["entity_id"] for c in (chains or [])], output,
            )
        )
        return output

    def test_round_trip(self, tmp_path: Path):
        # Export from one stub, import into another, verify the
        # bulk_save call carries the right shape.
        bundle = self._make_bundle(
            tmp_path,
            chains=[_chain_payload("c-1", "Alpha"), _chain_payload("c-2", "Beta")],
        )
        target_client = _StubClient()
        target = _StubMemory(target_client)
        result = asyncio.run(import_library_bundle(target, bundle))
        assert result.success
        assert result.imported_count == 2
        assert result.failed_count == 0
        # bulk_save called once with two items.
        assert len(target_client.bulk_calls) == 1
        items = target_client.bulk_calls[0]
        assert len(items) == 2
        assert {i["entity_type"] for i in items} == {"chain"}

    def test_dry_run_skips_bulk_save(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )
        target_client = _StubClient()
        target = _StubMemory(target_client)
        result = asyncio.run(
            import_library_bundle(target, bundle, dry_run=True)
        )
        assert result.success
        assert result.imported_count == 1
        # bulk_save NOT called.
        assert target_client.bulk_calls == []
        # Manifest available on the result.
        assert result.manifest is not None
        assert len(result.manifest.chains) == 1

    def test_namespace_stamping(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )
        target_client = _StubClient()
        target = _StubMemory(target_client)
        asyncio.run(
            import_library_bundle(target, bundle, namespace="alice")
        )
        items = target_client.bulk_calls[0]
        assert items[0]["meta"]["namespace"] == "alice"

    def test_skip_collision_strips_entity_id(self, tmp_path: Path):
        # Default `on_collision="skip"` removes entity_id so
        # Memory creates fresh rows.
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )
        target_client = _StubClient()
        target = _StubMemory(target_client)
        asyncio.run(import_library_bundle(target, bundle))
        items = target_client.bulk_calls[0]
        assert "entity_id" not in items[0]

    def test_overwrite_keeps_entity_id(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )
        target_client = _StubClient()
        target = _StubMemory(target_client)
        asyncio.run(
            import_library_bundle(target, bundle, on_collision="overwrite")
        )
        items = target_client.bulk_calls[0]
        assert items[0]["entity_id"] == "c-1"

    def test_failed_items_surface(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1"), _chain_payload("c-2")],
        )
        target_client = _StubClient(
            bulk_response={
                "results": [
                    {"index": 0, "success": True},
                    {"index": 1, "success": False, "error": "duplicate name"},
                ],
                "success_count": 1,
                "error_count": 1,
            }
        )
        target = _StubMemory(target_client)
        result = asyncio.run(import_library_bundle(target, bundle))
        assert result.imported_count == 1
        assert result.failed_count == 1
        assert any("duplicate" in f for f in result.failures)

    def test_missing_bulk_save_returns_error(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )

        class _NoBulk:
            pass

        target = _StubMemory(_NoBulk())
        result = asyncio.run(import_library_bundle(target, bundle))
        assert result.success is False
        assert "bulk_save" in (result.error or "")

    def test_missing_bundle_file_returns_error(self, tmp_path: Path):
        target = _StubMemory(_StubClient())
        result = asyncio.run(
            import_library_bundle(target, tmp_path / "nope.tar.gz")
        )
        assert result.success is False
        assert "not found" in (result.error or "")

    def test_bulk_save_exception_wraps(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )
        target_client = _StubClient(bulk_exc=RuntimeError("503"))
        target = _StubMemory(target_client)
        result = asyncio.run(import_library_bundle(target, bundle))
        assert result.success is False
        assert "503" in (result.error or "")

    def test_timeout_wraps(self, tmp_path: Path):
        bundle = self._make_bundle(
            tmp_path, chains=[_chain_payload("c-1")],
        )
        target_client = _StubClient(bulk_delay=0.5)
        target = _StubMemory(target_client)
        result = asyncio.run(
            import_library_bundle(target, bundle, timeout=0.05)
        )
        assert result.success is False
        assert "timed out" in (result.error or "")

    def test_empty_bundle_no_op(self, tmp_path: Path):
        bundle = self._make_bundle(tmp_path, chains=[])
        target_client = _StubClient()
        target = _StubMemory(target_client)
        result = asyncio.run(import_library_bundle(target, bundle))
        # Empty manifest → no bulk_save call, success=True.
        assert result.success
        assert result.imported_count == 0
        assert target_client.bulk_calls == []


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_runtime_re_exports(self):
        from care.runtime import (
            BundleEntry as Entry,
            BundleExportResult as ER,
            BundleImportResult as IR,
            BundleManifest as M,
            LibraryBundleError as Err,
            export_library_bundle as exp,
            import_library_bundle as imp,
            read_bundle_manifest as read,
        )

        assert Entry is BundleEntry
        assert ER is BundleExportResult
        assert IR is BundleImportResult
        assert M is BundleManifest
        assert Err is LibraryBundleError
        assert exp is export_library_bundle
        assert imp is import_library_bundle
        assert read is read_bundle_manifest
