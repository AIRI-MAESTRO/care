"""Tests for ``care.runtime.telemetry`` (TODO §9 P3).

Coverage layers:

1. **Event shape** — frozen, auto-stamped ``trace_id`` /
   ``timestamp``, sensible defaults.
2. **NullTelemetrySink** — no-op contract; ``dropped`` counter
   for test introspection.
3. **`build_telemetry_sink` factory** — opt-in gate (off →
   Null), explicit-Null path, unknown backend raises with the
   known-backend list, accepts both full ``CareConfig`` and
   bare ``TelemetryConfig``.
4. **Langfuse missing-dep** — the real "langfuse SDK not
   installed" path (`langfuse` isn't in the dev env).
5. **Custom backend registry** — register / dispatch /
   overwrite / unregister / error wrapping (parallels the
   ``register_provider_factory`` tests).
"""

from __future__ import annotations

import pytest

from care.config import CareConfig, TelemetryConfig
from care.runtime.telemetry import (
    NullTelemetrySink,
    TelemetryEvent,
    TelemetrySinkError,
    build_telemetry_sink,
    list_known_backends,
    register_telemetry_backend,
    unregister_telemetry_backend,
)


# ---------------------------------------------------------------------------
# TelemetryEvent
# ---------------------------------------------------------------------------


class TestTelemetryEvent:
    def test_default_factories_independent(self):
        a = TelemetryEvent(kind="x")
        b = TelemetryEvent(kind="x")
        # Distinct trace ids per event.
        assert a.trace_id != b.trace_id
        # Distinct attribute dicts so mutating one doesn't bleed.
        a.attributes["k"] = "v"
        assert b.attributes == {}

    def test_frozen(self):
        e = TelemetryEvent(kind="x")
        with pytest.raises(Exception):
            e.kind = "y"  # type: ignore[misc]

    def test_explicit_trace_id_preserved(self):
        e = TelemetryEvent(kind="x", trace_id="fixed-id")
        assert e.trace_id == "fixed-id"

    def test_default_timestamp_is_floatlike(self):
        e = TelemetryEvent(kind="x")
        assert isinstance(e.timestamp, float)
        assert e.timestamp > 0


# ---------------------------------------------------------------------------
# NullTelemetrySink
# ---------------------------------------------------------------------------


class TestNullSink:
    def test_record_increments_dropped(self):
        sink = NullTelemetrySink()
        assert sink.dropped == 0
        sink.record(TelemetryEvent(kind="x"))
        sink.record(TelemetryEvent(kind="y"))
        assert sink.dropped == 2

    def test_flush_close_no_op(self):
        sink = NullTelemetrySink()
        sink.flush()  # No exception
        sink.close()  # No exception
        assert sink.dropped == 0


# ---------------------------------------------------------------------------
# build_telemetry_sink — opt-in gate
# ---------------------------------------------------------------------------


class TestBuildSink:
    def test_default_disabled_returns_null(self):
        sink = build_telemetry_sink(CareConfig())
        assert isinstance(sink, NullTelemetrySink)

    def test_enabled_null_backend_returns_null(self):
        cfg = TelemetryConfig(enabled=True, backend="null")
        sink = build_telemetry_sink(cfg)
        assert isinstance(sink, NullTelemetrySink)

    def test_enabled_empty_backend_returns_null(self):
        cfg = TelemetryConfig(enabled=True, backend="")
        sink = build_telemetry_sink(cfg)
        assert isinstance(sink, NullTelemetrySink)

    def test_disabled_with_langfuse_backend_still_null(self):
        # Even if a user typos `backend=langfuse` while
        # `enabled=false`, we don't try to construct a real client.
        cfg = TelemetryConfig(enabled=False, backend="langfuse")
        sink = build_telemetry_sink(cfg)
        assert isinstance(sink, NullTelemetrySink)

    def test_unknown_backend_raises(self):
        cfg = TelemetryConfig(enabled=True, backend="weather")
        with pytest.raises(
            TelemetrySinkError, match="unknown telemetry backend"
        ):
            build_telemetry_sink(cfg)

    def test_error_lists_known_backends(self):
        cfg = TelemetryConfig(enabled=True, backend="nope")
        with pytest.raises(TelemetrySinkError) as exc:
            build_telemetry_sink(cfg)
        msg = str(exc.value)
        assert "null" in msg
        assert "langfuse" in msg

    def test_accepts_full_careconfig(self):
        full = CareConfig(telemetry=TelemetryConfig(enabled=False))
        sink = build_telemetry_sink(full)
        assert isinstance(sink, NullTelemetrySink)


# ---------------------------------------------------------------------------
# Langfuse missing-dep path (real for this dev env)
# ---------------------------------------------------------------------------


def _langfuse_installed() -> bool:
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


class TestLangfuseBackend:
    @pytest.mark.skipif(
        _langfuse_installed(),
        reason="langfuse SDK is installed; missing-dep path skipped",
    )
    def test_missing_langfuse_dep_raises(self):
        cfg = TelemetryConfig(
            enabled=True,
            backend="langfuse",
            public_key="pub",
            secret_key="sec",
        )
        with pytest.raises(
            TelemetrySinkError, match="langfuse SDK is not installed"
        ):
            build_telemetry_sink(cfg)

    def test_missing_keys_raises(self):
        # Even when the SDK is installed, missing creds must
        # surface clearly rather than yielding an SDK-layer crash.
        if not _langfuse_installed():
            pytest.skip("langfuse not installed; skip this path")
        cfg = TelemetryConfig(
            enabled=True, backend="langfuse", public_key=None, secret_key=None
        )
        with pytest.raises(TelemetrySinkError, match="public_key.*secret_key"):
            build_telemetry_sink(cfg)


# ---------------------------------------------------------------------------
# Custom backend registry
# ---------------------------------------------------------------------------


class TestCustomBackend:
    def teardown_method(self):
        unregister_telemetry_backend("custom-stub")
        unregister_telemetry_backend("rises")

    def test_register_and_dispatch(self):
        captured: list[TelemetryConfig] = []

        def factory(cfg: TelemetryConfig):
            captured.append(cfg)
            return NullTelemetrySink()

        register_telemetry_backend("custom-stub", factory)
        cfg = TelemetryConfig(enabled=True, backend="custom-stub")
        sink = build_telemetry_sink(cfg)
        assert isinstance(sink, NullTelemetrySink)
        assert captured == [cfg]

    def test_register_empty_name_raises(self):
        with pytest.raises(TelemetrySinkError, match="non-empty"):
            register_telemetry_backend("", lambda cfg: NullTelemetrySink())

    def test_unregister_returns_bool(self):
        register_telemetry_backend("custom-stub", lambda cfg: NullTelemetrySink())
        assert unregister_telemetry_backend("custom-stub") is True
        assert unregister_telemetry_backend("custom-stub") is False

    def test_register_overwrites_silently(self):
        register_telemetry_backend(
            "custom-stub",
            lambda cfg: NullTelemetrySink(),
        )

        marker = NullTelemetrySink()
        register_telemetry_backend("custom-stub", lambda cfg: marker)
        cfg = TelemetryConfig(enabled=True, backend="custom-stub")
        assert build_telemetry_sink(cfg) is marker

    def test_factory_generic_error_wrapped(self):
        def boom(cfg):
            raise RuntimeError("bad factory")

        register_telemetry_backend("rises", boom)
        cfg = TelemetryConfig(enabled=True, backend="rises")
        with pytest.raises(
            TelemetrySinkError, match="custom telemetry backend 'rises' raised"
        ):
            build_telemetry_sink(cfg)

    def test_factory_raising_telemetry_error_propagates(self):
        def boom(cfg):
            raise TelemetrySinkError("descriptive message")

        register_telemetry_backend("rises", boom)
        cfg = TelemetryConfig(enabled=True, backend="rises")
        with pytest.raises(
            TelemetrySinkError, match="^descriptive message$"
        ):
            build_telemetry_sink(cfg)

    def test_case_insensitive_backend_dispatch(self):
        register_telemetry_backend("custom-stub", lambda cfg: NullTelemetrySink())
        cfg = TelemetryConfig(enabled=True, backend="CUSTOM-STUB")
        sink = build_telemetry_sink(cfg)
        assert isinstance(sink, NullTelemetrySink)


# ---------------------------------------------------------------------------
# list_known_backends
# ---------------------------------------------------------------------------


class TestListKnownBackends:
    def teardown_method(self):
        unregister_telemetry_backend("custom-stub")

    def test_includes_builtins(self):
        names = list_known_backends()
        assert "null" in names
        assert "langfuse" in names

    def test_includes_registered_custom(self):
        register_telemetry_backend("custom-stub", lambda cfg: NullTelemetrySink())
        assert "custom-stub" in list_known_backends()

    def test_returns_sorted(self):
        names = list_known_backends()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Re-export check
# ---------------------------------------------------------------------------


class TestRuntimeReExport:
    def test_runtime_package_exports_telemetry_symbols(self):
        from care.runtime import (
            NullTelemetrySink as ExportedNull,
        )
        from care.runtime import (
            TelemetryEvent as ExportedEvent,
        )
        from care.runtime import (
            build_telemetry_sink as exported_factory,
        )
        assert ExportedNull is NullTelemetrySink
        assert ExportedEvent is TelemetryEvent
        assert exported_factory is build_telemetry_sink
