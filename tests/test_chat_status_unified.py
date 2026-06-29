"""`/status` shares care doctor's run_all_probes (TODO usability unify)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import care.screens.chat as chat


class _FakeReport:
    def format_text(self) -> str:
        return "✓ mage: ok\n✓ memory: ok\n· platform: skipped"


def test_status_worker_uses_run_all_probes_deep():
    posted: list = []
    captured: dict = {}

    async def _fake_probes(config, **kwargs):
        captured.update(kwargs)
        return _FakeReport()

    screen = SimpleNamespace(
        app=SimpleNamespace(config=object(), memory=object(), platform=None),
        _post_line=lambda role, text, **kw: posted.append((role, text)),
    )

    with patch("care.first_run.run_all_probes", _fake_probes):
        asyncio.run(chat._status_worker(screen))

    # Shared report + deep MAGE probe.
    assert any("care doctor" in text for _, text in posted)
    assert any("memory: ok" in text for _, text in posted)
    assert captured.get("deep") is True
    # App's existing facades injected (memory present, platform None).
    assert captured.get("memory_factory") is not None
    assert captured.get("platform_factory") is None


def test_status_worker_no_config_warns():
    posted: list = []
    screen = SimpleNamespace(
        app=SimpleNamespace(config=None),
        _post_line=lambda role, text, **kw: posted.append((role, text, kw)),
    )
    asyncio.run(chat._status_worker(screen))
    assert posted and "app.config" in posted[0][1]
