"""`CARE_CHAT__DEFAULT_MODE` normalization through `CareConfig.load`.

Regression: after the ``ad_hoc`` → ``interactive`` rename, the load-time
``_CHAT_MODE_ALIASES`` normalizer still mapped to the old canonical
``ad_hoc`` and rejected ``interactive`` (the current canonical mode),
warning and silently dropping it. The alias map must mirror
``care.screens.chat.normalise_mode``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from care.config import _CHAT_MODE_ALIASES, CareConfig

_NO_TOML = Path("/no-such-config.toml")


def _load_mode(value: str) -> str:
    cfg = CareConfig.load(path=_NO_TOML, env={"CARE_CHAT__DEFAULT_MODE": value})
    return cfg.chat.default_mode


class TestChatModeNormalization:
    def test_interactive_is_accepted(self):
        # The canonical current mode must NOT be rejected/dropped.
        assert _load_mode("interactive") == "interactive"

    @pytest.mark.parametrize("legacy", ["ad_hoc", "ad-hoc", "adhoc", "AD_HOC"])
    def test_legacy_ad_hoc_maps_to_interactive(self, legacy):
        assert _load_mode(legacy) == "interactive"

    @pytest.mark.parametrize("prod", ["production", "prod", "PROD"])
    def test_production_aliases(self, prod):
        assert _load_mode(prod) == "production"

    def test_unknown_falls_back_to_default(self):
        # Dropped → Pydantic field default (interactive) applies.
        assert _load_mode("bogus") == "interactive"


def test_alias_map_only_yields_canonical_modes():
    assert set(_CHAT_MODE_ALIASES.values()) == {"interactive", "production"}
