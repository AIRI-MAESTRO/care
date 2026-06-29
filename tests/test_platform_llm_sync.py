"""Tests for gigaevo-platform llm_models.yml sync from MAESTRO config."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from care.runtime.platform_llm_sync import (
    build_llm_models_registry,
    sync_platform_llm_registry,
)


def _cfg(
    *,
    mutation_key: str = "sk-mut",
    validation_key: str = "sk-val",
    mutation_model: str = "provider/mut",
    validation_model: str = "provider/val",
    mutation_url: str = "https://mut.example/v1",
    validation_url: str = "https://val.example/v1",
    mutation_max_tokens: int = 8192,
    validation_max_tokens: int = 2048,
    mage_key: str = "",
    mage_model: str = "mage/model",
    mage_url: str = "https://mage.example/v1",
) -> SimpleNamespace:
    return SimpleNamespace(
        platform=SimpleNamespace(
            mutation_base_url=mutation_url,
            mutation_model=mutation_model,
            mutation_api_key=mutation_key,
            validation_base_url=validation_url,
            validation_model=validation_model,
            validation_api_key=validation_key,
            mutation_max_tokens=mutation_max_tokens,
            validation_max_tokens=validation_max_tokens,
        ),
        mage=SimpleNamespace(
            base_url=mage_url,
            model=mage_model,
            api_key=mage_key,
        ),
    )


class TestBuildRegistry:
    def test_platform_fields_used(self) -> None:
        payload = build_llm_models_registry(_cfg())
        models = {m["id"]: m for m in payload["models"]}
        assert models["care-mutation"]["runtime"]["model"] == "provider/mut"
        assert models["care-validation"]["runtime"]["api_key"] == "sk-val"
        assert models["care-mutation"]["runtime"]["max_tokens"] == 8192
        assert models["care-validation"]["runtime"]["max_tokens"] == 2048

    def test_per_run_override(self) -> None:
        payload = build_llm_models_registry(
            _cfg(),
            mutation_max_tokens=16384,
            validation_max_tokens=4096,
        )
        models = {m["id"]: m for m in payload["models"]}
        assert models["care-mutation"]["runtime"]["max_tokens"] == 16384
        assert models["care-validation"]["runtime"]["max_tokens"] == 4096

    def test_mage_fallback_for_keys_and_urls(self) -> None:
        cfg = _cfg(
            mutation_key="",
            validation_key="",
            mutation_model="",
            validation_model="",
            mutation_url="",
            validation_url="",
            mage_key="sk-mage",
        )
        payload = build_llm_models_registry(cfg)
        mut = payload["models"][0]["runtime"]
        assert mut["api_key"] == "sk-mage"
        assert mut["model"] == "mage/model"
        assert mut["base_url"] == "https://mage.example/v1"

    def test_missing_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="API keys are empty"):
            build_llm_models_registry(_cfg(mutation_key="", validation_key=""))


class TestSync:
    def test_writes_yaml(self, tmp_path: Path) -> None:
        cfg = _cfg()
        result = sync_platform_llm_registry(cfg, platform_dir=tmp_path)
        assert result.wrote is True
        target = tmp_path / "llm_models.yml"
        assert target.is_file()
        text = target.read_text(encoding="utf-8")
        assert "care-mutation" in text
        assert "provider/mut" in text
        assert "sk-mut" in text
        assert "max_tokens: 8192" in text

    def test_override_writes_yaml(self, tmp_path: Path) -> None:
        result = sync_platform_llm_registry(
            _cfg(),
            platform_dir=tmp_path,
            mutation_max_tokens=12288,
        )
        assert result.wrote is True
        text = (tmp_path / "llm_models.yml").read_text(encoding="utf-8")
        assert "max_tokens: 12288" in text

    def test_missing_platform_dir_is_noop(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-platform"
        result = sync_platform_llm_registry(_cfg(), platform_dir=missing)
        assert result.wrote is False
        assert result.path is None


class TestStartEvolutionSync:
    def test_start_evolution_forwards_mutation_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from care.platform import CarePlatform, EvolutionRef

        captured: dict[str, object] = {}

        @staticmethod
        def _fake_sync(
            config: object,
            *,
            strict: bool = False,
            mutation_max_tokens: int | None = None,
            validation_max_tokens: int | None = None,
        ) -> None:
            captured["strict"] = strict
            captured["mutation_max_tokens"] = mutation_max_tokens
            captured["validation_max_tokens"] = validation_max_tokens

        monkeypatch.setattr(
            CarePlatform, "_sync_llm_registry", _fake_sync,
        )
        monkeypatch.setattr(
            "care.config.CareConfig.load",
            lambda: _cfg(),
        )

        class _Client:
            pass

        plat = CarePlatform(_Client())

        def _fake_start(**kw: object) -> EvolutionRef:
            return EvolutionRef(
                evolution_id="exp_test",
                base_chain_id="chain-1",
                status="queued",
                extras={},
            )

        monkeypatch.setattr(plat, "_start_chain_experiment", _fake_start)

        plat.start_evolution(
            base_chain_id="chain-1",
            base_chain_content={"steps": [], "version": 1},
            mutation_max_tokens=16384,
        )
        assert captured["strict"] is True
        assert captured["mutation_max_tokens"] == 16384
