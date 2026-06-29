"""Push MAESTRO LLM credentials into gigaevo-platform ``llm_models.yml``.

Platform evolution runs do not read MAESTRO's live ``[mage]`` client at
submit time. The runner resolves ``care-mutation`` / ``care-validation``
from the bind-mounted ``/llm_models.yml``. This module keeps that file
aligned with CARE config:

* ``[platform].mutation_*`` / ``validation_*`` when set
* otherwise ``[mage].base_url`` / ``model`` / ``api_key`` (keystore-resolved
  by :meth:`CareConfig.load`)

Call :func:`sync_platform_llm_registry` after Settings save and immediately
before :meth:`CarePlatform.start_evolution`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlatformLlmSyncResult:
    """Outcome of a registry sync attempt."""

    path: Path | None
    wrote: bool
    mutation_model: str
    mutation_base_url: str
    validation_model: str
    validation_base_url: str
    message: str


def _resolve_secret(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip()


def _platform_llm_fields(cfg: Any) -> tuple[str, str, str, str, str, str]:
    """Return mutation + validation (url, model, key) with mage fallback."""
    plat = cfg.platform
    mage = cfg.mage

    mutation_url = (plat.mutation_base_url or mage.base_url or "").strip()
    mutation_model = (plat.mutation_model or mage.model or "").strip()
    mutation_key = (
        _resolve_secret(plat.mutation_api_key)
        or _resolve_secret(plat.validation_api_key)
        or _resolve_secret(mage.api_key)
    )

    validation_url = (
        plat.validation_base_url or mutation_url or mage.base_url or ""
    ).strip()
    validation_model = (
        plat.validation_model or mutation_model or mage.model or ""
    ).strip()
    validation_key = _resolve_secret(plat.validation_api_key) or mutation_key

    return (
        mutation_url,
        mutation_model,
        mutation_key,
        validation_url,
        validation_model,
        validation_key,
    )


def _model_block(
    *,
    model_id: str,
    label: str,
    description: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "id": model_id,
        "label": label,
        "provider": "openrouter",
        "supports": ["chat", "json"],
        "ui": {"description": description, "tags": ["care"]},
        "runtime": {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "temperature": 0.7,
            "max_tokens": int(max_tokens),
            "top_p": 1.0,
            "request_timeout": 600,
            "max_retries": 3,
            "timeout": 600,
            "ssl_no_verify": False,
        },
    }


def build_llm_models_registry(
    cfg: Any,
    *,
    mutation_max_tokens: int | None = None,
    validation_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the ``llm_models.yml`` payload from a loaded :class:`CareConfig`."""
    (
        mutation_url,
        mutation_model,
        mutation_key,
        validation_url,
        validation_model,
        validation_key,
    ) = _platform_llm_fields(cfg)

    if not mutation_key or not validation_key:
        raise ValueError(
            "LLM API keys are empty — set CARE_PLATFORM__MUTATION_API_KEY, "
            "CARE_MAGE__API_KEY, or keystore slots in ~/.config/care/config.toml",
        )
    if not mutation_url or not mutation_model:
        raise ValueError(
            "Mutation LLM base_url and model are required "
            "(platform or mage section)",
        )
    if not validation_url or not validation_model:
        raise ValueError(
            "Validation LLM base_url and model are required "
            "(platform or mage section)",
        )

    plat = cfg.platform
    mut_tokens = (
        mutation_max_tokens
        if mutation_max_tokens is not None
        else int(getattr(plat, "mutation_max_tokens", 8192))
    )
    val_tokens = (
        validation_max_tokens
        if validation_max_tokens is not None
        else int(getattr(plat, "validation_max_tokens", 2048))
    )

    return {
        "version": 1,
        "defaults": {
            "llm_model": "care-mutation",
            "prompt_llm_model": "care-validation",
        },
        "models": [
            _model_block(
                model_id="care-mutation",
                label="CARE Mutation",
                description="CARE mutation LLM (GA chain proposals + CARL eval).",
                base_url=mutation_url,
                api_key=mutation_key,
                model=mutation_model,
                max_tokens=mut_tokens,
            ),
            _model_block(
                model_id="care-validation",
                label="CARE Validation",
                description="CARE validation LLM (GA judge / prompts).",
                base_url=validation_url,
                api_key=validation_key,
                model=validation_model,
                max_tokens=val_tokens,
            ),
        ],
    }


def default_platform_dir() -> Path:
    """Sibling gigaevo-platform checkout or ``CARE_PLATFORM__CHECKOUT_DIR``."""
    override = os.environ.get("CARE_PLATFORM__CHECKOUT_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / ".." / "gigaevo-platform").resolve()


def _yaml_quote(value: str) -> str:
    if value == "":
        return '""'
    if any(ch in value for ch in ':"\'\n\r\t#{}[],&*!|>'):
        return json.dumps(value, ensure_ascii=False)
    return value


def _dump_llm_models_yaml(payload: dict[str, Any]) -> str:
    """Minimal YAML writer for the fixed Platform registry schema."""
    lines = [
        "# Generated by care.runtime.platform_llm_sync — do not commit.",
        "# Updated from MAESTRO ~/.config/care/config.toml on Settings save",
        "# and before each evolution submit.",
        "",
        f"version: {payload['version']}",
        "defaults:",
        f"  llm_model: {_yaml_quote(payload['defaults']['llm_model'])}",
        "  prompt_llm_model: "
        f"{_yaml_quote(payload['defaults']['prompt_llm_model'])}",
        "models:",
    ]
    for model in payload["models"]:
        lines.extend([
            f"- id: {_yaml_quote(model['id'])}",
            f"  label: {_yaml_quote(model['label'])}",
            f"  provider: {_yaml_quote(model['provider'])}",
            "  supports:",
            "  - chat",
            "  - json",
            "  ui:",
            f"    description: {_yaml_quote(model['ui']['description'])}",
            "    tags:",
            "    - care",
            "  runtime:",
            f"    base_url: {_yaml_quote(model['runtime']['base_url'])}",
            f"    api_key: {_yaml_quote(model['runtime']['api_key'])}",
            f"    model: {_yaml_quote(model['runtime']['model'])}",
            f"    temperature: {model['runtime']['temperature']}",
            f"    max_tokens: {model['runtime']['max_tokens']}",
            f"    top_p: {model['runtime']['top_p']}",
            f"    request_timeout: {model['runtime']['request_timeout']}",
            f"    max_retries: {model['runtime']['max_retries']}",
            f"    timeout: {model['runtime']['timeout']}",
            f"    ssl_no_verify: {str(model['runtime']['ssl_no_verify']).lower()}",
        ])
    return "\n".join(lines) + "\n"


def sync_platform_llm_registry(
    cfg: Any,
    *,
    platform_dir: Path | None = None,
    dry_run: bool = False,
    mutation_max_tokens: int | None = None,
    validation_max_tokens: int | None = None,
) -> PlatformLlmSyncResult:
    """Write ``llm_models.yml`` under ``platform_dir`` from ``cfg``."""
    root = (platform_dir or default_platform_dir()).resolve()
    target = root / "llm_models.yml"
    payload = build_llm_models_registry(
        cfg,
        mutation_max_tokens=mutation_max_tokens,
        validation_max_tokens=validation_max_tokens,
    )
    (
        mutation_url,
        mutation_model,
        _mk,
        validation_url,
        validation_model,
        _vk,
    ) = _platform_llm_fields(cfg)
    text = _dump_llm_models_yaml(payload)

    if dry_run:
        return PlatformLlmSyncResult(
            path=target,
            wrote=False,
            mutation_model=mutation_model,
            mutation_base_url=mutation_url,
            validation_model=validation_model,
            validation_base_url=validation_url,
            message=f"dry-run: would write {target}",
        )

    if not root.is_dir():
        return PlatformLlmSyncResult(
            path=None,
            wrote=False,
            mutation_model=mutation_model,
            mutation_base_url=mutation_url,
            validation_model=validation_model,
            validation_base_url=validation_url,
            message=f"gigaevo-platform not found at {root}",
        )

    target.write_text(text, encoding="utf-8")
    target.chmod(0o644)
    return PlatformLlmSyncResult(
        path=target,
        wrote=True,
        mutation_model=mutation_model,
        mutation_base_url=mutation_url,
        validation_model=validation_model,
        validation_base_url=validation_url,
        message=(
            f"synced {target} "
            f"(mutation={mutation_model}, validation={validation_model})"
        ),
    )


def try_sync_platform_llm_registry(
    cfg: Any,
    **kwargs: Any,
) -> PlatformLlmSyncResult | None:
    """Best-effort sync — returns ``None`` on validation / IO errors."""
    try:
        return sync_platform_llm_registry(cfg, **kwargs)
    except (OSError, ValueError):
        return None


__all__ = [
    "PlatformLlmSyncResult",
    "build_llm_models_registry",
    "default_platform_dir",
    "sync_platform_llm_registry",
    "try_sync_platform_llm_registry",
]
