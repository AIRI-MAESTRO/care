"""Sync Platform chain experiment templates into a live runner problem dir.

master-api sometimes bakes an older ``helper.py`` (mmar_carl + broken
``SingleEndpointDict``) into ``exp_*`` folders.  MAESTRO overlays the
current templates on the runner clone so validation executes chains and
ROUGE/BERTScore behave as intended.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from care.runtime.runner_tools_sync import _docker_container_running

_log = logging.getLogger("care.evolution_chain_templates")

_CHAIN_FILES = (
    "helper.py",
    "validate.py",
    "chain_client.py",
    "chain_runner.py",
    "chain_types.py",
    "chain_validation.py",
    "context.py",
)

_TEMPLATED_FILES = frozenset({"validate.py", "context.py"})

_STALE_HELPER_MARKERS = (
    "from mmar_carl import",
    "SingleEndpointDict",
)


@dataclass(frozen=True)
class ChainTemplateSyncResult:
    copied: int
    experiment_id: str
    container: str
    dest_dir: str
    ok: bool
    message: str


def default_template_dir() -> Path:
    env = os.environ.get("CARE_PLATFORM__CHAIN_TEMPLATE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    care_root = here.parents[2]
    return (
        care_root.parent
        / "gigaevo-platform"
        / "master_api"
        / "src"
        / "folder_constructor"
        / "validate_templates"
        / "chain"
    )


def default_runner_container() -> str:
    return os.environ.get(
        "CARE_PLATFORM__RUNNER_CONTAINER",
        "gigaevo-platform-runner-api-1-1",
    )


def default_clone_root() -> str:
    return os.environ.get(
        "CARE_PLATFORM__GIGAVOLVE_CLONE_ROOT",
        "/tmp/gigavolve/gigaevo-core-1",
    )


def default_runner_user() -> str:
    """Unix user that executes ``run.py`` inside the runner container."""
    return os.environ.get("CARE_PLATFORM__RUNNER_USER", "gigaevouser")


def verify_chain_template_source() -> tuple[bool, str]:
    """Return whether patched chain templates are available locally."""
    tpl = default_template_dir()
    if not tpl.is_dir():
        return False, f"chain templates missing: {tpl}"
    helper = tpl / "helper.py"
    if not helper.is_file():
        return False, f"chain helper.py missing under {tpl}"
    text = helper.read_text(encoding="utf-8")
    if "chain_runner" not in text:
        return False, f"chain helper.py at {tpl} looks stale (no chain_runner)"
    return True, f"chain templates ready: {tpl}"


def sync_kwargs_from_experiment(experiment: dict[str, Any] | None) -> dict[str, Any]:
    """Extract template-render kwargs from a Platform experiment record."""
    if not experiment:
        return {}
    cfg = experiment.get("config") or {}
    params = cfg.get("parameters") or {}
    vc = params.get("validation_criteria") or {}
    if not isinstance(vc, dict):
        vc = {}
    return {
        "validation_type": vc.get("validation_type"),
        "continuous_metric": vc.get("continuous_metric"),
        "binary_method": vc.get("binary_method"),
        "target_column": params.get("target_column") or "expected",
        "regexp_pattern": vc.get("regexp_pattern") or "",
    }


def _build_placeholders(
    *,
    experiment_id: str,
    validation_type: str | None,
    continuous_metric: str | None,
    binary_method: str | None,
    target_column: str,
    regexp_pattern: str,
) -> dict[str, str]:
    from care.runtime.evolution_validation import build_chain_validation_criteria

    vc = build_chain_validation_criteria(
        validation_type=validation_type,
        continuous_metric=continuous_metric,
        binary_method=binary_method,
        regexp_pattern=regexp_pattern,
    )
    vtype = str(vc.get("validation_type") or "Continuous (0..1)")
    if vtype == "Binary (0/1)":
        metric = str(vc.get("binary_method") or "equality")
    else:
        metric = str(vc.get("continuous_metric") or "ROUGE-L")
    return {
        "validation_type": vtype,
        "metric": metric,
        "fitness_mode": "accuracy",
        "target_field": target_column or "expected",
        "task_name": experiment_id.removeprefix("exp_") or "chain",
        "regexp_pattern": regexp_pattern or r"Answer:\s*(.+?)$",
    }


def _render_template(content: str, placeholders: dict[str, str]) -> str:
    if "${" not in content:
        return content
    return Template(content).safe_substitute(placeholders)


def _runner_path_exists(container: str, path: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "test", "-e", path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _ensure_runner_dest_dir(container: str, dest_dir: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "mkdir", "-p", dest_dir],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _fix_runner_file_permissions(
    container: str,
    dest_dir: str,
    filenames: list[str],
    *,
    runner_user: str | None = None,
) -> None:
    """``docker cp`` lands host temp files as mode 600 + foreign uid.

    ``run.py`` executes as ``gigaevouser`` and cannot import
    ``validate.py`` / ``helper.py`` until we chmod/chown them.
    """
    if not filenames:
        return
    user = runner_user or default_runner_user()
    paths = " ".join(f"{dest_dir}/{name}" for name in filenames)
    try:
        subprocess.run(
            [
                "docker", "exec", container,
                "bash", "-lc",
                f"chmod a+r {paths} && chown {user}:{user} {paths}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def runner_helper_is_stale(
    experiment_id: str,
    *,
    container: str | None = None,
    clone_root: str | None = None,
) -> bool:
    """True when the live problem dir still ships the old mmar_carl helper."""
    if not experiment_id.startswith("exp_"):
        return False
    cont = container or default_runner_container()
    root = clone_root or default_clone_root()
    helper_path = f"{root}/problems/{experiment_id}/helper.py"
    if not _runner_path_exists(cont, helper_path):
        return True
    user = default_runner_user()
    try:
        proc = subprocess.run(
            [
                "docker", "exec", "-u", user, cont,
                "python3", "-c",
                f"print(open({helper_path!r}, encoding='utf-8').read())",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if proc.returncode != 0:
        return True
    text = proc.stdout
    if "chain_runner" in text:
        return False
    return any(marker in text for marker in _STALE_HELPER_MARKERS)


def sync_chain_templates_to_runner(
    experiment_id: str,
    *,
    template_dir: Path | None = None,
    container: str | None = None,
    clone_root: str | None = None,
    validation_type: str | None = None,
    continuous_metric: str | None = None,
    binary_method: str | None = None,
    target_column: str = "expected",
    regexp_pattern: str = "",
) -> ChainTemplateSyncResult:
    """Copy chain templates into ``problems/<experiment_id>/`` on the runner.

    Templated files (``validate.py``, ``context.py``) are rendered with the
    experiment's validation settings so we never overwrite a live problem with
    raw ``${validation_type}`` placeholders.
    """
    cont = container or default_runner_container()
    root = clone_root or default_clone_root()
    dest_dir = f"{root}/problems/{experiment_id}"

    if not experiment_id.startswith("exp_"):
        return ChainTemplateSyncResult(
            copied=0,
            experiment_id=experiment_id,
            container=cont,
            dest_dir=dest_dir,
            ok=False,
            message="not a chain experiment id",
        )

    tpl = template_dir or default_template_dir()
    if not tpl.is_dir():
        return ChainTemplateSyncResult(
            copied=0,
            experiment_id=experiment_id,
            container=cont,
            dest_dir=dest_dir,
            ok=False,
            message=f"template dir missing: {tpl}",
        )

    if not _docker_container_running(cont):
        return ChainTemplateSyncResult(
            copied=0,
            experiment_id=experiment_id,
            container=cont,
            dest_dir=dest_dir,
            ok=False,
            message=f"runner container not running: {cont}",
        )

    _ensure_runner_dest_dir(cont, dest_dir)

    placeholders = _build_placeholders(
        experiment_id=experiment_id,
        validation_type=validation_type,
        continuous_metric=continuous_metric,
        binary_method=binary_method,
        target_column=target_column,
        regexp_pattern=regexp_pattern,
    )
    copied = 0
    copied_names: list[str] = []
    for name in _CHAIN_FILES:
        src = tpl / name
        if not src.is_file():
            continue
        content = src.read_text(encoding="utf-8")
        if name in _TEMPLATED_FILES:
            content = _render_template(content, placeholders)
        tmp_path: str | None = None
        proc = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=f"_{name}",
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                os.chmod(tmp_path, 0o644)
            except OSError:
                pass
            proc = subprocess.run(
                ["docker", "cp", tmp_path, f"{cont}:{dest_dir}/{name}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ChainTemplateSyncResult(
                copied=copied,
                experiment_id=experiment_id,
                container=cont,
                dest_dir=dest_dir,
                ok=copied > 0,
                message=f"docker cp failed for {name}: {exc}",
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        if proc is not None and proc.returncode == 0:
            copied += 1
            copied_names.append(name)

    if copied_names:
        _fix_runner_file_permissions(cont, dest_dir, copied_names)

    stale = runner_helper_is_stale(
        experiment_id, container=cont, clone_root=root,
    )
    if copied > 0 and not stale:
        msg = f"copied {copied} files → {cont}:{dest_dir}"
        ok = True
    elif not stale:
        msg = f"templates already current at {dest_dir}"
        ok = True
    elif copied > 0:
        msg = f"copied {copied} files but helper still looks stale"
        ok = False
    else:
        msg = f"no templates copied (dest={dest_dir})"
        ok = False
    _log.info("chain template sync %s: %s", experiment_id, msg)
    return ChainTemplateSyncResult(
        copied=copied,
        experiment_id=experiment_id,
        container=cont,
        dest_dir=dest_dir,
        ok=ok,
        message=msg,
    )


def sync_chain_templates_until_ready(
    experiment_id: str,
    *,
    timeout: float = 180.0,
    poll_interval: float = 2.0,
    **kwargs: Any,
) -> ChainTemplateSyncResult:
    """Poll until the runner problem dir exists, then overlay templates."""
    cont = kwargs.get("container") or default_runner_container()
    root = kwargs.get("clone_root") or default_clone_root()
    dest_dir = f"{root}/problems/{experiment_id}"
    deadline = time.monotonic() + max(5.0, timeout)
    last: ChainTemplateSyncResult | None = None
    while time.monotonic() < deadline:
        if not runner_helper_is_stale(
            experiment_id, container=cont, clone_root=root,
        ):
            return last or ChainTemplateSyncResult(
                copied=0,
                experiment_id=experiment_id,
                container=cont,
                dest_dir=dest_dir,
                ok=True,
                message=f"templates already current at {dest_dir}",
            )
        last = sync_chain_templates_to_runner(experiment_id, **kwargs)
        if last.ok and not runner_helper_is_stale(
            experiment_id, container=cont, clone_root=root,
        ):
            return last
        time.sleep(poll_interval)
    return last or ChainTemplateSyncResult(
        copied=0,
        experiment_id=experiment_id,
        container=cont,
        dest_dir=dest_dir,
        ok=False,
        message="timed out waiting for runner problem dir",
    )


_sync_lock = threading.Lock()
_last_sync_attempt: dict[str, float] = {}
_scheduled_syncs: set[str] = set()


def schedule_chain_template_sync(
    experiment_id: str,
    **kwargs: Any,
) -> None:
    """Background overlay loop — survives runner restarts and late initialize."""
    if not experiment_id.startswith("exp_"):
        return
    with _sync_lock:
        if experiment_id in _scheduled_syncs:
            return
        _scheduled_syncs.add(experiment_id)

    def _worker() -> None:
        try:
            result = sync_chain_templates_until_ready(experiment_id, **kwargs)
            _log.info(
                "scheduled chain sync finished for %s: %s",
                experiment_id,
                result.message,
            )
        except Exception as exc:
            _log.warning(
                "scheduled chain sync failed for %s: %s",
                experiment_id,
                exc,
            )
        finally:
            with _sync_lock:
                _scheduled_syncs.discard(experiment_id)

    threading.Thread(
        target=_worker,
        name=f"care-chain-sync-{experiment_id[:12]}",
        daemon=True,
    ).start()


def maybe_sync_chain_templates(
    experiment_id: str,
    *,
    experiment: dict[str, Any] | None = None,
    min_interval: float = 8.0,
) -> ChainTemplateSyncResult | None:
    """Best-effort re-sync when polling observes a stale runner problem dir."""
    if not experiment_id.startswith("exp_"):
        return None
    if os.environ.get("CARE_PLATFORM__SYNC_CHAIN_TEMPLATES", "1") == "0":
        return None
    now = time.monotonic()
    with _sync_lock:
        last = _last_sync_attempt.get(experiment_id, 0.0)
        if now - last < min_interval:
            return None
        _last_sync_attempt[experiment_id] = now
    if not runner_helper_is_stale(experiment_id):
        return None
    kwargs = sync_kwargs_from_experiment(experiment)
    schedule_chain_template_sync(experiment_id, **kwargs)
    return None


__all__ = [
    "ChainTemplateSyncResult",
    "default_template_dir",
    "maybe_sync_chain_templates",
    "runner_helper_is_stale",
    "schedule_chain_template_sync",
    "sync_chain_templates_to_runner",
    "sync_chain_templates_until_ready",
    "sync_kwargs_from_experiment",
    "verify_chain_template_source",
]
