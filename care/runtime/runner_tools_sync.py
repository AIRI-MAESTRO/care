"""Copy gigaevo-core metric tools into the live runner Docker clone."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TOOL_MODULES = ("comparison", "redis2pd", "utils")


@dataclass(frozen=True)
class RunnerToolsSyncResult:
    copied: tuple[str, ...]
    container: str
    clone_tools_dir: str
    message: str


def default_gigaevo_core_dir() -> Path:
    env = os.environ.get("CARE_PLATFORM__GIGAVOLVE_CORE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    care_root = Path(__file__).resolve().parents[2]
    for candidate in (
        care_root.parent / "gigaevo-core",
        Path.home() / "gigaevo-core",
    ):
        if (candidate / "tools" / "comparison.py").is_file():
            return candidate
    return care_root.parent / "gigaevo-core"


def default_runner_container() -> str:
    return os.environ.get(
        "CARE_PLATFORM__RUNNER_CONTAINER",
        "gigaevo-platform-runner-api-1-1",
    )


def default_clone_tools_dir() -> str:
    clone_root = os.environ.get(
        "CARE_PLATFORM__GIGAVOLVE_CLONE_ROOT",
        "/tmp/gigavolve/gigaevo-core-1",
    )
    return f"{clone_root.rstrip('/')}/tools"


def _docker_container_running(name: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def sync_runner_gigaevo_tools(
    *,
    core_dir: Path | None = None,
    container: str | None = None,
    clone_tools_dir: str | None = None,
) -> RunnerToolsSyncResult | None:
    """Best-effort ``docker cp`` of gigaevo metric tools (comparison, redis2pd, utils)."""
    if os.environ.get("CARE_PLATFORM__SYNC_RUNNER_TOOLS", "1") == "0":
        return None

    root = core_dir or default_gigaevo_core_dir()
    cont = container or default_runner_container()
    dest = clone_tools_dir or default_clone_tools_dir()

    if not _docker_container_running(cont):
        return RunnerToolsSyncResult(
            copied=(),
            container=cont,
            clone_tools_dir=dest,
            message=f"runner container not running: {cont}",
        )

    copied: list[str] = []
    for mod in _TOOL_MODULES:
        src = root / "tools" / f"{mod}.py"
        if not src.is_file():
            continue
        try:
            proc = subprocess.run(
                ["docker", "cp", str(src), f"{cont}:{dest}/{mod}.py"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return RunnerToolsSyncResult(
                copied=tuple(copied),
                container=cont,
                clone_tools_dir=dest,
                message=f"docker cp failed for {mod}: {exc}",
            )
        if proc.returncode == 0:
            copied.append(mod)

    if not copied:
        return RunnerToolsSyncResult(
            copied=(),
            container=cont,
            clone_tools_dir=dest,
            message=f"no tools copied (core_dir={root})",
        )
    return RunnerToolsSyncResult(
        copied=tuple(copied),
        container=cont,
        clone_tools_dir=dest,
        message=f"copied {', '.join(copied)} → {cont}:{dest}",
    )


__all__ = [
    "RunnerToolsSyncResult",
    "default_gigaevo_core_dir",
    "sync_runner_gigaevo_tools",
]
