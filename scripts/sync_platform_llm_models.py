#!/usr/bin/env python3
"""Write gigaevo-platform/llm_models.yml from MAESTRO config.

Platform runners read ``/llm_models.yml`` (bind-mounted from the checkout).
Evolution uses model ids ``care-mutation`` and ``care-validation`` — they
must exist in that file with working keys aligned to MAESTRO settings.

Also runs automatically on Settings save and before each evolution submit.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform-dir",
        type=Path,
        default=None,
        help="Path to gigaevo-platform (default: ../gigaevo-platform)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print path that would be written, do not write",
    )
    args = parser.parse_args(argv)

    from care.config import CareConfig
    from care.runtime.platform_llm_sync import (
        default_platform_dir,
        sync_platform_llm_registry,
    )

    cfg = CareConfig.load()
    platform_dir = args.platform_dir or default_platform_dir()
    result = sync_platform_llm_registry(
        cfg,
        platform_dir=platform_dir,
        dry_run=args.dry_run,
    )
    print(result.message)
    if not result.wrote and not args.dry_run:
        if result.path is None:
            return 1
        return 0
    if result.wrote:
        print(
            f"  care-mutation → {result.mutation_model} "
            f"@ {result.mutation_base_url}",
        )
        print(
            f"  care-validation → {result.validation_model} "
            f"@ {result.validation_base_url}",
        )
        print("Restart runner to reload bind-mounted llm_models.yml:")
        print(f"  cd {platform_dir} && make restart SERVICE=runner-api")
        print("  # or: ./deploy.sh restart runner-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
