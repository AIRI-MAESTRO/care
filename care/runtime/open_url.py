"""Best-effort "open a URL in the user's browser" (PRODUCTION_TODO B3).

The chat surface always POSTS the url as text too (terminals make it
clickable and it survives SSH sessions where no local browser exists), so
this helper is allowed to silently fail — its return value only decorates
the chat line with "(opened in browser)".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def open_url(url: str) -> bool:
    """Open ``url`` in the default browser; True when a browser took it."""
    try:
        import webbrowser

        return bool(webbrowser.open(url))
    except Exception:  # noqa: BLE001 — headless / no display / odd platform
        logger.debug("open_url: could not open %s", url, exc_info=True)
        return False
