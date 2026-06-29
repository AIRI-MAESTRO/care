"""UI-language catalogs bundled with the CARE package.

Each ``<lang>.json`` here is a nested key→text catalog consumed by
:mod:`care.runtime.i18n` (``en.json`` is the fallback / source of truth for
the key set). Catalogs are resolved via :mod:`importlib.resources` so both
editable installs and wheels keep the package data reachable.
"""
