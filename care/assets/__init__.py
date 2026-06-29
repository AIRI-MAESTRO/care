"""Static assets bundled with the CARE package.

Currently:

* ``airi_logo.png`` — 12×12 down-sized AIRI logo. Renders
  as a half-block (6×6 cell) pixel art block to the left of
  the boot-banner text in :meth:`care.screens.chat.ChatScreen._post_boot_header`.

Asset paths are resolved via :mod:`importlib.resources` so
both editable installs and wheels keep the package data
reachable.
"""
