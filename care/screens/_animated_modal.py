"""AnimatedModalScreen — shared modal entrance animation (TODO §Animations A-2).

A tiny :class:`~textual.screen.ModalScreen` subclass that fades its inner
content box in (opacity 0 → 1) with a subtle 1-row rise (offset y 1 → 0) over
~0.15 s on mount, using Textual's native ``styles.animate()`` API.

Reduced-motion safety: the entrance is a strict NO-OP whenever the app's
animation level is ``"none"`` — the box is left at its default fully-visible,
un-offset state and neither ``opacity`` nor ``offset`` is touched. The test
suite forces ``animation_level="none"`` globally (``tests/conftest.py``), and a
future reduced-motion config sets it too, so every modal that mixes this in
gets reduced-motion handling for free.
"""

from __future__ import annotations

from typing import Generic

from textual.css.scalar import ScalarOffset
from textual.screen import ModalScreen, ScreenResultType


class AnimatedModalScreen(ModalScreen[ScreenResultType], Generic[ScreenResultType]):
    """ModalScreen that fades its content box in on mount.

    Stays generic over ``ScreenResultType`` so subclasses keep the usual
    ``class FooModal(AnimatedModalScreen[Result])`` subscript that
    :meth:`~textual.screen.Screen.dismiss` is typed against.
    """

    #: CSS id (without ``#``) of the inner box to animate. Subclasses set this.
    ANIM_BOX_ID: str | None = None

    def _animate_modal_in(self) -> None:
        try:
            if self.app.animation_level == "none":
                return
            box = self.query_one(f"#{self.ANIM_BOX_ID}") if self.ANIM_BOX_ID else self
            box.styles.opacity = 0.0
            box.styles.offset = (0, 1)
            box.styles.animate(
                "opacity", value=1.0, duration=0.15, easing="out_cubic",
            )
            # NB: animate to a `ScalarOffset`, not a plain ``(0, 0)`` tuple —
            # ``styles.animate`` rejects raw tuples and would otherwise leave the
            # box stuck at offset ``(0, 1)`` (a permanent 1-row shift).
            box.styles.animate(
                "offset",
                value=ScalarOffset.from_offset((0, 0)),
                duration=0.15,
                easing="out_cubic",
            )
        except Exception:
            pass


__all__ = ["AnimatedModalScreen"]
