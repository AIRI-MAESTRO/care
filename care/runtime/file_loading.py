"""Canonical "read a file as chain-ready content" used by every attach surface.

Before this, four surfaces read files independently and diverged: the chat
``@``-refs, the chat document-skill bridge, the RunContextModal context files,
and ``care run --file``. Some capped the size, some didn't; some crashed on a
binary (``UnicodeDecodeError``); only one handled images. This module is the
single implementation they all call:

* office documents (docx/pptx/xlsx/odf/rtf) and PDFs are extracted to text
  via the same extractors the chat ``@``-refs use;
* images become a base64 ``data:`` URI (so vision models can consume them);
* everything else is read as UTF-8 with undecodable bytes replaced — a binary
  never crashes the handoff;
* text is capped at :data:`MAX_CONTEXT_FILE_CHARS` with a truncation flag.

:class:`LoadedFile.as_block` renders the ready-to-inject prompt block, matching
the chat ``@``-ref envelopes (``<file …>`` / ``<image …>``).
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path

#: Cap on text pulled from a single file (chars). Bigger files are truncated
#: with a notice rather than ballooning the prompt / the run.
MAX_CONTEXT_FILE_CHARS = 200_000

#: Cap on an image attached inline as base64 (bytes). Larger images are
#: rejected with an error rather than bloating the request.
MAX_IMAGE_BYTES = 4_000_000

_OFFICE_EXTS = frozenset(
    {".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".rtf"}
)
_IMAGE_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
)


def is_image_path(path: str | Path) -> bool:
    return Path(str(path)).suffix.lower() in _IMAGE_EXTS


@dataclass(frozen=True)
class LoadedFile:
    """Result of :func:`load_file` — chain-ready content for one file."""

    path: str
    content: str
    """Extracted text for documents; ``""`` for images / errors."""
    image_data_uri: str | None
    truncated: bool
    error: str | None
    size_bytes: int

    @property
    def is_image(self) -> bool:
        return self.image_data_uri is not None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def memory_value(self) -> str:
        """The string to store in ``context.memory["input"][…]`` — the image
        data URI for images, the extracted text otherwise."""
        return self.image_data_uri if self.is_image else self.content

    def as_block(self, label: str | None = None) -> str:
        """Render the file as a prompt block (``<file>`` / ``<image>``),
        matching the chat ``@``-ref envelopes so every surface looks the same.
        """
        name = label or Path(self.path).name or self.path
        if self.is_image:
            mime = mimetypes.guess_type(name)[0] or "image/png"
            return (
                f'<image path="{name}" mime="{mime}" '
                f'size_bytes="{self.size_bytes}">\n'
                f"{self.image_data_uri}\n</image>"
            )
        note = (
            f"\n[… truncated to {len(self.content)} chars]"
            if self.truncated
            else ""
        )
        return f'<file path="{name}">\n{self.content}{note}\n</file>'


def load_file(
    path: str | Path,
    *,
    max_chars: int = MAX_CONTEXT_FILE_CHARS,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> LoadedFile:
    """Read ``path`` into chain-ready content. Never raises — a problem is
    reported via :attr:`LoadedFile.error` with empty content."""
    p = Path(str(path)).expanduser()
    try:
        size = p.stat().st_size
    except OSError:
        return LoadedFile(str(path), "", None, False, "file not found", 0)

    if is_image_path(p):
        if size > max_image_bytes:
            return LoadedFile(
                str(path), "", None, False,
                f"image too large ({size} bytes)", size,
            )
        try:
            data = p.read_bytes()
        except OSError as exc:
            return LoadedFile(
                str(path), "", None, False, type(exc).__name__, size,
            )
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        return LoadedFile(str(path), "", uri, False, None, size)

    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader

            text = "\n".join(
                (page.extract_text() or "") for page in PdfReader(str(p)).pages
            )
        elif ext in _OFFICE_EXTS:
            from care.runtime.document_extract import extract_document_text

            text = extract_document_text(p)
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — never crash the attach/handoff
        return LoadedFile(str(path), "", None, False, type(exc).__name__, size)

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return LoadedFile(str(path), text, None, truncated, None, size)


def load_file_text(
    path: str | Path, *, max_chars: int = MAX_CONTEXT_FILE_CHARS,
) -> tuple[str, bool]:
    """Convenience: ``(content, truncated)``. Images yield their data URI as
    the content. Kept small so existing ``(text, truncated)`` call sites can
    delegate here."""
    lf = load_file(path, max_chars=max_chars)
    return lf.memory_value, lf.truncated


__all__ = [
    "MAX_CONTEXT_FILE_CHARS",
    "MAX_IMAGE_BYTES",
    "LoadedFile",
    "is_image_path",
    "load_file",
    "load_file_text",
]
