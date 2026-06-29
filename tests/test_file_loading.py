"""Tests for the canonical file loader (care.runtime.file_loading)."""

from __future__ import annotations

import base64
from pathlib import Path

from care.runtime.file_loading import (
    MAX_CONTEXT_FILE_CHARS,
    LoadedFile,
    is_image_path,
    load_file,
    load_file_text,
)


class TestText:
    def test_plain_text(self, tmp_path: Path):
        f = tmp_path / "n.txt"
        f.write_text("hello world")
        lf = load_file(str(f))
        assert lf.ok and not lf.is_image
        assert lf.content == "hello world"
        assert lf.truncated is False
        assert lf.memory_value == "hello world"

    def test_truncation(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("A" * (MAX_CONTEXT_FILE_CHARS + 100))
        lf = load_file(str(f))
        assert lf.truncated is True
        assert len(lf.content) == MAX_CONTEXT_FILE_CHARS
        assert "truncated" in lf.as_block("big.txt")

    def test_custom_cap(self, tmp_path: Path):
        f = tmp_path / "x.txt"
        f.write_text("hello world")
        assert load_file(str(f), max_chars=5).content == "hello"

    def test_binary_safe(self, tmp_path: Path):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\x80")
        lf = load_file(str(f))
        assert isinstance(lf.content, str)  # no crash
        assert lf.ok

    def test_missing(self):
        lf = load_file("/no/such/file-xyz.txt")
        assert lf.error == "file not found"
        assert lf.content == ""
        assert lf.memory_value == ""

    def test_docx(self, tmp_path: Path):
        from docx import Document

        f = tmp_path / "r.docx"
        d = Document()
        d.add_paragraph("Hello from DOCX")
        d.save(str(f))
        lf = load_file(str(f))
        assert "Hello from DOCX" in lf.content

    def test_load_file_text_convenience(self, tmp_path: Path):
        f = tmp_path / "x.txt"
        f.write_text("hi")
        assert load_file_text(str(f)) == ("hi", False)


class TestImages:
    def test_is_image_path(self):
        assert is_image_path("a.PNG") is True
        assert is_image_path("a.jpeg") is True
        assert is_image_path("a.txt") is False

    def test_image_to_data_uri(self, tmp_path: Path):
        f = tmp_path / "pic.png"
        raw = b"\x89PNG\r\n\x1a\n" + b"fakepngdata"
        f.write_bytes(raw)
        lf = load_file(str(f))
        assert lf.is_image
        assert lf.image_data_uri.startswith("data:image/png;base64,")
        assert base64.b64decode(lf.image_data_uri.split(",", 1)[1]) == raw
        # memory_value is the data URI (not empty text)
        assert lf.memory_value == lf.image_data_uri

    def test_image_block_envelope(self, tmp_path: Path):
        f = tmp_path / "shot.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\nxx")
        block = load_file(str(f)).as_block()
        assert block.startswith('<image path="shot.png"')
        assert "data:image/png;base64," in block
        assert block.rstrip().endswith("</image>")

    def test_oversize_image_rejected(self, tmp_path: Path):
        f = tmp_path / "huge.png"
        f.write_bytes(b"x" * 50)
        lf = load_file(str(f), max_image_bytes=10)
        assert lf.error is not None
        assert lf.image_data_uri is None


class TestAsBlock:
    def test_file_block(self):
        lf = LoadedFile("/x/a.txt", "body", None, False, None, 4)
        block = lf.as_block()
        assert block == '<file path="a.txt">\nbody\n</file>'
