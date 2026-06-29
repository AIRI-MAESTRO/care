"""Tests for ChatScreen's document-skill file helpers — the chat-side of the
"a chain that reads a file ran without one" guard (care.skill_file_inputs)."""

from __future__ import annotations

from pathlib import Path

from care.screens.chat import ChatScreen


class TestTaskHasInlineFile:
    def test_plain_task_has_no_inline_file(self):
        assert ChatScreen._task_has_inline_file("summarise the document") is False

    def test_inlined_file_block_detected(self):
        # _read_file_ref wraps @path content in a <file …> block.
        task = 'do it\n<file path="/x/a.txt">\n```\nbody\n```\n</file>\n'
        assert ChatScreen._task_has_inline_file(task) is True

    def test_inlined_image_block_detected(self):
        task = 'caption this\n<image path="/x/a.png" mime="image/png" size="1">'
        assert ChatScreen._task_has_inline_file(task) is True


class TestExtractSkillFileText:
    def test_plain_text(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("hello world", encoding="utf-8")
        assert ChatScreen._extract_skill_file_text(str(f)) == "hello world"

    def test_docx(self, tmp_path: Path):
        from docx import Document

        f = tmp_path / "report.docx"
        d = Document()
        d.add_paragraph("Quarterly revenue up 20 percent.")
        d.save(str(f))
        out = ChatScreen._extract_skill_file_text(str(f))
        assert out is not None
        assert "Quarterly revenue up 20 percent." in out

    def test_missing_file_returns_none(self):
        assert ChatScreen._extract_skill_file_text("/no/such/file-xyz.docx") is None
