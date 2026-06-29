"""Tests for CARE.md user/project context loading (P1.1)."""

from __future__ import annotations

from care import context_md
from care.config import CareConfig


def test_absent_files_is_empty(tmp_path):
    cfg = CareConfig()
    cfg.context.global_path = tmp_path / "nope" / "CARE.md"
    assert context_md.load_user_context(cfg, project_dir=tmp_path / "empty") == ""


def test_global_only(tmp_path):
    g = tmp_path / "CARE.md"
    g.write_text("Prefer metric units.", encoding="utf-8")
    cfg = CareConfig()
    cfg.context.global_path = g
    out = context_md.load_user_context(cfg, project_dir=tmp_path / "empty")
    assert "Prefer metric units." in out
    assert "Global user context" in out


def test_project_last_so_it_augments_global(tmp_path):
    gdir = tmp_path / "global"
    gdir.mkdir()
    (gdir / "CARE.md").write_text("global: answer in Russian", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "CARE.md").write_text("project: domain is finance", encoding="utf-8")
    cfg = CareConfig()
    cfg.context.global_path = gdir / "CARE.md"
    out = context_md.load_user_context(cfg, project_dir=proj)
    assert "global: answer in Russian" in out
    assert "project: domain is finance" in out
    # project block comes LAST (augments / overrides the global one)
    assert out.index("global: answer in Russian") < out.index("project: domain is finance")


def test_disabled_returns_empty(tmp_path):
    g = tmp_path / "CARE.md"
    g.write_text("x", encoding="utf-8")
    cfg = CareConfig()
    cfg.context.global_path = g
    cfg.context.enabled = False
    assert context_md.load_user_context(cfg, project_dir=tmp_path) == ""


def test_max_chars_truncates(tmp_path):
    g = tmp_path / "CARE.md"
    g.write_text("A" * 5000, encoding="utf-8")
    cfg = CareConfig()
    cfg.context.global_path = g
    cfg.context.max_chars = 500
    out = context_md.load_user_context(cfg, project_dir=tmp_path / "empty")
    assert len(out) <= 600
    assert "truncated" in out


def test_no_config_reads_project_from_cwd(tmp_path, monkeypatch):
    (tmp_path / "CARE.md").write_text("hello from project", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = context_md.load_user_context(None)  # defaults; cwd = tmp_path
    assert "hello from project" in out


# ---------------------------------------------------------------------------
# Learned-profile injection (P1.4)
# ---------------------------------------------------------------------------


def test_format_profile_block_renders_non_tool_signals():
    block = context_md.format_profile_block(
        {"tool_usage": {"web_search": 3, "calculator": 1}, "recent_domains": ["finance"], "last_mode": "deep"}
    )
    assert "Learned user preferences" in block
    assert "finance" in block and "deep" in block


def test_format_profile_block_never_lists_tool_names():
    # Listing learned tool names biased the planner to re-invent them (P1.4
    # pollution loop) — they must NOT appear in the generation context.
    block = context_md.format_profile_block(
        {"tool_usage": {"get_current_time": 9, "web_search": 3}, "recent_domains": ["x"]}
    )
    assert "get_current_time" not in block and "web_search" not in block


def test_format_profile_block_empty_is_blank():
    assert context_md.format_profile_block(None) == ""
    assert context_md.format_profile_block({}) == ""


def test_load_user_context_appends_profile_block(tmp_path):
    cfg = CareConfig()
    cfg.context.global_path = tmp_path / "none" / "CARE.md"  # absent
    out = context_md.load_user_context(
        cfg,
        project_dir=tmp_path / "empty",  # no project CARE.md
        profile={"last_mode": "deep", "recent_domains": ["finance"], "tool_usage": {"web_search": 2}},
    )
    # profile flows into the generation context even with no CARE.md files,
    # but NEVER lists tool names (the P1.4 pollution loop)
    assert "Learned user preferences" in out
    assert "finance" in out and "deep" in out
    assert "web_search" not in out


# ---------------------------------------------------------------------------
# Personalization shaping (P2.7)
# ---------------------------------------------------------------------------


def test_two_profiles_produce_visibly_different_shaping():
    """The Verify for P2.7 — two distinct profiles yield distinct
    personalization blocks (different depth + domain directives)."""
    a = context_md.format_profile_block(
        {"recent_domains": ["finance"], "last_mode": "fast"}
    )
    b = context_md.format_profile_block(
        {"recent_domains": ["biology"], "last_mode": "deep", "run_count": 12}
    )
    assert a != b
    assert "finance" in a and "concise" in a  # fast → concise shaping
    assert "biology" in b and "thorough" in b  # deep → thorough shaping
    assert "12 prior task" in b  # experienced-user directive


def test_format_profile_block_renders_run_count_directive():
    block = context_md.format_profile_block({"run_count": 1, "last_mode": "deep"})
    assert "1 prior task)" in block  # singular, no plural 's'
    assert "thorough" in block


# ---------------------------------------------------------------------------
# P5.6 — CARE.md auto-create + auto-learn user facts
# ---------------------------------------------------------------------------


def test_ensure_care_md_writes_scaffold(tmp_path):
    p = tmp_path / "sub" / "CARE.md"
    out = context_md.ensure_care_md(p)
    assert out == p
    assert p.is_file()  # parents created
    text = p.read_text(encoding="utf-8")
    assert "# CARE.md" in text
    assert "## Auto-learned facts" in text


def test_ensure_care_md_idempotent_preserves_content(tmp_path):
    p = tmp_path / "CARE.md"
    context_md.merge_learned_fact(p, "Language", "Russian")
    context_md.ensure_care_md(p)  # must NOT clobber the existing file
    assert "- Language: Russian" in p.read_text(encoding="utf-8")


def test_merge_learned_fact_adds_and_dedups(tmp_path):
    p = tmp_path / "CARE.md"
    assert context_md.merge_learned_fact(p, "Language", "Russian") is True
    # exact dup → no-op, no second bullet
    assert context_md.merge_learned_fact(p, "Language", "Russian") is False
    text = p.read_text(encoding="utf-8")
    assert text.count("- Language: Russian") == 1
    assert "## Auto-learned facts" in text


def test_merge_learned_fact_supersedes_same_key(tmp_path):
    p = tmp_path / "CARE.md"
    context_md.merge_learned_fact(p, "Language", "Russian")
    assert context_md.merge_learned_fact(p, "Language", "French") is True
    text = p.read_text(encoding="utf-8")
    assert "- Language: French" in text
    assert "Russian" not in text  # stale value superseded, not duplicated
    assert text.count("- Language:") == 1  # one bullet for the key


def test_merge_distinct_keys_accumulate(tmp_path):
    p = tmp_path / "CARE.md"
    context_md.merge_learned_fact(p, "Language", "Russian")
    context_md.merge_learned_fact(p, "Role", "ML engineer")
    text = p.read_text(encoding="utf-8")
    assert "- Language: Russian" in text
    assert "- Role: ML engineer" in text


def test_learned_fact_feeds_personalization(tmp_path):
    # A fact written to the global CARE.md rides into the next generation's
    # context via load_user_context (the P2.7 personalization read path).
    g = tmp_path / "CARE.md"
    context_md.merge_learned_fact(g, "Language", "always answer in Russian")
    cfg = CareConfig()
    cfg.context.global_path = g
    out = context_md.load_user_context(cfg, project_dir=tmp_path / "empty")
    assert "always answer in Russian" in out
    assert "Global user context" in out
