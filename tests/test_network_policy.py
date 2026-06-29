"""Tests for ``care.sandbox.network_policy`` (TODO §6.2 P0).

Pure-function coverage — no IO, no async. Three areas:

1. ``parse_webfetch_domains`` — accept str / list / None, ignore
   bare ``WebFetch``, dedupe + sort, tolerate whitespace and casing.
2. ``translate_to_carl_policy`` — three known mappings + ValueError
   on unknown.
3. ``resolve_network_policy`` — end-to-end: produces the expected
   ``ResolvedNetworkPolicy`` for each CARE policy literal,
   including the operator-override path.
"""

from __future__ import annotations

import pytest

from care.sandbox import (
    CARE_TO_CARL_POLICY,
    ResolvedNetworkPolicy,
    parse_webfetch_domains,
    resolve_network_policy,
    translate_to_carl_policy,
)


# ---------------------------------------------------------------------------
# parse_webfetch_domains
# ---------------------------------------------------------------------------


class TestParseWebfetchDomains:
    def test_none_returns_empty_list(self):
        assert parse_webfetch_domains(None) == []

    def test_empty_string(self):
        assert parse_webfetch_domains("") == []

    def test_empty_list(self):
        assert parse_webfetch_domains([]) == []

    def test_string_input_single_domain(self):
        assert parse_webfetch_domains("WebFetch(domain:api.example.com)") == [
            "api.example.com"
        ]

    def test_list_input(self):
        assert parse_webfetch_domains(
            ["Bash", "WebFetch(domain:api.example.com)", "Read"]
        ) == ["api.example.com"]

    def test_multiple_domains_dedup_and_sort(self):
        tokens = [
            "WebFetch(domain:beta.example.com)",
            "WebFetch(domain:api.example.com)",
            "WebFetch(domain:api.example.com)",  # duplicate
        ]
        assert parse_webfetch_domains(tokens) == [
            "api.example.com",
            "beta.example.com",
        ]

    def test_whitespace_tolerated_around_tokens(self):
        assert parse_webfetch_domains(
            "WebFetch( domain : api.example.com )"
        ) == ["api.example.com"]

    def test_case_insensitive_token_name(self):
        """`webfetch` should still match — manifests aren't case-pure."""
        assert parse_webfetch_domains("webfetch(domain:api.example.com)") == [
            "api.example.com"
        ]

    def test_bare_webfetch_ignored(self):
        """Bare ``WebFetch`` (no domain) doesn't widen the
        allowlist — that's a chain-author choice via ``policy=open``."""
        assert parse_webfetch_domains(["WebFetch", "Bash(git:*)"]) == []

    def test_mixed_with_unrelated_tokens(self):
        tokens = [
            "Bash(git:*)",
            "Read",
            "Write",
            "WebFetch(domain:api.example.com)",
            "WebFetch",  # bare, ignored
        ]
        assert parse_webfetch_domains(tokens) == ["api.example.com"]


# ---------------------------------------------------------------------------
# translate_to_carl_policy
# ---------------------------------------------------------------------------


class TestTranslateToCarlPolicy:
    @pytest.mark.parametrize(
        "care_name,carl_name",
        [
            ("none", "none"),
            ("skill_declared", "allowlist"),
            ("open", "host"),
        ],
    )
    def test_known_mappings(self, care_name, carl_name):
        assert translate_to_carl_policy(care_name) == carl_name

    def test_mapping_table_covers_every_carre_policy(self):
        """Pin the table content — adding a new CARE policy without
        updating CARL_TO_CARE_POLICY would break translation
        silently otherwise."""
        assert set(CARE_TO_CARL_POLICY.keys()) == {
            "none",
            "skill_declared",
            "open",
        }

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown CARE network policy"):
            translate_to_carl_policy("bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_network_policy
# ---------------------------------------------------------------------------


class TestResolveNetworkPolicy:
    def test_none_policy_yields_empty_domains(self):
        result = resolve_network_policy("none")
        assert isinstance(result, ResolvedNetworkPolicy)
        assert result.policy == "none"
        assert result.carl_policy == "none"
        assert result.domains == ()
        assert result.enforced is True

    def test_open_policy_yields_empty_domains(self):
        """Open means "trust the chain author"; the manifest's
        WebFetch hints are intentionally ignored."""
        result = resolve_network_policy(
            "open",
            allowed_tools=["WebFetch(domain:api.example.com)"],
        )
        assert result.carl_policy == "host"
        assert result.domains == ()

    def test_skill_declared_pulls_domains_from_allowed_tools(self):
        result = resolve_network_policy(
            "skill_declared",
            allowed_tools=[
                "Bash",
                "WebFetch(domain:api.example.com)",
                "WebFetch(domain:beta.example.com)",
            ],
        )
        assert result.carl_policy == "allowlist"
        assert result.domains == ("api.example.com", "beta.example.com")

    def test_skill_declared_with_no_allowed_tools(self):
        """No declarations + no overrides → empty allowlist (which
        is a valid "deny everything" outcome — the user explicitly
        chose skill_declared but the skill declared nothing)."""
        result = resolve_network_policy("skill_declared")
        assert result.domains == ()

    def test_override_domains_merge_in(self):
        result = resolve_network_policy(
            "skill_declared",
            allowed_tools=["WebFetch(domain:api.example.com)"],
            override_domains=["operator-added.example"],
        )
        assert result.domains == (
            "api.example.com",
            "operator-added.example",
        )

    def test_override_domains_strip_blanks(self):
        result = resolve_network_policy(
            "skill_declared",
            allowed_tools=None,
            override_domains=["", "  ", "good.example"],
        )
        assert result.domains == ("good.example",)

    def test_override_domains_dedupe_against_manifest(self):
        result = resolve_network_policy(
            "skill_declared",
            allowed_tools=["WebFetch(domain:dup.example)"],
            override_domains=["dup.example"],
        )
        assert result.domains == ("dup.example",)

    def test_enforced_override_false(self):
        """``LocalSandboxBackend`` calls with ``enforced_override=False``
        so CARE's TUI banner can warn."""
        result = resolve_network_policy(
            "none", enforced_override=False
        )
        assert result.enforced is False

    def test_enforced_override_true(self):
        result = resolve_network_policy(
            "skill_declared",
            allowed_tools=["WebFetch(domain:x.example)"],
            enforced_override=True,
        )
        assert result.enforced is True

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="unknown CARE network policy"):
            resolve_network_policy("not-a-policy")  # type: ignore[arg-type]


class TestResolvedNetworkPolicyShape:
    def test_frozen(self):
        result = resolve_network_policy("none")
        with pytest.raises(AttributeError):
            result.policy = "open"  # type: ignore[misc]

    def test_domains_is_tuple_not_list(self):
        """Tuple keeps the dataclass hashable + immutable; backends
        that snapshot the policy don't have to defensively copy."""
        result = resolve_network_policy(
            "skill_declared",
            allowed_tools=["WebFetch(domain:x.example)"],
        )
        assert isinstance(result.domains, tuple)
