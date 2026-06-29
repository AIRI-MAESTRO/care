"""Tests for `care.runtime.chain_title.suggest_chain_title`
(TODO §3 P2)."""

from __future__ import annotations

from types import SimpleNamespace

from care.runtime.chain_title import (
    MAX_TITLE_CHARS,
    _build_user_prompt,
    _clean_suggestion,
    suggest_chain_title,
)


# ---------------------------------------------------------------------------
# _clean_suggestion (pure)
# ---------------------------------------------------------------------------


class TestCleanSuggestion:
    def test_trims_whitespace(self):
        assert _clean_suggestion("  Weather Forecaster  ") == (
            "Weather Forecaster"
        )

    def test_strips_double_quotes(self):
        assert _clean_suggestion('"Weather Forecaster"') == (
            "Weather Forecaster"
        )

    def test_strips_single_quotes(self):
        assert _clean_suggestion("'Weather Forecaster'") == (
            "Weather Forecaster"
        )

    def test_strips_backticks(self):
        assert _clean_suggestion("`Weather Forecaster`") == (
            "Weather Forecaster"
        )

    def test_drops_trailing_period(self):
        assert _clean_suggestion("Weather Forecaster.") == (
            "Weather Forecaster"
        )

    def test_drops_trailing_punctuation_run(self):
        assert _clean_suggestion("Forecaster!!! ") == "Forecaster"

    def test_collapses_internal_whitespace(self):
        assert _clean_suggestion("Foo\n\nbar   baz") == "Foo bar baz"

    def test_caps_at_max_chars(self):
        long = "X" * 200
        out = _clean_suggestion(long)
        assert len(out) <= MAX_TITLE_CHARS
        assert out == "X" * MAX_TITLE_CHARS

    def test_empty_input(self):
        assert _clean_suggestion("") == ""

    def test_only_whitespace(self):
        assert _clean_suggestion("   \n\t  ") == ""

    def test_non_string_input(self):
        assert _clean_suggestion(None) == ""
        assert _clean_suggestion(42) == ""


# ---------------------------------------------------------------------------
# _build_user_prompt (pure)
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_extracts_chain_name_and_steps(self):
        chain = {
            "name": "weather-forecaster",
            "steps": [
                {
                    "title": "Fetch forecast",
                    "aim": "Pull JSON from API",
                    "step_type": "tool",
                },
                {
                    "name": "Summarise",
                    "description": "Distill into prose",
                    "type": "llm",
                },
            ],
        }
        out = _build_user_prompt(chain)
        assert "weather-forecaster" in out
        assert "Fetch forecast" in out
        # Falls back to `name` when `title` is missing.
        assert "Summarise" in out
        # Step type echoed.
        assert "tool" in out
        assert "llm" in out

    def test_caps_step_count(self):
        chain = {
            "name": "big",
            "steps": [
                {"title": f"step-{i}", "aim": "x", "step_type": "llm"}
                for i in range(30)
            ],
        }
        out = _build_user_prompt(chain)
        # Only first 12 steps render.
        assert "step-0" in out
        assert "step-11" in out
        assert "step-12" not in out

    def test_non_dict_chain_safe(self):
        # Non-dict chain → empty-shell payload (no crash).
        out = _build_user_prompt([1, 2, 3])
        assert "steps" in out
        # No name, no real steps.
        assert "step-" not in out

    def test_skips_non_dict_step_entries(self):
        chain = {
            "name": "x",
            "steps": [
                "junk-string",
                {"title": "real-step", "aim": "y"},
            ],
        }
        out = _build_user_prompt(chain)
        assert "real-step" in out
        assert "junk-string" not in out


# ---------------------------------------------------------------------------
# suggest_chain_title (mocked client)
# ---------------------------------------------------------------------------


def _make_client(response_text: str | Exception):
    """Build a mock client whose `chat.completions.create`
    returns a response carrying `choices[0].message.content =
    response_text`, OR raises when given an Exception."""
    captured: dict = {}

    def _create(**kwargs):
        captured.update(kwargs)
        if isinstance(response_text, Exception):
            raise response_text
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_text),
                ),
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    return client, captured


class TestSuggestChainTitle:
    def test_returns_cleaned_suggestion(self):
        client, captured = _make_client('"Weather Forecaster" ')
        out = suggest_chain_title(
            {"name": "w", "steps": []},
            client=client,
            model="m",
        )
        assert out == "Weather Forecaster"
        # Call shape: temperature + max_tokens land.
        assert captured["model"] == "m"
        assert captured["temperature"] == 0.2
        assert captured["max_tokens"] == 40

    def test_non_dict_chain_returns_fallback(self):
        client, _ = _make_client("X")
        out = suggest_chain_title(
            [1, 2, 3], client=client, model="m",
            fallback="default-name",
        )
        assert out == "default-name"

    def test_no_client_returns_fallback(self):
        out = suggest_chain_title(
            {}, client=None, model="m", fallback="fb",
        )
        assert out == "fb"

    def test_empty_model_returns_fallback(self):
        client, _ = _make_client("X")
        out = suggest_chain_title(
            {}, client=client, model="", fallback="fb",
        )
        assert out == "fb"

    def test_exception_swallowed_returns_fallback(self):
        client, _ = _make_client(RuntimeError("offline"))
        out = suggest_chain_title(
            {"steps": []}, client=client, model="m",
            fallback="fb-fallback",
        )
        assert out == "fb-fallback"

    def test_empty_response_returns_fallback(self):
        client, _ = _make_client("   \n\t  ")
        out = suggest_chain_title(
            {"steps": []}, client=client, model="m",
            fallback="fb",
        )
        assert out == "fb"

    def test_malformed_response_returns_fallback(self):
        # No `choices` attribute on the response.
        bad_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kw: SimpleNamespace(),
                ),
            ),
        )
        out = suggest_chain_title(
            {"steps": []}, client=bad_client, model="m",
            fallback="fb",
        )
        assert out == "fb"

    def test_long_response_capped(self):
        client, _ = _make_client("X" * 500)
        out = suggest_chain_title(
            {"steps": []}, client=client, model="m",
        )
        assert len(out) <= MAX_TITLE_CHARS
