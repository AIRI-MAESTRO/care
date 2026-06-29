"""Tests for the key-based UI localization layer (care.runtime.i18n).

The active language is a process-global, so each test sets what it needs and
the autouse fixture restores the Russian default afterwards (so the state
never leaks into other test modules' screen rendering).
"""

from __future__ import annotations

import pytest

from care.config import CareConfig
from care.runtime import i18n


@pytest.fixture(autouse=True)
def _restore_ru():
    yield
    i18n.set_ui_language("ru")


def test_default_is_russian():
    i18n.set_ui_language("ru")
    assert i18n.get_ui_language() == "ru"
    assert i18n.t("chat.mode.chat") == "Генерация агентной цепочки под задачу"
    assert i18n.t("settings.action.save") == "Сохранить"


def test_english_selected():
    i18n.set_ui_language("en")
    assert i18n.get_ui_language() == "en"
    assert i18n.t("chat.mode.chat") == "Agent chain generation for your task"
    assert i18n.t("chat.mode.libraryQuick") == "My chains library"
    assert i18n.t("chat.mode.evolutionQuick") == "More about evolution"
    assert i18n.t("chat.mode.skillStub") == "Add as a coding-agent skill"
    assert i18n.t("chat.chainBar.titleAction") == "Choose your next step:"
    assert i18n.t("chat.revise.saveLabel") == "Apply"
    assert i18n.t("chat.revise.applied").startswith("✓ Edit applied")
    assert "More about evolution" in i18n.t("chat.welcomeTail.chat")
    assert i18n.t("settings.action.save") == "Save"


def test_russian_library_quick_label():
    i18n.set_ui_language("ru")
    assert i18n.t("chat.mode.libraryQuick") == "Библиотека моих цепочек"


def test_unknown_and_none_fall_back_to_russian():
    i18n.set_ui_language("fr")
    assert i18n.get_ui_language() == "ru"
    i18n.set_ui_language(None)
    assert i18n.get_ui_language() == "ru"
    i18n.set_ui_language("EN")  # case-insensitive
    assert i18n.get_ui_language() == "en"


def test_interpolation_fills_named_placeholders():
    i18n.set_ui_language("en")
    assert (
        i18n.t("chat.mode.current", label="Chat", mode="ad_hoc")
        == "Current mode: Chat (ad_hoc)."
    )
    # The `{arg!r}` conversion is preserved through str.format.
    assert i18n.t("chat.mode.unknown", arg="xyz").startswith(
        "Unknown mode 'xyz'."
    )


def test_missing_key_surfaces_the_key():
    i18n.set_ui_language("en")
    assert i18n.t("nope.not.a.real.key") == "nope.not.a.real.key"


def test_missing_param_leaves_template_untouched():
    i18n.set_ui_language("en")
    # No `label`/`mode` supplied — must not raise, returns the raw template.
    assert i18n.t("chat.mode.current") == "Current mode: {label} ({mode})."


def test_catalogs_share_the_same_key_set():
    """ru.json and en.json must cover identical keys so no string can be
    English-only (or Russian-only) by accident."""
    ru_keys = set(i18n._catalog("ru"))
    en_keys = set(i18n._catalog("en"))
    assert ru_keys == en_keys, {
        "ru_only": sorted(ru_keys - en_keys),
        "en_only": sorted(en_keys - ru_keys),
    }


def test_command_blurbs_match_en_catalog():
    """English fallback blurbs in ChatScreen stay in lockstep with en.json."""
    import ast
    from pathlib import Path

    en_cmd = i18n._catalog("en")
    text = Path("care/screens/chat.py").read_text()
    start = text.index("_COMMAND_BLURBS: dict[str, str] = {")
    end = text.index("\n    }", start) + len("\n    }")
    dict_src = text[start + len("_COMMAND_BLURBS: dict[str, str] = ") : end]
    blurbs = ast.literal_eval(dict_src.strip())
    mismatches = {
        k: (blurbs[k], en_cmd[f"chat.cmd.{k}"])
        for k in blurbs
        if blurbs[k] != en_cmd.get(f"chat.cmd.{k}")
    }
    assert not mismatches, mismatches


def test_help_mode_labels_localized():
    i18n.set_ui_language("en")
    assert i18n.t("chat.help.modeLabelInteractive") == "Interactive"
    assert i18n.t("chat.help.modeLabelProduction") == "Production"
    i18n.set_ui_language("ru")
    assert i18n.t("chat.help.modeLabelInteractive") == "Интерактивный"


def test_config_ui_language_defaults_ru_and_is_separate_from_agent_language():
    cfg = CareConfig()
    assert cfg.defaults.ui_language == "ru"  # TUI default Russian
    assert cfg.defaults.language == "en"  # agent (CARL) language untouched
