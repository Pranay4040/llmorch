"""The question asked before the first thing is said.

Three modes, and none of them is a separate implementation: chat is the question
lane as a standing choice, one agent is every job pinned at once, and a crew is
what the system does anyway. So the tests are mostly about the *edges* — that
nobody is asked who is not there to answer, that a mode is a default rather than
a lock, and that the reduction one agent makes is stated rather than discovered.
"""

from __future__ import annotations

import pytest

from llmorch import mode as mode_module
from llmorch import settings as settings_module
from llmorch.chat import Conversation
from llmorch.mode import Mode, ask, describe, parse, render_menu


# ==========================================================================
# Reading an answer
# ==========================================================================


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("1", Mode.CHAT),
        ("2", Mode.AGENT),
        ("3", Mode.CREW),
        ("chat", Mode.CHAT),
        ("AGENT", Mode.AGENT),
        ("cre", Mode.CREW),
        ("  crew  ", Mode.CREW),
    ],
)
def test_the_number_or_the_name_both_work(typed, expected):
    """The menu shows numbers, and people type what they see."""
    assert parse(typed) is expected


@pytest.mark.parametrize("typed", ["", "   ", "9", "0", "nonsense", None])
def test_anything_else_is_not_an_answer(typed):
    assert parse(typed) is None


def test_an_unreadable_answer_takes_the_default():
    assert ask(reader=lambda _: "banana") is Mode.CREW
    assert ask(reader=lambda _: "") is Mode.CREW


def test_a_chosen_answer_is_taken():
    assert ask(reader=lambda _: "1") is Mode.CHAT


def test_walking_away_from_the_question_takes_the_default():
    def gone(_):
        raise EOFError

    assert ask(reader=gone) is Mode.CREW


def test_nobody_is_asked_who_is_not_there_to_answer(monkeypatch, capsys):
    """A menu printed at a pipe would consume the first line of input as the
    answer to a question the pipe never saw."""

    class NotATty:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", NotATty())

    assert ask() is Mode.CREW
    assert capsys.readouterr().out == ""


def test_the_menu_names_what_each_mode_gives_up():
    menu = render_menu()
    assert "nothing gets built" in menu
    assert "no second vendor to ask" in menu
    assert "(default)" in menu

    # The reduction one agent makes is stated, not discovered from a report with
    # no review section in it.
    assert "no cross-vendor review" in describe(Mode.AGENT)


# ==========================================================================
# What each mode does to a session
# ==========================================================================


def _offline(monkeypatch, tmp_path, *, answer_response: str = "An answer."):
    from llmorch import __main__ as cli
    from llmorch.demo.website import ARTIFACTS
    from llmorch.providers.base import ProviderRegistry
    from llmorch.providers.mock import MockProvider

    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("LLMORCH_SETTINGS", str(tmp_path / "settings.json"))

    provider = MockProvider(
        responses=dict(ARTIFACTS),
        revise_response='{"nodes": []}',
        answer_response=answer_response,
    )

    def registry(manifest):
        built = ProviderRegistry()
        for model in manifest.enabled_models:
            built.register(model.id, provider)
        return built, provider

    monkeypatch.setattr(cli, "_mock_registry", registry)
    return provider


def _script(monkeypatch, *lines: str) -> None:
    remaining = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(remaining))


def _session_id(out: str) -> str:
    return out.split("Session ", 1)[1].split(" saved", 1)[0]


def test_chat_mode_answers_an_instruction_instead_of_building_it(
    tmp_path, monkeypatch, capsys
):
    """The whole point of asking: someone who wanted to ask a question and got a
    six-file project has already paid for the misunderstanding."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, answer_response="I would build it like this.")
    _script(monkeypatch, "build a notes app", "/quit")

    assert cli.main(["start", "--mock", "--mode", "chat"]) == 0

    out = capsys.readouterr().out
    assert "I would build it like this." in out

    conversation = Conversation.load(_session_id(out))
    assert [t.kind for t in conversation.turns] == ["ask"]
    assert not conversation.files
    assert not (tmp_path / conversation.session_id / "output").exists()


def test_build_still_overrides_chat_mode(tmp_path, monkeypatch, capsys):
    """A mode is a default, not a lock."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    _script(monkeypatch, "/build a notes app", "/quit")

    assert cli.main(["start", "--mock", "--mode", "chat"]) == 0

    conversation = Conversation.load(_session_id(capsys.readouterr().out))
    assert [t.kind for t in conversation.turns] == ["build"]
    assert conversation.files


def test_one_agent_writes_every_file_itself(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    _script(monkeypatch, "build a notes app", "/quit")

    assert cli.main(["start", "--mock", "--mode", "agent"]) == 0

    conversation = Conversation.load(_session_id(capsys.readouterr().out))
    authors = {note.model_id for note in conversation.files.values()}
    assert len(authors) == 1


def test_a_crew_spreads_the_work(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    _script(monkeypatch, "build a notes app", "/quit")

    assert cli.main(["start", "--mock", "--mode", "crew"]) == 0

    conversation = Conversation.load(_session_id(capsys.readouterr().out))
    authors = {note.model_id for note in conversation.files.values()}
    assert len(authors) > 1


def test_one_agent_says_what_it_gives_up(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    _script(monkeypatch, "build a notes app", "/quit")
    cli.main(["start", "--mock", "--mode", "agent", "--tables"])

    out = capsys.readouterr().out
    assert "writes every file" in out
    assert "no second vendor to ask" in out


# ==========================================================================
# Where the answer comes from
# ==========================================================================


def test_a_resumed_session_keeps_the_mode_it_was_opened_in(
    tmp_path, monkeypatch, capsys
):
    """A conversation whose files a crew wrote is not one that a single agent
    has been having."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    _script(monkeypatch, "build a notes app", "/quit")
    cli.main(["start", "--mock", "--mode", "agent"])
    session_id = _session_id(capsys.readouterr().out)

    assert Conversation.load(session_id).mode == "agent"

    # Reopened with no mode named at all: it must not be asked again.
    _script(monkeypatch, "/quit")
    cli.main(["start", "--mock", "--session", session_id])

    assert "One agent" in capsys.readouterr().out


def test_a_saved_setting_answers_the_question(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    settings_module.Settings(live=False, mode="chat").save()
    _script(monkeypatch, "/quit")

    cli.main(["start", "--mock"])
    assert "Chat" in capsys.readouterr().out


def test_the_flag_wins_over_the_saved_setting(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    settings_module.Settings(live=False, mode="chat").save()
    _script(monkeypatch, "/quit")

    cli.main(["start", "--mock", "--mode", "crew"])
    assert "A crew" in capsys.readouterr().out


def test_the_mode_survives_the_settings_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMORCH_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))

    settings_module.Settings(mode="agent").save()
    assert settings_module.load().mode == "agent"


def test_an_unknown_saved_mode_falls_through_to_being_asked():
    """`parse` returning None is what makes the question happen, so a settings
    file with nonsense in it must not silently pick a mode."""
    assert mode_module.parse("multi-agentic-super-mode") is None
