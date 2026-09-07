"""Telling an instruction from a question, and answering the question.

Two things are under test, and only one of them costs anything.

The classifier is the free half, and its asymmetry is the point: reading an
instruction as a question wastes a request and builds nothing, while reading a
question as an instruction is what the system did before this existed. So the
table below pins the shapes that are recognised, and — just as deliberately —
that everything unrecognised still falls through to a build.

The answering half is one request that writes no files. What the tests hold it
to is the accounting: it goes through the governor like everything else, it
lands in the ledger under its own purpose, it never grows the prompt with the
project, and a question that would not fit is shortened rather than refused.
"""

from __future__ import annotations

import json

import pytest

from llmorch.chat import Conversation, FileNote, Turn
from llmorch.negotiate.answer import (
    Answer,
    AnswerError,
    answer,
    build_answer_prompt,
    clean_answer,
    files_named,
    pick_answerer,
    take_excerpts,
)
from llmorch.negotiate.intent import Intent, classify
from llmorch.providers.base import ProviderRegistry
from llmorch.providers.mock import FaultMode, MockProvider
from llmorch.quota.estimator import TokenEstimator
from llmorch.quota.governor import Governor
from llmorch.quota.store import LedgerStore
from llmorch.registry.manifest import load_manifest
from llmorch.engine.blackboard import Blackboard
from llmorch.engine.worker import WorkerDeps
from llmorch.engine.health import HealthTracker
from llmorch.types import InterfaceContract, NodeResult, NodeState


# ==========================================================================
# Which lane a line belongs in
# ==========================================================================


@pytest.mark.parametrize(
    "line",
    [
        "what does the server do?",
        "why is the page so slow?",
        "how does the frontend get its data?",
        "does it store anything on disk?",
        "is there a test for the schema",
        "which model wrote index.html?",
        "explain the launch command",
        "tell me what the styles are doing",
        "walk me through the delete flow",
        "what's left to do?",
        "should the schema have an index",
    ],
)
def test_a_question_is_answered_not_built(line):
    assert classify(line) is Intent.ASK


@pytest.mark.parametrize(
    "line",
    [
        "build a notes app",
        "add tags to each note",
        "now add a delete button",
        "make the header sticky",
        "use SQLite instead of a JSON file",
        "the list should be sorted by date",
        "can you add a delete button?",
        "could you please make the header sticky?",
        "please remove the sidebar",
        "how about a dark mode",
        "what about pagination?",
        "what if the list were paginated",
        "tidy up the spacing",
        "show the created date on each note",
    ],
)
def test_an_instruction_is_built(line):
    """Including the politely-phrased ones. "can you add a delete button?" ends
    in a question mark and is not a question."""
    assert classify(line) is Intent.BUILD


@pytest.mark.parametrize(
    "line",
    [
        "ok",
        "Thanks!",
        "looks good",
        "LGTM",
        "nice work",
        "makes sense",
        "  ",
        # Two acknowledgements in one breath. Found by running the live session
        # and watching this line buy a planning request.
        "thanks, that helps",
        "ok, makes sense",
        "great, thanks!",
    ],
)
def test_an_acknowledgement_is_neither(line):
    assert classify(line) is Intent.REMARK


def test_an_instruction_after_a_thank_you_is_still_an_instruction():
    """Every clause has to be an acknowledgement for the line to be one."""
    assert classify("thanks, now add tags") is Intent.BUILD
    assert classify("ok, remove the sidebar") is Intent.BUILD
    assert classify("nice — what does it store?") is Intent.ASK


def test_anything_unrecognised_falls_through_to_a_build():
    """The default is the old behaviour, which is what makes the classifier safe
    to be wrong about: a shape it does not know is an instruction, exactly as it
    was before this module existed."""
    assert classify("notes, but for recipes") is Intent.BUILD
    assert classify("dark mode") is Intent.BUILD
    assert classify("server.py is doing too much") is Intent.BUILD


# ==========================================================================
# What a question is allowed to carry
# ==========================================================================


def test_a_question_carries_only_the_files_it_names():
    available = ["server.py", "index.html", "style.css"]
    assert files_named("what does server.py do?", available) == ("server.py",)
    assert files_named("what does the server do?", available) == ("server.py",)
    assert files_named("how do the pages look?", available) == ()


def test_an_ambiguous_name_quotes_nothing():
    """Two files share the stem `app`. Picking one would answer confidently
    about the wrong file, which is worse than answering from the summaries."""
    assert files_named("what does app do?", ["app.py", "app.js"]) == ()


def test_quoted_files_are_trimmed_to_a_budget_and_say_so():
    files = {"server.py": "x" * 5000, "index.html": "y" * 5000}
    excerpts = take_excerpts(files, ("server.py", "index.html"), budget=2000)

    assert set(excerpts) == {"server.py", "index.html"}
    assert all(len(text) < 2000 for text in excerpts.values())
    assert all("truncated" in text for text in excerpts.values())


def test_the_prompt_never_carries_a_file_that_was_not_asked_about():
    system, user = build_answer_prompt(
        "what does the server do?",
        memory="## The files\n- `server.py` — serves the API",
        interface_text="## Interface contract",
        excerpts={"server.py": "print('hello')"},
    )
    assert "print('hello')" in user
    assert "index.html" not in user
    assert "no files" in system.lower() or "not changing it" in system.lower()


def test_an_answer_cannot_move_the_cursor():
    """Model output on its way to a terminal is data. A reply that repainted the
    screen would be the dashboard's escaping bug with a different exit."""
    cleaned = clean_answer("\x1b[2Jwiped\x07 the \x00screen")
    assert cleaned == "wiped the screen"


def test_a_long_answer_is_capped():
    assert len(clean_answer("word " * 5000)) < 6200


# ==========================================================================
# What answering costs
# ==========================================================================


def _deps(tmp_path, *, provider: MockProvider | None = None, ledger=None):
    manifest = load_manifest()
    provider = provider or MockProvider()
    registry = ProviderRegistry()
    for model in manifest.enabled_models:
        registry.register(model.id, provider)
    return (
        WorkerDeps(
            manifest=manifest,
            governor=Governor(manifest),
            registry=registry,
            estimator=TokenEstimator(),
            health=HealthTracker(threshold=2),
            blackboard=Blackboard(interface=InterfaceContract()),
            ledger=ledger,
            run_id="ask-test",
        ),
        provider,
        manifest,
    )


@pytest.mark.asyncio
async def test_a_question_costs_one_request_and_writes_nothing(tmp_path):
    deps, provider, manifest = _deps(tmp_path)
    model_id = pick_answerer(manifest, sorted(deps.registry.model_ids))

    reply = await answer(
        "what does the server do?",
        deps=deps,
        model_id=model_id,
        memory="## The files\n- `server.py` — serves the API",
    )

    assert isinstance(reply, Answer)
    assert reply.text
    assert len(provider.calls) == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_a_question_queues_behind_the_work(tmp_path):
    """NORMAL, not HIGH. The reserve exists so a critical-path retry is not
    crowded out, and no question is on the critical path."""
    deps, provider, manifest = _deps(tmp_path)
    model_id = pick_answerer(manifest, sorted(deps.registry.model_ids))

    seen = []
    original = deps.governor.try_acquire

    def spy(model, prompt, completion, priority=None, **kw):
        seen.append(priority)
        return original(model, prompt, completion, priority=priority, **kw)

    deps.governor.try_acquire = spy  # type: ignore[method-assign]
    await answer("what is this?", deps=deps, model_id=model_id, memory="")

    from llmorch.types import Priority

    assert seen == [Priority.NORMAL]


@pytest.mark.asyncio
async def test_a_question_is_on_the_record(tmp_path):
    """A request against a real daily allowance that nobody wrote down is quota
    tomorrow's process believes it still holds."""
    store = LedgerStore(tmp_path / "state.db").open()
    deps, _, manifest = _deps(tmp_path, ledger=store)
    model_id = pick_answerer(manifest, sorted(deps.registry.model_ids))

    await answer("what does it do?", deps=deps, model_id=model_id, memory="")

    rows = store.recent(10, run_id="ask-test")
    store.close()
    assert [r.purpose for r in rows] == ["answer"]
    assert rows[0].ok


@pytest.mark.asyncio
async def test_an_empty_reply_is_a_failure_rather_than_a_blank_line(tmp_path):
    deps, _, manifest = _deps(tmp_path, provider=MockProvider(answer_response=""))
    model_id = pick_answerer(manifest, sorted(deps.registry.model_ids))

    with pytest.raises(AnswerError):
        await answer("what is this?", deps=deps, model_id=model_id, memory="")


@pytest.mark.asyncio
async def test_a_failed_question_gives_its_reservation_back(tmp_path):
    """The reservation is made on an estimate before the call. A failure that
    kept it would ration the day against a request that never happened."""
    from llmorch.errors import TransportError

    class Dead:
        name = "dead"

        async def chat(self, request):
            raise TransportError("mock: connection reset", status=503)

        async def count_tokens(self, request):
            return None

    deps, _, manifest = _deps(tmp_path)
    model_id = pick_answerer(manifest, sorted(deps.registry.model_ids))
    deps.registry.register(model_id, Dead())

    before = deps.governor.headroom()[model_id].requests_used
    with pytest.raises(AnswerError):
        await answer("what is this?", deps=deps, model_id=model_id, memory="")

    assert deps.governor.headroom()[model_id].requests_used == before


@pytest.mark.asyncio
async def test_a_question_too_big_to_send_is_shortened_not_refused(tmp_path):
    """Groq leaves roughly 3,100 tokens for a prompt. A question that named a
    large file used to be `UNSERVABLE`; dropping the excerpt answers it from
    summaries instead, which is worse than the file and better than nothing."""
    deps, provider, manifest = _deps(tmp_path)
    model_id = "groq/gpt-oss-120b"

    reply = await answer(
        "what does server.py do?",
        deps=deps,
        model_id=model_id,
        memory="## The files\n- `server.py` — serves the API",
        excerpts={"server.py": "x" * 200_000},
    )

    assert reply.files_read == ()
    assert reply.text


# ==========================================================================
# A question in a conversation
# ==========================================================================


def _conversation() -> Conversation:
    conversation = Conversation(session_id="20260101-000000")
    conversation.files["server.py"] = FileNote(
        path="server.py",
        node_id="server",
        role="backend",
        summary="Serves /api/items from SQLite.",
        model_id="groq/gpt-oss-120b",
    )
    conversation.turns.append(Turn(instruction="build a notes app", kind="build"))
    return conversation


def test_the_planner_is_never_shown_a_question():
    """The planner's memory is "what you have been asked so far". A question was
    never asked of it, and leaving it in invites a plan that answers it in a
    file."""
    conversation = _conversation()
    conversation.record_said("what does the server do?", kind="ask", answer="It serves.")
    conversation.record_said("thanks", kind="remark")

    memory = conversation.render_memory()
    assert "build a notes app" in memory
    assert "what does the server do?" not in memory
    assert "thanks" not in memory


def test_the_answerer_is_shown_who_wrote_what():
    """A different document from the planner's, for a different reader."""
    text = _conversation().render_for_answer()
    assert "server.py" in text
    assert "groq/gpt-oss-120b" in text


def test_a_follow_up_can_see_the_last_answer():
    conversation = _conversation()
    conversation.record_said("what does it store?", kind="ask", answer="Items, in SQLite.")

    exchanges = conversation.render_exchanges()
    assert "what does it store?" in exchanges
    assert "Items, in SQLite." in exchanges


def test_the_thread_of_questions_is_bounded():
    """The one part of the memory that would otherwise grow without limit."""
    conversation = _conversation()
    for index in range(10):
        conversation.record_said(f"q{index}", kind="ask", answer="a" * 2000)

    exchanges = conversation.render_exchanges(limit=2)
    assert "q9" in exchanges and "q8" in exchanges
    assert "q7" not in exchanges
    assert len(exchanges) < 1200


def test_a_session_written_before_questions_existed_still_loads(tmp_path, monkeypatch):
    """`kind` and `answer` are additive, which is why the file version did not
    move: bumping it would decline every session already on disk."""
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    session = tmp_path / "20260101-000000"
    session.mkdir()
    (session / "conversation.json").write_text(
        json.dumps(
            {
                "version": 1,
                "session_id": "20260101-000000",
                "turns": [{"instruction": "build a notes app", "planned": ["server"]}],
                "files": [{"path": "server.py", "node_id": "server"}],
                "interface": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = Conversation.load("20260101-000000")
    assert [t.kind for t in loaded.turns] == ["build"]
    assert loaded.instructions


# ==========================================================================
# The three lanes, through the CLI
# ==========================================================================


def _offline(monkeypatch, tmp_path, *, answer_response: str = "It serves the API."):
    from llmorch import __main__ as cli
    from llmorch.demo.website import ARTIFACTS

    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))

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


def test_a_question_in_a_session_is_answered_and_builds_nothing(
    tmp_path, monkeypatch, capsys
):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, answer_response="It serves /api/items from SQLite.")
    _script(monkeypatch, "build a notes app", "what does server.py do?", "/quit")

    assert cli.main(["chat"]) == 0

    out = capsys.readouterr().out
    assert "It serves /api/items from SQLite." in out

    session_id = out.split("Session ", 1)[1].split(" saved", 1)[0]
    conversation = Conversation.load(session_id)
    assert [t.kind for t in conversation.turns] == ["build", "ask"]
    assert conversation.turns[-1].answer
    assert conversation.turns[-1].planned == ()
    assert len(conversation.files) == 6  # unchanged by the question


def test_a_remark_spends_nothing_at_all(tmp_path, monkeypatch, capsys):
    """The reason the classifier exists: "looks good" used to buy a planning
    request in order to be told there was nothing to plan."""
    from llmorch import __main__ as cli

    provider = _offline(monkeypatch, tmp_path)
    _script(monkeypatch, "build a notes app", "looks good", "/quit")

    assert cli.main(["chat"]) == 0
    built = len(provider.calls)

    out = capsys.readouterr().out
    assert "noted" in out

    session_id = out.split("Session ", 1)[1].split(" saved", 1)[0]
    conversation = Conversation.load(session_id)
    assert [t.kind for t in conversation.turns] == ["build", "remark"]

    # Every call the mock saw belongs to the build; the remark added none.
    assert built == len(provider.calls)


def test_the_lane_can_be_named_outright(tmp_path, monkeypatch, capsys):
    """`/ask` takes the decision away from the heuristic, which is what lets the
    heuristic stay small."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, answer_response="Answered on request.")
    _script(monkeypatch, "build a notes app", "/ask add a delete button", "/quit")

    assert cli.main(["chat"]) == 0

    out = capsys.readouterr().out
    assert "Answered on request." in out

    session_id = out.split("Session ", 1)[1].split(" saved", 1)[0]
    conversation = Conversation.load(session_id)
    assert [t.kind for t in conversation.turns] == ["build", "ask"]


def test_ask_answers_the_most_recent_session_from_a_shell(
    tmp_path, monkeypatch, capsys
):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, answer_response="Six files, one server.")
    _script(monkeypatch, "/quit")
    cli.main(["build a notes app"])
    capsys.readouterr()

    assert cli.main(["ask", "what did you build?"]) == 0
    assert "Six files, one server." in capsys.readouterr().out


def test_ask_with_no_session_says_so_rather_than_inventing_one(
    tmp_path, monkeypatch, capsys
):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path)
    assert cli.main(["ask", "what did you build?"]) == 2
    assert "No session" in capsys.readouterr().err


def test_a_question_records_no_artifacts(tmp_path, monkeypatch, capsys):
    """A question asked before anything is built is still a question, and the
    session is still empty afterwards."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, answer_response="Nothing built yet.")
    _script(monkeypatch, "what have you built?", "/quit")

    assert cli.main(["chat"]) == 0

    out = capsys.readouterr().out
    session_id = out.split("Session ", 1)[1].split(" saved", 1)[0]
    conversation = Conversation.load(session_id)
    assert not conversation.started
    assert [t.kind for t in conversation.turns] == ["ask"]
