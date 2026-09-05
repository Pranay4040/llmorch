"""A conversation with the orchestrator.

The thing under test is not the REPL — it is what a later turn knows about an
earlier one, and what it costs to know it. A conversation that remembered by
feeding the previous files back to the planner would work perfectly and be
unusable by the fourth turn, so the tests below pin the cheap form: instructions,
the contract, and one summary per file.
"""

from __future__ import annotations

import json

from llmorch.chat import (
    Conversation,
    FileNote,
    Turn,
    latest_session,
    merge_interfaces,
)
from llmorch.demo.website import build_nodes
from llmorch.types import (
    InterfaceContract,
    LaunchSpec,
    NodeResult,
    NodeState,
    Usage,
)

NODES = {n.id: n for n in build_nodes()}


def _results(*node_ids: str, degraded: set[str] = frozenset()) -> dict[str, NodeResult]:
    return {
        node_id: NodeResult(
            node_id=node_id,
            state=NodeState.DEGRADED if node_id in degraded else NodeState.DONE,
            artifact="" if node_id in degraded else "x" * 4000,
            summary=f"{node_id}: what it does",
            model_id="groq/gpt-oss-120b",
            usage=Usage(prompt_tokens=10, completion_tokens=20),
        )
        for node_id in node_ids
    }


def _conversation(tmp_path, monkeypatch) -> Conversation:
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    return Conversation(session_id="20260101-000000")


# ==========================================================================
# What a turn remembers
# ==========================================================================


def test_the_first_turn_is_a_build_and_the_rest_are_changes(tmp_path, monkeypatch):
    """The only branch in the module: nothing built yet, or something to change."""
    conversation = _conversation(tmp_path, monkeypatch)
    assert not conversation.started

    conversation.record(
        "build a notes app", NODES, _results("server"), InterfaceContract()
    )

    assert conversation.started


def test_memory_holds_summaries_and_never_file_contents(tmp_path, monkeypatch):
    """The design decision the whole feature turns on.

    Feeding the artifacts back would grow every prompt with the project rather
    than with the request, against a 6,000 tokens-per-minute ceiling.
    """
    conversation = _conversation(tmp_path, monkeypatch)
    conversation.record(
        "build it", NODES, _results("server", "index"), InterfaceContract()
    )

    memory = conversation.render_memory()

    assert "server.py" in memory and "server: what it does" in memory
    assert "x" * 100 not in memory
    assert len(memory) < 1000


def test_a_rewritten_file_replaces_its_note(tmp_path, monkeypatch):
    """The next turn must see the project as it stands, not its history."""
    conversation = _conversation(tmp_path, monkeypatch)
    conversation.record("build it", NODES, _results("index"), InterfaceContract())

    revised = _results("index")
    revised["index"].summary = "index: now with a detail link"
    conversation.record("add a link", NODES, revised, InterfaceContract())

    notes = [n for n in conversation.files.values() if n.node_id == "index"]
    assert len(notes) == 1
    assert notes[0].summary == "index: now with a detail link"


def test_a_degraded_node_leaves_the_previous_file_remembered(tmp_path, monkeypatch):
    """The file on disk is still the old one, so the memory must still be too."""
    conversation = _conversation(tmp_path, monkeypatch)
    conversation.record("build it", NODES, _results("index"), InterfaceContract())

    conversation.record(
        "change it", NODES, _results("index", degraded={"index"}), InterfaceContract()
    )

    assert conversation.files["index.html"].summary == "index: what it does"
    assert conversation.turns[-1].degraded == ("index",)


def test_prior_work_is_offered_to_new_nodes_as_summaries(tmp_path, monkeypatch):
    """So a new node can declare `needs: ["server.summary"]` against a file some
    earlier turn wrote — with the artifact left empty, because it is on disk."""
    conversation = _conversation(tmp_path, monkeypatch)
    conversation.record("build it", NODES, _results("server"), InterfaceContract())

    seeded = conversation.seed_results()

    assert seeded["server"].summary == "server: what it does"
    assert seeded["server"].artifact == ""


def test_the_instructions_are_remembered_in_order(tmp_path, monkeypatch):
    conversation = _conversation(tmp_path, monkeypatch)
    for instruction in ("build a notes app", "add tags", "make it dark"):
        conversation.record(instruction, NODES, _results("index"), InterfaceContract())

    memory = conversation.render_memory()

    assert memory.index("1. build a notes app") < memory.index("2. add tags")
    assert memory.index("2. add tags") < memory.index("3. make it dark")


# ==========================================================================
# Persistence
# ==========================================================================


def test_a_conversation_survives_the_process(tmp_path, monkeypatch):
    conversation = _conversation(tmp_path, monkeypatch)
    conversation.record(
        "build it",
        NODES,
        _results("server"),
        InterfaceContract(
            pages=("index.html",),
            launch=LaunchSpec(command=("python", "server.py")),
        ),
    )
    conversation.save()

    restored = Conversation.load("20260101-000000")

    assert restored is not None
    assert [t.instruction for t in restored.turns] == ["build it"]
    assert restored.files["server.py"].summary == "server: what it does"
    assert restored.interface.pages == ("index.html",)
    assert restored.interface.launch.command == ("python", "server.py")


def test_a_conversation_is_written_after_every_turn(tmp_path, monkeypatch):
    """A quota wall costs one turn, not the session."""
    conversation = _conversation(tmp_path, monkeypatch)
    conversation.record("first", NODES, _results("server"), InterfaceContract())
    conversation.save()
    conversation.record("second", NODES, _results("index"), InterfaceContract())
    conversation.save()

    stored = json.loads(conversation.path.read_text(encoding="utf-8"))

    assert [t["instruction"] for t in stored["turns"]] == ["first", "second"]


def test_an_unreadable_conversation_is_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    (tmp_path / "20260101-000000").mkdir(parents=True)
    (tmp_path / "20260101-000000" / "conversation.json").write_text("{ broken")

    assert Conversation.load("20260101-000000") is None


def test_a_conversation_from_a_future_version_is_declined(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    (tmp_path / "s").mkdir(parents=True)
    (tmp_path / "s" / "conversation.json").write_text(json.dumps({"version": 99}))

    assert Conversation.load("s") is None


def test_the_most_recent_session_is_the_one_continued(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    for session_id in ("20260101-000000", "20260301-120000", "20260201-000000"):
        conversation = Conversation(session_id=session_id)
        conversation.record("x", NODES, _results("index"), InterfaceContract())
        conversation.save()

    assert latest_session() == "20260301-120000"


def test_no_sessions_yet_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path / "empty"))
    assert latest_session() is None


# ==========================================================================
# Merging the contract
# ==========================================================================


def test_a_revision_adds_to_the_contract_rather_than_replacing_it():
    """A revision answers about the change, so what it leaves out it is keeping.

    Replacing would silently unserve every route it did not restate — and every
    check downstream measures the artifacts against this contract, so it would
    report the wrong thing with total confidence.
    """
    current = InterfaceContract(
        routes=({"method": "GET", "path": "/api/notes"},),
        pages=("index.html",),
        data_models=({"name": "Note", "fields": {"id": "integer"}},),
        runtime="Python 3.11",
    )
    update = InterfaceContract(
        routes=({"method": "GET", "path": "/api/notes/{id}"},),
        pages=("note.html",),
    )

    merged = merge_interfaces(current, update)

    assert {r["path"] for r in merged.routes} == {"/api/notes", "/api/notes/{id}"}
    assert merged.pages == ("index.html", "note.html")
    assert merged.data_models == current.data_models
    assert merged.runtime == "Python 3.11"


def test_a_restated_route_is_updated_not_duplicated():
    current = InterfaceContract(routes=({"method": "GET", "path": "/api/notes"},))
    update = InterfaceContract(
        routes=({"method": "GET", "path": "/api/notes", "returns": "Note[]"},)
    )

    merged = merge_interfaces(current, update)

    assert len(merged.routes) == 1
    assert merged.routes[0]["returns"] == "Note[]"


def test_a_declared_launch_replaces_an_earlier_one():
    """The launch is a single fact about the project, not a set to accumulate."""
    current = InterfaceContract(launch=LaunchSpec(command=("python", "server.py")))
    update = InterfaceContract(launch=LaunchSpec(command=("node", "server.js")))

    assert merge_interfaces(current, update).launch.command == ("node", "server.js")
    assert merge_interfaces(current, InterfaceContract()).launch.command == (
        "python",
        "server.py",
    )


# ==========================================================================
# Reading the folder back
# ==========================================================================


def test_a_dependency_tree_is_not_walked(tmp_path):
    """`--smoke-install` can leave tens of thousands of files here, and a smoke
    run leaves a SQLite file. Neither is an artifact any model wrote."""
    from llmorch.__main__ import _read_output

    (tmp_path / "server.js").write_text("// mine\n", encoding="utf-8")
    (tmp_path / "notes.db").write_bytes(b"SQLite format 3\x00")
    deep = tmp_path / "node_modules" / "left-pad"
    deep.mkdir(parents=True)
    (deep / "index.js").write_text("// theirs\n", encoding="utf-8")

    files = _read_output(tmp_path)

    assert list(files) == ["server.js"]


def test_the_generated_readme_is_not_an_artifact(tmp_path):
    """`materialize` writes it; no node produced it, so no check should judge it."""
    from llmorch.__main__ import _read_output

    (tmp_path / "README.md").write_text("# Generated project\n", encoding="utf-8")
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")

    assert list(_read_output(tmp_path)) == ["index.html"]


def test_a_folder_that_does_not_exist_yet_reads_as_empty(tmp_path):
    from llmorch.__main__ import _read_output

    assert _read_output(tmp_path / "nope") == {}


# ==========================================================================
# A turn, end to end
# ==========================================================================


def _offline(monkeypatch, tmp_path, revise_response: str) -> None:
    """Run chat against the mock, with a revision answer this test chooses."""
    from llmorch import __main__ as cli
    from llmorch.demo.website import ARTIFACTS
    from llmorch.providers.base import ProviderRegistry
    from llmorch.providers.mock import MockProvider

    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))

    def registry(manifest):
        provider = MockProvider(
            responses=dict(ARTIFACTS), revise_response=revise_response
        )
        built = ProviderRegistry()
        for model in manifest.enabled_models:
            built.register(model.id, provider)
        return built, provider

    monkeypatch.setattr(cli, "_mock_registry", registry)


def _script(monkeypatch, *lines: str) -> None:
    remaining = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(remaining))


def test_a_first_turn_builds_and_a_second_can_change_nothing(
    tmp_path, monkeypatch, capsys
):
    """The whole loop offline: build, then an instruction the planner answers
    with no nodes at all — which is an answer, not a failure."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, revise_response='{"nodes": []}')
    _script(monkeypatch, "build a notes app", "looks good", "/quit")

    args = cli.build_parser().parse_args(["chat"])
    assert args.func(args) == 0

    out = capsys.readouterr().out
    assert "nothing to change" in out

    session_id = out.split("New session ", 1)[1].split(".", 1)[0]
    conversation = Conversation.load(session_id)
    assert [t.instruction for t in conversation.turns] == [
        "build a notes app",
        "looks good",
    ]
    assert conversation.turns[-1].planned == ()
    assert len(conversation.files) == 6


def test_a_second_turn_rewrites_only_what_the_change_needs(
    tmp_path, monkeypatch, capsys
):
    """The reason the whole feature exists: a change to one file must not cost a
    rewrite of six."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, revise_response=_one_node_change())
    _script(monkeypatch, "build a notes app", "add a detail link", "/quit")

    args = cli.build_parser().parse_args(["chat"])
    args.func(args)

    out = capsys.readouterr().out
    session_id = out.split("New session ", 1)[1].split(".", 1)[0]
    conversation = Conversation.load(session_id)

    assert len(conversation.turns[0].planned) == 6
    assert conversation.turns[1].planned == ("index",)
    assert len(conversation.files) == 6  # rewritten, not added to


def _one_node_change() -> str:
    return json.dumps(
        {
            "nodes": [
                {
                    "id": "index",
                    "title": "Index page, revised",
                    "role": "frontend",
                    "spec": "Add a detail link.",
                    "output_path": "index.html",
                    "output_kind": "code",
                    "est_output_tokens": 400,
                }
            ]
        }
    )


# ==========================================================================
# The shortest way in
# ==========================================================================


def test_a_bare_invocation_opens_a_session(tmp_path, monkeypatch, capsys):
    """`llmorch` on its own used to be an argparse error."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, revise_response='{"nodes": []}')
    _script(monkeypatch, "/quit")

    assert cli.main([]) == 0
    assert "New session" in capsys.readouterr().out


def test_an_opening_instruction_is_said_for_you(tmp_path, monkeypatch, capsys):
    """`llmorch "build a notes app"` is one line instead of two."""
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, revise_response='{"nodes": []}')
    _script(monkeypatch, "/quit")

    assert cli.main(["build a notes app"]) == 0

    out = capsys.readouterr().out
    session_id = out.split("New session ", 1)[1].split(".", 1)[0]
    conversation = Conversation.load(session_id)
    assert [t.instruction for t in conversation.turns] == ["build a notes app"]


def test_cli_is_the_same_door_as_chat(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, revise_response='{"nodes": []}')
    _script(monkeypatch, "/quit")

    assert cli.main(["cli"]) == 0
    assert "New session" in capsys.readouterr().out


def test_flags_alone_still_open_a_session(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    _offline(monkeypatch, tmp_path, revise_response='{"nodes": []}')
    _script(monkeypatch, "/quit")

    assert cli.main(["--smoke"]) == 0
    assert "New session" in capsys.readouterr().out


def test_every_subcommand_still_routes_to_itself():
    """The shortcut must not swallow a real command: `llmorch run` is a run."""
    from llmorch import __main__ as cli

    parser = cli.build_parser()
    assert "run" in parser.subcommand_names
    assert "chat" in parser.subcommand_names and "cli" in parser.subcommand_names

    for name, expected in (
        ("run", cli.cmd_run),
        ("plan", cli.cmd_plan),
        ("quota", cli.cmd_quota),
        ("doctor", cli.cmd_doctor),
        ("chat", cli.cmd_chat),
        ("cli", cli.cmd_chat),
    ):
        assert parser.parse_args([name]).func is expected
