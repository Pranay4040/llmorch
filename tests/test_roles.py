"""Choosing which model does which job.

The assignment normally decides this by fitness, remaining quota and an even
split, and it is usually right. This is the override — and what the tests hold
it to is the *boundary* of the override, because a preference that quietly
disabled the machinery underneath it would be worse than no preference at all.

So: a pin binds the assignment and nothing else. Failover still runs its whole
ladder, because a model that has tripped its circuit breaker is not the one
anybody meant to insist on. A reviewer pin cannot put an author in review of its
own vendor. And a pin that cannot physically serve a node falls back to the
automatic choice out loud, rather than degrading the node to honour it literally.
"""

from __future__ import annotations

import argparse

import pytest

from llmorch import settings as settings_module
from llmorch.engine.graph import TaskGraph
from llmorch.engine.verify import pick_reviewer
from llmorch.negotiate.answer import pick_answerer
from llmorch.negotiate.decompose import pick_planner
from llmorch.negotiate.reconcile import ReconcileInput, reconcile
from llmorch.registry.manifest import load_manifest
from llmorch.demo.website import build_nodes
from llmorch.types import Role, TaskNode

MANIFEST = load_manifest()
EVERY_MODEL = sorted(m.id for m in MANIFEST.enabled_models)


def _input(nodes=None, **kw) -> ReconcileInput:
    return ReconcileInput(
        graph=TaskGraph.build(nodes or build_nodes()),
        manifest=MANIFEST,
        candidates=list(EVERY_MODEL),
        **kw,
    )


# ==========================================================================
# The assignment
# ==========================================================================


def test_a_pinned_role_goes_to_the_model_that_was_chosen():
    plan = reconcile(_input(pins={Role.FRONTEND: "openrouter/minimax-m3"}))

    frontend = [
        node_id
        for node_id, node in TaskGraph.build(build_nodes()).nodes.items()
        if node.role is Role.FRONTEND
    ]
    assert frontend, "the demo graph should have frontend nodes to pin"
    for node_id in frontend:
        assert plan.assignments[node_id].model_id == "openrouter/minimax-m3"


def test_the_unpinned_jobs_are_still_chosen_automatically():
    """A pin is not a takeover: everything else is assigned as before."""
    plan = reconcile(_input(pins={Role.FRONTEND: "openrouter/minimax-m3"}))

    others = {
        a.model_id
        for node_id, a in plan.assignments.items()
        if TaskGraph.build(build_nodes()).nodes[node_id].role is not Role.FRONTEND
    }
    assert others - {"openrouter/minimax-m3"}


def test_a_pin_survives_the_swap_pass():
    """2-opt improves the total score by trading assignments. A pinned node has
    no second option to trade into, which is what makes this hold by
    construction rather than by a special case."""
    pinned = "openrouter/north-mini-code"
    plan = reconcile(_input(pins={Role.BACKEND: pinned}))

    graph = TaskGraph.build(build_nodes())
    backend = [n for n, node in graph.nodes.items() if node.role is Role.BACKEND]
    assert all(plan.assignments[n].model_id == pinned for n in backend)


def test_a_pin_that_cannot_serve_the_node_falls_back_out_loud():
    """Honouring it literally would degrade the node, which is a worse answer to
    "I prefer this model" than doing the work with a note attached."""
    huge = TaskNode(
        id="huge",
        title="an enormous file",
        role=Role.BACKEND,
        spec="x",
        output_path="huge.py",
        # Past every Groq model's 4,096-token output cap, inside Gemini's.
        est_output_tokens=9000,
    )
    plan = reconcile(
        _input(nodes=[huge], pins={Role.BACKEND: "groq/gpt-oss-120b"})
    )

    assert "huge" in plan.assignments  # it still got done
    assert any("assigned automatically" in note for note in plan.notes)


def test_a_pin_naming_a_model_outside_the_roster_is_noted():
    plan = reconcile(
        ReconcileInput(
            graph=TaskGraph.build(build_nodes()),
            manifest=MANIFEST,
            candidates=["groq/gpt-oss-120b"],
            pins={Role.BACKEND: "gemini/3.6-flash"},
        )
    )
    assert any("not in this run's roster" in note for note in plan.notes)
    assert all(a.model_id == "groq/gpt-oss-120b" for a in plan.assignments.values())


# ==========================================================================
# The three jobs no node ever carries
# ==========================================================================


def test_a_pin_chooses_the_planner():
    assert pick_planner(MANIFEST, EVERY_MODEL, prefer="groq/qwen3-27b") == "groq/qwen3-27b"


def test_a_pin_chooses_who_answers_questions():
    assert (
        pick_answerer(MANIFEST, EVERY_MODEL, prefer="groq/gpt-oss-20b")
        == "groq/gpt-oss-20b"
    )


def test_being_unreachable_outranks_being_chosen():
    """A pinned model that is not in this run's candidates is not an error and
    not a refusal — the job still gets done by whoever can."""
    reachable = ["groq/gpt-oss-120b", "groq/qwen3-27b"]
    chosen = pick_planner(MANIFEST, reachable, prefer="gemini/3.6-flash")

    assert chosen in reachable


def test_a_reviewer_pin_never_beats_the_cross_vendor_rule():
    """The rule this project enforces in code rather than asking for in a
    prompt. A same-family reviewer shares the author's blind spots, and
    self-review in particular re-approves its own mistake."""
    chosen = pick_reviewer(
        MANIFEST,
        author_model_id="groq/gpt-oss-120b",
        candidates=EVERY_MODEL,
        prefer_model="groq/qwen3-27b",  # the author's own vendor
    )

    assert chosen is not None
    assert MANIFEST.vendor_of(chosen) != "groq"


def test_a_reviewer_pin_is_honoured_when_it_is_allowed_to_be():
    chosen = pick_reviewer(
        MANIFEST,
        author_model_id="groq/gpt-oss-120b",
        candidates=EVERY_MODEL,
        prefer_model="gemini/3.6-flash",
    )
    assert chosen == "gemini/3.6-flash"


# ==========================================================================
# What the file may say
# ==========================================================================


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMORCH_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path / "runs"))


def test_pins_survive_the_process():
    settings_module.Settings(
        role_models={"backend": "groq/gpt-oss-120b", "review": "gemini/3.6-flash"}
    ).save()

    loaded = settings_module.load()
    assert loaded.role_models == {
        "backend": "groq/gpt-oss-120b",
        "review": "gemini/3.6-flash",
    }


def test_a_cleared_dropdown_leaves_no_residue():
    """"Automatic" is the absence of a pin, not a pin to nothing."""
    chosen = settings_module.from_dict({"role_models": {"backend": "", "review": None}})
    assert chosen.role_models == {}


def test_a_pin_that_is_not_two_strings_is_dropped():
    chosen = settings_module.from_dict(
        {"role_models": {"backend": ["a", "b"], 7: "x", "frontend": "groq/qwen3-27b"}}
    )
    assert chosen.role_models == {"frontend": "groq/qwen3-27b"}


# ==========================================================================
# Through the CLI
# ==========================================================================


def _args(**kw) -> argparse.Namespace:
    base = dict(
        task="x", live=False, providers="groq", review="code", max_nodes=10,
        concurrency=4, allow_paid=False, max_usd=0, models=(), role_models={},
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_a_stale_pin_is_dropped_with_a_note_rather_than_raising():
    """The file was written when models.yaml said something else, and a retired
    model must not be the reason a session refuses to start."""
    from llmorch import __main__ as cli

    warnings: list[str] = []
    pins = cli._pins(_args(role_models={"backend": "groq/retired-yesterday"}),
                     MANIFEST, warnings)

    assert pins == {}
    assert any("not in this run's roster" in w for w in warnings)


def test_an_unknown_role_is_dropped_with_a_note():
    from llmorch import __main__ as cli

    warnings: list[str] = []
    pins = cli._pins(_args(role_models={"chief architect": "groq/gpt-oss-120b"}),
                     MANIFEST, warnings)

    assert pins == {}
    assert any("unknown role" in w for w in warnings)


def test_role_names_are_read_as_roles():
    from llmorch import __main__ as cli

    pins = cli._pins(
        _args(role_models={"backend": "groq/gpt-oss-120b"}), MANIFEST, []
    )
    assert pins == {Role.BACKEND: "groq/gpt-oss-120b"}


def test_start_carries_the_saved_pins_into_the_assignment(tmp_path, monkeypatch, capsys):
    """End to end: what the page saved is what writes the files."""
    from llmorch import __main__ as cli
    from llmorch.demo.website import ARTIFACTS
    from llmorch.providers.base import ProviderRegistry
    from llmorch.providers.mock import MockProvider
    from llmorch.chat import Conversation

    settings_module.Settings(
        live=False,
        role_models={"frontend": "openrouter/minimax-m3"},
    ).save()

    def registry(manifest):
        provider = MockProvider(responses=dict(ARTIFACTS))
        built = ProviderRegistry()
        for model in manifest.enabled_models:
            built.register(model.id, provider)
        return built, provider

    monkeypatch.setattr(cli, "_mock_registry", registry)
    monkeypatch.setattr("builtins.input", lambda prompt="": "/quit")

    assert cli.main(["start", "build a notes app"]) == 0

    out = capsys.readouterr().out
    session_id = out.split("New session ", 1)[1].split(".", 1)[0]
    conversation = Conversation.load(session_id)

    frontend = [n for n in conversation.files.values() if n.role == "frontend"]
    assert frontend
    assert all(note.model_id == "openrouter/minimax-m3" for note in frontend)
