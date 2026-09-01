"""Cross-artifact validation.

Every other check in this system reads one file. These read the *set*, looking
for the failure a split build makes likely and a single author would never make:
each file impeccable against its own spec, and the project broken because two
models agreed with the spec and not with each other.

The fixtures below start from a consistent artifact set and break exactly one
thing, because a checker that has only ever been shown healthy input is a
checker with no evidence behind it.
"""

from __future__ import annotations

import pytest

from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine.contracts import (
    ContractReport,
    artifacts_from_results,
    check_contract,
    check_python_calls,
    route_pattern,
)
from llmorch.types import InterfaceContract, NodeResult, NodeState

PATHS = {n.id: n.output_path for n in build_nodes()}


@pytest.fixture
def artifacts():
    """The reference artifact set: consistent by construction."""
    return {PATHS[node_id]: text for node_id, text in ARTIFACTS.items()}


def _errors(report: ContractReport) -> list[str]:
    return [i.what for i in report.errors]


# ==========================================================================
# The healthy case
# ==========================================================================


def test_the_reference_artifacts_agree_with_each_other(artifacts):
    """If this ever fails, the demo itself has drifted from its contract."""
    report = check_contract(INTERFACE, artifacts)
    assert report.ok, _errors(report)
    assert not report.warnings, [i.what for i in report.warnings]
    assert len(report.checks_run) == 6


def test_an_empty_artifact_set_is_not_silently_fine():
    """Nothing produced means every promised page is missing."""
    report = check_contract(INTERFACE, {})
    assert not report.ok
    assert len(report.errors) == len(INTERFACE.pages)


# ==========================================================================
# Route agreement — the mismatch neither file can see alone
# ==========================================================================


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/api/notes", ("api", "notes")),
        ("/api/notes/{id}", ("api", "notes", "*")),
        ("/api/notes/:id", ("api", "notes", "*")),
        ("/api/notes/42?x=1", ("api", "notes", "42")),
        ("/api/notes/${id}", ("api", "notes", "*")),
    ],
)
def test_route_patterns_normalise_placeholders(path, expected):
    """A frontend interpolating a real id must still match the declared route,
    or every correct call would be reported as a violation."""
    assert route_pattern(path) == expected


def test_a_singular_plural_mismatch_is_caught(artifacts):
    """`/api/note/1` against `/api/notes/1`: both files are internally
    consistent, and the project does not work."""
    artifacts["app.js"] = artifacts["app.js"].replace("/api/notes", "/api/note")
    report = check_contract(INTERFACE, artifacts)

    assert not report.ok
    assert any("/api/note" in e for e in _errors(report))


def test_a_route_the_backend_never_mentions_is_caught(artifacts):
    artifacts["server.py"] = artifacts["server.py"].replace("/api/notes", "/api/things")
    report = check_contract(INTERFACE, artifacts)
    assert any("no backend file mentions" in e for e in _errors(report))


def test_a_parameterised_route_hiding_behind_its_collection_is_flagged(artifacts):
    """`/api/notes/{id}` shares its base with `/api/notes`, so the base alone
    proves nothing. Dropping only the parameterised handler must still surface —
    as a warning, since a segment-splitting dispatch can be correct without the
    literal prefix."""
    artifacts["server.py"] = artifacts["server.py"].replace("/api/notes/", "/api/x/")
    report = check_contract(INTERFACE, artifacts)

    assert any("may not be handled" in i.what for i in report.warnings)


def test_navigation_links_are_not_mistaken_for_api_calls(artifacts):
    """A page linking to another page is not a contract violation."""
    artifacts["index.html"] = artifacts["index.html"].replace(
        "</body>", '<a href="/about.html">about</a></body>'
    )
    report = check_contract(INTERFACE, artifacts)
    assert not any("/about.html" in e and "does not declare" in e for e in _errors(report))


# ==========================================================================
# Assets
# ==========================================================================


def test_a_page_linking_a_file_nobody_wrote_is_caught(artifacts):
    """The classic split-build failure: the page author links `style.css`
    because the contract implies one, and the styling node degraded."""
    del artifacts["style.css"]
    report = check_contract(INTERFACE, artifacts)

    assert not report.ok
    assert all("style.css" in e for e in _errors(report))


def test_external_and_inline_references_are_left_alone(artifacts):
    """A CDN link, an anchor, and a data URI are not this project's files."""
    artifacts["index.html"] = artifacts["index.html"].replace(
        "</head>",
        '<link href="https://cdn.example/x.css" rel="stylesheet">'
        '<a href="#top">top</a>'
        '<img src="data:image/png;base64,iVBOR">'
        "</head>",
    )
    report = check_contract(INTERFACE, artifacts)
    assert report.ok, _errors(report)


def test_a_missing_page_is_caught(artifacts):
    del artifacts["note.html"]
    report = check_contract(INTERFACE, artifacts)
    assert any("note.html" in e for e in _errors(report))


# ==========================================================================
# Schema
# ==========================================================================


def test_a_field_with_nowhere_to_live_is_a_warning_not_an_error(artifacts):
    """A field can legitimately be computed rather than stored, so this points
    rather than fails."""
    artifacts["schema.sql"] = artifacts["schema.sql"].replace("created_at", "made_at")
    report = check_contract(INTERFACE, artifacts)

    assert report.ok, "a naming difference must not fail the contract"
    assert any("created_at" in i.what for i in report.warnings)


def test_no_schema_artifact_means_no_schema_claims(artifacts):
    """Absence of a schema is a different finding from a schema that disagrees;
    this check has nothing to say about it."""
    del artifacts["schema.sql"]
    report = check_contract(INTERFACE, artifacts)
    assert not any("column in the schema" in i.what for i in report.warnings)


# ==========================================================================
# Wiring
# ==========================================================================


def test_only_completed_nodes_contribute_artifacts():
    """A degraded node's stub must not satisfy a reference that nothing really
    answers — that would turn a missing file into a silent pass."""
    nodes = {n.id: n for n in build_nodes()}
    results = {
        "index": NodeResult(node_id="index", state=NodeState.DONE, artifact="<html>"),
        "style": NodeResult(
            node_id="style", state=NodeState.DEGRADED, artifact="/* TODO */"
        ),
    }
    built = artifacts_from_results(nodes, results)

    assert "index.html" in built
    assert "style.css" not in built


def test_a_contract_with_no_routes_makes_no_route_claims(artifacts):
    empty = InterfaceContract(pages=INTERFACE.pages)
    report = check_contract(empty, artifacts)
    assert not any("declare" in e for e in _errors(report))


# ==========================================================================
# Python call agreement
#
# The failure that made the case for this file, taken from a real run: one
# model wrote `parse_csv(data, delimiter, strip_whitespace)`, another called
# `parse_csv(text=..., has_header=...)`. Both files parse. Both match their own
# spec. The tool raises TypeError on the first line of real work.
# ==========================================================================


def _check(files: dict[str, str]) -> ContractReport:
    report = ContractReport()
    check_python_calls(files, report)
    return report


def test_two_models_disagreeing_on_a_signature_is_caught():
    report = _check({
        "pkg/parser.py": "def parse_csv(data, delimiter=None):\n    return data\n",
        "pkg/cli.py": "from pkg.parser import parse_csv\n"
                      "def main():\n    return parse_csv(text='x', has_header=True)\n",
    })
    assert len(report.errors) == 2
    assert "text=" in report.errors[0].what


def test_a_matching_call_is_left_alone():
    report = _check({
        "pkg/parser.py": "def parse_csv(data, delimiter=None):\n    return data\n",
        "pkg/cli.py": "def main():\n    return parse_csv('x', delimiter=',')\n",
    })
    assert report.ok, [i.what for i in report.errors]


def test_too_many_positional_arguments_is_caught():
    report = _check({
        "pkg/a.py": "def render(rows):\n    return rows\n",
        "pkg/b.py": "def main():\n    return render(1, 2, 3)\n",
    })
    assert any("positional" in i.what for i in report.errors)


def test_a_function_taking_kwargs_is_never_accused():
    """It accepts anything, so no keyword can be wrong."""
    report = _check({
        "pkg/a.py": "def render(rows, **options):\n    return rows\n",
        "pkg/b.py": "def main():\n    return render([], anything_at_all=1)\n",
    })
    assert report.ok


def test_an_ambiguous_name_is_left_alone():
    """Two definitions of one name mean any call could be either, and a false
    accusation about working code is worse than a missed fault."""
    report = _check({
        "pkg/a.py": "def render(rows):\n    return rows\n",
        "pkg/b.py": "def render(text, style):\n    return text\n",
        "pkg/c.py": "def main():\n    return render(text='x', style='y')\n",
    })
    assert report.ok


def test_calls_to_code_this_project_did_not_write_are_ignored():
    """The stdlib and third-party libraries are not ours to check."""
    report = _check({
        "pkg/a.py": "import json\n"
                    "def main():\n    return json.dumps({}, indent=2, sort_keys=True)\n",
    })
    assert report.ok


def test_a_method_on_an_unrelated_object_is_not_a_module_call():
    """`writer.render(...)` is only checked when `writer` is one of our own
    modules — otherwise every same-named method would be a false positive."""
    report = _check({
        "pkg/a.py": "def render(rows):\n    return rows\n",
        "pkg/b.py": "def main(engine):\n    return engine.render(template='x')\n",
    })
    assert report.ok


def test_a_qualified_call_to_our_own_module_is_checked():
    report = _check({
        "parser.py": "def parse_csv(data):\n    return data\n",
        "cli.py": "import parser\n"
                  "def main():\n    return parser.parse_csv(text='x')\n",
    })
    assert any("parse_csv" in i.what for i in report.errors)


def test_an_unparseable_file_does_not_stop_the_check():
    """A degraded stub must not blind the checker to the files around it."""
    report = _check({
        "pkg/broken.py": "def oops(:\n",
        "pkg/a.py": "def render(rows):\n    return rows\n",
        "pkg/b.py": "def main():\n    return render(wrong=1)\n",
    })
    assert any("render" in i.what for i in report.errors)


def test_the_reference_artifacts_have_no_call_mismatches(artifacts):
    assert _check(artifacts).ok
