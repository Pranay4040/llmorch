"""Salvage and materialization tests.

The `safe_join` cases are security-critical: `output_path` comes from LLM
output, and this is the only place untrusted model output reaches the
filesystem.
"""

from __future__ import annotations

import sys

import pytest

from llmorch.engine.materialize import materialize, safe_join
from llmorch.engine.salvage import (
    all_fenced_blocks,
    extract_code,
    extract_json,
    looks_truncated,
    strip_fences,
)
from llmorch.errors import UnsafePath
from llmorch.types import NodeResult, NodeState, OutputKind, Role, TaskNode

# ==========================================================================
# Path safety
# ==========================================================================


@pytest.mark.parametrize(
    "hostile",
    [
        "../../etc/passwd",
        "../outside.txt",
        "a/../../b.txt",
        "/etc/passwd",
        "C:\\Windows\\System32\\evil.dll",
        "C:/Windows/evil.dll",
        "\\\\server\\share\\evil.dll",
        "//server/share/evil.dll",
        "",
        "   ",
        ".",
        "con.txt",
        "nul",
        "COM1.js",
        # Windows silently strips a trailing space or dot from a path
        # component, so the path written would differ from the path checked.
        "dir /file.txt",
        "dir./file.txt",
        "trailing.",
        "with\x00null.txt",
    ],
)
def test_hostile_output_paths_are_rejected(tmp_path, hostile):
    with pytest.raises(UnsafePath):
        safe_join(tmp_path, hostile)


def test_surrounding_whitespace_is_stripped_rather_than_rejected(tmp_path):
    """Incidental whitespace around a whole path is a formatting artefact, not
    an attack — unlike whitespace inside a component."""
    assert safe_join(tmp_path, "  index.html  ") == safe_join(tmp_path, "index.html")


@pytest.mark.parametrize(
    "ok",
    ["index.html", "css/style.css", "./app.js", "a/b/c/deep.py", "schema.sql"],
)
def test_legitimate_relative_paths_are_accepted(tmp_path, ok):
    resolved = safe_join(tmp_path, ok)
    assert tmp_path.resolve() in resolved.parents


def test_backslash_separators_are_normalised(tmp_path):
    """Models emit Windows separators unpredictably; both must land identically."""
    assert safe_join(tmp_path, "css\\style.css") == safe_join(tmp_path, "css/style.css")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs admin")
def test_symlink_escape_is_rejected(tmp_path):
    """String inspection alone would pass this — the check must resolve first."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "output"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafePath):
        safe_join(root, "link/evil.txt")


# ==========================================================================
# Salvage
# ==========================================================================


def test_strip_fences_returns_inner_content():
    assert strip_fences("```python\nprint('hi')\n```") == "print('hi')"


def test_text_without_fences_passes_through():
    assert strip_fences("plain content") == "plain content"


def test_json_recovered_from_surrounding_prose():
    """The single most common malformed-response shape."""
    text = 'Sure! Here you go:\n\n{"verdict": "pass"}\n\nHope that helps!'
    assert extract_json(text) == {"verdict": "pass"}


def test_json_recovered_from_a_fenced_block():
    text = 'Result:\n```json\n{"a": [1, 2, 3]}\n```'
    assert extract_json(text) == {"a": [1, 2, 3]}


def test_json_extraction_ignores_braces_inside_strings():
    """A naive brace scan would stop at the wrong place here."""
    text = '{"note": "a } inside a string", "ok": true}'
    assert extract_json(text) == {"note": "a } inside a string", "ok": True}


def test_json_extraction_handles_escaped_quotes():
    text = r'{"quote": "she said \"hi\"", "n": 1}'
    assert extract_json(text) == {"quote": 'she said "hi"', "n": 1}


def test_json_extraction_returns_none_when_nothing_parses():
    assert extract_json("no json here at all") is None
    assert extract_json("") is None


def test_code_extraction_prefers_the_requested_language():
    """Models often precede the real file with a short usage example."""
    text = "Usage:\n```bash\npython server.py\n```\nCode:\n```python\nx = 1\n```"
    assert extract_code(text, prefer_lang="python") == "x = 1"


def test_code_extraction_falls_back_to_the_longest_block():
    text = "```\nshort\n```\n```\nmuch longer block of content here\n```"
    assert "much longer" in extract_code(text)


def test_all_fenced_blocks_reports_languages():
    blocks = all_fenced_blocks("```py\na\n```\n```sql\nb\n```")
    assert blocks == [("py", "a"), ("sql", "b")]


def test_unclosed_fence_reads_as_truncated():
    assert looks_truncated("```python\nprint('unfinis")
    assert not looks_truncated("```python\nprint('done')\n```")


def test_empty_output_reads_as_truncated():
    assert looks_truncated("")


# ==========================================================================
# Materialization
# ==========================================================================


def _node(node_id, path, kind=OutputKind.CODE, role=Role.FRONTEND):
    return TaskNode(
        id=node_id,
        title=f"build {path}",
        role=role,
        spec=f"produce {path}",
        output_path=path,
        output_kind=kind,
    )


def _done(node_id, artifact, model="groq/llama-3.3-70b"):
    return NodeResult(
        node_id=node_id, state=NodeState.DONE, artifact=artifact, model_id=model
    )


def test_artifacts_are_written_to_a_real_folder(tmp_path):
    nodes = {"n1": _node("n1", "index.html"), "n2": _node("n2", "app.js")}
    results = {
        "n1": _done("n1", "<h1>Notes</h1>"),
        "n2": _done("n2", "```javascript\nfetch('/api/notes')\n```"),
    }

    report = materialize(tmp_path, nodes, results)

    assert (tmp_path / "index.html").read_text() == "<h1>Notes</h1>\n"
    # Fences are stripped, so the file is immediately usable.
    assert (tmp_path / "app.js").read_text() == "fetch('/api/notes')\n"
    assert report.total == 2
    assert not report.rejected


def test_nested_directories_are_created(tmp_path):
    nodes = {"n1": _node("n1", "static/css/style.css")}
    materialize(tmp_path, nodes, {"n1": _done("n1", "body{}")})
    assert (tmp_path / "static" / "css" / "style.css").is_file()


def test_degraded_nodes_are_stubbed_not_skipped(tmp_path):
    """The folder structure stays complete and the gap explains itself."""
    nodes = {"n1": _node("n1", "server.py", OutputKind.CODE, Role.BACKEND)}
    results = {
        "n1": NodeResult(
            node_id="n1",
            state=NodeState.DEGRADED,
            error="all vendors exhausted",
            vendors_tried=("groq", "gemini"),
        )
    }

    report = materialize(tmp_path, nodes, results)
    content = (tmp_path / "server.py").read_text()

    assert report.stubbed == ("server.py",)
    assert "DEGRADED" in content
    assert "all vendors exhausted" in content
    assert "groq, gemini" in content
    assert "produce server.py" in content  # original spec preserved


def test_a_hostile_path_degrades_only_its_own_file(tmp_path):
    """One bad path from one model must not discard everyone else's work."""
    nodes = {
        "good": _node("good", "index.html"),
        "bad": _node("bad", "../../escape.txt"),
    }
    results = {"good": _done("good", "<h1>ok</h1>"), "bad": _done("bad", "pwned")}

    report = materialize(tmp_path, nodes, results)

    assert report.written == ("index.html",)
    assert len(report.rejected) == 1
    assert not (tmp_path.parent.parent / "escape.txt").exists()


def test_readme_documents_how_to_run_the_result(tmp_path):
    nodes = {"n1": _node("n1", "index.html")}
    materialize(tmp_path, nodes, {"n1": _done("n1", "<h1>hi</h1>")})

    readme = (tmp_path / "README.md").read_text()
    assert "python server.py" in readme
    assert "localhost:8000" in readme
    assert "index.html" in readme


def test_readme_attributes_each_file_to_its_model(tmp_path):
    """Seeing which vendor produced which file is the point of the exercise."""
    nodes = {"n1": _node("n1", "index.html"), "n2": _node("n2", "server.py")}
    results = {
        "n1": _done("n1", "<h1>hi</h1>", model="gemini/2.5-flash"),
        "n2": _done("n2", "print(1)", model="groq/llama-3.3-70b"),
    }
    materialize(tmp_path, nodes, results)

    readme = (tmp_path / "README.md").read_text()
    assert "gemini/2.5-flash" in readme
    assert "groq/llama-3.3-70b" in readme
