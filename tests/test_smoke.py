"""Running the generated project.

Every other test in this suite reads artifacts. These execute them, which is the
whole point: the two faults that reached a real output folder — a path handler
that resolved every page to the drive root, and two models that disagreed about
a function's arity — were both invisible to static checks and both immediately
obvious to a process that started and answered a request.

The fixtures are therefore real servers on real sockets. Each one is healthy
except for the single thing it is built to break, because a checker that has
only been shown working code has no evidence behind it.
"""

from __future__ import annotations

import json
import socket

import pytest

from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine import smoke
from llmorch.engine.smoke import (
    Probe,
    SmokeReport,
    candidate_ports,
    fill_placeholders,
    find_entrypoint,
    port_is_free,
    sample_body,
    smoke_run,
    usable_ports,
)
from llmorch.report.render import render_smoke
from llmorch.types import InterfaceContract

PATHS = {n.id: n.output_path for n in build_nodes()}


def free_port() -> int:
    """An unused port, released immediately so the fixture can bind it.

    The window between release and bind is a real race, but the alternative —
    a fixed port — collides with whatever else is on the machine, which is the
    failure this whole module is careful about.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --------------------------------------------------------------------------
# Fixture servers
# --------------------------------------------------------------------------

_FIXTURE = '''\
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = {port}
TABLE = json.loads(r"""{table}""")
RAISE_ON = json.loads(r"""{raise_on}""")
BOOT = "{boot}"

if BOOT == "crash":
    raise SystemExit("schema.sql is missing")
if BOOT == "hang":
    import time
    time.sleep(30)
    raise SystemExit(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        path = self.path.split("?")[0]
        if path in RAISE_ON:
            raise RuntimeError("fixture server failed on " + path)
        status, body = TABLE.get(path, [404, "not found"])
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _handle
    do_POST = _handle


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


def write_fixture_server(
    directory,
    *,
    port: int,
    table: dict | None = None,
    raise_on: tuple[str, ...] = (),
    boot: str = "serve",
) -> None:
    """A server that answers exactly what the test says and nothing else."""
    (directory / "server.py").write_text(
        _FIXTURE.format(
            port=port,
            table=json.dumps(table or {}),
            raise_on=json.dumps(list(raise_on)),
            boot=boot,
        ),
        encoding="utf-8",
    )


def write_demo_project(directory, port: int) -> None:
    """The canned notes app, on a port this test owns."""
    for node_id, text in ARTIFACTS.items():
        target = directory / PATHS[node_id]
        target.parent.mkdir(parents=True, exist_ok=True)
        if PATHS[node_id] == "server.py":
            text = text.replace("PORT = 8000", f"PORT = {port}")
        target.write_text(text, encoding="utf-8")


# ==========================================================================
# Launch preliminaries
# ==========================================================================


def test_server_py_is_preferred_over_the_other_entrypoints(tmp_path):
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "server.py").write_text("", encoding="utf-8")
    assert find_entrypoint(tmp_path).name == "server.py"


def test_a_folder_with_nothing_runnable_has_no_entrypoint(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")
    assert find_entrypoint(tmp_path) is None


def test_the_port_literal_is_read_out_of_the_source():
    assert candidate_ports("PORT = 8931\n")[0] == 8931
    assert candidate_ports('serve(("127.0.0.1", 9100))')[0] == 9100
    assert candidate_ports("app.run(port=5000)")[0] == 5000


def test_the_conventional_port_is_always_a_fallback():
    """A server that computes its port still gets probed somewhere."""
    assert candidate_ports("print('hello')") == (8000,)
    assert 8000 in candidate_ports("PORT = 8931\n")


def test_a_nonsense_port_number_is_ignored():
    assert candidate_ports("VERSION = 999999") == (8000,)


# ==========================================================================
# Skips — the absence of evidence, reported as such
# ==========================================================================


def test_nothing_to_run_is_a_skip_rather_than_a_failure(tmp_path):
    report = smoke_run(tmp_path, InterfaceContract())
    assert not report.ran
    assert not report.errors
    assert "no entrypoint" in report.skipped


def test_a_degraded_stub_is_never_launched(tmp_path):
    """`materialize` writes a commented placeholder when a node produced nothing.

    Launching it would spend fifteen seconds proving that a file of comments
    does not serve HTTP.
    """
    (tmp_path / "server.py").write_text(
        "# DEGRADED — not generated\n# node: server (backend)\n", encoding="utf-8"
    )
    report = smoke_run(tmp_path, InterfaceContract())
    assert not report.ran
    assert "degraded stub" in report.skipped


def test_a_port_somebody_is_already_serving_on_is_not_a_candidate():
    """Detected by connecting, not by binding.

    The two disagree: a port left in TIME_WAIT by a previous run refuses a bind
    while nothing is listening on it, and calling that "in use" would skip the
    smoke run on a port that was in fact free.
    """
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        spare = free_port()

        assert usable_ports((port, spare)) == (spare,)


def test_a_port_this_run_did_not_open_is_never_probed(tmp_path, monkeypatch):
    """The rule that keeps a stranger's 200s from being reported as success."""
    monkeypatch.setattr(smoke, "port_is_free", lambda port, host="127.0.0.1": False)
    launched = []
    monkeypatch.setattr(
        smoke, "_launch", lambda *a, **kw: launched.append(a) or pytest.fail("launched")
    )

    write_fixture_server(tmp_path, port=free_port())
    report = smoke_run(tmp_path, InterfaceContract(), boot_timeout=2.0)

    assert not report.ran
    assert "already in use" in report.skipped
    assert not report.probes
    assert not launched


# ==========================================================================
# The healthy case
# ==========================================================================


def test_the_demo_project_actually_runs(tmp_path):
    """The reference artifacts, executed rather than inspected.

    This is the milestone claim the orchestrator has never been able to make:
    not that the files parse and agree, but that the assembled project serves
    its pages, its stylesheet, its script, and every route in the contract.
    """
    port = free_port()
    write_demo_project(tmp_path, port)

    report = smoke_run(tmp_path, INTERFACE)

    assert report.ran, report.skipped
    assert report.port == port
    assert not report.errors, [i.what for i in report.errors]
    assert report.ok

    by_label = {p.label: p.status for p in report.probes}
    assert by_label["GET /index.html"] == 200
    assert by_label["GET /note.html"] == 200
    assert by_label["GET /style.css"] == 200
    assert by_label["GET /app.js"] == 200
    assert by_label["GET /api/notes"] == 200
    assert by_label["POST /api/notes"] == 201

    # The POST ran first so the parameterized read has a record to find, and the
    # id came back from the server rather than being guessed.
    detail = [p for p in report.probes if p.path.startswith("/api/notes/")]
    assert detail and detail[0].status == 200


# ==========================================================================
# The failures worth catching
# ==========================================================================


def test_a_project_that_will_not_start_is_reported_with_its_stderr(tmp_path):
    write_fixture_server(tmp_path, port=free_port(), boot="crash")

    report = smoke_run(tmp_path, InterfaceContract(), boot_timeout=10.0)

    assert not report.ran
    assert [i.what for i in report.errors] == [
        "server.py exited with code 1 instead of serving"
    ]
    assert "schema.sql is missing" in report.stderr_tail


def test_a_startup_crash_is_one_finding_not_two(tmp_path):
    """The exit and the traceback that caused it are the same fault."""
    (tmp_path / "server.py").write_text(
        "raise RuntimeError('no database')\n", encoding="utf-8"
    )

    report = smoke_run(tmp_path, InterfaceContract(), boot_timeout=10.0)

    assert len(report.errors) == 1
    assert "Traceback (most recent call last)" in report.stderr_tail
    assert "never reached a serving state" in render_smoke(report)


def test_a_project_that_binds_no_port_is_timed_out_and_killed(tmp_path):
    write_fixture_server(tmp_path, port=free_port(), boot="hang")

    report = smoke_run(tmp_path, InterfaceContract(), boot_timeout=1.5)

    assert not report.ran
    assert any("bound no port" in i.what for i in report.errors)


def test_a_declared_page_the_server_will_not_serve_is_an_error(tmp_path):
    port = free_port()
    write_fixture_server(
        tmp_path, port=port, table={"/index.html": [200, "<html></html>"]}
    )

    report = smoke_run(
        tmp_path, InterfaceContract(pages=("index.html", "note.html"))
    )

    assert report.ran, report.skipped
    assert [i.what for i in report.errors] == ["`/note.html` returned 404"]


def test_an_asset_the_page_links_but_the_server_loses_is_an_error(tmp_path):
    """The drive-root fault, in the only form that reveals it.

    `contracts.check_assets_resolve` proves `style.css` was written. It was.
    The server serves it from somewhere else, and only a request finds that out.
    """
    port = free_port()
    write_fixture_server(
        tmp_path,
        port=port,
        table={"/index.html": [200, '<html><link href="style.css"></html>']},
    )

    report = smoke_run(tmp_path, InterfaceContract(pages=("index.html",)))

    assert report.ran, report.skipped
    assert any("links `/style.css`" in i.what for i in report.errors)


def test_a_page_already_fetched_is_not_fetched_again_as_a_link(tmp_path):
    """Pages link to each other; the report should say so once."""
    port = free_port()
    write_fixture_server(
        tmp_path,
        port=port,
        table={
            "/index.html": [200, '<html><a href="note.html">n</a></html>'],
            "/note.html": [200, '<html><a href="index.html">back</a></html>'],
        },
    )

    report = smoke_run(
        tmp_path, InterfaceContract(pages=("index.html", "note.html"))
    )

    assert report.ran, report.skipped
    assert [p.label for p in report.probes] == [
        "GET /index.html",
        "GET /note.html",
    ]


def test_a_route_that_raises_is_an_error_and_the_traceback_is_kept(tmp_path):
    port = free_port()
    write_fixture_server(tmp_path, port=port, raise_on=("/api/notes",))

    report = smoke_run(
        tmp_path,
        InterfaceContract(routes=({"method": "GET", "path": "/api/notes"},)),
    )

    assert report.ran, report.skipped
    assert report.errors
    assert "Traceback (most recent call last)" in report.stderr_tail
    assert any("raised while serving" in i.what for i in report.errors)


def test_a_five_hundred_on_a_declared_route_is_an_error(tmp_path):
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/api/notes": [500, "boom"]})

    report = smoke_run(
        tmp_path,
        InterfaceContract(routes=({"method": "GET", "path": "/api/notes"},)),
    )

    assert report.ran, report.skipped
    assert [i.what for i in report.errors] == ["`GET /api/notes` returned 500"]


def test_a_refused_synthesized_body_is_only_a_warning(tmp_path):
    """The request body was invented here, so a 400 may be the server being right.

    Reporting it as a failure would be the checker accusing working code, which
    costs more than the fault it would occasionally catch.
    """
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/api/notes": [400, "bad input"]})

    report = smoke_run(
        tmp_path,
        InterfaceContract(
            routes=({"method": "POST", "path": "/api/notes", "accepts": "NoteInput"},),
            data_models=({"name": "NoteInput", "fields": {"title": "string"}},),
        ),
    )

    assert report.ran, report.skipped
    assert not report.errors
    assert any("synthesized body" in i.what for i in report.warnings)


def test_an_empty_store_does_not_convict_a_parameterized_route(tmp_path):
    """Nothing was created, so a 404 for `{id}` proves nothing either way."""
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={})

    report = smoke_run(
        tmp_path,
        InterfaceContract(routes=({"method": "GET", "path": "/api/notes/{id}"},)),
    )

    assert report.ran, report.skipped
    assert not report.errors
    assert any("no record to fetch" in i.what for i in report.warnings)


def test_a_destructive_method_is_never_driven(tmp_path):
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/api/notes/1": [200, "{}"]})

    report = smoke_run(
        tmp_path,
        InterfaceContract(routes=({"method": "DELETE", "path": "/api/notes/{id}"},)),
    )

    assert report.ran, report.skipped
    assert not report.probes


# ==========================================================================
# Request synthesis
# ==========================================================================


def test_a_body_is_built_from_the_contract_by_type():
    interface = InterfaceContract(
        data_models=(
            {
                "name": "NoteInput",
                "fields": {
                    "title": "string",
                    "count": "integer",
                    "pinned": "boolean",
                    "tags": "string[]",
                },
            },
        )
    )
    body = sample_body(interface, "NoteInput")
    assert isinstance(body["title"], str)
    assert body["count"] == 1
    assert body["pinned"] is True
    assert body["tags"] == []


def test_an_unknown_model_yields_an_empty_body():
    assert sample_body(InterfaceContract(), "Missing") == {}


def test_placeholders_are_filled_in_every_convention():
    assert fill_placeholders("/api/notes/{id}", "7") == "/api/notes/7"
    assert fill_placeholders("/api/notes/:id", "7") == "/api/notes/7"
    assert fill_placeholders("/api/notes/${id}", "7") == "/api/notes/7"
    assert fill_placeholders("/api/notes", "7") == "/api/notes"


def test_a_free_port_reads_as_free():
    assert port_is_free(free_port())


# ==========================================================================
# Rendering
# ==========================================================================


def test_a_skipped_run_never_renders_as_a_pass():
    text = render_smoke(SmokeReport(skipped="no entrypoint in output/"))
    assert "skipped" in text
    assert "runs" not in text


def test_a_clean_run_says_what_it_proved():
    report = SmokeReport(
        ran=True,
        entrypoint="server.py",
        port=8000,
        probes=[Probe("GET", "/index.html", 200)],
    )
    text = render_smoke(report)
    assert "200  GET /index.html" in text
    assert "the assembled project runs" in text


def test_failures_are_rendered_with_the_stderr_that_explains_them():
    report = SmokeReport(ran=True, entrypoint="server.py", port=8000)
    report.add("error", "`GET /api/notes` returned 500", why="the server raised")
    report.stderr_tail = "Traceback (most recent call last)\nKeyError: 'title'"
    text = render_smoke(report)
    assert "[FAIL]" in text
    assert "KeyError" in text
