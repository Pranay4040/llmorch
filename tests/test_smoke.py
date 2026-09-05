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
from dataclasses import replace

import pytest

from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine import smoke
from llmorch.engine.checkpoint import plan_from_dict, plan_to_dict
from llmorch.engine.smoke import (
    INSTALL_RECIPES,
    Probe,
    SmokeReport,
    candidate_ports,
    fill_placeholders,
    find_entrypoint,
    port_is_free,
    sample_body,
    install_plan,
    plan_launch,
    smoke_run,
    usable_ports,
)
from llmorch.report.render import render_smoke
from llmorch.types import InterfaceContract, LaunchSpec

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


def demo_interface(port: int) -> InterfaceContract:
    """The demo contract, declaring the port this test's copy actually binds."""
    return replace(INTERFACE, launch=replace(INTERFACE.launch, port=port))


def test_the_demo_project_actually_runs(tmp_path):
    """The reference artifacts, executed rather than inspected.

    This is the milestone claim the orchestrator has never been able to make:
    not that the files parse and agree, but that the assembled project serves
    its pages, its stylesheet, its script, and every route in the contract.
    """
    port = free_port()
    write_demo_project(tmp_path, port)

    report = smoke_run(tmp_path, demo_interface(port))

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


def test_the_demo_project_runs_without_a_declared_launch_too(tmp_path):
    """The inference path, kept working for a plan that predates `launch`.

    Checkpoints and cached plans written before the contract carried a launch
    command still resume, so this is not a legacy branch — it is the one every
    stored plan takes.
    """
    port = free_port()
    write_demo_project(tmp_path, port)

    report = smoke_run(tmp_path, replace(INTERFACE, launch=LaunchSpec()))

    assert report.ran, report.skipped
    assert report.port == port  # read out of the source, as before
    assert not report.errors, [i.what for i in report.errors]


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
# The declared launch command
# ==========================================================================


def declaring(command, **kwargs) -> InterfaceContract:
    return InterfaceContract(launch=LaunchSpec(command=command, **kwargs))


def test_a_declared_command_is_used_instead_of_the_search(tmp_path):
    """The point of the whole field: the contract says how to start itself."""
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/": [200, "up"]})
    (tmp_path / "boot.py").write_text(
        (tmp_path / "server.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "server.py").write_text(
        "raise SystemExit('wrong file')\n", encoding="utf-8"
    )

    launch, refusal = plan_launch(
        tmp_path, declaring(("python", "boot.py"), port=port), python="/usr/bin/python3"
    )

    assert refusal == ""
    assert launch.argv == ("/usr/bin/python3", "boot.py")
    assert launch.ports == (port,)
    assert launch.label == "boot.py"


def test_a_declared_port_is_taken_over_the_source_literal(tmp_path):
    write_fixture_server(tmp_path, port=9999)
    launch, _ = plan_launch(tmp_path, declaring(("python", "server.py"), port=4321))
    assert launch.ports == (4321,)
    assert launch.port_declared


def test_a_declared_launch_without_a_port_still_reads_the_source(tmp_path):
    write_fixture_server(tmp_path, port=9999)
    launch, _ = plan_launch(tmp_path, declaring(("python", "server.py")))
    assert launch.ports[0] == 9999
    assert not launch.port_declared


def test_a_subcommand_and_a_flag_pass_through(tmp_path):
    """`go run main.go`, `node --enable-source-maps server.js`."""
    (tmp_path / "main.go").write_text("package main\n", encoding="utf-8")
    launch, refusal = plan_launch(tmp_path, declaring(("go", "run", "main.go")))
    assert refusal == ""
    assert launch.argv == ("go", "run", "main.go")

    (tmp_path / "server.js").write_text("// js\n", encoding="utf-8")
    launch, refusal = plan_launch(
        tmp_path, declaring(("node", "--enable-source-maps", "server.js"))
    )
    assert refusal == ""
    assert launch.argv == ("node", "--enable-source-maps", "server.js")


def test_a_file_in_a_subdirectory_is_addressed_from_the_output_root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("# app\n", encoding="utf-8")
    launch, refusal = plan_launch(
        tmp_path, declaring(("python", "src/app.py")), python="py"
    )
    assert refusal == ""
    assert launch.argv == ("py", "src/app.py")
    assert launch.cwd == tmp_path.resolve()


def test_a_program_outside_the_allowlist_is_refused(tmp_path):
    """Not because `bash` could run code the artifacts could not — it is the
    blast radius that narrows, from any binary to the project just written."""
    (tmp_path / "server.py").write_text("# ok\n", encoding="utf-8")
    launch, refusal = plan_launch(tmp_path, declaring(("bash", "-c", "server.py")))
    assert launch is None
    assert "not one of" in refusal


def test_a_path_escaping_the_output_folder_is_refused(tmp_path):
    launch, refusal = plan_launch(
        tmp_path, declaring(("python", "../../etc/passwd"))
    )
    assert launch is None
    assert "traverse upwards" in refusal


def test_an_absolute_path_is_refused(tmp_path):
    launch, refusal = plan_launch(tmp_path, declaring(("python", "/etc/hosts")))
    assert launch is None
    assert "relative" in refusal


def test_a_file_no_node_produced_is_refused(tmp_path):
    launch, refusal = plan_launch(tmp_path, declaring(("python", "server.py")))
    assert launch is None
    assert "which no node produced" in refusal


def test_a_module_flag_cannot_stand_in_for_the_project(tmp_path):
    """`python -m http.server` would serve the folder and prove nothing about
    the code the models wrote. It is refused as a module nothing produced."""
    launch, refusal = plan_launch(tmp_path, declaring(("python", "-m", "http.server")))
    assert launch is None
    assert "http.server" in refusal


def test_a_command_with_no_file_argument_at_all_is_refused(tmp_path):
    launch, refusal = plan_launch(tmp_path, declaring(("python", "-u")))
    assert launch is None
    assert "names no file" in refusal


def test_a_shell_shaped_argument_is_refused(tmp_path):
    """Nothing here reaches a shell — `shell=True` is never set — so this is
    about not passing through a token whose shape we cannot account for."""
    (tmp_path / "server.py").write_text("# ok\n", encoding="utf-8")

    # Caught as a path, by the same containment check materialization uses.
    _, refusal = plan_launch(
        tmp_path, declaring(("python", "server.py", "; rm -rf /"))
    )
    assert refusal

    # Caught as neither a path nor an argument shape we recognise.
    _, refusal = plan_launch(
        tmp_path, declaring(("python", "server.py", "$(whoami)"))
    )
    assert "will not pass" in refusal


def test_a_command_with_too_many_arguments_is_refused(tmp_path):
    launch, refusal = plan_launch(
        tmp_path, declaring(("python",) + tuple(f"a{i}" for i in range(20)))
    )
    assert launch is None
    assert "over the" in refusal


def test_a_declared_launch_naming_a_degraded_stub_is_refused(tmp_path):
    (tmp_path / "server.py").write_text(
        "# DEGRADED — not generated\n", encoding="utf-8"
    )
    launch, refusal = plan_launch(tmp_path, declaring(("python", "server.py")))
    assert launch is None
    assert "degraded stub" in refusal


def test_a_refused_launch_never_falls_back_to_guessing(tmp_path):
    """A contract that states how to start itself and states something we will
    not run gets a skip, not a different program started in its place."""
    write_fixture_server(tmp_path, port=free_port())
    report = smoke_run(tmp_path, declaring(("bash", "server.py")))

    assert not report.ran
    assert "not one of" in report.skipped
    assert not report.probes


def test_the_ready_path_is_probed_before_anything_else(tmp_path):
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/health": [200, "ok"]})

    report = smoke_run(
        tmp_path, declaring(("python", "server.py"), port=port, ready_path="/health")
    )

    assert report.ran, report.skipped
    assert report.probes[0].label == "GET /health"
    assert not report.errors


def test_a_ready_path_that_is_also_a_page_is_probed_once(tmp_path):
    """The page check is the stricter of the two, so it is the one that runs."""
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/index.html": [200, "<html>"]})

    report = smoke_run(
        tmp_path,
        replace(
            declaring(("python", "server.py"), port=port, ready_path="/index.html"),
            pages=("index.html",),
        ),
    )

    assert report.ran, report.skipped
    assert [p.label for p in report.probes] == ["GET /index.html"]


def test_a_ready_path_that_five_hundreds_is_an_error(tmp_path):
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/": [503, "starting"]})

    report = smoke_run(tmp_path, declaring(("python", "server.py"), port=port))

    assert report.ran, report.skipped
    assert any("`/` returned 503" in i.what for i in report.errors)


def test_a_ready_path_that_404s_is_fine(tmp_path):
    """An API with no root route is a normal shape, not a broken server."""
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={})

    report = smoke_run(tmp_path, declaring(("python", "server.py"), port=port))

    assert report.ran, report.skipped
    assert not report.errors


def test_a_declared_port_the_server_never_binds_says_which_disagreed(tmp_path):
    write_fixture_server(tmp_path, port=free_port())

    report = smoke_run(
        tmp_path,
        declaring(("python", "server.py"), port=free_port()),
        boot_timeout=2.0,
    )

    assert not report.ran
    assert any("the declaration and" in i.why for i in report.errors)


# ==========================================================================
# Dependencies
# ==========================================================================


def test_every_recipe_is_pinned_and_runs_no_package_scripts():
    """Both properties are the reason an install is allowed here at all.

    A package's install hooks are third-party code the plan never mentioned, and
    an unpinned install resolves to whatever the registry serves today.
    """
    for recipe in INSTALL_RECIPES:
        assert "--ignore-scripts" in recipe.argv
        assert any(
            flag in recipe.argv for flag in ("ci", "--frozen-lockfile")
        ), recipe.argv


def test_a_lockfile_without_its_install_is_a_plan(tmp_path):
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    recipe = install_plan(tmp_path)
    assert recipe is not None
    assert recipe.argv[0] == "npm"


def test_an_install_already_done_is_not_planned_again(tmp_path):
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    assert install_plan(tmp_path) is None


def test_a_folder_with_no_lockfile_needs_nothing(tmp_path):
    (tmp_path / "server.py").write_text("# ok\n", encoding="utf-8")
    assert install_plan(tmp_path) is None


def test_python_and_go_are_left_alone(tmp_path):
    """`go run` fetches its own modules, and installing model-chosen packages
    into the interpreter running llmorch would change the user's environment."""
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "go.sum").write_text("", encoding="utf-8")
    assert install_plan(tmp_path) is None


def test_needing_an_install_that_was_not_asked_for_is_a_skip(tmp_path):
    """Named precisely, with the command that would fix it — the project is not
    started into a folder where it can only fail."""
    port = free_port()
    write_fixture_server(tmp_path, port=port)
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    report = smoke_run(tmp_path, declaring(("python", "server.py"), port=port))

    assert not report.ran
    assert "npm ci --ignore-scripts" in report.skipped
    assert not report.errors


def test_a_failed_install_is_not_the_project_s_fault(tmp_path, monkeypatch):
    port = free_port()
    write_fixture_server(tmp_path, port=port)
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        smoke, "_run_install", lambda recipe, cwd: (False, "npm ERR! 404 not found")
    )

    report = smoke_run(
        tmp_path, declaring(("python", "server.py"), port=port), install=True
    )

    assert not report.ran
    assert [i.what for i in report.errors] == ["`npm ci --ignore-scripts` failed"]
    assert "nothing here is its fault" in report.errors[0].why
    assert "404" in report.stderr_tail


def test_a_successful_install_is_recorded_and_the_project_then_runs(
    tmp_path, monkeypatch
):
    port = free_port()
    write_fixture_server(tmp_path, port=port, table={"/": [200, "up"]})
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    ran: list = []

    def fake_install(recipe, cwd):
        ran.append((recipe.argv, cwd))
        (cwd / recipe.marker).mkdir()
        return True, ""

    monkeypatch.setattr(smoke, "_run_install", fake_install)

    report = smoke_run(
        tmp_path, declaring(("python", "server.py"), port=port), install=True
    )

    assert ran and ran[0][1] == tmp_path.resolve()
    assert report.ran, report.skipped
    assert report.installed == "npm ci --ignore-scripts"
    assert not report.errors


def test_a_death_from_a_missing_dependency_is_not_blamed_on_the_code(tmp_path):
    """The bug this exists to fix: without it the report reads as the model
    writing a bad import, when the harness started it in an empty folder."""
    (tmp_path / "server.py").write_text(
        "import a_package_nobody_installed\n", encoding="utf-8"
    )

    report = smoke_run(
        tmp_path,
        declaring(("python", "server.py"), port=free_port()),
        boot_timeout=10.0,
    )

    assert not report.ran
    assert any("want of a dependency" in i.what for i in report.warnings)
    assert any("no lockfile was found" in i.why for i in report.warnings)


def test_an_ordinary_crash_is_still_the_code_s_fault(tmp_path):
    """The hint must not fire on every failure, or it stops meaning anything."""
    (tmp_path / "server.py").write_text("raise ValueError('bad')\n", encoding="utf-8")

    report = smoke_run(
        tmp_path,
        declaring(("python", "server.py"), port=free_port()),
        boot_timeout=10.0,
    )

    assert not report.ran
    assert not report.warnings


def test_stderr_keeps_the_cause_as_well_as_the_stack():
    """Node names the missing module on line one and stacks thirty frames after
    it, so a tail alone shows the stack and loses the cause."""
    text = "Error: Cannot find module 'express'\n" + "\n".join(
        f"    at frame {i}" for i in range(40)
    )
    rendered = render_smoke(
        SmokeReport(entrypoint="server.js", stderr_tail=text, skipped="")
    )
    assert "Cannot find module" in rendered
    assert "more lines" in rendered
    assert "at frame 39" in rendered


# ==========================================================================
# Reading a launch out of untrusted JSON
# ==========================================================================


def test_a_command_line_string_is_accepted_where_a_list_was_asked_for():
    assert LaunchSpec.from_payload({"command": "python server.py"}).command == (
        "python",
        "server.py",
    )


def test_a_port_that_arrives_as_a_string_is_read():
    spec = LaunchSpec.from_payload({"command": ["node", "a.js"], "port": "3000"})
    assert spec.port == 3000


def test_junk_yields_a_contract_that_declares_nothing():
    for junk in ("rm -rf /", None, 42, {"command": {"not": "a list"}}, {"command": []}):
        assert not LaunchSpec.from_payload(junk).declared


def test_a_launch_survives_the_checkpoint_round_trip(tmp_path):
    """Resuming a run must start the project the same way the plan said to."""
    interface = declaring(("node", "server.js"), port=3000, ready_path="/health")
    nodes = {n.id: n for n in build_nodes()}

    restored = plan_from_dict(plan_to_dict(nodes, interface))[1]

    assert restored.launch == interface.launch


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
