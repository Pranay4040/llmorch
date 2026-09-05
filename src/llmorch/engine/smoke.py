"""Execute the generated project and see whether it actually runs.

Every other check in this system reads the artifacts. Tier 0 asks whether a file
parses, Tier 1 asks whether it does its job, and `contracts.py` asks whether the
files agree with each other. All three are static, and all three passed the two
faults that reached a real output folder: a path handler that resolved every
page to the drive root on Windows, and two models that agreed on a function's
name and disagreed on its arity. Both were found by running the result.

So this starts the project, drives the contract's own pages and routes against
it over HTTP, and reports what came back.

**This executes model-written code.** Materialization is the only place where
untrusted output reaches the filesystem; this is the only place where it reaches
the interpreter. That is a decision for the caller, not for this module: it is
opt-in (`llmorch run --smoke`) and never a default.

Two rules keep it from making false accusations, which against generated code
cost more than a missed fault:

- **Never probe a port this run did not open.** A port already answering before
  launch belongs to somebody else's server, and its 200s would be reported as
  this project working.
- **Only a server-side failure is an error.** A 5xx, a declared route the server
  does not serve, a page that will not load, a linked asset that 404s, a
  traceback on stderr, a process that dies. A 4xx on synthesized input is a
  warning — the request body was invented here, so the server may be right to
  reject it.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..errors import UnsafePath
from ..types import InterfaceContract, LaunchSpec
from .contracts import local_asset_refs
from .materialize import safe_join

# Tried in order, when the contract declares no launch of its own. `server.py`
# first because it is what the pinned stack and the generated README both name.
ENTRYPOINTS = ("server.py", "app.py", "main.py")

# What a declared `command` is allowed to start.
#
# The allowlist is not what stops a hostile plan from running code — `--smoke`
# already runs a model-written file, so that door is open by the time we get
# here, and the user opened it. What it stops is the blast radius widening from
# "the project we just materialized" to "any binary on this machine, with any
# arguments". A contract that wants `bash -c` does not get it; it gets a skip
# and a reason.
INTERPRETERS = frozenset(
    {"python", "python3", "node", "deno", "bun", "ruby", "php", "go"}
)

# Interpreters we substitute the running interpreter for, so a project is
# started with the environment the tests and the CLI are already using rather
# than whatever `python` means on this PATH.
_PYTHON = frozenset({"python", "python3"})

# An argument that is neither a path nor a flag: a subcommand like `go run`, or
# a value like `--port=8080`. Bounded and character-restricted so nothing that
# reaches a shell-like context can hide in one — though nothing here ever
# reaches a shell, since `shell=True` is never used.
_PLAIN_ARG = re.compile(r"^[A-Za-z0-9._:=@-]{1,64}$")

# What makes an argument look like it names a file rather than a switch.
_PATHISH = re.compile(r"[/\\]|\.[A-Za-z0-9]{1,6}$")

MAX_LAUNCH_ARGS = 12

# Signatures of a process that died because a dependency was not there. The
# point of recognising them is attribution: without it, a project that dies on
# its first `require` reads as the model writing bad code, when what actually
# happened is that this harness started it without installing anything.
_MISSING_DEPENDENCY = (
    "cannot find module",
    "modulenotfounderror",
    "no module named",
    "err_module_not_found",
    "no required module provides package",
)

INSTALL_TIMEOUT = 300.0

DEFAULT_PORT = 8000

# Written by `materialize._stub_for` when no model could produce the node. There
# is nothing to learn from launching it.
_DEGRADED = "DEGRADED — not generated"

# Ordered by how strongly each says "this is the listening port".
_PORT_PATTERNS = (
    re.compile(r"""(?im)^\s*(?:PORT|_PORT|SERVER_PORT)\s*=\s*(\d{2,5})\b"""),
    re.compile(r"""(?i)\(\s*["'][\w.]*["']\s*,\s*(\d{2,5})\s*\)"""),
    re.compile(r"""(?i)\bport\s*=\s*(\d{2,5})\b"""),
    re.compile(r"""(?i)(?:localhost|127\.0\.0\.1)\s*:\s*(\d{2,5})"""),
)

_TRACEBACK = "Traceback (most recent call last)"

# A path segment the contract left as a placeholder: {id}, :id, ${id}, <id>.
_PLACEHOLDER = re.compile(r"^(\{.*\}|:.+|\$\{.*\}|<.+>)$")


@dataclass(frozen=True, slots=True)
class SmokeIssue:
    severity: str  # "error" | "warning"
    what: str
    where: str = ""
    why: str = ""


@dataclass(frozen=True, slots=True)
class Probe:
    """One request made against the running server."""

    method: str
    path: str
    status: int | None = None
    """None means the request never got an answer — refused, reset, timed out."""
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.method} {self.path}"


@dataclass(slots=True)
class SmokeReport:
    ran: bool = False
    skipped: str = ""
    """Why nothing was launched. Empty when `ran` is True."""
    entrypoint: str = ""
    port: int | None = None
    installed: str = ""
    """The install command that ran first, when one did."""
    probes: list[Probe] = field(default_factory=list)
    issues: list[SmokeIssue] = field(default_factory=list)
    stderr_tail: str = ""

    def add(self, severity: str, what: str, where: str = "", why: str = "") -> None:
        self.issues.append(SmokeIssue(severity, what, where, why))

    @property
    def errors(self) -> list[SmokeIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[SmokeIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True only when the project was actually run and nothing failed.

        A skipped run is not a pass — it is the absence of evidence, and the
        renderer says so rather than showing a tick.
        """
        return self.ran and not self.errors


# --------------------------------------------------------------------------
# Launch preliminaries
# --------------------------------------------------------------------------


def find_entrypoint(output_dir: Path) -> Path | None:
    for name in ENTRYPOINTS:
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    return None


def candidate_ports(source: str) -> tuple[int, ...]:
    """Ports the entrypoint looks like it will bind, best guess first.

    Read out of the source because the generated code owns the number: nothing
    here can inject one, and a stdlib `http.server` build hardcodes it. The
    fallback exists so a server that computes its port still gets probed on the
    conventional one rather than not at all.
    """
    found: list[int] = []
    for pattern in _PORT_PATTERNS:
        for raw in pattern.findall(source):
            port = int(raw)
            if 1 <= port <= 65535 and port not in found:
                found.append(port)
    if DEFAULT_PORT not in found:
        found.append(DEFAULT_PORT)
    return tuple(found[:4])


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether nothing is currently *serving* on this port.

    Deliberately a connect rather than a bind. They disagree: a port whose last
    connection is still in TIME_WAIT refuses a bind while nothing is listening
    on it, so a bind probe reports a perfectly usable port as taken. The
    question here is only ever "would a request reach somebody else's server",
    and connect is what answers that one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def usable_ports(ports: tuple[int, ...]) -> tuple[int, ...]:
    """The candidates nothing is already answering on."""
    return tuple(p for p in ports if port_is_free(p))


def _looks_degraded(source: str) -> bool:
    return _DEGRADED in source[:400]


@dataclass(frozen=True, slots=True)
class Install:
    """A lockfile-pinned dependency install for the folder that was written."""

    argv: tuple[str, ...]
    lockfile: str
    marker: str
    """The directory whose presence means the install already happened."""

    @property
    def label(self) -> str:
        return " ".join(self.argv)


# Keyed on the lockfile, not on anything a model said. That is the whole design:
# `LaunchSpec.command` is declared and therefore validated, while this is
# inferred from a file the build either produced or did not, so there is no new
# untrusted input to check.
#
# Every recipe is pinned to the lockfile and passes `--ignore-scripts`, because
# a package's install hooks are third-party code the plan never mentioned and
# nobody reviewed. Each was run against a real install before being written
# down; the flags differ per manager and guessing them produces a recipe that
# fails in a way that looks like the project's fault.
INSTALL_RECIPES: tuple[Install, ...] = (
    Install(("npm", "ci", "--ignore-scripts"), "package-lock.json", "node_modules"),
    Install(
        ("pnpm", "install", "--frozen-lockfile", "--ignore-scripts"),
        "pnpm-lock.yaml",
        "node_modules",
    ),
    Install(
        ("yarn", "install", "--frozen-lockfile", "--ignore-scripts"),
        "yarn.lock",
        "node_modules",
    ),
)


def install_plan(output_dir: Path) -> Install | None:
    """The install this folder needs and has not had, or None.

    Nothing is inferred for Python or Go on purpose. `go run` fetches its own
    modules, so a Go build needs network rather than a step; and installing
    model-chosen packages into the interpreter running llmorch would put them in
    the user's environment, which is not this module's to change. Both are
    instead recognised on failure, by `_MISSING_DEPENDENCY`.
    """
    for recipe in INSTALL_RECIPES:
        if (output_dir / recipe.lockfile).is_file():
            if (output_dir / recipe.marker).is_dir():
                return None
            return recipe
    return None


def _run_install(recipe: Install, cwd: Path) -> tuple[bool, str]:
    """Run it and return (ok, output tail). Never raises."""
    try:
        finished = subprocess.run(
            list(recipe.argv),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=INSTALL_TIMEOUT,
        )
    except FileNotFoundError:
        return False, f"{recipe.argv[0]} is not installed on this machine"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {INSTALL_TIMEOUT:.0f}s"
    except OSError as exc:
        return False, str(exc)

    if finished.returncode == 0:
        return True, ""
    tail = (finished.stderr or finished.stdout or b"").decode("utf-8", "replace")
    return False, tail.strip()[-2000:]


@dataclass(frozen=True, slots=True)
class Launch:
    """A start command that has been checked, with what to probe once it runs."""

    argv: tuple[str, ...]
    cwd: Path
    label: str
    ports: tuple[int, ...]
    ready_path: str = ""
    """Probed once the port opens, for a declared launch only. When the contract
    declares nothing, its pages are the readiness evidence and this stays empty."""
    declared: bool = False
    port_declared: bool = False
    """True when the port came from the contract rather than from the source.
    Changes what a boot timeout means: a stated port that never opens is most
    likely the statement being wrong, not the server being slow."""


def _source_of(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _declared_launch(
    output_dir: Path, spec: LaunchSpec, *, python: str
) -> tuple[Launch | None, str]:
    """Check a contract's own launch command, or refuse it with a reason.

    Refusing is not falling back. A contract that states how to start itself and
    states something we will not run gets a skip that names the problem —
    quietly guessing instead would start a different program from the one the
    plan declared and report the result as that plan's.
    """
    raw = [str(a) for a in spec.command]
    if len(raw) > MAX_LAUNCH_ARGS:
        return None, (
            f"declared launch has {len(raw)} arguments, over the "
            f"{MAX_LAUNCH_ARGS} allowed"
        )

    program = PurePosixPath(raw[0].replace("\\", "/")).name.lower()
    if program.endswith(".exe"):
        program = program[:-4]
    if program not in INTERPRETERS:
        return None, (
            f"declared launch starts {raw[0]!r}, which is not one of "
            + ", ".join(sorted(INTERPRETERS))
        )

    root = output_dir.resolve()
    argv = [python if program in _PYTHON else program]
    files: list[Path] = []

    for arg in raw[1:]:
        if "\x00" in arg:
            return None, "declared launch contains a null byte"
        if _PATHISH.search(arg):
            # Anything naming a file goes through the same containment check
            # that materialization uses, for the same reason: it came from a
            # model, and this one ends in an exec rather than a write.
            try:
                target = safe_join(root, arg)
            except UnsafePath as exc:
                return None, f"declared launch names {arg!r} — {exc}"
            if not target.is_file():
                return None, f"declared launch names {arg!r}, which no node produced"
            files.append(target)
            argv.append(target.relative_to(root).as_posix())
        elif _PLAIN_ARG.match(arg):
            argv.append(arg)
        else:
            return None, f"declared launch has an argument this will not pass: {arg!r}"

    if not files:
        return None, "declared launch names no file from the output folder"

    source = _source_of(files[0])
    if _looks_degraded(source):
        return None, f"{files[0].name} is a degraded stub, not an implementation"

    port_declared = spec.port is not None
    if spec.port is None:
        ports = candidate_ports(source)
    elif isinstance(spec.port, int) and 1 <= spec.port <= 65535:
        ports = (spec.port,)
    else:
        return None, f"declared port {spec.port!r} is not a port number"

    ready = str(spec.ready_path or "/").strip()
    if not ready.startswith("/") or "\x00" in ready or len(ready) > 200:
        ready = "/"

    return (
        Launch(
            argv=tuple(argv),
            cwd=root,
            label=files[0].name,
            ports=ports,
            ready_path=ready,
            declared=True,
            port_declared=port_declared,
        ),
        "",
    )


def _discovered_launch(output_dir: Path, *, python: str) -> tuple[Launch | None, str]:
    """The inference for a contract that declares nothing: the pinned stack."""
    entrypoint = find_entrypoint(output_dir)
    if entrypoint is None:
        return None, (
            f"no entrypoint in {output_dir} and the contract declares no launch "
            f"command (looked for {', '.join(ENTRYPOINTS)})"
        )

    source = _source_of(entrypoint)
    if _looks_degraded(source):
        return None, f"{entrypoint.name} is a degraded stub, not an implementation"

    return (
        Launch(
            argv=(python, entrypoint.name),
            cwd=output_dir.resolve(),
            label=entrypoint.name,
            ports=candidate_ports(source),
        ),
        "",
    )


def plan_launch(
    output_dir: Path, interface: InterfaceContract, *, python: str | None = None
) -> tuple[Launch | None, str]:
    """How to start this project, and on what ports. Returns (launch, refusal).

    Validated here rather than where the contract is parsed, because a contract
    reaches this point from three directions — a model's decomposition, a
    checkpoint, or a plan-cache file somebody edited — and only one of those
    passes through the parser.
    """
    spec = interface.launch
    if not isinstance(spec, LaunchSpec):
        spec = LaunchSpec()
    interpreter = python or sys.executable
    if spec.declared:
        return _declared_launch(Path(output_dir), spec, python=interpreter)
    return _discovered_launch(Path(output_dir), python=interpreter)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

# Built once, with proxies explicitly disabled: an `http_proxy` in the
# environment would otherwise send a request for 127.0.0.1 to the proxy, and the
# failure would be reported as the generated server not answering.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(
    base: str, method: str, path: str, *, body: bytes | None = None, timeout: float
) -> tuple[int | None, bytes, str]:
    """Returns (status, body, detail). Status is None when nothing answered."""
    request = urllib.request.Request(base + path, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(65536), ""
    except urllib.error.HTTPError as exc:
        # A 4xx/5xx is an answer, not a failure to answer.
        return exc.code, exc.read(65536), exc.reason or ""
    except urllib.error.URLError as exc:
        return None, b"", str(exc.reason)
    except (TimeoutError, socket.timeout):
        return None, b"", f"no response within {timeout:.0f}s"
    except OSError as exc:
        return None, b"", str(exc)


def sample_body(interface: InterfaceContract, model_name: str) -> dict[str, Any]:
    """Invent a request body from the contract's data model.

    Values are placeholders by type, so anything the server rejects it rejects
    on shape. That is why a 4xx here is only ever a warning.
    """
    for model in interface.data_models:
        if not isinstance(model, dict) or model.get("name") != model_name:
            continue
        fields = model.get("fields")
        if not isinstance(fields, dict):
            return {}
        body: dict[str, Any] = {}
        for name, kind in fields.items():
            text = str(kind).lower()
            if "int" in text or "number" in text or "float" in text:
                body[name] = 1
            elif "bool" in text:
                body[name] = True
            elif "[]" in text or "array" in text or "list" in text:
                body[name] = []
            else:
                body[name] = f"llmorch smoke test {name}"
        return body
    return {}


def _created_id(payload: bytes) -> str | None:
    try:
        data = json.loads(payload or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(data, dict):
        for key in ("id", "note_id", "item_id"):
            value = data.get(key)
            if isinstance(value, (int, str)) and str(value).strip():
                return str(value)
    return None


def fill_placeholders(path: str, value: str) -> str:
    parts = [p for p in path.split("/") if p]
    return "/" + "/".join(value if _PLACEHOLDER.match(p) else p for p in parts)


def _has_placeholder(path: str) -> bool:
    return any(_PLACEHOLDER.match(p) for p in path.split("/") if p)


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------


def _launch(launch: Launch) -> tuple[subprocess.Popen, Any, Any]:
    """Start the project. Output goes to temp files, not pipes.

    A pipe would deadlock the moment a chatty server filled the OS buffer, and
    the whole point of this step is to survive whatever the generated code does.
    """
    out = tempfile.TemporaryFile(mode="w+b")
    err = tempfile.TemporaryFile(mode="w+b")
    proc = subprocess.Popen(
        # Executing the artifact is the point of this module; the decision to
        # allow it was made by the caller, before we got here. `shell` is never
        # set, so the argv `plan_launch` validated is the argv that runs.
        list(launch.argv),
        cwd=str(launch.cwd),
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        # Its own process group, so a server that spawns workers can be taken
        # down as a unit rather than leaving orphans holding the port.
        start_new_session=(sys.platform != "win32"),
    )
    return proc, out, err


def _signal(proc: subprocess.Popen, sig: int) -> None:
    """Signal the whole process group where the platform has one.

    A server that forked workers leaves them holding the port if only the parent
    is signalled, and the next run then finds the port occupied and skips.
    Windows has no process group to signal, so the child alone is the best
    available.
    """
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, sig)
        else:
            # No process group to signal, and `terminate` and `kill` are the
            # same TerminateProcess call, so the escalation has nowhere to go.
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        # Already gone, or not ours to signal. Fall back to the direct child.
        try:
            proc.terminate()
        except OSError:
            pass


def _shutdown(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    _signal(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    _signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Unkillable — reported nowhere, because there is nothing the caller
        # could do about it and the run's findings are still valid.
        pass


def _read_tail(handle: Any, limit: int = 2000) -> str:
    try:
        handle.seek(0)
        data = handle.read()
    except (OSError, ValueError):
        return ""
    text = data.decode("utf-8", errors="replace").strip()
    return text[-limit:]


def _wait_for_boot(
    proc: subprocess.Popen, ports: tuple[int, ...], deadline: float
) -> int | None:
    """Poll until one candidate port answers, or the process dies, or time runs out."""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        for port in ports:
            if not port_is_free(port):
                return port
        time.sleep(0.1)
    return None


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _blame_missing_dependency(report: SmokeReport, *, installed: bool) -> None:
    """Say so when a project died for want of something nobody installed.

    Without this the report blames the code: a `Cannot find module` traceback
    under a heading about the project failing to start reads as the model having
    written a bad import, when the harness started it in an empty folder.
    """
    haystack = report.stderr_tail.lower()
    if not any(sign in haystack for sign in _MISSING_DEPENDENCY):
        return
    report.add(
        "warning",
        "it died for want of a dependency, which this run did not install",
        why=(
            "the install that ran did not cover it"
            if installed
            else "no lockfile was found here that `--smoke-install` knows how to use"
        ),
    )


def _probe_ready(
    base: str, launch: Launch, report: SmokeReport, timeout: float
) -> None:
    """One request at the path the contract says answers when the server is up.

    An open port means a socket was bound, which is not the same as a server
    that serves — and for an API with no pages, this is the only thing that
    would notice the difference. A 404 passes: a project with no root route is a
    normal shape, and the contract only claimed this path is where to look.
    """
    status, _, detail = _request(base, "GET", launch.ready_path, timeout=timeout)
    report.probes.append(Probe("GET", launch.ready_path, status, detail))
    if status is None:
        report.add(
            "error",
            f"`{launch.ready_path}` got no response ({detail})",
            where=launch.label,
            why="the port is open but nothing is answering on it",
        )
    elif status >= 500:
        report.add(
            "error",
            f"`{launch.ready_path}` returned {status}",
            where=launch.label,
            why="the contract names this path as the sign the server is up",
        )


def _probe_pages(
    base: str, interface: InterfaceContract, report: SmokeReport, timeout: float
) -> dict[str, str]:
    """GET every page the contract promises. Returns the bodies that loaded."""
    bodies: dict[str, str] = {}
    for page in interface.pages:
        path = "/" + str(page).lstrip("/")
        status, payload, detail = _request(base, "GET", path, timeout=timeout)
        report.probes.append(Probe("GET", path, status, detail))
        if status == 200:
            bodies[path] = payload.decode("utf-8", errors="replace")
        elif status is None:
            report.add(
                "error",
                f"`{path}` got no response ({detail})",
                where=str(page),
                why="the server is running but not answering for this page",
            )
        else:
            report.add(
                "error",
                f"`{path}` returned {status}",
                where=str(page),
                why="the contract promises this page and the server does not serve it",
            )
    return bodies


def _probe_assets(
    base: str, pages: dict[str, str], report: SmokeReport, timeout: float
) -> None:
    """Fetch what the loaded pages link to.

    `contracts.check_assets_resolve` already proves the file was written. This
    proves the server hands it back — the difference between the two is exactly
    the fault where correct files are served from the wrong root.
    """
    # The pages themselves are already probed, and they link to each other: a
    # back-link from the detail page to the index is a real reference, and
    # fetching the index a second time to confirm it is still 200 only makes the
    # report harder to read.
    seen: set[str] = set(pages)
    for page_path, html in pages.items():
        for _, target in local_asset_refs(html):
            path = "/" + target
            if path in seen:
                continue
            seen.add(path)
            status, _, detail = _request(base, "GET", path, timeout=timeout)
            report.probes.append(Probe("GET", path, status, detail))
            if status != 200:
                report.add(
                    "error",
                    f"`{page_path}` links `{path}`, which returns "
                    f"{status if status is not None else detail}",
                    where=page_path,
                    why="the file exists but the server does not serve it from there",
                )


def _probe_routes(
    base: str, interface: InterfaceContract, report: SmokeReport, timeout: float
) -> None:
    """Drive the declared API, creating a record first so reads have one to find."""
    routes = [
        r
        for r in interface.routes
        if isinstance(r, dict) and str(r.get("path", "")).strip()
    ]

    def _method(route: dict) -> str:
        return str(route.get("method", "GET")).upper()

    created_id: str | None = None

    # POST first: a GET for `{id}` is only meaningful once something exists.
    for route in routes:
        if _method(route) != "POST" or _has_placeholder(str(route["path"])):
            continue
        path = "/" + str(route["path"]).lstrip("/")
        payload = sample_body(interface, str(route.get("accepts", "")))
        body = json.dumps(payload).encode("utf-8")
        status, response, detail = _request(
            base, "POST", path, body=body, timeout=timeout
        )
        report.probes.append(Probe("POST", path, status, detail))
        if status is None:
            report.add("error", f"`POST {path}` got no response ({detail})", where=path)
        elif status >= 500:
            report.add(
                "error",
                f"`POST {path}` returned {status}",
                where=path,
                why="the server raised on a request its own contract describes",
            )
        elif status >= 400:
            report.add(
                "warning",
                f"`POST {path}` returned {status} for a synthesized body",
                where=path,
                why="the body was invented here, so the server may be "
                "right to refuse it",
            )
        elif created_id is None:
            created_id = _created_id(response)

    for route in routes:
        if _method(route) != "GET":
            continue
        raw = "/" + str(route["path"]).lstrip("/")
        parameterized = _has_placeholder(raw)
        path = fill_placeholders(raw, created_id or "1") if parameterized else raw
        status, _, detail = _request(base, "GET", path, timeout=timeout)
        report.probes.append(Probe("GET", path, status, detail))

        if status is None:
            report.add("error", f"`GET {path}` got no response ({detail})", where=path)
        elif status >= 500:
            report.add(
                "error",
                f"`GET {path}` returned {status}",
                where=path,
                why="the server raised on a route its own contract declares",
            )
        elif status == 404 and parameterized and created_id is None:
            # Nothing was created, so an empty store is a legitimate 404 rather
            # than a missing route.
            report.add(
                "warning",
                f"`GET {path}` returned 404 with no record to fetch",
                where=path,
                why="nothing was created first, so this proves nothing either way",
            )
        elif status >= 400:
            report.add(
                "error",
                f"`GET {path}` returned {status}",
                where=path,
                why="the contract declares this route and the server does not serve it",
            )

    # Methods beyond GET and POST are left alone: the contract gives nothing to
    # assert about a DELETE, and running one would destroy the record just made.


def smoke_run(
    output_dir: Path,
    interface: InterfaceContract,
    *,
    python: str | None = None,
    install: bool = False,
    boot_timeout: float = 15.0,
    request_timeout: float = 5.0,
) -> SmokeReport:
    """Launch the generated project, drive its contract, and shut it down.

    Never raises on a bad artifact: a project that cannot start is the finding,
    not an error in the harness. `SmokeReport.ran` distinguishes "ran and passed"
    from "never got far enough to tell".
    """
    report = SmokeReport()
    output_dir = Path(output_dir)

    launch, refusal = plan_launch(output_dir, interface, python=python)
    if launch is None:
        report.skipped = refusal
        return report
    report.entrypoint = launch.label

    # Dependencies, before anything is started. A project launched without them
    # dies on its first import, and the resulting traceback is about the wrong
    # thing entirely.
    recipe = install_plan(launch.cwd)
    if recipe is not None:
        if not install:
            report.skipped = (
                f"{launch.label} needs dependencies: {recipe.lockfile} is present "
                f"and {recipe.marker}/ is not. Re-run with --smoke-install to run "
                f"`{recipe.label}` first (it reaches the network)."
            )
            return report
        ok, detail = _run_install(recipe, launch.cwd)
        report.installed = recipe.label
        if not ok:
            report.stderr_tail = detail
            report.add(
                "error",
                f"`{recipe.label}` failed",
                where=recipe.lockfile,
                why="the project was never started, so nothing here is its fault",
            )
            return report

    ports = launch.ports
    free = usable_ports(ports)
    if not free:
        report.skipped = (
            f"port {', '.join(str(p) for p in ports)} already in use — refusing to "
            "probe a server this run did not start"
        )
        return report
    if len(free) < len(ports):
        busy = ", ".join(str(p) for p in ports if p not in free)
        report.add(
            "warning",
            f"port {busy} was already in use and was not probed",
            why="a response from it would have come from another server",
        )

    proc, out, err = _launch(launch)
    try:
        port = _wait_for_boot(proc, free, time.monotonic() + boot_timeout)

        if port is None:
            report.stderr_tail = _read_tail(err) or _read_tail(out)
            if proc.poll() is not None:
                report.add(
                    "error",
                    f"{launch.label} exited with code {proc.returncode} "
                    "instead of serving",
                    where=launch.label,
                    why="the project does not start at all",
                )
                _blame_missing_dependency(report, installed=install)
            else:
                report.add(
                    "error",
                    f"{launch.label} bound no port within {boot_timeout:.0f}s "
                    f"(tried {', '.join(str(p) for p in free)})",
                    where=launch.label,
                    why=(
                        "the contract declares this port and the server is "
                        "running without it — most likely the declaration and "
                        "the code disagree"
                        if launch.port_declared
                        else "it is running but never began serving"
                    ),
                )
            return report

        report.ran = True
        report.port = port
        base = f"http://127.0.0.1:{port}"

        # Skipped when the ready path is also a declared page: the page check is
        # the stricter of the two — it counts a 404 as a failure, where this one
        # accepts it — so probing both would add a row and take evidence away.
        pages_declared = {str(p).lstrip("/") for p in interface.pages}
        if launch.ready_path and launch.ready_path.lstrip("/") not in pages_declared:
            _probe_ready(base, launch, report, request_timeout)
        pages = _probe_pages(base, interface, report, request_timeout)
        _probe_assets(base, pages, report, request_timeout)
        _probe_routes(base, interface, report, request_timeout)

        if proc.poll() is not None:
            report.add(
                "error",
                f"{launch.label} died while serving (exit code {proc.returncode})",
                where=launch.label,
            )
        return report
    finally:
        _shutdown(proc)
        report.stderr_tail = report.stderr_tail or _read_tail(err)
        # Only for a server that got as far as serving. A process that died on
        # startup already has its exit reported, and its traceback is the same
        # traceback — saying it "raised while serving" would be two findings for
        # one fault, and the second one untrue.
        if report.ran and _TRACEBACK in report.stderr_tail:
            report.add(
                "error",
                f"{launch.label} raised while serving",
                where=launch.label,
                why="a traceback reached stderr; the tail is below",
            )
        out.close()
        err.close()
