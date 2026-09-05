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
from pathlib import Path
from typing import Any

from ..types import InterfaceContract
from .contracts import local_asset_refs

# Tried in order. `server.py` first because it is what the pinned stack and the
# generated README both name.
ENTRYPOINTS = ("server.py", "app.py", "main.py")

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


def _launch(entrypoint: Path, *, python: str) -> tuple[subprocess.Popen, Any, Any]:
    """Start the project. Output goes to temp files, not pipes.

    A pipe would deadlock the moment a chatty server filled the OS buffer, and
    the whole point of this step is to survive whatever the generated code does.
    """
    out = tempfile.TemporaryFile(mode="w+b")
    err = tempfile.TemporaryFile(mode="w+b")
    proc = subprocess.Popen(
        # Executing the artifact is the point of this module; the decision to
        # allow it was made by the caller, before we got here.
        [python, entrypoint.name],
        cwd=str(entrypoint.parent),
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

    entrypoint = find_entrypoint(output_dir)
    if entrypoint is None:
        report.skipped = (
            f"no entrypoint in {output_dir} (looked for "
            f"{', '.join(ENTRYPOINTS)})"
        )
        return report
    report.entrypoint = entrypoint.name

    source = entrypoint.read_text(encoding="utf-8", errors="replace")
    if _looks_degraded(source):
        report.skipped = f"{entrypoint.name} is a degraded stub, not an implementation"
        return report

    ports = candidate_ports(source)
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

    proc, out, err = _launch(entrypoint, python=python or sys.executable)
    try:
        port = _wait_for_boot(proc, free, time.monotonic() + boot_timeout)

        if port is None:
            report.stderr_tail = _read_tail(err) or _read_tail(out)
            if proc.poll() is not None:
                report.add(
                    "error",
                    f"{entrypoint.name} exited with code {proc.returncode} "
                    "instead of serving",
                    where=entrypoint.name,
                    why="the project does not start at all",
                )
            else:
                report.add(
                    "error",
                    f"{entrypoint.name} bound no port within {boot_timeout:.0f}s "
                    f"(tried {', '.join(str(p) for p in free)})",
                    where=entrypoint.name,
                    why="it is running but never began serving",
                )
            return report

        report.ran = True
        report.port = port
        base = f"http://127.0.0.1:{port}"

        pages = _probe_pages(base, interface, report, request_timeout)
        _probe_assets(base, pages, report, request_timeout)
        _probe_routes(base, interface, report, request_timeout)

        if proc.poll() is not None:
            report.add(
                "error",
                f"{entrypoint.name} died while serving (exit code {proc.returncode})",
                where=entrypoint.name,
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
                f"{entrypoint.name} raised while serving",
                where=entrypoint.name,
                why="a traceback reached stderr; the tail is below",
            )
        out.close()
        err.close()
