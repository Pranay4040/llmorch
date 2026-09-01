"""The read-only dashboard.

Two constraints were decided before this existed and are the reason it was
cheap enough to build last: it can only look, and it can only be reached from
this machine. Both are tested here against a real socket, because a handler
exercised only through mocks proves the mock works.

The third property is subtler. Much of what this page displays was written by a
language model — error text, node ids, task descriptions — so the server sends
no interpolated markup at all, and a provider's error string has nowhere to
execute.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from llmorch.dashboard.page import PAGE
from llmorch.dashboard.server import DashboardError, build_server, serve_in_thread
from llmorch.dashboard.state import snapshot


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """A server on an ephemeral port, against an empty state directory."""
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path / "runs"))
    httpd, _thread = serve_in_thread(port=0)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.headers, response.read().decode("utf-8")


# ==========================================================================
# It can only look
# ==========================================================================


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_nothing_can_be_changed_through_the_dashboard(dashboard, method):
    """Refused by design rather than by omission. A page that could spend quota
    would need authentication and a threat model; a page that can only look
    needs neither."""
    request = urllib.request.Request(f"{dashboard}/api/state", method=method, data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=10)
    assert exc.value.code == 405


def test_the_page_is_served(dashboard):
    status, headers, body = _get(f"{dashboard}/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert "llmorch" in body


def test_state_is_json_and_complete(dashboard):
    status, headers, body = _get(f"{dashboard}/api/state")
    assert status == 200
    assert headers["Content-Type"] == "application/json"

    state = json.loads(body)
    for key in ("quota", "spend", "runs", "track_record", "recent", "roster", "paths"):
        assert key in state, key


def test_an_unknown_path_is_a_plain_404(dashboard):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{dashboard}/../../etc/passwd")
    assert exc.value.code == 404


def test_health_check_needs_no_ledger(dashboard):
    assert _get(f"{dashboard}/healthz")[0] == 200


# ==========================================================================
# It can only be reached from here
# ==========================================================================


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", ""])
def test_binding_beyond_loopback_is_refused(host):
    """There is no authentication, and the ledger is a complete record of what
    this account has spent."""
    with pytest.raises(DashboardError) as exc:
        build_server(host=host, port=0)
    assert "loopback" in str(exc.value)


def test_loopback_is_allowed():
    httpd = build_server(host="127.0.0.1", port=0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


# ==========================================================================
# It cannot be made to execute what a model wrote
# ==========================================================================


def test_the_page_never_assigns_markup():
    """Every value is written with textContent. One markup assignment would be
    enough for a provider's error message to run in the watcher's browser.

    Checks for the dangerous *forms*, not the words: the page mentions
    innerHTML in a comment explaining why it does not use it.
    """
    import re

    for pattern in (
        r"innerHTML\s*=",
        r"outerHTML\s*=",
        r"insertAdjacentHTML\s*\(",
        r"document\.write\s*\(",
        r"eval\s*\(",
        r"new\s+Function\s*\(",
    ):
        assert not re.search(pattern, PAGE), pattern

    assert "textContent" in PAGE


def test_the_page_fetches_nothing_from_outside(dashboard):
    """Self-contained by construction — and the response says so, so a stray
    CDN link becomes a visible failure rather than a silent request to someone
    else's server."""
    _status, headers, body = _get(f"{dashboard}/")
    assert "https://" not in body.replace("http://127.0.0.1", "")
    assert "default-src 'none'" in headers["Content-Security-Policy"]


def test_model_written_text_survives_as_data(dashboard, tmp_path):
    """An error string containing markup must come back as characters, not as
    something the browser would parse."""
    from llmorch.quota.store import LedgerStore, make_event
    from llmorch.registry.manifest import load_manifest
    from llmorch.types import Usage

    hostile = "<script>alert('x')</script>"
    with LedgerStore(tmp_path / "state.db") as store:
        store.record(
            make_event(
                run_id="r1", node_id="n1", purpose="execute",
                manifest=load_manifest(), model_id="groq/gpt-oss-120b",
                usage=Usage(), ok=False, http_status=500, error=hostile,
            )
        )

    state = json.loads(_get(f"{dashboard}/api/state")[1] and _get(f"{dashboard}/api/state")[2])
    errors = [row["error"] for row in state["recent"]]
    assert hostile in errors, "the ledger row should be visible as data"


# ==========================================================================
# It agrees with the CLI
# ==========================================================================


def test_the_snapshot_replays_the_ledger_like_every_other_command(tmp_path, monkeypatch):
    """A dashboard computing its own numbers would eventually disagree with
    `llmorch quota`, and then nobody knows which to believe."""
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("LLMORCH_RUNS_DIR", str(tmp_path / "runs"))

    from llmorch.quota.store import LedgerStore, make_event
    from llmorch.registry.manifest import load_manifest
    from llmorch.types import Usage

    manifest = load_manifest()
    with LedgerStore(tmp_path / "state.db") as store:
        for i in range(3):
            store.record(
                make_event(
                    run_id="r1", node_id=f"n{i}", purpose="execute",
                    manifest=manifest, model_id="groq/gpt-oss-120b",
                    usage=Usage(prompt_tokens=100, completion_tokens=50),
                )
            )

    state = snapshot()
    row = next(q for q in state["quota"] if q["model_id"] == "groq/gpt-oss-120b")
    assert row["requests_used"] == 3
    assert 0 < row["fraction"] < 1


def test_estimated_limits_are_flagged_as_such(tmp_path, monkeypatch):
    """OpenRouter's limits are a guess until a 429 says otherwise, and the page
    should not present a guess as a measurement."""
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    state = snapshot()
    openrouter = [q for q in state["quota"] if q["provider"] == "openrouter"]
    assert openrouter and all(q["estimated"] for q in openrouter)
    groq = [q for q in state["quota"] if q["provider"] == "groq"]
    assert groq and not any(q["estimated"] for q in groq)


def test_the_snapshot_exposes_locations_but_never_contents(tmp_path, monkeypatch):
    """Paths help someone find their data; a key or an artifact on the page
    would be a leak with no upside."""
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))
    state = snapshot()
    blob = json.dumps(state).lower()
    assert set(state["paths"]) == {"ledger", "profiles", "runs", "plans"}
    for secret_ish in ("api_key", "authorization", "bearer", "sk-"):
        assert secret_ish not in blob


def test_a_second_client_is_not_blocked_by_the_first(dashboard):
    """The page holds a keep-alive connection and polls it every five seconds.
    On a single-threaded server that one connection owns the process, and
    anything else — another tab, a curl to check a number — waits for the
    browser to close. Found by doing exactly that."""
    import http.client

    holder = http.client.HTTPConnection(dashboard.replace("http://", ""), timeout=10)
    holder.request("GET", "/api/state")
    holder.getresponse().read()          # connection stays open, keep-alive

    try:
        assert _get(f"{dashboard}/healthz")[0] == 200
    finally:
        holder.close()
