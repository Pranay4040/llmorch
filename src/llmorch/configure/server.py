"""The setup page's server — the one place in this project that accepts a write.

The dashboard's header states the reason it needs no authentication: it is
read-only, so a page that can only look needs no threat model. This server can
save settings and write an API key into `.env`, so that argument does not cover
it and three things are enforced here instead.

**A token, kept for this machine.** It is in the URL the browser is opened with,
and every request must carry it. Without one, a page on any other origin could
post to `http://127.0.0.1:8788/api/keys` while this is running — the browser
would send the request happily, and same-origin policy only stops it *reading*
the reply, which an attacker writing a key does not need.

It was minted per launch to begin with, and that was wrong for a reason worth
recording: it made every bookmark and every reopened tab into a dead end. A
stable secret stops a cross-origin post exactly as well as a fresh one, and the
thing it does not stop — a local process reading the token file — is a process
that can already read `.env`. See `load_or_create_token`.

**A loopback `Host`.** The socket binds to 127.0.0.1, but binding is not enough
on its own: a hostile name resolving to 127.0.0.1 makes a cross-origin page
same-origin as far as the browser is concerned. Checking the header the browser
sends is what closes that.

**Keys go one way.** `GET /api/config` reports whether each key is *set*. There
is no endpoint that returns one, so nothing that can read this page can read the
account's secrets — which matters because the page is on a port that anything
on this machine may connect to.

Runs still cannot be started from here. Configuration is a small, bounded
surface; spending quota is not, and `llmorch start` is a terminal away.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .. import settings as settings_module
from ..config import (
    KeyRejected,
    env_path,
    has_api_key,
    load_dotenv,
    state_db_path,
    write_env_key,
)
from ..registry.manifest import load_manifest
from .page import PAGE

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORT = 8788
MAX_BODY = 64 * 1024

# The role taxonomy is closed and internal — `profiles.json` keys a track record
# on it, so the names cannot move. What can move is what they are called on a
# page: "research" is not what anybody would call the thing that answers their
# questions, and a person choosing a model for a job should be reading the job.
ROLE_LABELS = {
    "planning": (
        "Planner",
        "Splits your instruction into files and writes the contract the others "
        "are held to. One request the whole run depends on.",
    ),
    "research": (
        "Chat & questions",
        "Answers `llmorch ask` and any question you type in a session. Never "
        "writes a file.",
    ),
    "backend": (
        "Backend",
        "Servers, routes, database access.",
    ),
    "frontend": (
        "Frontend",
        "Pages and browser scripts.",
    ),
    "styling": (
        "Styling",
        "Stylesheets.",
    ),
    "content": (
        "Docs & text",
        "READMEs, sample data, prose.",
    ),
    "integration": (
        "Glue & setup",
        "Wiring, config, entry points.",
    ),
    "review": (
        "Reviewer",
        "Reads another model's file and says whether it does what was asked. "
        "Always from a different vendor than the author — a pin that shares the "
        "author's vendor is skipped for that file.",
    ),
}


class ConfigureError(RuntimeError):
    pass


# A browser asked for a page, so it gets one. The old reply was a line of plain
# text telling somebody to find a URL they no longer had, which is a dead end
# reached by doing something reasonable.
STALE_LINK = """<!doctype html>
<meta charset="utf-8">
<title>llmorch setup — wrong link</title>
<style>
  body { background:#0f1115; color:#e6e9ef; margin:0; padding:48px 24px;
         font:14px/1.7 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  @media (prefers-color-scheme: light) { body { background:#f7f8fa; color:#12151b; } }
  main { max-width: 640px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0 0 14px; }
  p { color:#8b93a3; }
  code { border:1px solid #242a34; border-radius:4px; padding:1px 6px; }
</style>
<main>
  <h1>This link is missing its token.</h1>
  <p>
    The setup page will only answer a URL that carries the token for this
    machine, because it can write API keys and any other page in your browser
    could otherwise post to it.
  </p>
  <p>
    Run <code>llmorch</code> in a terminal and open the URL it prints. That link
    now stays the same, so this one will keep working once you have used it.
  </p>
</main>
"""


TOKEN_NAME = "configure-token"
_TOKEN_SHAPE = re.compile(r"^[A-Za-z0-9_-]{20,128}$")


def new_token() -> str:
    return secrets.token_urlsafe(24)


def token_path() -> Path:
    """Beside the ledger and the settings, not in the checkout."""
    return state_db_path().parent / TOKEN_NAME


def load_or_create_token(path: Path | None = None) -> str:
    """The token for this machine, minted once and kept.

    It began as a fresh secret per launch, which is stronger and was wrong: a
    bookmarked page, a reopened tab, or simply restarting the server left you
    with a URL that no longer worked and a line of text telling you to find a
    URL you no longer had. That is a dead end reached by doing something
    reasonable.

    Keeping it costs little that matters. What the token defends against is
    another *web page* posting to this port — CSRF, and a hostile name resolving
    to 127.0.0.1 — and a stable secret stops that exactly as well as a fresh
    one. The threat it does not stop is a local process reading the file, and
    such a process can already read `.env`, which holds the keys themselves.

    A file that is missing, unreadable, or does not look like a token is
    replaced rather than trusted.
    """
    target = path or token_path()
    try:
        existing = target.read_text(encoding="utf-8").strip()
        if _TOKEN_SHAPE.match(existing):
            return existing
    except OSError:
        pass

    token = new_token()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(token, encoding="utf-8")
        # Advisory on Windows, real everywhere else, and cheap in both places.
        os.chmod(target, 0o600)
    except OSError:
        # A token that cannot be saved still works for this run; the URL just
        # stops being stable, which is where this came in.
        pass
    return token


def snapshot() -> dict[str, Any]:
    """Everything the page draws, and nothing that could leak a key."""
    manifest = load_manifest()
    current = settings_module.load()

    providers = [
        {
            "name": name,
            "key_env": spec.api_key_env,
            "key_set": has_api_key(spec.api_key_env),
            "enabled": spec.enabled,
            "paid": spec.paid,
        }
        for name, spec in sorted(manifest.providers.items())
    ]
    models = [
        {
            "id": model.id,
            "provider": model.provider,
            "wire_name": model.wire_name,
            "context": model.context,
            "max_output": model.max_output,
        }
        for model in manifest.models
    ]
    roles = [
        {
            "name": role.value,
            "label": ROLE_LABELS.get(role.value, (role.value, ""))[0],
            "blurb": ROLE_LABELS.get(role.value, (role.value, ""))[1],
            # The declared chain, for the "which models are meant for this job"
            # ordering, and every model as the wider choice: a pin is a person
            # overruling the manifest's preference, so the manifest's preference
            # must not also be the limit of what they can pick.
            "models": list(chain),
            "pinned": current.role_models.get(role.value, ""),
        }
        for role, chain in sorted(manifest.roles.items(), key=lambda kv: kv[0].value)
    ]
    return {
        "providers": providers,
        "models": models,
        "roles": roles,
        "settings": current.to_dict(),
        "env_path": str(env_path()),
        "settings_path": str(settings_module.settings_path()),
    }


def apply_settings(payload: Any) -> dict[str, Any]:
    """Validate and save what the page posted.

    The whole document is rebuilt through `settings.from_dict`, which clamps
    every field, rather than merged key by key into what is on disk: a partial
    write from a page that failed halfway is a configuration nobody chose.
    """
    saved = settings_module.from_dict(payload)
    saved.save()
    return {"ok": True, "settings": saved.to_dict()}


def apply_keys(payload: Any) -> dict[str, Any]:
    """Write the keys that were typed, and no others.

    Reports names, never values. An empty box is absent from the payload
    entirely, so "leave it as it is" and "clear it" stay different requests.
    """
    if not isinstance(payload, dict):
        raise KeyRejected("expected an object of KEY: value")

    manifest = load_manifest()
    known = {spec.api_key_env for spec in manifest.providers.values()}

    written: list[str] = []
    for env_var, value in payload.items():
        if env_var not in known:
            # Only the variables the manifest actually names. Otherwise this is
            # an endpoint for writing arbitrary environment variables into a
            # file the shell may later source.
            raise KeyRejected(f"{env_var} is not a key any declared provider uses")
        if not isinstance(value, str):
            raise KeyRejected(f"the value for {env_var} is not text")
        write_env_key(env_var, value)
        written.append(env_var)

    # So the page's "key set" pills are right on the next load, and so a session
    # started from this process sees the key without a restart.
    load_dotenv(override=True)
    return {"ok": True, "saved": written}


class _Handler(BaseHTTPRequestHandler):
    server_version = "llmorch-configure"
    protocol_version = "HTTP/1.1"
    token = ""

    # -- guards -----------------------------------------------------------

    def _host_is_loopback(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        name = host.rsplit(":", 1)[0].strip("[]") if host else ""
        return name in LOOPBACK

    def _authorised(self) -> bool:
        sent = self.headers.get("X-Llmorch-Token") or ""
        if not sent:
            query = self.path.partition("?")[2]
            for part in query.split("&"):
                key, _, value = part.partition("=")
                if key == "t":
                    sent = value
                    break
        return bool(self.token) and secrets.compare_digest(sent, self.token)

    def _guarded(self) -> bool:
        if not self._host_is_loopback():
            self._send(403, "text/plain", b"loopback only")
            return False
        if not self._authorised():
            self._send(401, "text/html; charset=utf-8", STALE_LINK.encode("utf-8"))
            return False
        return True

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the stdlib
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/healthz":
            self._send(200, "text/plain", b"ok")
            return
        if not self._guarded():
            return

        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/config":
            self._json(200, snapshot())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        body = self._read_body()
        if not self._guarded():
            return

        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "body was not JSON"})
            return

        try:
            if path == "/api/settings":
                self._json(200, apply_settings(payload))
            elif path == "/api/keys":
                self._json(200, apply_keys(payload))
            else:
                self._send(404, "text/plain", b"not found")
        except KeyRejected as exc:
            # The message is written never to quote the value.
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # a bad save must not take the page down
            self._json(500, {"error": type(exc).__name__})

    do_PUT = do_DELETE = do_PATCH = do_POST

    # -- plumbing ---------------------------------------------------------

    def _read_body(self) -> bytes:
        """Read the body before deciding anything about it.

        On a keep-alive connection an unread body is parsed as the start of the
        next request, so a refusal that skipped this would reach the browser as
        a connection reset rather than the 401 it actually was.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return b""
        if length > MAX_BODY:
            return b""
        return self.rfile.read(max(0, length))

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        # The page reflects which keys are set. Nothing about it should sit in a
        # disk cache after the process that served it is gone.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class _Server(ThreadingHTTPServer):
    """Threading, and refusing to share a port.

    `HTTPServer` sets `allow_reuse_address`, which on POSIX only skips
    TIME_WAIT and is wanted. On Windows it means something else entirely: a
    *second* process may bind an address another process is already listening
    on, and which of them a connection reaches is arbitrary. So it is switched
    off there and left alone everywhere else.

    That is not theoretical here. Two `llmorch` setup servers were live on 8788
    at once, and the older one answered a link the newer one had just printed —
    which surfaces as "missing or wrong token" against a URL that is, as far as
    anyone can see, the right one. Refusing the bind turns that into a sentence
    saying it is already running.
    """

    # Windows only. `SO_REUSEADDR` does not mean the same thing on both
    # platforms: on Windows it permits a *second* process to bind an address a
    # first is already listening on, which is the hazard above; on POSIX it only
    # skips TIME_WAIT, and turning it off there means restarting this server
    # shortly after a browser was connected fails with EADDRINUSE — reported by
    # the message below as another instance running, which would be a lie.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        """A browser hanging up is not an error worth a traceback.

        `socketserver` prints fifteen lines to stderr when a client closes a
        keep-alive connection, which on this page happens every time a tab is
        closed or reloaded — landing in the middle of whatever the person was
        reading in that terminal. Anything else still gets printed.
        """
        import sys
        import traceback

        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError,
                                          BrokenPipeError)):
            return
        traceback.print_exc()


def build_server(
    host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, token: str
) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise ConfigureError(
            f"refusing to bind {host!r}: this page writes API keys and settings, "
            "so it serves loopback only"
        )
    if not token:
        raise ConfigureError("refusing to serve without a token")

    handler = type("_BoundHandler", (_Handler,), {"token": token})
    try:
        return _Server((host, port), handler)
    except OSError as exc:
        raise ConfigureError(
            f"port {port} is already in use — llmorch setup may already be "
            f"running. Open http://{host}:{port}/?t=<token> in the terminal "
            f"that started it, or use --port to run a second one ({exc})"
        ) from exc


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    open_browser: bool = True,
) -> None:
    """Run until interrupted, opening the page on the way up."""
    token = load_or_create_token()
    try:
        httpd = build_server(host, port, token=token)
    except ConfigureError as exc:
        # The token is the same one the other instance is using, so the link
        # below is the one that works — which is only true because it is kept
        # rather than minted per launch.
        print(str(exc))
        print("\n  if it is already running, this is its link:")
        print(f"  http://{host}:{port}/?t={token}")
        return
    bound = httpd.server_address
    url = f"http://{bound[0]}:{bound[1]}/?t={token}"

    print(f"llmorch setup on {url}")
    print("  loopback only; the token in that URL is what lets it save.")
    print("  the same link works next time — it is kept, not minted per run.")
    print("  when you are done here:  llmorch start")

    if open_browser:
        # In a thread: on Windows this can block for as long as it takes a
        # browser to start, and the page it opens is served by this process.
        import webbrowser

        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
