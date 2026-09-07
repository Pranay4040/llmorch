"""The setup page's server — the one place in this project that accepts a write.

The dashboard's header states the reason it needs no authentication: it is
read-only, so a page that can only look needs no threat model. This server can
save settings and write an API key into `.env`, so that argument does not cover
it and three things are enforced here instead.

**A token, minted per launch.** It is in the URL the browser is opened with, and
every request must carry it. Without one, a page on any other origin could post
to `http://127.0.0.1:8788/api/keys` while this is running — the browser would
send the request happily, and same-origin policy only stops it *reading* the
reply, which an attacker writing a key does not need.

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
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .. import settings as settings_module
from ..config import KeyRejected, env_path, has_api_key, load_dotenv, write_env_key
from ..registry.manifest import load_manifest
from .page import PAGE

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORT = 8788
MAX_BODY = 64 * 1024


class ConfigureError(RuntimeError):
    pass


def new_token() -> str:
    return secrets.token_urlsafe(24)


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
        {"name": role.value, "models": list(chain)}
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
            self._send(
                401,
                "text/plain",
                b"missing or wrong token; open the URL llmorch printed",
            )
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
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    open_browser: bool = True,
) -> None:
    """Run until interrupted, opening the page on the way up."""
    token = new_token()
    httpd = build_server(host, port, token=token)
    bound = httpd.server_address
    url = f"http://{bound[0]}:{bound[1]}/?t={token}"

    print(f"llmorch setup on {url}")
    print("  loopback only; the token in that URL is what lets it save.")
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
