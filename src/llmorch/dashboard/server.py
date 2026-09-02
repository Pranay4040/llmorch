"""A read-only window on the orchestrator, served to localhost only.

Two constraints were decided before a line of this existed, and both are
enforced here rather than documented and hoped for.

**Read-only.** There is no endpoint that starts a run, cancels one, or edits
anything. The dashboard shows state that other commands produce. A page that
could spend quota would need authentication, rate limiting, and a threat model;
a page that can only look needs none of that, and the difference is the reason
this milestone was cheap enough to leave until last.

**Localhost only.** The socket binds to 127.0.0.1, and a non-loopback host is
refused rather than quietly accepted. The ledger holds a full record of what
this account has spent, and nothing here asks who is asking.

One further rule follows from what the data *is*: every string on the page may
have come from a model — error text, node ids, task descriptions. The server
therefore ships no interpolated markup at all. It sends a static page and a
JSON document, and the page writes values through `textContent`, which cannot
execute anything it is handed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .page import PAGE
from .state import snapshot

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORT = 8787


class DashboardError(RuntimeError):
    pass


class _Handler(BaseHTTPRequestHandler):
    """Three routes, all GET, none of them able to change anything."""

    server_version = "llmorch-dashboard"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the stdlib
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif path == "/api/state":
            try:
                body = json.dumps(snapshot(), default=str).encode("utf-8")
            except Exception as exc:  # a broken ledger must not take the page down
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, "application/json", body)
                return
            self._send(200, "application/json", body)
        elif path == "/healthz":
            self._send(200, "text/plain", b"ok")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        """Refused by design, not by omission."""
        self._drain()
        self._send(405, "text/plain", b"this dashboard is read-only")

    do_PUT = do_DELETE = do_PATCH = do_POST

    def _drain(self) -> None:
        """Read and discard a request body before refusing it.

        On a keep-alive connection the unread body would sit in the socket and
        be parsed as the start of the next request, which surfaces to the client
        as a connection reset rather than the 405 it was actually sent. Capped,
        because this is a body nobody asked for.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        remaining = min(max(0, length), 1 << 20)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # No external anything: the page is entirely self-contained, and saying
        # so means a stray CDN link becomes a visible failure rather than a
        # silent request to somebody else's server.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default: this runs alongside a build, and one line per poll
        would bury whatever the person was actually watching."""
        return


def build_server(
    host: str = "127.0.0.1", port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise DashboardError(
            f"refusing to bind {host!r}: this dashboard has no authentication and "
            "shows a full record of account spend, so it serves loopback only"
        )
    # Threading, not the plain HTTPServer. The page holds a keep-alive
    # connection open and polls it every five seconds; a single-threaded server
    # would spend its whole life inside that one connection, and a second
    # client — another tab, or a curl to check something — would hang until the
    # browser was closed. Found exactly that way.
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    return server


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Run until interrupted."""
    httpd = build_server(host, port)
    address = httpd.server_address
    print(f"llmorch dashboard on http://{address[0]}:{address[1]}  (ctrl-c to stop)")
    print("  read-only, loopback only, no authentication")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()


def serve_in_thread(port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start on a background thread and return immediately.

    Used by the tests, which need a real socket to prove the routes behave —
    a handler exercised only through mocks proves the mock works.
    """
    httpd = build_server("127.0.0.1", port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread
