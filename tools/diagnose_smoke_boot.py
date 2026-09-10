"""Why does a started server never answer on this machine?

Temporary. Every smoke test fails on macOS with the child process alive, silent,
and never listening — and the two obvious causes are ruled out by reading the
code: output already goes to temp files rather than pipes, and `python` is
substituted with `sys.executable`. So this reproduces the exact pattern the
smoke run uses, one step at a time, and says which step behaves differently.

Delete this once the answer is known.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SERVER = '''\
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = {port}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *a):
        pass

print("about to bind", PORT, flush=True)
server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
print("bound", server.server_address, flush=True)
server.serve_forever()
'''


def free_port() -> int:
    """Exactly what tests/test_smoke.py does."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def can_connect(host: str, port: int, family: int = socket.AF_INET) -> bool:
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    print(f"platform     : {sys.platform}")
    print(f"executable   : {sys.executable}")
    print(f"version      : {sys.version.splitlines()[0]}")

    port = free_port()
    print(f"free_port()  : {port}")

    # Can this process bind it back immediately? If not, neither can the child.
    try:
        probe = socket.socket()
        probe.bind(("127.0.0.1", port))
        probe.close()
        print("rebind       : ok")
    except OSError as exc:
        print(f"rebind       : FAILED {exc}")

    work = Path(tempfile.mkdtemp())
    (work / "server.py").write_text(SERVER.format(port=port), encoding="utf-8")

    out = tempfile.TemporaryFile(mode="w+b")
    err = tempfile.TemporaryFile(mode="w+b")
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=str(work),
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        start_new_session=(sys.platform != "win32"),
    )
    print(f"spawned pid  : {proc.pid}")

    answered = None
    deadline = started + 20.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print(f"child exited : code {proc.returncode} "
                  f"after {time.monotonic() - started:.2f}s")
            break
        if can_connect("127.0.0.1", port):
            answered = time.monotonic() - started
            print(f"answered     : after {answered:.2f}s")
            break
        time.sleep(0.1)
    else:
        print("answered     : NEVER within 20s")

    print(f"alive at end : {proc.poll() is None}")
    for host, family in (("127.0.0.1", socket.AF_INET),
                         ("localhost", socket.AF_INET),
                         ("::1", socket.AF_INET6)):
        try:
            print(f"connect {host:<10}: {can_connect(host, port, family)}")
        except OSError as exc:
            print(f"connect {host:<10}: error {exc}")

    for name, handle in (("stdout", out), ("stderr", err)):
        handle.seek(0)
        body = handle.read().decode("utf-8", "replace").strip()
        print(f"{name}       : {body!r}")

    try:
        proc.kill()
    except OSError:
        pass

    if sys.platform != "win32":
        listening = subprocess.run(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=20,
        )
        rows = [r for r in listening.stdout.splitlines() if str(port) in r]
        print(f"lsof rows    : {rows or 'none for this port'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
