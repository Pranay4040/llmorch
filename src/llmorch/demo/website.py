"""The notes-app demo: task graph, interface contract, and canned artifacts.

The artifacts here are what the mock provider returns, and they are real working
code rather than filler. That is deliberate: it makes Milestone 1's exit
criterion meaningful — the dry run produces a folder you can actually serve and
open, so the whole pipeline is verifiable by eye before a single token is spent.

The stack is pinned (plain HTML/CSS/JS, stdlib `http.server`, SQLite) so the
contract checker knows what it is parsing and the per-role prompts have a fixed
target.
"""

from __future__ import annotations

from ..types import InterfaceContract, OutputKind, Role, SplitHint, TaskNode

TASK = "build a notes app"

INTERFACE = InterfaceContract(
    routes=(
        {"method": "GET", "path": "/api/notes", "returns": "Note[]"},
        {"method": "GET", "path": "/api/notes/{id}", "returns": "Note"},
        {"method": "POST", "path": "/api/notes", "accepts": "NoteInput", "returns": "Note"},
    ),
    data_models=(
        {
            "name": "Note",
            "fields": {
                "id": "integer",
                "title": "string",
                "body": "string",
                "created_at": "string (ISO 8601)",
            },
        },
        {"name": "NoteInput", "fields": {"title": "string", "body": "string"}},
    ),
    pages=("index.html", "note.html"),
    notes=(
        "Plain HTML/CSS/JS with no framework and no build step. "
        "Backend is Python's stdlib http.server over SQLite. "
        "The API is served from the same origin as the pages."
    ),
)


def build_nodes() -> list[TaskNode]:
    """The DAG.

    Shaped so the three-way split is genuine: the schema must exist before the
    API, and the API contract must exist before the pages that call it. Every
    node stays under ~1,300 tokens of output so it remains servable on a
    6,000 TPM provider.
    """
    return [
        TaskNode(
            id="schema",
            title="SQLite schema",
            role=Role.BACKEND,
            spec=(
                "Write schema.sql defining a `notes` table matching the Note "
                "data model: id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT "
                "NOT NULL, body TEXT NOT NULL DEFAULT '', created_at TEXT NOT "
                "NULL. Add an index on created_at for reverse-chronological "
                "listing. DDL only."
            ),
            output_path="schema.sql",
            output_kind=OutputKind.SCHEMA,
            est_output_tokens=250,
        ),
        TaskNode(
            id="server",
            title="JSON API over SQLite",
            role=Role.BACKEND,
            spec=(
                "Write server.py using only the standard library. Serve the "
                "three routes in the interface contract, plus the static files "
                "index.html, note.html, style.css and app.js from the same "
                "directory. Create the database from schema.sql on first run. "
                "Listen on port 8000."
            ),
            output_path="server.py",
            output_kind=OutputKind.CODE,
            deps=("schema",),
            needs=("schema.summary",),
            est_output_tokens=1300,
            split_hint=SplitHint.PER_ROUTE,
        ),
        TaskNode(
            id="index",
            title="Notes list page",
            role=Role.FRONTEND,
            spec=(
                "Write index.html: a heading, a form for creating a note "
                "(title and body), and an empty <ul id='notes'> the script "
                "fills in. Link style.css and app.js. No frameworks."
            ),
            output_path="index.html",
            output_kind=OutputKind.CODE,
            deps=("schema",),
            est_output_tokens=400,
        ),
        TaskNode(
            id="detail",
            title="Single note page",
            role=Role.FRONTEND,
            spec=(
                "Write note.html: reads ?id= from the query string and shows "
                "one note's title, body and created_at, with a link back to the "
                "list. Link style.css and app.js."
            ),
            output_path="note.html",
            output_kind=OutputKind.CODE,
            deps=("schema",),
            est_output_tokens=350,
        ),
        TaskNode(
            id="client",
            title="Browser API client",
            role=Role.FRONTEND,
            spec=(
                "Write app.js. On index.html: fetch GET /api/notes and render "
                "the list, and POST /api/notes on form submit. On note.html: "
                "fetch GET /api/notes/{id} and render it. Use the exact route "
                "paths from the interface contract."
            ),
            output_path="app.js",
            output_kind=OutputKind.CODE,
            deps=("server", "index", "detail"),
            needs=("server.summary",),
            est_output_tokens=700,
        ),
        TaskNode(
            id="style",
            title="Stylesheet",
            role=Role.STYLING,
            spec=(
                "Write style.css: a readable single-column layout, max-width "
                "around 40rem, sensible spacing, and a dark-mode variant via "
                "prefers-color-scheme."
            ),
            output_path="style.css",
            output_kind=OutputKind.CODE,
            deps=("index",),
            est_output_tokens=450,
        ),
    ]


# ==========================================================================
# Canned artifacts returned by the mock provider.
# Real, working implementations — the dry run must produce a runnable app.
# ==========================================================================

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes (created_at DESC);
"""

_SERVER = '''\
"""Notes API — standard library only."""

import json
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).parent
DB_PATH = HERE / "notes.db"
PORT = 8000

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/note.html": ("note.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    schema = HERE / "schema.sql"
    if not schema.is_file():
        raise SystemExit("schema.sql is missing")
    with connect() as conn:
        conn.executescript(schema.read_text(encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet by default

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, filename, content_type):
        path = HERE / filename
        if not path.is_file():
            self._json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path in STATIC:
            self._static(*STATIC[path])
            return

        if path == "/api/notes":
            with connect() as conn:
                rows = conn.execute(
                    "SELECT id, title, body, created_at FROM notes "
                    "ORDER BY created_at DESC"
                ).fetchall()
            self._json([dict(r) for r in rows])
            return

        if path.startswith("/api/notes/"):
            raw = path.rsplit("/", 1)[-1]
            if not raw.isdigit():
                self._json({"error": "invalid id"}, 400)
                return
            with connect() as conn:
                row = conn.execute(
                    "SELECT id, title, body, created_at FROM notes WHERE id = ?",
                    (int(raw),),
                ).fetchone()
            if row is None:
                self._json({"error": "not found"}, 404)
                return
            self._json(dict(row))
            return

        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/notes":
            self._json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON"}, 400)
            return

        title = str(payload.get("title", "")).strip()
        if not title:
            self._json({"error": "title is required"}, 400)
            return

        note = {
            "title": title,
            "body": str(payload.get("body", "")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with connect() as conn:
            cursor = conn.execute(
                "INSERT INTO notes (title, body, created_at) VALUES (?, ?, ?)",
                (note["title"], note["body"], note["created_at"]),
            )
            note["id"] = cursor.lastrowid
        self._json(note, 201)


if __name__ == "__main__":
    init_db()
    print(f"Notes app running at http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''

_INDEX = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Notes</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <h1>Notes</h1>

    <form id="new-note">
      <label>
        Title
        <input name="title" required maxlength="200" placeholder="Note title">
      </label>
      <label>
        Body
        <textarea name="body" rows="4" placeholder="Write something..."></textarea>
      </label>
      <button type="submit">Add note</button>
    </form>

    <ul id="notes"></ul>
    <p id="empty" hidden>No notes yet.</p>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""

_DETAIL = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Note</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <p><a href="index.html">&larr; All notes</a></p>
    <article id="note" hidden>
      <h1 id="note-title"></h1>
      <time id="note-created"></time>
      <p id="note-body"></p>
    </article>
    <p id="error" hidden>Note not found.</p>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""

_CLIENT = """\
// Browser client. Route paths come from the shared interface contract.

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

async function loadList() {
  const list = document.getElementById('notes');
  const empty = document.getElementById('empty');
  const response = await fetch('/api/notes');
  const notes = await response.json();

  list.textContent = '';
  empty.hidden = notes.length > 0;

  for (const note of notes) {
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = `note.html?id=${note.id}`;
    link.textContent = note.title;
    const when = document.createElement('time');
    when.textContent = formatDate(note.created_at);
    item.append(link, when);
    list.append(item);
  }
}

async function createNote(event) {
  event.preventDefault();
  const form = event.target;
  const payload = {
    title: form.title.value.trim(),
    body: form.body.value,
  };
  if (!payload.title) return;

  const response = await fetch('/api/notes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (response.ok) {
    form.reset();
    await loadList();
  }
}

async function loadDetail(id) {
  const response = await fetch(`/api/notes/${id}`);
  if (!response.ok) {
    document.getElementById('error').hidden = false;
    return;
  }
  const note = await response.json();
  document.getElementById('note-title').textContent = note.title;
  document.getElementById('note-body').textContent = note.body;
  document.getElementById('note-created').textContent = formatDate(note.created_at);
  document.getElementById('note').hidden = false;
}

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('new-note');
  if (form) {
    form.addEventListener('submit', createNote);
    loadList();
    return;
  }
  const id = new URLSearchParams(location.search).get('id');
  if (id) loadDetail(id);
});
"""

_STYLE = """\
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #6b7280;
  --line: #e5e7eb;
  --accent: #2563eb;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111317;
    --fg: #e8e8e8;
    --muted: #9aa1ab;
    --line: #2a2f37;
    --accent: #7aa2f7;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 2rem 1rem;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
}

main { max-width: 40rem; margin: 0 auto; }

h1 { font-size: 1.75rem; margin: 0 0 1.5rem; }

form {
  display: grid;
  gap: 0.75rem;
  padding: 1.25rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 2rem;
}

label { display: grid; gap: 0.35rem; font-size: 0.875rem; color: var(--muted); }

input, textarea {
  font: inherit;
  padding: 0.5rem 0.65rem;
  color: var(--fg);
  background: var(--bg);
  border: 1px solid var(--line);
  border-radius: 6px;
}

input:focus, textarea:focus { outline: 2px solid var(--accent); outline-offset: 1px; }

button {
  font: inherit;
  font-weight: 600;
  padding: 0.55rem 1rem;
  color: #fff;
  background: var(--accent);
  border: 0;
  border-radius: 6px;
  cursor: pointer;
  justify-self: start;
}

ul { list-style: none; margin: 0; padding: 0; }

li {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  padding: 0.85rem 0;
  border-bottom: 1px solid var(--line);
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

time { font-size: 0.8125rem; color: var(--muted); white-space: nowrap; }

article p { white-space: pre-wrap; }
"""

ARTIFACTS: dict[str, str] = {
    "schema": _SCHEMA,
    "server": _SERVER,
    "index": _INDEX,
    "detail": _DETAIL,
    "client": _CLIENT,
    "style": _STYLE,
}

SUMMARIES: dict[str, str] = {
    "schema": "notes(id, title, body, created_at) with an index on created_at DESC",
    "server": (
        "http.server on :8000. GET /api/notes, GET /api/notes/{id}, "
        "POST /api/notes; serves index.html, note.html, style.css, app.js"
    ),
    "index": "list page with a create form and <ul id='notes'>",
    "detail": "detail page reading ?id= and filling #note-title/#note-body",
    "client": "fetch-based client wired to the contract's route paths",
    "style": "single-column layout, max-width 40rem, dark mode via media query",
}
