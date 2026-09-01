"""The dashboard page: one string, no assets, no dependencies.

Held as a Python string rather than a file so the dashboard cannot half-exist —
there is no state where the server starts and the page is missing, and nothing
to package or locate at runtime.

The page never receives markup. It fetches JSON and writes every value with
`textContent`, because much of that data was written by a language model: error
text, node ids, task descriptions, model names. Handing any of it to
`innerHTML` would mean a provider's error string could execute in the browser
of whoever is watching the build.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>llmorch</title>
<style>
  :root {
    --bg: #0f1115; --panel: #161a21; --line: #242a34; --text: #e6e9ef;
    --muted: #8b93a3; --ok: #4ade80; --warn: #fbbf24; --bad: #f87171;
    --bar: #3b82f6;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f7f8fa; --panel: #ffffff; --line: #e3e6ec; --text: #12151b;
      --muted: #667085; --ok: #15803d; --warn: #b45309; --bad: #b91c1c;
      --bar: #2563eb;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  header {
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
    padding: 18px 24px; border-bottom: 1px solid var(--line);
  }
  h1 { font-size: 16px; margin: 0; letter-spacing: 0.08em; text-transform: uppercase; }
  .muted { color: var(--muted); }
  main { display: grid; gap: 18px; padding: 18px 24px 48px;
         grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
  section h2 {
    font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase;
    margin: 0; padding: 12px 16px; border-bottom: 1px solid var(--line);
    color: var(--muted);
  }
  .body { padding: 8px 16px 16px; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-weight: 500; color: var(--muted);
       padding: 6px 8px 6px 0; font-size: 12px; }
  td { padding: 5px 8px 5px 0; border-top: 1px solid var(--line);
       white-space: nowrap; }
  td.wrap { white-space: normal; color: var(--muted); font-size: 12px; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .ok { color: var(--ok); } .warn { color: var(--warn); } .bad { color: var(--bad); }
  .track { background: var(--line); border-radius: 3px; height: 6px;
           width: 110px; overflow: hidden; display: inline-block;
           vertical-align: middle; }
  .fill { background: var(--bar); height: 100%; display: block; }
  .fill.warn { background: var(--warn); } .fill.bad { background: var(--bad); }
  .tag { color: var(--muted); font-size: 11px; border: 1px solid var(--line);
         border-radius: 3px; padding: 0 5px; }
  footer { padding: 0 24px 32px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>llmorch</h1>
  <span class="muted" id="generated">loading…</span>
  <span class="tag">read-only</span>
  <span class="tag">loopback only</span>
</header>

<main>
  <section><h2>Quota — today</h2><div class="body"><table id="quota"></table></div></section>
  <section><h2>Runs</h2><div class="body"><table id="runs"></table></div></section>
  <section><h2>Spend by day</h2><div class="body"><table id="spend"></table></div></section>
  <section><h2>Track record</h2><div class="body"><table id="track"></table></div></section>
  <section style="grid-column: 1 / -1">
    <h2>Recent calls</h2><div class="body"><table id="recent"></table></div>
  </section>
</main>

<footer id="paths"></footer>

<script>
// Every value below arrives from the ledger, and much of it was written by a
// model. It is placed with textContent throughout — never innerHTML — so a
// provider's error string cannot execute here.
const $ = (id) => document.getElementById(id);

function table(el, columns, rows, cell) {
  el.replaceChildren();
  const head = el.insertRow();
  for (const name of columns) {
    const th = document.createElement("th");
    th.textContent = name;
    head.appendChild(th);
  }
  if (!rows.length) {
    const td = el.insertRow().insertCell();
    td.colSpan = columns.length;
    td.className = "wrap";
    td.textContent = "nothing recorded yet";
    return;
  }
  for (const row of rows) cell(el.insertRow(), row);
}

function put(tr, text, cls) {
  const td = tr.insertCell();
  td.textContent = text === null || text === undefined ? "—" : String(text);
  if (cls) td.className = cls;
  return td;
}

function bar(tr, fraction) {
  const td = tr.insertCell();
  const track = document.createElement("span");
  track.className = "track";
  const fill = document.createElement("span");
  fill.className = "fill" + (fraction > 0.9 ? " bad" : fraction > 0.6 ? " warn" : "");
  fill.style.width = Math.min(100, Math.round(fraction * 100)) + "%";
  track.appendChild(fill);
  td.appendChild(track);
}

function duration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60);
  return h + "h" + String(m).padStart(2, "0") + "m";
}

function render(state) {
  $("generated").textContent = "updated " + state.generated_utc.slice(11, 19) + " UTC";

  table($("quota"), ["model", "requests", "", "per-minute tokens", "resets in"],
    state.quota, (tr, q) => {
      put(tr, q.model_id);
      put(tr, q.requests_limit ? q.requests_used + "/" + q.requests_limit : "—", "num");
      bar(tr, q.fraction);
      put(tr, q.tokens_limit_minute
            ? q.tokens_minute + "/" + q.tokens_limit_minute : "—", "num");
      put(tr, duration(q.seconds_to_reset) + (q.estimated ? " (est)" : ""));
    });

  table($("runs"), ["run", "done", "left", "state", "task"],
    state.runs, (tr, r) => {
      put(tr, r.run_id);
      put(tr, r.done, "num");
      put(tr, r.left, "num");
      put(tr, r.complete ? "complete" : (r.blocked_until ? "blocked" : "resumable"),
          r.complete ? "ok" : "warn");
      put(tr, r.task || "—", "wrap");
    });

  table($("spend"), ["day", "model", "calls", "failed", "tokens"],
    state.spend.days, (tr, d) => {
      put(tr, d.day);
      put(tr, d.model_id);
      put(tr, d.requests, "num");
      put(tr, d.failures, d.failures ? "num bad" : "num");
      put(tr, d.tokens.toLocaleString(), "num");
    });

  table($("track"), ["model", "role", "score", "tries", "rejected"],
    state.track_record, (tr, t) => {
      put(tr, t.model_id);
      put(tr, t.role);
      put(tr, t.score.toFixed(2), t.score >= 0.6 ? "num ok" : t.score < 0.4 ? "num bad" : "num");
      put(tr, t.attempts, "num");
      put(tr, t.rejections, "num");
    });

  table($("recent"), ["when", "model", "purpose", "node", "tokens", "ms", "result"],
    state.recent, (tr, c) => {
      put(tr, c.ts_utc.slice(11, 19));
      put(tr, c.model_id);
      put(tr, c.purpose);
      put(tr, c.node_id);
      put(tr, c.tokens, "num");
      put(tr, c.latency_ms, "num");
      put(tr, c.ok ? "ok" : (c.error || ("HTTP " + c.status)), c.ok ? "ok" : "bad wrap");
    });

  const paths = $("paths");
  paths.replaceChildren();
  for (const [name, value] of Object.entries(state.paths)) {
    const line = document.createElement("div");
    line.textContent = name + ": " + value;
    paths.appendChild(line);
  }
}

async function poll() {
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    const state = await response.json();
    if (state.error) {
      $("generated").textContent = "error: " + state.error;
      return;
    }
    render(state);
  } catch (err) {
    $("generated").textContent = "not reachable — is llmorch dashboard still running?";
  }
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""
