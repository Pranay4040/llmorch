"""The setup page: one string, no assets, no dependencies.

Held the same way as the dashboard's, for the same reason — there is no state
where the server starts and the page is missing — and it follows the same rule
about markup: every value is written with `textContent` or set as an input
`value`, never as HTML. The data here comes from `models.yaml` rather than from
a model, but the rule is not conditional on today's data source.

Two things this page does that the dashboard deliberately does not:

**It writes.** So every request carries the token the server minted at launch,
taken from the URL the browser was opened with.

**It handles a secret.** A key field is `type="password"`, is never populated
from the server, and is cleared the moment it has been sent. What comes back is
only whether a key is now set — the page has no way to read one back out, which
is what makes a shoulder-surfer or a screenshot a non-event.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llmorch setup</title>
<style>
  /* One theme, on purpose. This is a control surface for a terminal tool and is
     read beside one; a light variant nobody asked for was a second set of
     colours to keep true of every control. */
  :root {
    --bg: #0f1115; --panel: #161a21; --panel-2: #1b2029; --raised: #14181f;
    --line: #242a34; --line-soft: #1e232c; --field: #2c3340;
    --text: #e6e9ef; --text-2: #c8cede; --muted: #98a0af; --muted-2: #767e8e;
    --faint: #6b7385;
    --ok: #4ade80; --warn: #fbbf24; --bad: #f87171; --accent: #3b82f6;

    /* System faces, not downloaded ones: this page is served under
       `default-src 'none'` with no font-src, so a webfont link would be blocked
       and fall back silently — and widening that CSP on the one page that
       writes API keys is not a trade worth making for typography. */
    --ui: "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system,
          "Helvetica Neue", sans-serif;
    --mono: "Cascadia Mono", "Cascadia Code", ui-monospace, SFMono-Regular,
            Consolas, "Liberation Mono", monospace;
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }

  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.65 var(--ui);
    -webkit-font-smoothing: antialiased;
  }
  /* Monospace now means "this is an identifier" — a model id, an environment
     variable, a path, a command — rather than being the whole page's voice. */
  code, input[type=password], select, input[type=number] { font-family: var(--mono); }

  /* -- header ------------------------------------------------------------ */
  header.top {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 32px; padding: 26px 32px 20px; border-bottom: 1px solid var(--line);
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  h1 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
  p.lede { margin: 4px 0 0 28px; color: var(--muted); font-size: 13.5px; }
  code {
    color: var(--text-2); background: var(--panel); border: 1px solid var(--line);
    border-radius: 4px; padding: 1px 6px; font-size: 12.5px;
  }
  .saver { display: flex; align-items: center; gap: 14px; padding-top: 4px; }
  .saver .status { color: var(--muted-2); font-size: 13px; white-space: nowrap; }
  .saver .status.dirty { color: var(--warn); }
  .saver .status.bad { color: var(--bad); }

  /* -- shell ------------------------------------------------------------- */
  .shell {
    display: grid; grid-template-columns: 232px minmax(0, 1fr); gap: 32px;
    padding: 24px 32px 56px; align-items: start;
  }
  @media (max-width: 860px) {
    header.top { flex-direction: column; gap: 16px; padding: 20px; }
    .shell { grid-template-columns: 1fr; gap: 20px; padding: 20px; }
    nav.tabs { position: static; }
  }

  /* -- rail -------------------------------------------------------------- */
  nav.tabs { display: flex; flex-direction: column; gap: 22px; position: sticky; top: 20px; }
  nav.tabs .group { display: flex; flex-direction: column; gap: 2px; }
  nav.tabs .grouplabel {
    color: var(--faint); font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.09em; text-transform: uppercase; padding: 0 10px 6px;
  }
  nav.tabs button {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    width: 100%; text-align: left; background: transparent; color: var(--muted);
    border: 0; border-radius: 6px; padding: 7px 10px; height: auto;
    font: inherit; font-weight: 400; cursor: pointer;
  }
  nav.tabs button:hover { color: var(--text-2); background: var(--raised); filter: none; }
  nav.tabs button[aria-selected="true"] {
    color: var(--text); font-weight: 500; background: var(--panel-2);
    box-shadow: inset 2px 0 0 var(--accent);
  }
  nav.tabs .count { font-family: var(--mono); font-size: 11.5px; color: var(--muted-2); }
  nav.tabs .count.ok { color: var(--ok); }
  nav.tabs .count.warn { color: var(--warn); }
  nav.tabs button[aria-selected="true"] .count { color: var(--accent); }
  nav.tabs .count.tag {
    font-family: var(--ui); font-size: 11px; color: var(--muted-2);
    border: 1px solid var(--field); border-radius: 999px; padding: 0 7px;
  }

  /* -- panels ------------------------------------------------------------ */
  section { min-width: 0; }
  h2 { margin: 0 0 6px; font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
  .hint {
    color: var(--muted); margin: 0 0 4px; font-size: 13.5px;
    max-width: 72ch; text-wrap: pretty;
  }
  .hint em { color: var(--text-2); font-style: normal; }
  .hint.warn { color: var(--warn); }
  .hint.bad { color: var(--bad); }
  .note {
    margin: 14px 0 0; padding: 11px 14px;
    border: 1px solid var(--line); border-left: 2px solid var(--accent);
    border-radius: 0 8px 8px 0; background: var(--raised);
    color: var(--muted); font-size: 12.5px; line-height: 1.6;
    max-width: 76ch; text-wrap: pretty;
  }
  .note.warn { border-left-color: var(--warn); color: var(--warn); }
  .card {
    margin-top: 18px; border: 1px solid var(--line); border-radius: 10px;
    background: var(--panel); overflow: hidden;
  }
  .subhead {
    color: var(--faint); font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.09em; text-transform: uppercase; margin: 22px 0 8px;
  }

  /* -- fields ------------------------------------------------------------ */
  input[type=password], input[type=number], select {
    height: 32px; background: var(--bg); color: var(--text);
    border: 1px solid var(--field); border-radius: 6px; padding: 0 10px;
    font-size: 12.5px; width: 100%;
  }
  select { padding-right: 6px; }
  input[type=number] { width: 84px; text-align: center; }
  input:focus-visible, select:focus-visible, button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
  }
  input[type=checkbox], input[type=radio] {
    accent-color: var(--accent); width: 15px; height: 15px;
  }

  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 6px;
    padding: 0 18px; height: 34px; font: inherit; font-weight: 500; cursor: pointer;
  }
  button:hover { filter: brightness(1.08); }
  button.ghost {
    background: transparent; color: var(--text); border: 1px solid var(--field);
    font-weight: 400;
  }
  button.ghost:hover { background: var(--raised); filter: none; }
  button:disabled {
    background: var(--panel); color: var(--faint); border: 1px solid var(--line);
    cursor: default; filter: none;
  }

  /* -- keys -------------------------------------------------------------- */
  .row {
    display: grid; grid-template-columns: 184px minmax(0, 1fr) 92px; gap: 16px;
    align-items: center; padding: 13px 18px; border-top: 1px solid var(--line-soft);
  }
  .row:first-of-type { border-top: 0; }
  .name { font-weight: 500; }
  .row .muted { font-family: var(--mono); color: var(--muted-2); font-size: 11.5px; }
  .muted { color: var(--muted-2); }
  .pill {
    display: flex; align-items: center; justify-content: flex-end; gap: 6px;
    font-size: 12px; white-space: nowrap; border: 0; padding: 0;
  }
  .pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
  .pill.set { color: var(--ok); }
  .pill.set::before { background: var(--ok); }
  .pill.unset { color: var(--warn); }
  .pill.unset::before { border: 1px solid var(--warn); }

  /* -- models ------------------------------------------------------------ */
  .vendor {
    margin-top: 14px; border: 1px solid var(--line); border-radius: 10px;
    background: var(--panel); overflow: hidden;
  }
  .vendor:first-child { margin-top: 18px; }
  .vendor > h3 {
    margin: 0; padding: 10px 18px; font-size: 13px; font-weight: 600;
    color: var(--text); background: var(--raised);
    border-bottom: 1px solid var(--line);
  }
  .models { display: flex; flex-direction: column; }
  .models label.check {
    display: grid; grid-template-columns: 18px minmax(0, 1fr) auto; gap: 12px;
    align-items: center; padding: 8px 18px; border-top: 1px solid var(--line-soft);
    font-family: var(--mono); font-size: 13px;
  }
  .models label.check:first-child { border-top: 0; }
  .models label.check:hover { background: var(--raised); }
  .models label.check .muted { text-align: right; font-size: 11.5px; }

  /* -- roles ------------------------------------------------------------- */
  .role {
    display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 24px;
    align-items: start; padding: 14px 18px; border-top: 1px solid var(--line-soft);
  }
  .role:first-of-type { border-top: 0; }
  .role .blurb {
    color: var(--muted-2); font-size: 12.5px; line-height: 1.55; margin-top: 2px;
    text-wrap: pretty;
  }
  .role select { max-width: 340px; }

  /* -- mode picker ------------------------------------------------------- */
  .pick { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0 0; }
  @media (max-width: 720px) { .pick { grid-template-columns: 1fr; } }
  .pick label {
    display: flex; gap: 11px; align-items: flex-start; padding: 13px 15px;
    border: 1px solid var(--line); border-radius: 9px; background: var(--raised);
    color: var(--text-2); cursor: pointer;
  }
  .pick label:hover { border-color: var(--field); }
  .pick label:has(input:checked) {
    border-color: var(--accent); background: var(--panel); color: var(--text);
  }
  .pick input { margin-top: 4px; flex: none; }

  /* -- run options ------------------------------------------------------- */
  .opts { display: flex; flex-wrap: wrap; gap: 14px 28px; align-items: center; }
  label.check { display: flex; gap: 9px; align-items: center; white-space: nowrap; }
  .warn { color: var(--warn); }
  .bad { color: var(--bad); }
</style>
</head>
<body>
<header class="top">
  <div>
    <div class="brand">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6"
           stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="5" r="2.4"></circle>
        <circle cx="5" cy="18" r="2.4"></circle>
        <circle cx="19" cy="18" r="2.4"></circle>
        <path d="M12 7.4v3.2M10.6 12.4 6.6 15.8M13.4 12.4l4 3.4"></path>
      </svg>
      <h1>llmorch setup</h1>
    </div>
    <p class="lede">Choose once. Then run <code>llmorch start</code> and talk to it.</p>
  </div>
  <!-- The save control sits where the thing being saved is, rather than
       floating at the bottom of the viewport detached from it. -->
  <div class="saver">
    <span class="status" id="status">loading…</span>
    <button id="save">Save</button>
  </div>
</header>

<div class="shell">
  <nav class="tabs" id="tabs" role="tablist"></nav>

  <main>

  <section data-tab="keys">
    <h2>Keys</h2>
    <p class="hint">
      Written to <span id="envpath" class="muted"></span>, which is gitignored.
      A key is never sent back to this page — you can set one, not read one.
      Leave a box empty to keep what is already there.
    </p>
    <div id="providers" class="card"></div>
    <div style="margin-top:14px"><button id="savekeys" class="ghost">Save keys</button></div>
  </section>

  <section data-tab="models">
    <h2>Models</h2>
    <p class="hint">
      Which models this account may use. Unticking every model of a vendor
      switches that vendor off. A role with nothing left ticked cannot be built.
    </p>
    <div id="models"></div>
    <p id="unstaffed" class="hint warn"></p>
  </section>

  <section data-tab="roles">
    <h2>Who does what</h2>
    <p class="hint">
      One model per job, when you want to choose. <em>Automatic</em> lets the
      assignment pick by fitness, remaining quota and an even split — which is
      usually better, and is what every unpinned job does.
    </p>
    <p class="note">
      A pin binds the <em>assignment</em>, not failover. If the model you chose
      breaks mid-run its work still moves to another vendor, because a model
      that has tripped its circuit breaker is not the one you meant to insist on.
    </p>
    <div id="roles" class="card"></div>
    <p id="rolewarn" class="hint warn"></p>
  </section>

  <section data-tab="chat">
    <h2>Chat</h2>
    <p class="hint">Questions only. Nothing is built, and no file is written.</p>
    <div id="pick-chat" class="pick"></div>
    <div class="role">
      <div>
        <div class="name">Who answers</div>
        <div class="blurb">Also used for any question typed inside a build
          session. Same control as the <em>Who does what</em> tab.</div>
      </div>
      <select data-role="research" data-mirror="1"></select>
    </div>
    <div class="opts" style="margin-top:12px">
      <label class="check">
        <input type="checkbox" id="answer_reads_files">
        answers may quote a file the question names
      </label>
    </div>
    <p class="hint">
      On, an answer is grounded in the file rather than inferred from a one-line
      summary. Off, nothing you built is ever sent to a provider — this is the
      only path by which it would be.
    </p>
  </section>

  <section data-tab="agent">
    <h2>AI agent</h2>
    <p class="hint">One model plans the work and writes every file in it.</p>
    <div id="pick-agent" class="pick"></div>
    <div class="role">
      <div>
        <div class="name">The agent</div>
        <div class="blurb">Empty picks the best planner available, which is the
          same choice made for the one request every run depends on.</div>
      </div>
      <select id="agent_model"></select>
    </div>
    <p class="note warn">
      Cross-vendor review cannot run in this mode. A reviewer must come from a
      different vendor than the author — a model tends to re-approve its own
      mistake — and with one model there is no second vendor to ask. The eight
      cross-artifact checks still run; they cost nothing and read the files.
    </p>
  </section>

  <section data-tab="crew">
    <h2>Crew</h2>
    <p class="hint">
      Several models from different vendors split the work, review each other,
      and are held to one shared contract.
    </p>
    <div id="pick-crew" class="pick"></div>
    <div class="opts" style="margin-top:12px">
      <label class="check">divide work by
        <select id="assignment">
          <option value="fitness">best fit and fair share</option>
          <option value="round_robin">round robin</option>
        </select>
      </label>
      <label class="check">review
        <select id="review">
          <option value="off">off — no cross-vendor review</option>
          <option value="code">code — review the files that are code</option>
          <option value="all">all — review everything written</option>
        </select>
      </label>
      <label class="check">max nodes <input type="number" id="max_nodes" min="1" max="25"></label>
      <label class="check">at once <input type="number" id="concurrency" min="1" max="16"></label>
    </div>
    <p class="hint" id="assignmenthint"></p>
    <p class="hint">
      Who gets which job is on the <em>Who does what</em> tab. A model pinned
      there is used whichever way the work is divided.
    </p>
  </section>

  <section data-tab="runs">
    <h2>How runs behave</h2>
    <p class="hint">True of every mode.</p>
    <div class="opts">
      <label class="check"><input type="checkbox" id="live"> call real providers</label>
      <label class="check"><input type="checkbox" id="smoke"> start what it builds</label>
      <label class="check"><input type="checkbox" id="smoke_install"> ...installing deps first</label>
    </div>
    <p class="hint" id="livehint"></p>
  </section>
  </main>
</div>

<script>
const token = new URLSearchParams(location.search).get("t") || "";
let config = null;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;   // never innerHTML
  return node;
}

async function call(path, options) {
  const opts = options || {};
  opts.headers = Object.assign({}, opts.headers, {"X-Llmorch-Token": token});
  const response = await fetch(path, opts);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function status(text, kind) {
  const node = document.getElementById("status");
  node.textContent = text;
  node.className = "status" + (kind ? " " + kind : "");
}

// Whether anything on this page differs from what is saved. Tracked so the
// Save button can say which of its two states it is in: a button that looks
// identical whether or not there is anything to save tells you nothing, and
// the answer is the one thing you want before closing the tab.
let dirty = false;

function markClean(text) {
  dirty = false;
  document.getElementById("save").disabled = true;
  status(text);
}

function markDirty() {
  if (dirty) return;
  dirty = true;
  document.getElementById("save").disabled = false;
  status("unsaved changes", "dirty");
}

function drawProviders() {
  const host = document.getElementById("providers");
  host.replaceChildren();
  for (const provider of config.providers) {
    const row = el("div", "row");
    const left = el("div");
    left.append(el("div", "name", provider.name));
    left.append(el("div", "muted", provider.key_env));
    row.append(left);

    const field = el("input");
    field.type = "password";
    field.placeholder = provider.key_set ? "set — type to replace" : "paste key here";
    field.autocomplete = "off";
    field.dataset.env = provider.key_env;
    row.append(field);

    row.append(el("span", "pill " + (provider.key_set ? "set" : "unset"),
                 provider.key_set ? "key set" : "no key"));
    host.append(row);
  }
  document.getElementById("envpath").textContent = config.env_path;
}

function drawModels() {
  const host = document.getElementById("models");
  host.replaceChildren();
  const chosen = new Set(config.settings.models);
  const all = chosen.size === 0;

  for (const provider of config.providers) {
    const models = config.models.filter(m => m.provider === provider.name);
    if (!models.length) continue;
    const group = el("div", "vendor");
    group.append(el("h3", null, provider.name + (provider.key_set ? "" : "  (no key)")));
    const grid = el("div", "models");
    for (const model of models) {
      const label = el("label", "check");
      const box = el("input");
      box.type = "checkbox";
      box.value = model.id;
      box.checked = all || chosen.has(model.id);
      box.dataset.model = "1";
      box.addEventListener("change", () => { drawUnstaffed(); drawRoles(); drawTabs(); });
      label.append(box);
      label.append(el("span", null, model.id));
      label.append(el("span", "muted", "  " + model.context.toLocaleString() + " ctx"));
      grid.append(label);
    }
    group.append(grid);
    host.append(group);
  }
  drawUnstaffed();
}

function chosenModels() {
  return [...document.querySelectorAll("[data-model]")]
    .filter(box => box.checked).map(box => box.value);
}

function drawUnstaffed() {
  const chosen = new Set(chosenModels());
  const missing = config.roles
    .filter(role => !role.models.some(id => chosen.has(id)))
    .map(role => role.name);
  const node = document.getElementById("unstaffed");
  node.textContent = missing.length
    ? "Nothing left to serve: " + missing.join(", ") + ". A node with that role cannot run."
    : "";
}

function drawSettings() {
  const s = config.settings;
  document.getElementById("live").checked = s.live;
  document.getElementById("review").value = s.review;
  document.getElementById("assignment").value = s.assignment || "fitness";
  document.getElementById("assignmenthint").textContent =
    (s.assignment === "round_robin")
      ? "Rotation. Ignores affinity and track record, and spreads per-minute "
        + "token pressure across more vendors — which is the pressure that "
        + "actually stalls a run against a tight ceiling."
      : "Weighs affinity, track record and remaining quota, then caps how much "
        + "of the work any one model may take.";
  document.getElementById("smoke").checked = s.smoke;
  document.getElementById("smoke_install").checked = s.smoke_install;
  document.getElementById("max_nodes").value = s.max_nodes;
  document.getElementById("concurrency").value = s.concurrency;
  document.getElementById("livehint").textContent = s.live
    ? "Live: every turn spends real quota against the keys above."
    : "Mock: no network, no quota — and the same canned answer whatever you type.";
}

document.getElementById("assignment").addEventListener("change", () => {
  config.settings.assignment = document.getElementById("assignment").value;
  drawSettings();
});

document.getElementById("live").addEventListener("change", () => {
  config.settings.live = document.getElementById("live").checked;
  drawSettings();
  drawTabs();
});

// Seven flat tabs, grouped. The groups are the three questions the page
// actually asks: what it may call, how a session opens, and what is true of
// every run.
const TABS = [
  {id: "keys", label: "Keys", group: "Setup"},
  {id: "models", label: "Models", group: "Setup"},
  {id: "roles", label: "Who does what", group: "Setup"},
  {id: "chat", label: "Chat", mode: "chat", group: "Session mode"},
  {id: "agent", label: "AI agent", mode: "agent", group: "Session mode"},
  {id: "crew", label: "Crew", mode: "crew", group: "Session mode"},
  {id: "runs", label: "Runs", group: "Every run"},
];
// Prefixed, because a bare "#roles" also matches the <div id="roles"> inside
// the tab and the browser scrolls to it — landing past the header on a page
// whose header is how you get anywhere else.
let active = location.hash.replace("#tab-", "") || "keys";

function showTab(id) {
  active = id;
  history.replaceState(null, "", "#tab-" + id);
  for (const section of document.querySelectorAll("section[data-tab]")) {
    section.hidden = section.dataset.tab !== id;
  }
  for (const button of document.querySelectorAll("nav.tabs button")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === id));
  }
}

function tabState(tab) {
  // What you would otherwise have to open the tab to find out. Returned as
  // [text, class] so the rail can carry it beside the name.
  if (!config) return null;
  if (tab.id === "keys") {
    const set = config.providers.filter(p => p.key_set).length;
    return [set + "/" + config.providers.length, set ? "ok" : "warn"];
  }
  if (tab.id === "models") {
    const picked = chosenModels().length;
    return [String(picked), picked ? "" : "warn"];
  }
  if (tab.id === "roles") {
    const pinned = config.roles.filter(r => r.pinned).length;
    return pinned ? [pinned + " pinned", ""] : null;
  }
  if (tab.id === "runs") {
    return config.settings.live ? ["live", "ok"] : ["mock", "warn"];
  }
  // A tick on the mode sessions will actually open in, so the choice is visible
  // from every tab rather than only from the one holding it.
  if (tab.mode && config.settings.mode === tab.mode) return ["in use", "tag"];
  return null;
}

function drawTabs() {
  const nav = document.getElementById("tabs");
  nav.replaceChildren();

  let group = null;
  let host = null;
  for (const tab of TABS) {
    if (tab.group !== group) {
      group = tab.group;
      host = el("div", "group");
      host.append(el("div", "grouplabel", group));
      nav.append(host);
    }
    const button = el("button", null, tab.label);
    button.dataset.tab = tab.id;
    button.setAttribute("role", "tab");
    const state = tabState(tab);
    if (state) button.append(el("span", ("count " + state[1]).trim(), state[0]));
    button.addEventListener("click", () => showTab(tab.id));
    host.append(button);
  }
  showTab(active);
}

// Selecting a mode is its own control rather than a side effect of opening the
// tab: looking at what a mode would do must not change which one you get.
const MODES = ["chat", "agent", "crew"];

function drawModePickers() {
  for (const mode of MODES) {
    const host = document.getElementById("pick-" + mode);
    if (!host) continue;
    host.replaceChildren();
    for (const [value, text] of [
      [mode, "Always use this mode"],
      ["", "Ask me at the start of each session"],
    ]) {
      const label = el("label");
      const radio = el("input");
      radio.type = "radio";
      radio.name = "mode-" + mode;
      radio.checked = (config.settings.mode || "") === value;
      radio.addEventListener("change", () => {
        config.settings.mode = value;
        drawModePickers();
        drawTabs();
      });
      label.append(radio, el("span", null, text));
      host.append(label);
    }
  }
}

function drawModeSettings() {
  const s = config.settings;
  document.getElementById("answer_reads_files").checked = s.answer_reads_files;

  const agent = document.getElementById("agent_model");
  agent.replaceChildren();
  agent.append(new Option("Automatic — the best planner available", ""));
  for (const id of chosenModels()) agent.append(new Option(id, id));
  agent.value = chosenModels().includes(s.agent_model) ? s.agent_model : "";
}

function fillRolePicker(picker, role, available) {
  picker.replaceChildren();
  picker.dataset.role = role.name;
  picker.append(new Option("Automatic — best fit, fair share", ""));

  // The models the manifest lists for this job first, then everything else:
  // a pin is a person overruling that preference, so it must not also be the
  // limit of what they can pick.
  const suggested = role.models.filter(id => available.has(id));
  const rest = config.models.map(m => m.id)
    .filter(id => available.has(id) && !suggested.includes(id));

  for (const [group, ids] of [["suited to this job", suggested], ["others", rest]]) {
    if (!ids.length) continue;
    const box = document.createElement("optgroup");
    box.label = group;
    for (const id of ids) box.append(new Option(id, id));
    picker.append(box);
  }
  picker.value = available.has(role.pinned) ? role.pinned : "";
  picker.onchange = () => { drawRoleWarning(); mirrorRole(role.name); };
}

function drawRoles() {
  const host = document.getElementById("roles");
  host.replaceChildren();
  const available = new Set(chosenModels());

  for (const role of config.roles) {
    const row = el("div", "role");
    const left = el("div");
    left.append(el("div", "name", role.label));
    left.append(el("div", "blurb", role.blurb));
    row.append(left);

    const picker = el("select");
    fillRolePicker(picker, role, available);
    row.append(picker);
    host.append(row);

    // The same job can be shown on the tab for the mode it belongs to. That
    // copy is filled from the same data rather than kept as a second source of
    // truth, so the two cannot drift.
    for (const mirror of document.querySelectorAll(
      `select[data-mirror][data-role="${role.name}"]`
    )) {
      fillRolePicker(mirror, role, available);
    }
  }
  drawRoleWarning();
}

function chosenRoles() {
  const out = {};
  // A role may be shown twice — once on "Who does what" and once on the tab for
  // the mode it belongs to. Last non-empty wins; they are kept in step by
  // `mirrorRole`, so they never actually disagree.
  for (const picker of document.querySelectorAll("[data-role]")) {
    if (picker.value) out[picker.dataset.role] = picker.value;
  }
  return out;
}

function mirrorRole(role) {
  const pickers = [...document.querySelectorAll(`[data-role="${role}"]`)];
  const source = pickers.find(p => p.value) || pickers[0];
  for (const picker of pickers) picker.value = source ? source.value : "";
}

function drawRoleWarning() {
  const pins = chosenRoles();
  const notes = [];

  // Every author is a candidate reviewer's author. If the only thing left that
  // could review is the vendor doing the writing, review is skipped per file
  // rather than done badly — worth saying before the run rather than after.
  const vendorOf = (id) => (config.models.find(m => m.id === id) || {}).provider;
  if (pins.review) {
    const writers = Object.entries(pins)
      .filter(([role]) => role !== "review")
      .map(([, id]) => vendorOf(id));
    if (writers.length && writers.every(v => v === vendorOf(pins.review))) {
      notes.push("every pinned writer shares the reviewer's vendor, so review "
                 + "will be skipped for those files — a reviewer never shares "
                 + "the author's vendor.");
    }
  }
  document.getElementById("rolewarn").textContent = notes.join(" ");
}

for (const kind of ["change", "input"]) {
  document.querySelector(".shell").addEventListener(kind, (event) => {
    if (event.target.type !== "password") markDirty();
  });
}

async function load() {
  try {
    config = await call("/api/config");
    drawProviders();
    drawModels();
    drawRoles();
    drawModePickers();
    drawModeSettings();
    drawSettings();
    drawTabs();
    markClean(config.settings.configured ? "no changes" : "not configured yet");
  } catch (err) {
    status("could not load: " + err.message, "bad");
  }
}

document.getElementById("savekeys").addEventListener("click", async () => {
  const fields = [...document.querySelectorAll("input[type=password]")];
  const keys = {};
  for (const field of fields) {
    if (field.value) keys[field.dataset.env] = field.value;
  }
  if (!Object.keys(keys).length) { status("no key typed"); return; }
  status("saving keys…");
  try {
    const result = await call("/api/keys", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(keys),
    });
    for (const field of fields) field.value = "";   // never linger in the DOM
    status(result.saved.length + " key(s) written");
    await load();
  } catch (err) {
    status("keys not saved: " + err.message, "bad");
  }
});

document.getElementById("save").addEventListener("click", async () => {
  const every = config.models.length;
  const picked = chosenModels();
  const payload = {
    live: document.getElementById("live").checked,
    mode: config.settings.mode || "",
    agent_model: document.getElementById("agent_model").value,
    answer_reads_files: document.getElementById("answer_reads_files").checked,
    assignment: document.getElementById("assignment").value,
    review: document.getElementById("review").value,
    smoke: document.getElementById("smoke").checked,
    smoke_install: document.getElementById("smoke_install").checked,
    max_nodes: Number(document.getElementById("max_nodes").value),
    concurrency: Number(document.getElementById("concurrency").value),
    // Every model ticked means "no opinion", which keeps working when a model
    // is added to the manifest later.
    models: picked.length === every ? [] : picked,
    role_models: chosenRoles(),
    providers: [],
  };
  status("saving…");
  try {
    await call("/api/settings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    markClean("saved — now run  llmorch start  in your terminal");
  } catch (err) {
    status("not saved: " + err.message, "bad");
  }
});

load();
</script>
</body>
</html>
"""
