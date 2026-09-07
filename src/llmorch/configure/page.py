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
  :root {
    --bg: #0f1115; --panel: #161a21; --line: #242a34; --text: #e6e9ef;
    --muted: #8b93a3; --ok: #4ade80; --warn: #fbbf24; --bad: #f87171;
    --accent: #3b82f6;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f7f8fa; --panel: #ffffff; --line: #e3e6ec; --text: #12151b;
      --muted: #667085; --ok: #15803d; --warn: #b45309; --bad: #b91c1c;
      --accent: #2563eb;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  main { max-width: 900px; margin: 0 auto; padding: 24px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: .04em; }
  h2 { font-size: 14px; margin: 0 0 4px; letter-spacing: .06em; text-transform: uppercase; }
  p.lede { color: var(--muted); margin: 0 0 24px; }
  section {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 16px 18px; margin-bottom: 18px;
  }
  .hint { color: var(--muted); margin: 0 0 14px; font-size: 13px; }
  .row {
    display: grid; grid-template-columns: 150px 1fr auto; gap: 12px;
    align-items: center; padding: 8px 0; border-top: 1px solid var(--line);
  }
  .row:first-of-type { border-top: 0; }
  .name { font-weight: 600; }
  .muted { color: var(--muted); }
  .pill { font-size: 12px; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--line); }
  .pill.set { color: var(--ok); border-color: var(--ok); }
  .pill.unset { color: var(--warn); border-color: var(--warn); }
  input[type=password], input[type=number], select {
    background: var(--bg); color: var(--text); border: 1px solid var(--line);
    border-radius: 6px; padding: 6px 8px; font: inherit; width: 100%;
  }
  input[type=number] { width: 90px; }
  label.check { display: flex; gap: 8px; align-items: baseline; padding: 4px 0; }
  .models { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 4px 18px; }
  .vendor { margin-top: 12px; }
  .vendor:first-child { margin-top: 0; }
  .vendor > h3 { font-size: 13px; margin: 0 0 4px; color: var(--muted); }
  .opts { display: flex; flex-wrap: wrap; gap: 18px 32px; align-items: center; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 6px;
    padding: 9px 18px; font: inherit; cursor: pointer;
  }
  button.ghost { background: transparent; color: var(--text); border: 1px solid var(--line); }
  button:disabled { opacity: .5; cursor: default; }
  .bar {
    position: fixed; left: 0; right: 0; bottom: 0; background: var(--panel);
    border-top: 1px solid var(--line); padding: 12px 20px;
    display: flex; gap: 14px; align-items: center; justify-content: center;
  }
  .bar .status { color: var(--muted); }
  code {
    background: var(--bg); border: 1px solid var(--line); border-radius: 4px;
    padding: 1px 6px;
  }
  .warn { color: var(--warn); }
  .bad { color: var(--bad); }
</style>
</head>
<body>
<main>
  <h1>llmorch setup</h1>
  <p class="lede">Choose once. Then run <code>llmorch start</code> and talk to it.</p>

  <section>
    <h2>Keys</h2>
    <p class="hint">
      Written to <span id="envpath" class="muted"></span>, which is gitignored.
      A key is never sent back to this page — you can set one, not read one.
      Leave a box empty to keep what is already there.
    </p>
    <div id="providers"></div>
    <div style="margin-top:14px"><button id="savekeys" class="ghost">Save keys</button></div>
  </section>

  <section>
    <h2>Models</h2>
    <p class="hint">
      Which models this account may use. Unticking every model of a vendor
      switches that vendor off. A role with nothing left ticked cannot be built.
    </p>
    <div id="models"></div>
    <p id="unstaffed" class="hint warn"></p>
  </section>

  <section>
    <h2>How runs behave</h2>
    <div class="opts">
      <label class="check"><input type="checkbox" id="live"> call real providers</label>
      <label class="check">review
        <select id="review">
          <option value="off">off</option>
          <option value="code">code</option>
          <option value="all">all</option>
        </select>
      </label>
      <label class="check"><input type="checkbox" id="smoke"> start what it builds</label>
      <label class="check"><input type="checkbox" id="smoke_install"> ...installing deps first</label>
      <label class="check">max nodes <input type="number" id="max_nodes" min="1" max="25"></label>
      <label class="check">concurrency <input type="number" id="concurrency" min="1" max="16"></label>
    </div>
    <p class="hint" id="livehint"></p>
  </section>
</main>

<div class="bar">
  <button id="save">Save</button>
  <span class="status" id="status">loading…</span>
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
      box.addEventListener("change", drawUnstaffed);
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
  document.getElementById("smoke").checked = s.smoke;
  document.getElementById("smoke_install").checked = s.smoke_install;
  document.getElementById("max_nodes").value = s.max_nodes;
  document.getElementById("concurrency").value = s.concurrency;
  document.getElementById("livehint").textContent = s.live
    ? "Live: every turn spends real quota against the keys above."
    : "Mock: no network, no quota — and the same canned answer whatever you type.";
}

document.getElementById("live").addEventListener("change", () => {
  config.settings.live = document.getElementById("live").checked;
  drawSettings();
});

async function load() {
  try {
    config = await call("/api/config");
    drawProviders();
    drawModels();
    drawSettings();
    status(config.settings.configured ? "loaded" : "not configured yet");
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
    review: document.getElementById("review").value,
    smoke: document.getElementById("smoke").checked,
    smoke_install: document.getElementById("smoke_install").checked,
    max_nodes: Number(document.getElementById("max_nodes").value),
    concurrency: Number(document.getElementById("concurrency").value),
    // Every model ticked means "no opinion", which keeps working when a model
    // is added to the manifest later.
    models: picked.length === every ? [] : picked,
    providers: [],
  };
  status("saving…");
  try {
    await call("/api/settings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    status("saved — now run  llmorch start  in your terminal");
  } catch (err) {
    status("not saved: " + err.message, "bad");
  }
});

load();
</script>
</body>
</html>
"""
