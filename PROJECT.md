# llmorch — Project Blueprint & Status

> Self-contained handoff. Everything needed to resume with no prior context.
> Last updated: 2026-08-24, end of Milestone 3.

---

## 1. What this is

A **multi-provider LLM orchestrator**. It takes a task, splits it into a
dependency graph, assigns each slice to whichever model is best suited to it,
executes them across vendors under strict quota control, and writes a real
runnable project folder to disk.

Original goal: *have AI models from different companies collaborate, split work
by their strengths, and conserve scarce API quota.*

---

## 2. Current status — READ THIS FIRST

**M0–M3 complete. 301 tests pass, 1 skipped** (symlink test needs admin on
Windows).

`llmorch run --live "build a notes app"` completes **6 of 6 nodes, 0 degraded**,
across both vendors at ~74% quota efficiency. Verified twice, and the generated
app was started and exercised both times: POST creates a note, GET lists,
detail fetches, `/` and `/style.css` serve.

**`server.py` was written by Gemini; the schema, both pages, the client script
and the stylesheet by Groq models.** They interoperate correctly through the
interface contract without ever exchanging a message. That is the thing this
project exists to demonstrate, and it works.

### The honest caveat

**The task string is ignored.** `llmorch run --live "build a chess engine"`
produces the notes-app DAG. `build_nodes()` returns a hand-written six-node
graph regardless of input; the task text is echoed in the header and used
nowhere else. Decomposition is `negotiate/decompose.py`, **M4, not built**.

So what works end to end is: *one fixed, hand-authored task, executed across
real models from two vendors, under real quota control, producing a real
working app.* That is a substantial result. It is not yet "give it any task."

Also inert until M4: the bidding round. The scoring formula's `z_conf` and
`track_record` terms are always zero, so assignment currently runs on static
priors and quota pressure alone.

---

## 3. The roster — verified live 2026-08-24

Read from real `x-ratelimit-*` response headers, not from documentation.
**Every wire name originally in this file was wrong.** Five of five: two Groq
models had been retired, one renamed, Gemini's refused on a new key. A wrong
wire name fails exactly like a dead model — mid-run, after quota has been spent
on its neighbours.

| model id | wire name | RPD | TPM | notes |
|---|---|---|---|---|
| `groq/gpt-oss-120b` | `openai/gpt-oss-120b` | 1,000 | 8,000 | 130–770ms |
| `groq/gpt-oss-20b` | `openai/gpt-oss-20b` | 1,000 | 8,000 | faster, weaker |
| `groq/qwen3.6-27b` | `qwen/qwen3.6-27b` | 1,000 | 8,000 | see `<think>` below |
| `gemini/3.6-flash` | `gemini-3.6-flash` | unknown | unknown | ~13s, 1M context |

**Groq's daily cap is a leaky bucket, not a midnight reset.**
`x-ratelimit-reset-requests` returns `86400/limit` seconds per request spent
(86.4s each at 1,000/day, confirmed across three models with different caps).
Capacity trickles back all day.

**Gemini's endpoint sends no `x-ratelimit-*` headers at all**, so its limits
cannot be calibrated the way Groq's were. Marked `limits_are_estimated`; the old
2.5-flash free-tier figures are carried forward as a guess. Local counting is
the only defence there, which is why its reserve is kept generous.

**`gemini-2.5-flash` is still listed by the models endpoint but refused on a new
key** ("no longer available to new users"). **Being listed is not proof of
access — only a request is.** That is what `llmorch doctor --live` is for.
`gemini-3.7-flash` exists but timed out at 120s.

### On the account but NOT enabled

`groq/compound` and `groq/compound-mini` (250/day, **70,000 TPM**) and
`allam-2-7b` (7,000/day, 6,000 TPM). Their limits differ from the enabled three,
and the manifest declares limits **per provider**, not per model. Enabling them
needs per-model limit overrides — see §8, it is the highest-value next change.

### Two model behaviours that cost a whole afternoon

- **`gpt-oss` spends reasoning tokens before emitting anything.** A probe with
  `max_tokens=16` returned an **empty string** and 14 reasoning tokens.
- **`qwen3.6` emits `<think>` blocks inline in the message content**, not in a
  reasoning field, and reports `reasoning_tokens=0` while doing it — so the
  spend is invisible to accounting. Handled by `reasoning_headroom` in
  models.yaml and `salvage.strip_reasoning`.

---

## 4. Architecture

```
task text
  |
  +-- decompose ----> task DAG + shared interface contract   (M4, NOT BUILT)
  +-- bid ----------> each model rates itself per node       (M4, NOT BUILT)
  +-- reconcile ----> feasibility -> normalize -> score -> assign (0 requests)
  +-- execute ------> DAG scheduler, gated by the quota governor
  +-- materialize --> artifacts -> runs/<id>/output/         (0 requests)
```

### The dispatcher (`negotiate/reconcile.py`) — the "middle man"

Deterministic Python, **not** an LLM. Costs zero requests, runs instantly, is
testable offline, and cannot hallucinate an assignment that violates a rate
limit. Four inputs, in decreasing order of trustworthiness:

| Source | Contributes | Lives in | Live? |
|---|---|---|---|
| Capability sheet | Hand-written priors per role | `role_affinity` in models.yaml | yes |
| Self-reported bids | What a model claims, z-normalized per bidder | `bidding.py` | **M4** |
| Track record | Realized per-(model, role) performance (EWMA) | `profiles.json` | **M4** |
| Live quota | How close a model is to its daily wall | `governor.headroom()` | yes |

Score:
`0.35*z_conf + 0.25*role_affinity + 0.15*track_record + 0.15*quality_prior - 0.10*quota_pressure`

Then a **capacity-constrained assignment measured in tokens** (not node count),
plus a 2-opt swap pass. Evenness is a hard constraint in an algorithm, never an
instruction in a prompt.

### The interface contract

Decomposition emits a shared spec (routes, data models) that every node receives
verbatim. This is how a frontend from one vendor works against a backend from
another **without the models ever talking to each other**. Verified live.

---

## 5. File status

| File | Status | Notes |
|---|---|---|
| `types.py` | done | Core dataclasses; imports nothing internal |
| `errors.py` | done | Split on `is_retryable` — drives the failover ladder |
| `config.py` | done | Paths, .env loader, RunConfig. Never logs a key value |
| `models.yaml` | done | **Verified live.** Groq ×3 + Gemini active |
| `registry/manifest.py` | done | Validation incl. the cross-vendor chain rule |
| `quota/windows.py` | done | SlidingWindow, DayCounter, Clock (owns `sleep`) |
| `quota/estimator.py` | done | Char-based + per-provider EWMA self-calibration |
| `quota/governor.py` | done | **The core.** Admission control |
| `quota/store.py` | done | SQLite ledger; counters derived from events |
| `providers/base.py` | done | Provider protocol + registry |
| `providers/mock.py` | done | Canned responses + fault injection |
| `providers/openai_compat.py` | done | stdlib HTTP, injectable transport |
| `providers/headers.py` | done | Durations, Retry-After, per-minute vs daily 429 |
| `engine/graph.py` | done | Kahn levels, cycle repair, budget pruning |
| `engine/salvage.py` | done | Fences, balanced-JSON, `<think>` stripping |
| `engine/verify.py` | done (Tier 0) | Tier 1 review designed, wired at M4 |
| `engine/health.py` | done | Failover ladder + circuit breaker |
| `engine/worker.py` | done | One node; failover, budget growth, ledger writes |
| `engine/scheduler.py` | done | DAG execution + bulk reassignment |
| `engine/blackboard.py` | done | Summaries only, never whole artifacts |
| `engine/materialize.py` | done | Path safety + artifact writeout |
| `report/render.py` | done | plan/outcome/spend/quota tables |
| `report/ledger.py` | done | run history, today, lifetime totals, run detail |
| `demo/website.py` | done | Notes-app DAG + canned artifacts for dry runs |
| `__main__.py` | done | `run [--live]`, `plan`, `quota`, `ledger`, `doctor [--live]` |
| `negotiate/decompose.py` | **NOT BUILT** | **M4 — the big one.** Task text → DAG |
| `negotiate/bidding.py` | **NOT BUILT** | M4 |
| `negotiate/profiles.py` | **NOT BUILT** | M4 |
| `negotiate/plancache.py` | **NOT BUILT** | M4 |
| `engine/contracts.py` | **NOT BUILT** | Cross-artifact validation — M5 |
| `engine/checkpoint.py` | **NOT BUILT** | Resume across quota days — M5 |

### Tests (301 passing, 1 skipped)

| File | Tests | Covers |
|---|---|---|
| `test_foundations.py` | 16 | types, errors, config |
| `test_manifest.py` | 20 | manifest loading + validation rules |
| `test_governor.py` | 38 | admission, day rollovers, reserves, wait-vs-refuse |
| `test_materialize.py` | 23 | salvage + path safety (security-critical) |
| `test_reconcile.py` | 29 | graph + dispatcher + bid normalization |
| `test_health_verify.py` | 44 | failover, breaker, Tier 0, reasoning stripping |
| `test_integration.py` | 23 | full pipeline, faults, degradation, budget growth |
| `test_providers.py` | 31 | header parsing + adapter status/usage mapping |
| `test_store.py` | 25 | ledger, day boundaries, restore, limit scope |
| `test_live_pipeline.py` | 11 | the live path end to end, only the socket faked |

`test_live_pipeline.py` is the important one: it runs the **real** adapter,
scheduler and ledger against a canned HTTP transport. It caught two bugs a mock
provider structurally cannot reach.

---

## 6. Bugs found (all fixed)

Each would have cost live quota to discover. The first three are from the
offline build; the rest only appear once real requests are made — which is the
argument for building offline first *and* for going live early.

1. **models.yaml was internally invalid.** Groq models declared
   `max_output: 8192` against a per-request ceiling that made a full-size
   response impossible.
2. **Fair-share cap could be smaller than a single node**, making it
   unsatisfiable, so balancing silently never ran.
3. **check_css passed unparseable Python as valid CSS.** Bracket-balance alone
   accepts it. Worse, the wrongly-successful node *reset the breaker's streak*.
4. **`llmorch run` crashed on a default Windows console.** cp1252 stdout cannot
   encode the report tables' em dashes; the run died mid-render before writing
   anything. It had only ever been run in a UTF-8 terminal.
5. **Every scope bucket got every counter**, so each limit was enforced at
   whichever scope it was *not* declared at. Five calls to one Groq model made
   its siblings report five requests used against their own daily caps.
6. **A WAIT verdict was raised as `QuotaExhausted`**, benching a model for the
   whole run — the mirror image of the UNSERVABLE-is-not-WAIT invariant.
   Against Groq's account-scoped 30 RPM the first fan-out benched the entire
   roster and every remaining node degraded.
7. **Provider 429s tripped the circuit breaker.** A 429 is the provider
   enforcing a quota, not a model returning garbage.
8. **`sync_from_headers` derived usage from the manifest's ceiling, not the
   server's.** With a declared 14,400 and a real 1,000, `remaining=998` read as
   13,402 spent — a healthy provider benched on its second call.
9. **Restoring the ledger crashed on a retired model id.** The ledger is
   permanent and the manifest is not; correcting the wire names instantly broke
   every command that reads history.
10. **A model with no API key stayed in the fallback chains.** Excluding it from
    planning was not enough — the chains come from the manifest, so failover
    routed straight back to it and every node rediscovered the missing key.
11. **Reasoning models were starved.** qwen3.6 given 900 tokens for a stylesheet
    produced 900 tokens of `<think>` and half a CSS rule, three times in one
    run, tripping its breaker.
12. **Truncation failed over instead of growing the budget.** A truncated output
    with `max_output` still unspent means the *estimate* was too small; failing
    over handed the same under-budget to the next vendor, which truncated
    identically.

---

## 7. Load-bearing invariants — do not break these

- **UNSERVABLE is not WAIT.** A request larger than a provider's per-minute
  ceiling never fits. Treating it as a wait hangs the scheduler forever.
- **WAIT is not EXHAUSTED_TODAY.** Busy is not spent. Collapsing them benches a
  healthy model for the whole run over a pause of seconds. (`QuotaBusy`.)
- **max_tokens is always set explicitly.** It turns the completion estimate into
  a hard bound, which is what makes admission sound.
- **Reserve on estimate, reconcile on commit**, or concurrent fan-out races.
- **Monotonic clock for sliding windows; wall clock for day boundaries. Never
  mixed.** And **a clock owns `sleep`** — waiting on a real clock while
  measuring a fake one never advances toward the deadline.
- **Response headers override local counters**, and usage is derived from the
  **server's** stated ceiling, never the manifest's.
- **Running out of quota is NOT a health failure.** Nor is a 429. Nor is a
  missing key. None of them are the model being broken.
- **Failover prefers a DIFFERENT VENDOR**; chains must span ≥2 vendors,
  enforced at manifest load.
- **A reviewer must never share the author's vendor.**
- **output_path is untrusted input.** Resolve-then-verify-containment.
- **Downstream nodes get summaries, never whole artifacts.**
- **A degraded node must never fail the whole run.**
- **Dry runs never touch the ledger.** Mock traffic there would make the
  governor refuse real requests tomorrow over quota never spent.
- **Being listed is not proof of access.** Only a live request is.

---

## 8. What next

### Immediately valuable, in order

**1. Per-model limit overrides in the manifest.** *(small, unblocks a lot)*
`ProviderSpec` holds `limits`; `ModelSpec` has none, so every model on a
provider must share one set. This is the only thing keeping `groq/compound` out
of the roster, and its **70,000 TPM is nine times** the ceiling everything else
works under — which is exactly the constraint that shapes the whole design.
Touches `registry/manifest.py` and `quota/governor.py` (`_states_for` already
keys by model).

**2. `negotiate/decompose.py` — M4.** *(the big one)*
Turns task text into a DAG plus the interface contract, in one request. Until
this exists the CLI's task argument is decorative. Best served by Gemini's 1M
context; it is the one job Groq's 8,000 TPM makes impossible. Needs a strict
JSON schema and must route through `salvage.extract_json`, which already exists
for exactly this. Then `bidding.py` and `profiles.json` light up the two inert
scoring terms.

**3. Fix the leaky-bucket display.** *(small, currently misleading)*
`llmorch quota` claims Groq "resets in 15h00m" when a slot actually frees every
~86 seconds. `DayCounter` models a midnight reset. Conservative — it under-uses
quota rather than overspending — but wrong on screen.

**4. Widen the live sample.** Both clean runs needed retries; `style` took 5
attempts and crossed vendors both times. It converges, but n=2 is thin. Runs are
cheap (~7 requests of 1,000/day).

### Then

| # | Milestone | Status | Needs |
|---|---|---|---|
| M0 | Setup, secret hygiene | **DONE** | — |
| M1 | Offline skeleton, zero API calls | **DONE** | — |
| M2 | First real requests, Groq only | **DONE** | — |
| M3 | Add Gemini; cross-vendor failover live | **DONE** | — |
| M3.5 | NVIDIA NIM + Mistral (adaptive limit discovery) | | those keys |
| M4 | Negotiation live + Tier 1 cross-vendor review | next | — |
| M5 | `contracts.py`, `checkpoint.py`, Perplexity behind `--allow-paid` | | — |
| M6 | Web dashboard (read-only, localhost-only) | | — |

### Considered and rejected

- **Ox Alpha** (`stealth/ox-alpha` on OpenRouter, free, 1M context, appeared
  2026-08-20). Tempting — its context is 145× Groq's ceiling. Rejected because:
  the free window is ~1 week and stealth models vanish and return renamed
  (exactly the failure this project spent a day fixing); **OpenRouter is a
  router, not a vendor**, so it satisfies the cross-vendor rule on the letter
  while a gateway failure correlates across everything behind it; and its data
  terms are contradictory (the model page says prompts are retained by an
  anonymous provider, the launch post claims zero retention). Worth
  reconsidering *only* as opportunistic capacity for M4 decompose and M5
  contracts — never as a chain's sole rung.
- **The `openai` SDK** for the provider adapter. It retries internally, and a
  retry it performs is a request the governor never reserved and never counted,
  which breaks the invariant that every call passed through a ticket. The
  `providers` extra in `pyproject.toml` is now unused.

### Decisions already made (do not re-litigate)

- Demo is a **notes app, no auth** (list page, detail page, SQLite CRUD).
- Stack is **pinned** (plain HTML/CSS/JS + stdlib http.server + SQLite) so the
  contract checker knows what it is parsing.
- Dashboard is **deferred to M6**; output is terminal text + report.md.
- Perplexity is **paid** and needs both `--allow-paid` and a non-zero budget.
- Coordination is **negotiate once, then route** — not free-form model chatter.

---

## 9. How to run

```bash
.venv/Scripts/python.exe -m pytest -q
```

Preflight — offline checks, then confirm every wire name with one live call
each. **Re-run this whenever models.yaml changes.** It is cheap, and it is the
only thing between a renamed model and a mid-run failure that looks exactly
like a dead one:

```bash
.venv/Scripts/python.exe -m llmorch doctor --live
```

Dry run against the mock, no network, no keys:

```bash
.venv/Scripts/python.exe -m llmorch run "build a notes app"
```

The real thing:

```bash
.venv/Scripts/python.exe -m llmorch run --live "build a notes app"
```

Quota headroom, and what the ledger has recorded:

```bash
.venv/Scripts/python.exe -m llmorch quota
```

```bash
.venv/Scripts/python.exe -m llmorch ledger --totals
```

Then run the generated app (serves on http://localhost:8000):

```bash
.venv/Scripts/python.exe runs/<id>/output/server.py
```

---

## 10. Environment and secrets

- Python **3.14.1**, venv at `.venv/`
- PyYAML, pydantic 2.13, tzdata 2026.3, pytest, pytest-asyncio
- **`tzdata` is a hard requirement on Windows** — `zoneinfo` cannot resolve
  `America/Los_Angeles` without it, and Gemini's quota day depends on it
- Ledger lives at `%LOCALAPPDATA%\llmorch\state.db`, **outside the checkout**:
  quota belongs to the account, not the working copy, so two clones must share
  one ledger or both believe they hold the full daily allowance

`.env` is gitignored and holds the real keys. `.gitignore` was written *before*
`.env` existed, so no key ever sat in a trackable file. `.env.example`
documents each provider, its limits, and its signup URL.

`GROQ_API_KEY` and `GEMINI_API_KEY` are both set. NVIDIA NIM, Mistral and
Perplexity are unset, which is what gates M3.5 and M5.

Dry runs need **no keys at all**.

Full original plan, with the rationale behind every decision and an 18-entry
decisions log:
`C:\Users\prana\.claude\plans\this-is-totally-a-glimmering-snail.md`
