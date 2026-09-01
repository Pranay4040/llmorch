# llmorch — Project Blueprint & Status

> Self-contained handoff. Everything needed to resume work with no prior context.
> Last updated: M4b (negotiation) live on 2026-09-01. The system now plans and builds tasks it has never seen; §13 covers what that exposed.

---

## 1. What this is

A **multi-provider LLM orchestrator**. It takes a task ("build a notes app"),
splits it into a dependency graph, assigns each slice to whichever AI model is
best suited to it, executes them across vendors, and writes a real runnable
project folder to disk.

Original goal: *have AI models from different companies collaborate, split work
by their strengths, and conserve scarce API quota.*

---

## 2. Three findings that shaped the design

**(a) Claude and ChatGPT are not in the roster.** *(Superseded 2026-09-01 —
see §10.)* No Anthropic or OpenAI API key was available, and subscriptions
cannot be called from code. Vendor diversity came from Google (Gemini) and the
open-model vendors Groq hosts. The architecture is provider-agnostic — adding a
key later is a manifest entry, not a rewrite, and §10 is that claim being
cashed.

One correction the live probe forced: **Groq serves no Meta chat model on this
account.** The only `meta-llama/*` entries are prompt-guard classifiers. Within
Groq the vendors are OpenAI (open-weights) and Alibaba, not Meta.

**(b) "Save credits" is a quota-scheduling problem, not a cost problem.**
Nothing in the roster bills per token. The scarce resources are rate limits:

| Provider | RPM | TPM | RPD | Reset TZ | Source |
|---|---|---|---|---|---|
| Groq | 30 | **8,000** | **1,000** | UTC | its own `x-ratelimit-*` headers |
| Gemini 2.5 Flash | 10 | 250,000 | **250** | America/Los_Angeles | docs — sends no headers |

Two walls dominate every decision:
- **Groq's 8,000 TPM** caps a single request at ~7,200 tokens. Larger requests
  are *permanently unservable*, not merely delayed.
- **Daily requests are scarce on both.** 1,000 and 250 — Groq is four times
  Gemini, not fifty times. The original plan called Groq "abundant" on the
  strength of a published 14,400/day that this account does not have.

The Groq row was wrong in both directions until the first live call read the
headers, which is the whole argument for treating headers as fact and local
counting as inference.

**(c) Free models are unreliable self-reporters.**
Asked to rate themselves they all claim high competence at everything. So models
*bid*, but a deterministic Python pass makes the final assignment.

---

## 3. Architecture

```
task text
  |
  +-- decompose ----> task DAG + shared interface contract      (1 request)
  +-- bid ----------> each model rates itself per node          (N <= 4 requests)
  +-- reconcile ----> feasibility -> normalize -> score -> assign (0 requests)
  +-- execute ------> DAG scheduler, gated by the quota governor
  +-- materialize --> artifacts -> runs/<id>/output/            (0 requests)
```

### The dispatcher (`negotiate/reconcile.py`) — the "middle man"

Deterministic Python, **not** an LLM. Costs zero requests, runs instantly, is
testable offline, and cannot hallucinate an assignment that violates a rate
limit. Four inputs, in decreasing order of trustworthiness:

| Source | Contributes | Lives in |
|---|---|---|
| Capability sheet | Hand-written priors per role | `role_affinity` in models.yaml |
| Self-reported bids | What a model claims, z-normalized per bidder | `bidding.py` (M4) |
| Track record | Realized per-(model, role) performance (EWMA) | `profiles.json` (M4) |
| Live quota | How close a model is to its daily wall | `governor.headroom()` |

Score:
`0.35*z_conf + 0.25*role_affinity + 0.15*track_record + 0.15*quality_prior - 0.10*quota_pressure`

Then a **capacity-constrained assignment measured in tokens** (not node count),
plus a 2-opt swap pass. Evenness is a hard constraint in an algorithm, never an
instruction in a prompt.

### The interface contract

Decomposition emits a shared spec (routes, data models) that every node receives
verbatim. This is how a frontend from one vendor works against a backend from
another **without the models ever talking to each other**.

---

## 4. Current status

**M0-M6 complete; the quota layer is now a library. 434 tests pass, 1 skipped** (symlink
test needs admin on Windows).

Verified working end to end:

```bash
python -m llmorch run "build a notes app"
```

Splits work across 4 models / 2 vendors, writes `runs/<id>/output/`, and the
generated app genuinely serves: POST creates a note, GET lists, detail fetches,
`/` returns the page. All against mocks — zero network calls.

**First live cross-vendor build: 2026-09-01.** Six nodes, two vendors, real
models. Gemini failed the `server` node and Groq picked it up — cross-vendor
failover observed rather than simulated. One node degraded on a quota wall;
`llmorch resume` finished it by re-requesting exactly one node and carrying the
other five over at zero cost. The generated API genuinely serves
(POST creates, GET lists); its static handler does not, on Windows — see §5.10.

M2 added the live path: an OpenAI-wire HTTP adapter, header parsing, a SQLite
usage ledger that survives the process, and `llmorch doctor`. The whole live
pipeline is exercised in tests through an injected transport — `test_providers.py`
runs the real adapter end to end over a fake wire, including cross-vendor
failover on a 404 — so the only thing still unverified is what the providers
themselves say back.

**The one thing M2 has not done: call a real endpoint.** `GROQ_API_KEY` is
present in the environment (not in `.env`, which is still blank), so
`llmorch doctor --probe` is ready to run and will confirm the three Groq wire
names with one 16-token call each. Until it does, those names remain guesses.

### Environment
- Python **3.14.1**, venv at `.venv/`
- Installed: PyYAML, pydantic 2.13, tzdata 2026.3, pytest, pytest-asyncio
- **No HTTP or provider SDK dependency.** The adapter is stdlib `urllib` on a
  worker thread. The vendor SDKs each carry their own retry/rate-limit layer,
  which would sit *underneath* the governor and retry requests it never
  admitted — and admission control only works if every call goes through it
- `tzdata` is a **hard requirement** on Windows — `zoneinfo` cannot resolve
  `America/Los_Angeles` without it (verified working)

### File status

| File | Status | Notes |
|---|---|---|
| `types.py` | done | All core dataclasses; imports nothing internal |
| `errors.py` | done | Split on `is_retryable` — drives the failover ladder |
| `config.py` | done | Paths, .env loader, RunConfig. Never logs a key value |
| `models.yaml` | done | Groq + Gemini active; NIM/Mistral/Perplexity inactive |
| `registry/manifest.py` | done | Validation incl. the cross-vendor chain rule |
| `quota/windows.py` | done | SlidingWindow, DayCounter, injectable Clock |
| `quota/estimator.py` | done | Char-based + per-provider EWMA self-calibration |
| `quota/governor.py` | done | **The core.** Admission control |
| `quota/store.py` | done | SQLite ledger; day keys per provider tz, replay into the governor |
| `providers/base.py` | done | Provider protocol + registry |
| `providers/mock.py` | done | Canned responses + fault injection |
| `providers/openai_compat.py` | done | stdlib-urllib HTTP adapter, injectable transport |
| `providers/headers.py` | done | Rate-limit headers → snapshot; daily-vs-burst detection |
| `engine/graph.py` | done | Kahn levels, cycle repair, budget pruning |
| `engine/salvage.py` | done | Fence stripping, balanced-JSON recovery |
| `engine/verify.py` | done | Tier 0 checks + Tier 1 schema, reviewer selection, verdict parsing |
| `engine/health.py` | done | Failover ladder + circuit breaker |
| `engine/worker.py` | done | Executes one node, fails over across vendors |
| `engine/scheduler.py` | done | DAG execution + bulk reassignment |
| `engine/blackboard.py` | done | Summaries only, never whole artifacts |
| `engine/materialize.py` | done | Path safety + artifact writeout |
| `engine/review.py` | done | Tier 1 execution: review, one bounded repair |
| `engine/contracts.py` | done | Cross-artifact agreement: pages, assets, routes, schema |
| `engine/checkpoint.py` | done | Atomic per-run checkpoint; resume never re-buys work |
| `negotiate/roles.py` | done | Fixed taxonomy + alias parsing |
| `negotiate/reconcile.py` | done | **The dispatcher** |
| `negotiate/decompose.py` | done | Task -> DAG + contract; output validated, never trusted |
| `negotiate/bidding.py` | done | One request per model, entirely optional |
| `negotiate/profiles.py` | done | Per-(model, role) EWMA, shrunk by sample count |
| `negotiate/plancache.py` | done | Same task + roster -> zero planning requests |
| `report/render.py` | done | plan/outcome/spend/quota tables, doctor sweep, resume list |
| `report/ledger.py` | done | Cross-run usage tables; what was carried over today |
| `demo/website.py` | done | Notes-app DAG + real working canned artifacts |
| `doctor.py` | done | Pre-flight sweep; `--probe` verifies wire names live |
| `discover.py` | done | Asks each spare key what it can reach, spending no tokens |
| `dashboard/state.py` | done | One snapshot: quota, runs, spend, track record |
| `dashboard/server.py` | done | stdlib HTTP, loopback-only, GET-only |
| `dashboard/page.py` | done | Self-contained page; writes values as text, never markup |
| `__main__.py` | done | `run` (+`--live`), `resume`, `plan`, `quota`, `ledger`, `doctor` |

### Tests (434 passing)

| File | Covers |
|---|---|
| `test_foundations.py` | types, errors, config |
| `test_manifest.py` | manifest loading + validation rules |
| `test_governor.py` | admission control, day rollovers, reserves |
| `test_materialize.py` | salvage + path safety (security-critical) |
| `test_reconcile.py` | graph + dispatcher + bid normalization |
| `test_health_verify.py` | failover, circuit breaker, Tier 0 checks |
| `test_integration.py` | full pipeline, fault injection, degradation |
| `test_providers.py` | header dialects, error mapping, full run over a fake wire |
| `test_store.py` | ledger persistence, provider-local days, governor replay, doctor |
| `test_checkpoint.py` | checkpoint format, graph fingerprint, resume skips paid-for work |
| `test_review.py` | who reviews, what gets reviewed, and the three ways review could make things worse |
| `test_contracts.py` | cross-artifact agreement, one deliberate break at a time |
| `test_negotiate.py` | plan validation, bid normalisation, track-record shrinkage |
| `test_dashboard.py` | read-only and loopback-only, proved against a real socket |
| `test_public_api.py` | the import surface, and the README example executed verbatim |

---

## 5. Bugs found during the build (all fixed)

Each would have cost live quota to discover — the reason for building offline first.

1. **models.yaml was internally invalid.** Groq models declared
   `max_output: 8192`, but Groq's 6,000 TPM caps a request at ~5,400 tokens, so
   those models could never serve a full-size request. Now capped at 4,096 —
   which leaves only ~1,300 tokens for the prompt. **Groq nodes must stay small.**

2. **Fair-share cap could be smaller than a single node**, making it
   unsatisfiable — every node fell through to "best available" and balancing
   silently never ran. Fixed with a floor of `max(largest_node, even_share)`.

3. **check_css passed unparseable Python as valid CSS.** Bracket-balance alone
   accepts it (zero braces is balanced). Worse, the wrongly-successful node
   *reset the circuit breaker's failure streak*. Now requires at least one rule block.

4. **A finished run could die while printing its own summary.** The plan and
   quota tables use box-drawing characters; on a console still defaulting to
   cp1252 (Git Bash here) `print` raised UnicodeEncodeError *after* every
   artifact had been written. The work survived, the report did not. `main()`
   now reconfigures stdout to UTF-8 with `errors="replace"`.

5. **Two of three Groq wire names did not exist.** `llama-3.3-70b-versatile`
   and `qwen3-32b` both returned "model does not exist or you do not have
   access to it". `GET /v1/models` shows the account serves no Meta chat model
   at all. Found by `doctor --probe` for the cost of two 16-token calls; found
   mid-run it would have cost the planning request first. This is the single
   clearest argument for the probe existing.

6. **The published Groq limits were wrong in both directions.** Documentation
   said 6,000 TPM and 14,400 requests/day; the account's headers say 8,000 and
   1,000. Every "Groq is the abundant one" decision rested on a number nothing
   had ever checked.

7. **Groq's edge returns 403 to the default urllib User-Agent.** The adapter
   sets `llmorch/0.1` on every request and never hit it; a diagnostic script
   written without one did, and looked exactly like an auth failure.

8. **A per-minute wait was recorded as running out of quota for the day.**
   `worker._call` mapped every non-UNSERVABLE denial to `QuotaExhausted`, so a
   WAIT — a token window that clears in seconds — marked a healthy model
   EXHAUSTED for the whole run and handed its work to a worse one. The
   blueprint's own invariant had a sibling nobody had written down: **WAIT is
   not EXHAUSTED.** Invisible offline, because the mock never refuses
   admission. `test_a_per_minute_wait_does_not_write_a_model_off_for_the_day`
   fails against the old mapping and passes against the new one.

9. **Gemini charges hidden thinking tokens against `max_tokens`.** They are not
   reported in `usage`. Measured on the same prompt: `max_tokens=900` returned
   `finish_reason=length` with 36 visible tokens; `max_tokens=8000` returned
   9,734 characters and `finish_reason=stop`. Neither `reasoning_effort: none`
   nor `google.thinking_config` is accepted (HTTP 400), so thinking cannot be
   switched off — only budgeted for. Hence `min_output_tokens` on ModelSpec.
   The floor costs nothing real: tokens are reserved at that size and
   reconciled to actual usage on commit.

10. **The generated app's API worked and its static handler did not — on
    Windows only.** `translate_path` ran `os.path.normpath` *before* splitting
    on `/`, so `/index.html` became `\index.html`, and `os.path.join` treats a
    leading backslash as drive-absolute: every page resolved to `C:\index.html`
    and 404'd. Valid Python, passes Tier 0, passes every API call. Found only by
    running the thing. This is the case for Tier 1 review (M4) and the contract
    checker (M5), and the reason "the tests pass" is not the finish line.

11. **The `response_format` envelope had been wrong since M2.** The bare JSON
    Schema was sent as `json_schema`, where the API expects
    `{"name": ..., "schema": ...}`. Never noticed because nothing sent a schema
    until Tier 1 did — dead code is untested code even when it looks obviously
    right.

12. **Fixing WAIT-as-EXHAUSTED exposed the next collapsed distinction.** With
    waits raised as `RateLimited`, two of them in a row tripped the circuit
    breaker and marked a healthy model *broken*. Same error as counting a daily
    cap as a fault, one rung down: the model answered correctly every time it
    was asked, it just could not be asked yet. Rate limits now carry no health
    penalty and get a longer same-model retry allowance, because waiting
    genuinely fixes a full window and failing over just moves identical load
    onto a model whose window is equally full.

13. **Nothing carried quota across processes.** Counters lived only in memory, so
   a second run an hour later started from zero and believed it held the whole
   daily allowance. On Gemini that discovery costs one of 250 requests to make.
   This is what `quota/store.py` exists for — and why every command that reads
   quota replays the ledger first.

---

## 6. Load-bearing invariants — do not break these

- **UNSERVABLE is not WAIT.** A request larger than a provider's per-minute
  ceiling never fits. Treating it as a wait hangs the scheduler forever.
- **max_tokens is always set explicitly** on every request. It turns the
  completion estimate into a hard bound, which is what makes admission sound.
- **Reserve on estimate, reconcile on commit.** Without it, concurrent fan-out
  races — several callers each read a counter none has yet incremented.
- **Monotonic clock for sliding windows; wall clock for day boundaries. Never
  mixed.** Wall time for RPM would let a clock adjustment grant free quota;
  monotonic for RPD would never reset.
- **Response headers override local counters.** Local counting is inference;
  headers are fact and cost nothing to read.
- **Running out of quota is NOT a health failure.** A model at its daily cap is
  unavailable, not broken — it must not accumulate a track-record penalty.
- **Failover prefers a DIFFERENT VENDOR.** Failure modes correlate within a
  vendor; a third attempt at the same one tends to fail identically.
- **Fallback chains must span at least 2 vendors** — enforced at manifest load.
- **A reviewer must never share the author's vendor** — self-review re-approves
  its own mistakes.
- **output_path is untrusted input.** It is the only place model output reaches
  the filesystem. Resolve-then-verify-containment, never string inspection.
- **Downstream nodes get summaries, never whole artifacts.** Pasting upstream
  files into downstream prompts is the fastest way to exhaust 6,000 TPM.
- **A degraded node must never fail the whole run.**
- **A dry run never writes to the ledger.** Mock calls consume no real quota;
  recording them would tell tomorrow's admission control that requests were
  spent which never were.
- **The ledger is the source of truth; counters are a cache of it.** Every
  command that reasons about quota replays today's rows before answering.
- **Replay restores daily counters only.** Sliding windows are monotonic-clocked
  and meaningless across processes — and they drain within a minute anyway.
- **A request that reached the provider counts, even when it failed.** A 429 has
  usually already consumed its slot. Only a call that never left this machine
  (status 0) is free, which is what the ledger's `http_status > 0` keys off.
- **The transport never retries.** Retry policy lives above the governor. A
  vendor SDK retrying underneath it would fire requests admission control never
  granted — which is the one thing that would make the whole design a fiction.

---

## 7. Roadmap

| # | Milestone | Status | Needs |
|---|---|---|---|
| M0 | Setup, secret hygiene | **DONE** | — |
| M1 | Offline skeleton, zero API calls | **DONE** | — |
| M2 | First real requests, **Groq only** | **DONE + VERIFIED LIVE** | — |
| M2.5 | Checkpoint + resume across a quota day | **DONE** | — |
| M3 | Add Gemini; cross-vendor failover live | **DONE + OBSERVED LIVE** | — |
| M3.5 | Roster expansion | **DONE** (see §14) | — |
| M4 | Tier 1 cross-vendor review | **DONE + LIVE** (see §11) | — |
| M4b | Negotiation live (decompose, bidding, profiles) | **DONE + LIVE** (see §13) | — |
| M5 | Cross-artifact contract checking | **DONE** (see §12) | — |
| M5b | Optimization; Perplexity behind `--allow-paid` | | — |
| M6 | Web dashboard (read-only, localhost-only) | **DONE** (see §15) | — |

### What M2.5 added (checkpoint + resume)

A daily cap is not a crash, it is a scheduled event — so the run now writes
`runs/<id>/checkpoint.json` after every wave, artifact text included, and
`llmorch resume` runs only what is missing.

Measured on the demo graph: a run blocked on one node costs **6 requests** to
redo from scratch and **1** to resume. The wall gets paid for once.

- Writes are atomic (temp file + `os.replace`) — a crash mid-write cannot
  destroy the last good checkpoint.
- The graph is fingerprinted, so a checkpoint cannot be replayed onto a changed
  plan and silently stitch two sets of assumptions together.
- Only DONE nodes are restored; a degraded node's stub is discarded and re-run.
- `resume` refuses while the blocked model is still out (exit 2) unless
  `--force`: spending three other models on work that is waiting for a fourth is
  how one quota wall becomes two.

### The immediate next step

**Run `llmorch doctor --probe`.** It is the last unverified thing in M2: the
Groq wire names (`llama-3.3-70b-versatile`, `qwen3-32b`, `openai/gpt-oss-120b`)
are still guesses. Three calls, 16 output tokens each, against a 14,400/day
budget. `GROQ_API_KEY` is present in the environment.

Then **M3**, which needs `GEMINI_API_KEY`: with two live vendors, cross-vendor
failover stops being a tested behaviour and becomes an observed one.

### Decisions already made (do not re-litigate)

- Demo is a **notes app, no auth** (list page, detail page, SQLite CRUD).
- Stack is **pinned** (plain HTML/CSS/JS + stdlib http.server + SQLite) so the
  contract checker knows what it is parsing. Swapping to a richer stack later
  touches only the decompose prompt and contracts.py.
- Dashboard is **deferred to M6**; M1–M5 output is terminal text + report.md.
- Perplexity is **paid** (~$1/Mtok PLUS $5–14 per 1,000 requests) and needs both
  `--allow-paid` and a non-zero budget.
- Coordination is **negotiate once, then route** — not free-form model chatter.

---

## 8. How to run

```bash
.venv/Scripts/python.exe -m pytest -q
```

```bash
.venv/Scripts/python.exe -m llmorch plan --explain "build a notes app"
```

```bash
.venv/Scripts/python.exe -m llmorch run "build a notes app"
```

```bash
.venv/Scripts/python.exe -m llmorch quota
```

```bash
.venv/Scripts/python.exe -m llmorch doctor          # add --probe for live checks
```

```bash
.venv/Scripts/python.exe -m llmorch ledger --days 3 --recent 10
```

```bash
.venv/Scripts/python.exe -m llmorch resume --list   # then: resume [<run_id>]
```

Then run the generated app (serves on http://localhost:8000):

```bash
.venv/Scripts/python.exe runs/<id>/output/server.py
```

---

## 9. Secrets

`.env` exists with **all keys blank**, gitignored. `.env.example` documents each
provider, its free-tier limits, and its signup URL. `.gitignore` was written
*before* `.env` existed, so no key ever sat in a trackable file.

Milestones 1 and 2.5 need **no keys at all** — both run fully on the mock.

`GROQ_API_KEY` is currently set **in the environment**, not in `.env`. That is
enough for `llmorch doctor --probe` and `llmorch run --live` to work; it also
means a live call is one flag away, so the `--live` flag is deliberately
explicit and never the default.

---

## 10. The roster as of 2026-09-01

Eleven keys are now in `.env`. Two are wired into models.yaml and verified live;
the rest are held back because an unverified provider in the manifest is a
liability, not an asset — that is the lesson of §5.5.

| Key | Status |
|---|---|
| `GROQ_API_KEY` | **live** — 3 models verified |
| `GEMINI_API_KEY` | **live** — gemini-2.5-flash verified |
| `OPENROUTER_API_KEY` | held — *this is the one that changes the premise* |
| `CEREBRAS_API_KEY` | held |
| `MOONSHOT_API_KEY` (Kimi) | held |
| `DEEPSEEK_API_KEY` | held |
| `MISTRAL_API_KEY` | held — provider already declared, no models |
| `FIREWORKS_API_KEY` | held |
| `OPENCODE_ZEN_API_KEY` | held |
| `WAFER_API_KEY` | held — vendor unidentified |
| `NVIDIA_NIM_API_KEY` | held — M3.5, provider declared, no models |

**OpenRouter overturns finding (a).** It fronts Anthropic and OpenAI models
behind one OpenAI-compatible key — the wire format the M2 adapter already
speaks. "Claude and ChatGPT are not in the roster" is now a choice rather than a
constraint. What it does not change: those models are *paid*, so they belong
behind `--allow-paid` and a non-zero budget alongside Perplexity, and the
`cost_usd` column that has read zero since M2 starts to matter.

Every addition needs the same three things before it is trusted: a base URL, a
wire name confirmed by `doctor --probe`, and limits — declared
`limits_are_estimated: true` until headers or a 429 say otherwise.

---

---

## 11. What Tier 1 review actually catches

Wired 2026-09-01 and measured against the one artifact known to be broken —
the `normpath` server from §5.10. Four attempts, in order:

| What the reviewer was given | Verdict |
|---|---|
| "Review this file against its spec" | **pass** — missed it |
| Same, plus "trace one request end to end first" | **pass** — missed it |
| Same, plus *"the server runs on Windows"* | **revise** — exact diagnosis |
| Trace instruction + **runtime in the interface contract** | **revise** — exact diagnosis, no hints |

The prompt was never the problem. **Correctness is relative to where the code
runs, and the contract never said.** On the platform the reviewer assumed, the
code was right — "pass" was a defensible answer to the question it was actually
asked.

So the fix went into the contract rather than the prompt: `InterfaceContract`
now carries a `runtime` field — OS, interpreter, working directory, how it is
launched — and every node sees it, author and reviewer alike. The next live run
produced a server that does not hand-roll path handling at all: told where it
would run, the model reached for the resolver that already works. Six of six
nodes, every route 200, POST creates and GET lists.

Both halves matter and neither is sufficient:

- **The trace requirement** is enforced through the schema, not asked for in
  prose: `trace` is the first property and is required, so the reviewer must
  walk the file before it is allowed to judge it. The trace is then discarded —
  its value is in being produced before the verdict, not after.
- **Review is advisory, never fatal.** No reviewer, no quota, an unparseable
  reply — each skips review and accepts the artifact. Work a model actually
  produced is not discarded because its critic was unavailable.
- **Reviews never draw on the reserve** (NORMAL priority only) and repairs are
  capped at one per node, un-re-reviewed. Otherwise a harsh reviewer and a
  stubborn author trade requests until the day's budget is gone.

What it still would not catch is anything about the *set* of artifacts — that
gap is closed by §12.

---

## 12. Cross-artifact checking (M5)

Tier 0 asks whether a file parses. Tier 1 asks whether a file does its job.
Neither can see the failure a split build makes likely and a single author would
never make: **each file impeccable against its own spec, and the project broken
because two models agreed with the spec and not with each other.**

Five checks, all deterministic, no request and no model:

| Check | Catches |
|---|---|
| pages exist | a promised page whose node degraded |
| assets resolve | `index.html` linking a `style.css` nobody wrote |
| frontend calls are declared | `/api/note/1` against `/api/notes/1` |
| routes are served | a declared route the backend never mentions |
| schema covers models | an API field with no column — **warning**, since a field can be computed rather than stored |

Two judgements worth keeping:

**It reports, it does not gate.** By the time these run the artifacts exist and
were paid for. A half-matching project someone can fix in two minutes beats an
empty folder and a clean conscience.

**A parameterised route hides behind its collection.** `/api/notes/{id}` shares
its base with `/api/notes`, so finding the base proves nothing about the
parameterised handler. Dropping only that handler is therefore reported as a
*warning*: a segment-splitting dispatch can be correct without ever containing
the literal prefix, and a checker this shallow should say "look here", not
"this is broken".

Measured against a real degraded run: the styling node hit a quota wall, and the
check named both pages left pointing at a stylesheet that does not exist —
before anyone opened a browser.

---

## 13. Negotiation live (M4b)

`llmorch run "<anything>"` now plans its own graph. Proven on a task the
codebase had never seen — *"build a command-line tool that converts CSV files to
a formatted markdown table"* — which Gemini split into five nodes (parser,
formatter, CLI, packaging, tests) that four models across two vendors then
built.

Three inputs to one decision, each trusted differently:

- **The plan** is a model deciding *structure*, so its output is validated
  hard: ids normalised, dangling deps dropped, oversized nodes clamped to what
  the tightest model in the roster can serve, unusable plans refused outright.
  Cached by task **and roster** — a graph split for a 4,096-token ceiling is the
  wrong graph once a 65,536-token model joins. A repeat run costs zero requests.
- **Bids** are a model describing *itself*, so they are z-normalised within each
  bidder: what survives is not "how good am I" but "which of these do I prefer,
  relative to my own baseline". One request per model for the whole run, and
  `auto` skips the round entirely when there is nothing to decide.
- **The track record** is the only input grounded in what happened. Per
  (model, role) EWMA, shrunk toward neutral by sample count so one lucky result
  does not outrank a long history, persisted because every data point cost a
  live request. A quota wall is never recorded as a failure.

### What the first novel build exposed

**The interface contract is web-shaped.** For the CSV tool the planner emitted
no routes and no pages — correctly, there are none — and the real coordination
surface, the module API, had no field to live in. So nothing coordinated it and
nothing checked it. The result:

```
parser.py:  def parse_csv(data, delimiter=None, strip_whitespace=True)
cli.py:     parse_csv(text=csv_text, delimiter=..., has_header=...)
            -> TypeError on the first line of real work
```

Both files parse. Both match their own spec. Tier 1 reviewed them and passed
them, because each *is* correct on its own.

The fix reads the code instead of asking the planner for more declarations:
`check_python_calls` parses every Python artifact, collects module-level
signatures, and verifies that calls between them agree. It found all four
mismatches in that build for free. It is deliberately conservative — only names
defined exactly once across the set, never a function taking `**kwargs`, never a
method on an object that is not one of our own modules — because a false
accusation about working code is worse than a missed fault.

### Two more corrections from the same run

**Gemini's real limit is 20 requests per *minute*, not 250 per day.** The 429
says so outright: `Quota exceeded for metric: generate_content_free_tier_requests,
limit: 20 ... Please retry in 12.39s`. Two bugs sat in the way of reading it: the
error message was trimmed to 400 characters, cutting off the quota metric, and
Gemini wraps its error object in a JSON *array*, so the parser fell through to a
stringified list.

**A stated wait now outranks any keyword.** That same 429 lists several quota
ids, some of them per-day, while the metric that actually tripped is per-minute
— so scanning the body for "per day" read the wrong line and marked a healthy
model exhausted for the rest of the run. If the server says come back in twelve
seconds, waiting works, and it is not the daily wall. The keyword heuristic now
only stands when no short retry-after contradicts it.

---

## 14. Roster expansion (M3.5)

Nine spare keys. `llmorch discover` asks each one what it can reach — a plain
`GET /models`, no tokens, safe to run against a key whose pricing nobody has
checked. Seven answered. **One turned out to be usable.**

| Key | `GET /models` | A completion |
|---|---|---|
| OpenRouter | 419 models, 18 free | **works** — added |
| Cerebras | 2 models | "Payment required to access this resource" |
| NVIDIA NIM | 82 models, ten companies | "Function … Not found for account" |
| OpenCode Zen | 63 models, incl. Claude and GPT | paid; not probed |
| DeepSeek / Moonshot / Fireworks | 3 / 4 / 24 | paid; not probed |
| Mistral | — | 401, the key is rejected |
| Wafer | — | 404, vendor unidentified |

**A model list is not an entitlement.** Cerebras and NIM both answer `/models`
perfectly and refuse every completion. Discovery proves a base URL and a key;
only `doctor --probe` proves a model. That distinction is now enforced by a
test: a provider may be `enabled` only if a live call to it has succeeded.

The NIM result is the sharpest disappointment of the exercise — its catalogue
lists Meta, Mistral, IBM, Writer, DeepSeek, Microsoft, Moonshot, OpenAI, Google
and 01-ai behind one key, which is precisely the multi-company diversity §1
set out to get. The account simply is not entitled to any of it.

### What was added

Three OpenRouter free models, each confirmed by a live probe, each from a
different company: **MiniMax** (`minimax-m3`, 1M context), **NVIDIA**
(`nemotron-3-ultra-550b`), **Cohere** (`north-mini-code`). Four other free
models were tried and rejected — two upstream provider errors, one that
returned no choices, one gated to a tier this key lacks — and they are named in
models.yaml so nobody re-adds them hopefully.

The roster is now **7 models across 3 vendors**, and every role chain reaches
all three. That matters beyond redundancy: the failover rule prefers a vendor
that has not yet failed, and with two vendors a chain runs out of unfamiliar
ground after one failure.

Their `quality_prior` values are placeholders and say so. Nothing here has a
track record, and inventing one would be fabricating evidence — §13's
`profiles.json` will replace them with what actually happens, per role.

### The architectural claim, cashed

§1 said adding a vendor "is a manifest entry, not a rewrite". Adding OpenRouter
was a YAML block, three model entries, and a line in each role chain. **No
Python changed.** The adapter, the governor, the ledger, review, and the
contract checker all took a third vendor without noticing.

---

## 15. The dashboard (M6)

`llmorch dashboard` serves one page on 127.0.0.1 showing quota with reset
times, runs and whether each is resumable, spend by day, the learned track
record, and the last thirty calls. It polls every five seconds.

Three properties, each enforced in code rather than intended:

**Read-only.** No endpoint starts a run, cancels one, or edits anything; POST,
PUT, PATCH and DELETE return 405 *by design rather than by omission*, and there
is a test per verb. This is why the milestone was cheap enough to leave for
last: a page that could spend quota needs authentication, rate limiting and a
threat model, and a page that can only look needs none of them.

**Loopback only.** A non-loopback bind raises rather than being quietly
accepted. There is no authentication, and the ledger is a complete record of
what this account has spent.

**It cannot execute what a model wrote.** Error text, node ids and task
descriptions all originate from providers. The server therefore sends no
interpolated markup at all — a static page plus a JSON document — and the page
writes every value through `textContent`. A test greps the page for
`innerHTML =`, `insertAdjacentHTML(`, `document.write(` and `eval(`, and a
ledger row containing `<script>` is asserted to arrive as characters.

It reads the same functions the CLI does. A dashboard that computed its own
numbers would eventually disagree with `llmorch quota`, and then nobody would
know which to believe. Estimated limits are labelled `(est)` on the page, so
OpenRouter's guessed ceiling never reads as a measurement.

One real bug surfaced while testing it: refusing a request without draining its
body left the unread bytes in the socket, where the next parse treated them as
a new request — the client saw a connection reset instead of the 405 it was
actually sent. Only PUT failed, which is exactly the kind of asymmetry a
per-verb test catches and a single happy-path test does not.

---

## 15b. Too small is not incapable

The third instance of one mistake, and the clearest.

A node's `est_output_tokens` is a guess made by the planner before any of the
work exists, and it is systematically low for the files nobody can size in
advance — test suites, complete pages, anything enumerating cases. The budget
was `estimate x 2`, and when a reply came back cut off at that budget the worker
recorded a failure against the model and **failed over to the next one with the
same budget**, which truncated identically. Observed live twice: a test-suite
node estimated at 1,000 tokens burned two models and degraded, and nobody
involved was incapable of writing the file.

It is the same error as counting a per-minute wait as a daily exhaustion, and
counting a rate limit as a fault: *blaming a model for a constraint the system
imposed on it.* Truncation is now evidence about the budget:

- The budget doubles and the **same** model is asked again, up to whatever that
  model allows and at most `max_escalations` times.
- Growing it carries no health penalty — the events list records "output budget
  raised to N" instead, so the retry does not read as misbehaviour.
- The larger budget **travels with the node** across a failover, because a file
  that did not fit in 2,000 tokens will not fit for the next model either.
- Truncation with no room left to grow is still a real failure.

The planner is also now told what an estimate costs — that a file is cut off at
roughly twice it, and that a test file for a small module is rarely under 1,200
tokens. It could not have known that otherwise.

Verifying this exposed a second defect. Resuming the affected run made the
resume **re-plan**, because the plan signature includes the roster and
OpenRouter had been added in between — so it spent the scarcest request in the
system to rediscover a known answer, and could have produced a different graph.
Checkpoints now carry the plan they were run against, serialised in
`engine/checkpoint.py` rather than by the negotiation package, so a checkpoint
can be read back by something other than whatever happened to write it.

---

## 16. The quota layer as a library

The orchestrator is the demo. The part with value outside this repository is the
quota arithmetic, so `llmorch.quota` is now a documented import surface rather
than an internal package:

```python
from llmorch.quota import Governor, Ticket, model_spec, provider_spec, quota_manifest
```

`quota_manifest` declares providers and models in keyword arguments named the
way documentation states them — `rpm`, `tpm`, `rpd`, `reserve_requests`,
`account_scoped` — so using the governor does not mean adopting models.yaml, the
role taxonomy, or fallback chains. Role chains are left empty: they belong to
routing, not to accounting, and requiring them would force a caller to invent a
taxonomy they have no use for.

Three tests keep the boundary honest:

- **The README example is executed, not eyeballed.** The first Python block is
  extracted from the file and run. A quickstart that has quietly stopped working
  is worse than none — it is the first thing anyone tries.
- **It runs with no API key and no network**, because `api_key_env` names a
  variable rather than reading one, and quota arithmetic is offline work.
- **Importing `llmorch.quota` must not drag in the orchestrator.** A subprocess
  imports it and asserts that no `engine`, `negotiate` or `demo` module was
  loaded. Someone governing their own calls should not be paying for the
  scheduler.

`llmorch.providers` is exported alongside it — the stdlib-only client and the
header parser — since a governor with nothing to govern is not much use.

---

The original plan — the rationale behind every decision, with an 18-entry
decisions log — is kept outside this repository, alongside the notes it was
drafted in. Everything load-bearing from it has been folded into the sections
above.
