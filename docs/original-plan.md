# Multi-Provider LLM Orchestrator (`llmorch`)

## Context

You want agents from different AI vendors to collaborate on a task: negotiate once about
who is good at what, split the work, then each execute their slice — demoed on "build a
website" (frontend / backend / research split across different models). Goals are even
work distribution, efficiency, and conserving scarce API quota.

Three findings from research reshape the original vision, and the plan reflects them:

**1. Claude and ChatGPT are not in the roster.** You have no Anthropic or OpenAI API key,
and subscriptions (Claude Pro, ChatGPT Plus) cannot be called from code. The literal
"ChatGPT does frontend, Claude does backend" split is not buildable today. Vendor diversity
instead comes from Google (Gemini) and the several open-model vendors Groq hosts (Meta,
Alibaba, Moonshot) — still genuinely multi-company, and the architecture is provider-agnostic,
so adding an Anthropic or OpenAI key, or a vendor like OpenCode Zen, later is a manifest
entry, not a rewrite.

**2. "Save credits" is not a cost problem — it is a quota-scheduling problem.** Nothing in
the default roster is billed per token. The scarce resources are heterogeneous free-tier
rate limits:

| Provider | RPM | TPM | RPD | Reset TZ | Notes |
|---|---|---|---|---|---|
| Groq | 30 | **6,000** | 14,400 | UTC | org-scoped; open models only; very fast |
| Gemini 2.5 Flash | 10 | 250,000 | **250** | **America/Los_Angeles** | 1M context |

**v1 roster is Groq + Gemini only** — you have keys for both today, and both publish exact,
reliably-enforced free-tier numbers, which is what the quota governor needs to trust. NVIDIA
NIM and Mistral are the two confirmed additions for a later milestone (see M3.5 below):

| Provider | Why it's deferred, not v1 |
|---|---|
| NVIDIA NIM | Genuinely the best fit for "different companies" — one key reaches 100+ models each labeled by their real vendor (Mistral, Meta, Microsoft, Qwen, DeepSeek). But NVIDIA publishes no official free-tier numbers; real-world reports put it around 40 RPM with a daily cap that runs out fast and isn't documented. Needs the adaptive-discovery path, not the trusted-numbers path. |
| Mistral (La Plateforme) | A genuinely distinct company/model family (not an open-model host), with huge total volume (~1B tokens/month free). But throughput is ~1–2 requests/minute, which would stall a DAG expecting some concurrency, and the free tier opts your prompts into their training data. |

Perplexity remains manifest-documented-but-inactive, opt-in only, as already decided.

With OpenRouter dropped, requests are abundant (~14.4k/day from Groq). The two real walls
are **Groq's 6,000 TPM** — which makes any large-context codegen call *permanently
unservable*, not merely delayed — and **Gemini's 250 RPD**, which makes Gemini the
precious long-context resource. The design optimizes against those two, not against dollars.

**3. Free models are unreliable self-reporters.** Asked to rate themselves, they all claim
high competence at everything, and a Llama model has no knowledge of what Qwen is good at.
So negotiation produces *bids*, and a deterministic Python scoring pass — not an LLM —
makes the final assignment. This is also what makes "split evenly" real: evenness is a
hard capacity constraint in an assignment algorithm, not an instruction in a prompt.

**Demo target:** a small notes app — a home/list page, a note-detail page, and a SQLite-backed
API (create/read/list notes), no auth. Small enough to fit Groq's 6,000 TPM per node, real
enough to need an actual schema and actual page-to-API wiring, so the frontend/backend/content
split is genuine rather than cosmetic.

**The stack is pinned, not model-chosen.** Plain HTML/CSS/JS frontend (no framework, no build
step), a Python `http.server`-based JSON API, and a SQLite file. The decompose step fills in
*routes and data models* but never picks the stack. Two reasons this is architectural rather
than cosmetic: the deterministic contract-check can only parse `fetch()` calls and route
declarations if it knows what it is parsing, and per-role prompts can only be tuned if the
target is fixed. A moving stack would also let two runs produce mutually incompatible output.

**Scaling up to a "proper website" later is a config change, not a rewrite.** The notes app is
a proving ground; the orchestrator itself is task-agnostic and takes arbitrary task text. When
you want a real site, only two components care about the stack — the decompose prompt and
`contracts.py` — so the path is either swapping the pin to a richer target (React + FastAPI,
say) or promoting `stack` to a run-level config option with one profile per supported stack.
Keeping the pin *now* is what keeps M1 verifiable by eye; keeping it *swappable* is why the
demo choice does not lock in the project's ceiling.

**Intended outcome:** `llmorch run "build a notes app"` produces a plan showing which model
got which slice and why, executes the DAG within quota, **writes a real runnable project
folder to disk**, and reports exactly what was spent where — with the entire system runnable
offline against fakes before a single real request is spent.

---

## Scope of this plan

Milestone 0 (setup) and Milestone 1 (below) are the deliverable: **the full engine running
end-to-end against a mock provider, with zero API calls.** This is deliberate — you cannot
debug admission control by spending requests when Gemini gives you 250/day. Milestones 2–6
are sketched for direction but are not this build.

---

## Architecture

```
task text
   │
   ├─ decompose ────────► task DAG + shared `interface` contract   (1 request)
   │
   ├─ bid ──────────────► each model rates itself per node          (N ≤ 4 requests)
   │
   ├─ reconcile ────────► feasibility → normalize → score → assign  (0 requests, pure Python)
   │
   ├─ execute ──────────► DAG scheduler, gated by the quota governor
   │
   └─ materialize ──────► artifacts → runs/<id>/output/ real project folder  (0 requests)
```

### `reconcile` is the dispatcher — the "middle man"

This step is the heart of the system, so it is worth naming plainly: **`reconcile.py` is the
manager that decides which AI gets which job.** It is deterministic Python, not an LLM, and it
draws on four sources of knowledge about each model:

| Source | What it contributes | Where it lives |
|---|---|---|
| **Capability sheet** | Hand-written priors: Gemini is strong at research (1M context, web grounding), Llama-70B is strong at backend logic, weak at research | `role_affinity` in `models.yaml` |
| **Self-reported bids** | What each model claims it is good at *for this specific task* — z-normalized so bragging gains nothing | `bidding.py` → `reconcile.py` |
| **Track record** | How each model has actually performed per role across past runs (EWMA) | `profiles.json` |
| **Live quota** | How much budget the model has left today; near its wall → deprioritized automatically | `governor.headroom()` |

Worked example, the user's own case: a `research` node scores highest for **Gemini** (0.90
research affinity, huge context, search grounding), while a `backend` node scores highest for
**Groq's Llama-3.3-70B** (0.70 backend affinity) — which is also the quota-correct outcome,
since it preserves Gemini's scarce 250/day for work that genuinely needs the long context.

Choosing code over an LLM manager here is deliberate: it costs zero requests, runs instantly,
is fully testable offline, cannot hallucinate an assignment that violates a rate limit, and
`llmorch plan --explain` can show the exact arithmetic behind every decision.

### The `interface` contract keeps the outputs compatible

Decomposition emits a shared spec (routes, data models) that every downstream node receives
verbatim. That is how a frontend from model A and a backend from model B end up compatible
*without the models ever talking to each other* — which is what makes "negotiate once, then
route" work at all, and what keeps it cheap.

---

## File layout

```
pyproject.toml
models.yaml                     # provider/model manifest
models.local.yaml               # gitignored per-machine overrides
src/llmorch/
  types.py                      # all core dataclasses; imports nothing internal
  config.py                     # paths, env loading, RunConfig
  errors.py                     # Unservable, QuotaExhausted, SchemaInvalid
  __main__.py                   # CLI: run | plan | quota | report | resume | doctor
  registry/manifest.py          # YAML -> validated models; overlay merge
  providers/
    base.py                     # Provider protocol, ChatRequest/ChatResponse/Usage
    mock.py                     # deterministic fake — Milestone 1's only provider
    openai_compat.py            # Groq, Gemini shim [M2]; NVIDIA NIM, Mistral [M3.5]; Perplexity [M5]
    gemini_native.py            # thinking_budget=0, caching, free count_tokens [M4]
    headers.py                  # x-ratelimit-* / retry-after -> RateLimitSnapshot
  quota/
    limits.py, windows.py       # LimitSpec; SlidingWindow, DayCounter
    governor.py                 # ★ admission control
    estimator.py                # char-based estimate + EWMA self-calibration
    store.py                    # sqlite3 ledger + day counters
  negotiate/
    roles.py                    # FIXED role taxonomy
    decompose.py, bidding.py
    reconcile.py                # ★ assignment algorithm
    profiles.py, plancache.py   # caches that make negotiation amortized
  engine/
    graph.py, scheduler.py, worker.py
    blackboard.py               # artifacts + summaries
    health.py                   # ★ circuit breaker + cross-vendor failover
    verify.py                   # ★ Tier-0 static checks + cross-vendor LLM review
    materialize.py              # ★ artifacts -> real project folder on disk
    contracts.py                # deterministic cross-artifact validation
    salvage.py                  # lenient JSON/code-fence extraction
    checkpoint.py               # resume across a quota-day boundary
  report/ledger.py, report/render.py
  demo/website.py               # notes-app task + golden DAG for offline dev
tests/
```

**Python 3.11+ required** — for `zoneinfo`, `tomllib`, and modern typing syntax. Verify before
starting: `python --version`.

**Dependencies, kept small:** `PyYAML`, `pydantic`, `tzdata`, plus `openai` and
`google-genai` from Milestone 2. Dev: `pytest`, `pytest-asyncio`.

`tzdata` is **required, not optional** — Windows ships no timezone database, and
`zoneinfo.ZoneInfo("America/Los_Angeles")` for Gemini's Pacific-midnight reset raises
`ZoneInfoNotFoundError` without it.

Deliberately avoided: `langchain` / `litellm` (they abstract away exactly the rate-limit
semantics this project is about), `tiktoken` (wrong tokenizer for every model here),
`scipy` (the assignment solver is ~60 lines).

---

## Component detail

### The quota governor (`quota/governor.py`) — build this first

Every limit is one generic type, so there is a single code path:

```python
LimitSpec(kind: rpm|tpm|rpd|cost_usd_run|requests_per_run,
          scope: account|model, value: int, reserve: int, reset_tz: str)
```

```python
class Admission(Enum):
    GRANTED, WAIT, EXHAUSTED_TODAY, UNSERVABLE, COST_BLOCKED

class Governor:
    def try_acquire(model_id, est_prompt, est_completion, priority) -> Ticket | Denial
    async def acquire(..., deadline: float) -> Ticket
    def wait_time(model_id, est_tokens) -> float | None      # None => never today
    def commit(ticket, usage: Usage, cost: Decimal)          # reconcile est -> actual
    def release(ticket, reason)                              # refund on transport failure
    def sync_from_headers(provider, snap: RateLimitSnapshot) # server is authoritative
    def headroom() -> dict[str, Headroom]
```

Five semantics that make it sound:

1. **`UNSERVABLE` is distinct from `WAIT`.** If `est_prompt + max_tokens > tpm_limit`, the
   model can never serve that node — waiting is pointless and hangs the scheduler. Against
   Groq's 6,000 TPM this fires constantly and is *the* correctness-critical distinction in
   this build. The manifest derives `max_request_tokens = 0.9 × tpm` automatically.
2. **`max_tokens` is always set explicitly on every request.** This turns the completion
   side of the estimate from a guess into a hard upper bound, which is what makes admission
   control sound rather than hopeful. Non-negotiable.
3. **Reserve on estimate, reconcile on commit.** True token counts only exist after the
   response. Reserve `(est_prompt + max_tokens) × 1.25`; at commit, swap the reserved TPM
   entry for the actual. The reservation is what makes concurrent fan-out safe.
4. **Monotonic clock for RPM/TPM sliding windows; wall clock for RPD day boundaries. Never
   mixed.** If the laptop suspends, monotonic windows return stale-but-conservative, which
   is the safe direction.
5. **Response headers override local counters.** Local counting is an estimate;
   `x-ratelimit-remaining-*` and `retry-after` are ground truth and cost nothing to read.
   Sync on every response including 429s. This is also how bonus vendors with undocumented
   limits (NVIDIA NIM, Mistral — M3.5) get discovered — start conservative (10 RPM / 100 RPD
   assumed), then let observed headers and 429s widen or tighten it, persisting what is
   learned.

Data structures: RPM as `deque[float]` of monotonic stamps; TPM as `deque[(ts, tokens)]`
with an incrementally maintained running sum so admission is O(evicted) not O(n); RPD as an
int plus a lazily recomputed `next_reset_utc` derived per provider's own `reset_tz`.

**Persistence:** SQLite at `%LOCALAPPDATA%\llmorch\state.db`, WAL mode, stdlib `sqlite3`.
A `usage_events` table is the single source of truth, and counters are *derived from it* on
startup rather than stored redundantly — a separate counter table drifts from reality after
a crash. A small `day_counters` table exists only as a cross-process reservation lock under
`BEGIN IMMEDIATE`, so two concurrent runs cannot both believe Gemini has 250 requests left.

**Token estimation:** `ceil(len(text) / 3.6)` with a per-provider correction factor, then
self-calibrated — every commit records `actual/est` into a per-provider EWMA (α=0.2). After
~20 calls it lands within a few percent, with no dependencies and no extra requests.

### Manifest (`models.yaml`)

YAML rather than Python constants, because these are *operational facts about this machine*,
not code: NVIDIA NIM's and Mistral's rosters/limits (added in M3.5) are undocumented and will
be learned at runtime, and Groq's per-model limits change. A gitignored `models.local.yaml`
overlay holds per-machine values. Loaded into frozen pydantic models for validation.

```yaml
providers:
  groq:
    kind: openai_compat
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
    reset_tz: UTC
    limits:
      - {kind: rpm, scope: account, value: 30}    # org-scoped, not per-key
      - {kind: tpm, scope: model,   value: 6000}
      - {kind: rpd, scope: model,   value: 14400}
  gemini:
    reset_tz: America/Los_Angeles                 # NOT UTC
    limits:
      - {kind: rpm, scope: model, value: 10}
      - {kind: tpm, scope: model, value: 250000}
      - {kind: rpd, scope: model, value: 250, reserve: 30}
  perplexity:
    paid: true                                    # requires --allow-paid
    cost: {input_per_mtok: 1.0, output_per_mtok: 1.0, per_request: 0.009}
    limits: [{kind: requests_per_run, scope: account, value: 2}]

models:
  - id: groq/llama-3.3-70b
    wire_name: llama-3.3-70b-versatile
    context: 131072
    max_output: 8192
    quality_prior: 0.72
    role_affinity: {frontend: 0.45, backend: 0.70, research: 0.25, review: 0.75}
```

`scope: account | model` and `reserve` carry disproportionate weight. `reserve` withholds
headroom from normal-priority admission so a runaway fan-out cannot consume the last Gemini
requests needed for retries on the critical path.

### Negotiation — 1 + N requests worst case, 0 when cached

- **Decompose (1 request).** Best long-context model (Gemini) emits the `interface` contract
  plus nodes `{id, title, role, spec, deps, output_kind, est_output_tokens, split_hint}`.
  `role` must come from the **fixed taxonomy** in `roles.py` (`planning, research, backend,
  frontend, styling, content, review, integration`) — free-text roles would make the profile
  cache useless across tasks.
- **Budget prune (0 requests).** Enforce `max_nodes` (default 10) *before* bidding, merging
  same-role leaves, so the bid prompt stays small.
- **Bid (N ≤ 4 parallel).** Each bidder sees only the role taxonomy and node `id + title +
  one line` — never full specs, never the interface. ~300–500 prompt tokens, `max_tokens=400`.
  This short-prompt/short-output shape is exactly what Groq's 6,000 TPM handles well, so
  bidding runs there almost free while Gemini's scarce RPD is reserved for execution.
- **Reconcile (0 requests, pure Python)** — where "split evenly" actually gets satisfied:
  1. **Hard feasibility filter** — drop pairs where the node cannot fit the model's context,
     `max_output`, or the provider's `max_request_tokens`. Eliminated before scoring, not penalized.
  2. **Within-bidder z-score normalization** — the counter to overclaiming. A model returning
     0.95 across the board contributes a flat, uninformative signal instead of dominating.
     If a bidder's confidence stdev < 0.05, discard its bids and fall back to priors.
  3. **Score:** `0.35·z_conf + 0.25·role_affinity + 0.15·profile_ewma + 0.15·quality_prior
     − 0.10·quota_pressure`, where `quota_pressure = est_tokens / remaining_budget(model)`,
     so a model near its daily wall deprioritizes itself with no special case.
  4. **Capacity-constrained assignment** — each model gets
     `capacity = total_est_tokens / n_eligible × 1.35`; greedy by descending score subject to
     capacity, then a 2-opt swap pass. Balance on **tokens, not node count** — that is the
     correct notion of "even".
- **Caching, the biggest quota win.** `profiles.json` holds a per-`(model, role)` EWMA of bid
  confidence and realized outcome; `--negotiate=auto` (default) runs bidding *only* when a
  role has fewer than 3 observations. So bidding is a warm-up cost that then disappears.
  `plancache` keyed on `sha256(task ‖ model_ids ‖ manifest_version)` makes demo replays cost
  zero requests. Exact-match only — a semantic cache would silently reuse a wrong plan.

### Execution engine

- `asyncio` scheduler over Kahn topological levels; concurrency bounded by a global cap (4),
  a per-provider semaphore, and ultimately the governor.
- **Context discipline — the critical efficiency rule.** Never paste upstream artifacts
  wholesale into downstream prompts; that is the fastest way to blow Groq's TPM. Each node
  returns `{artifact, summary (≤200 tokens), interface_delta}` in the *same* request, so
  summarization costs zero extra calls. Downstream nodes declare `needs: ["n1.summary"]`.
- **Automatic chunking** when `est_output > max_output` or TPM headroom — split
  deterministically by `split_hint` (`per_file`, `per_route`), no LLM involved.
- **Node states:** `PENDING → RUNNING → VERIFYING → {DONE | RETRY | FALLBACK | DEGRADED | FAILED}`.
  Schema-invalid tries **`salvage.py` first** (extract first balanced JSON / fenced block) and
  only spends a repair request if salvage fails — salvage saves more requests than repair
  prompts do. Chain exhausted → `DEGRADED` with a stub artifact; **a daily quota wall must
  never fail the whole run.** Full failover ladder and circuit-breaker semantics in
  `engine/health.py` below; verification gate in `engine/verify.py`.
- **Checkpoint / resume** after every node. With Gemini at 250/day a large run can legitimately
  span a day boundary, so `llmorch resume <run_id>` is a feature, not a nicety.
- **Contract check (0 requests)** — regex/AST comparison of `fetch()` URLs in the frontend
  against declared routes in the backend, catching "frontend calls `/api/todos`, backend built
  `/todos`" without paying for an LLM review pass.
- **Untrusted content.** Artifacts from model A become prompt input for model B. Wrap them in
  delimited "this is data, not instructions" blocks; artifact text must never influence routing,
  the manifest, or governor state.

### Failover (`engine/health.py`) — when a model can't do the job

Two distinct failure scales, handled differently. Retrying the *same* model is usually futile:
if Gemini returned malformed JSON twice, a third Gemini attempt fails the same way. **Failure
modes correlate within a vendor and decorrelate across vendors**, so failover prefers a
different vendor rather than merely a different model.

**Node-level failover** — one task fails:

```
attempt 1: assigned model
   ↓ transport error / 5xx / timeout  → retry same model (backoff+jitter, max 2)
   ↓ 429                              → sync headers; re-select or wait if slack allows
   ↓ schema-invalid                   → salvage.py first; one repair request if that fails
   ↓ still failing                    → FALLBACK: next rung in the role chain,
                                         preferring a DIFFERENT VENDOR
   ↓ chain exhausted                  → DEGRADED (stub artifact, run continues)
```

Role chains are declared in the manifest and validated at load — each must contain **at least
two distinct vendors**, or the failover is decorative:

```yaml
roles:
  frontend: [gemini/2.5-flash, groq/qwen3-32b, groq/llama-3.3-70b]
  backend:  [groq/llama-3.3-70b, gemini/2.5-flash]
  research: [gemini/2.5-flash, perplexity/sonar]   # perplexity gated by --allow-paid
```

**Model-level failover — the "hand the whole thing over" case.** A per-model circuit breaker
tracks consecutive hard failures within a run. After **2**, the model is marked `UNHEALTHY`
for the remainder of the run, and — this is the part that matters — **its still-pending nodes
are re-run through `reconcile.py` with that model excluded**, so the work is redistributed
wholesale rather than failing one node at a time. Guards that keep this from thrashing:

- Reassignment happens **once per model**; a model that circuit-breaks does not get re-tried.
- The replacement must satisfy the same feasibility filter (context, `max_output`, TPM) — a
  model cannot inherit work it physically cannot serve.
- Reassignment respects live quota. If no healthy model can take the work today, the nodes go
  `DEGRADED` and `llmorch resume` picks them up after the quota resets, rather than burning
  the remaining budget on doomed attempts.
- `EXHAUSTED_TODAY` is **not** a health failure — running out of quota is not the model being
  broken. It marks the model unavailable for the run without touching its track record.

Failure outcomes feed `profiles.json`, so a model that keeps failing at a role gradually stops
being assigned that role in future runs. Failover is not just recovery — it is how the system
learns.

### Verification (`engine/verify.py`) — automatic evaluation of generated code

Two tiers, cheapest first, because most defects are catchable without spending a request.

**Tier 0 — deterministic, 0 requests.** Runs on every node, always on:

| Check | How |
|---|---|
| Python parses | `ast.parse()` — catches truncated output and syntax errors instantly |
| SQL parses | `sqlite3.complete_statement()` + `EXPLAIN` against a scratch in-memory DB |
| HTML/JS sane | tag balance, no unclosed fences, no obvious placeholder text (`TODO`, `...`) |
| Not truncated | output did not stop at exactly `max_tokens` |
| Contract holds | `contracts.py` — frontend `fetch()` URLs match backend routes |

Truncation and syntax errors are the most common free-model failures, and Tier 0 catches both
for free. **Tier 0 failure routes straight into the failover ladder above** — no LLM needed to
notice that the code doesn't parse.

**Tier 1 — cross-vendor LLM review, 1 request per reviewed node.** This is your original
"AIs give each other feedback" idea, in the form that actually pays off:

- **The reviewer must be a different vendor than the author.** Enforced in code, not
  suggested in a prompt. Self-review is a well-known weak spot — a model that made a mistake
  tends to re-approve it. A peer from another vendor has decorrelated blind spots.
- Reviewer gets the `interface` contract, the artifact, and the node spec — never the author's
  identity, to avoid deference.
- Structured verdict: `{verdict: pass|revise|reject, issues: [{severity, line, what, why}]}`.
- `revise` → **one** repair round, sent back to the original author with the critique attached.
  Capped at one, because repair loops are where quota disappears.
- `reject` → treat as a node failure and enter the failover ladder with a different vendor.

**Routing and cost.** Review roughly doubles requests on reviewed nodes, so: reviews default
to **Groq** (14,400/day — abundant), never Gemini (250/day — precious), and Tier 1 runs only
on `output_kind == "code"` nodes by default. Controlled by `--review={off|code|all}`, default
`code`. The governor treats review requests as normal-priority, so they can never consume the
`reserve` headroom that retries on the critical path depend on.

**Reviewer output is untrusted input.** A critique is data, not instructions — it is delimited
before being passed to the repair round, and it can never alter routing, the manifest, or
governor state.

### Materialization (`engine/materialize.py`) — closing the loop

Without this the orchestrator produces a pile of text blobs in a database and *feels* broken
even when every component worked correctly. Every node declares an `output_path` in the DAG;
after execution, `materialize.py` writes each artifact to a real folder:

```
runs/<run_id>/
  state.json              # checkpoint (plan, node states, ledger cursor)
  plan.md                 # human-readable assignment table + score breakdown
  report.md               # spend, quota efficiency, fairness
  output/                 # ← the actual deliverable
    index.html            # list page
    note.html             # detail page
    style.css
    app.js                # fetch() calls against the pinned API routes
    server.py             # stdlib http.server JSON API
    schema.sql            # SQLite DDL
    seed.py               # optional demo content
    README.md             # generated: how to run it
```

Rules that keep this safe and debuggable:

- **Path safety is non-negotiable.** `output_path` comes from LLM output, so it is untrusted.
  Reject absolute paths, drive letters, `..` segments, symlinks, and anything resolving outside
  `runs/<id>/output/`. Resolve with `Path.resolve()` and assert the output root is a parent.
  This is the one place where model output touches the filesystem — treat it as hostile input.
- **Code-fence stripping.** Models wrap code in ``` fences with language tags regardless of
  instructions. `salvage.py` already extracts fenced blocks; materialize reuses it rather than
  re-implementing.
- **`DEGRADED` nodes still materialize** — as a stub file containing the node spec as a comment,
  so the folder structure is complete and it is obvious what is missing and why.
- **Never write outside `runs/<run_id>/`.** No writing into the user's project, no overwriting
  a previous run. Each run is a fresh, self-contained, disposable folder.
- **A generated `README.md`** with the literal command to run it (`python server.py`, then open
  `http://localhost:8000`), so "did it work?" takes one command to answer.

`llmorch run` prints the output path on completion. Verification is then human and immediate:
open the folder, run the server, look at the page.

### Reporting

`usage_events` is the only source of truth. `llmorch quota` shows per-provider headroom and
time-to-reset *in each provider's own timezone*. `llmorch report <run_id>` gives per-node and
per-provider spend, estimator calibration error, assignment fairness (token share vs. ideal
even split), and the headline metric — **quota efficiency = useful output tokens ÷ total
tokens spent**, where retries, repairs, and degraded nodes count as waste.
`llmorch plan --explain` shows the score decomposition so you can see *why* a model won a node.

---

## Milestone 0 — project setup and secret hygiene

Small, but done first so keys never land somewhere they can leak.

1. **Verify Python 3.11+** — `python --version`.
2. **`.gitignore`** — must exclude `.env`, `runs/`, `*.db` (+ `-wal`/`-shm`), `models.local.yaml`,
   `tmp/`, `__pycache__/`, `.venv/`. Written *before* any key file exists, so there is never a
   window where a real key sits in a trackable file.
3. **`.env.example`** — committed template listing every key name with blank values and a short
   note on each provider's free-tier limits and signup URL. Real keys never go in this file.
4. **`.env`** — the user's actual keys, gitignored. Blank until needed: **Milestone 1 requires no
   keys at all**, since it runs entirely against the mock provider.
5. **`pyproject.toml`** + a virtualenv.

Key names, matching `api_key_env` in `models.yaml`:

| Variable | Milestone | Notes |
|---|---|---|
| `GROQ_API_KEY` | M2 | Free, no card — console.groq.com/keys |
| `GEMINI_API_KEY` | M3 | Free tier — aistudio.google.com/apikey |
| `NVIDIA_NIM_API_KEY` | M3.5 | Free, no card — build.nvidia.com |
| `MISTRAL_API_KEY` | M3.5 | Free tier, phone verification; opts prompts into training |
| `PERPLEXITY_API_KEY` | M5 | **Paid.** Also requires `--allow-paid` at runtime |
| `LLMORCH_MAX_USD` | — | Hard spend ceiling per run. Default `0.00` = paid providers off |

`config.py` loads `.env` with a ~12-line stdlib parser (no `python-dotenv`), and **must never
log, print, or write a key value** — not in the ledger, not in `report.md`, not in errors. The
ledger records provider and model names only.

---

## Milestone 1 — offline skeleton, zero API calls

Build, in order:

1. `types.py`, `errors.py`, `config.py`
2. `registry/manifest.py` + `models.yaml` (Groq + Gemini active; NIM/Mistral/Perplexity
   present as inactive entries so the schema is proven against real shapes — none called)
3. `quota/` complete — `limits.py`, `windows.py`, `estimator.py`, `store.py`, `governor.py`,
   **with an injectable clock** so tests control time
4. `providers/base.py` + `providers/mock.py` — deterministic canned artifacts and simulated
   token counts
5. `engine/graph.py`, `scheduler.py`, `worker.py`, `blackboard.py`, `salvage.py`, `checkpoint.py`
6. `engine/health.py` — failover ladder + per-model circuit breaker + bulk reassignment
7. `engine/verify.py` — **Tier 0 only** in M1 (`ast.parse`, SQL parse, truncation, contract
   check). Tier 1 cross-vendor review is designed now but wired at M4, since it needs two
   live vendors to be meaningful.
8. `engine/materialize.py` — path-safety validation + artifact writeout
9. `negotiate/roles.py`, `reconcile.py` — the assignment algorithm, testable with fixture bids
10. `demo/website.py` with a hand-written golden DAG for the notes app — the pinned stack
    (`index.html`, `note.html`, `style.css`, `app.js`, `server.py`, `schema.sql`), each node
    carrying its `output_path`
11. `report/` + `__main__.py`

The mock provider gains a **fault-injection mode** (`MockProvider(fail_nodes=..., fail_mode=...)`)
so failover and verification are testable offline: it can return malformed JSON, truncated
output, code that does not parse, or transport errors on demand. Failover logic that has never
been exercised is failover logic that does not work.

**Exit criterion:** `llmorch run --dry-run "build a notes app"` executes the full DAG
against the mock — schema design, API, list page, detail page, content/copy — writes a
complete project folder to `runs/<id>/output/`, prints the assignment table with score
breakdowns, and prints a spend report — having made zero network calls. Because the mock
returns canned-but-valid artifacts for the pinned stack, **the materialized folder actually
runs**: `python runs/<id>/output/server.py` serves a working notes app. That makes the whole
pipeline verifiable by eye before a single real token is spent.

The governor and the mock come before any real provider deliberately: every later decision
depends on admission control being correct, and it is the one component that cannot be
debugged by spending requests.

### Verification

```bash
pytest -q
```

Tests that must pass, all with a fake clock and no network:

- **Governor:** RPM window eviction; TPM reserve-then-reconcile; RPD rollover across
  **Pacific midnight** for Gemini and **UTC midnight** for Groq in the same run; `UNSERVABLE`
  returned (not `WAIT`) when `est_prompt + max_tokens > 6000` on Groq; `reserve` blocking
  normal-priority while admitting high-priority; concurrent `acquire` not oversubscribing;
  refund on `release`; header sync overriding local counters.
- **Reconcile:** a bidder returning 0.95 on everything does not dominate; infeasible pairs
  are filtered before scoring; token-share imbalance stays within tolerance; 2-opt never
  violates capacity.
- **Graph/scheduler:** dependency order respected; cycle detected and repaired; chunking
  triggers on oversized nodes; a `DEGRADED` node does not fail the run; checkpoint→resume
  reproduces state.
- **Estimator:** EWMA converges on synthetic actual/est pairs.
- **Salvage:** recovers JSON from fenced blocks, leading prose, and trailing commentary.
- **Failover (`health.py`)** — using the mock's fault-injection mode: a failing node advances
  to the next chain rung and **lands on a different vendor**; 2 consecutive failures trip the
  circuit breaker; a circuit-broken model's pending nodes are **bulk-reassigned** via reconcile;
  a model is never reassigned work it cannot physically serve; reassignment happens at most once
  per model (no thrashing); `EXHAUSTED_TODAY` does **not** count as a health failure; when no
  healthy model remains, nodes go `DEGRADED` rather than burning the remaining quota.
- **Verify Tier 0:** unparseable Python, invalid SQL, and output truncated at exactly
  `max_tokens` are each caught with zero requests and routed into the failover ladder.
- **Materialize (security-critical):** rejects `../../etc/passwd`, `C:\Windows\...`, absolute
  POSIX paths, and symlinked escapes; strips code fences; writes stubs for `DEGRADED` nodes;
  never writes outside the run's `output/` root.

Then, manually:

```bash
python -m llmorch run --dry-run "build a notes app"
python -m llmorch plan --explain "build a notes app"
python -m llmorch quota
```

Then confirm the generated app actually runs:

```bash
python runs/latest/output/server.py
```

---

## Later milestones (direction only, not this build)

- **M2 — first real requests, Groq only.** `openai_compat.py` + `headers.py` + ledger wiring.
  Groq's 14,400/day makes it the safe place to burn requests while debugging. `llmorch doctor`
  sends one trivial call and verifies header parsing and counter sync.
- **M3 — multi-provider (Groq + Gemini).** Gemini via its OpenAI-compat endpoint, role
  fallback chains and the circuit breaker now operating across **two real vendors** (the first
  point at which cross-vendor failover is genuinely exercised), retry state machine. Real
  website build with a static plan, still no negotiation.
- **M3.5 — bonus vendors, adaptive discovery.** Add NVIDIA NIM and Mistral through the
  same undocumented-limits path: start conservative (e.g. assume 10 RPM / 100 RPD), widen
  or tighten from observed response headers and 429s, persist what's learned per provider.
  Neither becomes load-bearing for daily volume — they add genuine vendor variety on top of
  the Groq/Gemini backbone, gated so their unreliable limits can never stall the run
  (fallback chains always have a Groq or Gemini rung beneath them).
- **M4 — negotiation live + Tier 1 cross-vendor review.** `decompose.py`, `bidding.py`,
  `profiles.py`, `plancache.py`, and `verify.py`'s LLM review tier — which needs two live
  vendors to be meaningful, hence its placement here. Negotiation comes last deliberately, so
  a broken negotiation cannot burn quota discovering a broken executor.
- **M5 — optimization.** `gemini_native.py` (`thinking_budget=0` matters — Gemini 2.5 Flash
  thinks by default and silently inflates output and TPM), explicit context caching, free
  `count_tokens`, `contracts.py` checks, Perplexity behind `--allow-paid`.

- **M6 — web dashboard.** A local UI for watching the orchestrator work, deliberately built
  only after the engine is proven. **Until this lands (M1–M5), all output is terminal text
  plus generated `report.md` / `plan.md` files — accept that the early milestones look plain.**

  The dashboard is mostly *presentation of data that already exists*, not new plumbing — the
  SQLite ledger holds every token spent, `state.json` holds the plan and live node states, and
  `reconcile.py` already emits the score breakdown. Views worth building:

  | View | Shows | Source |
  |---|---|---|
  | **Live DAG** | Nodes lighting up as models finish; colour-coded by assigned vendor; `DEGRADED` nodes flagged | `state.json` |
  | **Assignment table** | Which AI won which job and the exact scoring math behind it | `reconcile.py` output |
  | **Quota gauges** | Per-provider headroom, TPM pressure, time-to-reset in each provider's own timezone | `governor.headroom()` |
  | **Spend timeline** | Tokens/requests over the run; retries and repairs marked as waste | `usage_events` |
  | **Run history** | Past runs, quota efficiency trend, per-model track record | ledger + `profiles.json` |

  Stack: stdlib `http.server` + Server-Sent Events for live updates + vanilla JS — zero new
  dependencies, consistent with the rest of the project, and sufficient for a local read-only
  dashboard. FastAPI + uvicorn is the fallback if SSE-over-stdlib proves awkward.

  Two constraints: the dashboard is **read-only** (it observes runs, never launches or mutates
  them, so it cannot corrupt governor state), and it **binds to localhost only** — it exposes
  API-key-adjacent quota data and must never be reachable off the machine.

Out of scope throughout: streaming, tool-calling inside worker nodes, multi-round negotiation,
semantic plan caching, and actually serving the generated site.

---

## Known risks

| Risk | Mitigation |
|---|---|
| Groq 6,000 TPM makes long codegen permanently unservable | `UNSERVABLE` verdict distinct from `WAIT`; `max_request_tokens = 0.9×tpm`; Groq restricted by role affinity to short tasks; Gemini carries long-context work |
| Gemini 250 RPD is the tightest real budget | `reserve: 30`; bidding routed to Groq; plan + profile caching; checkpoint/resume across days |
| NVIDIA NIM / Mistral limits undocumented or throttled (M3.5) | Start conservative, learn from headers and 429s, persist what is discovered; never the only rung in a fallback chain |
| Gemini resets at Pacific midnight, Groq at UTC | Per-provider `reset_tz` + `zoneinfo`; **`tzdata` is a hard dependency on Windows** |
| Suspend/DST corrupting windows | Monotonic for RPM/TPM, wall clock for RPD, never mixed |
| Two concurrent runs double-spend one key | `BEGIN IMMEDIATE` day-counter reservation in SQLite |
| Free models 400 on `response_format` | Per-model capability flags, one-time detection, persisted downgrade, salvage parser |
| Degenerate auction (all bids ~0.95) | Within-bidder z-scoring; discard bidders with stdev < 0.05 |
| Incompatible artifacts across vendors | Shared `interface` contract + deterministic `contract_check` (0 requests) |
| Perplexity priced **per request** ($5–14/1k), not just per token | `cost.per_request` in manifest; `requests_per_run: 2`; gated behind `--allow-paid`; research defaults to Gemini grounding |
| Prompt injection via artifact chaining | Delimited data blocks; artifacts never influence routing or governor state |
| **API key leaking into a committed file or a report** | `.gitignore` written before `.env` exists; keys never logged, printed, or persisted; ledger stores provider/model names only; `.env.example` holds blank placeholders |
| **Path traversal via model-chosen `output_path`** | Materialize rejects absolute paths, drive letters, `..`, and symlinks; `Path.resolve()` must stay under the run's `output/` root; writes never escape `runs/<id>/` |
| Generated folder looks complete but does not run | Mock returns valid pinned-stack artifacts, so M1's exit criterion is a *runnable* app; generated `README.md` gives the exact run command |
| **Retrying a failed model on the same vendor fails identically** | Failure modes correlate within a vendor; fallback chains must contain ≥2 distinct vendors, validated at manifest load |
| **Circuit-breaker thrashing / reassignment loops** | Reassign at most once per model; feasibility re-checked before inheriting work; `EXHAUSTED_TODAY` never counts as a health failure |
| **Review loop burns the daily budget** | Tier 0 is free and catches most defects; Tier 1 capped at one repair round, defaults to code nodes only, and routes to Groq (14,400/day) never Gemini (250/day) |
| **Self-review rubber-stamps bad code** | Reviewer vendor ≠ author vendor, enforced in code; reviewer never told who wrote the artifact |
| **Reviewer critique used as an injection vector** | Critiques are delimited untrusted data; they cannot alter routing, the manifest, or governor state |

---

## Decisions log

Every settled choice and the reason, so none of it has to be re-litigated later.

| # | Decision | Why |
|---|---|---|
| 1 | **Coordination = negotiate once, then route** | Free-form inter-model chatter costs O(rounds × models) requests and is non-deterministic. One bid round + deterministic assignment gets the same benefit at a fraction of the quota. |
| 2 | **v1 providers = Gemini + Groq only** | The only two with exact, reliably-enforced, published free-tier numbers — which is what the governor must be able to trust. |
| 3 | **OpenRouter dropped** | 50 requests/day without a $10 top-up; user declined the top-up, making it unusable as a workhorse. |
| 4 | **NVIDIA NIM + Mistral deferred to M3.5** | Both add real vendor diversity, but NIM publishes no official limits and Mistral runs at ~1–2 req/min and trains on free-tier prompts. Bonus, never load-bearing. |
| 5 | **Perplexity opt-in only** | Genuinely paid ($5–14 per 1k requests on top of tokens); Gemini's free search grounding covers the research role. |
| 6 | **The dispatcher is code, not an LLM** | Zero requests, instant, testable offline, fully explainable, and cannot hallucinate an assignment that violates a rate limit. |
| 7 | **"Even split" = capacity constraint on tokens** | An LLM told to "split evenly" will not. Evenness must be enforced by the assignment algorithm, and measured in tokens rather than node count. |
| 8 | **Bids are z-normalized per bidder** | Free models overclaim uniformly; without normalization the loudest model wins everything. |
| 9 | **Offline mock skeleton before any real provider** | Admission control cannot be debugged by spending requests when Gemini allows 250/day. |
| 10 | **Stack pinned (HTML/JS + stdlib Python + SQLite)** | Contract-checking needs a known target to parse; prompts need a fixed target to tune. Swappable later — see the "proper website" note above. |
| 11 | **Materialization writes a real project folder** | Otherwise a fully-working orchestrator produces only text blobs and *feels* broken. |
| 12 | **Demo = notes app, no auth** | Enough to force a genuine 3-way split (schema/API, pages, content) without auth's security-critical complexity eating scarce quota. |
| 13 | **Web dashboard deferred to M6** | Engine correctness first; M1–M5 output is terminal text plus `report.md`. Read-only and localhost-only when built. |
| 14 | **SQLite ledger is the single source of truth** | Counters derived from events rather than stored separately, so a crash cannot leave them drifting from reality. |
| 15 | **Failover prefers a different vendor** | Failure modes correlate within a vendor — a third Gemini attempt fails the way the first two did. Cross-vendor retry has decorrelated blind spots. |
| 16 | **Circuit breaker reassigns work in bulk** | The user's "hand the whole thing over" case: after 2 consecutive failures a model is unhealthy for the run and its pending nodes go back through `reconcile`, rather than dying one node at a time. |
| 17 | **Verification is two-tier, cheap first** | Truncation and syntax errors are the most common free-model failures and are catchable for 0 requests. Paying an LLM to notice that code does not parse is waste. |
| 18 | **Reviewer must be a different vendor than the author** | Self-review is a known weak spot — a model tends to re-approve its own mistake. Also the form in which "AIs give each other feedback" actually produces value. |
