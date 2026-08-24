# llmorch — Project Blueprint & Status

> Self-contained handoff. Everything needed to resume work with no prior context.
> Last updated: end of Milestone 1.

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

**(a) Claude and ChatGPT are not in the roster.**
No Anthropic or OpenAI API key is available, and subscriptions (Claude Pro,
ChatGPT Plus) cannot be called from code. Vendor diversity therefore comes from
Google (Gemini) and the open-model vendors Groq hosts (Meta, Alibaba, Moonshot).
The architecture is provider-agnostic — adding a key later is a manifest entry,
not a rewrite.

**(b) "Save credits" is a quota-scheduling problem, not a cost problem.**
Nothing in the roster bills per token. The scarce resources are rate limits:

| Provider | RPM | TPM | RPD | Reset TZ |
|---|---|---|---|---|
| Groq | 30 | **6,000** | 14,400 | UTC |
| Gemini 2.5 Flash | 10 | 250,000 | **250** | America/Los_Angeles |

Two walls dominate every decision:
- **Groq's 6,000 TPM** caps a single request at ~5,400 tokens. Larger requests
  are *permanently unservable*, not merely delayed.
- **Gemini's 250/day** makes each Gemini request precious.

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

**Milestones 0 and 1 are COMPLETE. 210 tests pass, 1 skipped** (symlink test
needs admin on Windows).

Verified working end to end:

```bash
python -m llmorch run "build a notes app"
```

Splits work across 4 models / 2 vendors, writes `runs/<id>/output/`, and the
generated app genuinely serves: POST creates a note, GET lists, detail fetches,
`/` returns the page. All against mocks — zero network calls.

### Environment
- Python **3.14.1**, venv at `.venv/`
- Installed: PyYAML, pydantic 2.13, tzdata 2026.3, pytest, pytest-asyncio
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
| `quota/store.py` | **NOT BUILT** | SQLite ledger — M2 |
| `providers/base.py` | done | Provider protocol + registry |
| `providers/mock.py` | done | Canned responses + fault injection |
| `providers/openai_compat.py` | **NOT BUILT** | M2 |
| `providers/headers.py` | **NOT BUILT** | M2 |
| `engine/graph.py` | done | Kahn levels, cycle repair, budget pruning |
| `engine/salvage.py` | done | Fence stripping, balanced-JSON recovery |
| `engine/verify.py` | done (Tier 0) | Tier 1 review designed, wired at M4 |
| `engine/health.py` | done | Failover ladder + circuit breaker |
| `engine/worker.py` | done | Executes one node, fails over across vendors |
| `engine/scheduler.py` | done | DAG execution + bulk reassignment |
| `engine/blackboard.py` | done | Summaries only, never whole artifacts |
| `engine/materialize.py` | done | Path safety + artifact writeout |
| `engine/contracts.py` | **NOT BUILT** | Cross-artifact validation — M5 |
| `engine/checkpoint.py` | **NOT BUILT** | Resume across quota days — M2/M3 |
| `negotiate/roles.py` | done | Fixed taxonomy + alias parsing |
| `negotiate/reconcile.py` | done | **The dispatcher** |
| `negotiate/decompose.py` | **NOT BUILT** | M4 |
| `negotiate/bidding.py` | **NOT BUILT** | M4 |
| `negotiate/profiles.py` | **NOT BUILT** | M4 |
| `negotiate/plancache.py` | **NOT BUILT** | M4 |
| `report/render.py` | done | plan/outcome/spend/quota tables |
| `report/ledger.py` | **NOT BUILT** | M2, with store.py |
| `demo/website.py` | done | Notes-app DAG + real working canned artifacts |
| `__main__.py` | done | `run`, `plan`, `quota` |

### Tests (210 passing)

| File | Covers |
|---|---|
| `test_foundations.py` | types, errors, config |
| `test_manifest.py` | manifest loading + validation rules |
| `test_governor.py` | admission control, day rollovers, reserves |
| `test_materialize.py` | salvage + path safety (security-critical) |
| `test_reconcile.py` | graph + dispatcher + bid normalization |
| `test_health_verify.py` | failover, circuit breaker, Tier 0 checks |
| `test_integration.py` | full pipeline, fault injection, degradation |

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

---

## 7. Roadmap

| # | Milestone | Status | Needs |
|---|---|---|---|
| M0 | Setup, secret hygiene | **DONE** | — |
| M1 | Offline skeleton, zero API calls | **DONE** | — |
| M2 | First real requests, **Groq only** | next | GROQ_API_KEY |
| M3 | Add Gemini; cross-vendor failover live | | GEMINI_API_KEY |
| M3.5 | NVIDIA NIM + Mistral (adaptive limit discovery) | | those keys |
| M4 | Negotiation live + Tier 1 cross-vendor review | | — |
| M5 | Optimization; Perplexity behind --allow-paid | | — |
| M6 | Web dashboard (read-only, localhost-only) | | — |

### M2 specifics (the immediate next step)

Build `providers/openai_compat.py`, `providers/headers.py`, `quota/store.py`,
`report/ledger.py`, and an `llmorch doctor` command.

Groq first **deliberately**: 14,400 requests/day makes it the safe place to
debug header parsing and counter sync.

**Unverified:** the Groq wire names in models.yaml
(`llama-3.3-70b-versatile`, `qwen3-32b`, `openai/gpt-oss-120b`) have not been
checked against the live API. `llmorch doctor` should confirm each with one
trivial call before anything depends on them.

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

Then run the generated app (serves on http://localhost:8000):

```bash
.venv/Scripts/python.exe runs/<id>/output/server.py
```

---

## 9. Secrets

`.env` exists with **all keys blank**, gitignored. `.env.example` documents each
provider, its free-tier limits, and its signup URL. `.gitignore` was written
*before* `.env` existed, so no key ever sat in a trackable file.

Milestone 1 needs **no keys at all**.

Full original plan, with the rationale behind every decision and an 18-entry
decisions log:
`C:\Users\prana\.claude\plans\this-is-totally-a-glimmering-snail.md`
