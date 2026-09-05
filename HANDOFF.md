# llmorch — handoff

**State:** M0–M6 done, plus the smoke run. 527 tests pass, 1 skipped on Windows
(a symlink test needing admin). Published at github.com/Pranay4040/llmorch,
tagged `v0.1.0`.

Two things in one repo:

- **`llmorch.quota`** — a library for rationing calls across LLM providers.
  Admission control, a durable usage ledger, a stdlib-only OpenAI-wire client.
  This is the part with value outside the repo; see README.md.
- **The orchestrator** — plans a task into a DAG, assigns each node by fitness
  and remaining quota, executes across vendors with failover, writes a runnable
  folder.

---

## Run it

```bash
.venv/Scripts/python.exe -m pytest -q                  # 527 tests, no network
.venv/Scripts/python.exe -m llmorch run "build a notes app"        # mock, offline
.venv/Scripts/python.exe -m llmorch run --smoke "<task>"          # ...then run the result
.venv/Scripts/python.exe -m llmorch run --smoke-install "<task>"  # ...installing its deps first
.venv/Scripts/python.exe -m llmorch run --live --providers all "<task>"
.venv/Scripts/python.exe -m llmorch resume <run_id>    # after a quota wall
.venv/Scripts/python.exe -m llmorch doctor --probe     # verify wire names live
.venv/Scripts/python.exe -m llmorch discover           # what each key can reach
.venv/Scripts/python.exe -m llmorch dashboard          # read-only, localhost
```

Keys live in `.env` (gitignored). Nothing above needs one except `--live`,
`--probe` and `discover`.

`.github/workflows/tests.yml` runs the suite on push and pull request across
Linux (3.11, 3.13) and Windows (3.12), then does a full offline demo run with
`--smoke`. Everything it does is offline, so CI needs no secrets and never
spends quota.

## Roster (verified live 2026-09-01)

| Vendor | Models | Real limits |
|---|---|---|
| Groq | gpt-oss-120b, gpt-oss-20b, qwen3-27b | 1,000 req/day, 8,000 TPM — **from its own headers** |
| Gemini | 3.6-flash | 20 req/**minute** — from a 429 body. Daily figure unverified |
| OpenRouter | minimax-m3, nemotron-ultra, north-mini-code | free tier; limits estimated |

---

## Next

Ordered by value. Issues #1–#4 are filed on GitHub.

1. **Arity agreement in JavaScript** (#1, the half still open). Routes, imports
   and imported names now work for JS and Go; what `check_python_calls` does and
   nothing else can is compare *signatures*. `ast` gives Python exact ones, and
   JavaScript defaults, rest parameters and destructured options objects mean a
   regex would report working code as broken. This needs a real parser, which
   means a dependency this project has so far refused — and the missed fault is
   cheaper than the false accusation, so it stays open on purpose.
2. **A compiled build step, if it is ever worth it.** `--smoke-install` covers
   the Node case: a lockfile is enough to install from, and the recipe is
   inferred from which lockfile exists rather than declared by a model. What is
   still not covered is anything needing a *compile* — a Go binary, a TypeScript
   build — and the case for adding it is weaker than it looks: `go run` already
   fetches and compiles, and a build step is where an arbitrary command would
   have to come back in. Worth doing only when a real plan is blocked by it.
3. **Paid providers** (#2). DeepSeek, Moonshot, Fireworks, OpenCode Zen are
   discovered and undeclared. OpenCode Zen fronts Claude and GPT families, so it
   would reopen the roster the way OpenRouter did. Blocked on *verified*
   pricing — the manifest rejects a paid provider with no cost, and inventing
   numbers to satisfy that check defeats it.
4. **Gemini's daily limit** (#4) and **the Mistral/Wafer keys** (#3).

## Do not break these

Each was learned by getting it wrong against a live API.

- **`UNSERVABLE` ≠ `WAIT` ≠ `EXHAUSTED_TODAY`.** Too big to ever fit, busy for
  seconds, and gone until midnight are three different answers. Collapsing any
  two writes off a healthy model.
- **Blaming a model for a constraint the system imposed is the recurring bug.**
  It happened three times: a per-minute wait recorded as daily exhaustion, a
  rate limit counted toward the circuit breaker, and a reply cut off at too
  small a budget then failed over *with the same budget*. Truncation now grows
  the budget and carries no health penalty.
- **A stated wait outranks any keyword.** A 429 body may list several quota ids,
  some per-day, while the metric that tripped is per-minute. If the server says
  come back in 12s, waiting works.
- **Headers are fact; local counting is inference.** Published limits were wrong
  in both directions for Groq.
- **A model list is not an entitlement.** Cerebras and NVIDIA NIM both answer
  `GET /models` and refuse every completion. `discover` proves a URL and a key;
  only `doctor --probe` proves a model.
- **Correctness is relative to the runtime.** `InterfaceContract.runtime` carries
  OS, interpreter, working directory and launch command to every node. Without
  it a reviewer passed a file that resolved every page to the drive root.
- **A reviewer never shares the author's vendor**, review is advisory and never
  fatal, and repairs are capped at one per node.
- **Which files are the backend is the portable part of a route check, not the
  matching.** A route literal looks the same in Python, Go and JavaScript. What
  broke outside the pinned stack was `.py` as a stand-in for "the server": in a
  Node build the browser script is `.js` too, and counting it as backend makes
  every route it fetches look served — a check that can never fail is worse than
  no check, because it reports a pass.
- **A module whose exports cannot be read confidently leaves the check.**
  `exports.foo = …`, a re-export, a default export, a spread in the exports
  object: any of them and the module is not judged at all. Judging an import
  against most of a module's exports would accuse working code.
- **The install recipe is inferred, never declared.** `LaunchSpec.command` is
  model input and is therefore validated; the install is keyed on which lockfile
  the build produced, so there is no second untrusted command to check. Every
  recipe is pinned to that lockfile and passes `--ignore-scripts`: a package's
  install hooks are third-party code the plan never mentioned.
- **An install failure is never the project's fault.** It happens before
  anything is started, so the report says so — and a process that dies naming a
  module nobody installed gets the same attribution rather than reading as the
  model writing a bad import.
- **A refused launch is never a fallback.** A contract that states how to start
  itself and states something the allowlist will not run gets a skip naming the
  reason. Quietly guessing `server.py` instead would start a different program
  from the one the plan declared and report the result as that plan's.
- **The interpreter allowlist narrows the blast radius; it does not stop code
  execution.** `--smoke` already runs a model-written file, so that door is open
  by the time `plan_launch` is reached. What the allowlist keeps out is "any
  binary on the machine with any arguments": every path argument goes through
  `materialize.safe_join`, and a command naming no file from the output folder
  is refused, so what runs is always the project just written.
- **The smoke run never probes a port it did not open.** A port already
  answering before launch belongs to another server, and its 200s would be
  reported as this project working. Detected by connecting, not by binding —
  they disagree, since a port in TIME_WAIT refuses a bind while nothing is
  listening on it.
- **Executing generated code is opt-in.** `--smoke` is the only place model
  output reaches the interpreter, and the only step whose absence is reported as
  "no evidence" rather than a pass.
- **Dry runs never touch the ledger.** Recording mock calls would tell tomorrow's
  admission control that quota was spent which never was.
- **Model output is untrusted** wherever it reaches a filesystem, a prompt, or
  the dashboard.

## Gotchas

- `--smoke-install` is the **only step in the system that reaches the network
  without an API key**, and the only one that runs third-party code. It is off
  by default and CI never passes it, which is what keeps `pytest` and the CI
  demo run offline.
- The three install recipes were each **run against a real install** before
  being written down. The flags differ per manager — yarn v1 takes
  `--frozen-lockfile`, yarn berry does not — and a guessed flag fails in a way
  that looks like the project's fault.
- A **declared port that disagrees with the code** is a boot timeout, not a
  quick failure: the run waits on the port the contract named. The error says
  which declaration is suspect, but the fifteen seconds are spent either way.
- The smoke run's HTTP client **disables proxies explicitly**. An `http_proxy` in
  the environment otherwise sends a request for 127.0.0.1 to the proxy, and the
  failure reads as the generated server not answering.
- Groq's edge returns **403 to the default urllib User-Agent**. The adapter always
  sets one.
- Gemini charges **invisible thinking tokens** against `max_tokens` and reports
  them nowhere. Hence `min_output_tokens: 8000` on its model entry; there is no
  way to switch thinking off on that endpoint.
- The ledger stamps rows from the **wall clock**, so a test using `FakeClock` for
  a restore will silently find nothing.
- `llmorch.quota` must not import `engine`, `negotiate` or `demo` — there is a
  test asserting it.

---

Design rationale for every decision above, and the six faults the live runs
exposed, is in the git history and the `v0.1.0` release notes. `docs/original-plan.md`
holds the original 45k plan.
