# llmorch — handoff

**State:** M0–M6 done. 449 tests pass, 1 skipped (a symlink test needing admin on
Windows). Published at github.com/Pranay4040/llmorch, tagged `v0.1.0`.

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
.venv/Scripts/python.exe -m pytest -q                  # 449 tests, no network
.venv/Scripts/python.exe -m llmorch run "build a notes app"        # mock, offline
.venv/Scripts/python.exe -m llmorch run --live --providers all "<task>"
.venv/Scripts/python.exe -m llmorch resume <run_id>    # after a quota wall
.venv/Scripts/python.exe -m llmorch doctor --probe     # verify wire names live
.venv/Scripts/python.exe -m llmorch discover           # what each key can reach
.venv/Scripts/python.exe -m llmorch dashboard          # read-only, localhost
```

Keys live in `.env` (gitignored). Nothing above needs one except `--live`,
`--probe` and `discover`.

## Roster (verified live 2026-09-01)

| Vendor | Models | Real limits |
|---|---|---|
| Groq | gpt-oss-120b, gpt-oss-20b, qwen3-27b | 1,000 req/day, 8,000 TPM — **from its own headers** |
| Gemini | 3.6-flash | 20 req/**minute** — from a 429 body. Daily figure unverified |
| OpenRouter | minimax-m3, nemotron-ultra, north-mini-code | free tier; limits estimated |

---

## Next

Ordered by value. Issues #1–#4 are filed on GitHub.

1. **Contract checking beyond Python/web** (#1). `engine/contracts.py` catches
   cross-artifact mismatches — it found a real one where two models disagreed on
   a function signature and both files were individually perfect. But the route
   and asset checks are HTTP-shaped and the call check is `ast`-based, so a JS or
   Go build gets nothing. Keep any new check conservative: a false accusation
   about working code is worse than a missed fault.
2. **A smoke-run step.** The orchestrator writes a project and never runs it.
   Both real bugs that reached the output — a Windows path fault, and the
   signature mismatch — were found by executing the result, not by testing it.
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
- **Dry runs never touch the ledger.** Recording mock calls would tell tomorrow's
  admission control that quota was spent which never was.
- **Model output is untrusted** wherever it reaches a filesystem, a prompt, or
  the dashboard.

## Gotchas

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
