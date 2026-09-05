# llmorch

[![tests](https://github.com/Pranay4040/llmorch/actions/workflows/tests.yml/badge.svg)](https://github.com/Pranay4040/llmorch/actions/workflows/tests.yml)

**Quota governance for applications that call several LLM providers** — plus an
orchestrator built on top of it, as the worked example.

The hard part of using free and low tiers is not cost. It is that they are
rationed along several axes at once, with different numbers per provider, days
that end at different midnights, and limits that are sometimes shared across
every model on an account. Get that arithmetic wrong and you do not get a
slightly worse result — you get a 429 in the middle of work you have already
partly paid for.

```python
from llmorch.quota import Governor, Ticket, model_spec, provider_spec, quota_manifest

manifest = quota_manifest(
    providers=[
        provider_spec(
            "groq",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            rpm=30, tpm=8000, rpd=1000,   # as the provider's docs state them
            reserve_requests=50,          # held back for critical retries
        ),
    ],
    models=[
        model_spec(
            "groq/gpt-oss-120b",
            provider="groq",
            wire_name="openai/gpt-oss-120b",
            context=131072,
            max_output=4096,
        ),
    ],
)

governor = Governor(manifest)

verdict = governor.try_acquire("groq/gpt-oss-120b", 400, 1200)
if isinstance(verdict, Ticket):
    ...                                    # call the provider, then:
    # governor.commit(verdict, response.usage)
else:
    print(verdict.verdict.value, verdict.reason)
```

No YAML, no role taxonomy, no dependency on the orchestrator.

## The distinction the whole thing turns on

Every refusal is not the same refusal:

| Verdict | Meaning | What a caller should do |
|---|---|---|
| `GRANTED` | proceed | send the request |
| `WAIT` | a per-minute window is full | sleep `retry_after_s`, then retry |
| `EXHAUSTED_TODAY` | daily cap reached | try another provider; resume after its local midnight |
| `UNSERVABLE` | larger than the per-minute ceiling | never send it at this size — split it |
| `COST_BLOCKED` | would spend real money unauthorised | nothing, until a budget is set |

**`UNSERVABLE` is not `WAIT`.** A request bigger than a provider's per-minute
token ceiling will not fit however long you wait, and treating it as a wait
condition hangs the caller forever. Against an 8,000 TPM provider this case is
routine, not exotic.

**`WAIT` is not `EXHAUSTED_TODAY`.** A window that clears in nine seconds is not
a model being unavailable for the day, and conflating them writes off a healthy
provider until midnight. Both distinctions were learned by getting them wrong.

## What else is in here

| Piece | What it does |
|---|---|
| `llmorch.quota.Governor` | Admission control across RPM / TPM / RPD / TPD, account- and model-scoped, with a reserve for critical work |
| `llmorch.quota.LedgerStore` | Append-only SQLite record of every call, stamped with the **provider's own** day key |
| `llmorch.quota.restore_governor` | Replays today's ledger at startup, so a fresh process does not think it holds the whole daily allowance |
| `llmorch.quota.TokenEstimator` | Character-based estimation, self-correcting per provider from real usage |
| `llmorch.providers.OpenAICompatProvider` | Dependency-free client for any OpenAI-shaped endpoint |
| `llmorch.providers.parse_rate_limit_headers` | Turns `x-ratelimit-*` headers — and prose like *"retry in 12.4s"* — into a snapshot |

Three properties worth knowing, because they are easy to get wrong and all
three cost a live 429 to discover:

- **Reserve on the estimate, reconcile on commit.** Without it, concurrent
  fan-out races: several callers each read a counter none of them has yet
  incremented.
- **Monotonic clock for sliding windows, wall clock for day boundaries, never
  mixed.** Wall time for a per-minute window lets a clock adjustment grant free
  quota; monotonic time for a daily counter never resets.
- **Response headers override local counters.** Local counting is inference;
  headers are fact, and they cost nothing to read. A stated wait also outranks
  any keyword in the body — if the server says come back in twelve seconds,
  waiting works, whatever else the error text mentions.

## No dependencies you did not ask for

`PyYAML`, `pydantic` and `tzdata` — the last only because Windows ships no
timezone database, and a provider whose day ends at Pacific midnight cannot be
tracked without one.

Deliberately **not** an HTTP library and **not** a vendor SDK. The provider
client is stdlib `urllib` on a worker thread, because every SDK ships its own
retry and rate-limit machinery, which would sit *underneath* the governor and
silently retry requests it never admitted. Admission control only works if every
call goes through it.

## The orchestrator

The application this was built for splits a task across models from different
vendors, assigns each slice by fitness and remaining quota, and writes a
runnable project folder:

```bash
llmorch run "build a notes app"                  # mock provider, no network
llmorch run --live --providers all "build a CLI that converts CSV to markdown"
llmorch run --smoke "build a notes app"          # ...and then run what it wrote
llmorch chat                                     # a session, not one shot
llmorch resume <run_id>                          # after a quota wall
```

`chat` keeps the conversation: the first instruction builds a project, and each
one after it is planned as a *change* to what already exists, so "now add tags"
rewrites the two files that need it rather than the six that do not. What a later
turn remembers is the instructions, the interface contract, and one summary per
file — never the file contents, because a conversation that pasted its artifacts
back into the planner would grow every prompt with the project instead of with
the request. Sessions are saved after every turn; `--continue` picks the last one
back up.

`--smoke` starts the generated project, drives the contract's pages and routes
against it over HTTP, and reports what came back. How to start it is part of the
contract the planner emits, so it is stated rather than guessed:

```json
"launch": {"command": ["node", "server.js"], "port": 3000, "ready_path": "/"}
```

The command is checked before it runs: the program must be a known interpreter,
and every path in it must resolve inside the output folder, through the same
containment check that writing those files used. A command that fails either
test is refused by name — never quietly replaced with a guess.

A project with dependencies gets `--smoke-install`, which runs a lockfile-pinned
install first — `npm ci --ignore-scripts` and its pnpm and yarn equivalents. The
recipe is chosen by which lockfile the build produced, not by anything a model
said, and package install scripts stay disabled. Without that flag, a project
needing an install is skipped with the command that would fix it, rather than
being started into a folder where it can only fail. It is off by default and has
to be asked for: every other step in the system treats model output as untrusted
data, and this one hands it the interpreter. What it buys is the only evidence in
a run that did not come from reading the code — a project whose files all parse,
all pass review, and all agree with each other can still serve every page from
the wrong directory, and nothing static will say so.

Before any of that, eight deterministic checks read the finished artifacts as a
set rather than one at a time — the pages the contract promised exist, the assets
and modules they reference were written, the frontend calls only declared routes,
the backend mentions every declared route, and the modules agree with each other
on names and signatures. They cost no requests and they catch the failure a split
build makes likely and a single author never would: every file impeccable against
its own spec, and the project broken because two models agreed with the spec and
not with each other.

Each run leaves `runs/<run_id>/report.md` next to the folder it produced —
verdict first, then which model wrote which file, what the quota bought, how
evenly the work landed, and what the checks and the smoke run found. The
artifacts stay on disk indefinitely and look equally plausible either way; the
evidence about them should not be the one part that lives in scrollback.

Supporting commands: `doctor --probe` (verify wire names before depending on
them), `discover` (ask a key which models it can reach, spending no tokens),
`quota`, `ledger`, `dashboard` (read-only, loopback only).

Current state, what is next, and the invariants not to break are in
[HANDOFF.md](HANDOFF.md). The original 45k plan is in
[docs/original-plan.md](docs/original-plan.md).

## Install

```bash
pip install -e ".[dev]"
python -m pytest -q
```

CI runs that suite on Linux (3.11 and 3.13) and Windows (3.12), then does a full
offline demo run with `--smoke` — plan, execute against the mock provider, write
the folder, and start what it wrote. Windows is in the matrix on purpose: three
of the invariants this project holds were faults that only appear there.
