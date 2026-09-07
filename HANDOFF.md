# llmorch — handoff

**State:** M0–M6 done, plus the smoke run, the question lane, the setup page, live run tracking per-job model choice and the three session modes. 743 tests pass,
1 skipped on Windows (a symlink test needing admin). Published at
github.com/Pranay4040/llmorch, tagged `v0.1.0`.

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
.venv/Scripts/python.exe -m pytest -q                  # 743 tests, no network
.venv/Scripts/python.exe -m llmorch run "build a notes app"        # mock, offline
.venv/Scripts/python.exe -m llmorch run --smoke "<task>"          # ...then run the result
.venv/Scripts/python.exe -m llmorch run --smoke-install "<task>"  # ...installing its deps first
.venv/Scripts/python.exe -m llmorch run --live --providers all "<task>"
.venv/Scripts/python.exe -m llmorch configure          # the setup page, in a browser
.venv/Scripts/python.exe -m llmorch start             # a session, on what it saved
llmorch.cmd                                            # the setup page — the short way in
.venv/Scripts/llmorch.exe                              # ...same thing, once PATH is set
.venv/Scripts/llmorch.exe "build a notes app"          # ...with the first thing said
.venv/Scripts/python.exe -m llmorch chat               # a session, not one shot
.venv/Scripts/python.exe -m llmorch chat --continue    # ...pick the last one back up
.venv/Scripts/python.exe -m llmorch ask "what does server.js do?"  # ask, don't build
.venv/Scripts/python.exe -m llmorch resume <run_id>    # after a quota wall
.venv/Scripts/python.exe -m llmorch doctor --probe     # verify wire names live
.venv/Scripts/python.exe -m llmorch discover           # what each key can reach
.venv/Scripts/python.exe -m llmorch dashboard          # read-only, localhost
```

Keys live in `.env` (gitignored). Nothing above needs one except `--live`,
`--probe` and `discover`.

Every run writes `runs/<run_id>/report.md` beside its output folder: the verdict
first, then nodes, spend, fair share, cross-artifact checks and the smoke run.
It is the same information the terminal prints, kept for someone reading it
after the scrollback is gone.

`.github/workflows/tests.yml` runs the suite on push and pull request across
Linux (3.11, 3.13) and Windows (3.12), then does a full offline demo run with
`--smoke`. Everything it does is offline, so CI needs no secrets and never
spends quota.

## Roster (verified live 2026-09-01; OpenRouter additions 2026-09-07)

| Vendor | Models | Real limits |
|---|---|---|
| Groq | gpt-oss-120b, gpt-oss-20b, qwen3-27b | 1,000 req/day, 8,000 TPM — **from its own headers** |
| Gemini | 3.6-flash | 20 req/**minute** — from a 429 body. Daily figure unverified |
| OpenRouter | minimax-m3, nemotron-ultra, north-mini-code, laguna-s, laguna-xs, dots-3-note, minimax-m2.7, ling-flash-fin, lfm-2.5, nemotron-lightning, nemotron-super | free tier; **50 req/day account-scoped**, so more models here is diversity, not capacity |

---

## Next

Ordered by value. Issues #1–#4 are filed on GitHub.

1. **Enforce the staffing rule instead of asking for it.** `build_revise_prompt`
   now tells the planner that a route it adds to the contract needs a node that
   serves it — found by running `chat` and watching a revision add a route,
   staff nothing, fail the cross-artifact check and 404 under the smoke run.
   But a prompt is a request, and decision 7 in the original plan is that a
   constraint you can enforce in code does not belong in a prompt: an LLM told
   to "split evenly" will not, which is why evenness is an assignment
   constraint. The same argument applies here. After `revise` returns and
   before any execution request is spent, the routes a revision adds can be
   compared against the delta nodes and the existing backend files; an unstaffed
   promise is then caught for free, rather than after the whole change is paid
   for.
2. **Planning and bidding are not on the ledger.** `decompose`, `revise` and
   `collect_bids` commit to the governor and observe the estimator, but write no
   `UsageEvent` — only execution, review, repair, doctor and now `answer` do. So
   `restore_governor` under-counts the day by every planning request a previous
   process made, which is the one thing the ledger exists to prevent. The fix is
   the four-line `_record` that `answer.py` already carries; what makes it worth
   care rather than a copy-paste is that `make_event` wants a `node_id` and a
   plan has none, so the column has to mean "not a node" rather than "unknown".
3. **Arity agreement in JavaScript** (#1, the half still open). Routes, imports
   and imported names now work for JS and Go; what `check_python_calls` does and
   nothing else can is compare *signatures*. `ast` gives Python exact ones, and
   JavaScript defaults, rest parameters and destructured options objects mean a
   regex would report working code as broken. This needs a real parser, which
   means a dependency this project has so far refused — and the missed fault is
   cheaper than the false accusation, so it stays open on purpose.
4. **A compiled build step, if it is ever worth it.** `--smoke-install` covers
   the Node case: a lockfile is enough to install from, and the recipe is
   inferred from which lockfile exists rather than declared by a model. What is
   still not covered is anything needing a *compile* — a Go binary, a TypeScript
   build — and the case for adding it is weaker than it looks: `go run` already
   fetches and compiles, and a build step is where an arbitrary command would
   have to come back in. Worth doing only when a real plan is blocked by it.
5. **Paid providers** (#2). DeepSeek, Moonshot, Fireworks, OpenCode Zen are
   discovered and undeclared. OpenCode Zen fronts Claude and GPT families, so it
   would reopen the roster the way OpenRouter did. Blocked on *verified*
   pricing — the manifest rejects a paid provider with no cost, and inventing
   numbers to satisfy that check defeats it.
6. **Gemini's daily limit** (#4) and **the Mistral/Wafer keys** (#3).

## Do not break these

Each was learned by getting it wrong against a live API.

- **Neither local server may share its port.** `HTTPServer` sets
  `allow_reuse_address`, which on POSIX only skips TIME_WAIT and on Windows lets
  a *second* process bind an address a first is already listening on — with
  connections going to whichever wins. Two setup servers were live on 8788 at
  once and the older one answered a link the newer had just printed, which
  surfaces as "missing or wrong token" against a URL that is visibly correct.
  Both servers now refuse the bind and say what is already there.
- **The setup token is kept, not minted per launch.** A fresh secret each time is
  stronger and made every bookmark and reopened tab a dead end. A stable secret
  stops a cross-origin post exactly as well; what it does not stop is a local
  process reading `configure-token`, and such a process can already read `.env`.
  A refused link now gets a page explaining itself rather than one line of text.
- **A closed browser tab is not an error.** `socketserver` prints a traceback
  when a client drops a keep-alive connection, which the dashboard's five-second
  poll makes routine — fifteen lines into the middle of whatever the person was
  reading in that terminal. Both servers swallow connection resets and nothing
  else.
- **Opening a tab is not choosing.** Each mode tab carries its own "always
  use this mode / ask me at the start" control, because a tab that selected the
  mode by being opened would mean you could not look at what a mode does without
  changing which one you get.
- **A control shown twice has one source.** The answering model appears on both
  *Who does what* and *Chat*; both are filled by `fillRolePicker` from the same
  data and kept in step by `mirrorRole`, rather than being two places that
  separately remember the same fact.
- **An answer quoting a file is the only path by which what you built reaches a
  provider.** `answer_reads_files` is on by default because grounding beats
  inference, and exists at all because that sentence is worth being able to act
  on.
- **The three session modes are settings, not implementations.** Chat is the
  question lane as a standing choice, one agent is every job pinned at once, and
  a crew is the default behaviour. Adding a fourth mode should mean finding
  another setting of the same machinery, not another code path.
- **A mode says what it gives up.** One agent cannot have cross-vendor review —
  `pick_reviewer` requires a different vendor than the author and there is not
  one — so the menu says so before the choice rather than leaving it to be
  inferred from a report with no review section.
- **Nobody is asked who is not there to answer.** The mode prompt checks
  `stdin.isatty()` first. A menu printed at a pipe would consume the first line
  of input as the answer to a question the pipe never saw, which is exactly what
  happened the first time — the whole test suite failed on it.
- **A session is named once, before anything of it is on disk.** The id is a
  directory name, a checkpoint key and the `run_id` on every ledger row the
  session produces, so the one safe moment to name it is the first instruction —
  when none of those exist yet. `Conversation.name_for` refuses after that
  rather than renaming and orphaning all three.
- **The timestamp stays in front of the slug.** `latest_session`,
  `resume --list` and the dashboard's run list all order runs by the directory
  name and nothing else, so the prefix has to stay fixed-width and sortable. The
  slug is for the reader; the prefix is load-bearing.
- **A bigger roster does not mean a bigger allowance, and can mean less
  negotiation.** OpenRouter's 50 requests a day are account-scoped, so the eight
  models added on 2026-09-07 share one bucket with the three that were already
  there — what they buy is vendor diversity for review and failover, not
  throughput. And `should_bid` skips the bidding round when there are more
  models than nodes, so a 15-model roster retires it under `auto` for any
  normal-sized graph. Both are the right behaviour and neither is obvious.
- **A newly verified model is not a better model.** The 2026-09-07 additions sit
  at the tail of every chain and score below the incumbents, so the default
  assignment did not move. They are reachable through failover, through
  cross-vendor review, and through a pin — and `profiles.json` promotes them on
  their own if they earn it. Raising their priors to make them get picked would
  be inventing the evidence the priors are supposed to record.
- **A pin binds the assignment and nothing else.** Choosing a model for a job
  overrides the reconciler's choice for that role and leaves every other
  mechanism intact: failover still runs its whole ladder, since a model that has
  tripped its circuit breaker is not the one anybody meant to insist on, and
  `pick_reviewer` filters a review pin through the cross-vendor rule rather than
  letting it outrank it. Enforced structurally — a pinned node is scored against
  one model, so the 2-opt swap pass has nothing to trade it into.
- **A pin that cannot be honoured falls back out loud.** Not in the roster, or
  too small an output ceiling for the node: the automatic choice takes over and
  the run says which pin it could not use. Degrading a node to honour a
  preference literally is a worse answer to "I prefer this model" than doing the
  work with a note attached.
- **Every chooser reads the pins, not just the reconciler.** Three jobs are
  never carried by a node — the planner, the answerer and the reviewer — and
  wiring only the reconciler left the two most visible ones (chat and planning)
  silently unpinnable. Found by pinning them and watching the affinity win.
- **A run's numbers belong in the browser, because they move.** The terminal
  printed five tables describing the state everything was in when the run
  ended. A run now publishes `runs/<id>/progress.json` as it goes — the same
  file-and-poll shape the checkpoint already established, since the dashboard is
  a different process — and the terminal keeps a verdict line and a URL.
  `--tables` restores the old output.
- **A node in flight reports no tokens.** The governor reserves on an estimate
  and reconciles on commit, so a node that has not come back has a budget and
  not a bill. Publishing the estimate as spend would show a total that walks
  backwards when the wave lands, which is the sort of number people stop
  trusting.
- **Nothing writes "finished" on a dead run's behalf.** The progress file is all
  a killed process leaves behind, so a run is *active* only while its file is
  moving. The staleness window is 90s, because a node waiting out a per-minute
  rate limit is working, and calling that stopped misreports the most
  interesting moment there is.
- **Monitoring may never fail the run it describes.** Every write is wrapped and
  every headroom reading is guarded; a progress file that cannot be written is
  silence, not an exception.
- **The two "today" figures on the dashboard disagree on purpose, and each says
  which it is.** The run panel's comes from the vendor's rate-limit headers —
  fact, and what admission control believes — while the quota table's is the
  ledger replay. Found by watching them differ by nine on one screen with
  nothing on the page to explain it, which is how a reader learns to trust
  neither.
- **The setup page is the only thing here that accepts a write, and it is
  guarded as one.** The dashboard's header states that being read-only is *why*
  it needs no authentication; that argument does not extend to a page which
  saves settings and writes an API key, so `configure` mints a token at launch,
  requires it on every request, and refuses a non-loopback `Host`. Binding to
  127.0.0.1 alone is not enough — a hostile name resolving there is same-origin
  as far as a browser is concerned, and same-origin policy only stops an
  attacker *reading* the reply, which somebody writing a key does not need.
- **Keys go one way through it.** `GET /api/config` reports whether each key is
  set; nothing returns one. And only the variables a declared provider names can
  be written, or the endpoint is a way to put arbitrary variables into a file a
  shell may later source.
- **Read every file on this machine as `utf-8-sig`.** Notepad and PowerShell's
  `Set-Content -Encoding utf8` both write a byte-order mark. In `settings.json` a
  BOM made `json.loads` fail on a perfect file and the roster silently widened
  back to every model; in `.env` it becomes part of the first variable's *name*,
  which reads as a wrong key rather than a mis-encoded file. Found by saving
  settings from PowerShell and watching Gemini reappear in an assignment that
  had excluded it.
- **A settings file that cannot be read is reported, not just replaced.**
  Falling back to the defaults is right — configuration must never be why a
  build cannot start — but a file that fails to parse looks exactly like a file
  nobody wrote, and one of those is a configuration somebody made and is not
  getting.
- **The roster is narrowed to what can actually be called, before anything is
  built on it.** `--providers groq` used to narrow only the client registry, so
  the assignment still handed a node to Gemini and the run died looking for a
  provider that was never built — after the plan was printed and the planning
  request already spent. A provider enabled in `models.yaml` with no key in the
  environment is the same fault through a quieter door, so `restricted_to`
  covers both. Disabling a vendor is a state the manifest is *designed* to be
  valid in, which is why `_validate` checks the declared chain rather than the
  enabled subset.
- **The line decides the lane, and the default is a build.** A question is
  answered, an acknowledgement costs nothing, and anything unrecognised is an
  instruction — which is the behaviour that was there before, so the classifier
  is never worse than what it replaced. The two mistakes are not symmetrical:
  reading an instruction as a question spends one request and builds nothing;
  reading a question as an instruction is the old bug. `/ask` and `/build` exist
  so the rules can stay small instead of growing a case per phrasing.
- **A question grows the prompt with the request, never with the project.** The
  same reason a conversation remembers summaries rather than file contents. What
  an answer may quote is the files the question *names* — naming one is part of
  the request — and against Groq's ~3,100 tokens of prompt headroom even that is
  fitted before the call, dropping the largest excerpt first. A question that
  does not fit is shortened, never refused: an answer from summaries is worse
  than an answer from the file and much better than `UNSERVABLE`.
- **An answer runs at NORMAL priority and lands in the ledger.** Planning is HIGH
  because the run depends on it and that is what the reserve is for; nothing
  depends on a question. And it is a real request against a real daily
  allowance, so a run that did not write it down would leave tomorrow's process
  believing it still holds that quota.
- **The suite is isolated from `.env`.** `main` calls `load_dotenv`, so any test
  that ran the CLI used to copy the developer's real keys into the process, and
  every later test that described a keyless provider was describing a machine
  with the *other* keys still set. It passed in CI, which has no `.env`, and
  failed on the machine that had one. `tests/conftest.py` unsets every
  `*_API_KEY` per test; a suite that is green only where the secrets are absent
  is testing the machine.
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
- **A conversation remembers summaries, never file contents.** Feeding the
  artifacts back to the planner would grow every prompt with the project rather
  than with the request, against a 6,000 TPM ceiling. What a turn remembers is
  what the blackboard already passes between nodes: the instructions, the
  contract, and one summary per file — written by the model that wrote the file,
  in the same response, so it cost nothing extra.
- **A revision adds to the contract, it does not replace it.** A planner asked
  about a change answers about the change; a revision that adds one route and
  does not restate the other three is describing an addition. Replacing would
  silently unserve the other three, and every check downstream measures the
  artifacts against this contract, so it would report the wrong thing with total
  confidence.
- **A turn's contract check sees the whole folder, not the turn's delta.** Given
  one turn's three changed files it would otherwise report the other five pages
  as missing — the checker looking at a fragment and describing it as the
  project.
- **`report.md` recomputes nothing and holds no key.** Every figure comes from
  the same objects the terminal renderers read, so the file and the screen
  cannot disagree about what happened; and it is as publishable as the output
  folder beside it, which is why the rule that keeps secrets out of the ledger
  and out of error strings covers it too.
- **The verdict reports the result, not the steps.** A run whose nodes all
  succeeded can still have produced a project that does not run — that is the
  case the section exists for, and why a smoke run that never happened is
  written as "not started" rather than left blank.
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
- **`setup.ps1` is proved by CI or not at all**, and the same goes for the
  shim. Neither can run on the machine they are written on, so the
  windows-latest job runs `setup.ps1` for real — venv, install, PATH entry,
  Desktop shortcut — and asserts each of the four. Editing either without that
  job passing is editing blind.
- **The Windows shim is proved by CI or not at all.** `llmorch.cmd` cannot run
  on the machine it was written on, so `test_the_repository_shim_runs_the_cli`
  picks the shim for its platform and the windows-latest job is what actually
  exercises the batch file. Changing it without that job passing is changing it
  blind.
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
