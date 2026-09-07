"""Answering a question about the project, instead of planning a change to it.

The cheap half of this is in `intent.py`, which decides that a line is a
question without spending anything. This is the other half: one request that
returns prose for a person to read, and writes nothing.

Three properties it holds, each for the same reason the rest of the system holds
them:

**It costs one request, at NORMAL priority.** Planning runs HIGH because the
whole run depends on it and that is what the reserve is for. A question is not
on anyone's critical path, so it queues behind the work.

**It grows with the request, never with the project.** A conversation remembers
summaries rather than file contents precisely so the twentieth turn costs what
the second did, and answering a question is not a licence to paste the project
into a prompt — against Groq's ~3,100-token prompt headroom it would not fit
anyway. What a question may carry is the files it *names*: naming a file is part
of the request, so the prompt still grows with what was asked and not with what
has been built. Excerpts that do not fit are dropped before the request is made,
loudest first, so a question is never refused as unservable when a shorter form
of it would have been answered.

**The answer is untrusted output.** It reaches a terminal rather than a file, so
it is stripped of control characters and capped, on the same rule that governs
model output reaching the filesystem or the dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..errors import EngineError, LLMOrchError
from ..quota.store import make_event
from ..registry.manifest import Manifest
from ..types import ChatRequest, Message, Priority, Role, Ticket, Usage

ANSWER_MAX_TOKENS = 700
"""An answer is a paragraph for a person, not a file. Small on purpose: the
output reservation is the half of a request the governor cannot refund, and a
question that returns an essay has usually not been answered."""

EXCERPT_MAX_CHARS = 6000
"""Ceiling on all quoted files together, before the per-model fit is applied."""

EXCERPT_MAX_FILES = 3

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSWER_CAP = 6000


class AnswerError(EngineError):
    """The question could not be answered."""


@dataclass(slots=True)
class Answer:
    text: str
    model_id: str = ""
    usage: Usage = None  # type: ignore[assignment]
    files_read: tuple[str, ...] = ()
    """Which artifacts were quoted into the prompt. Printed with the answer, so
    a reader can tell an answer grounded in a file from one inferred from a
    one-line summary."""


def pick_answerer(
    manifest: Manifest, candidates: list[str], *, prefer: str | None = None
) -> str | None:
    """Whoever is best at reading and explaining, and is still reachable.

    Keyed on the declared research affinity rather than a hardcoded name, for
    the same reason `pick_planner` is: adding a better model to the manifest
    should be enough to change who answers. A person's choice outranks the
    affinity, and being unreachable outranks the choice.
    """
    eligible = [m for m in candidates if m in {x.id for x in manifest.enabled_models}]
    if not eligible:
        return None
    if prefer and prefer in eligible:
        return prefer
    return max(
        eligible,
        key=lambda m: (
            manifest.model(m).affinity(Role.RESEARCH),
            manifest.model(m).quality_prior,
        ),
    )


def files_named(question: str, available: list[str]) -> tuple[str, ...]:
    """Which of the project's files the question actually names.

    Deterministic and free. A bare stem counts — "what does the server do?"
    names `server.py` — but only when exactly one file has that stem, because
    resolving an ambiguous name by guessing would quote the wrong file and
    answer confidently about it.
    """
    lowered = question.lower()
    named: list[str] = []

    for path in available:
        if path.lower() in lowered:
            named.append(path)

    if not named:
        stems: dict[str, list[str]] = {}
        for path in available:
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if stem:
                stems.setdefault(stem, []).append(path)
        for stem, paths in stems.items():
            if len(paths) == 1 and re.search(rf"\b{re.escape(stem)}\b", lowered):
                named.append(paths[0])

    return tuple(dict.fromkeys(named))[:EXCERPT_MAX_FILES]


def take_excerpts(
    files: dict[str, str], paths: tuple[str, ...], *, budget: int = EXCERPT_MAX_CHARS
) -> dict[str, str]:
    """The named files, trimmed to a shared character budget.

    Trimmed from the end and marked, rather than summarised: a truncated file
    that says it was truncated is honest input, and the alternative — quietly
    handing over the first half — invites an answer about the parts that were
    dropped.
    """
    if not paths:
        return {}
    share = max(400, budget // len(paths))
    out: dict[str, str] = {}
    for path in paths:
        text = files.get(path)
        if text is None:
            continue
        if len(text) > share:
            text = text[:share] + "\n… (truncated; this is the start of the file)"
        out[path] = text
    return out


def build_answer_prompt(
    question: str,
    *,
    memory: str,
    interface_text: str,
    excerpts: dict[str, str] | None = None,
    exchanges: str = "",
) -> tuple[str, str]:
    """Return (system, user) for a question about the project.

    The system prompt's real work is the last rule. A model handed summaries and
    asked what a file does will describe what a file with that summary would
    plausibly do; saying so is an answer, and saying it as if it had read the
    file is not.
    """
    system = """\
You are answering a question about a project that several different models \
built together. You are not changing it: no files, no plan, no JSON.

What you have been given is what the system itself remembers between turns — \
the instructions so far, the interface contract every model was held to, and \
one summary per file, written by whichever model wrote that file. Where the \
question named a file, its contents are quoted too.

Answer in plain prose, briefly — a short paragraph, or a few lines of a list. \
No code fences unless you are quoting something you were given.

Say what you actually know. If the answer is not in what you were given, say \
that, and say what would settle it — the file to read, or the command to run. \
An answer inferred from a one-line summary must be marked as inferred. \
Describing a file you were not shown as if you had read it is the one failure \
that matters here, because everything else in this project is checked and this \
is not.
"""
    parts = [memory]
    if interface_text:
        parts.append(interface_text)
    for path, text in (excerpts or {}).items():
        parts.append(f"## `{path}`, as it is on disk now\n\n```\n{text}\n```")
    if exchanges:
        parts.append(exchanges)
    parts.append(f"## The question\n\n{question}")
    return system, "\n\n".join(parts)


def clean_answer(text: str) -> str:
    """Model output on its way to a terminal.

    Control characters are stripped for the same reason artifacts are contained
    to the output folder: a reply is data, and data does not get to move the
    cursor, clear the screen or repaint what was printed above it.
    """
    text = _ANSI.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.strip()
    if len(text) > _ANSWER_CAP:
        text = text[:_ANSWER_CAP].rstrip() + "\n… (answer truncated)"
    return text


def _fit(
    question: str,
    *,
    memory: str,
    interface_text: str,
    excerpts: dict[str, str],
    exchanges: str,
    deps,
    provider_name: str,
    ceiling: int,
    max_tokens: int,
) -> tuple[str, str, dict[str, str], int]:
    """Shrink the prompt until it fits the model's per-request ceiling.

    Excerpts go first and largest first, because they are the part of the prompt
    the question can do without: an answer from summaries alone is worse than an
    answer from the file, and both are better than `UNSERVABLE`.
    """
    quoted = dict(excerpts)
    while True:
        system, user = build_answer_prompt(
            question,
            memory=memory,
            interface_text=interface_text,
            excerpts=quoted,
            exchanges=exchanges,
        )
        est = deps.estimator.estimate_prompt(
            system=system, messages=[user], provider=provider_name
        )
        if est + max_tokens <= ceiling or not quoted:
            return system, user, quoted, est
        widest = max(quoted, key=lambda p: len(quoted[p]))
        del quoted[widest]


def _record(
    deps,
    model_id: str,
    est_prompt: int,
    est_completion: int,
    *,
    usage: Usage | None = None,
    latency_ms: int = 0,
    error: Exception | None = None,
) -> None:
    """Put the call on the record, under its own purpose.

    A question is a real request against a real daily allowance, so it belongs
    in the ledger for the same reason execution does: `restore_governor` replays
    the day at startup, and a request that was never written down is quota
    tomorrow's process believes it still holds.
    """
    if deps.ledger is None:
        return
    status = (getattr(error, "status", None) or 0) if error is not None else 200
    deps.ledger.record(
        make_event(
            run_id=deps.run_id,
            node_id=None,
            purpose="answer",
            manifest=deps.manifest,
            model_id=model_id,
            usage=usage or Usage(),
            est_prompt_tokens=est_prompt,
            est_completion_tokens=est_completion,
            ok=error is None,
            http_status=status,
            latency_ms=latency_ms,
            error=str(error) if error is not None else None,
        )
    )


async def answer(
    question: str,
    *,
    deps,
    model_id: str,
    memory: str,
    interface_text: str = "",
    excerpts: dict[str, str] | None = None,
    exchanges: str = "",
) -> Answer:
    """Ask one model the question, through the same governed path as everything else.

    Routing it around admission control would let the cheapest request in the
    system be the one that blows the daily cap.
    """
    manifest: Manifest = deps.manifest
    model = manifest.model(model_id)
    provider_name = manifest.vendor_of(model_id)

    max_tokens = min(model.max_output, max(model.min_output_tokens, ANSWER_MAX_TOKENS))
    ceiling = manifest.max_request_tokens(model_id)

    system, user, quoted, est_prompt = _fit(
        question,
        memory=memory,
        interface_text=interface_text,
        excerpts=excerpts or {},
        exchanges=exchanges,
        deps=deps,
        provider_name=provider_name,
        ceiling=ceiling,
        max_tokens=max_tokens,
    )

    # NORMAL, not HIGH: the reserve exists so a critical-path retry is not
    # crowded out, and a question is never on the critical path.
    ticket = deps.governor.try_acquire(
        model_id, est_prompt, max_tokens, priority=Priority.NORMAL
    )
    if not isinstance(ticket, Ticket):
        raise AnswerError(
            f"cannot answer with {model_id}: {getattr(ticket, 'reason', 'refused')}"
        )

    request = ChatRequest(
        model_id=model_id,
        messages=(Message("user", "[answer]\n" + user),),
        system=system,
        max_tokens=max_tokens,
    )

    try:
        response = await deps.registry.get(model_id).chat(request)
    except LLMOrchError as exc:
        deps.governor.release(ticket, "answer failed")
        _record(deps, model_id, est_prompt, max_tokens, error=exc)
        raise AnswerError(f"{model_id} could not answer: {exc}") from exc

    deps.governor.commit(ticket, response.usage)
    _record(
        deps,
        model_id,
        est_prompt,
        max_tokens,
        usage=response.usage,
        latency_ms=response.latency_ms,
    )
    deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
    if response.rate_limit:
        deps.governor.sync_from_headers(model_id, response.rate_limit)

    text = clean_answer(response.text)
    if not text:
        raise AnswerError(f"{model_id} returned an empty answer")

    return Answer(
        text=text,
        model_id=model_id,
        usage=response.usage,
        files_read=tuple(quoted),
    )
