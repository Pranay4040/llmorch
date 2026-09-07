"""Telling an instruction from a question, before either costs anything.

`llmorch chat` planned every line as a change. "what does the server do?" and
"looks good" both bought a planning request — one of them to be told there was
nothing to plan. Against 250 requests a day that is the cost worth removing, and
it is removable for free: which lane a line belongs in is a property of the line,
readable here rather than discoverable from a model's reply.

**The classifier is conservative on purpose, and the default is BUILD.** The two
mistakes are not symmetrical. Reading an instruction as a question spends one
request and builds nothing — the user says it again and is annoyed. Reading a
question as an instruction is what happens today, so falling through to BUILD is
never worse than the behaviour being replaced. So every rule here recognises a
*shape*, and anything unrecognised is an instruction.

For the residue, both lanes can be named outright: `/ask` and `/build` take the
decision away from the heuristic entirely. That escape hatch is why the rules can
stay small instead of growing a special case for every phrasing.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    """What a line of a conversation is asking for."""

    BUILD = "build"
    """An instruction: plan it and write the files it needs."""

    ASK = "ask"
    """A question about the project: answer it, change nothing."""

    REMARK = "remark"
    """Neither: an acknowledgement. Costs nothing and is recorded as said."""


# Acknowledgements, as a closed set. A closed set rather than a rule, because
# the failure of a rule here is silently discarding an instruction — and the
# whole set is short enough to read.
_REMARKS = frozenset(
    {
        "ok",
        "okay",
        "k",
        "kk",
        "fine",
        "good",
        "great",
        "nice",
        "cool",
        "perfect",
        "lovely",
        "excellent",
        "beautiful",
        "lgtm",
        "yes",
        "yep",
        "yeah",
        "no",
        "nope",
        "sure",
        "right",
        "hm",
        "hmm",
        "huh",
        "wow",
        "ah",
        "oh",
        "done",
        "stop",
        "wait",
        "thanks",
        "thank you",
        "thanks!",
        "ta",
        "cheers",
        "nevermind",
        "never mind",
        "looks good",
        "look good",
        "looks great",
        "looks right",
        "looks fine",
        "that works",
        "that works!",
        "it works",
        "works",
        "makes sense",
        "good job",
        "nice work",
        "well done",
        "got it",
        "understood",
        "noted",
        "i see",
        "fair enough",
        "carry on",
        "keep going",
        "that helps",
        "that helped",
        "helpful",
        "very helpful",
        "good to know",
        "brilliant",
        "awesome",
        "amazing",
        "ty",
        "thx",
        "much appreciated",
        "appreciated",
        "thank you very much",
    }
)

# A question that opens with one of these is a question about the project.
_ASK_OPENERS = frozenset(
    {
        "what",
        "whats",
        "why",
        "how",
        "when",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "am",
        "have",
        "has",
        "had",
        "should",
        "explain",
        "describe",
        "summarise",
        "summarize",
        "remind",
    }
)

# Openers that read as questions but are proposals: "how about tags",
# "what if the list were paginated". A proposal is an instruction.
_PROPOSALS = ("how about", "what about", "what if", "how come we")

# Phrases that open a *request* rather than a question, when a build verb
# follows: "can you add tags?" ends in a question mark and is not a question.
_POLITE = (
    "can you please",
    "could you please",
    "would you please",
    "can you",
    "could you",
    "would you",
    "will you",
    "can we",
    "could we",
    "can i get",
    "i want you to",
    "i want",
    "i need",
    "i'd like",
    "id like",
    "please",
    "lets",
    "let's",
    "go ahead and",
    "now",
)

# What a line asks to be done. Only used to rescue a politely-phrased
# instruction from the question mark at the end of it.
_BUILD_VERBS = frozenset(
    {
        "add",
        "build",
        "create",
        "make",
        "write",
        "implement",
        "change",
        "update",
        "remove",
        "delete",
        "drop",
        "rename",
        "fix",
        "refactor",
        "move",
        "rewrite",
        "redo",
        "support",
        "include",
        "replace",
        "split",
        "merge",
        "style",
        "restyle",
        "use",
        "switch",
        "convert",
        "set",
        "enable",
        "disable",
        "wire",
        "hook",
        "generate",
        "put",
        "give",
        "turn",
        "extend",
        "extract",
        "sort",
        "order",
        "validate",
        "test",
        "document",
        "handle",
        "store",
        "save",
        "load",
        "show",
        "display",
        "render",
        "paginate",
        "cache",
        "log",
    }
)

_WORD = re.compile(r"[a-z']+")


def _normalise(line: str) -> str:
    """Lowercase, collapse whitespace, and drop trailing punctuation.

    Used only for the acknowledgement set, so that "thanks!" and "Thanks" are
    the same remark. The question rules read the original, because the question
    mark is the strongest signal there is.
    """
    return " ".join(line.lower().split()).rstrip(".!,;:")


def _is_remark(normalised: str) -> bool:
    """Whether the whole line is acknowledgement and nothing else.

    Clause by clause, because people thank you in two breaths: "thanks, that
    helps" is two acknowledgements and no instruction. Every clause has to be in
    the set, which is what keeps "thanks, now add tags" an instruction — the
    second clause is not an acknowledgement, so the line is not one.
    """
    clauses = [c.strip(" .!") for c in re.split(r"[,;]| - ", normalised)]
    clauses = [c for c in clauses if c]
    return bool(clauses) and all(c in _REMARKS for c in clauses)


def _first_words(line: str, count: int) -> list[str]:
    return _WORD.findall(line.lower())[:count]


def _strip_politeness(lowered: str) -> str:
    """Remove one leading request phrase, so the verb after it can be read."""
    for prefix in _POLITE:
        if lowered.startswith(prefix + " "):
            return lowered[len(prefix) + 1 :].lstrip()
    return lowered


def classify(line: str) -> Intent:
    """Which lane a line of conversation belongs in.

    Order matters, and it is the order of confidence:

    1. an acknowledgement from the closed set is a remark;
    2. a politely-phrased instruction is an instruction, question mark and all;
    3. a proposal that opens like a question is an instruction;
    4. a question mark, or a question opener, is a question;
    5. anything else is an instruction.
    """
    text = line.strip()
    if not text:
        return Intent.REMARK

    normalised = _normalise(text)
    if _is_remark(normalised):
        return Intent.REMARK

    lowered = " ".join(text.lower().split())

    # "can you add tags?" — the question mark is grammar, not a question.
    stripped = _strip_politeness(lowered)
    if stripped != lowered:
        head = _first_words(stripped, 1)
        if head and head[0] in _BUILD_VERBS:
            return Intent.BUILD

    if lowered.startswith(_PROPOSALS):
        return Intent.BUILD

    if text.endswith("?"):
        return Intent.ASK

    words = _first_words(text, 2)
    if words and words[0] in _ASK_OPENERS:
        # "show" and "tell" only open a question with "me": "tell me what the
        # server does" asks, "tell the user their name is required" instructs.
        return Intent.ASK
    if words[:1] in (["tell"], ["show"], ["walk"]) and words[1:2] == ["me"]:
        return Intent.ASK

    return Intent.BUILD
