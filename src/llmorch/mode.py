"""What kind of session this is, asked before the first thing is said.

The three modes are not three implementations. They are three settings of
machinery that already exists, which is the reason they are cheap and the reason
each one can be described honestly:

- **chat** routes every line into the question lane, so nothing is ever built.
  Already what `/ask` does per line; this is that as a standing choice.
- **agent** pins every job to one model. Already what the *Who does what* tab
  does per job; this is that for all of them at once.
- **crew** is what the system does by default: several models from different
  vendors split the work and review each other.

Asking up front is worth a prompt because the alternative is discovering the
mismatch after a request is spent — a person who wanted to ask a question and
got a six-file project has paid for the misunderstanding.

**One agent is a real reduction, and it says so.** Tier 1 review requires a
reviewer from a different vendor than the author, enforced in code because a
model tends to re-approve its own mistake. Pin every job to one model and there
is no second vendor left to ask, so review stops happening. That is a
consequence of the choice rather than a fault, and the mode description states
it rather than letting it be discovered from a report that has no review section.
"""

from __future__ import annotations

import sys
from enum import Enum


class Mode(str, Enum):
    CHAT = "chat"
    AGENT = "agent"
    CREW = "crew"


DEFAULT_MODE = Mode.CREW

#: (label, one-line summary, what it costs or gives up)
DESCRIPTIONS: dict[Mode, tuple[str, str, str]] = {
    Mode.CHAT: (
        "Chat",
        "questions only — nothing gets built",
        "one request per question, no files written",
    ),
    Mode.AGENT: (
        "One agent",
        "a single model plans and writes every file",
        "no cross-vendor review: there is no second vendor to ask",
    ),
    Mode.CREW: (
        "A crew",
        "several models split the work and review each other",
        "the default — cross-vendor review and eight cross-artifact checks",
    ),
}

ORDER = (Mode.CHAT, Mode.AGENT, Mode.CREW)


def parse(value: str | None) -> Mode | None:
    """Read a mode from a flag, a settings file, or a person's keypress.

    Accepts the number shown in the menu as well as the name, because the menu
    shows numbers and people type what they see.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.isdigit():
        index = int(text) - 1
        return ORDER[index] if 0 <= index < len(ORDER) else None
    for mode in Mode:
        if text == mode.value or text.startswith(mode.value[:3]):
            return mode
    return None


def render_menu(default: Mode = DEFAULT_MODE) -> str:
    lines = ["", "How should this session work?", ""]
    for index, mode in enumerate(ORDER, start=1):
        label, summary, note = DESCRIPTIONS[mode]
        mark = "  (default)" if mode is default else ""
        lines.append(f"  {index}  {label:<11} {summary}{mark}")
        lines.append(f"     {'':<11} {note}")
    return "\n".join(lines)


def ask(default: Mode = DEFAULT_MODE, *, reader=None) -> Mode:
    """Put the three options in front of the person and read one back.

    Nobody is asked who is not there to answer. This runs at the top of every
    session, including the ones a script pipes into and the ones a test drives,
    and a menu printed at a pipe would consume the first line of input as the
    answer to a question the pipe never saw. So a non-interactive stdin takes
    the default without printing anything; `--mode` is how a script says which.

    `reader` is looked up at call time rather than defaulted to `input` in the
    signature, because a default is bound once at import and could not then be
    replaced by a test.
    """
    if reader is None:
        if not sys.stdin or not sys.stdin.isatty():
            return default
        reader = input

    print(render_menu(default))
    try:
        answer = reader(f"Choose [1/2/3, Enter for {ORDER.index(default) + 1}]: ")
    except (EOFError, KeyboardInterrupt, OSError):
        print()
        return default
    return parse(answer) or default


def describe(mode: Mode) -> str:
    label, summary, note = DESCRIPTIONS[mode]
    return f"{label} — {summary}. {note}."
