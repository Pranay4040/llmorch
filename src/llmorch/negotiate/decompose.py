"""Turn a task sentence into a task graph and the contract that binds it.

This is the one request the whole run depends on, and the only place where a
model decides structure rather than content. Two things follow from that.

**It must be cheap to repeat and cheaper to not repeat.** A decomposition is
cached by task text, so re-running the same build costs zero requests — which
matters when the model best suited to planning is also the one with 250 requests
a day.

**Its output is untrusted and is validated hard.** Everything downstream trusts
the graph: output paths reach the filesystem, node ids key the checkpoint, roles
select the fallback chain. So nothing here is taken on faith — ids are
regenerated if they collide, dangling dependencies are dropped, cycles are
broken by `TaskGraph.build`, unknown roles fall back through alias parsing, and
oversized nodes are clamped to what the roster can actually serve. A
decomposition that cannot be repaired into a valid graph is rejected outright
rather than half-run.

The interface contract emitted alongside the nodes is the coordination
mechanism for the entire system: it is the only thing every model sees, and it
is why a frontend from one vendor works against a backend from another without
the two ever exchanging a message.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..errors import EngineError, LLMOrchError
from ..engine.salvage import extract_json
from ..registry.manifest import Manifest
from ..engine.smoke import INTERPRETERS
from ..types import (
    ChatRequest,
    InterfaceContract,
    LaunchSpec,
    Message,
    OutputKind,
    Priority,
    Role,
    TaskNode,
    Ticket,
    Usage,
)
from .roles import parse_role

DECOMPOSE_MAX_TOKENS = 4000

# A node bigger than this cannot be served by the tightest model in the roster,
# so the decomposer is told the ceiling rather than being corrected afterwards.
DEFAULT_NODE_TOKENS = 800

_SAFE_ID = re.compile(r"[^a-z0-9_-]+")


class DecomposeError(EngineError):
    """The task could not be turned into a runnable graph."""


DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "interface": {
            "type": "object",
            "properties": {
                "runtime": {"type": "string"},
                "notes": {"type": "string"},
                "launch": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "array", "items": {"type": "string"}},
                        "port": {"type": ["integer", "null"]},
                        "ready_path": {"type": "string"},
                    },
                },
                "pages": {"type": "array", "items": {"type": "string"}},
                "routes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string"},
                            "path": {"type": "string"},
                            "accepts": {"type": "string"},
                            "returns": {"type": "string"},
                        },
                        "required": ["method", "path"],
                    },
                },
                "data_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "fields": {"type": "object"},
                        },
                        "required": ["name"],
                    },
                },
            },
            "required": ["runtime", "notes"],
        },
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "role": {"type": "string"},
                    "spec": {"type": "string"},
                    "output_path": {"type": "string"},
                    "output_kind": {"type": "string"},
                    "deps": {"type": "array", "items": {"type": "string"}},
                    "needs": {"type": "array", "items": {"type": "string"}},
                    "est_output_tokens": {"type": "integer"},
                },
                "required": ["id", "title", "role", "spec", "output_path"],
            },
        },
    },
    "required": ["interface", "nodes"],
}


# A revision reuses the node schema exactly. The interface is optional here
# because a revision answers about a change: `merge_interfaces` folds whatever
# comes back into the standing contract rather than replacing it.
REVISE_SCHEMA = {
    "type": "object",
    "properties": {
        "interface": DECOMPOSE_SCHEMA["properties"]["interface"],
        "nodes": DECOMPOSE_SCHEMA["properties"]["nodes"],
    },
    "required": ["nodes"],
}


def build_revise_prompt(
    instruction: str,
    *,
    memory: str,
    interface_text: str,
    max_nodes: int,
    max_node_tokens: int,
    roles: list[str],
) -> tuple[str, str]:
    """Return (system, user) for a change to a project that already exists.

    The difference from planning fresh is the instruction to emit *only* what
    changes. A revision that re-plans the whole build would rewrite six files to
    add a field to one, spending the quota of a new project to make a small
    edit — and every rewritten file is a fresh chance for a model to disagree
    with the ones it is not rewriting.
    """
    system = f"""\
You are changing a project that already exists. Several different models will \
carry out the change, each writing exactly one whole file, none of them able to \
talk to each other.

Emit ONLY the nodes whose files must be written or rewritten for this change. A \
file that does not need to change must not appear. Reusing an existing file's \
`output_path` means that file is rewritten completely — there are no patches, \
so a node that rewrites a file must produce the whole thing, including the \
parts that are not changing.

Hard constraints — properties of the machine, not preferences:

- At most {max_nodes} nodes, and no node may need more than {max_node_tokens} \
tokens of output. A node over the ceiling can never be run at all.
- `role` must be one of: {", ".join(roles)}.
- `deps` lists node ids that must finish first, and may name nodes from earlier \
turns. No cycles.
- `needs` lists what a node reads from upstream, as "<node_id>.summary". You may \
name a node from an earlier turn; you will get its summary, never its contents.
- `est_output_tokens` must cover the WHOLE file being rewritten, not the size of \
the change.

In `interface`, emit only what the change adds or alters. What you leave out is \
kept as it is, so there is no need to restate the parts that do not move.

Whatever you add there, you must also staff. A route you declare needs a node \
that serves it; a page you declare needs a node that writes it. That the file \
which would serve it already exists is not enough — if it does not serve the \
route today, the node that rewrites it is one of the nodes you emit. A contract \
promising what no node delivers is not a smaller change, it is a broken one: it \
fails the cross-artifact check, and the route returns 404 when the project is \
run.

Reply with ONLY a JSON object. No prose, no code fence.
"""
    user = (
        f"{memory}\n\n{interface_text}\n\n"
        f"## The change now being asked for\n\n{instruction}\n\n"
        "Produce the nodes this change requires."
    )
    return system, user


def build_decompose_prompt(
    task: str, *, max_nodes: int, max_node_tokens: int, roles: list[str]
) -> tuple[str, str]:
    """Return (system, user).

    The prompt states the constraints as facts about the machine rather than as
    requests, because the model cannot negotiate them: a node larger than the
    ceiling is unservable no matter how good the plan is, and a role outside the
    taxonomy has no fallback chain behind it.
    """
    system = f"""\
You are planning a build that will be carried out by several different models \
working in parallel, each writing exactly one file, none of them able to talk \
to each other. Your output is the only coordination they get.

Hard constraints — these are properties of the machine, not preferences:

- At most {max_nodes} nodes. Fewer is better if the work genuinely divides that way.
- Each node produces ONE file, and no node may need more than \
{max_node_tokens} tokens of output. Split a large file into several nodes rather \
than exceeding this; a node over the ceiling can never be run at all.
- `est_output_tokens` is a real budget, not a formality: the file is cut off at \
roughly twice it, and a cut-off file costs a wasted request to discover. Size it \
for the finished file. Test suites, complete pages, and anything that enumerates \
cases run far longer than they look — a test file for a small module is rarely \
under 1,200 tokens.
- `role` must be one of: {", ".join(roles)}.
- `deps` lists node ids that must finish first. No cycles.
- `needs` lists what a node reads from upstream, as "<node_id>.summary". \
Downstream nodes receive short summaries, never whole files.

The `interface` you emit is handed verbatim to every node. It is what lets a \
file written by one model work against a file written by another. Put in it \
everything two files would need to agree on: routes, data shapes, page names, \
and `runtime` — the OS, interpreter, working directory and launch command the \
result must actually work under. Be specific; a file is only correct relative \
to where it runs.

Also emit `launch`, which is `runtime` in a form a program can act on, so the \
result can be started and checked rather than only written:

- `command`: argv as a list, e.g. ["python", "server.py"] or ["node", "server.js"]. \
The first entry must be an interpreter — one of: {", ".join(sorted(INTERPRETERS))}. \
Every other entry naming a file must name a file one of your nodes produces.
- `port`: the port it listens on, as an integer.
- `ready_path`: a path that answers once it is up, e.g. "/" or "/health".

Emit `launch` only if the build is a thing that can be started and left running. \
For a library, a script that exits, or a document, leave it out.

Reply with ONLY a JSON object. No prose, no code fence.
"""
    user = f"The task: {task}\n\nProduce the interface contract and the nodes."
    return system, user


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _clean_id(raw: str, used: set[str], index: int) -> str:
    """A node id is used as a dict key, a checkpoint key and a prompt marker, so
    it is normalised rather than trusted."""
    candidate = _SAFE_ID.sub("", str(raw).strip().lower().replace(" ", "_"))[:32]
    if not candidate:
        candidate = f"n{index}"
    base = candidate
    suffix = 2
    while candidate in used:
        candidate = f"{base}{suffix}"
        suffix += 1
    return candidate


def _output_kind(raw: Any, output_path: str) -> OutputKind:
    try:
        return OutputKind(str(raw).strip().lower())
    except ValueError:
        pass
    lowered = output_path.lower()
    if lowered.endswith((".py", ".js", ".ts", ".mjs", ".rb", ".go")):
        return OutputKind.CODE
    if lowered.endswith(".sql"):
        return OutputKind.SCHEMA
    if lowered.endswith((".md", ".txt")):
        return OutputKind.TEXT
    return OutputKind.TEXT


def parse_interface(payload: dict[str, Any]) -> InterfaceContract:
    raw = payload.get("interface")
    if not isinstance(raw, dict):
        return InterfaceContract()
    routes = tuple(r for r in (raw.get("routes") or []) if isinstance(r, dict))
    models = tuple(m for m in (raw.get("data_models") or []) if isinstance(m, dict))
    pages = tuple(str(p) for p in (raw.get("pages") or []) if isinstance(p, str))
    return InterfaceContract(
        routes=routes,
        data_models=models,
        pages=pages,
        runtime=str(raw.get("runtime") or "")[:2000],
        launch=LaunchSpec.from_payload(raw.get("launch")),
        notes=str(raw.get("notes") or "")[:4000],
    )


def parse_nodes(
    payload: dict[str, Any], *, max_nodes: int, max_node_tokens: int
) -> list[TaskNode]:
    """Turn the model's node list into TaskNodes, repairing what can be repaired.

    Repair rather than reject, up to a point: a plan with one malformed
    dependency is still a good plan, and re-asking costs a request from the
    scarcest budget in the system. What cannot be repaired — no usable nodes at
    all — raises.
    """
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise DecomposeError("the decomposition contained no nodes")

    used: set[str] = set()
    nodes: list[TaskNode] = []
    id_map: dict[str, str] = {}

    for index, raw in enumerate(raw_nodes[:max_nodes], start=1):
        if not isinstance(raw, dict):
            continue
        output_path = str(raw.get("output_path") or "").strip()
        spec = str(raw.get("spec") or "").strip()
        if not output_path or not spec:
            continue

        node_id = _clean_id(raw.get("id") or f"n{index}", used, index)
        used.add(node_id)
        id_map[str(raw.get("id") or "").strip()] = node_id

        tokens = raw.get("est_output_tokens")
        try:
            tokens = int(tokens)
        except (TypeError, ValueError):
            tokens = DEFAULT_NODE_TOKENS
        # Clamped, not rejected: an over-ambitious estimate is the decomposer
        # being wrong about size, and the node itself may still be fine.
        tokens = max(128, min(tokens, max_node_tokens))

        nodes.append(
            TaskNode(
                id=node_id,
                title=str(raw.get("title") or node_id)[:120],
                role=parse_role(str(raw.get("role") or "")),
                spec=spec[:4000],
                output_path=output_path,
                output_kind=_output_kind(raw.get("output_kind"), output_path),
                deps=tuple(str(d) for d in (raw.get("deps") or []) if isinstance(d, str)),
                needs=tuple(
                    str(n) for n in (raw.get("needs") or []) if isinstance(n, str)
                ),
                est_output_tokens=tokens,
            )
        )

    if not nodes:
        raise DecomposeError("no usable nodes survived validation")

    # Remap dependencies onto the cleaned ids, dropping any that point nowhere.
    # TaskGraph.build would drop dangling deps anyway; doing it here keeps the
    # remapping and the dropping in one place where both are visible.
    known = {n.id for n in nodes}
    repaired: list[TaskNode] = []
    import dataclasses

    for node in nodes:
        deps = tuple(
            dict.fromkeys(
                id_map.get(d, _SAFE_ID.sub("", d.strip().lower().replace(" ", "_")))
                for d in node.deps
            )
        )
        deps = tuple(d for d in deps if d in known and d != node.id)
        needs = tuple(
            n for n in node.needs if n.split(".")[0] in known
        )
        repaired.append(dataclasses.replace(node, deps=deps, needs=needs))
    return repaired


def plan_signature(task: str, manifest: Manifest) -> str:
    """Cache key: the task, plus the roster that will execute it.

    The roster is included because a plan is sized against it. A graph split for
    a model with a 4,096-token ceiling is not the right graph once a model with
    65,536 joins the roster.
    """
    roster = ",".join(sorted(m.id for m in manifest.enabled_models))
    blob = f"{task.strip().lower()}|{roster}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Decomposition:
    nodes: list[TaskNode]
    interface: InterfaceContract
    model_id: str = ""
    usage: Usage = None  # type: ignore[assignment]
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        return {
            "model_id": self.model_id,
            "interface": {
                "routes": list(self.interface.routes),
                "data_models": list(self.interface.data_models),
                "pages": list(self.interface.pages),
                "runtime": self.interface.runtime,
                "launch": self.interface.launch.to_dict(),
                "notes": self.interface.notes,
            },
            "nodes": [
                {
                    **dataclasses.asdict(n),
                    "role": n.role.value,
                    "output_kind": n.output_kind.value,
                    "split_hint": n.split_hint.value,
                    "deps": list(n.deps),
                    "needs": list(n.needs),
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decomposition:
        interface = parse_interface(data)
        nodes = parse_nodes(
            data, max_nodes=len(data.get("nodes") or []) or 1, max_node_tokens=10**6
        )
        return cls(
            nodes=nodes,
            interface=interface,
            model_id=str(data.get("model_id") or ""),
            cached=True,
        )


async def decompose(
    task: str,
    *,
    deps,
    model_id: str,
    max_nodes: int = 10,
) -> Decomposition:
    """Ask one model to plan the build.

    `deps` is a WorkerDeps: the same governed path every other request takes, so
    planning draws on the same admission control and lands in the same ledger.
    Planning runs at HIGH priority — it is the one request the entire run
    depends on, and it is exactly what the reserve exists for.
    """
    manifest: Manifest = deps.manifest
    model = manifest.model(model_id)
    provider_name = manifest.vendor_of(model_id)

    # The ceiling is the tightest model in the roster, not this one: the plan
    # has to be executable by whoever ends up with each node.
    max_node_tokens = min(m.max_output for m in manifest.enabled_models)

    system, user = build_decompose_prompt(
        task,
        max_nodes=max_nodes,
        max_node_tokens=max_node_tokens,
        roles=[r.value for r in Role],
    )
    est_prompt = deps.estimator.estimate_prompt(
        system=system, messages=[user], provider=provider_name
    )
    max_tokens = min(
        model.max_output, max(model.min_output_tokens, DECOMPOSE_MAX_TOKENS)
    )

    ticket = deps.governor.try_acquire(
        model_id, est_prompt, max_tokens, priority=Priority.HIGH
    )
    if not isinstance(ticket, Ticket):
        raise DecomposeError(
            f"cannot plan with {model_id}: {getattr(ticket, 'reason', 'refused')}"
        )

    request = ChatRequest(
        model_id=model_id,
        messages=(Message("user", "[decompose]\n" + user),),
        system=system,
        max_tokens=max_tokens,
        json_schema=DECOMPOSE_SCHEMA if model.supports_json_schema else None,
    )

    try:
        response = await deps.registry.get(model_id).chat(request)
    except LLMOrchError as exc:
        deps.governor.release(ticket, "decompose failed")
        raise DecomposeError(f"{model_id} could not plan the task: {exc}") from exc

    deps.governor.commit(ticket, response.usage)
    deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
    if response.rate_limit:
        deps.governor.sync_from_headers(model_id, response.rate_limit)

    payload = extract_json(response.text)
    if not isinstance(payload, dict):
        raise DecomposeError(
            f"{model_id} did not return a usable plan (no JSON object in the reply)"
        )

    return Decomposition(
        nodes=parse_nodes(payload, max_nodes=max_nodes, max_node_tokens=max_node_tokens),
        interface=parse_interface(payload),
        model_id=model_id,
        usage=response.usage,
    )


async def revise(
    instruction: str,
    *,
    deps,
    model_id: str,
    memory: str,
    interface_text: str,
    max_nodes: int = 10,
) -> Decomposition:
    """Ask one model what must change, given what already exists.

    Same governed path, same priority and same failure handling as `decompose` —
    a revision is the request the rest of the turn depends on, exactly as the
    first plan was.
    """
    manifest: Manifest = deps.manifest
    model = manifest.model(model_id)
    provider_name = manifest.vendor_of(model_id)
    max_node_tokens = min(m.max_output for m in manifest.enabled_models)

    system, user = build_revise_prompt(
        instruction,
        memory=memory,
        interface_text=interface_text,
        max_nodes=max_nodes,
        max_node_tokens=max_node_tokens,
        roles=[r.value for r in Role],
    )
    est_prompt = deps.estimator.estimate_prompt(
        system=system, messages=[user], provider=provider_name
    )
    max_tokens = min(
        model.max_output, max(model.min_output_tokens, DECOMPOSE_MAX_TOKENS)
    )

    ticket = deps.governor.try_acquire(
        model_id, est_prompt, max_tokens, priority=Priority.HIGH
    )
    if not isinstance(ticket, Ticket):
        raise DecomposeError(
            f"cannot plan the change with {model_id}: "
            f"{getattr(ticket, 'reason', 'refused')}"
        )

    request = ChatRequest(
        model_id=model_id,
        messages=(Message("user", "[revise]\n" + user),),
        system=system,
        max_tokens=max_tokens,
        json_schema=REVISE_SCHEMA if model.supports_json_schema else None,
    )

    try:
        response = await deps.registry.get(model_id).chat(request)
    except LLMOrchError as exc:
        deps.governor.release(ticket, "revise failed")
        raise DecomposeError(f"{model_id} could not plan the change: {exc}") from exc

    deps.governor.commit(ticket, response.usage)
    deps.estimator.observe(provider_name, response.usage.prompt_tokens, est_prompt)
    if response.rate_limit:
        deps.governor.sync_from_headers(model_id, response.rate_limit)

    payload = extract_json(response.text)
    if not isinstance(payload, dict):
        raise DecomposeError(
            f"{model_id} did not return a usable change (no JSON object in the reply)"
        )

    # "Nothing needs to change" is a legitimate answer to a change request, and
    # a different thing from a malformed plan. Planning fresh keeps the stricter
    # rule: a *build* that produced no nodes did fail.
    if isinstance(payload.get("nodes"), list) and not payload["nodes"]:
        return Decomposition(
            nodes=[],
            interface=parse_interface(payload),
            model_id=model_id,
            usage=response.usage,
        )

    return Decomposition(
        nodes=parse_nodes(payload, max_nodes=max_nodes, max_node_tokens=max_node_tokens),
        interface=parse_interface(payload),
        model_id=model_id,
        usage=response.usage,
    )


def pick_planner(manifest: Manifest, candidates: list[str]) -> str | None:
    """Whoever is best at planning and is still available.

    Uses the declared planning affinity rather than a hardcoded name, so adding
    a stronger planner to the manifest is enough to change who plans.
    """
    eligible = [m for m in candidates if m in {x.id for x in manifest.enabled_models}]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda m: (
            manifest.model(m).affinity(Role.PLANNING),
            manifest.model(m).quality_prior,
        ),
    )
