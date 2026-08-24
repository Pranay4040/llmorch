"""The task DAG: validation, topological levels, and budget pruning.

Cycles are repaired rather than raised wherever possible. A decomposing model
will eventually emit one, and discarding an otherwise-good plan over a single
bad edge wastes the request that produced it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..errors import GraphError
from ..types import Role, TaskNode


@dataclass(slots=True)
class TaskGraph:
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    """Non-fatal repairs applied at construction, surfaced in the run report."""

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, nodes: list[TaskNode], *, repair: bool = True) -> TaskGraph:
        graph = cls()
        for node in nodes:
            if node.id in graph.nodes:
                raise GraphError(f"duplicate node id {node.id!r}")
            graph.nodes[node.id] = node

        graph._drop_dangling_deps(repair=repair)
        graph._break_cycles(repair=repair)
        return graph

    def _drop_dangling_deps(self, *, repair: bool) -> None:
        known = set(self.nodes)
        for node_id, node in list(self.nodes.items()):
            missing = [d for d in node.deps if d not in known]
            if not missing:
                continue
            if not repair:
                raise GraphError(f"node {node_id!r} depends on unknown {missing!r}")
            kept = tuple(d for d in node.deps if d in known)
            self.nodes[node_id] = _replace_deps(node, kept)
            self.warnings.append(
                f"node {node_id!r} referenced unknown dependencies {missing!r}; dropped"
            )

    def _break_cycles(self, *, repair: bool) -> None:
        """Remove back-edges until the graph is acyclic.

        Nodes are visited in insertion order, which usually matches the
        decomposer's intended sequence, so the edge removed is the one pointing
        backwards rather than an arbitrary one.
        """
        order = {node_id: i for i, node_id in enumerate(self.nodes)}
        for node_id, node in list(self.nodes.items()):
            back = [d for d in node.deps if order.get(d, -1) > order[node_id]]
            if not back:
                continue
            if not repair:
                raise GraphError(f"cycle through {node_id!r} via {back!r}")
            kept = tuple(d for d in node.deps if d not in back)
            self.nodes[node_id] = _replace_deps(node, kept)
            self.warnings.append(
                f"node {node_id!r} had back-edges {back!r}; removed to break a cycle"
            )

        if self._find_cycle():
            if not repair:
                raise GraphError("graph still contains a cycle")
            # Fallback: strip every edge that a topological sort cannot place.
            placed = set(self.levels_flat())
            for node_id, node in list(self.nodes.items()):
                if node_id not in placed:
                    self.nodes[node_id] = _replace_deps(node, ())
                    self.warnings.append(
                        f"node {node_id!r} was in an unresolvable cycle; "
                        "dependencies cleared"
                    )

    def _find_cycle(self) -> bool:
        return len(self.levels_flat()) != len(self.nodes)

    # -- traversal --------------------------------------------------------

    def levels(self) -> list[list[str]]:
        """Kahn levels: every node in a level is safe to run concurrently."""
        indegree: dict[str, int] = {n: 0 for n in self.nodes}
        dependents: dict[str, list[str]] = defaultdict(list)

        for node_id, node in self.nodes.items():
            for dep in node.deps:
                if dep in self.nodes:
                    indegree[node_id] += 1
                    dependents[dep].append(node_id)

        ready = deque(sorted(n for n, d in indegree.items() if d == 0))
        out: list[list[str]] = []
        while ready:
            level = sorted(ready)
            out.append(level)
            ready.clear()
            for node_id in level:
                for child in dependents[node_id]:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.append(child)
        return out

    def levels_flat(self) -> list[str]:
        return [n for level in self.levels() for n in level]

    def dependents_of(self, node_id: str) -> list[str]:
        return [n for n, node in self.nodes.items() if node_id in node.deps]

    def ready_nodes(self, done: set[str]) -> list[str]:
        """Nodes whose dependencies are all satisfied and which are not done."""
        return sorted(
            n
            for n, node in self.nodes.items()
            if n not in done and all(d in done for d in node.deps)
        )

    # -- budgeting --------------------------------------------------------

    def prune_to_budget(self, max_nodes: int) -> list[str]:
        """Merge same-role leaf nodes until the graph fits `max_nodes`.

        Applied before bidding, not after: the bid prompt lists every node, so
        pruning first keeps that request small. Leaves are merged because
        nothing depends on them, so collapsing them cannot break an edge.

        Returns the ids that were absorbed into others.
        """
        if len(self.nodes) <= max_nodes:
            return []

        absorbed: list[str] = []
        by_role: dict[Role, list[str]] = defaultdict(list)
        for node_id, node in self.nodes.items():
            if not self.dependents_of(node_id):
                by_role[node.role].append(node_id)

        for role, leaves in by_role.items():
            for victim in sorted(leaves)[1:]:
                if len(self.nodes) <= max_nodes:
                    break
                keeper = sorted(leaves)[0]
                if victim not in self.nodes or keeper not in self.nodes:
                    continue
                self._merge(keeper, victim)
                absorbed.append(victim)
            if len(self.nodes) <= max_nodes:
                break

        if absorbed:
            self.warnings.append(
                f"pruned {len(absorbed)} node(s) to fit a budget of {max_nodes}: "
                f"{', '.join(absorbed)}"
            )
        return absorbed

    def _merge(self, keeper_id: str, victim_id: str) -> None:
        keeper, victim = self.nodes[keeper_id], self.nodes[victim_id]
        merged_deps = tuple(
            sorted({*keeper.deps, *victim.deps} - {keeper_id, victim_id})
        )
        self.nodes[keeper_id] = TaskNode(
            id=keeper.id,
            title=f"{keeper.title} + {victim.title}",
            role=keeper.role,
            spec=f"{keeper.spec}\n\n---\n\n{victim.spec}",
            output_path=keeper.output_path,
            output_kind=keeper.output_kind,
            deps=merged_deps,
            needs=tuple(sorted({*keeper.needs, *victim.needs})),
            est_output_tokens=keeper.est_output_tokens + victim.est_output_tokens,
            split_hint=keeper.split_hint,
        )
        del self.nodes[victim_id]

    # -- inspection -------------------------------------------------------

    @property
    def total_est_tokens(self) -> int:
        return sum(n.est_output_tokens for n in self.nodes.values())

    def by_role(self) -> dict[Role, list[str]]:
        out: dict[Role, list[str]] = defaultdict(list)
        for node_id, node in self.nodes.items():
            out[node.role].append(node_id)
        return dict(out)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes


def _replace_deps(node: TaskNode, deps: tuple[str, ...]) -> TaskNode:
    return TaskNode(
        id=node.id,
        title=node.title,
        role=node.role,
        spec=node.spec,
        output_path=node.output_path,
        output_kind=node.output_kind,
        deps=deps,
        needs=node.needs,
        est_output_tokens=node.est_output_tokens,
        split_hint=node.split_hint,
    )
