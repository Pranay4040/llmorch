"""DAG execution.

Runs ready nodes concurrently, bounded by a global cap and — ultimately — by
the governor, which is the real limiter.

The bulk-reassignment path lives here: when a model trips its circuit breaker,
its *pending* nodes are re-run through the dispatcher with that model excluded,
so the work is redistributed in one move rather than each node discovering the
same breakage independently.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..config import RunConfig
from ..negotiate.reconcile import ReconcileInput, ReconcileResult, reconcile
from ..quota.estimator import TokenEstimator
from ..quota.governor import Governor
from ..registry.manifest import Manifest
from ..types import Assignment, NodeResult, NodeState, Priority
from .blackboard import Blackboard
from .graph import TaskGraph
from .health import HealthTracker
from .worker import WorkerDeps, execute_node


@dataclass(slots=True)
class RunOutcome:
    results: dict[str, NodeResult] = field(default_factory=dict)
    assignments: dict[str, Assignment] = field(default_factory=dict)
    reassignments: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def completed(self) -> list[str]:
        return sorted(
            n for n, r in self.results.items() if r.state is NodeState.DONE
        )

    @property
    def degraded(self) -> list[str]:
        return sorted(
            n for n, r in self.results.items() if r.state is NodeState.DEGRADED
        )

    @property
    def all_succeeded(self) -> bool:
        return not self.degraded and bool(self.results)


class Scheduler:
    def __init__(
        self,
        graph: TaskGraph,
        manifest: Manifest,
        governor: Governor,
        registry,
        *,
        config: RunConfig,
        blackboard: Blackboard,
        estimator: TokenEstimator | None = None,
        health: HealthTracker | None = None,
        sleep=asyncio.sleep,
        store=None,
    ) -> None:
        self.graph = graph
        self.manifest = manifest
        self.governor = governor
        self.registry = registry
        self.config = config
        self.blackboard = blackboard
        self.estimator = estimator or TokenEstimator()
        self.health = health or HealthTracker(
            threshold=config.circuit_breaker_threshold
        )
        self.sleep = sleep
        self.store = store
        """LedgerStore on a live run, None on a dry one. Mock traffic must not
        reach the ledger: the governor would refuse tomorrow's real requests on
        the strength of quota that was never spent."""

    # -- assignment -------------------------------------------------------

    def plan(self, *, bids=None) -> ReconcileResult:
        return reconcile(self._reconcile_input(bids or []))

    def _reconcile_input(self, bids, exclude: set[str] | None = None):
        exclude = exclude or set()
        candidates = [
            m.id
            for m in self.manifest.enabled_models
            if m.id not in exclude
            and m.id not in self.config.excluded_models
            and self.health.is_available(m.id)
        ]
        return ReconcileInput(
            graph=self.graph,
            manifest=self.manifest,
            candidates=candidates,
            bids=bids,
            quota_pressure=self._quota_pressure(),
            imbalance_tolerance=self.config.imbalance_tolerance,
        )

    def _quota_pressure(self) -> dict[str, float]:
        """0..1 per model — how close it is to its daily wall."""
        out: dict[str, float] = {}
        for model_id, head in self.governor.headroom().items():
            if head.requests_limit:
                out[model_id] = min(
                    1.0, head.requests_used / max(1, head.requests_limit)
                )
        return out

    # -- execution --------------------------------------------------------

    async def run(self, plan: ReconcileResult | None = None) -> RunOutcome:
        plan = plan or self.plan()
        outcome = RunOutcome(
            assignments=dict(plan.assignments), warnings=list(self.graph.warnings)
        )

        # Nodes no model can serve never enter the schedule.
        for node_id in plan.unassigned:
            outcome.results[node_id] = NodeResult(
                node_id=node_id,
                state=NodeState.DEGRADED,
                error="no model can serve this node within its provider's limits",
            )
        outcome.warnings.extend(plan.notes)

        deps = WorkerDeps(
            manifest=self.manifest,
            governor=self.governor,
            registry=self.registry,
            estimator=self.estimator,
            health=self.health,
            blackboard=self.blackboard,
            max_retries=self.config.max_retries,
            sleep=self.sleep,
            store=self.store,
            run_id=self.config.run_id,
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        settled: set[str] = set(outcome.results)

        while True:
            ready = [
                n
                for n in self.graph.ready_nodes(settled)
                if n in outcome.assignments
            ]
            if not ready:
                break

            async def run_one(node_id: str) -> tuple[str, NodeResult]:
                async with semaphore:
                    node = self.graph.nodes[node_id]
                    model_id = outcome.assignments[node_id].model_id
                    # A critical-path node may draw on reserved headroom.
                    priority = (
                        Priority.HIGH
                        if self.graph.dependents_of(node_id)
                        else Priority.NORMAL
                    )
                    result = await execute_node(node, model_id, deps, priority=priority)
                    return node_id, result

            for node_id, result in await asyncio.gather(
                *(run_one(n) for n in ready)
            ):
                outcome.results[node_id] = result
                self.blackboard.record(result)
                settled.add(node_id)

            self._reassign_unhealthy(outcome, settled)

        outcome.warnings.extend(self.health.events)
        return outcome

    def _reassign_unhealthy(self, outcome: RunOutcome, settled: set[str]) -> None:
        """Redistribute a broken model's pending work in one move.

        Only fires once per model — a flapping model must not trigger
        reassignment after reassignment. Replacements are re-derived through the
        dispatcher, so feasibility and fair-share are both re-checked rather
        than assumed.
        """
        for model_id in self.health.unhealthy_models:
            if not self.health.needs_reassignment(model_id):
                continue

            orphaned = [
                n
                for n, a in outcome.assignments.items()
                if a.model_id == model_id and n not in settled
            ]
            if not orphaned:
                continue

            replacement = reconcile(self._reconcile_input([], exclude={model_id}))
            moved = 0
            for node_id in orphaned:
                new = replacement.assignments.get(node_id)
                if new is not None:
                    outcome.assignments[node_id] = new
                    moved += 1
                else:
                    outcome.results[node_id] = NodeResult(
                        node_id=node_id,
                        state=NodeState.DEGRADED,
                        error=f"{model_id} is unhealthy and no replacement can serve this node",
                    )
                    settled.add(node_id)

            outcome.reassignments.append(
                f"{model_id} went unhealthy; reassigned {moved} pending node(s)"
            )
