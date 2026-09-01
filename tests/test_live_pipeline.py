"""The live path, end to end, without a network.

Everything real except the socket: the actual `OpenAICompatProvider`, the
actual scheduler, the actual ledger on disk. Only the HTTP transport is
canned. That combination is the point of M2 — it exercises exactly the code
that will run against Groq, including the parts a mock provider bypasses
entirely (wire-name translation, header parsing, usage extraction, ledger
writes), while costing nothing and depending on no key.

The invariant these tests exist to hold: a live run's spend outlives the
process, and a dry run's does not.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from llmorch.config import RunConfig
from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine.blackboard import Blackboard
from llmorch.engine.graph import TaskGraph
from llmorch.engine.scheduler import Scheduler
from llmorch.providers.base import ProviderRegistry
from llmorch.providers.openai_compat import HttpResponse, build_provider
from llmorch.quota.governor import Governor
from llmorch.quota.store import open_store
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import NodeState

GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


async def _no_sleep(_seconds):
    return None


class RecordingTransport:
    """Serves the demo's canned artifacts over the real wire format.

    Keys responses by the `[node:<id>]` marker the worker embeds, exactly as
    the mock provider does — so the same run produces the same project, but by
    the route a live run actually takes.
    """

    def __init__(
        self,
        *,
        headers=None,
        fail: dict[str, HttpResponse] | None = None,
        fail_times: dict[str, int] | None = None,
    ):
        self.requests: list[dict] = []
        self.headers = headers or {}
        self.fail = fail or {}
        self.fail_times = fail_times or {}
        """node_id -> how many times to fail before succeeding. Absent means
        always, which is how a node is driven all the way to degraded."""
        self._failed: dict[str, int] = {}

    def __call__(self, request) -> HttpResponse:
        payload = json.loads(request.body)
        self.requests.append(payload)

        node_id = ""
        for message in payload["messages"]:
            for line in message["content"].splitlines():
                if line.startswith("[node:") and line.endswith("]"):
                    node_id = line[len("[node:") : -1]

        if node_id in self.fail:
            budget = self.fail_times.get(node_id)
            fired = self._failed.get(node_id, 0)
            if budget is None or fired < budget:
                self._failed[node_id] = fired + 1
                return self.fail[node_id]

        text = ARTIFACTS.get(node_id, "placeholder")
        return HttpResponse(
            status=200,
            headers=self.headers,
            body=json.dumps(
                {
                    "model": payload["model"],
                    "choices": [
                        {"message": {"content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": max(1, len(text) // 4),
                    },
                }
            ),
        )


def _harness(manifest, store, transport, *, run_id="live-test"):
    """A scheduler wired the way `llmorch run --live` wires one."""
    registry = ProviderRegistry()
    for name in ("groq", "gemini"):
        provider = build_provider(
            manifest.providers[name], manifest.models, "test-key", transport=transport
        )
        for model in manifest.models:
            if model.provider == name:
                registry.register(model.id, provider)

    graph = TaskGraph.build(build_nodes())
    scheduler = Scheduler(
        graph,
        manifest,
        Governor(manifest, clock=FakeClock()),
        registry,
        config=RunConfig(task="build a notes app", run_id=run_id, dry_run=False),
        blackboard=Blackboard(interface=INTERFACE),
        sleep=_no_sleep,
        store=store,
    )
    return scheduler, graph


# ==========================================================================
# A live run, end to end
# ==========================================================================


def test_live_run_completes_through_the_real_adapter(manifest, tmp_path):
    transport = RecordingTransport()
    with open_store(tmp_path / "l.db") as store:
        scheduler, graph = _harness(manifest, store, transport)
        outcome = asyncio.run(scheduler.run(scheduler.plan()))

    assert len(outcome.completed) == len(graph.nodes)
    assert not outcome.degraded


def test_every_request_carries_the_vendors_wire_name(manifest, tmp_path):
    # The id/wire-name split only exists on the live path; a mock never sees it.
    transport = RecordingTransport()
    with open_store(tmp_path / "l.db") as store:
        scheduler, _ = _harness(manifest, store, transport)
        asyncio.run(scheduler.run(scheduler.plan()))

    sent = {r["model"] for r in transport.requests}
    assert sent, "no requests were made"
    assert sent <= {*GROQ_MODELS, "gemini-3.6-flash"}
    assert not any(m.startswith("groq/") for m in sent), "ids leaked onto the wire"


def test_every_request_sets_max_tokens(manifest, tmp_path):
    # The invariant admission control rests on: without an explicit cap the
    # completion estimate is a guess rather than a bound.
    transport = RecordingTransport()
    with open_store(tmp_path / "l.db") as store:
        scheduler, _ = _harness(manifest, store, transport)
        asyncio.run(scheduler.run(scheduler.plan()))

    assert all(r.get("max_tokens", 0) > 0 for r in transport.requests)


def test_no_request_exceeds_groqs_per_minute_ceiling(manifest, tmp_path):
    # A verified 8,000 TPM caps a Groq request at ~7,200 tokens. Anything above
    # that is permanently unservable, so the governor must never have let it
    # through; max_output holds each model well under the line.
    transport = RecordingTransport()
    with open_store(tmp_path / "l.db") as store:
        scheduler, _ = _harness(manifest, store, transport)
        asyncio.run(scheduler.run(scheduler.plan()))

    for request in transport.requests:
        if request["model"] in GROQ_MODELS:
            assert request["max_tokens"] <= 4096


# ==========================================================================
# The ledger
# ==========================================================================


def test_a_live_run_is_recorded(manifest, tmp_path):
    with open_store(tmp_path / "l.db") as store:
        scheduler, graph = _harness(manifest, store, RecordingTransport())
        asyncio.run(scheduler.run(scheduler.plan()))

        events = store.events_for_run("live-test")
        assert len(events) == len(graph.nodes)
        assert all(e.ok for e in events)
        assert all(e.purpose == "execute" for e in events)
        assert all(e.usage.prompt_tokens == 120 for e in events)


def test_a_dry_run_is_not_recorded(manifest, tmp_path):
    # Mock traffic reaching the ledger would have the governor refuse real
    # requests tomorrow over quota that was never spent.
    from llmorch.providers.mock import MockProvider

    registry = ProviderRegistry()
    provider = MockProvider(responses=dict(ARTIFACTS))
    for model in manifest.enabled_models:
        registry.register(model.id, provider)

    with open_store(tmp_path / "l.db") as store:
        scheduler = Scheduler(
            TaskGraph.build(build_nodes()),
            manifest,
            Governor(manifest, clock=FakeClock()),
            registry,
            config=RunConfig(task="t", run_id="dry-test"),
            blackboard=Blackboard(interface=INTERFACE),
            sleep=_no_sleep,
            store=None,  # what `_setup` passes on a dry run
        )
        asyncio.run(scheduler.run(scheduler.plan()))

        assert store.events_for_run("dry-test") == ()


def test_a_transient_429_is_recorded_and_the_node_still_lands(manifest, tmp_path):
    # A 429 consumed a request slot, so the ledger has to know — even though
    # the node went on to succeed on the next attempt.
    failure = HttpResponse(
        429, {"retry-after": "1"}, json.dumps({"error": {"message": "slow down"}})
    )
    transport = RecordingTransport(fail={"schema": failure}, fail_times={"schema": 1})

    with open_store(tmp_path / "l.db") as store:
        scheduler, _ = _harness(manifest, store, transport)
        outcome = asyncio.run(scheduler.run(scheduler.plan()))

        failed = [e for e in store.events_for_run("live-test") if not e.ok]

        assert failed, "the 429 was not recorded"
        assert failed[0].http_status == 429
        assert failed[0].node_id == "schema"
        assert not outcome.degraded, "a transient 429 must not cost a node"


def test_a_transient_429_does_not_bench_the_model_for_the_run(manifest, tmp_path):
    """The bug this pair of tests was written to catch.

    A per-minute 429 is the provider enforcing a quota, not a model being
    broken. Treating it as either — a daily exhaustion, or a circuit-breaker
    failure — takes a healthy model out of every later chain. Against Groq's
    account-wide 30 RPM that happens seconds into the first fan-out, and the
    whole roster is benched while every remaining node degrades.
    """
    failure = HttpResponse(
        429, {"retry-after": "1"}, json.dumps({"error": {"message": "slow down"}})
    )
    transport = RecordingTransport(fail={"schema": failure}, fail_times={"schema": 2})

    with open_store(tmp_path / "l.db") as store:
        scheduler, graph = _harness(manifest, store, transport)
        outcome = asyncio.run(scheduler.run(scheduler.plan()))

    assert len(outcome.completed) == len(graph.nodes)
    assert not scheduler.health.unhealthy_models, (
        f"a rate limit benched {scheduler.health.unhealthy_models}"
    )


def test_a_daily_429_does_bench_the_model(manifest, tmp_path):
    # The other side of the line: a daily cap means this model is done until
    # the provider's next local midnight, so it must leave the chains.
    failure = HttpResponse(
        429,
        {},
        json.dumps({"error": {"message": "Limit 1000, used 1000 per day"}}),
    )
    transport = RecordingTransport(fail={"schema": failure}, fail_times={"schema": 1})

    with open_store(tmp_path / "l.db") as store:
        scheduler, _ = _harness(manifest, store, transport)
        asyncio.run(scheduler.run(scheduler.plan()))

    benched = [
        m
        for m in ("groq/gpt-oss-120b", "groq/gpt-oss-20b", "groq/qwen3.6-27b")
        if scheduler.health.status(m).value == "exhausted"
    ]
    assert benched, "a daily cap should take the model out of the chains"


def test_spend_from_a_live_run_survives_into_the_next_process(manifest, tmp_path):
    """The whole reason the ledger is durable.

    Run once, throw the governor away, build a fresh one from the ledger, and
    the quota is still spent — which is what stops the second run of the day
    from rediscovering the wall by walking into it.
    """
    path = tmp_path / "shared.db"

    with open_store(path) as store:
        scheduler, graph = _harness(manifest, store, RecordingTransport())
        asyncio.run(scheduler.run(scheduler.plan()))

    with open_store(path) as store:
        governor = Governor(manifest, clock=FakeClock())
        assert sum(h.requests_used for h in governor.headroom().values()) == 0

        governor.restore_daily(store.usage_by_model_today(governor.reset_timezones()))
        assert sum(h.requests_used for h in governor.headroom().values()) == len(
            graph.nodes
        )


def test_rate_limit_headers_are_believed_over_local_counting(manifest, tmp_path):
    # The server is authoritative. Groq reporting 9,000 requests already spent
    # must override a local counter that has only seen this run's handful.
    # Groq states its own ceiling alongside the remainder, and both are read:
    # deriving usage from the manifest's number instead is how two requests
    # came to look like thirteen thousand before the probe corrected it.
    transport = RecordingTransport(
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "940",
        }
    )
    with open_store(tmp_path / "l.db") as store:
        scheduler, _ = _harness(manifest, store, transport)
        asyncio.run(scheduler.run(scheduler.plan()))

        used = scheduler.governor.headroom()["groq/gpt-oss-120b"].requests_used
        assert used == 1_000 - 940
