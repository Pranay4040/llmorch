"""Live-provider adapter tests, run entirely off a fake wire.

Every path here is one that only appears against a real endpoint — a 429 whose
body names a daily cap, a 404 on an unverified wire name, a completion stopped
at max_tokens. Those are exactly the paths that must not be debugged live:
Gemini allows 250 requests a day, and a single afternoon of iterating on header
parsing would spend the lot.

So the transport is injected, and the assertions are about what goes *onto* the
wire and what is made of what comes back.
"""

from __future__ import annotations

import json

import pytest

from llmorch.config import RunConfig
from llmorch.demo.website import ARTIFACTS, INTERFACE, build_nodes
from llmorch.engine.blackboard import Blackboard
from llmorch.engine.graph import TaskGraph
from llmorch.engine.scheduler import Scheduler
from llmorch.errors import (
    ConfigError,
    ProviderError,
    RateLimited,
    SchemaInvalid,
    TransportError,
)
from llmorch.providers.headers import (
    looks_like_daily_limit,
    parse_duration,
    parse_rate_limit_headers,
)
from llmorch.providers.base import ProviderRegistry
from llmorch.providers.openai_compat import (
    HttpResponse,
    OpenAICompatProvider,
    build_live_registry,
)
from llmorch.quota.governor import Governor
from llmorch.quota.store import LedgerStore
from llmorch.quota.windows import FakeClock
from llmorch.registry.manifest import load_manifest
from llmorch.types import ChatRequest, Message, NodeState

GROQ = "groq/gpt-oss-120b"
GEMINI = "gemini/3.6-flash"

GROQ_HEADERS = {
    "x-ratelimit-limit-requests": "14400",
    "x-ratelimit-remaining-requests": "14370",
    "x-ratelimit-reset-requests": "2m59.56s",
    "x-ratelimit-limit-tokens": "6000",
    "x-ratelimit-remaining-tokens": "4800",
    "x-ratelimit-reset-tokens": "7.66s",
}


# ==========================================================================
# Fake wire
# ==========================================================================


class FakeTransport:
    """Canned HTTP, with every request kept for inspection."""

    def __init__(self, responses=None, *, handler=None) -> None:
        self.responses = list(responses or [])
        self.handler = handler
        self.requests: list[dict] = []

    async def post(self, url, *, headers, body, timeout_s):
        payload = json.loads(body.decode("utf-8"))
        self.requests.append(
            {"url": url, "headers": dict(headers), "payload": payload,
             "timeout_s": timeout_s}
        )
        if self.handler is not None:
            return self.handler(payload)
        if not self.responses:
            raise AssertionError("FakeTransport ran out of canned responses")
        return self.responses.pop(0)


def _completion(text: str, *, finish: str = "stop", headers=None, prompt=120, out=40):
    return HttpResponse(
        status=200,
        headers=headers if headers is not None else dict(GROQ_HEADERS),
        body=json.dumps(
            {
                "id": "chatcmpl-1",
                "model": "llama-3.3-70b-versatile",
                "choices": [
                    {"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text}}
                ],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": out,
                    "total_tokens": prompt + out,
                },
            }
        ),
    )


def _error(status: int, message: str, *, headers=None, code=None):
    return HttpResponse(
        status=status,
        headers=headers or {},
        body=json.dumps({"error": {"message": message, "code": code or "error"}}),
    )


def _provider(transport) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="sk-test-not-a-real-key",
        wire_names={GROQ: "llama-3.3-70b-versatile"},
        transport=transport,
    )


def _request(**kw) -> ChatRequest:
    return ChatRequest(
        model_id=GROQ,
        messages=(Message("user", "[node:n1]\nwrite the thing"),),
        system="you are one of several models",
        max_tokens=kw.pop("max_tokens", 512),
        **kw,
    )


# ==========================================================================
# Header parsing
# ==========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),
        ("1h2m3s", 3723.0),
        ("500ms", 0.5),
        ("88", 88.0),
        ("1d", 86400.0),
    ],
)
def test_parse_duration_handles_every_dialect(text, expected):
    assert parse_duration(text) == pytest.approx(expected)


def test_parse_duration_gives_up_quietly():
    """An unreadable header leaves the local estimate in place rather than
    poisoning the counters with a wrong number."""
    assert parse_duration(None) is None
    assert parse_duration("") is None
    assert parse_duration("soon") is None


def test_groq_headers_become_a_snapshot():
    snap = parse_rate_limit_headers(GROQ_HEADERS)
    assert snap.remaining_requests == 14370
    assert snap.limit_requests == 14400
    assert snap.remaining_tokens == 4800
    assert snap.reset_tokens_s == pytest.approx(7.66)
    assert snap.daily_limit_hit is False


def test_headers_are_case_insensitive():
    snap = parse_rate_limit_headers({"X-RateLimit-Remaining-Requests": "5"})
    assert snap.remaining_requests == 5


def test_zero_remaining_with_an_hours_long_reset_reads_as_the_daily_wall():
    """A per-minute bucket always clears within a minute. One that will not
    reset for hours is the daily cap wearing a per-request header — and the
    scheduler must not sit waiting on it."""
    snap = parse_rate_limit_headers(
        {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7h12m"}
    )
    assert snap.daily_limit_hit is True


def test_zero_remaining_with_a_short_reset_is_only_a_burst_limit():
    snap = parse_rate_limit_headers(
        {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "12s"}
    )
    assert snap.daily_limit_hit is False


def test_daily_limit_is_recognised_from_the_body_when_headers_are_silent():
    """Gemini's OpenAI-compatible endpoint returns no rate-limit headers at all;
    the fact lives in the quota id inside the error body."""
    body = json.dumps(
        {"error": {"message": "You exceeded your current quota",
                   "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}}
    )
    assert parse_rate_limit_headers({}, body=body).daily_limit_hit is True


@pytest.mark.parametrize(
    "text,expected",
    [
        ("rate limit reached: 14400 requests per day", True),
        ("GenerateRequestsPerDayPerProjectPerModel", True),
        ("daily quota exceeded", True),
        ("rate limit reached: 30 requests per minute", False),
        ("", False),
        (None, False),
    ],
)
def test_daily_versus_burst_is_read_from_the_message(text, expected):
    assert looks_like_daily_limit(text) is expected


# ==========================================================================
# What goes onto the wire
# ==========================================================================


async def test_request_carries_the_wire_name_not_the_internal_id():
    """models.yaml ids are ours; wire names are the provider's. Sending the
    wrong one is a 404 discovered mid-run, after quota has been spent."""
    transport = FakeTransport([_completion("ok")])
    await _provider(transport).chat(_request())

    payload = transport.requests[0]["payload"]
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert transport.requests[0]["url"].endswith("/v1/chat/completions")


async def test_max_tokens_is_always_sent():
    """The hard bound is what makes admission control sound rather than
    hopeful — an unbounded completion could blow a 6,000 TPM window by itself."""
    transport = FakeTransport([_completion("ok")])
    await _provider(transport).chat(_request(max_tokens=256))
    assert transport.requests[0]["payload"]["max_tokens"] == 256


async def test_system_prompt_leads_the_message_list():
    transport = FakeTransport([_completion("ok")])
    await _provider(transport).chat(_request())
    messages = transport.requests[0]["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


async def test_base_url_with_a_trailing_slash_still_joins_cleanly():
    """Gemini's compatibility base_url ends in a slash and Groq's does not."""
    provider = OpenAICompatProvider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="k",
        transport=FakeTransport([_completion("ok")]),
    )
    assert provider.endpoint("chat/completions").endswith("/v1beta/openai/chat/completions")


# ==========================================================================
# What is made of the response
# ==========================================================================


async def test_usage_and_headers_come_back_on_the_response():
    transport = FakeTransport([_completion("hello", prompt=321, out=45)])
    response = await _provider(transport).chat(_request())

    assert response.text == "hello"
    assert response.usage.prompt_tokens == 321
    assert response.usage.total_tokens == 366
    assert response.rate_limit.remaining_requests == 14370
    assert response.truncated is False


async def test_finish_reason_length_is_truncation():
    """The provider stating it stopped at the cap is the authoritative signal —
    free to read, and among the commonest free-model failures."""
    response = await _provider(
        FakeTransport([_completion("def handler(", finish="length")])
    ).chat(_request())
    assert response.truncated is True


async def test_reasoning_and_cached_tokens_are_kept():
    body = json.dumps(
        {
            "model": "x",
            "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 300},
                "prompt_tokens_details": {"cached_tokens": 60},
            },
        }
    )
    response = await _provider(
        FakeTransport([HttpResponse(200, {}, body)])
    ).chat(_request())
    # Reasoning tokens are billed and rate-limited like any other output, so
    # leaving them out would under-count the run against a TPM ceiling.
    assert response.usage.reasoning_tokens == 300
    assert response.usage.total_tokens == 420


# ==========================================================================
# Failure mapping — the ladder branches on is_retryable
# ==========================================================================


async def test_429_becomes_a_retryable_rate_limit_with_its_retry_after():
    transport = FakeTransport(
        [_error(429, "rate limit reached: 30 requests per minute",
                headers={"retry-after": "8"})]
    )
    with pytest.raises(RateLimited) as exc:
        await _provider(transport).chat(_request())
    assert exc.value.is_retryable is True
    assert exc.value.retry_after_s == pytest.approx(8.0)
    assert exc.value.daily is False


async def test_429_naming_a_daily_cap_is_flagged_as_daily():
    """`daily` is what stops the worker retrying into a wall that will not move
    until the provider's next local midnight."""
    transport = FakeTransport(
        [_error(429, "rate limit reached for model on requests per day")]
    )
    with pytest.raises(RateLimited) as exc:
        await _provider(transport).chat(_request())
    assert exc.value.daily is True


async def test_5xx_is_a_transport_error_and_worth_retrying():
    with pytest.raises(TransportError) as exc:
        await _provider(FakeTransport([_error(503, "upstream unavailable")])).chat(
            _request()
        )
    assert exc.value.is_retryable is True


async def test_401_is_terminal_and_never_echoes_the_key():
    with pytest.raises(ProviderError) as exc:
        await _provider(FakeTransport([_error(401, "Invalid API Key")])).chat(_request())
    assert exc.value.is_retryable is False
    assert "sk-test-not-a-real-key" not in str(exc.value)


async def test_404_on_an_unverified_wire_name_is_terminal():
    """This is the failure `llmorch doctor --probe` exists to find first."""
    with pytest.raises(ProviderError) as exc:
        await _provider(
            FakeTransport([_error(404, "model `qwen3-32b` does not exist")])
        ).chat(_request())
    assert exc.value.status == 404
    assert "does not exist" in str(exc.value)


async def test_html_error_page_still_produces_a_readable_message():
    """Edge proxies answer with HTML, not JSON, and a screenful of markup in the
    log helps nobody."""
    with pytest.raises(TransportError) as exc:
        await _provider(
            FakeTransport([HttpResponse(502, {}, "<html><body>Bad Gateway</body></html>")])
        ).chat(_request())
    assert len(str(exc.value)) < 250


async def test_unparseable_body_is_schema_invalid_so_salvage_runs_first():
    with pytest.raises(SchemaInvalid):
        await _provider(FakeTransport([HttpResponse(200, {}, "not json")])).chat(
            _request()
        )


async def test_empty_choices_is_schema_invalid():
    body = json.dumps({"model": "x", "choices": [], "usage": {}})
    with pytest.raises(SchemaInvalid):
        await _provider(FakeTransport([HttpResponse(200, {}, body)])).chat(_request())


# ==========================================================================
# Registry wiring
# ==========================================================================


def test_live_registry_covers_only_keyed_providers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    registry, clients = build_live_registry(load_manifest())
    assert set(clients) == {"groq"}
    assert all(m.startswith("groq/") for m in registry.model_ids)


def test_one_client_serves_every_model_on_a_provider(monkeypatch):
    """Account-scoped limits mean the models share a quota bucket; sharing one
    client keeps that visible rather than implied."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    registry, clients = build_live_registry(load_manifest(), only_providers={"groq"})
    groq_models = [m for m in registry.model_ids if m.startswith("groq/")]
    assert len(groq_models) > 1
    assert len({id(registry.get(m)) for m in groq_models}) == 1


def test_no_keys_at_all_is_a_config_error_not_a_silent_empty_run(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ConfigError):
        build_live_registry(load_manifest())


# ==========================================================================
# End to end over the fake wire
# ==========================================================================


def _artifact_handler(payload):
    """Answer each request with the canned artifact for the node it names."""
    text = "\n".join(m["content"] for m in payload["messages"])
    node_id = next(
        (
            line[len("[node:") : -1]
            for line in text.splitlines()
            if line.startswith("[node:") and line.endswith("]")
        ),
        "",
    )
    artifact = ARTIFACTS.get(node_id, "# nothing")
    return _completion(artifact, prompt=len(text) // 4, out=len(artifact) // 4)


async def test_full_run_over_the_http_adapter_records_every_call(tmp_path, monkeypatch):
    """The whole pipeline against the real adapter — governed, executed, and
    written to the ledger — without a socket being opened."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")

    manifest = load_manifest()
    transport = FakeTransport(handler=_artifact_handler)
    registry, _ = build_live_registry(manifest, transport=transport)

    store = LedgerStore(tmp_path / "state.db").open()
    graph = TaskGraph.build(build_nodes())
    governor = Governor(manifest, clock=FakeClock())
    scheduler = Scheduler(
        graph,
        manifest,
        governor,
        registry,
        config=RunConfig(task="build a notes app", run_id="wire-test"),
        blackboard=Blackboard(interface=INTERFACE),
        ledger=store,
    )

    outcome = await scheduler.run()
    assert outcome.all_succeeded, outcome.degraded
    assert all(r.state is NodeState.DONE for r in outcome.results.values())

    # One ledger row per provider call, each with the usage the wire reported.
    rows = store.recent(50, run_id="wire-test")
    assert all(r.ok and r.prompt_tokens > 0 for r in rows)

    # Execution is one call per node; Tier 1 review is a request like any other
    # and is recorded under its own purpose rather than folded into the node's.
    executed = [r for r in rows if r.purpose == "execute"]
    assert len(executed) == len(outcome.results)
    assert {r.purpose for r in rows} == {"execute", "review"}

    # And the governor's day counters agree with what was recorded.
    spent = store.run_usage("wire-test")
    assert spent.requests == len(rows)
    store.close()


async def test_a_dead_model_fails_over_to_the_other_vendor(monkeypatch):
    """A 404 wire name on one vendor must not sink the node — the whole point
    of chains spanning two vendors."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")

    manifest = load_manifest()

    def handler(payload):
        if payload["model"].startswith("gemini"):
            return _error(404, "model does not exist")
        return _artifact_handler(payload)

    transport = FakeTransport(handler=handler)
    registry, _ = build_live_registry(manifest, transport=transport)

    graph = TaskGraph.build(build_nodes())
    scheduler = Scheduler(
        graph,
        manifest,
        Governor(manifest, clock=FakeClock()),
        registry,
        config=RunConfig(task="build a notes app", run_id="failover-test"),
        blackboard=Blackboard(interface=INTERFACE),
        sleep=_no_sleep,
    )

    outcome = await scheduler.run()
    assert outcome.all_succeeded, outcome.degraded
    assert all(
        r.model_id is not None and not r.model_id.startswith("gemini")
        for r in outcome.results.values()
    )


async def _no_sleep(_seconds):
    """Backoff without the wait."""
    return None


# ==========================================================================
# The output floor
# ==========================================================================


class _RecordingProvider:
    """Captures the ChatRequest the worker built, then answers plausibly."""

    name = "recorder"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest):
        self.requests.append(request)
        from llmorch.types import ChatResponse, Usage

        return ChatResponse(
            text="body { color: #222; }\n",
            usage=Usage(prompt_tokens=100, completion_tokens=40),
            model_reported=request.model_id,
            latency_ms=5,
        )

    async def count_tokens(self, request):
        return None


async def test_a_model_with_an_output_floor_is_never_asked_for_less():
    """Gemini charges hidden thinking tokens against max_tokens without
    reporting them. Budget only for the visible answer and the reply comes back
    stopped at `length` with nothing in it — measured: 900 tokens asked, 36
    visible returned; 8,000 asked, a complete file returned."""
    from llmorch.demo.website import INTERFACE
    from llmorch.engine.blackboard import Blackboard
    from llmorch.engine.worker import WorkerDeps, execute_node
    from llmorch.engine.health import HealthTracker
    from llmorch.quota.estimator import TokenEstimator
    from llmorch.types import OutputKind, Role, TaskNode

    manifest = load_manifest()
    model = manifest.model(GEMINI)
    assert model.min_output_tokens > 0, "this test is about the floor"

    recorder = _RecordingProvider()
    registry = ProviderRegistry()
    registry.register(GEMINI, recorder)

    # A small node: doubling its estimate would land far below the floor.
    node = TaskNode(
        id="n1", title="styles", role=Role.STYLING, spec="css please",
        output_path="style.css", output_kind=OutputKind.TEXT,
        est_output_tokens=200,
    )
    deps = WorkerDeps(
        manifest=manifest, governor=Governor(manifest, clock=FakeClock()),
        registry=registry, estimator=TokenEstimator(), health=HealthTracker(),
        blackboard=Blackboard(interface=INTERFACE),
    )
    await execute_node(node, GEMINI, deps)

    assert recorder.requests, "the provider was never called"
    assert recorder.requests[0].max_tokens >= model.min_output_tokens


async def test_the_floor_never_exceeds_the_model_cap():
    """A floor above max_output would make every request unsatisfiable."""
    manifest = load_manifest()
    for model in manifest.enabled_models:
        assert model.min_output_tokens <= model.max_output


# ==========================================================================
# A stated wait outranks a keyword
#
# Found live: Gemini answers a 429 with several quota ids — some of them
# per-day — while the metric that actually tripped is per-minute, and no
# rate-limit headers at all. Reading the wrong line took a healthy model out of
# the run for the rest of the day.
# ==========================================================================

GEMINI_429_BODY = (
    '{"error": {"code": 429, "message": "You exceeded your current quota. '
    "* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    'limit: 20, model: gemini-3.6-flash Please retry in 12.398389909s.", '
    '"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}}'
)


def test_a_wait_stated_in_prose_is_still_a_wait():
    """Gemini sends no rate-limit headers; the only number it gives is in the
    sentence, and the governor needs it."""
    snap = parse_rate_limit_headers({}, body=GEMINI_429_BODY)
    assert snap.retry_after_s == pytest.approx(12.398, abs=0.01)


def test_twelve_seconds_is_not_the_daily_wall():
    """The body mentions a per-day quota id, but the metric that tripped clears
    in twelve seconds. If the server says come back shortly, waiting works."""
    assert parse_rate_limit_headers({}, body=GEMINI_429_BODY).daily_limit_hit is False


def test_a_wait_measured_in_hours_still_reads_as_daily():
    """The veto must not make the daily case unreachable."""
    body = GEMINI_429_BODY.replace("12.398389909s", "7h12m")
    assert parse_rate_limit_headers({}, body=body).daily_limit_hit is True


def test_an_explicit_header_still_wins_over_the_prose():
    snap = parse_rate_limit_headers({"retry-after": "30"}, body=GEMINI_429_BODY)
    assert snap.retry_after_s == pytest.approx(30.0)


async def test_a_quota_message_survives_long_enough_to_read():
    """The quota metric and its value sit past the length an ordinary error is
    trimmed to — and they are the only place a provider states its real limit."""
    transport = FakeTransport([HttpResponse(429, {}, GEMINI_429_BODY)])
    with pytest.raises(RateLimited) as exc:
        await _provider(transport).chat(_request())

    assert "limit: 20" in str(exc.value)
    assert exc.value.daily is False
