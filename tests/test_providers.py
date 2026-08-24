"""Provider adapter and header parsing.

Every test here runs against a fake transport. That is the point: the failure
modes worth pinning down — 429s that mean tomorrow, truncated generations,
malformed 200s — are precisely the ones that cannot be summoned on demand from
a live provider, and rehearsing them against Gemini's 250 requests a day would
exhaust the quota before the behaviour was covered.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from llmorch.errors import ProviderError, RateLimited, SchemaInvalid, TransportError
from llmorch.providers.base import Provider
from llmorch.providers.headers import (
    looks_daily,
    observed_limits,
    parse_duration,
    parse_rate_limit,
    parse_retry_after,
)
from llmorch.providers.openai_compat import (
    HttpResponse,
    OpenAICompatProvider,
    build_provider,
)
from llmorch.registry.manifest import load_manifest
from llmorch.types import ChatRequest, Message

MODEL = "groq/gpt-oss-120b"
WIRE = "openai/gpt-oss-120b"


def _provider(transport, **kw) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="sk-not-a-real-key",
        wire_names={MODEL: WIRE},
        transport=transport,
        **kw,
    )


def _request(**kw) -> ChatRequest:
    defaults = dict(
        model_id=MODEL,
        messages=(Message("user", "write a function"),),
        max_tokens=512,
        system="you are a model",
    )
    return ChatRequest(**{**defaults, **kw})


def _ok_body(text="hello", finish="stop", **usage) -> str:
    return json.dumps(
        {
            "model": WIRE,
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, **usage},
        }
    )


def _chat(provider, request=None):
    return asyncio.run(provider.chat(request or _request()))


# ==========================================================================
# Duration parsing
# ==========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),  # Groq's compound form
        ("521ms", 0.521),
        ("1h30m", 5400.0),
        ("30", 30.0),  # bare number means seconds
        ("0s", 0.0),
    ],
)
def test_parse_duration_handles_every_provider_format(text, expected):
    assert parse_duration(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [None, "", "   ", "soon", "later today"])
def test_parse_duration_returns_none_rather_than_guessing(text):
    # A wrong duration is worse than no duration: the governor would trust it.
    assert parse_duration(text) is None


def test_retry_after_accepts_an_http_date():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    later = now + timedelta(seconds=90)
    stamp = later.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(stamp, now=now) == pytest.approx(90.0, abs=1.0)


def test_retry_after_never_returns_a_negative_wait():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    past = (now - timedelta(hours=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(past, now=now) == 0.0


# ==========================================================================
# Rate-limit snapshots
# ==========================================================================


def test_headers_are_matched_case_insensitively():
    snap = parse_rate_limit({"X-RateLimit-Remaining-Requests": "1200"})
    assert snap.remaining_requests == 1200


def test_absent_headers_leave_every_field_none():
    # Gemini's compatibility endpoint sends no rate-limit headers at all.
    # Empty must mean "the server said nothing", so local counters stay in
    # charge — not "the server said zero".
    snap = parse_rate_limit({})
    assert snap.remaining_requests is None
    assert snap.remaining_tokens is None
    assert snap.retry_after_s is None
    assert snap.daily_limit_hit is False


def test_daily_flag_is_only_considered_on_a_429():
    headers = {"x-ratelimit-remaining-requests": "0", "retry-after": "7200"}
    assert parse_rate_limit(headers, status=200).daily_limit_hit is False
    assert parse_rate_limit(headers, status=429).daily_limit_hit is True


def test_daily_cap_recognised_from_the_error_text():
    snap = parse_rate_limit(
        {"retry-after": "2"},
        status=429,
        body=json.dumps({"error": {"message": "Limit 14400, used 14400 per day"}}),
    )
    assert snap.daily_limit_hit is True


def test_per_minute_429_is_not_mistaken_for_a_daily_one():
    # The expensive misread: benching a healthy model all day over a 7s pause.
    snap = parse_rate_limit(
        {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7.66s"},
        status=429,
        body=json.dumps({"error": {"message": "rate limit reached for requests"}}),
    )
    assert snap.daily_limit_hit is False
    assert snap.reset_requests_s == pytest.approx(7.66)


def test_gemini_free_tier_quota_error_reads_as_daily():
    snap = parse_rate_limit(
        {},
        status=429,
        body=json.dumps(
            {"error": {"message": "Quota exceeded for generate_content_free_tier"}}
        ),
    )
    assert snap.daily_limit_hit is True


def test_long_reset_horizon_implies_daily_without_any_text():
    assert looks_daily(remaining_requests=0, reset_requests_s=7200.0) is True
    assert looks_daily(remaining_requests=0, reset_requests_s=45.0) is False


def test_observed_limits_reads_what_a_provider_volunteers():
    # How the undocumented free tiers (NIM, Mistral) get characterised.
    assert observed_limits(
        {"x-ratelimit-limit-requests": "30", "x-ratelimit-limit-tokens": "6000"}
    ) == {"rpm": 30, "tpm": 6000}
    assert observed_limits({}) == {}


# ==========================================================================
# The adapter: happy path
# ==========================================================================


def test_adapter_satisfies_the_provider_protocol():
    assert isinstance(_provider(lambda r: HttpResponse(200, {}, _ok_body())), Provider)


def test_request_is_built_in_the_openai_wire_shape():
    seen = {}

    def transport(request):
        seen.update(json.loads(request.body))
        seen["url"] = request.url
        seen["headers"] = dict(request.headers)
        return HttpResponse(200, {}, _ok_body())

    _chat(_provider(transport))

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["model"] == WIRE, "the id must be translated to the vendor's wire name"
    assert seen["max_tokens"] == 512
    assert seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "you are a model"}
    assert seen["messages"][1] == {"role": "user", "content": "write a function"}
    assert seen["headers"]["Authorization"].startswith("Bearer ")


def test_base_url_with_a_trailing_slash_does_not_double_up():
    # Gemini's base_url ends in "/" in models.yaml; Groq's does not.
    seen = {}

    def transport(request):
        seen["url"] = request.url
        return HttpResponse(200, {}, _ok_body())

    provider = OpenAICompatProvider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="k",
        transport=transport,
    )
    _chat(provider, _request(model_id="gemini/3.6-flash"))
    assert seen["url"].endswith("/v1beta/openai/chat/completions")
    assert "//chat" not in seen["url"]


def test_max_tokens_field_can_be_renamed_per_provider():
    seen = {}

    def transport(request):
        seen.update(json.loads(request.body))
        return HttpResponse(200, {}, _ok_body())

    _chat(_provider(transport, max_tokens_field="max_completion_tokens"))
    assert seen["max_completion_tokens"] == 512
    assert "max_tokens" not in seen


def test_successful_response_is_mapped_onto_chat_response():
    headers = {
        "x-ratelimit-remaining-requests": "14399",
        "x-ratelimit-reset-tokens": "7.66s",
    }
    response = _chat(_provider(lambda r: HttpResponse(200, headers, _ok_body("code"))))

    assert response.text == "code"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 4
    assert response.truncated is False
    assert response.rate_limit.remaining_requests == 14399
    assert response.model_reported == WIRE


def test_reasoning_tokens_are_counted_toward_the_token_budget():
    # Providers bill reasoning tokens but exclude them from completion_tokens,
    # so a governor that ignored them would under-count the models that spend
    # the most.
    body = _ok_body(completion_tokens_details={"reasoning_tokens": 900})
    response = _chat(_provider(lambda r: HttpResponse(200, {}, body)))

    assert response.usage.reasoning_tokens == 900
    assert response.usage.total_tokens == 10 + 4 + 900


def test_finish_reason_length_marks_the_response_truncated():
    body = _ok_body(text="def handler(", finish="length")
    assert _chat(_provider(lambda r: HttpResponse(200, {}, body))).truncated is True


def test_missing_usage_block_degrades_to_zeros_rather_than_raising():
    body = json.dumps(
        {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
    )
    assert _chat(_provider(lambda r: HttpResponse(200, {}, body))).usage.total_tokens == 0


# ==========================================================================
# The adapter: failure mapping
#
# `is_retryable` is what the failover ladder branches on, so each status has to
# land on the right side of it.
# ==========================================================================


def test_429_becomes_a_retryable_rate_limit_carrying_its_horizon():
    body = json.dumps({"error": {"message": "rate limit reached"}})
    provider = _provider(lambda r: HttpResponse(429, {"retry-after": "3"}, body))

    with pytest.raises(RateLimited) as caught:
        _chat(provider)

    assert caught.value.is_retryable is True
    assert caught.value.retry_after_s == 3.0
    assert caught.value.daily is False


def test_daily_429_is_flagged_so_the_scheduler_stops_for_the_day():
    body = json.dumps({"error": {"message": "Limit 14400 requests per day"}})
    provider = _provider(lambda r: HttpResponse(429, {}, body))

    with pytest.raises(RateLimited) as caught:
        _chat(provider)
    assert caught.value.daily is True


def test_401_is_not_retryable_and_never_echoes_the_key():
    provider = _provider(
        lambda r: HttpResponse(401, {}, json.dumps({"error": {"message": "bad key"}}))
    )

    with pytest.raises(ProviderError) as caught:
        _chat(provider)

    assert caught.value.is_retryable is False
    assert "sk-not-a-real-key" not in str(caught.value)


def test_404_names_the_wire_name_as_the_thing_to_check():
    # The one failure M2 most expects: models.yaml's unverified wire names.
    provider = _provider(lambda r: HttpResponse(404, {}, '{"error":"no such model"}'))

    with pytest.raises(ProviderError) as caught:
        _chat(provider)

    assert caught.value.is_retryable is False
    assert "wire_name" in str(caught.value)


def test_400_is_terminal_because_the_request_itself_is_wrong():
    provider = _provider(lambda r: HttpResponse(400, {}, '{"error":"bad request"}'))
    with pytest.raises(ProviderError) as caught:
        _chat(provider)
    assert caught.value.is_retryable is False


def test_5xx_is_a_retryable_transport_error():
    provider = _provider(lambda r: HttpResponse(503, {}, "upstream unavailable"))
    with pytest.raises(TransportError) as caught:
        _chat(provider)
    assert caught.value.is_retryable is True


def test_a_200_that_is_not_json_is_a_schema_failure():
    provider = _provider(lambda r: HttpResponse(200, {}, "<html>gateway</html>"))
    with pytest.raises(SchemaInvalid):
        _chat(provider)


def test_a_200_with_no_choices_is_malformed_not_empty():
    provider = _provider(lambda r: HttpResponse(200, {}, json.dumps({"choices": []})))
    with pytest.raises(SchemaInvalid):
        _chat(provider)


def test_an_empty_content_string_is_returned_not_raised():
    # Verification catches this for free; raising here would spend a retry
    # before Tier 0 ever got to look at it.
    body = _ok_body(text="")
    assert _chat(_provider(lambda r: HttpResponse(200, {}, body))).text == ""


def test_html_error_bodies_do_not_break_error_reporting():
    provider = _provider(lambda r: HttpResponse(502, {}, "<html>bad gateway</html>"))
    with pytest.raises(TransportError) as caught:
        _chat(provider)
    assert "bad gateway" in str(caught.value)


# ==========================================================================
# Construction from the manifest
# ==========================================================================


def test_build_provider_maps_every_model_id_to_its_wire_name():
    manifest = load_manifest()
    provider = build_provider(
        manifest.providers["groq"], manifest.models, "key", transport=lambda r: None
    )

    assert provider.wire_name("groq/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert provider.wire_name("groq/qwen3.6-27b") == "qwen/qwen3.6-27b"
    # Gemini's models belong to a different adapter instance.
    assert "gemini/3.6-flash" not in provider.wire_names
