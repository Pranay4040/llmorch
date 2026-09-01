"""Find out what a key actually buys, before anything depends on it.

Adding a provider by hand means writing down four things a person is likely to
get wrong: the base URL, the model names, the rate limits, and whether calling
it costs money. Every one of those has already been wrong once in this project —
two of three Groq wire names did not exist, Groq's published limits were out by
an order of magnitude in both directions, and Gemini's "250 a day" turned out to
be twenty a minute.

So this module asks instead. `GET /models` is free on every OpenAI-compatible
endpoint: no tokens, no completion, and on most providers it is not even
metered. That single request answers "is this base URL right", "does this key
work", and "what is actually served" — the three questions that otherwise get
answered by a 404 in the middle of a run.

What it deliberately does **not** do is call a chat endpoint. Several of these
providers bill per token, and discovery must be safe to run against a key whose
pricing nobody has checked yet.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import has_api_key, load_dotenv
from .providers.headers import parse_rate_limit_headers
from .providers.openai_compat import USER_AGENT

# Candidate endpoints for keys that are not in models.yaml yet. Several
# providers publish more than one host, so each entry may carry alternatives:
# discovery tries them in order and reports the one that answered.
CANDIDATES: dict[str, dict] = {
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "urls": ["https://openrouter.ai/api/v1"],
        "note": "fronts many vendors, including ones with no direct key here",
    },
    "cerebras": {
        "env": "CEREBRAS_API_KEY",
        "urls": ["https://api.cerebras.ai/v1"],
        "note": "free tier, very fast inference",
    },
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "urls": ["https://api.moonshot.ai/v1", "https://api.moonshot.cn/v1"],
        "note": "Kimi",
    },
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "urls": ["https://api.deepseek.com/v1"],
        "note": "paid per token",
    },
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "urls": ["https://api.mistral.ai/v1"],
        "note": "declared in models.yaml, no models yet",
    },
    "fireworks": {
        "env": "FIREWORKS_API_KEY",
        "urls": ["https://api.fireworks.ai/inference/v1"],
        "note": "paid, trial credits",
    },
    "nvidia_nim": {
        "env": "NVIDIA_NIM_API_KEY",
        "urls": ["https://integrate.api.nvidia.com/v1"],
        "note": "declared in models.yaml, no models yet",
    },
    "opencode_zen": {
        "env": "OPENCODE_ZEN_API_KEY",
        "urls": [
            "https://opencode.ai/zen/v1",
            "https://api.opencode.ai/v1",
            "https://zen.opencode.ai/v1",
        ],
        "note": "endpoint unconfirmed",
    },
    "wafer": {
        "env": "WAFER_API_KEY",
        "urls": ["https://api.wafer.ai/v1", "https://wafer.ai/api/v1"],
        "note": "vendor unidentified",
    },
}


@dataclass(slots=True)
class Discovery:
    """What one key turned out to be worth."""

    provider: str
    env_var: str
    status: str  # "ok" | "no key" | "auth" | "unreachable" | "unexpected"
    base_url: str = ""
    detail: str = ""
    models: list[str] = field(default_factory=list)
    rate_limit_headers: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(self.models)


def _get(url: str, api_key: str, timeout: float = 20.0):
    """A plain authenticated GET. Returns (status, headers, body)."""
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read().decode(
                "utf-8", errors="replace"
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, dict(exc.headers.items() if exc.headers else {}), body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {}, str(exc)


def _model_ids(body: str) -> list[str]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    entries = data.get("data") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    ids = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id"):
            ids.append(str(entry["id"]))
        elif isinstance(entry, str):
            ids.append(entry)
    return ids


def discover_provider(name: str, spec: dict) -> Discovery:
    """Try each candidate URL until one answers, and report what it said."""
    env_var = spec["env"]
    if not has_api_key(env_var):
        return Discovery(name, env_var, "no key", detail=f"{env_var} is unset")

    import os

    api_key = os.environ[env_var].strip()
    last: Discovery | None = None

    for url in spec["urls"]:
        status, headers, body = _get(f"{url.rstrip('/')}/models", api_key)
        limits = {
            k: v for k, v in headers.items() if "ratelimit" in k.lower()
        }

        if status == 200:
            models = _model_ids(body)
            return Discovery(
                name, env_var, "ok" if models else "unexpected",
                base_url=url,
                detail=f"{len(models)} models" if models else "200 but no model list",
                models=models,
                rate_limit_headers=limits,
            )
        if status in (401, 403):
            last = Discovery(
                name, env_var, "auth", base_url=url,
                detail=f"{status}: the key was rejected here",
            )
            continue
        if status == 0:
            last = Discovery(
                name, env_var, "unreachable", base_url=url, detail=body[:120]
            )
            continue
        last = Discovery(
            name, env_var, "unexpected", base_url=url,
            detail=f"HTTP {status}: {body[:120]}",
        )

    return last or Discovery(name, env_var, "unreachable", detail="no candidate answered")


def discover_all(only: set[str] | None = None) -> list[Discovery]:
    load_dotenv()
    names = sorted(CANDIDATES) if only is None else sorted(only & set(CANDIDATES))
    return [discover_provider(name, CANDIDATES[name]) for name in names]
