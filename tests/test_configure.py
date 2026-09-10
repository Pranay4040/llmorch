"""The setup page, the settings file, and the one endpoint that writes a secret.

The dashboard's tests can be about what it shows, because the dashboard cannot
change anything. This page can: it saves a roster and it writes an API key into
`.env`. So most of what follows is about refusal — what happens without the
token, from a non-loopback `Host`, with a variable no provider declares, and
with a value carrying a newline that would turn one key into two.

The rest pins the property that makes the whole thing safe to leave running: a
key goes in and cannot come back out. There is no endpoint that returns one, and
`GET /api/config` reports only whether each is set.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from llmorch import settings as settings_module
from llmorch.config import KeyRejected, load_dotenv, write_env_key
from llmorch.configure.server import (
    ConfigureError,
    build_server,
    new_token,
    snapshot,
)
from llmorch.registry.manifest import ManifestError, load_manifest
from llmorch.types import Role


# ==========================================================================
# The settings file
# ==========================================================================


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Never the developer's real `.env` or settings. This module writes both."""
    monkeypatch.setenv("LLMORCH_ENV", str(tmp_path / ".env"))
    monkeypatch.setenv("LLMORCH_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("LLMORCH_STATE_DB", str(tmp_path / "state.db"))


def test_no_file_yet_is_the_defaults_not_an_error():
    chosen = settings_module.load()
    assert chosen.live is True
    assert chosen.configured is False


def test_a_corrupt_file_is_the_defaults_not_a_crash(tmp_path):
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
    assert settings_module.load().live is True


def test_an_unreadable_file_says_so_rather_than_pretending_to_be_empty(tmp_path):
    """Falling back to the defaults is right; doing it silently is not. A file
    that cannot be parsed looks exactly like a file nobody wrote, and one of
    those is a configuration somebody made and is not getting."""
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
    chosen, complaint = settings_module.read()

    assert chosen.live is True
    assert "not valid JSON" in complaint


def test_a_byte_order_mark_does_not_discard_the_settings(tmp_path):
    """Notepad and `Set-Content -Encoding utf8` both write one. Found by saving
    this file from PowerShell and watching the roster silently widen back to
    every model."""
    (tmp_path / "settings.json").write_text(
        json.dumps({"live": False, "models": ["groq/gpt-oss-120b"]}),
        encoding="utf-8-sig",
    )

    chosen, complaint = settings_module.read()
    assert complaint == ""
    assert chosen.models == ("groq/gpt-oss-120b",)


def test_a_byte_order_mark_does_not_break_the_env_file(tmp_path, monkeypatch):
    """The same fault one file over: a BOM would otherwise become part of the
    first variable's name, which reads as a wrong key rather than a mis-encoded
    file."""
    (tmp_path / ".env").write_text("GROQ_API_KEY=works\n", encoding="utf-8-sig")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    load_dotenv(override=True)

    import os

    assert os.environ.get("GROQ_API_KEY") == "works"


def test_choices_survive_the_process():
    settings_module.Settings(
        live=False, providers=("groq",), models=("groq/gpt-oss-120b",), review="all"
    ).save()

    loaded = settings_module.load()
    assert loaded.live is False
    assert loaded.providers == ("groq",)
    assert loaded.models == ("groq/gpt-oss-120b",)
    assert loaded.review == "all"
    assert loaded.configured is True


def test_saving_is_what_makes_it_configured():
    """"The file exists" and "somebody chose this" are different facts, and only
    the second one should stop `start` saying it is running on defaults."""
    assert settings_module.Settings().configured is False
    settings_module.Settings().save()
    assert settings_module.load().configured is True


@pytest.mark.parametrize(
    "raw, field, expected",
    [
        ({"max_nodes": 10_000}, "max_nodes", 25),
        ({"max_nodes": 0}, "max_nodes", 1),
        ({"concurrency": -4}, "concurrency", 1),
        ({"concurrency": "abc"}, "concurrency", 4),
        ({"review": "everything"}, "review", "code"),
    ],
)
def test_a_value_from_the_browser_is_clamped_not_trusted(raw, field, expected):
    assert getattr(settings_module.from_dict(raw), field) == expected


def test_a_name_that_is_not_a_string_is_dropped():
    """A stringified dict would become a provider name matching nothing, which
    disables the roster rather than failing."""
    chosen = settings_module.from_dict({"providers": ["groq", {"x": 1}, "", None]})
    assert chosen.providers == ("groq",)


# ==========================================================================
# Writing a key into .env
# ==========================================================================


def test_a_key_is_added_and_then_replaced(tmp_path):
    env = tmp_path / ".env"
    write_env_key("GROQ_API_KEY", "first-value")
    assert "GROQ_API_KEY=first-value" in env.read_text(encoding="utf-8")

    write_env_key("GROQ_API_KEY", "second-value")
    body = env.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=second-value" in body
    assert "first-value" not in body
    assert body.count("GROQ_API_KEY") == 1


def test_the_comments_that_say_what_each_key_is_survive(tmp_path):
    """`.env` is hand-edited and its comments carry the free-tier limits.
    Regenerating the file from a template would be tidier and would lose them."""
    env = tmp_path / ".env"
    env.write_text(
        "# Groq  https://console.groq.com  free tier\nGROQ_API_KEY=old\n"
        "# Gemini\nGEMINI_API_KEY=keep-me\n",
        encoding="utf-8",
    )

    write_env_key("GROQ_API_KEY", "new")
    body = env.read_text(encoding="utf-8")
    assert "# Groq  https://console.groq.com  free tier" in body
    assert "GEMINI_API_KEY=keep-me" in body
    assert "GROQ_API_KEY=new" in body


def test_an_empty_value_clears_rather_than_deletes(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# Groq\nGROQ_API_KEY=old\n", encoding="utf-8")

    write_env_key("GROQ_API_KEY", "")
    body = env.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=" in body
    assert "old" not in body
    assert "# Groq" in body


def test_a_value_with_a_line_break_is_refused(tmp_path):
    """It would not be one variable with a strange value. It would be that
    variable plus whatever the next line parsed as."""
    with pytest.raises(KeyRejected):
        write_env_key("GROQ_API_KEY", "abc\nOPENROUTER_API_KEY=stolen")
    assert not (tmp_path / ".env").exists()


def test_a_variable_name_that_is_not_one_is_refused():
    with pytest.raises(KeyRejected):
        write_env_key("PATH; rm -rf /", "x")


def test_a_written_key_is_readable_by_the_loader(tmp_path, monkeypatch):
    write_env_key("GROQ_API_KEY", "round-trips")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    load_dotenv(override=True)

    import os

    assert os.environ["GROQ_API_KEY"] == "round-trips"


# ==========================================================================
# Choosing models
# ==========================================================================


def test_choosing_models_narrows_the_roster_and_the_chains():
    manifest = load_manifest()
    narrowed = manifest.restricted_to_models({"groq/gpt-oss-120b"})

    assert [m.id for m in narrowed.models] == ["groq/gpt-oss-120b"]
    assert [m.id for m in narrowed.enabled_models] == ["groq/gpt-oss-120b"]
    for role in narrowed.roles:
        assert set(narrowed.chain(role)) <= {"groq/gpt-oss-120b"}


def test_a_vendor_with_nothing_ticked_is_switched_off():
    manifest = load_manifest()
    narrowed = manifest.restricted_to_models({"groq/gpt-oss-120b"})
    assert narrowed.providers["gemini"].enabled is False


def test_a_role_left_with_no_model_is_named():
    manifest = load_manifest()
    narrowed = manifest.restricted_to_models({"groq/gpt-oss-120b"})
    unstaffed = narrowed.unstaffed_roles()

    # Whichever roles they are, every one of them has an empty chain and none of
    # the staffed ones appear.
    assert all(not narrowed.chain(role) for role in unstaffed)
    assert Role.BACKEND not in unstaffed


def test_an_unknown_model_is_refused_by_name():
    with pytest.raises(ManifestError) as excinfo:
        load_manifest().restricted_to_models({"groq/does-not-exist"})
    assert "groq/does-not-exist" in str(excinfo.value)


# ==========================================================================
# The server
# ==========================================================================


@pytest.fixture
def site():
    """A real server on an ephemeral port, torn down after the test."""
    token = new_token()
    httpd = build_server("127.0.0.1", 0, token=token)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}", token
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str, *, token: str = "", host: str | None = None):
    request = urllib.request.Request(url)
    if token:
        request.add_header("X-Llmorch-Token", token)
    if host:
        request.add_header("Host", host)
    return urllib.request.urlopen(request, timeout=10)


def _post(url: str, payload: dict, *, token: str = ""):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("X-Llmorch-Token", token)
    return urllib.request.urlopen(request, timeout=10)


def test_the_page_is_served_with_the_token(site):
    base, token = site
    with _get(f"{base}/", token=token) as response:
        assert response.status == 200
        assert b"llmorch setup" in response.read()


def test_without_the_token_nothing_is_served(site):
    base, _ = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/api/config")
    assert excinfo.value.code == 401


def test_a_write_without_the_token_is_refused(site):
    """The case the token exists for: a page on another origin can post here,
    and same-origin policy only stops it reading the reply — which somebody
    writing a key does not need to do."""
    base, _ = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/api/keys", {"GROQ_API_KEY": "planted"})
    assert excinfo.value.code == 401
    assert not settings_module.load().configured


def test_a_non_loopback_host_is_refused(site):
    """Binding to 127.0.0.1 is not enough on its own: a hostile name resolving
    there is same-origin as far as the browser is concerned."""
    base, token = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/api/config", token=token, host="evil.example.com")
    assert excinfo.value.code == 403


def test_the_config_never_carries_a_key_value(site, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-secret-value-123")
    base, token = site
    with _get(f"{base}/api/config", token=token) as response:
        body = response.read().decode("utf-8")

    assert "sk-secret-value-123" not in body
    groq = next(p for p in json.loads(body)["providers"] if p["name"] == "groq")
    assert groq["key_set"] is True
    assert "key" not in {k for k in groq if k.endswith("value")}


def test_settings_posted_from_the_page_are_saved(site):
    base, token = site
    with _post(
        f"{base}/api/settings",
        {"live": False, "review": "all", "models": ["groq/gpt-oss-120b"]},
        token=token,
    ) as response:
        assert json.loads(response.read())["ok"] is True

    saved = settings_module.load()
    assert saved.live is False
    assert saved.review == "all"
    assert saved.models == ("groq/gpt-oss-120b",)
    assert saved.configured is True


def test_a_key_posted_from_the_page_reaches_the_env_file(site, tmp_path):
    base, token = site
    with _post(f"{base}/api/keys", {"GROQ_API_KEY": "from-the-browser"}, token=token) as r:
        assert json.loads(r.read())["saved"] == ["GROQ_API_KEY"]

    assert "GROQ_API_KEY=from-the-browser" in (tmp_path / ".env").read_text("utf-8")


def test_only_variables_a_declared_provider_uses_can_be_written(site, tmp_path):
    """Otherwise this is an endpoint for writing arbitrary variables into a file
    a shell may later source."""
    base, token = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/api/keys", {"PATH": "/tmp/evil"}, token=token)

    assert excinfo.value.code == 400
    assert not (tmp_path / ".env").exists()


def test_the_error_from_a_rejected_key_never_quotes_it(site):
    base, token = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/api/keys", {"GROQ_API_KEY": "bad\nvalue"}, token=token)

    body = excinfo.value.read().decode("utf-8")
    assert "bad" not in body


def test_binding_anywhere_but_loopback_is_refused():
    with pytest.raises(ConfigureError):
        build_server("0.0.0.0", 0, token=new_token())


def test_serving_without_a_token_is_refused():
    with pytest.raises(ConfigureError):
        build_server("127.0.0.1", 0, token="")


def test_the_snapshot_describes_the_manifest():
    state = snapshot()
    assert {p["name"] for p in state["providers"]} == set(load_manifest().providers)
    assert all("key_set" in p for p in state["providers"])
    assert state["models"]


# ==========================================================================
# start
# ==========================================================================


def test_start_uses_what_the_page_saved(tmp_path, monkeypatch, capsys):
    from llmorch import __main__ as cli

    settings_module.Settings(
        live=False, providers=("groq",), review="off", max_nodes=3
    ).save()

    seen = {}
    monkeypatch.setattr(cli, "cmd_chat", lambda args: seen.update(vars(args)) or 0)

    assert cli.main(["start"]) == 0
    assert seen["live"] is False
    assert seen["providers"] == "groq"
    assert seen["review"] == "off"
    assert seen["max_nodes"] == 3
    assert "mock" in capsys.readouterr().out


def test_the_command_line_still_wins_over_the_file(monkeypatch):
    """A stored preference must never be the reason you cannot do something
    once."""
    from llmorch import __main__ as cli

    settings_module.Settings(live=True).save()

    seen = {}
    monkeypatch.setattr(cli, "cmd_chat", lambda args: seen.update(vars(args)) or 0)

    assert cli.main(["start", "--mock"]) == 0
    assert seen["live"] is False


def test_start_says_when_nothing_was_ever_configured(monkeypatch, capsys):
    from llmorch import __main__ as cli

    monkeypatch.setattr(cli, "cmd_chat", lambda args: 0)
    assert cli.main(["start"]) == 0
    assert "Nothing saved yet" in capsys.readouterr().out


def test_the_mode_can_be_chosen_on_the_page(site):
    """`Settings.mode` is reachable from the UI, not only by hand-editing JSON."""
    base, token = site
    with _post(f"{base}/api/settings", {"mode": "agent"}, token=token) as response:
        assert json.loads(response.read())["ok"] is True

    assert settings_module.load().mode == "agent"


def test_leaving_the_mode_empty_means_ask_every_time(site):
    base, token = site
    with _post(f"{base}/api/settings", {"mode": ""}, token=token) as response:
        assert json.loads(response.read())["ok"] is True

    from llmorch import mode as mode_module

    assert settings_module.load().mode == ""
    assert mode_module.parse(settings_module.load().mode) is None


def test_the_agent_model_and_the_answer_setting_round_trip(site):
    """The two settings the mode tabs exist to hold."""
    base, token = site
    with _post(
        f"{base}/api/settings",
        {"mode": "agent", "agent_model": "groq/qwen3-27b", "answer_reads_files": False},
        token=token,
    ) as response:
        assert json.loads(response.read())["ok"] is True

    saved = settings_module.load()
    assert saved.mode == "agent"
    assert saved.agent_model == "groq/qwen3-27b"
    assert saved.answer_reads_files is False


def test_answers_may_quote_files_unless_told_otherwise():
    """On by default: an answer grounded in the file beats one inferred from a
    one-line summary, and it is what the answer prompt is written around."""
    assert settings_module.Settings().answer_reads_files is True
    assert settings_module.from_dict({}).answer_reads_files is True
    assert settings_module.from_dict({"answer_reads_files": False}).answer_reads_files is False


def test_the_page_has_a_tab_for_each_mode():
    from llmorch.configure.page import PAGE

    for tab in ('data-tab="chat"', 'data-tab="agent"', 'data-tab="crew"'):
        assert tab in PAGE
    # Selecting a mode is its own control, so opening a tab to look at what a
    # mode would do does not change which one you get.
    assert 'id="pick-chat"' in PAGE and "Always use this mode" in PAGE


# ==========================================================================
# The link has to keep working
# ==========================================================================


def test_the_token_is_the_same_next_time(tmp_path):
    """It was minted per launch to begin with, which turned every bookmark and
    every reopened tab into a dead end."""
    from llmorch.configure.server import load_or_create_token

    first = load_or_create_token()
    assert load_or_create_token() == first
    assert (tmp_path / "configure-token").read_text(encoding="utf-8") == first


def test_a_token_file_that_is_not_a_token_is_replaced_not_trusted(tmp_path):
    from llmorch.configure.server import load_or_create_token

    (tmp_path / "configure-token").write_text("nope", encoding="utf-8")
    token = load_or_create_token()

    assert token != "nope"
    assert len(token) >= 20


def test_a_token_that_cannot_be_saved_still_serves_this_run(tmp_path, monkeypatch):
    """The URL stops being stable, which is where this came in — but the run
    itself must not fail over somewhere to put a file."""
    from llmorch.configure import server as configure_server

    def refuse(*_args, **_kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(configure_server.Path, "write_text", refuse)
    assert len(configure_server.load_or_create_token()) >= 20


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None  # urllib then raises the 3xx as an HTTPError


def test_a_stale_page_link_heals_itself(site):
    """A reopened tab or an old bookmark carries a dead token. For the page
    itself the server bounces it to the working URL, so the address bar just
    corrects — no error, no trip to the terminal."""
    base, token = site
    opener = urllib.request.build_opener(_NoRedirect)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        opener.open(f"{base}/?t=aStaleTokenFromAnEarlierRun", timeout=10)
    assert excinfo.value.code == 302
    assert excinfo.value.headers["Location"] == f"/?t={token}"

    # And urllib, like a browser, follows it to the real page.
    with _get(f"{base}/?t=aStaleTokenFromAnEarlierRun") as response:
        assert response.status == 200
        assert b"llmorch setup" in response.read()


def test_only_the_page_is_redirected_not_an_api_call(site):
    """A state-changing request is never bounced — it gets the 401, and the
    401 carries no token."""
    base, token = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/api/config?t=stale")

    assert excinfo.value.code == 401
    assert token not in excinfo.value.read().decode("utf-8")


def test_the_refusal_never_carries_the_real_token(site):
    base, token = site
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/api/config")

    assert token not in excinfo.value.read().decode("utf-8")


def test_a_second_server_refuses_the_port_rather_than_racing_for_it(site):
    """`allow_reuse_address` on Windows lets a second process bind an address a
    first is already listening on, with connections going to whichever wins. Two
    setup servers were live on 8788 at once and the older one answered a link
    the newer had just printed — which reads as a wrong token on a URL that is,
    as far as anyone can see, the right one."""
    from llmorch.configure.server import build_server, new_token

    base, _ = site
    port = int(base.rsplit(":", 1)[1])

    with pytest.raises(ConfigureError) as excinfo:
        build_server("127.0.0.1", port, token=new_token())

    assert "already in use" in str(excinfo.value)
