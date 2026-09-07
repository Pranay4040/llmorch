"""The setup page: choose the roster and the run mode once, in a browser.

Separate from `dashboard` on purpose. The dashboard is read-only and says so as
the reason it needs no authentication; this one writes settings and API keys, so
it carries a per-launch token and a Host check that the dashboard does not need.
Keeping them apart keeps that argument true of each.
"""

from .server import DEFAULT_PORT, ConfigureError, build_server, new_token, serve, snapshot

__all__ = [
    "ConfigureError",
    "DEFAULT_PORT",
    "build_server",
    "new_token",
    "serve",
    "snapshot",
]
