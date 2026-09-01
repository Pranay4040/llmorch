"""Role taxonomy helpers.

The taxonomy itself lives in `types.Role` and is deliberately closed. A learned
track record is keyed on (model, role), so free-text roles would make that
history unmatchable from one run to the next — every run would start cold.
"""

from __future__ import annotations

from ..types import OutputKind, Role

# Roles that produce code or schema, and therefore attract stricter
# verification and cross-vendor review.
CODE_ROLES: frozenset[Role] = frozenset(
    {Role.BACKEND, Role.FRONTEND, Role.STYLING, Role.INTEGRATION}
)

DEFAULT_OUTPUT_KIND: dict[Role, OutputKind] = {
    Role.PLANNING: OutputKind.SPEC,
    Role.RESEARCH: OutputKind.TEXT,
    Role.BACKEND: OutputKind.CODE,
    Role.FRONTEND: OutputKind.CODE,
    Role.STYLING: OutputKind.CODE,
    Role.CONTENT: OutputKind.TEXT,
    Role.REVIEW: OutputKind.TEXT,
    Role.INTEGRATION: OutputKind.CODE,
}


def parse_role(raw: str) -> Role:
    """Map a model-supplied string onto the taxonomy.

    Decomposers return near-misses constantly — "front-end", "API", "database".
    Normalising them is cheaper than spending a repair request, and refusing a
    whole plan over the word "front-end" would be absurd.
    """
    text = (raw or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    aliases = {
        "frontend": Role.FRONTEND,
        "ui": Role.FRONTEND,
        "client": Role.FRONTEND,
        "markup": Role.FRONTEND,
        "html": Role.FRONTEND,
        "backend": Role.BACKEND,
        "api": Role.BACKEND,
        "server": Role.BACKEND,
        "database": Role.BACKEND,
        "db": Role.BACKEND,
        "schema": Role.BACKEND,
        "styling": Role.STYLING,
        "style": Role.STYLING,
        "css": Role.STYLING,
        "design": Role.STYLING,
        "research": Role.RESEARCH,
        "investigation": Role.RESEARCH,
        "analysis": Role.RESEARCH,
        "planning": Role.PLANNING,
        "plan": Role.PLANNING,
        "architecture": Role.PLANNING,
        "content": Role.CONTENT,
        "copy": Role.CONTENT,
        "text": Role.CONTENT,
        "writing": Role.CONTENT,
        "review": Role.REVIEW,
        "qa": Role.REVIEW,
        "testing": Role.REVIEW,
        "integration": Role.INTEGRATION,
        "glue": Role.INTEGRATION,
        "wiring": Role.INTEGRATION,
    }

    if text in aliases:
        return aliases[text]

    for key, role in aliases.items():
        if key in text:
            return role

    # Unrecognisable: integration is the least-wrong default, since it carries
    # no strong affinity in any direction.
    return Role.INTEGRATION


def output_kind_for(role: Role) -> OutputKind:
    return DEFAULT_OUTPUT_KIND.get(role, OutputKind.TEXT)


def needs_review(role: Role, kind: OutputKind, policy: str) -> bool:
    """Whether a node gets Tier 1 cross-vendor review under `policy`."""
    if policy == "off":
        return False
    if policy == "all":
        return True
    return kind in (OutputKind.CODE, OutputKind.SCHEMA) or role in CODE_ROLES
