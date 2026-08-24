"""llmorch — a quota-governed multi-provider LLM orchestrator.

Splits a task across models from different vendors, assigning each slice to the
model best suited to it, while respecting heterogeneous free-tier rate limits.
"""

__version__ = "0.1.0"
