"""Lexical backstop for an already semantically reviewed memory projection.

These are topic names, not descriptions of acts or bodily responses. They do
not grant permission to raw chat: the caller must also have an exact Opus A
projection receipt. All other configured keywords remain blocking, including
unknown/custom entries. The source keyword table is never changed.
"""

# Deployments may list topic names here; an empty set keeps every keyword blocking.
_FACT_TOPICS = frozenset()


def projection_keyword_hit(text, keywords):
    """Reject detail/unknown terms even when a model erroneously calls them A."""
    folded = text.casefold()
    return any(word not in _FACT_TOPICS and word in folded for word in keywords)
