"""Limit only classification calls, without mutating the shared chat client."""

from copy import copy
from types import SimpleNamespace

CHECK_MAX_TOKENS = 4096


class CheckOutputTruncated(ValueError):
    pass


class _CheckedStream:
    def __init__(self, stream):
        self.stream = stream
        self.iterator = stream.__aiter__()

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.iterator.__anext__()
        if any(
            getattr(choice, "finish_reason", None) == "length"
            for choice in (getattr(chunk, "choices", None) or [])
        ):
            raise CheckOutputTruncated("classification_output_truncated")
        return chunk


class _LimitedCompletions:
    def __init__(self, completions):
        self.completions = completions

    async def create(self, **kwargs):
        kwargs["max_tokens"] = CHECK_MAX_TOKENS
        result = await self.completions.create(**kwargs)
        return _CheckedStream(result) if kwargs.get("stream") else result


class _CheckClient:
    def __init__(self, client):
        self.client = client
        self.chat = SimpleNamespace(
            completions=_LimitedCompletions(client.chat.completions)
        )

    def __getattr__(self, name):
        return getattr(self.client, name)


def bounded_check_model(model):
    # Two-argument synthetic classifiers have no SDK transport to limit.
    client = getattr(model, "client", None)
    if client is None or not hasattr(client, "chat"):
        return model
    result = copy(model)
    result.client = _CheckClient(client)
    return result
