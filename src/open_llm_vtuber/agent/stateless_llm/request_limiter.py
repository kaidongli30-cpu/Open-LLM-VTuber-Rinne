import asyncio
import hashlib
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from loguru import logger


BackendKey = tuple[str, str, str, str, str]
RegistryKey = tuple[int, BackendKey]

_registry_lock = threading.Lock()
_limiters: dict[RegistryKey, asyncio.Semaphore] = {}
_limits: dict[RegistryKey, int] = {}
_spacing_locks: dict[RegistryKey, asyncio.Lock] = {}
_next_request_ready_at: dict[RegistryKey, float] = {}


def build_backend_key(
    provider_name: str,
    base_url: str | None,
    organization_id: str | None = None,
    project_id: str | None = None,
    api_key: str | None = None,
) -> BackendKey:
    api_key_fingerprint = ""
    if api_key:
        api_key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]

    return (
        provider_name,
        base_url or "",
        organization_id or "",
        project_id or "",
        api_key_fingerprint,
    )


def _normalize_limit(max_concurrent_requests: int | None) -> int:
    if max_concurrent_requests is None:
        return 1
    return max(1, int(max_concurrent_requests))


def _normalize_min_interval(min_request_interval_seconds: float | None) -> float:
    if min_request_interval_seconds is None:
        return 0.0
    return max(0.0, float(min_request_interval_seconds))


def _build_registry_key(backend_key: BackendKey) -> RegistryKey:
    return (id(asyncio.get_running_loop()), backend_key)


def _get_request_limiter(
    backend_key: BackendKey, max_concurrent_requests: int | None
) -> asyncio.Semaphore:
    normalized_limit = _normalize_limit(max_concurrent_requests)
    registry_key = _build_registry_key(backend_key)

    with _registry_lock:
        limiter = _limiters.get(registry_key)
        current_limit = _limits.get(registry_key)

        if limiter is None or current_limit != normalized_limit:
            limiter = asyncio.Semaphore(normalized_limit)
            _limiters[registry_key] = limiter
            _limits[registry_key] = normalized_limit

        return limiter


def _get_request_spacing_lock(backend_key: BackendKey) -> tuple[RegistryKey, asyncio.Lock]:
    registry_key = _build_registry_key(backend_key)

    with _registry_lock:
        spacing_lock = _spacing_locks.get(registry_key)
        if spacing_lock is None:
            spacing_lock = asyncio.Lock()
            _spacing_locks[registry_key] = spacing_lock

    return registry_key, spacing_lock


@asynccontextmanager
async def limit_request_concurrency(
    backend_key: BackendKey,
    max_concurrent_requests: int | None,
    min_request_interval_seconds: float | None = None,
) -> AsyncIterator[None]:
    limiter = _get_request_limiter(backend_key, max_concurrent_requests)
    normalized_interval = _normalize_min_interval(min_request_interval_seconds)

    if limiter.locked():
        logger.debug(
            f"Waiting for an available LLM request slot for backend '{backend_key[1] or backend_key[0]}'."
        )

    await limiter.acquire()
    try:
        if normalized_interval > 0:
            registry_key, spacing_lock = _get_request_spacing_lock(backend_key)
            loop = asyncio.get_running_loop()
            async with spacing_lock:
                now = loop.time()
                ready_at = _next_request_ready_at.get(registry_key, 0.0)
                wait_time = ready_at - now
                if wait_time > 0:
                    logger.debug(
                        "Delaying request for backend "
                        f"'{backend_key[1] or backend_key[0]}' by {wait_time:.2f}s "
                        "to respect the minimum request interval."
                    )
                    await asyncio.sleep(wait_time)
                _next_request_ready_at[registry_key] = (
                    loop.time() + normalized_interval
                )
        yield
    finally:
        limiter.release()
