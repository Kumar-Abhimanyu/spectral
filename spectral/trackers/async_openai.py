"""Opt-in instrumentation for asynchronous OpenAI Chat Completions."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from datetime import datetime, timezone
from typing import Any

from spectral.storage import TraceStore
from spectral.trackers._common import copy_tags, prepare_request, record_call
from spectral.trackers._streaming import AsyncTrackedStream

logger = logging.getLogger(__name__)


class AsyncOpenAITracker:
    """Wrap an injected asynchronous client without changing SDK internals.

    Only metadata is captured. Provider exceptions and response objects are
    preserved, including cancellation; telemetry failures produce a generic warning and never replace
    the provider's outcome. SQLite writes stay on the store's owning event-loop
    thread and may briefly block it. No work is offloaded to worker threads.
    """

    def __init__(self, client: Any, store: TraceStore, *, tags: dict | None = None):
        self._create = client.chat.completions.create
        if not inspect.iscoroutinefunction(inspect.unwrap(self._create)):
            raise TypeError("AsyncOpenAITracker requires an asynchronous client")
        self._store = store
        self._tags = copy_tags(tags if tags is not None else {})

    async def create(self, *, tags: dict | None = None, **kwargs: Any) -> Any:
        """Call chat.completions.create once and persist usage, timing and status.

        Call tags override constructor tags. Streaming returns a lifecycle wrapper. Prompts, completions, credentials and exception messages are
        never copied to the trace. Tags are explicitly user-supplied metadata.
        """
        model, trace_tags = prepare_request(self._tags, tags, kwargs, "AsyncOpenAITracker")
        timestamp = datetime.now(timezone.utc)
        start = time.perf_counter()
        try:
            response = await self._create(**kwargs)
        except asyncio.CancelledError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self._record(model, timestamp, latency_ms, trace_tags, error=exc)
            raise
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self._record(model, timestamp, latency_ms, trace_tags, error=exc)
            raise
        if kwargs.get("stream"):
            return AsyncTrackedStream(response, self._store, logger, model, timestamp, start, trace_tags)
        latency_ms = (time.perf_counter() - start) * 1000
        self._record(model, timestamp, latency_ms, trace_tags, response=response)
        return response

    def _record(
        self, model: str, timestamp: datetime, latency_ms: float, tags: dict,
        *, response: Any = None, error: BaseException | None = None,
    ) -> None:
        record_call(self._store, logger, model, timestamp, latency_ms, tags,
                    response=response, error=error)
