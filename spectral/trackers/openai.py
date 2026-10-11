"""Opt-in instrumentation for synchronous OpenAI Chat Completions."""

from __future__ import annotations

import inspect
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from spectral.models import Provider, Status, Trace
from spectral.storage import TraceStore

logger = logging.getLogger(__name__)


class OpenAITracker:
    """Wrap an injected synchronous client without changing SDK internals.

    Only metadata is captured. Provider exceptions and response objects are
    preserved; telemetry failures produce a generic warning and never replace
    the provider's outcome. Use the store on its owning thread.
    """

    def __init__(self, client: Any, store: TraceStore, *, tags: dict | None = None):
        self._create = client.chat.completions.create
        if inspect.iscoroutinefunction(inspect.unwrap(self._create)):
            raise TypeError("OpenAITracker requires a synchronous client")
        self._store = store
        self._tags = self._copy_tags(tags if tags is not None else {})

    @staticmethod
    def _copy_tags(tags: dict) -> dict:
        if not isinstance(tags, dict) or any(not isinstance(key, str) for key in tags):
            raise TypeError("tags must be a dictionary with string keys")
        # Snapshot nested metadata before a paid request; reject non-JSON tags early.
        return json.loads(json.dumps(tags, allow_nan=False))

    def create(self, *, tags: dict | None = None, **kwargs: Any) -> Any:
        """Call chat.completions.create once and persist usage, timing and status.

        Call tags override constructor tags. Streaming is rejected before any
        request. Prompts, completions, credentials and exception messages are
        never copied to the trace. Tags are explicitly user-supplied metadata.
        """
        if kwargs.get("stream"):
            raise ValueError("Streaming is not supported by OpenAITracker")
        model = kwargs.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        trace_tags = self._copy_tags(self._tags | self._copy_tags(tags if tags is not None else {}))
        timestamp = datetime.now(timezone.utc)
        start = time.perf_counter()
        try:
            response = self._create(**kwargs)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            self._record(model, timestamp, latency_ms, trace_tags, error=exc)
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        self._record(model, timestamp, latency_ms, trace_tags, response=response)
        return response

    def _record(
        self, model: str, timestamp: datetime, latency_ms: float, tags: dict,
        *, response: Any = None, error: Exception | None = None,
    ) -> None:
        try:
            input_tokens = output_tokens = 0
            if error is None:
                response_model = getattr(response, "model", None)
                if isinstance(response_model, str) and response_model:
                    model = response_model
                usage = getattr(response, "usage", None)
                for field, target in (("prompt_tokens", "input"), ("completion_tokens", "output")):
                    value = getattr(usage, field, None)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        if target == "input":
                            input_tokens = value
                        else:
                            output_tokens = value
            self._store.add(Trace(
                timestamp=timestamp,
                provider=Provider.OPENAI,
                model=model,
                status=Status.ERROR if error is not None else Status.SUCCESS,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error_type=type(error).__name__ if error is not None else None,
                tags=tags,
            ))
        except Exception:
            # Exception strings/tracebacks may contain request or credential data.
            logger.warning("Spectral could not record OpenAI call metadata")
