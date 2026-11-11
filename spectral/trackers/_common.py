"""Shared metadata-only validation and recording for OpenAI adapters."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from spectral.models import Provider, Status, Trace
from spectral.storage import TraceStore


def copy_tags(tags: dict) -> dict:
    if not isinstance(tags, dict) or any(not isinstance(key, str) for key in tags):
        raise TypeError("tags must be a dictionary with string keys")
    return json.loads(json.dumps(tags, allow_nan=False))


def prepare_request(default_tags: dict, tags: dict | None, kwargs: dict, adapter: str) -> tuple[str, dict]:
    model = kwargs.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a nonempty string")
    return model, copy_tags(default_tags | copy_tags(tags if tags is not None else {}))


def record_call(
    store: TraceStore, logger: logging.Logger, model: str, timestamp: datetime,
    latency_ms: float, tags: dict, *, response: Any = None, error: BaseException | None = None,
    status: Status | None = None,
) -> None:
    try:
        input_tokens = output_tokens = 0
        if response is not None:
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
        store.add(Trace(
            timestamp=timestamp,
            provider=Provider.OPENAI,
            model=model,
            status=status if status is not None else (Status.ERROR if error is not None else Status.SUCCESS),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            error_type=type(error).__name__ if error is not None else None,
            tags=tags,
        ))
    except Exception:
        # Exception strings/tracebacks may contain request or credential data.
        logger.warning("Spectral could not record OpenAI call metadata")
