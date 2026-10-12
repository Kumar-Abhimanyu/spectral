# Spectral

Local observability for LLM calls. Store trace metadata in SQLite and inspect
latency, token usage, errors, and estimated costs from the command line.
Includes SQLite persistence, CLI reporting, and opt-in synchronous OpenAI Chat
Completions instrumentation. Other provider adapters and the dashboard remain placeholders.

## Install

Requires Python 3.10 or newer. Runtime code uses only the standard library.

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

## Record a trace

Use the storage API from your application or an instrumentation adapter:

```python
from datetime import datetime, timezone

from spectral.models import Provider, Trace
from spectral.storage import TraceStore

with TraceStore("traces.sqlite3") as store:
    store.add(Trace(
        timestamp=datetime.now(timezone.utc),
        provider=Provider.OLLAMA,
        model="local-model",
        input_tokens=120,
        output_tokens=40,
        latency_ms=850,
        tags={"feature": "document-search"},
    ))
```

`add_many(iterable)` writes a batch atomically. Duplicate IDs raise
`sqlite3.IntegrityError`; they never overwrite existing traces. JSON-compatible
tags, error details, and usage metadata survive reopening the database.
Timestamps are stored in UTC. Existing naive timestamps are interpreted as UTC.
Use a separate `TraceStore` connection per thread and close it with `with`.

## Instrument OpenAI calls

Install the optional SDK extra: `python -m pip install -e '.[openai]'`.
Inject your existing synchronous client and store:

```python
from openai import OpenAI
from spectral.storage import TraceStore
from spectral.trackers import OpenAITracker

with OpenAI() as client, TraceStore("traces.sqlite3") as store:
    tracker = OpenAITracker(client, store, tags={"application": "support"})
    response = tracker.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        tags={"feature": "chat"},
    )
```

`create` forwards request arguments to `client.chat.completions.create` and
returns the original response object. It records provider, returned model
(falling back to the requested model), UTC start time, token usage, estimated
cost, elapsed request time, status, and optional tags. Call tags override default
tags; nested values are snapshotted before the request. Tags must be JSON-compatible
with string keys and finite numeric values. No prompts, completions, API keys,
request headers, or provider exception messages are captured. Failure traces
contain only the exception class name. Keep sensitive data out of tags.

Provider exceptions propagate unchanged. Storage or metadata failures emit a
generic warning without exception details and preserve the provider's result;
those calls may have no trace. Missing or invalid usage counts become zero, so
zero tokens does not prove a call was free. Costs use the existing static table;
snapshot model names and unknown models can report zero. Latency includes SDK
retries but excludes trace persistence. Streaming is rejected before a request;
asynchronous clients, Responses API, raw responses, and other providers are
outside this adapter's scope. Do not pass an asynchronous callable disguised as
a synchronous client. Use one store per thread.

## Instrument asynchronous OpenAI calls

Use `AsyncOpenAITracker` with an injected `AsyncOpenAI` client for non-streaming
Chat Completions. The synchronous and asynchronous adapters share metadata
validation and trace recording.

```python
import asyncio
from openai import AsyncOpenAI
from spectral.storage import TraceStore
from spectral.trackers import AsyncOpenAITracker

async def main():
    with TraceStore("traces.sqlite3") as store:
        async with AsyncOpenAI() as client:
            tracker = AsyncOpenAITracker(client, store, tags={"application": "support"})
            response = await tracker.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                tags={"feature": "chat"},
            )

asyncio.run(main())
```

Provider response objects and exceptions are preserved. Cancellation reaches the
provider and is re-raised; best-effort telemetry records it as an error with type
`CancelledError`, without the cancellation message. Cancellation before the
coroutine starts creates no trace. Call tags are snapshotted when the coroutine
starts, before awaiting the provider. Concurrent tasks on one event-loop thread
may share a tracker and store; each call has independent tags and timing.
Create, use, and close the store on that same thread. SQLite writes are synchronous
and may briefly block the event loop (including waiting for a database lock).
This adapter does not offload writes to worker threads or add a storage queue.
Streaming, Responses API, and async callables hidden behind wrappers without
`__wrapped__` remain unsupported. Privacy, usage fallbacks, static cost estimates,
and best-effort recording limitations are the same as for `OpenAITracker`.

## Inspect traces

```sh
spectral --db traces.sqlite3 summary
spectral --db traces.sqlite3 summary --provider ollama --json
spectral --db traces.sqlite3 traces --model local-model --limit 20 --json
spectral --db traces.sqlite3 summary --since 2026-10-10T00:00:00Z --until 2026-10-11T00:00:00Z
spectral --db traces.sqlite3 traces --status error
```

`--db` comes before the subcommand. If omitted, the database is
`~/.spectral/traces.sqlite3`; missing parent directories are created automatically.
All filters combine with AND. Time bounds are inclusive; ISO 8601 dates mean
midnight UTC, and timestamps may include an offset. Results are ordered newest
first, with trace ID as the tie breaker. The default trace display limit is 100;
summaries always include **all** matching traces.

The Python API offers `get(id)`, `query(...)`, and `summary(...)` with the same
filters. CLI JSON summaries include provider/model breakdowns, token totals,
error rate, and the existing model's average/p50/p95 latency calculations.
Cost reporting uses the existing static model estimates, not provider invoices;
unknown models currently report zero. Summary calculation loads matching traces
into memory, so large datasets will need SQL aggregation or streaming in a later milestone.

No prompts, completions, or API credentials are captured automatically. Tags and
error messages are stored verbatim: include only metadata you intend to retain.

## Validation

The test suite needs no network, API keys, paid calls, or downloaded models:

```sh
python -m unittest discover -s tests -v
```

Tests cover persistence across connections, atomic rollback, duplicate IDs,
Unicode metadata, UTC time filters, parameterized queries, CLI errors, and summaries.

## Next milestones

1. Add streaming instrumentation, then Anthropic and Ollama adapters.
2. Add request correlation, configurable cost rates, and latency comparisons.
3. Add a terminal dashboard, followed by trace export and retention policies.
