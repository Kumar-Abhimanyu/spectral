# Spectral

Local observability for LLM calls. Store trace metadata in SQLite and inspect
latency, token usage, errors, and estimated costs from the command line.
This milestone implements persistence and reporting; provider trackers and the
dashboard remain placeholders.

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

1. Add a provider adapter that records successful and failed calls.
2. Add request correlation, configurable cost rates, and latency comparisons.
3. Add a terminal dashboard, followed by trace export and retention policies.
