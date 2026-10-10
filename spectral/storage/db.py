"""SQLite storage for trace metadata, with atomic inserts and UTC filters."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from spectral.models import AggregatedStats, Provider, Status, Trace


def utc_timestamp(value: datetime) -> str:
    """Normalize timestamps; existing naive Trace timestamps are interpreted as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


class TraceStore:
    """A single-thread SQLite connection. Use a context manager to close it.

    Duplicate trace IDs raise sqlite3.IntegrityError rather than silently replacing data.
    A batch either commits entirely or leaves the database unchanged.
    """

    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        try:
            with self._connection:
                self._connection.execute(
                    """CREATE TABLE IF NOT EXISTS traces (
                        id TEXT PRIMARY KEY NOT NULL,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )"""
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS traces_timestamp ON traces(timestamp, id)"
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS traces_provider_model "
                    "ON traces(provider, model, timestamp)"
                )
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def add(self, trace: Trace) -> None:
        self.add_many([trace])

    def add_many(self, traces: Iterable[Trace]) -> None:
        rows = []
        for trace in traces:
            data = trace.to_dict()
            data["timestamp"] = utc_timestamp(trace.timestamp)
            rows.append(
                (
                    trace.id,
                    data["timestamp"],
                    trace.provider.value,
                    trace.model,
                    trace.status.value,
                    json.dumps(data, ensure_ascii=False, allow_nan=False),
                )
            )
        with self._connection:
            self._connection.executemany(
                "INSERT INTO traces (id, timestamp, provider, model, status, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def get(self, trace_id: str) -> Trace | None:
        row = self._connection.execute(
            "SELECT payload FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        return Trace.from_dict(json.loads(row["payload"])) if row else None

    def query(
        self,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        status: Status | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[Trace]:
        """Return newest first; time bounds are inclusive and normalized to UTC."""
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        if since is not None and until is not None and utc_timestamp(since) > utc_timestamp(until):
            raise ValueError("since must be earlier than or equal to until")
        clauses = []
        params: list[str | int] = []
        for column, value in (
            ("provider", provider.value if provider is not None else None),
            ("model", model),
            ("status", status.value if status is not None else None),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(utc_timestamp(since))
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(utc_timestamp(until))
        sql = "SELECT payload FROM traces"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            Trace.from_dict(json.loads(row["payload"]))
            for row in self._connection.execute(sql, params)
        ]

    def summary(
        self,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        status: Status | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> AggregatedStats:
        """Aggregate every matching trace, without a display limit."""
        return AggregatedStats.from_traces(
            self.query(
                provider=provider,
                model=model,
                status=status,
                since=since,
                until=until,
            )
        )
