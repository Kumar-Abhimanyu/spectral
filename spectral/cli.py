"""Inspect locally stored traces with no provider credentials or network calls."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from spectral.models import Provider, Status
from spectral.storage import TraceStore


def _timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use an ISO 8601 date or timestamp") from exc


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("use a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local LLM trace metadata")
    parser.add_argument("--db", type=Path, default=Path.home() / ".spectral" / "traces.sqlite3")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "traces"):
        command = commands.add_parser(name)
        command.add_argument("--provider", choices=[p.value for p in Provider])
        command.add_argument("--model")
        command.add_argument("--status", choices=[s.value for s in Status])
        command.add_argument("--since", type=_timestamp)
        command.add_argument("--until", type=_timestamp)
        command.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        if name == "traces":
            command.add_argument("--limit", type=_positive_int, default=100)
    args = parser.parse_args(argv)
    filters = {
        "provider": Provider(args.provider) if args.provider else None,
        "model": args.model,
        "status": Status(args.status) if args.status else None,
        "since": args.since,
        "until": args.until,
    }
    try:
        with TraceStore(args.db) as store:
            if args.command == "summary":
                stats = store.summary(**filters)
                data = {**asdict(stats), "error_rate": stats.error_rate}
                if args.json:
                    print(json.dumps(data, indent=2))
                else:
                    print(
                        f"Calls: {stats.total_calls} ({stats.success_calls} success, {stats.partial_calls} partial, {stats.error_calls} errors)"
                    )
                    print(
                        f"Tokens: {stats.total_tokens} ({stats.total_input} input, {stats.total_output} output)"
                    )
                    print(f"Estimated cost (USD): {stats.total_cost_usd:.6f}")
                    print(
                        f"Latency (ms): avg={stats.avg_latency_ms:.2f} p50={stats.p50_latency_ms:.2f} p95={stats.p95_latency_ms:.2f}"
                    )
                    print(f"Error rate: {stats.error_rate:.2%}")
            else:
                traces = store.query(**filters, limit=args.limit)
                if args.json:
                    print(json.dumps([trace.to_dict() for trace in traces], indent=2))
                elif not traces:
                    print("No matching traces.")
                else:
                    for trace in traces:
                        print(
                            f"{trace.timestamp.isoformat()} {trace.id} {trace.provider.value}/{trace.model} {trace.status.value} tokens={trace.total_tokens} latency_ms={trace.latency_ms:.2f}"
                        )
    except ValueError as exc:
        parser.error(str(exc))
    except (OSError, sqlite3.Error) as exc:
        print(
            f"Unable to access trace database: {exc}. Check --db and directory permissions.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
