import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from spectral.models import Provider, Status, Trace
from spectral.storage import TraceStore


class TraceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "nested" / "traces.sqlite3"
        self.store = TraceStore(self.path)
        self.addCleanup(self.store.close)
        self.time = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)

    def test_round_trip_survives_reopening(self):
        trace = Trace(
            timestamp=self.time,
            provider=Provider.ANTHROPIC,
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=123.45,
            cost_usd=0.012,
            tags={"feature": "search", "nested": {"labels": ["café", 42]}},
        )
        self.store.add(trace)
        with TraceStore(self.path) as other:
            self.assertEqual(other.get(trace.id).to_dict(), trace.to_dict())

    def test_error_trace_round_trip(self):
        trace = Trace(
            timestamp=self.time,
            status=Status.ERROR,
            error_type="TimeoutError",
            error_message="provider timed out",
        )
        self.store.add(trace)
        self.assertEqual(self.store.get(trace.id).to_dict(), trace.to_dict())

    def test_missing_id_and_empty_summary(self):
        self.assertIsNone(self.store.get("missing"))
        self.assertEqual(self.store.query(), [])
        self.assertEqual(self.store.summary().total_calls, 0)
        self.assertEqual(self.store.summary().error_rate, 0)

    def test_duplicate_does_not_overwrite(self):
        trace = Trace(id="same", model="original")
        self.store.add(trace)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add(Trace(id="same", model="replacement"))
        self.assertEqual(self.store.get("same").model, "original")

    def test_batch_rolls_back_on_duplicate(self):
        self.store.add(Trace(id="existing"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_many([Trace(id="new"), Trace(id="existing")])
        self.assertIsNone(self.store.get("new"))
        self.assertEqual(len(self.store.query()), 1)

    def test_serialization_failure_leaves_batch_unchanged(self):
        with self.assertRaises(TypeError):
            self.store.add_many([Trace(id="good"), Trace(tags={"bad": object()})])
        self.assertEqual(self.store.query(), [])

    def test_nonfinite_metadata_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add(Trace(latency_ms=float("nan")))
        self.assertEqual(self.store.query(), [])

    def test_generator_batch(self):
        self.store.add_many(Trace(id=str(i)) for i in range(3))
        self.assertEqual(len(self.store.query()), 3)

    def test_naive_and_offset_timestamps_normalized_without_mutating_trace(self):
        local_time = self.time.astimezone(timezone(timedelta(hours=5, minutes=30)))
        aware = Trace(id="aware", timestamp=local_time)
        naive = Trace(id="naive", timestamp=self.time.replace(tzinfo=None))
        self.store.add_many([aware, naive])
        for trace in self.store.query(since=self.time, until=local_time):
            self.assertEqual(trace.timestamp, self.time)
            self.assertEqual(trace.timestamp.tzinfo, timezone.utc)
        self.assertEqual(len(self.store.query(since=self.time, until=local_time)), 2)
        self.assertEqual(aware.timestamp, local_time)
        self.assertIsNone(naive.timestamp.tzinfo)

    def test_order_limit_and_inclusive_time_filters(self):
        self.store.add_many(
            [
                Trace(id="old", timestamp=self.time - timedelta(seconds=1)),
                Trace(id="a", timestamp=self.time),
                Trace(id="b", timestamp=self.time),
                Trace(id="new", timestamp=self.time + timedelta(seconds=1)),
            ]
        )
        self.assertEqual([t.id for t in self.store.query(limit=2)], ["new", "b"])
        self.assertEqual(
            [t.id for t in self.store.query(since=self.time, until=self.time)], ["b", "a"]
        )

    def test_combined_filters(self):
        self.store.add_many(
            [
                Trace(id="match", model="search", provider=Provider.OLLAMA, status=Status.ERROR),
                Trace(
                    id="other-model", model="chat", provider=Provider.OLLAMA, status=Status.ERROR
                ),
                Trace(id="other-status", model="search", provider=Provider.OLLAMA),
                Trace(id="other-provider", model="search", status=Status.ERROR),
            ]
        )
        traces = self.store.query(provider=Provider.OLLAMA, model="search", status=Status.ERROR)
        self.assertEqual([t.id for t in traces], ["match"])

    def test_model_filter_is_parameterized(self):
        model = "x' OR 1=1 --"
        self.store.add_many([Trace(id="match", model=model), Trace(id="other", model="normal")])
        self.assertEqual([t.id for t in self.store.query(model=model)], ["match"])
        self.assertEqual(len(self.store.query()), 2)

    def test_invalid_query_arguments(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.store.query(limit=limit)
        with self.assertRaises(ValueError):
            self.store.query(since=self.time, until=self.time - timedelta(seconds=1))

    def test_summary_uses_every_matching_trace(self):
        self.store.add_many(
            Trace(
                id=str(i),
                model="test",
                timestamp=self.time,
                input_tokens=2,
                output_tokens=1,
                latency_ms=10,
                cost_usd=0.01,
            )
            for i in range(101)
        )
        self.store.add(Trace(model="other", status=Status.ERROR))
        stats = self.store.summary(model="test")
        self.assertEqual(stats.total_calls, 101)
        self.assertEqual(stats.total_tokens, 303)
        self.assertEqual(stats.total_cost_usd, 1.01)
        self.assertEqual(stats.avg_latency_ms, 10)
        self.assertEqual(stats.by_model["test"]["calls"], 101)
        self.assertEqual(self.store.summary().error_calls, 1)

    def test_context_manager_closes_connection(self):
        with TraceStore(":memory:") as store:
            store.add(Trace())
        with self.assertRaises(sqlite3.ProgrammingError):
            store.query()


if __name__ == "__main__":
    unittest.main()
