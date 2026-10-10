import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from spectral.cli import main
from spectral.models import Provider, Status, Trace
from spectral.storage import TraceStore


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "traces.sqlite3"
        with TraceStore(self.path) as store:
            store.add_many(
                [
                    Trace(
                        id="one",
                        timestamp=datetime(2026, 10, 10, tzinfo=timezone.utc),
                        model="local",
                        provider=Provider.OLLAMA,
                        input_tokens=3,
                        output_tokens=2,
                        latency_ms=10,
                    ),
                    Trace(
                        id="two",
                        timestamp=datetime(2026, 10, 11, tzinfo=timezone.utc),
                        model="remote",
                        status=Status.ERROR,
                        error_type="TimeoutError",
                    ),
                ]
            )

    def invoke(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--db", str(self.path), *args])
        self.assertEqual(code, 0)
        return output.getvalue()

    def test_summary_json_and_filters(self):
        data = json.loads(self.invoke("summary", "--json"))
        self.assertEqual(data["total_calls"], 2)
        self.assertEqual(data["total_tokens"], 5)
        self.assertEqual(data["error_rate"], 0.5)
        filtered = json.loads(
            self.invoke("summary", "--json", "--provider", "ollama", "--model", "local")
        )
        self.assertEqual(filtered["total_calls"], 1)
        self.assertEqual(filtered["error_calls"], 0)

    def test_time_and_status_filters(self):
        data = json.loads(
            self.invoke(
                "traces",
                "--json",
                "--status",
                "error",
                "--since",
                "2026-10-11T00:00:00Z",
                "--until",
                "2026-10-11",
            )
        )
        self.assertEqual([t["id"] for t in data], ["two"])

    def test_traces_limit_and_text(self):
        data = json.loads(self.invoke("traces", "--json", "--limit", "1"))
        self.assertEqual([t["id"] for t in data], ["two"])
        self.assertIn("ollama/local", self.invoke("traces", "--provider", "ollama"))
        self.assertIn("Calls: 2 (1 success, 1 errors)", self.invoke("summary"))

    def test_empty_results(self):
        self.assertEqual(self.invoke("traces", "--model", "missing").strip(), "No matching traces.")
        data = json.loads(self.invoke("summary", "--model", "missing", "--json"))
        self.assertEqual(data["total_calls"], 0)

    def test_bad_arguments_have_usage_errors(self):
        for args in (
            ["traces", "--limit", "0"],
            ["traces", "--limit", "many"],
            ["summary", "--since", "yesterday"],
            ["summary", "--provider", "unknown"],
            ["summary", "--since", "2026-10-12", "--until", "2026-10-10"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(*args)
                self.assertEqual(error.exception.code, 2)

    def test_bad_database_returns_failure_without_traceback(self):
        self.path.write_text("not a SQLite database")
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = main(["--db", str(self.path), "summary"])
        self.assertEqual(code, 1)
        self.assertIn("Check --db", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())


if __name__ == "__main__":
    unittest.main()
