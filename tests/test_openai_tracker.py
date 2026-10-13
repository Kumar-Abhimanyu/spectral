"""No paid requests: fake clients and an optional real SDK/HTTP stub."""

import json
import tempfile
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from spectral.models import Provider, Status, estimate_cost
from spectral.storage import TraceStore
from spectral.trackers import OpenAITracker


def client_with(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def completion(model="gpt-4o", usage=None):
    return SimpleNamespace(model=model, usage=usage)


class OpenAITrackerTests(unittest.TestCase):
    def setUp(self):
        self.store = TraceStore(":memory:")
        self.addCleanup(self.store.close)

    def test_response_identity_kwargs_usage_cost_and_timing(self):
        response = completion(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40))
        create = Mock(return_value=response)
        tracker = OpenAITracker(client_with(create), self.store)
        messages = [{"role": "user", "content": "private prompt"}]
        with patch("spectral.trackers.openai.time.perf_counter", side_effect=[10, 10.25]):
            result = tracker.create(model="requested-alias", messages=messages, temperature=0)
        self.assertIs(result, response)
        create.assert_called_once_with(model="requested-alias", messages=messages, temperature=0)
        trace, = self.store.query()
        self.assertEqual(trace.provider, Provider.OPENAI)
        self.assertEqual(trace.status, Status.SUCCESS)
        self.assertEqual(trace.model, "gpt-4o")
        self.assertEqual((trace.input_tokens, trace.output_tokens), (120, 40))
        self.assertEqual(trace.cost_usd, estimate_cost("gpt-4o", 120, 40))
        self.assertEqual(trace.latency_ms, 250)
        self.assertIsNotNone(trace.timestamp.tzinfo)
        self.assertNotIn("private prompt", json.dumps(trace.to_dict()))

    def test_exception_identity_and_message_privacy(self):
        error = RuntimeError("private prompt and API credential")
        create = Mock(side_effect=error)
        tracker = OpenAITracker(client_with(create), self.store)
        with patch("spectral.trackers.openai.time.perf_counter", side_effect=[1, 1.125]):
            with self.assertRaises(RuntimeError) as caught:
                tracker.create(model="gpt-4o", messages=[])
        self.assertIs(caught.exception, error)
        create.assert_called_once()
        trace, = self.store.query()
        self.assertEqual(trace.status, Status.ERROR)
        self.assertEqual(trace.error_type, "RuntimeError")
        self.assertIsNone(trace.error_message)
        self.assertEqual(trace.latency_ms, 125)
        self.assertEqual(trace.total_tokens, 0)
        self.assertEqual(trace.cost_usd, 0)
        self.assertNotIn("credential", json.dumps(trace.to_dict()))

    def test_missing_usage_and_model_fall_back(self):
        for response in [completion(model=None), SimpleNamespace(), completion(usage=None)]:
            with self.subTest(response=response):
                tracker = OpenAITracker(client_with(Mock(return_value=response)), self.store)
                self.assertIs(tracker.create(model="unknown-model"), response)
        for trace in self.store.query():
            self.assertEqual(trace.total_tokens, 0)
            self.assertEqual(trace.cost_usd, 0)

    def test_invalid_usage_never_becomes_negative_or_boolean_tokens(self):
        for invalid in [-1, True, "12", None, 1.5]:
            with self.subTest(invalid=invalid):
                response = completion(usage=SimpleNamespace(prompt_tokens=invalid, completion_tokens=7))
                tracker = OpenAITracker(client_with(Mock(return_value=response)), self.store)
                tracker.create(model="gpt-4o")
        for trace in self.store.query():
            self.assertEqual((trace.input_tokens, trace.output_tokens), (0, 7))

    def test_tags_are_snapshotted_merged_and_not_forwarded(self):
        tags = {"feature": "search", "nested": {"value": 1}}
        create = Mock(return_value=completion())
        tracker = OpenAITracker(client_with(create), self.store, tags=tags)
        tags["nested"]["value"] = 2
        overrides = {"feature": "chat", "nested": {"value": 3}}
        tracker.create(model="gpt-4o", tags=overrides)
        overrides["nested"]["value"] = 4
        first, = self.store.query()
        self.assertEqual(first.tags, {"feature": "chat", "nested": {"value": 3}})
        tracker.create(model="gpt-4o")
        self.assertIn({"feature": "search", "nested": {"value": 1}}, [t.tags for t in self.store.query()])
        create.assert_called_with(model="gpt-4o")

    def test_invalid_tags_fail_before_provider_call(self):
        create = Mock()
        tracker = OpenAITracker(client_with(create), self.store)
        for tags in [[], {1: "value"}, {"x": object()}, {"x": float("nan")}]:
            with self.subTest(tags=tags):
                with self.assertRaises((TypeError, ValueError)):
                    tracker.create(model="gpt-4o", tags=tags)
                with self.assertRaises((TypeError, ValueError)):
                    OpenAITracker(client_with(create), self.store, tags=tags)
        create.assert_not_called()
        self.assertEqual(self.store.query(), [])

    def test_streaming_and_invalid_models_fail_before_request(self):
        create = Mock()
        tracker = OpenAITracker(client_with(create), self.store)
        for model in [None, "", 123]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                tracker.create(model=model)
        create.assert_not_called()
        self.assertEqual(self.store.query(), [])

    def test_async_client_is_rejected(self):
        create = AsyncMock()
        with self.assertRaises(TypeError):
            OpenAITracker(client_with(create), self.store)
        create.assert_not_called()

    def test_decorated_async_client_is_rejected(self):
        async def create(**kwargs):
            return completion()

        @wraps(create)
        def decorated(**kwargs):
            return create(**kwargs)

        with self.assertRaises(TypeError):
            OpenAITracker(client_with(decorated), self.store)

    def test_real_async_sdk_is_rejected(self):
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError:
            self.skipTest("Install the openai extra to exercise SDK compatibility")
        import asyncio

        async def check():
            async with AsyncOpenAI(api_key="test-only", http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            )) as client:
                with self.assertRaises(TypeError):
                    OpenAITracker(client, self.store)

        asyncio.run(check())

    def test_storage_failure_preserves_success_and_exception(self):
        response = completion()
        error = RuntimeError("sensitive provider message")
        store = Mock()
        store.add.side_effect = RuntimeError("sensitive storage message")
        tracker = OpenAITracker(client_with(Mock(return_value=response)), store)
        with self.assertLogs("spectral.trackers.openai", level="WARNING") as logs:
            self.assertIs(tracker.create(model="gpt-4o"), response)
            failing = OpenAITracker(client_with(Mock(side_effect=error)), store)
            with self.assertRaises(RuntimeError) as caught:
                failing.create(model="gpt-4o")
        self.assertIs(caught.exception, error)
        self.assertEqual(len(logs.records), 2)
        self.assertNotIn("sensitive", " ".join(logs.output))
        self.assertTrue(all(record.exc_info is None for record in logs.records))

    def test_response_metadata_failure_preserves_response(self):
        class Response:
            @property
            def usage(self):
                raise RuntimeError("sensitive response metadata")

        response = Response()
        tracker = OpenAITracker(client_with(Mock(return_value=response)), self.store)
        with self.assertLogs("spectral.trackers.openai", level="WARNING"):
            self.assertIs(tracker.create(model="gpt-4o"), response)

    def test_records_survive_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.sqlite3"
            with TraceStore(path) as store:
                tracker = OpenAITracker(client_with(Mock(return_value=completion())), store)
                tracker.create(model="gpt-4o")
                failing = OpenAITracker(client_with(Mock(side_effect=ValueError("private"))), store)
                with self.assertRaises(ValueError):
                    failing.create(model="gpt-4o")
            with TraceStore(path) as store:
                stats = store.summary()
                self.assertEqual((stats.total_calls, stats.success_calls, stats.error_calls), (2, 1, 1))

    def test_real_sdk_response_with_mock_http_transport(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("Install the openai extra to exercise SDK compatibility")

        def respond(request):
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "private completion"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        with OpenAI(api_key="test-only", http_client=httpx.Client(
            transport=httpx.MockTransport(respond),
        )) as client:
            tracker = OpenAITracker(client, self.store)
            response = tracker.create(model="gpt-4o", messages=[{"role": "user", "content": "private prompt"}])
        self.assertEqual(response.choices[0].message.content, "private completion")
        trace, = self.store.query()
        self.assertEqual(trace.total_tokens, 15)
        self.assertNotIn("private", json.dumps(trace.to_dict()))


if __name__ == "__main__":
    unittest.main()
