"""Async adapter regressions using fake clients and mocked SDK HTTP."""

import asyncio
import json
import tempfile
import threading
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from spectral.models import Status, estimate_cost
from spectral.storage import TraceStore
from spectral.trackers import AsyncOpenAITracker


def client_with(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def completion(model="gpt-4o", usage=None):
    return SimpleNamespace(model=model, usage=usage)


class AsyncOpenAITrackerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = TraceStore(":memory:")
        self.addCleanup(self.store.close)

    async def test_identity_arguments_usage_cost_timing_and_privacy(self):
        response = completion(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40))
        create = AsyncMock(return_value=response)
        tracker = AsyncOpenAITracker(client_with(create), self.store)
        messages = [{"role": "user", "content": "private prompt"}]
        with patch("spectral.trackers.async_openai.time.perf_counter", side_effect=[1, 1.25]):
            self.assertIs(await tracker.create(model="alias", messages=messages), response)
        create.assert_awaited_once_with(model="alias", messages=messages)
        trace, = self.store.query()
        self.assertEqual(trace.model, "gpt-4o")
        self.assertEqual((trace.input_tokens, trace.output_tokens), (120, 40))
        self.assertEqual(trace.cost_usd, estimate_cost("gpt-4o", 120, 40))
        self.assertEqual(trace.latency_ms, 250)
        self.assertIsNotNone(trace.timestamp.tzinfo)
        self.assertNotIn("private", json.dumps(trace.to_dict()))

    async def test_exception_identity_and_privacy(self):
        error = ValueError("private credential")
        create = AsyncMock(side_effect=error)
        tracker = AsyncOpenAITracker(client_with(create), self.store)
        with self.assertRaises(ValueError) as caught:
            await tracker.create(model="gpt-4o")
        self.assertIs(caught.exception, error)
        create.assert_awaited_once()
        trace, = self.store.query()
        self.assertEqual(trace.status, Status.ERROR)
        self.assertEqual(trace.error_type, "ValueError")
        self.assertIsNone(trace.error_message)
        self.assertEqual(trace.total_tokens, 0)
        self.assertEqual(trace.cost_usd, 0)
        self.assertNotIn("private", json.dumps(trace.to_dict()))

    async def test_cancelled_error_identity_and_trace(self):
        error = asyncio.CancelledError("private")
        tracker = AsyncOpenAITracker(client_with(AsyncMock(side_effect=error)), self.store)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await tracker.create(model="gpt-4o")
        self.assertIs(caught.exception, error)
        trace, = self.store.query()
        self.assertEqual((trace.status, trace.error_type), (Status.ERROR, "CancelledError"))
        self.assertIsNone(trace.error_message)

    async def test_task_cancellation_reaches_provider(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def create(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        tracker = AsyncOpenAITracker(client_with(create), self.store)
        task = asyncio.create_task(tracker.create(model="gpt-4o"))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(task.cancelled())
        self.assertTrue(cancelled.is_set())
        trace, = self.store.query()
        self.assertEqual(trace.error_type, "CancelledError")

    async def test_tag_snapshot_before_await_and_override(self):
        defaults = {"nested": {"value": 1}, "feature": "default"}
        tags = {"nested": {"value": 2}}
        entered, release = asyncio.Event(), asyncio.Event()

        async def create(**kwargs):
            entered.set()
            await release.wait()
            return completion()

        tracker = AsyncOpenAITracker(client_with(create), self.store, tags=defaults)
        defaults["nested"]["value"] = 9
        task = asyncio.create_task(tracker.create(model="gpt-4o", tags=tags))
        await entered.wait()
        tags["nested"]["value"] = 99
        release.set()
        await task
        trace, = self.store.query()
        self.assertEqual(trace.tags, {"nested": {"value": 2}, "feature": "default"})
        await tracker.create(model="gpt-4o")
        self.assertIn({"nested": {"value": 1}, "feature": "default"}, [t.tags for t in self.store.query()])

    async def test_validation_before_provider(self):
        create = AsyncMock()
        tracker = AsyncOpenAITracker(client_with(create), self.store)
        for kwargs in [dict(model="gpt-4o", stream=True), dict(model=""), dict(model=None),
                       dict(model=7), dict(model="gpt-4o", tags=[]),
                       dict(model="gpt-4o", tags={1: "bad"}),
                       dict(model="gpt-4o", tags={"x": float("nan")})]:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                await tracker.create(**kwargs)
        with self.assertRaises(TypeError):
            AsyncOpenAITracker(client_with(create), self.store, tags=[])
        create.assert_not_called()
        self.assertEqual(self.store.query(), [])

    async def test_sync_rejected_and_decorated_async_accepted(self):
        with self.assertRaises(TypeError):
            AsyncOpenAITracker(client_with(Mock()), self.store)

        async def create(**kwargs):
            return completion()

        @wraps(create)
        def decorated(**kwargs):
            return create(**kwargs)

        tracker = AsyncOpenAITracker(client_with(decorated), self.store)
        await tracker.create(model="gpt-4o")
        self.assertEqual(len(self.store.query()), 1)

    async def test_missing_and_invalid_metadata(self):
        for invalid in [None, -1, True, "2", 1.5]:
            response = completion(model=None, usage=SimpleNamespace(prompt_tokens=invalid, completion_tokens=7))
            await AsyncOpenAITracker(client_with(AsyncMock(return_value=response)), self.store).create(model="unknown")
        for trace in self.store.query():
            self.assertEqual((trace.model, trace.input_tokens, trace.output_tokens), ("unknown", 0, 7))
        await AsyncOpenAITracker(client_with(AsyncMock(return_value=SimpleNamespace())), self.store).create(model="unknown")
        self.assertEqual(self.store.query()[0].total_tokens, 0)

    async def test_storage_failure_preserves_all_outcomes(self):
        store = Mock()
        store.add.side_effect = RuntimeError("private storage error")
        response = completion()
        for outcome in [response, ValueError("private"), asyncio.CancelledError("private")]:
            create = AsyncMock(return_value=outcome) if outcome is response else AsyncMock(side_effect=outcome)
            tracker = AsyncOpenAITracker(client_with(create), store)
            with self.assertLogs("spectral.trackers.async_openai", level="WARNING") as logs:
                if outcome is response:
                    self.assertIs(await tracker.create(model="gpt-4o"), response)
                else:
                    with self.assertRaises(type(outcome)) as caught:
                        await tracker.create(model="gpt-4o")
                    self.assertIs(caught.exception, outcome)
            self.assertNotIn("private", " ".join(logs.output))
            self.assertIsNone(logs.records[0].exc_info)

    async def test_broken_metadata_preserves_response(self):
        class Response:
            @property
            def usage(self):
                raise RuntimeError("private")

        response = Response()
        tracker = AsyncOpenAITracker(client_with(AsyncMock(return_value=response)), self.store)
        with self.assertLogs("spectral.trackers.async_openai", level="WARNING"):
            self.assertIs(await tracker.create(model="gpt-4o"), response)

    async def test_concurrent_calls_isolate_tags_and_stay_on_store_thread(self):
        owner = threading.get_ident()
        threads = []
        original_add = self.store.add

        def add(trace):
            threads.append(threading.get_ident())
            original_add(trace)

        self.store.add = add
        entered = asyncio.Event()
        count = 0

        async def create(**kwargs):
            nonlocal count
            count += 1
            if count == 20:
                entered.set()
            await entered.wait()
            return completion(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))

        tracker = AsyncOpenAITracker(client_with(create), self.store)
        responses = await asyncio.wait_for(asyncio.gather(*[
            tracker.create(model="gpt-4o", tags={"request": i}) for i in range(20)
        ]), timeout=5)
        self.assertEqual(len(responses), 20)
        self.assertEqual({t.tags["request"] for t in self.store.query()}, set(range(20)))
        self.assertEqual(self.store.summary().total_tokens, 300)
        self.assertEqual(threads, [owner] * 20)

    async def test_sqlite_reopen_and_real_async_sdk(self):
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError:
            self.skipTest("Install openai extra for SDK compatibility")

        async def respond(request):
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "private completion"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.sqlite3"
            with TraceStore(path) as store:
                async with AsyncOpenAI(api_key="test-only", http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(respond),
                )) as client:
                    result = await AsyncOpenAITracker(client, store).create(
                        model="gpt-4o", messages=[{"role": "user", "content": "private prompt"}],
                    )
                    self.assertEqual(result.choices[0].message.content, "private completion")
            with TraceStore(path) as store:
                trace, = store.query()
                self.assertEqual(trace.total_tokens, 15)
                self.assertNotIn("private", json.dumps(trace.to_dict()))
