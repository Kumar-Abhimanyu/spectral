"""Streaming lifecycle regressions; no paid requests."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from spectral.models import Status
from spectral.storage import TraceStore
from spectral.trackers import AsyncOpenAITracker, OpenAITracker


def client(fn):
    return NS(chat=NS(completions=NS(create=fn)))


def chunk(usage=None):
    return NS(model="gpt-4o", usage=usage, choices=[NS(delta=NS(content="private"))])


class Stream:
    def __init__(self, chunks, error=None):
        self.chunks = iter(chunks)
        self.error = error
        self.close = Mock()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            if self.error:
                raise self.error
            raise


class AsyncStream(Stream):
    def __init__(self, chunks, error=None):
        super().__init__(chunks, error)
        self.close = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration from None


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.store = TraceStore(":memory:")
        self.addCleanup(self.store.close)

    def tracked(self, source, **kwargs):
        return OpenAITracker(client(Mock(return_value=source)), self.store).create(model="gpt-4o", stream=True, **kwargs)

    def test_complete_identity_usage_privacy_and_once(self):
        chunks = [chunk(), chunk(NS(prompt_tokens=10, completion_tokens=5))]
        source = Stream(chunks)
        with patch("spectral.trackers.openai.time.perf_counter", return_value=10):
            tracked = self.tracked(source)
        self.assertEqual(self.store.query(), [])
        with patch("spectral.trackers._streaming.time.perf_counter", return_value=10.25):
            results = list(tracked)
        for actual, original in zip(results, chunks):
            self.assertIs(actual, original)
        tracked.close()
        self.assertEqual(list(tracked), [])
        source.close.assert_called_once()
        trace, = self.store.query()
        self.assertEqual(trace.status, Status.SUCCESS)
        self.assertEqual(trace.latency_ms, 250)
        self.assertEqual(trace.total_tokens, 15)
        self.assertNotIn("private", json.dumps(trace.to_dict()))

    def test_early_close_partial_summary_and_storage_filter(self):
        tracked = self.tracked(Stream([chunk()]))
        with tracked:
            next(tracked)
        trace, = self.store.query(status=Status.PARTIAL)
        self.assertEqual(trace.total_tokens, 0)
        stats = self.store.summary()
        self.assertEqual((stats.total_calls, stats.success_calls, stats.partial_calls, stats.error_calls), (1, 0, 1, 0))
        self.assertEqual(tracked._metadata.usage, None)

    def test_partial_survives_database_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.sqlite3"
            with TraceStore(path) as store:
                tracked = OpenAITracker(client(Mock(return_value=Stream([]))), store).create(model="gpt-4o", stream=True)
                tracked.close()
            with TraceStore(path) as store:
                self.assertEqual(len(store.query(status=Status.PARTIAL)), 1)
                self.assertEqual(store.summary().partial_calls, 1)

    def test_never_iterated_closed_is_partial(self):
        tracked = self.tracked(Stream([]))
        tracked.close()
        self.assertEqual(self.store.query()[0].status, Status.PARTIAL)

    def test_empty_and_missing_usage_complete(self):
        for chunks in [[], [chunk()]]:
            list(self.tracked(Stream(chunks)))
        self.assertTrue(all(t.status == Status.SUCCESS and t.total_tokens == 0 for t in self.store.query()))

    def test_provider_exception_identity_preserves_observed_usage(self):
        error = RuntimeError("private error")
        tracked = self.tracked(Stream([chunk(NS(prompt_tokens=4, completion_tokens=2))], error))
        next(tracked)
        with self.assertRaises(RuntimeError) as caught:
            next(tracked)
        self.assertIs(caught.exception, error)
        trace, = self.store.query()
        self.assertEqual((trace.status, trace.total_tokens), (Status.ERROR, 6))
        self.assertIsNone(trace.error_message)

    def test_context_exception_preserved_when_close_fails(self):
        source = Stream([chunk()])
        source.close.side_effect = ValueError("private cleanup")
        error = RuntimeError("caller error")
        with self.assertLogs("spectral.trackers.openai", level="WARNING"):
            with self.assertRaises(RuntimeError) as caught:
                with self.tracked(source):
                    raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(self.store.query()[0].status, Status.PARTIAL)

    def test_close_failure_is_error_and_repeated_close_noop(self):
        source = Stream([])
        error = ValueError("private")
        source.close.side_effect = error
        tracked = self.tracked(source)
        with self.assertRaises(ValueError) as caught:
            tracked.close()
        self.assertIs(caught.exception, error)
        tracked.close()
        self.assertEqual(self.store.query()[0].status, Status.ERROR)

    def test_storage_and_chunk_metadata_failure_do_not_change_chunks(self):
        class BadChunk:
            @property
            def usage(self):
                raise ValueError("private")
        item = BadChunk()
        source = Stream([item])
        tracked = self.tracked(source)
        with self.assertLogs("spectral.trackers.openai", level="WARNING"):
            self.assertIs(next(tracked), item)
        self.store.add = Mock(side_effect=RuntimeError("private storage"))
        with self.assertLogs("spectral.trackers.openai", level="WARNING") as logs:
            self.assertEqual(list(tracked), [])
        self.assertNotIn("private", str(logs.output))


class AsyncStreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = TraceStore(":memory:")
        self.addCleanup(self.store.close)

    async def tracked(self, source, **kwargs):
        return await AsyncOpenAITracker(client(AsyncMock(return_value=source)), self.store).create(model="gpt-4o", stream=True, **kwargs)

    async def test_complete_and_close_once(self):
        chunks = [chunk(), chunk(NS(prompt_tokens=10, completion_tokens=5))]
        source = AsyncStream(chunks)
        tracked = await self.tracked(source)
        results = [c async for c in tracked]
        self.assertTrue(all(a is b for a, b in zip(results, chunks)))
        await tracked.aclose()
        source.close.assert_awaited_once()
        trace, = self.store.query()
        self.assertEqual((trace.status, trace.total_tokens), (Status.SUCCESS, 15))
        self.assertNotIn("private", json.dumps(trace.to_dict()))

    async def test_cancel_during_close_is_error_and_preserved(self):
        source = AsyncStream([])
        error = asyncio.CancelledError("private")
        source.close.side_effect = error
        tracked = await self.tracked(source)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await tracked.close()
        self.assertIs(caught.exception, error)
        await tracked.close()
        source.close.assert_awaited_once()
        self.assertEqual(self.store.query()[0].error_type, "CancelledError")

    async def test_context_break_is_partial(self):
        tracked = await self.tracked(AsyncStream([chunk(), chunk()]))
        async with tracked:
            async for item in tracked:
                break
        self.assertEqual(self.store.query()[0].status, Status.PARTIAL)

    async def test_cancellation_identity_and_cleanup(self):
        error = asyncio.CancelledError("private")
        source = AsyncStream([], error)
        tracked = await self.tracked(source)
        with self.assertRaises(asyncio.CancelledError) as caught:
            await anext(tracked)
        self.assertIs(caught.exception, error)
        source.close.assert_awaited_once()
        trace, = self.store.query()
        self.assertEqual((trace.status, trace.error_type), (Status.ERROR, "CancelledError"))

    async def test_concurrent_stream_tag_isolation(self):
        async def consume(i):
            tracked = await self.tracked(AsyncStream([chunk()]), tags={"request": i})
            async with tracked:
                return [c async for c in tracked]
        await asyncio.gather(*(consume(i) for i in range(10)))
        self.assertEqual({t.tags["request"] for t in self.store.query()}, set(range(10)))
        self.assertTrue(all(t.status == Status.SUCCESS for t in self.store.query()))

    async def test_task_cancel_during_next_finalizes_and_closes(self):
        entered = asyncio.Event()
        class WaitingStream:
            close = AsyncMock()
            def __aiter__(self):
                return self
            async def __anext__(self):
                entered.set()
                await asyncio.Event().wait()
        source = WaitingStream()
        tracked = await self.tracked(source)
        task = asyncio.create_task(anext(tracked))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(task.cancelled())
        source.close.assert_awaited_once()
        trace, = self.store.query()
        self.assertEqual(trace.error_type, "CancelledError")

    async def test_storage_failure_preserves_provider_error(self):
        error = ValueError("private")
        self.store.add = Mock(side_effect=RuntimeError("private"))
        tracked = await self.tracked(AsyncStream([], error))
        with self.assertLogs("spectral.trackers.async_openai", level="WARNING") as logs:
            with self.assertRaises(ValueError) as caught:
                await anext(tracked)
        self.assertIs(caught.exception, error)
        self.assertNotIn("private", str(logs.output))


class RealSDKStreamingTests(unittest.IsolatedAsyncioTestCase):
    def payload(self):
        events = [
            {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
             "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "private"}, "finish_reason": None}]},
            {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
             "model": "gpt-4o", "choices": [],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ]
        return "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"

    async def test_real_sync_and_async_sse_usage_and_http_cleanup(self):
        import httpx
        from openai import AsyncOpenAI, OpenAI

        responses = []
        def respond(request):
            response = httpx.Response(200, content=self.payload(), headers={"content-type": "text/event-stream"})
            responses.append(response)
            return response

        with TraceStore(":memory:") as store:
            with OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
                with OpenAITracker(sdk, store).create(model="gpt-4o", messages=[], stream=True,
                                                     stream_options={"include_usage": True}) as tracked:
                    chunks = list(tracked)
                    self.assertEqual(chunks[0].choices[0].delta.content, "private")
            async with AsyncOpenAI(api_key="test-only", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as sdk:
                async with await AsyncOpenAITracker(sdk, store).create(model="gpt-4o", messages=[], stream=True,
                                                                      stream_options={"include_usage": True}) as tracked:
                    chunks = [c async for c in tracked]
                    self.assertEqual(chunks[0].choices[0].delta.content, "private")
            self.assertEqual([t.total_tokens for t in store.query()], [15, 15])
            self.assertEqual(store.summary().success_calls, 2)
        self.assertTrue(all(response.is_closed for response in responses))
