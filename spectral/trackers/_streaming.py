"""Metadata-only stream wrappers; finalize once at exhaustion, close or failure."""

import asyncio
import time
from types import SimpleNamespace

from spectral.models import Status
from spectral.trackers._common import record_call


class _Lifecycle:
    def __init__(self, source, store, logger, model, timestamp, start, tags):
        self._source = source
        self._store = store
        self._logger = logger
        self._model = model
        self._timestamp = timestamp
        self._start = start
        self._tags = tags
        self._done = False
        self._iterator = None
        self._metadata = SimpleNamespace(model=model, usage=None)

    def _observe(self, chunk):
        try:
            model = getattr(chunk, "model", None)
            if isinstance(model, str) and model:
                self._metadata.model = model
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._metadata.usage = SimpleNamespace(
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
        except Exception:
            self._logger.warning("Spectral could not read streaming usage metadata")

    def _finish(self, status, error=None):
        if self._done:
            return
        self._done = True
        record_call(self._store, self._logger, self._model, self._timestamp,
                    (time.perf_counter() - self._start) * 1000, self._tags,
                    response=self._metadata, error=error, status=status)


class TrackedStream(_Lifecycle):
    """Use as a context manager, or close explicitly when stopping early."""

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        try:
            if self._iterator is None:
                self._iterator = iter(self._source)
            chunk = next(self._iterator)
        except StopIteration:
            self._close(Status.SUCCESS)
            raise
        except Exception as exc:
            self._finish(Status.ERROR, exc)
            try:
                self._source.close()
            except Exception:
                self._logger.warning("Spectral could not close provider stream")
            raise
        self._observe(chunk)
        return chunk

    def _close(self, status):
        if self._done:
            return
        try:
            self._source.close()
        except Exception as exc:
            self._finish(Status.ERROR, exc)
            raise
        self._finish(status)

    def close(self):
        self._close(Status.PARTIAL)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._done:
            return False
        if exc is None:
            self.close()
        else:
            self._finish(Status.PARTIAL)
            try:
                self._source.close()
            except Exception:
                self._logger.warning("Spectral could not close provider stream")
        return False


class AsyncTrackedStream(_Lifecycle):
    """Use async with, or await close() explicitly when stopping early."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        try:
            if self._iterator is None:
                self._iterator = aiter(self._source)
            chunk = await anext(self._iterator)
        except StopAsyncIteration:
            await self._close(Status.SUCCESS)
            raise
        except (Exception, asyncio.CancelledError) as exc:
            self._finish(Status.ERROR, exc)
            try:
                await self._source.close()
            except (Exception, asyncio.CancelledError):
                self._logger.warning("Spectral could not close provider stream")
            raise
        self._observe(chunk)
        return chunk

    async def _close(self, status):
        if self._done:
            return
        try:
            await self._source.close()
        except (Exception, asyncio.CancelledError) as exc:
            self._finish(Status.ERROR, exc)
            raise
        self._finish(status)

    async def close(self):
        await self._close(Status.PARTIAL)

    async def aclose(self):
        await self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._done:
            return False
        if exc is None:
            await self.close()
        else:
            self._finish(Status.ERROR if isinstance(exc, asyncio.CancelledError) else Status.PARTIAL,
                         exc if isinstance(exc, asyncio.CancelledError) else None)
            try:
                await self._source.close()
            except (Exception, asyncio.CancelledError):
                self._logger.warning("Spectral could not close provider stream")
        return False
