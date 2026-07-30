import time

import anyio
import pytest

import httpx
from httpx._client import BoundAsyncStream


class BlockingCloseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.close_calls = 0
        self.cleaned = False

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()
        self.cleaned = True


class FailOnceCloseStream(httpx.AsyncByteStream):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.close_calls = 0
        self.cleaned = False

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise self.error
        self.cleaned = True


class BlockingFailOnceCloseStream(httpx.AsyncByteStream):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.close_calls = 0
        self.cleaned = False

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            self.started.set()
            await self.release.wait()
            raise self.error
        self.cleaned = True


@pytest.mark.anyio
async def test_cancelled_close_remains_retryable() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)
    cancel_scope = anyio.CancelScope()

    async def close_once() -> None:
        with cancel_scope:
            await response.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_once)
        await stream.started.wait()
        cancel_scope.cancel()

    assert response.is_closed is False
    assert stream.close_calls == 1
    assert stream.cleaned is False

    stream.release.set()
    await response.aclose()

    assert response.is_closed is True
    assert stream.close_calls == 2
    assert stream.cleaned is True


@pytest.mark.anyio
async def test_already_cancelled_close_remains_retryable() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)

    with anyio.CancelScope() as cancel_scope:
        cancel_scope.cancel()
        await response.aclose()

    assert response.is_closed is False
    assert stream.close_calls == 1
    assert stream.cleaned is False

    stream.release.set()
    await response.aclose()

    assert response.is_closed is True
    assert stream.close_calls == 2
    assert stream.cleaned is True


@pytest.mark.anyio
async def test_close_failure_remains_retryable() -> None:
    error = RuntimeError("close failed")
    stream = FailOnceCloseStream(error)
    response = httpx.Response(200, stream=stream)

    with pytest.raises(RuntimeError, match="close failed"):
        await response.aclose()

    assert response.is_closed is False
    assert stream.close_calls == 1
    assert stream.cleaned is False

    await response.aclose()

    assert response.is_closed is True
    assert stream.close_calls == 2
    assert stream.cleaned is True


@pytest.mark.anyio
async def test_concurrent_close_callers_share_one_attempt() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)
    completed: list[str] = []

    async def close_and_record(name: str) -> None:
        await response.aclose()
        completed.append(name)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_and_record, "first")
        await stream.started.wait()
        task_group.start_soon(close_and_record, "second")
        await anyio.sleep(0)

        assert response.is_closed is False
        assert stream.close_calls == 1
        assert completed == []

        stream.release.set()

    assert response.is_closed is True
    assert stream.close_calls == 1
    assert sorted(completed) == ["first", "second"]


@pytest.mark.anyio
async def test_cancelling_one_waiter_does_not_cancel_owner() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)
    owner_completed = anyio.Event()
    waiter_scope = anyio.CancelScope()

    async def owner() -> None:
        await response.aclose()
        owner_completed.set()

    async def waiter() -> None:
        with waiter_scope:
            await response.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(owner)
        await stream.started.wait()
        task_group.start_soon(waiter)
        await anyio.sleep(0)
        waiter_scope.cancel()
        await anyio.sleep(0)

        assert owner_completed.is_set() is False
        assert stream.close_calls == 1

        stream.release.set()

    assert owner_completed.is_set() is True
    assert response.is_closed is True
    assert stream.close_calls == 1
    assert stream.cleaned is True


@pytest.mark.anyio
async def test_close_failure_is_shared_with_current_waiters() -> None:
    error = RuntimeError("close failed")
    stream = BlockingFailOnceCloseStream(error)
    response = httpx.Response(200, stream=stream)
    errors: list[BaseException] = []

    async def close_and_record_error() -> None:
        try:
            await response.aclose()
        except BaseException as exc:
            errors.append(exc)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_and_record_error)
        await stream.started.wait()
        task_group.start_soon(close_and_record_error)
        await anyio.sleep(0)
        assert stream.close_calls == 1
        stream.release.set()

    assert len(errors) == 2
    assert all(exc is error for exc in errors)
    assert response.is_closed is False
    assert stream.close_calls == 1

    await response.aclose()

    assert response.is_closed is True
    assert stream.close_calls == 2
    assert stream.cleaned is True


@pytest.mark.anyio
async def test_close_start_blocks_new_body_iteration() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)
    cancel_scope = anyio.CancelScope()

    async def close_once() -> None:
        with cancel_scope:
            await response.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_once)
        await stream.started.wait()

        assert response.is_closed is False
        with pytest.raises(httpx.StreamClosed):
            await response.aread()

        cancel_scope.cancel()

    assert response.is_closed is False
    with pytest.raises(httpx.StreamClosed):
        await response.aread()

    stream.release.set()
    await response.aclose()
    assert response.is_closed is True


@pytest.mark.anyio
async def test_repeated_successful_close_is_idempotent() -> None:
    stream = BlockingCloseStream()
    stream.release.set()
    response = httpx.Response(200, stream=stream)

    await response.aclose()
    await response.aclose()

    assert response.is_closed is True
    assert stream.close_calls == 1
    assert stream.cleaned is True


@pytest.mark.anyio
async def test_elapsed_is_finalized_only_after_close_completes() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)
    response.stream = BoundAsyncStream(stream, response, time.perf_counter())

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(response.aclose)
        await stream.started.wait()

        assert response.is_closed is False
        with pytest.raises(RuntimeError, match="may only be accessed"):
            response.elapsed  # noqa: B018

        stream.release.set()

    assert response.is_closed is True
    assert response.elapsed.total_seconds() >= 0
