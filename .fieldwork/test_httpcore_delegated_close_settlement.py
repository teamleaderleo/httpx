from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
import pytest

from httpcore import Request
from httpcore._async.http11 import HTTP11ConnectionByteStream
from httpcore._async.http2 import HTTP2ConnectionByteStream


class BlockingResponseClose:
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.calls = 0
        self.cleaned = False

    async def _response_closed(self, stream_id: int | None = None) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        self.cleaned = True


class FailOnceResponseClose:
    def __init__(self) -> None:
        self.calls = 0
        self.cleaned = False

    async def _response_closed(self, stream_id: int | None = None) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("delegated close failed")
        self.cleaned = True


def build_http11_stream(connection: Any) -> HTTP11ConnectionByteStream:
    request = Request("GET", "https://example.org/")
    return HTTP11ConnectionByteStream(connection, request)


def build_http2_stream(connection: Any) -> HTTP2ConnectionByteStream:
    request = Request("GET", "https://example.org/")
    return HTTP2ConnectionByteStream(connection, request, stream_id=1)


def build_http11_stream_with_request(
    connection: Any, request: Request
) -> HTTP11ConnectionByteStream:
    return HTTP11ConnectionByteStream(connection, request)


def build_http2_stream_with_request(
    connection: Any, request: Request
) -> HTTP2ConnectionByteStream:
    return HTTP2ConnectionByteStream(connection, request, stream_id=1)


StreamFactory = Callable[[Any], Any]
TraceStreamFactory = Callable[[Any, Request], Any]


@pytest.mark.anyio
@pytest.mark.parametrize("stream_factory", [build_http11_stream, build_http2_stream])
async def test_cancelled_delegated_close_is_not_retried(
    stream_factory: StreamFactory,
) -> None:
    connection = BlockingResponseClose()
    stream = stream_factory(connection)
    cancel_scope = anyio.CancelScope()

    async def close_once() -> None:
        with cancel_scope:
            await stream.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_once)
        await connection.started.wait()
        cancel_scope.cancel()

    assert connection.calls == 1
    assert connection.cleaned is False

    connection.release.set()
    await stream.aclose()

    assert connection.calls == 1
    assert connection.cleaned is False


@pytest.mark.anyio
@pytest.mark.parametrize("stream_factory", [build_http11_stream, build_http2_stream])
async def test_ordinary_delegated_close_failure_is_not_retried(
    stream_factory: StreamFactory,
) -> None:
    connection = FailOnceResponseClose()
    stream = stream_factory(connection)

    with pytest.raises(RuntimeError, match="delegated close failed"):
        await stream.aclose()

    assert connection.calls == 1
    assert connection.cleaned is False

    await stream.aclose()

    assert connection.calls == 1
    assert connection.cleaned is False


@pytest.mark.anyio
@pytest.mark.parametrize("stream_factory", [build_http11_stream, build_http2_stream])
async def test_second_close_returns_before_delegated_release_finishes(
    stream_factory: StreamFactory,
) -> None:
    connection = BlockingResponseClose()
    stream = stream_factory(connection)
    second_completed = anyio.Event()
    completed: list[str] = []

    async def first_close() -> None:
        await stream.aclose()
        completed.append("first")

    async def second_close() -> None:
        await stream.aclose()
        completed.append("second")
        second_completed.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(first_close)
        await connection.started.wait()
        task_group.start_soon(second_close)
        await second_completed.wait()

        assert connection.calls == 1
        assert completed == ["second"]

        connection.release.set()

    assert completed == ["second", "first"]
    assert connection.cleaned is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stream_factory",
    [build_http11_stream_with_request, build_http2_stream_with_request],
)
async def test_response_closed_trace_callback_can_reenter_close_without_cycle(
    stream_factory: TraceStreamFactory,
) -> None:
    connection = BlockingResponseClose()
    connection.release.set()
    stream_ref: dict[str, Any] = {}
    nested_completed = anyio.Event()

    async def trace(name: str, info: dict[str, Any]) -> None:
        if name.endswith("response_closed.started"):
            await stream_ref["stream"].aclose()
            nested_completed.set()

    request = Request(
        "GET", "https://example.org/", extensions={"trace": trace}
    )
    stream = stream_factory(connection, request)
    stream_ref["stream"] = stream

    with anyio.fail_after(1):
        await stream.aclose()

    assert nested_completed.is_set()
    assert connection.calls == 1
    assert connection.cleaned is True


@pytest.mark.anyio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "HTTPCore publishes byte-stream closed state before delegated release settles"
    ),
)
@pytest.mark.parametrize("stream_factory", [build_http11_stream, build_http2_stream])
async def test_interrupted_delegated_close_should_remain_retryable(
    stream_factory: StreamFactory,
) -> None:
    connection = BlockingResponseClose()
    stream = stream_factory(connection)
    cancel_scope = anyio.CancelScope()

    async def close_once() -> None:
        with cancel_scope:
            await stream.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_once)
        await connection.started.wait()
        cancel_scope.cancel()

    connection.release.set()
    await stream.aclose()

    assert connection.calls == 2
    assert connection.cleaned is True
