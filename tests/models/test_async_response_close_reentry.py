from __future__ import annotations

import anyio
import pytest

import httpx

REENTRY_MESSAGE = (
    "Attempted to re-enter response close from the stream close operation."
)


def assert_error_request(
    error: httpx.CloseError, request: httpx.Request | None
) -> None:
    if request is None:
        with pytest.raises(RuntimeError):
            _ = error.request
    else:
        assert error.request is request


class EscapingReentrantStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.response: httpx.Response | None = None
        self.close_calls = 0

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        assert self.response is not None
        await self.response.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("with_request", [False, True])
async def test_reentrant_close_fails_without_waiting(with_request: bool) -> None:
    request = httpx.Request("GET", "https://example.org") if with_request else None
    stream = EscapingReentrantStream()
    response = httpx.Response(200, request=request, stream=stream)
    stream.response = response

    with anyio.fail_after(1):
        with pytest.raises(httpx.CloseError, match=REENTRY_MESSAGE) as owner:
            await response.aclose()

    assert_error_request(owner.value, request)
    assert stream.close_calls == 1
    assert response.is_closed is False

    with pytest.raises(httpx.CloseError, match="close outcome is unknown") as later:
        await response.aclose()
    assert_error_request(later.value, request)
    assert later.value is not owner.value
    assert stream.close_calls == 1


class CatchingReentrantStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.response: httpx.Response | None = None
        self.started = anyio.Event()
        self.allow_reentry = anyio.Event()
        self.close_calls = 0
        self.reentry_error: httpx.CloseError | None = None

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.allow_reentry.wait()
        assert self.response is not None
        try:
            await self.response.aclose()
        except httpx.CloseError as error:
            self.reentry_error = error


@pytest.mark.anyio
async def test_caught_reentrant_close_does_not_poison_external_waiter() -> None:
    request = httpx.Request("GET", "https://example.org")
    stream = CatchingReentrantStream()
    response = httpx.Response(200, request=request, stream=stream)
    stream.response = response
    waiter_started = anyio.Event()
    waiter_done = anyio.Event()

    async def owner() -> None:
        await response.aclose()

    async def waiter() -> None:
        waiter_started.set()
        try:
            await response.aclose()
        finally:
            waiter_done.set()

    with anyio.fail_after(1):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(owner)
            await stream.started.wait()
            task_group.start_soon(waiter)
            await waiter_started.wait()
            await anyio.sleep(0)
            assert not waiter_done.is_set()
            stream.allow_reentry.set()

    assert stream.close_calls == 1
    assert response.is_closed is True
    assert stream.reentry_error is not None
    assert stream.reentry_error.args == (REENTRY_MESSAGE,)
    assert stream.reentry_error.request is request


class DescendantReentrantStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.response: httpx.Response | None = None
        self.close_calls = 0
        self.reentry_error: httpx.CloseError | None = None

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        assert self.response is not None

        async def descendant() -> None:
            assert self.response is not None
            try:
                await self.response.aclose()
            except httpx.CloseError as error:
                self.reentry_error = error

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(descendant)


@pytest.mark.anyio
async def test_descendant_reentrant_close_does_not_deadlock() -> None:
    request = httpx.Request("GET", "https://example.org")
    stream = DescendantReentrantStream()
    response = httpx.Response(200, request=request, stream=stream)
    stream.response = response

    with anyio.fail_after(1):
        await response.aclose()

    assert stream.close_calls == 1
    assert response.is_closed is True
    assert stream.reentry_error is not None
    assert stream.reentry_error.args == (REENTRY_MESSAGE,)
    assert stream.reentry_error.request is request


class NestedCloseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.inner_response: httpx.Response | None = None
        self.close_calls = 0

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        assert self.inner_response is not None
        await self.inner_response.aclose()


class OuterBackReferenceStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.outer_response: httpx.Response | None = None
        self.close_calls = 0
        self.reentry_error: httpx.CloseError | None = None

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        assert self.outer_response is not None
        try:
            await self.outer_response.aclose()
        except httpx.CloseError as error:
            self.reentry_error = error


@pytest.mark.anyio
async def test_nested_response_close_cycle_does_not_deadlock() -> None:
    outer_request = httpx.Request("GET", "https://outer.example.org")
    outer_stream = NestedCloseStream()
    inner_stream = OuterBackReferenceStream()
    outer_response = httpx.Response(200, request=outer_request, stream=outer_stream)
    inner_response = httpx.Response(200, stream=inner_stream)
    outer_stream.inner_response = inner_response
    inner_stream.outer_response = outer_response

    with anyio.fail_after(1):
        await outer_response.aclose()

    assert outer_stream.close_calls == 1
    assert inner_stream.close_calls == 1
    assert outer_response.is_closed is True
    assert inner_response.is_closed is True
    assert inner_stream.reentry_error is not None
    assert inner_stream.reentry_error.args == (REENTRY_MESSAGE,)
    assert inner_stream.reentry_error.request is outer_request
