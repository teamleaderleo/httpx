from __future__ import annotations

import gc
import weakref

import anyio
import pytest

import httpx


OBSERVER_CAUSE_MESSAGE = (
    "The original stream close failure was delivered to the initiating caller."
)


def traceback_depth(exc: BaseException) -> int:
    depth = 0
    traceback = exc.__traceback__
    while traceback is not None:
        depth += 1
        traceback = traceback.tb_next
    return depth


def assert_neutral_observer_failure(
    exc: httpx.CloseError, request: httpx.Request | None
) -> httpx.CloseError:
    if request is None:
        with pytest.raises(RuntimeError, match="request property has not been set"):
            assert exc.request is None
    else:
        assert exc.request is request

    cause = exc.__cause__
    assert isinstance(cause, httpx.CloseError)
    assert cause.args == (OBSERVER_CAUSE_MESSAGE,)
    assert cause.__traceback__ is None
    return cause


class ControlFlowAbort(BaseException):
    pass


class CommitThenRaiseStream(httpx.AsyncByteStream):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.close_calls = 0
        self.cleanup_commits = 0

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()
        self.cleanup_commits += 1
        raise self.error


class SuccessfulBlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.close_calls = 0

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()


class FrameMarker:
    pass


class FrameLocalFailureStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.marker_ref: weakref.ReferenceType[FrameMarker] | None = None
        self.close_calls = 0

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        marker = FrameMarker()
        self.marker_ref = weakref.ref(marker)
        raise RuntimeError("frame-local close failure")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [RuntimeError("ordinary close failed"), ControlFlowAbort("caller aborted")],
)
async def test_waiters_observe_distinct_neutral_terminal_failures(
    error: BaseException,
) -> None:
    request = httpx.Request("GET", "https://example.org")
    stream = CommitThenRaiseStream(error)
    response = httpx.Response(200, request=request, stream=stream)
    owner_errors: list[BaseException] = []
    waiter_errors: list[httpx.CloseError] = []

    async def owner() -> None:
        try:
            await response.aclose()
        except BaseException as exc:
            owner_errors.append(exc)

    async def waiter() -> None:
        try:
            await response.aclose()
        except httpx.CloseError as exc:
            waiter_errors.append(exc)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(owner)
        await stream.started.wait()
        task_group.start_soon(waiter)
        task_group.start_soon(waiter)
        await anyio.sleep(0)
        stream.release.set()

    assert owner_errors == [error]
    assert len(waiter_errors) == 2
    assert waiter_errors[0] is not waiter_errors[1]
    assert waiter_errors[0].args == waiter_errors[1].args
    waiter_causes = [
        assert_neutral_observer_failure(exc, request) for exc in waiter_errors
    ]
    assert waiter_causes[0] is not waiter_causes[1]
    assert error not in waiter_causes
    assert waiter_errors[0].__traceback__ is not waiter_errors[1].__traceback__
    assert stream.close_calls == 1
    assert stream.cleanup_commits == 1
    assert response.is_closed is False

    waiter_depths = [traceback_depth(exc) for exc in waiter_errors]
    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert later.value is not waiter_errors[0]
    assert later.value is not waiter_errors[1]
    later_cause = assert_neutral_observer_failure(later.value, request)
    assert later_cause not in waiter_causes
    later_depth = traceback_depth(later.value)

    with pytest.raises(httpx.CloseError) as repeated:
        await response.aclose()
    assert repeated.value is not later.value
    repeated_cause = assert_neutral_observer_failure(repeated.value, request)
    assert repeated_cause is not later_cause
    assert repeated_cause not in waiter_causes
    assert traceback_depth(repeated.value) == later_depth
    assert [traceback_depth(exc) for exc in waiter_errors] == waiter_depths
    assert stream.close_calls == 1

    iterator = response.aiter_raw()
    with pytest.raises(httpx.StreamClosed):
        await iterator.__anext__()


@pytest.mark.anyio
async def test_successful_waiters_share_one_close_attempt() -> None:
    stream = SuccessfulBlockingStream()
    response = httpx.Response(200, stream=stream)
    completed = 0

    async def close() -> None:
        nonlocal completed
        await response.aclose()
        completed += 1

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close)
        await stream.started.wait()
        task_group.start_soon(close)
        task_group.start_soon(close)
        await anyio.sleep(0)
        stream.release.set()

    assert completed == 3
    assert stream.close_calls == 1
    assert response.is_closed is True
    await response.aclose()
    assert stream.close_calls == 1


@pytest.mark.anyio
async def test_later_close_does_not_repeat_committed_cleanup() -> None:
    error = RuntimeError("cleanup committed")
    stream = CommitThenRaiseStream(error)
    response = httpx.Response(200, stream=stream)

    async def release() -> None:
        await stream.started.wait()
        stream.release.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(release)
        with pytest.raises(RuntimeError) as owner:
            await response.aclose()
    assert owner.value is error

    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert_neutral_observer_failure(later.value, None)
    assert stream.close_calls == 1
    assert stream.cleanup_commits == 1


@pytest.mark.anyio
async def test_terminal_response_does_not_retain_owner_traceback_frame() -> None:
    stream = FrameLocalFailureStream()
    response = httpx.Response(200, stream=stream)

    async def run_owner() -> None:
        try:
            await response.aclose()
        except RuntimeError as exc:
            assert str(exc) == "frame-local close failure"
        else:  # pragma: no cover
            raise AssertionError("owner failure did not escape")

    await run_owner()
    marker_ref = stream.marker_ref
    assert marker_ref is not None
    gc.collect()
    assert marker_ref() is None

    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert_neutral_observer_failure(later.value, None)
    assert stream.close_calls == 1


@pytest.mark.anyio
async def test_terminal_observer_failure_is_valid_without_request() -> None:
    error = RuntimeError("requestless close failed")
    stream = CommitThenRaiseStream(error)
    response = httpx.Response(200, stream=stream)

    async def release() -> None:
        await stream.started.wait()
        stream.release.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(release)
        with pytest.raises(RuntimeError) as owner:
            await response.aclose()
    assert owner.value is error

    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert_neutral_observer_failure(later.value, None)
    assert stream.close_calls == 1
