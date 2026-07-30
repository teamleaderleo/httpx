import anyio
import pytest

import httpx


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


class CommitThenBlockStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.close_calls = 0
        self.cleanup_commits = 0

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.cleanup_commits += 1
        self.started.set()
        await anyio.sleep_forever()


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


def traceback_names(error: BaseException) -> list[str]:
    names: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return names


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [RuntimeError("ordinary close failed"), ControlFlowAbort("caller aborted")],
)
async def test_waiters_receive_distinct_neutral_terminal_failures(
    error: BaseException,
) -> None:
    stream = CommitThenRaiseStream(error)
    response = httpx.Response(200, stream=stream)
    owner_errors: list[BaseException] = []
    waiter_errors: dict[str, httpx.CloseError] = {}

    async def owner() -> None:
        try:
            await response.aclose()
        except BaseException as exc:
            owner_errors.append(exc)

    async def waiter_one() -> None:
        try:
            await response.aclose()
        except httpx.CloseError as exc:
            waiter_errors["one"] = exc

    async def waiter_two() -> None:
        try:
            await response.aclose()
        except httpx.CloseError as exc:
            waiter_errors["two"] = exc

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(owner)
        await stream.started.wait()
        task_group.start_soon(waiter_one)
        task_group.start_soon(waiter_two)
        await anyio.sleep(0)
        stream.release.set()

    first = waiter_errors["one"]
    second = waiter_errors["two"]
    assert owner_errors == [error]
    assert first is not second
    assert first.__cause__ is error
    assert second.__cause__ is error
    assert "waiter_one" in traceback_names(first)
    assert "waiter_two" not in traceback_names(first)
    assert "waiter_two" in traceback_names(second)
    assert "waiter_one" not in traceback_names(second)
    assert stream.close_calls == 1
    assert stream.cleanup_commits == 1
    assert response.is_closed is False

    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert later.value is not first
    assert later.value is not second
    assert later.value.__cause__ is error
    assert "waiter_one" not in traceback_names(later.value)
    assert "waiter_two" not in traceback_names(later.value)
    assert stream.close_calls == 1

    iterator = response.aiter_raw()
    with pytest.raises(httpx.StreamClosed):
        await iterator.__anext__()


@pytest.mark.anyio
async def test_actual_owner_cancellation_is_preserved_and_neutralized_for_waiters() -> None:
    stream = CommitThenBlockStream()
    response = httpx.Response(200, stream=stream)
    owner_errors: list[BaseException] = []
    waiter_errors: list[httpx.CloseError] = []
    owner_scope: list[anyio.CancelScope] = []

    async def owner() -> None:
        with anyio.CancelScope() as scope:
            owner_scope.append(scope)
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
        await anyio.sleep(0)
        owner_scope[0].cancel()

    assert len(owner_errors) == 1
    assert isinstance(owner_errors[0], anyio.get_cancelled_exc_class())
    assert len(waiter_errors) == 1
    assert waiter_errors[0].__cause__ is owner_errors[0]
    assert stream.close_calls == 1
    assert stream.cleanup_commits == 1

    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert later.value is not waiter_errors[0]
    assert later.value.__cause__ is owner_errors[0]
    assert stream.close_calls == 1


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
async def test_later_retry_does_not_repeat_committed_cleanup() -> None:
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

    with pytest.raises(httpx.CloseError) as first:
        await response.aclose()
    with pytest.raises(httpx.CloseError) as second:
        await response.aclose()
    assert first.value is not second.value
    assert first.value.__cause__ is error
    assert second.value.__cause__ is error
    assert stream.close_calls == 1
    assert stream.cleanup_commits == 1
