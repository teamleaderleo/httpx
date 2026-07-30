import anyio
import pytest

import httpx


class ControlFlowAbort(BaseException):
    pass


class BlockingAbortOnceCloseStream(httpx.AsyncByteStream):
    def __init__(self, error: BaseException) -> None:
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
async def test_control_flow_abort_is_not_replayed_to_waiter() -> None:
    error = ControlFlowAbort("stop this caller")
    stream = BlockingAbortOnceCloseStream(error)
    response = httpx.Response(200, stream=stream)
    owner_errors: list[BaseException] = []
    waiter_completed = anyio.Event()

    async def owner() -> None:
        try:
            await response.aclose()
        except ControlFlowAbort as exc:
            owner_errors.append(exc)

    async def waiter() -> None:
        await response.aclose()
        waiter_completed.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(owner)
        await stream.started.wait()
        task_group.start_soon(waiter)
        await anyio.sleep(0)
        stream.release.set()

    assert owner_errors == [error]
    assert waiter_completed.is_set() is True
    assert response.is_closed is True
    assert stream.close_calls == 2
    assert stream.cleaned is True
