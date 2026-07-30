import pickle

import anyio
import pytest

import httpx


class BlockingCloseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def __aiter__(self):
        if False:
            yield b""

    async def aclose(self) -> None:
        self.started.set()
        await self.release.wait()


@pytest.mark.anyio
async def test_pickle_discards_in_progress_close_attempt() -> None:
    stream = BlockingCloseStream()
    response = httpx.Response(200, stream=stream)
    cancel_scope = anyio.CancelScope()

    async def close_once() -> None:
        with cancel_scope:
            await response.aclose()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(close_once)
        await stream.started.wait()

        restored = pickle.loads(pickle.dumps(response))

        assert restored.is_closed is True
        assert restored._async_close_started is True
        assert restored._async_close_state is None
        with pytest.raises(httpx.StreamClosed):
            await restored.aread()

        cancel_scope.cancel()

    assert response.is_closed is False
