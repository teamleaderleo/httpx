import anyio
import pytest

import httpx


class BlockingDelegatingCloseStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self.stream = stream
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.close_calls = 0
        self.delegated_close_calls = 0

    async def __aiter__(self):
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()
        self.delegated_close_calls += 1
        await self.stream.aclose()


@pytest.mark.anyio
async def test_cancelled_close_retry_releases_default_transport_pool_slot(server) -> None:
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    timeout = httpx.Timeout(5.0, pool=0.2)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        request = client.build_request("GET", server.url)
        response = await client.send(request, stream=True)
        assert isinstance(response.stream, httpx.AsyncByteStream)

        stream = BlockingDelegatingCloseStream(response.stream)
        response.stream = stream
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
        assert stream.delegated_close_calls == 0

        stream.release.set()
        await response.aclose()

        assert response.is_closed is True
        assert stream.close_calls == 2
        assert stream.delegated_close_calls == 1

        follow_up = await client.get(server.url)
        assert follow_up.status_code == 200
        assert follow_up.text == "Hello, world!"
