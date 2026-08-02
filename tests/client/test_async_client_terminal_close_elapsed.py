from __future__ import annotations

import anyio
import pytest

import httpx


class FailingCloseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.close_calls = 0
        self.error = RuntimeError("stream cleanup failed")

    async def __aiter__(self):
        yield b"body"  # pragma: no cover

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self.error


class FailingCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self, stream: FailingCloseStream) -> None:
        self.stream = stream

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, stream=self.stream)


@pytest.mark.anyio
async def test_elapsed_is_not_published_when_stream_cleanup_fails() -> None:
    stream = FailingCloseStream()
    transport = FailingCloseTransport(stream)

    async with httpx.AsyncClient(transport=transport) as client:
        request = client.build_request("GET", "https://example.org")
        response = await client.send(request, stream=True)

        with pytest.raises(RuntimeError) as owner:
            await response.aclose()
        assert owner.value is stream.error
        assert stream.close_calls == 1

        with pytest.raises(
            RuntimeError,
            match="may only be accessed after the response has been read or closed",
        ):
            _ = response.elapsed

        with pytest.raises(httpx.CloseError) as observer:
            await response.aclose()
        assert observer.value.request is request
        assert stream.close_calls == 1

        with pytest.raises(
            RuntimeError,
            match="may only be accessed after the response has been read or closed",
        ):
            _ = response.elapsed


class BlockingCloseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.started.set()
        await self.release.wait()


class BlockingCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self, stream: BlockingCloseStream) -> None:
        self.stream = stream

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, stream=self.stream)


class Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.mark.anyio
async def test_elapsed_excludes_stream_cleanup_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock(10.0)
    monkeypatch.setattr("httpx._client.time.perf_counter", clock)
    stream = BlockingCloseStream()
    transport = BlockingCloseTransport(stream)
    client = httpx.AsyncClient(transport=transport)
    request = client.build_request("GET", "https://example.org")
    response = await client.send(request, stream=True)
    clock.value = 12.0

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(response.aclose)
        await stream.started.wait()
        with pytest.raises(
            RuntimeError,
            match="may only be accessed after the response has been read or closed",
        ):
            _ = response.elapsed
        clock.value = 20.0
        stream.release.set()

    assert response.elapsed.total_seconds() == 2.0
    await client.aclose()
