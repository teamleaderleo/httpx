from __future__ import annotations

import anyio
import pytest

import httpx


class CancellableCloseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.close_calls = 0
        self.cancelled_error: BaseException | None = None

    async def __aiter__(self):
        if False:  # pragma: no cover
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await anyio.sleep_forever()
        except BaseException as exc:
            self.cancelled_error = exc
            raise


@pytest.mark.anyio
async def test_cancelled_owner_preserves_backend_error_and_terminalizes_observers() -> (
    None
):
    request = httpx.Request("GET", "https://example.org")
    stream = CancellableCloseStream()
    response = httpx.Response(200, request=request, stream=stream)
    owner_errors: list[BaseException] = []
    waiter_errors: list[httpx.CloseError] = []
    owner_scope: anyio.CancelScope | None = None

    async def owner() -> None:
        nonlocal owner_scope
        with anyio.CancelScope() as scope:
            owner_scope = scope
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
        assert owner_scope is not None
        task_group.start_soon(waiter)
        task_group.start_soon(waiter)
        await anyio.sleep(0)
        owner_scope.cancel()

    assert len(owner_errors) == 1
    assert isinstance(owner_errors[0], anyio.get_cancelled_exc_class())
    assert owner_errors[0] is stream.cancelled_error
    assert not isinstance(owner_errors[0], httpx.CloseError)

    assert len(waiter_errors) == 2
    assert waiter_errors[0] is not waiter_errors[1]
    for error in waiter_errors:
        assert error.request is request
        assert isinstance(error.__cause__, httpx.CloseError)
        assert error.__cause__ is not owner_errors[0]
        assert error.__cause__.__traceback__ is None

    assert stream.close_calls == 1
    assert response.is_closed is False

    with pytest.raises(httpx.CloseError) as later:
        await response.aclose()
    assert later.value is not waiter_errors[0]
    assert later.value is not waiter_errors[1]
    assert later.value.request is request
    assert isinstance(later.value.__cause__, httpx.CloseError)
    assert later.value.__cause__ is not owner_errors[0]
    assert stream.close_calls == 1

    iterator = response.aiter_raw()
    with pytest.raises(httpx.StreamClosed):
        await iterator.__anext__()
