import pytest

import httpx


class FailOnceSyncStream(httpx.SyncByteStream):
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.close_calls = 0
        self.cleaned = False

    def __iter__(self):
        return iter(())

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise self.error
        self.cleaned = True


class SuccessfulSyncStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.close_calls = 0

    def __iter__(self):
        return iter(())

    def close(self) -> None:
        self.close_calls += 1


def test_current_failed_close_is_terminal_and_nonretryable() -> None:
    stream = FailOnceSyncStream(RuntimeError("sync close failed"))
    response = httpx.Response(200, stream=stream)

    with pytest.raises(RuntimeError, match="sync close failed"):
        response.close()

    assert response.is_closed is True
    assert stream.close_calls == 1
    assert stream.cleaned is False

    response.close()

    assert response.is_closed is True
    assert stream.close_calls == 1
    assert stream.cleaned is False


@pytest.mark.xfail(
    strict=True,
    reason="Response.close publishes is_closed before stream cleanup completes",
)
def test_failed_close_can_be_retried_after_error() -> None:
    stream = FailOnceSyncStream(RuntimeError("sync close failed"))
    response = httpx.Response(200, stream=stream)

    with pytest.raises(RuntimeError, match="sync close failed"):
        response.close()

    assert response.is_closed is False

    response.close()

    assert response.is_closed is True
    assert stream.close_calls == 2
    assert stream.cleaned is True


def test_failed_close_keeps_body_iteration_blocked() -> None:
    stream = FailOnceSyncStream(RuntimeError("sync close failed"))
    response = httpx.Response(200, stream=stream)

    with pytest.raises(RuntimeError, match="sync close failed"):
        response.close()

    with pytest.raises(httpx.StreamClosed):
        list(response.iter_raw())


def test_close_error_retains_request_context() -> None:
    request = httpx.Request("GET", "https://example.org")
    error = httpx.CloseError("sync close failed")
    stream = FailOnceSyncStream(error)
    response = httpx.Response(200, stream=stream, request=request)

    with pytest.raises(httpx.CloseError) as exc_info:
        response.close()

    assert exc_info.value is error
    assert exc_info.value.request is request


def test_repeated_successful_close_is_idempotent() -> None:
    stream = SuccessfulSyncStream()
    response = httpx.Response(200, stream=stream)

    response.close()
    response.close()

    assert response.is_closed is True
    assert stream.close_calls == 1
