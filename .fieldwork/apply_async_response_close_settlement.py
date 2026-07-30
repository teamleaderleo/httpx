#!/usr/bin/env python3
"""Apply the Fieldwork async response close candidate to the pinned HTTPX source."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_exact(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one source anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    client = ROOT / "httpx" / "_client.py"
    models = ROOT / "httpx" / "_models.py"

    replace_exact(
        client,
        '''    async def aclose(self) -> None:
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
        await self._stream.aclose()
''',
        '''    async def aclose(self) -> None:
        await self._stream.aclose()
        elapsed = time.perf_counter() - self._start
        self._response.elapsed = datetime.timedelta(seconds=elapsed)
''',
        "BoundAsyncStream elapsed ordering",
    )

    replace_exact(
        models,
        '''from collections.abc import Mapping
from http.cookiejar import Cookie, CookieJar

from ._content import ByteStream, UnattachedStream, encode_request, encode_response
''',
        '''from collections.abc import Mapping
from http.cookiejar import Cookie, CookieJar

import anyio

from ._content import ByteStream, UnattachedStream, encode_request, encode_response
''',
        "AnyIO import",
    )

    replace_exact(
        models,
        '''SENSITIVE_HEADERS = {"authorization", "proxy-authorization"}


def _is_known_encoding(encoding: str) -> bool:
''',
        '''SENSITIVE_HEADERS = {"authorization", "proxy-authorization"}


class _AsyncCloseState:
    def __init__(self) -> None:
        self.event = anyio.Event()
        self.error: BaseException | None = None


def _is_known_encoding(encoding: str) -> bool:
''',
        "async close state",
    )

    replace_exact(
        models,
        '''        self.is_closed = False
        self.is_stream_consumed = False

        self.default_encoding = default_encoding
''',
        '''        self.is_closed = False
        self.is_stream_consumed = False
        self._async_close_started = False
        self._async_close_state: _AsyncCloseState | None = None

        self.default_encoding = default_encoding
''',
        "response close fields",
    )

    replace_exact(
        models,
        '''    def __getstate__(self) -> dict[str, typing.Any]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name not in ["extensions", "stream", "is_closed", "_decoder"]
        }

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.is_closed = True
        self.extensions = {}
        self.stream = UnattachedStream()
''',
        '''    def __getstate__(self) -> dict[str, typing.Any]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if name
            not in [
                "extensions",
                "stream",
                "is_closed",
                "_decoder",
                "_async_close_state",
            ]
        }

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.is_closed = True
        self._async_close_started = True
        self._async_close_state = None
        self.extensions = {}
        self.stream = UnattachedStream()
''',
        "response pickle state",
    )

    replace_exact(
        models,
        '''        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed:
            raise StreamClosed()
        if not isinstance(self.stream, AsyncByteStream):
''',
        '''        if self.is_stream_consumed:
            raise StreamConsumed()
        if self.is_closed or self._async_close_started:
            raise StreamClosed()
        if not isinstance(self.stream, AsyncByteStream):
''',
        "async body-read close barrier",
    )

    replace_exact(
        models,
        '''        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError("Attempted to call an async close on a sync stream.")

        if not self.is_closed:
            self.is_closed = True
            with request_context(request=self._request):
                await self.stream.aclose()
''',
        '''        if not isinstance(self.stream, AsyncByteStream):
            raise RuntimeError("Attempted to call an async close on a sync stream.")

        while not self.is_closed:
            state = self._async_close_state
            if state is None:
                state = _AsyncCloseState()
                self._async_close_state = state
                self._async_close_started = True
                try:
                    with request_context(request=self._request):
                        await self.stream.aclose()
                except BaseException as exc:
                    if isinstance(exc, Exception):
                        state.error = exc
                    self._async_close_state = None
                    state.event.set()
                    raise
                else:
                    self.is_closed = True
                    self._async_close_state = None
                    state.event.set()
                    return

            await state.event.wait()
            if state.error is not None:
                raise state.error
''',
        "Response.aclose ownership",
    )


if __name__ == "__main__":
    main()
