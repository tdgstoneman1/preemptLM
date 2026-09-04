from __future__ import annotations

from typing import Any, Self
from collections.abc import Mapping

from abc import ABC, abstractmethod

from pathlib import Path

import asyncio

import os

import json

import pyarrow as pa
import pyarrow.parquet as pq

from preempt.core.enums import ParquetCompressionCodecs


class BaseTraceSink(ABC):
    """*Abstract, do not instantiate.*

    Destination for recorded trace events. Use as an async context manager.
    """

    @abstractmethod
    async def write(self, event: Mapping[str, Any]) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def flush(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.aclose()


class ParquetTraceSink(BaseTraceSink):
    """Buffers recorded trace events and persists them as a Parquet file."""

    path: Path
    batch_size: int
    row_group_size: int | None

    overwrite: bool
    compression: ParquetCompressionCodecs

    _buffer: list[dict[str, Any]]
    _writer: pq.ParquetWriter | None
    _lock: asyncio.Lock
    _closed: bool
    _records_written: int

    def __init__(
        self,
        path: str | Path,
        *,
        schema: pa.Schema,
        batch_size: int = 1024,
        row_group_size: int | None = None,  # TODO rename
        overwrite: bool = False,
        compression: ParquetCompressionCodecs = ParquetCompressionCodecs.NONE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("`batch_size` must be greater than 0.")

        if row_group_size is not None and row_group_size <= 0:
            raise ValueError("`row_group_size` must be greater than 0.")

        self.path = Path(path)
        self.schema = schema
        self.batch_size = batch_size
        self.row_group_size = row_group_size
        self.overwrite = overwrite
        self.compression = compression

        self._buffer = []
        self._writer = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._records_written = 0

    @property
    def records_written(self) -> int:
        return self._records_written

    async def write(self, event: Mapping[str, pa.Field]) -> None:
        """Buffers an event and writes it to file when buffer reaches or
        exceeds batch size.
        """

        async with self._lock:
            self._raise_if_closed()
            self._buffer.append(dict(event))

            if len(self._buffer) >= self.batch_size:
                await self._write_buffer_locked()

    async def flush(self) -> None:
        """Writes all buffered events to the current Parquet file."""

        async with self._lock:
            self._raise_if_closed()
            await self._write_buffer_locked()

    async def aclose(self) -> None:
        """Writes pending events, then finalizes and closes current Parquet file."""

        async with self._lock:
            if self._closed:
                return
            try:
                await self._write_buffer_locked()

            finally:
                if self._writer is not None:
                    writer, self._writer = self._writer, None
                    await asyncio.to_thread(writer.close)

                self._closed = True

    async def _write_buffer_locked(self) -> None:
        """Writes the current buffer. Caller must hold `self._lock`."""

        if not self._buffer:
            return

        records, self._buffer = self._buffer, []
        try:
            batch = await asyncio.to_thread(
                pa.RecordBatch.from_pylist,
                records,
                self.schema,
            )
            writer = await self._ensure_writer_locked()

            await asyncio.to_thread(
                writer.write_batch,
                batch,
                self.row_group_size,
            )
        except Exception as e:
            self._buffer += records
            raise e

        self._records_written += batch.num_rows

    async def _ensure_writer_locked(self) -> pq.ParquetWriter:
        """Opens writer lazily. Caller must hold `self._lock`."""

        if self._writer is not None:
            return self._writer

        await asyncio.to_thread(
            self.path.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(
                f"File already exists at {self.path.as_posix()!r}. "
                f"Either initialize {self.__class__.__name__!r} with a "
                "different file path or 'overwrite=True'."
            )
        self._writer = await asyncio.to_thread(
            pq.ParquetWriter,
            self.path,
            self.schema,
            compression=self.compression,
        )

        return self._writer

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__!r} is closed.")


class JsonlEventSink(BaseTraceSink):
    path: Path

    _append: bool
    _flush_every_event: bool
    _fsync_every_event: bool
    _file: Any | None
    _lock: asyncio.Lock

    def __init__(
        self,
        path: str | Path,
        *,
        append: bool = False,
        flush_every_event: bool = False,
        fsync_every_event: bool = False,
    ) -> None:
        if fsync_every_event and not flush_every_event:
            raise ValueError(
                "`fsync_every_event=True` requires `flush_every_event=True`"
            )

        self.path = Path(path)
        self._append = append
        self._flush_every_event = flush_every_event
        self._fsync_every_event = fsync_every_event
        self._file = None
        self._lock = asyncio.Lock()

    async def write(self, event: Mapping[str, Any]) -> None:
        """Appends a JSON-serializable event to disk."""

        payload = json.dumps(
            dict(event),
            ensure_ascii=False,
            separators=(",", ":"),
            default=self._json_default,
        )

        async with self._lock:
            f = await self._ensure_open()
            await asyncio.to_thread(f.write, payload + "\n")

            if self._flush_every_event:
                await asyncio.to_thread(f.flush)

            if self._fsync_every_event:
                await asyncio.to_thread(self._fsync, f)

    async def flush(self) -> None:
        async with self._lock:
            if self._file is not None:
                await asyncio.to_thread(self._file.flush)

    async def aclose(self) -> None:
        async with self._lock:
            if self._file is None:
                return

            f, self._file = self._file, None

            await asyncio.to_thread(f.flush)
            await asyncio.to_thread(f.close)

    async def _ensure_open(self) -> Any:
        if self._file is None:
            await asyncio.to_thread(
                self.path.parent.mkdir,
                parents=True,
                exist_ok=True,
            )
            mode = "a" if self._append else "w"
            self._file = await asyncio.to_thread(
                self.path.open,
                mode,
                encoding="utf-8",
                newline="\n",
            )

        return self._file

    @staticmethod
    def _fsync(file: Any) -> None:
        os.fsync(file.fileno())

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            # TODO remove this check and handle serialization elsewhere earlier
            return value.item()

        if isinstance(value, Path):
            return value.as_posix()

        raise TypeError(
            f"Object of type `{type(value).__name__}` is not JSON serializable."
        )
