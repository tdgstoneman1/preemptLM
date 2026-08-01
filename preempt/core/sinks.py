from __future__ import annotations

from typing import Any, Self
from collections.abc import Mapping

from abc import ABC, abstractmethod

import asyncio
import os

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from preempt.core.enums import ParquetCompressionCodecs


class BaseEventSink(ABC):
    """**Abstract, do not instantiate.**

    Base class for event sinks that write hidden/internal model states.
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


class ParquetEventSink(BaseEventSink):
    """Buffer capture events and persist them as a Parquet file.

    The sink owns one Parquet file for its lifetime. All accepted events must
    conform to the supplied Arrow schema. Events are written in batches to
    avoid creating a row group for every capture.

    This is appropriate for offline trace collection and training-dataset
    creation. It is not intended as the final inference-time tracing path.
    """

    _path: Path
    _schema: pa.Schema
    _batch_size: int
    _row_group_size: int | None
    _overwrite: bool
    _compression: ParquetCompressionCodecs

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
        row_group_size: int | None = None,
        overwrite: bool = False,
        compression: ParquetCompressionCodecs = ParquetCompressionCodecs.NONE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if row_group_size is not None and row_group_size <= 0:
            raise ValueError("row_group_size must be greater than zero")

        self._path = Path(path)
        self._schema = schema
        self._batch_size = batch_size
        self._row_group_size = row_group_size
        self._overwrite = overwrite
        self._compression = compression
        self._buffer = []
        self._writer: pq.ParquetWriter | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._records_written = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def records_written(self) -> int:
        return self._records_written

    async def write(self, event: Mapping[str, pa.Field]) -> None:
        """Buffer one event; write when the batch threshold is reached."""

        async with self._lock:
            self._raise_if_closed()
            self._buffer.append(dict(event))

            if len(self._buffer) >= self._batch_size:
                await self._write_buffer_locked()

    async def flush(self) -> None:
        """Write all buffered events to the current Parquet file."""

        async with self._lock:
            self._raise_if_closed()
            await self._write_buffer_locked()

    async def aclose(self) -> None:
        """Write pending events, finalize the Parquet file, and close it."""

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
        """Writes the current buffer, caller must hold `self._lock`."""

        if not self._buffer:
            return

        records, self._buffer = self._buffer, []
        try:
            batch = await asyncio.to_thread(
                pa.RecordBatch.from_pylist,
                records,
                self._schema,
            )
            writer = await self._ensure_writer_locked()

            await asyncio.to_thread(
                writer.write_batch,
                batch,
                self._row_group_size,
            )
        except Exception as e:
            self._buffer = records + self._buffer
            raise e

        self._records_written += batch.num_rows

    async def _ensure_writer_locked(self) -> pq.ParquetWriter:
        """Lazily opens writer, caller must hold `self._lock`."""

        if self._writer is not None:
            return self._writer

        await asyncio.to_thread(
            self._path.parent.mkdir,
            parents=True,
            exist_ok=True,
        )

        if self._path.exists() and not self._overwrite:
            raise FileExistsError(
                f"File already exists at `{self._path.as_posix()}`. "
                f"Initialize `{self.__class__.__name__}` with a "
                "different file path or `overwrite=True`."
            )

        self._writer = await asyncio.to_thread(
            pq.ParquetWriter,
            self._path,
            self._schema,
            compression=self._compression,
        )
        return self._writer

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"`{self.__class__.__name__}` instance is closed.")


class JsonlFileEventSink(BaseEventSink):
    """Append capture events to a UTF-8 JSON Lines file.

    This is intentionally a dataset-collection implementation, not the
    eventual inference-time tracing path. Each call appends exactly one JSON
    object followed by a newline.

    The sink serializes writes with an asyncio lock so multiple instrumented
    wrappers cannot interleave bytes in the output file.
    """

    _path: Path
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

        self._path = Path(path)
        self._append = append
        self._flush_every_event = flush_every_event
        self._fsync_every_event = fsync_every_event

        self._file = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def write(self, event: Mapping[str, Any]) -> None:
        """Appends a JSON-serializable event to disk."""

        payload = json.dumps(
            dict(event),
            ensure_ascii=False,
            separators=(",", ":"),
            default=self._json_default,
        )

        async with self._lock:
            file = await self._ensure_open()
            await asyncio.to_thread(file.write, payload + "\n")

            if self._flush_every_event:
                await asyncio.to_thread(file.flush)

            if self._fsync_every_event:
                await asyncio.to_thread(self._fsync, file)

    async def flush(self) -> None:
        async with self._lock:
            if self._file is not None:
                await asyncio.to_thread(self._file.flush)

    async def aclose(self) -> None:
        async with self._lock:
            if self._file is None:
                return

            file, self._file = self._file, None

            await asyncio.to_thread(file.flush)
            await asyncio.to_thread(file.close)

    async def _ensure_open(self) -> Any:
        if self._file is None:
            await asyncio.to_thread(
                self._path.parent.mkdir,
                parents=True,
                exist_ok=True,
            )

            mode = "a" if self._append else "w"
            self._file = await asyncio.to_thread(
                self._path.open,
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
        """Convert common scalar-like values; reject unknown objects loudly."""
        if hasattr(value, "item"):
            # TODO remove this check and handle serialization elsewhere earlier
            return value.item()

        if isinstance(value, Path):
            return value.as_posix()

        raise TypeError(
            f"Object of type `{type(value).__name__}` is not JSON serializable."
        )
