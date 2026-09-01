from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from preempt.engine.sinks import ParquetEventSink

SCHEMA = pa.schema([pa.field("a", pa.int64(), nullable=False)])


async def test_batching_creates_one_row_group_per_full_batch(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    async with ParquetEventSink(path, schema=SCHEMA, batch_size=2) as sink:
        for i in range(5):
            await sink.write({"a": i})
    parquet_file = pq.ParquetFile(path)
    assert parquet_file.metadata.num_rows == 5
    # 2 full batches + 1 partial written at aclose()
    assert parquet_file.metadata.num_row_groups == 3
    assert sink.records_written == 5


async def test_refuses_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    path.touch()
    sink = ParquetEventSink(path, schema=SCHEMA, batch_size=1)
    with pytest.raises(FileExistsError):
        await sink.write({"a": 1})


async def test_write_after_close_raises(tmp_path: Path) -> None:
    sink = ParquetEventSink(tmp_path / "out.parquet", schema=SCHEMA)
    await sink.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await sink.write({"a": 1})


def test_rejects_nonpositive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ParquetEventSink(tmp_path / "x.parquet", schema=SCHEMA, batch_size=0)
