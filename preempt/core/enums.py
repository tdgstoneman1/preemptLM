from enum import StrEnum, IntEnum


class Backends(StrEnum):
    MLX = "mlx_metal"


class ParquetCompressionCodecs(StrEnum):
    """Parquet column compression codecs for event sinks"""

    NONE = "NONE"
    SNAPPY = "SNAPPY"
    GZIP = "GZIP"
    BROTLI = "BROTLI"
    LZ4 = "LZ4"
    LZ4_RAW = "LZ4_RAW"
    ZSTD = "ZSTD"


class ReadPriority(IntEnum):
    """Defines the urgency of an expert bank read. Lower values indicate higher
    urgency. Possible values:
    - `DEMAND` reads block the forward pass.
    - `PREFETCH` reads are speculative and non-blocking.
    """

    DEMAND = 0
    PREFETCH = 1


class CacheEvictionPolicy(StrEnum):
    """Determines the eviction strategy for experts cached in memory.

    - `LFRU`: Least Frequently Recently Used (`ExpertCacheManager` default)
    - `LRU` Least Recently Used
    """

    LFRU = "lfru"
    LRU = "lru"


class CacheSlotState(IntEnum):
    EMPTY = -1
    INFLIGHT = 0
    READY = 1
    # FAILED = "FAILED"
