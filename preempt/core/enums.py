from enum import StrEnum, IntEnum


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
    """Defines the urgency of an expert bank read (from disk). Lower values
    indicate higher urgency.

    `DEMAND` reads block the forward pass. `PREFETCH` reads are speculative
    and non-blocking.
    """

    DEMAND = 0
    PREFETCH = 1


class CachePolicy(StrEnum):
    """Determines the eviction strategy for in-memory experts. Eviction occurs
    when a requested expert is not found in memory and there is insufficient
    memory available to load it from disk.

    `LFRU` (`ExpertCache` default) ranks by frequency, then recency. This helps
    frequently accessed experts resist eviction during temporary bursts of less
    frequent accesses, whereas plain `LRU` would evict them.

    `LRU` stays selectable as a baseline for comparison.
    """

    LFRU = "lfru"
    LRU = "lru"
