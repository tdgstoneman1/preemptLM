from enum import StrEnum


class ParquetCompressionCodecs(StrEnum):
    NONE = "NONE"
    SNAPPY = "SNAPPY"
    GZIP = "GZIP"
    BROTLI = "BROTLI"
    LZ4 = "LZ4"
    LZ4_RAW = "LZ4_RAW"
    ZSTD = "ZSTD"
