from typing import Final
import re
from string import Template

# Versioning
EXPERT_BANK_SCHEMA_VERSION: Final[int] = 1
EXPERT_SELECTION_TRACE_SCHEMA_VERSION: Final[int] = 1

# Global defaults
DEFAULT_MAX_SAFETENSOR_SHARD_MB: Final[int] = 9 * 1024  # 9 GB

# Tracing
TIMESTAMP_FMT = "%Y%m%dT%H%M%SZ"
RUN_ID_TEMPLATE: Final[Template] = Template("${prefix}-${timestamp}-${hex}")

# Expert bank
MANIFEST_FILENAME: Final[str] = "manifest.json"
EXPERTS_FILENAME: Final[str] = "experts.bin"

# Darwin F_NOCACHE (<sys/fcntl.h>) to bypass OS
# page cache since it's omitted from Python's `fcntl`
F_NOCACHE = 48

# Encoding
BACKENDS: Final[frozenset[str]] = frozenset({"mlx"})
KNOWN_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "uint8",
        "uint16",
        "uint32",
        "bfloat16",
        "float16",
        "float32",
    }
)
QUANTIZED_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(?P<backend>[a-z0-9]+)-(?P<mode>[a-z0-9_]+)-q(?P<bits>\d+)"
    r"-g(?P<group_size>\d+)-(?P<dtype>[a-z0-9]+)$"
)
UNQUANTIZED_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(?P<backend>[a-z0-9]+)-unquantized-(?P<dtype>[a-z0-9]+)$"
)
