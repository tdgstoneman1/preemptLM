from typing import Final

import re

# Payload encoding

TAG_GRAMMAR: Final[str] = (
    "'<family>-<mode>-q<bits>-g<group_size>-<scalar>' or "
    "'<family>-unquantized-<scalar>'"
)  # TODO make this a string template

KNOWN_FAMILIES: Final[frozenset[str]] = frozenset({"mlx"})  # TODO rename
KNOWN_SCALARS: Final[frozenset[str]] = frozenset(
    {"bf16", "f16", "f32"}
)  # TODO add other dtypes, e.g. int8; rename ('scalars' too ambiguous)

QUANTIZED_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>[a-z0-9]+)-(?P<mode>[a-z0-9_]+)-q(?P<bits>\d+)"
    r"-g(?P<group_size>\d+)-(?P<scalar>[a-z0-9]+)$"
)
UNQUANTIZED_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>[a-z0-9]+)-unquantized-(?P<scalar>[a-z0-9]+)$"
)

# Storage

MANIFEST_FILENAME: Final[str] = "manifest.json"
EXPERTS_FILENAME: Final[str] = "experts.bin"
EXPERT_BANK_SCHEMA: Final[int] = 1  # TODO rename to 'EXPERT_BANK_SCHEMA_VERSION'?

F_NOCACHE = 48  # TODO rename

# Tracing

RUN_ID_TIMESTAMP_FMT = "%Y%m%dT%H%M%SZ"


# Defaults

DEFAULT_MAX_SAFETENSOR_SHARD_MB: Final[int] = 9 * 1024  # 9 GB * 1024
