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

# Expert bank

MANIFEST_FILENAME: Final[str] = "manifest.json"
EXPERTS_FILENAME: Final[str] = "experts.bin"
EXPERT_BANK_SCHEMA: Final[int] = 1  # TODO rename to 'EXPERT_BANK_SCHEMA_VERSION'?

# TODO rewrite this comment slop
# macOS `<sys/fcntl.h>` value of `F_NOCACHE`; absent from Python's `fcntl`
# module, so it is spelled out here. Turns off page caching for reads/writes
# on the fd, so blobs come off the SSD rather than the kernel's file cache.
F_NOCACHE = 48

# Tracing

RUN_ID_TIMESTAMP_FMT = "%Y%m%dT%H%M%SZ"


# Defaults

DEFAULT_MAX_SAFETENSOR_SHARD_MB: Final[int] = 9 * 1024  # 9 GB * 1024
