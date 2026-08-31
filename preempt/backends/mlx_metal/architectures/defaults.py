from typing import Final

from ..types import MlxMoEFactory

from .qwen3_next import Qwen3_XArchAdapter

V1_MOE_FACTORIES: Final[dict[str, MlxMoEFactory]] = {"qwen3-next": Qwen3_XArchAdapter}
