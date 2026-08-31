from typing import Final

from ..types import MlxMoEFactory

from .qwen3_x import Qwen3_xArchAdapter

V1_MOE_FACTORIES: Final[dict[str, MlxMoEFactory]] = {
    "qwen3-next": Qwen3_xArchAdapter,
    "qwen3_x": Qwen3_xArchAdapter,
}
