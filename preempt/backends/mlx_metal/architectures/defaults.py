from typing import Final

from ..types import MlxMoEFactory

from .qwen3_next import Qwen3NextMoEArchitecture

# TODO add version
# TODO rename "qwen" key to "qwen3-next"

DEFAULT_MOE_FACTORIES: Final[dict[str, MlxMoEFactory]] = {
    "qwen": Qwen3NextMoEArchitecture
}
