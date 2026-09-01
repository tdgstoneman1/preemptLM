from enum import StrEnum


class MlxQuantMode(StrEnum):
    AFFINE = "affine"
    MXFP4 = "mxfp4"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"
