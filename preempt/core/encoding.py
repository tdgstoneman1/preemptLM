from __future__ import annotations

import attrs
from attrs import field, validators

from .constants import (
    TAG_GRAMMAR,
    KNOWN_FAMILIES,
    KNOWN_SCALARS,
    QUANTIZED_RE,
    UNQUANTIZED_RE,
)

# TODO move to datamodel/


# TODO add support for more scalar dtypes e.g. int8
@attrs.define(kw_only=True, frozen=True)
class PayloadEncoding:
    """An expert bank's payload encoding information corresponding to its
    `payload_encoding` tag

    Attributes
    ----------
    family : str
        The backend that produced the encoding (e.g. 'mlx') and defines the
        binary layout/quantization conventions for expert weights.
    mode : str | None
        Quantization mode (e.g. `affine`), or `None` if unquantized. Since this
        class is backend-agnostic, validation of non-empty strings against a
        list of known modes is left to concrete backends.
    bits : int | None
        Number of bits per quantized weight, or `None` if unquantized
    group_size : int | None
        Number of weights sharing one scale/bias pair, or `None` if unquantized
    scalar : str
        Scalar dtype of `scales`/`biases` (`bf16`, `f16`, or `f32`) or the weights
        themselves if unquantized
    """

    family: str = field(validator=validators.min_len(1))  # TODO rename to `backend`
    mode: str | None = field(validator=validators.optional(validators.min_len(1)))
    bits: int | None = field(validator=validators.optional(validators.ge(1)))
    group_size: int | None = field(validator=validators.optional(validators.ge(1)))
    scalar: str = field(validator=validators.min_len(1))  # TODO rename

    @property
    def is_quantized(self) -> bool:
        return self.mode is not None

    @classmethod
    def from_tag(cls, tag: str) -> PayloadEncoding:
        return parse_payload_encoding_tag(tag)


def parse_payload_encoding_tag(tag: str) -> PayloadEncoding:
    """Decodes a payload encoding tag into its components.

    Parameters
    ----------
    tag : str
        Encoding tag from an expert bank manifest or `ExpertPayload`, e.g.
        `mlx-affine-q4-g64-bf16` or `mlx-unquantized-bf16`

    Returns
    -------
    PayloadEncoding
        The tag's decoded components

    Raises
    ------
    ValueError
        If `tag` does not match a recognized regex pattern
    ValueError
        If `tag` names an unknown family
    ValueError
        If `tag` names an unknown scalar dtype
    """
    match = UNQUANTIZED_RE.match(tag) or QUANTIZED_RE.match(tag)

    if match is None:
        raise ValueError(
            f"Malformed payload encoding tag '{tag!r}'; expected '{TAG_GRAMMAR}'."
        )

    family = match["family"]
    if family not in KNOWN_FAMILIES:
        raise ValueError(
            f"Unknown payload encoding family '{family!r}' in tag '{tag!r}'; "
            f"known families: {sorted(KNOWN_FAMILIES)}."
        )

    scalar = match["scalar"]
    if scalar not in KNOWN_SCALARS:
        raise ValueError(
            f"Unknown payload encoding scalar '{scalar!r}' in tag '{tag!r}'; "
            f"known scalars: {sorted(KNOWN_SCALARS)}."
        )

    groups = match.groupdict()
    if "bits" not in groups:
        return PayloadEncoding(
            family=family, mode=None, bits=None, group_size=None, scalar=scalar
        )

    bits = int(groups["bits"])
    group_size = int(groups["group_size"])
    if bits < 1 or group_size < 1:
        raise ValueError(
            f"Malformed payload encoding tag '{tag!r}': bits and group size must "
            f"be positive, but got `{bits=}` and `{group_size=}`."
        )

    return PayloadEncoding(
        family=family,
        mode=groups["mode"],
        bits=bits,
        group_size=group_size,
        scalar=scalar,
    )
