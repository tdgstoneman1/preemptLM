from __future__ import annotations

import attrs
from attrs import field, validators

from preempt.core.constants import (
    BACKENDS,
    KNOWN_DTYPES,
    QUANTIZED_REGEX,
    UNQUANTIZED_REGEX,
)

_TAG_GRAMMAR: tuple[str, str] = (
    "<backend>-<mode>-q<bits>-g<group_size>-<scalar>",
    "<backend>-unquantized-<scalar>",
)


@attrs.define(kw_only=True, frozen=True)
class ExpertBankEncoding:
    """An expert bank's model encoding information corresponding to the `encoding`
    tag in its manifest, for example `'mlx-affine-q8-g64-bfloat16'`

    Attributes
    ----------
    backend : str
        The backend used to run the model, e.g. 'mlx'. Defines binary layout/
        quantization conventions for handling weights.
    mode : str | None
        Quantization mode (e.g. `affine`), or `None` if unquantized. Since this
        class is backend-agnostic, validation of non-empty strings against a
        list of known modes is left to concrete backends.
    bits : int | None
        Number of bits per quantized weight, or `None` if unquantized
    group_size : int | None
        Number of weights sharing one scale/bias pair, or `None` if unquantized
    dtype : str
        Scalar dtype of `scales`/`biases` (e.g. `bfloat16`) or the weights
        themselves if unquantized
    """

    backend: str = field(validator=validators.min_len(1))
    mode: str | None = field(validator=validators.optional(validators.min_len(1)))
    bits: int | None = field(validator=validators.optional(validators.ge(1)))
    group_size: int | None = field(validator=validators.optional(validators.ge(1)))
    dtype: str = field(validator=validators.min_len(1))

    @property
    def is_quantized(self) -> bool:
        return self.mode is not None

    @classmethod
    def from_tag(cls, tag: str) -> ExpertBankEncoding:
        return parse_encoding_tag(tag)


def parse_encoding_tag(tag: str) -> ExpertBankEncoding:
    """Decodes a encoding tag into its components.

    Parameters
    ----------
    tag : str
        Encoding tag from an expert bank manifest or `SerializedExpert`, e.g.
        `mlx-affine-q4-g64-bfloat16` or `mlx-unquantized-bfloat16`

    Returns
    -------
    ExpertBankEncoding
        The tag's decoded components

    Raises
    ------
    ValueError
        If `tag` does not match a recognized regex pattern
    ValueError
        If `tag` names an unknown backend
    ValueError
        If `tag` names an unknown dtype
    """
    match = UNQUANTIZED_REGEX.match(tag) or QUANTIZED_REGEX.match(tag)
    if match is None:
        raise ValueError(
            f"Malformed encoding tag {tag!r}. Valid formats: {', '.join(_TAG_GRAMMAR)!r}"
        )

    backend = match["backend"]
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown encoding backend {backend!r} in tag {tag!r}; "
            f"known families: {sorted(BACKENDS)!r}."
        )

    scalar = match["dtype"]
    if scalar not in KNOWN_DTYPES:
        raise ValueError(
            f"Unknown encoding scalar {scalar!r} in tag {tag!r}; "
            f"known scalars: {sorted(KNOWN_DTYPES)!r}."
        )

    groups = match.groupdict()
    if "bits" not in groups:
        return ExpertBankEncoding(
            backend=backend, mode=None, bits=None, group_size=None, dtype=scalar
        )

    bits = int(groups["bits"])
    group_size = int(groups["group_size"])
    if bits < 1 or group_size < 1:
        raise ValueError(
            f"Malformed encoding tag {tag!r}: bits and group size must "
            f"be positive, but got {bits=!r} and {group_size=!r}."
        )

    return ExpertBankEncoding(
        backend=backend,
        mode=groups["mode"],
        bits=bits,
        group_size=group_size,
        dtype=scalar,
    )
