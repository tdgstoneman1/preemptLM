import pytest

from preempt.expert_bank.encoding import ExpertBankEncoding, parse_payload_encoding_tag


def test_parses_the_real_stores_tag() -> None:
    encoding = parse_payload_encoding_tag("mlx-affine-q4-g64-bf16")
    assert encoding == ExpertBankEncoding(
        family="mlx", mode="affine", bits=4, group_size=64, scalar="bf16"
    )


def test_parses_unquantized_tag() -> None:
    encoding = parse_payload_encoding_tag("mlx-unquantized-f32")
    assert encoding == ExpertBankEncoding(
        family="mlx", mode=None, bits=None, group_size=None, scalar="f32"
    )


@pytest.mark.parametrize(
    "tag",
    [
        "mlx-affine-q4-g64-bf16",
        "mlx-affine-q8-g32-f16",
        "mlx-mxfp4-q4-g32-bf16",
        "mlx-unquantized-bf16",
        "mlx-unquantized-f16",
        "mlx-unquantized-f32",
    ],
)
def test_round_trips_both_tag_forms(tag: str) -> None:
    """Every accepted tag re-renders to itself, so parsing loses nothing."""
    encoding = parse_payload_encoding_tag(tag)
    if encoding.mode is None:
        rendered = f"{encoding.family}-unquantized-{encoding.scalar}"
    else:
        rendered = (
            f"{encoding.family}-{encoding.mode}-q{encoding.bits}"
            f"-g{encoding.group_size}-{encoding.scalar}"
        )
    assert rendered == tag


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "mlx",
        "mlx-affine-q4-g64",  # scalar missing: the pre-fix tag form
        "mlx-affine-4-64-bf16",  # missing the q/g markers
        "mlx-affine-qx-g64-bf16",
        "mlx-affine-q4-g64-bf16-extra",
        "mlx-affine-q0-g64-bf16",
        "mlx-affine-q4-g0-bf16",
        "mlx--q4-g64-bf16",
        "MLX-affine-q4-g64-bf16",
        "mlx-unquantized",
        "mlx-unquantized-bf16-g64",
    ],
)
def test_malformed_tag_raises_naming_the_tag(tag: str) -> None:
    with pytest.raises(ValueError, match="payload encoding"):
        parse_payload_encoding_tag(tag)


@pytest.mark.parametrize("tag", ["torch-affine-q4-g64-bf16", "gguf-unquantized-f32"])
def test_unknown_family_raises(tag: str) -> None:
    with pytest.raises(ValueError, match="family"):
        parse_payload_encoding_tag(tag)


@pytest.mark.parametrize(
    "tag", ["mlx-affine-q4-g64-fp8", "mlx-unquantized-f64", "mlx-affine-q4-g64-int8"]
)
def test_unknown_scalar_raises(tag: str) -> None:
    """An unrecognised scalar must never be guessed at: numpy has no bfloat16,
    so a mis-read scalar silently corrupts every scale in the model."""
    with pytest.raises(ValueError, match="scalar"):
        parse_payload_encoding_tag(tag)


def test_encoding_is_frozen() -> None:
    encoding = parse_payload_encoding_tag("mlx-affine-q4-g64-bf16")
    with pytest.raises(Exception):
        encoding.bits = 8  # type: ignore[misc]
