import pytest

from preempt.expert_bank.encoding import ExpertBankEncoding, parse_encoding_tag


def test_parses_the_real_stores_tag() -> None:
    encoding = parse_encoding_tag("mlx-affine-q4-g64-bfloat16")
    assert encoding == ExpertBankEncoding(
        backend="mlx", mode="affine", bits=4, group_size=64, dtype="bfloat16"
    )


def test_parses_unquantized_tag() -> None:
    encoding = parse_encoding_tag("mlx-unquantized-float32")
    assert encoding == ExpertBankEncoding(
        backend="mlx", mode=None, bits=None, group_size=None, dtype="float32"
    )


@pytest.mark.parametrize(
    "tag",
    [
        "mlx-affine-q4-g64-bfloat16",
        "mlx-affine-q8-g32-float16",
        "mlx-mxfp4-q4-g32-bfloat16",
        "mlx-unquantized-bfloat16",
        "mlx-unquantized-float16",
        "mlx-unquantized-float32",
    ],
)
def test_round_trips_both_tag_forms(tag: str) -> None:
    """Every accepted tag re-renders to itself, so parsing loses nothing."""
    encoding = parse_encoding_tag(tag)
    if encoding.mode is None:
        rendered = f"{encoding.backend}-unquantized-{encoding.dtype}"
    else:
        rendered = (
            f"{encoding.backend}-{encoding.mode}-q{encoding.bits}"
            f"-g{encoding.group_size}-{encoding.dtype}"
        )
    assert rendered == tag


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "mlx",
        "mlx-affine-q4-g64",  # scalar missing: the pre-fix tag form
        "mlx-affine-4-64-bfloat16",  # missing the q/g markers
        "mlx-affine-qx-g64-bfloat16",
        "mlx-affine-q4-g64-bfloat16-extra",
        "mlx-affine-q0-g64-bfloat16",
        "mlx-affine-q4-g0-bfloat16",
        "mlx--q4-g64-bfloat16",
        "MLX-affine-q4-g64-bfloat16",
        "mlx-unquantized",
        "mlx-unquantized-bfloat16-g64",
    ],
)
def test_malformed_tag_raises_naming_the_tag(tag: str) -> None:
    with pytest.raises(ValueError, match="encoding"):
        parse_encoding_tag(tag)


@pytest.mark.parametrize(
    "tag", ["torch-affine-q4-g64-bfloat16", "gguf-unquantized-float32"]
)
def test_unknown_family_raises(tag: str) -> None:
    with pytest.raises(ValueError, match="backend"):
        parse_encoding_tag(tag)


@pytest.mark.parametrize(
    "tag", ["mlx-affine-q4-g64-fp8", "mlx-unquantized-f64", "mlx-affine-q4-g64-int12"]
)
def test_unknown_dtype_raises(tag: str) -> None:
    with pytest.raises(ValueError):
        parse_encoding_tag(tag)


def test_encoding_is_frozen() -> None:
    encoding = parse_encoding_tag("mlx-affine-q4-g64-bfloat16")
    with pytest.raises(Exception):
        encoding.bits = 8  # type: ignore[misc]
