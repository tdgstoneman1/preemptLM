from __future__ import annotations

import pytest

import inspect

from preempt.backends.mlx_metal.expert_bank.base_adapter import BaseMoEArchAdapter
from preempt.backends.mlx_metal.expert_bank.qwen3_x import Qwen3_xArchAdapter
from preempt.backends.mlx_metal.expert_bank.serialization import model_to_expert_bank

from preempt.backends.mlx_metal.expert_kernel import sequential_expert_matmul
from preempt.backends.mlx_metal.quantization import QuantSettings


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BaseMoEArchAdapter()  # type: ignore[abstract]


def test_partial_subclass_cannot_instantiate() -> None:
    class Partial(BaseMoEArchAdapter):
        @property
        def projection_names(self) -> tuple[str, ...]:
            return ("a",)

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


@pytest.fixture
def arch() -> Qwen3_xArchAdapter:
    return Qwen3_xArchAdapter()


class TestQwenArchitectureProperties:
    def test_projection_names(self, arch: Qwen3_xArchAdapter) -> None:
        assert arch.projection_names == ("gate_proj", "up_proj", "down_proj")

    def test_quantized_tensor_parts(self, arch: Qwen3_xArchAdapter) -> None:
        assert arch.quantized_tensor_parts == ("weight", "scales", "biases")

    def test_layer_class_name(self, arch: Qwen3_xArchAdapter) -> None:
        assert arch.layer_class_name == "Qwen3NextSparseMoeBlock"

    def test_expert_module_pattern_contains_groups(
        self, arch: Qwen3_xArchAdapter
    ) -> None:
        assert "(?P<prefix>" in arch.expert_module_pattern
        assert "(?P<layer>" in arch.expert_module_pattern
        assert "(?P<projection>" in arch.expert_module_pattern


class TestQwenRegexes:
    def test_tensor_regex_matches_qwen_name(self, arch: Qwen3_xArchAdapter) -> None:
        name = "language_model.model.layers.5.mlp.switch_mlp.gate_proj.weight"
        m = arch.expert_tensor_regex.match(name)

        assert m is not None
        assert m.group("layer") == "5"
        assert m.group("projection") == "gate_proj"
        assert m.group("part") == "weight"

    def test_tensor_regex_rejects_non_expert(self, arch: Qwen3_xArchAdapter) -> None:
        m = arch.expert_tensor_regex.match("model.layers.5.self_attn.q_proj.weight")
        assert m is None

    def test_module_regex_matches_without_part(self, arch: Qwen3_xArchAdapter) -> None:
        name = "model.layers.3.mlp.switch_mlp.up_proj"
        m = arch.expert_module_regex.match(name)

        assert m is not None
        assert m.group("layer") == "3"
        assert m.group("projection") == "up_proj"


class TestQwenTensorOrder:
    def test_full_order(self, arch: Qwen3_xArchAdapter) -> None:
        order = arch.tensor_order

        assert order[0] == "gate_proj.weight"
        assert order[1] == "gate_proj.scales"
        assert order[2] == "gate_proj.biases"
        assert len(order) == 9  # 3 projections x 3 parts

    def test_validate_layer_tensors_full(self, arch: Qwen3_xArchAdapter) -> None:
        layer_tensors = {
            f"{p}.{part}": f"full.{p}.{part}"
            for p in arch.projection_names
            for part in arch.quantized_tensor_parts
        }
        result = arch.validate_weight_tensor_paths(layer_tensors)

        assert result == arch.tensor_order

    def test_validate_layer_tensors_missing_weight_raises(
        self, arch: Qwen3_xArchAdapter
    ) -> None:
        layer_tensors = {"gate_proj.scales": "x", "gate_proj.biases": "x"}
        with pytest.raises(
            ValueError,
            match="The following weight tensors are missing from `layer_tensors`",
        ):
            arch.validate_weight_tensor_paths(layer_tensors)

    def test_validate_layer_tensors_unquantized(self, arch: Qwen3_xArchAdapter) -> None:
        layer_tensors = {
            "gate_proj.weight": "x",
            "up_proj.weight": "x",
            "down_proj.weight": "x",
        }
        result = arch.validate_weight_tensor_paths(layer_tensors)

        assert result == ("gate_proj.weight", "up_proj.weight", "down_proj.weight")

    def test_validate_layer_tensors_unexpected_raises(
        self, arch: Qwen3_xArchAdapter
    ) -> None:
        layer_tensors = {
            "gate_proj.weight": "x",
            "up_proj.weight": "x",
            "down_proj.weight": "x",
            "unknown_proj.weight": "x",
        }
        with pytest.raises(
            ValueError,
            match="`layer_tensors` contains the following unexpected weight tensors",
        ):
            arch.validate_weight_tensor_paths(layer_tensors)


class TestQwenQuantization:
    def test_no_quantization_section(self, arch: Qwen3_xArchAdapter) -> None:
        result = arch.resolve_quantization({}, (0, 1))
        assert result is None

    def test_uniform_default(self, arch: Qwen3_xArchAdapter) -> None:
        config = {"quantization": {"mode": "affine", "bits": 4, "group_size": 64}}
        result = arch.resolve_quantization(config, (0, 1))
        assert result == QuantSettings(mode="affine", bits=4, group_size=64)  # type: ignore

    def test_per_module_override_mixed_raises(self, arch: Qwen3_xArchAdapter) -> None:
        config = {
            "quantization": {
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
                "model.layers.0.mlp.switch_mlp.gate_proj": {
                    "bits": 8,
                    "group_size": 32,
                },
            }
        }
        with pytest.raises(ValueError, match="not uniformly quantized"):
            arch.resolve_quantization(config, (0, 1))

    def test_all_overrides_uniform(self, arch: Qwen3_xArchAdapter) -> None:
        config = {
            "quantization": {
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
                **{
                    f"model.layers.{l}.mlp.switch_mlp.{p}": {
                        "bits": 8,
                        "group_size": 32,
                    }
                    for l in (0, 1)
                    for p in ("gate_proj", "up_proj", "down_proj")
                },
            }
        }
        result = arch.resolve_quantization(config, (0, 1))
        assert result == QuantSettings(mode="affine", bits=8, group_size=32)  # type: ignore

    def test_text_config_quantization(self, arch: Qwen3_xArchAdapter) -> None:
        config = {
            "text_config": {
                "quantization": {"mode": "affine", "bits": 4, "group_size": 64}
            }
        }
        result = arch.resolve_quantization(config, (0,))

        assert result == QuantSettings(mode="affine", bits=4, group_size=64)  # type: ignore


class TestQwenMoESpec:
    def test_extract(self, arch: Qwen3_xArchAdapter) -> None:
        config = {"text_config": {"num_routed_experts": 256, "num_experts_per_tok": 8}}
        topo = arch.extract_model_moe_spec(config, (0, 1, 2), num_routed_experts=256)

        assert topo.moe_block_idxs == (0, 1, 2)
        assert topo.num_routed_experts == 256
        assert topo.top_k == 8

    def test_extract_non_text_config(self, arch: Qwen3_xArchAdapter) -> None:
        config = {"num_routed_experts": 128, "num_experts_per_tok": 4}
        topo = arch.extract_model_moe_spec(config, (5,), num_routed_experts=128)

        assert topo.moe_block_idxs == (5,)
        assert topo.num_routed_experts == 128
        assert topo.top_k == 4


def test_convert_accepts_architecture_parameter() -> None:  # TODO remove, redundant
    sig = inspect.signature(model_to_expert_bank)

    assert "architecture" in sig.parameters


def test_apply_experts_accepts_apply_expert_fn() -> None:
    sig = inspect.signature(sequential_expert_matmul)

    assert "expert_forward_fn" in sig.parameters
    assert "activation" not in sig.parameters
