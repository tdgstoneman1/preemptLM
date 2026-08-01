import torch

from .qwen3_model import Qwen3Model
from .qwen3_config import Qwen3Config

from .qwen3_block import GQATransformerBlock

from preempt.utils.torch_utils import assign_weights

QWEN3_PARAM_NAME_MAP = (
    ("transformer_blocks", "layers"),
    ("rms_norm1.scale", "input_layernorm.weight"),
    ("rms_norm2.scale", "post_attention_layernorm.weight"),
    ("final_norm.scale", "norm.weight"),
    ("self_attn.out_proj", "self_attn.o_proj"),
    ("token_embedding", "embed_tokens"),
)


def qwen3_transfer_hf_weights(
    model: Qwen3Model, config: Qwen3Config, state_dict: dict[str, torch.Tensor]
) -> None:

    assign_weights(
        model.token_embedding.weight,
        state_dict["model.embed_tokens.weight"],
        # "model.embed_tokens.weight",
    )

    for l in range(config.num_transformer_blocks):
        block: GQATransformerBlock = model.transformer_blocks[l]  # type: ignore

        # QKV linear projections
        assign_weights(
            block.self_attn.q_proj.weight,
            state_dict[f"model.layers.{l}.self_attn.q_proj.weight"],
            # f"model.layers.{l}.self_attn.q_proj.weight",
        )
        assign_weights(
            block.self_attn.k_proj.weight,
            state_dict[f"model.layers.{l}.self_attn.k_proj.weight"],
            # f"model.layers.{l}.self_attn.k_proj.weight",
        )
        assign_weights(
            block.self_attn.v_proj.weight,
            state_dict[f"model.layers.{l}.self_attn.v_proj.weight"],
            # f"model.layers.{l}.self_attn.v_proj.weight",
        )

        # Output projection
        assign_weights(
            block.self_attn.out_proj.weight,
            state_dict[f"model.layers.{l}.self_attn.o_proj.weight"],
            # f"model.layers.{l}.self_attn.o_proj.weight",
        )

        # Q and K norm
        if hasattr(block.self_attn, "q_norm") and block.self_attn.q_norm is not None:
            assign_weights(
                block.self_attn.q_norm.scale,
                state_dict[f"model.layers.{l}.self_attn.q_norm.weight"],
                # f"model.layers.{l}.self_attn.q_norm.weight",
            )
        if hasattr(block.self_attn, "k_norm") and block.self_attn.k_norm is not None:
            assign_weights(
                block.self_attn.k_norm.scale,
                state_dict[f"model.layers.{l}.self_attn.k_norm.weight"],
                # f"model.layers.{l}.self_attn.k_norm.weight",
            )

        # Attention layer norm
        assign_weights(
            block.rms_norm1.scale,
            state_dict[f"model.layers.{l}.input_layernorm.weight"],
            # f"model.layers.{l}.input_layernorm.weight",
        )

        # MLP weights
        assign_weights(
            block.mlp.gate_proj.weight,
            state_dict[f"model.layers.{l}.mlp.gate_proj.weight"],
            # f"model.layers.{l}.mlp.gate_proj.weight",
        )
        assign_weights(
            block.mlp.up_proj.weight,
            state_dict[f"model.layers.{l}.mlp.up_proj.weight"],
            # f"model.layers.{l}.mlp.up_proj.weight",
        )
        assign_weights(
            block.mlp.down_proj.weight,
            state_dict[f"model.layers.{l}.mlp.down_proj.weight"],
            # f"model.layers.{l}.mlp.down_proj.weight",
        )
        assign_weights(
            block.rms_norm2.scale,
            state_dict[f"model.layers.{l}.post_attention_layernorm.weight"],
            # f"model.layers.{l}.post_attention_layernorm.weight",
        )

    # Final norm and output head
    assign_weights(
        model.final_norm.scale,
        state_dict["model.norm.weight"],  # "model.norm.weight"
    )
    if "lm_head.weight" in state_dict:
        assign_weights(
            model.out_head.weight,
            state_dict["lm_head.weight"],  # "lm_head.weight"
        )
    else:
        model.out_head.weight = model.token_embedding.weight
        print("Model uses weight tying.")  # TODO logging
