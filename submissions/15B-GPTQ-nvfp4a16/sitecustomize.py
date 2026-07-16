import os
import sys


def patch_llama_config_validation_for_decoupled_head_dim():
    """
    Allow LLaMA configs with explicit head_dim where:

        hidden_size != num_attention_heads * head_dim

    Example:
        hidden_size=4096
        num_attention_heads=48
        head_dim=128
        attention width = 48 * 128 = 6144

    This only relaxes the obsolete hidden_size % num_attention_heads check.
    It keeps the important GQA divisibility check.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig

    def validate_architecture_allow_decoupled_head_dim(self):
        head_dim = getattr(self, "head_dim", None)

        if head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    "Invalid LLaMA config without explicit head_dim: "
                    f"hidden_size={self.hidden_size}, "
                    f"num_attention_heads={self.num_attention_heads}"
                )
        else:
            if head_dim <= 0:
                raise ValueError(f"head_dim must be positive, got {head_dim}")

        num_kv_heads = getattr(self, "num_key_value_heads", None)
        if num_kv_heads is not None:
            if self.num_attention_heads % num_kv_heads != 0:
                raise ValueError(
                    "num_attention_heads must be divisible by num_key_value_heads: "
                    f"num_attention_heads={self.num_attention_heads}, "
                    f"num_key_value_heads={num_kv_heads}"
                )

    LlamaConfig.validate_architecture = validate_architecture_allow_decoupled_head_dim

    # Hugging Face Hub strict dataclasses cache validate_* methods in
    # __class_validators__, so replacing the method alone is not enough.
    if hasattr(LlamaConfig, "__class_validators__"):
        LlamaConfig.__class_validators__ = [
            validator
            for validator in LlamaConfig.__class_validators__
            if getattr(validator, "__name__", "") != "validate_architecture"
        ]
        LlamaConfig.__class_validators__.append(
            validate_architecture_allow_decoupled_head_dim
        )


if os.environ.get("ALLOW_LLAMA_DECOUPLED_HEAD_DIM") == "1":
    try:
        patch_llama_config_validation_for_decoupled_head_dim()
        print(
            "[sitecustomize] Patched Transformers LlamaConfig validation "
            "for explicit decoupled head_dim.",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[sitecustomize] Failed to patch LlamaConfig validation: {exc!r}",
            file=sys.stderr,
        )
        raise
