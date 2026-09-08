"""QLoRA loading + state contracts for the pgat-length Qwen decoder path.

Ported from pg-adaptor/src/pg_adaptor/training/pgat_translation_runtime.py
and trimmed to only what pgat-length needs. The recipe (nf4, bf16 compute,
double-quantization, LoRA on q/k/v/o at rank 16 / alpha 32) matches PGAT-v1
so the decoder swap is comparable at the visual-encoder level.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


def load_qwen_qlora_decoder(config: Mapping[str, Any], hf_cache_dir: str | None = None):
    """Load Qwen with 4-bit quantization and attach LoRA adapters.

    Requires `peft` and `bitsandbytes`. Returns (decoder, compute_dtype).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen QLoRA")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_name = str(config.get("compute_dtype", "bfloat16")).lower()
    if compute_name not in {"bfloat16", "bf16", "float16", "fp16"}:
        raise ValueError(f"Unsupported QLoRA compute dtype: {compute_name}")
    compute_dtype = (
        torch.bfloat16 if compute_name in {"bfloat16", "bf16"} else torch.float16
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(config.get("quantization", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(config.get("double_quantization", True)),
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        str(config["decoder_model"]),
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        cache_dir=hf_cache_dir,
    )
    decoder.config.use_cache = False
    decoder = prepare_model_for_kbit_training(
        decoder,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=list(config.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    decoder = get_peft_model(decoder, lora)
    return decoder, compute_dtype


def get_lora_state_dict(decoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(decoder)
    if not state:
        raise RuntimeError("LoRA state dictionary is empty")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def load_lora_state_dict(decoder: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    expected = set(get_peft_model_state_dict(decoder))
    received = set(state)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    if missing or unexpected:
        raise RuntimeError(f"LoRA key mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")
    incompatible = set_peft_model_state_dict(decoder, dict(state))
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA load keys: {incompatible.unexpected_keys[:5]}")


def cosine_lr_lambda(total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier
