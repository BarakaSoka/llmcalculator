"""The four things people want to do with a model, and what each one costs.

Memory cost per parameter differs by more than an order of magnitude across
these: inference at 4-bit needs ~0.6 bytes/param, full fine-tuning with Adam
needs ~16. That gap is why a machine that runs a 32B model comfortably can
only fully fine-tune a 1B one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    description: str

    grad_bytes: float = 0.0
    """Bytes per trainable parameter held for gradients."""
    optimizer_bytes: float = 0.0
    """Bytes per trainable parameter for optimizer state (Adam m, v, master)."""
    trainable_fraction: float = 0.0
    """Share of parameters that receive gradients. LoRA touches ~0.5%."""
    needs_activations: bool = False
    default_base_quant: str = "Q4_K_M"
    overhead_bytes: float = 0.8e9
    """Framework, CUDA/Metal context, allocator fragmentation."""


INFERENCE = Workload(
    key="inference",
    label="Inference",
    description="Running the model to generate text",
    default_base_quant="Q4_K_M",
    overhead_bytes=0.6e9,
)

QLORA = Workload(
    key="qlora",
    label="QLoRA fine-tune",
    description="Fine-tuning adapters on a 4-bit frozen base",
    grad_bytes=2.0,
    optimizer_bytes=8.0,
    trainable_fraction=0.005,
    needs_activations=True,
    default_base_quant="nf4",
    overhead_bytes=1.5e9,
)

LORA = Workload(
    key="lora",
    label="LoRA fine-tune",
    description="Fine-tuning adapters on a 16-bit frozen base",
    grad_bytes=2.0,
    optimizer_bytes=8.0,
    trainable_fraction=0.005,
    needs_activations=True,
    default_base_quant="bf16",
    overhead_bytes=1.5e9,
)

FULL_FINETUNE = Workload(
    key="full",
    label="Full fine-tune",
    description="Updating every weight, bf16 with Adam",
    grad_bytes=2.0,
    optimizer_bytes=12.0,  # Adam m + v in fp32, plus fp32 master weights
    trainable_fraction=1.0,
    needs_activations=True,
    default_base_quant="bf16",
    overhead_bytes=2.0e9,
)

TRAINING = Workload(
    key="train",
    label="Train from scratch",
    description="Pre-training a model from random initialisation",
    grad_bytes=2.0,
    optimizer_bytes=12.0,
    trainable_fraction=1.0,
    needs_activations=True,
    default_base_quant="bf16",
    overhead_bytes=2.5e9,
)

ALL: List[Workload] = [INFERENCE, QLORA, LORA, FULL_FINETUNE, TRAINING]
BY_KEY: Dict[str, Workload] = {w.key: w for w in ALL}

ALIASES = {
    "infer": "inference", "run": "inference", "serve": "inference", "chat": "inference",
    "finetune": "lora", "fine-tune": "lora", "ft": "lora", "peft": "lora",
    "q-lora": "qlora", "4bit-lora": "qlora",
    "full-finetune": "full", "sft": "full",
    "pretrain": "train", "scratch": "train", "training": "train",
}


def get(key: str) -> Workload:
    k = key.lower().strip()
    k = ALIASES.get(k, k)
    if k not in BY_KEY:
        raise KeyError("Unknown workload {!r}. Options: {}".format(
            key, ", ".join(BY_KEY) + ", " + ", ".join(ALIASES)))
    return BY_KEY[k]
