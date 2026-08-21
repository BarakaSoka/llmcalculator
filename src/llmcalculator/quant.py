"""Quantization formats and their real on-disk / in-memory cost.

Bytes-per-weight figures are measured effective rates for llama.cpp GGUF
k-quants (which mix bit-widths across tensor types) and for the standard
bitsandbytes / GPTQ / AWQ schemes. They are deliberately empirical rather
than the nominal bit count: Q4_K_M is nominally 4 bits but lands near 4.8
once scales, mins and the higher-precision attention tensors are counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class QuantFormat:
    """A weight quantization scheme."""

    name: str
    bytes_per_weight: float
    quality: float
    """Retained quality, 0-1, as a rough perplexity-preservation proxy."""
    family: str
    trainable: bool = False
    """Whether adapters can be trained on top of a base in this format."""
    note: str = ""

    @property
    def bits_per_weight(self) -> float:
        return self.bytes_per_weight * 8

    def weight_bytes(self, params: float) -> float:
        return params * self.bytes_per_weight


# --- registry -------------------------------------------------------------

_FORMATS: List[QuantFormat] = [
    QuantFormat("fp32", 4.0, 1.000, "float", True, "Full precision; almost never needed for inference"),
    QuantFormat("fp16", 2.0, 1.000, "float", True, "Reference quality"),
    QuantFormat("bf16", 2.0, 1.000, "float", True, "Reference quality, better training dynamics"),
    QuantFormat("fp8", 1.0, 0.995, "float", False, "Hopper/Ada and MI300 native"),
    # GGUF k-quants (llama.cpp / Ollama / LM Studio)
    QuantFormat("Q8_0", 1.09, 0.998, "gguf", False, "Near-lossless"),
    QuantFormat("Q6_K", 0.84, 0.995, "gguf", False, "Excellent; hard to distinguish from fp16"),
    QuantFormat("Q5_K_M", 0.73, 0.988, "gguf", False, "Very good"),
    QuantFormat("Q4_K_M", 0.63, 0.975, "gguf", False, "Best size/quality tradeoff for most users"),
    QuantFormat("Q4_K_S", 0.58, 0.966, "gguf", False, "Slightly smaller, slightly worse than Q4_K_M"),
    QuantFormat("IQ4_XS", 0.53, 0.960, "gguf", False, "Imatrix 4-bit; good at small sizes"),
    QuantFormat("Q3_K_M", 0.51, 0.930, "gguf", False, "Noticeable degradation"),
    QuantFormat("IQ3_XS", 0.43, 0.910, "gguf", False, "Last resort for large models"),
    QuantFormat("Q2_K", 0.40, 0.830, "gguf", False, "Severe degradation; emergency only"),
    # Training-capable quantized bases
    QuantFormat("nf4", 0.55, 0.970, "bnb", True, "bitsandbytes 4-bit; the QLoRA base format"),
    QuantFormat("int8", 1.06, 0.992, "bnb", True, "bitsandbytes 8-bit"),
    # Serving-oriented
    QuantFormat("gptq-4bit", 0.60, 0.968, "gptq", False, "vLLM/ExLlama 4-bit"),
    QuantFormat("awq-4bit", 0.60, 0.972, "awq", False, "Activation-aware 4-bit"),
    QuantFormat("mlx-4bit", 0.58, 0.968, "mlx", True, "Apple MLX 4-bit; supports LoRA training"),
    QuantFormat("mlx-8bit", 1.06, 0.994, "mlx", True, "Apple MLX 8-bit"),
]

FORMATS: Dict[str, QuantFormat] = {f.name.lower(): f for f in _FORMATS}

DEFAULT_INFERENCE = "Q4_K_M"
DEFAULT_TRAINING_BASE = "bf16"
DEFAULT_QLORA_BASE = "nf4"

# Ordered best-quality-first; used when searching for a format that fits.
LADDER = ["fp16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "IQ3_XS", "Q2_K"]


def get(name: str) -> QuantFormat:
    """Look up a format by name, case-insensitively."""
    key = name.lower().strip()
    if key in FORMATS:
        return FORMATS[key]
    aliases = {
        "q4": "q4_k_m", "q5": "q5_k_m", "q6": "q6_k", "q8": "q8_0",
        "q3": "q3_k_m", "4bit": "nf4", "8bit": "int8",
        "f16": "fp16", "half": "fp16", "full": "fp32",
    }
    if key in aliases:
        return FORMATS[aliases[key]]
    raise KeyError(
        "Unknown quantization {!r}. Known: {}".format(name, ", ".join(f.name for f in _FORMATS))
    )


def ladder(trainable_only: bool = False) -> List[QuantFormat]:
    """Formats ordered from highest to lowest quality."""
    out = [get(n) for n in LADDER]
    if trainable_only:
        out = [f for f in out if f.trainable]
    return out


def kv_cache_bytes_per_element(name: str) -> float:
    """Bytes per KV cache element for a cache quantization setting."""
    table = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "q8_0": 1.0, "q4_0": 0.5}
    return table.get(name.lower(), 2.0)
