"""Model architecture description.

Everything the estimator needs to size a model is here. The architecture
fields (layers, hidden size, KV head count) matter enormously: two 7B models
with different grouped-query-attention configs can differ by 8x in KV cache
at long context, which is usually what actually decides whether a model fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    """Architecture of a transformer language model."""

    name: str
    params: float
    """Total parameter count."""

    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    """Key/value head count. Equals n_heads for multi-head attention; smaller
    for grouped-query attention, which shrinks the KV cache proportionally."""

    max_context: int = 8192
    vocab_size: int = 32000
    head_dim: Optional[int] = None
    """Explicit head dim. Some models (Gemma) decouple this from hidden/heads."""

    active_params: Optional[float] = None
    """Parameters used per token. Less than `params` for mixture-of-experts."""

    family: str = ""
    license: str = ""
    tags: tuple = ()
    hf_id: str = ""

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.n_heads
        if self.active_params is None:
            self.active_params = self.params

    # --- derived properties ------------------------------------------------

    @property
    def is_moe(self) -> bool:
        return self.active_params < self.params * 0.95

    @property
    def kv_dim(self) -> int:
        """Width of one layer's K (or V) vector."""
        return self.n_kv_heads * self.head_dim

    @property
    def gqa_ratio(self) -> float:
        return self.n_heads / max(self.n_kv_heads, 1)

    @property
    def params_b(self) -> float:
        return self.params / 1e9

    def kv_cache_bytes(self, context: int, batch: int = 1, kv_bytes: float = 2.0) -> float:
        """Bytes of KV cache for a given context length.

        Two tensors (K and V) per layer, each `kv_dim` wide per token.
        """
        return 2 * self.n_layers * self.kv_dim * context * batch * kv_bytes

    def activation_bytes(self, context: int, batch: int = 1, checkpointing: bool = True) -> float:
        """Peak activation memory during a training step.

        Without gradient checkpointing every layer's intermediates are retained
        for the backward pass. With checkpointing only layer boundaries are kept
        and the rest is recomputed, which trades ~30% more compute for a large
        memory saving.
        """
        per_layer_per_token = self.hidden_size * 20  # attn + MLP intermediates, bf16
        if checkpointing:
            effective_layers = max(1.0, self.n_layers ** 0.5)
        else:
            effective_layers = float(self.n_layers)
        acts = batch * context * per_layer_per_token * effective_layers
        # logits are a real cost at large vocab: [batch, seq, vocab] in fp32
        logits = batch * context * self.vocab_size * 4
        return acts + logits

    def describe(self) -> str:
        bits = ["{:.1f}B params".format(self.params_b)]
        if self.is_moe:
            bits.append("{:.1f}B active (MoE)".format(self.active_params / 1e9))
        bits.append("{} layers".format(self.n_layers))
        if self.gqa_ratio > 1:
            bits.append("GQA {:.0f}:1".format(self.gqa_ratio))
        bits.append("{}k ctx".format(self.max_context // 1024))
        return ", ".join(bits)
