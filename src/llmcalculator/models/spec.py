"""Model architecture description.

Everything the estimator needs to size a model is here. The architecture
fields (layers, hidden size, KV head count) matter enormously: two 7B models
with different grouped-query-attention configs can differ by 8x in KV cache
at long context, which is usually what actually decides whether a model fits.

Sizing is not the whole story, though. Two models with identical memory
profiles can be useless to each other's users, so a spec also carries what
the model is for and what will run it - see `capabilities.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import capabilities as caps


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

    # --- feed-forward and expert shape ------------------------------------
    intermediate_size: Optional[int] = None
    """Feed-forward width. None when the catalog entry predates the field;
    every model resolved from a config.json has it."""

    n_experts: int = 0
    """Routed experts per MoE layer. Zero for a dense model."""

    n_active_experts: int = 0
    """Experts each token is routed through."""

    moe_intermediate_size: Optional[int] = None
    """Width of one routed expert, usually far narrower than the dense FFN."""

    # --- other config facts that change what you can do with the model ----
    tie_word_embeddings: bool = False
    """Input and output embedding matrices share storage."""

    rope_theta: Optional[float] = None
    """RoPE base frequency. A large value is how a model earns a long window."""

    sliding_window: Optional[int] = None
    """Local attention span, if the model uses one. Caps KV cache growth."""

    torch_dtype: str = ""
    """Precision the weights are published in."""

    architecture: str = ""
    """The `architectures` entry from config.json, e.g. LlamaForCausalLM."""

    # --- what the model is for --------------------------------------------
    capabilities: tuple = ()
    """Curated capability keys; see `capabilities.CAPABILITIES`. Left empty,
    they are inferred from tags, name and architecture."""

    formats: tuple = ()
    """Weight formats known to exist. Empty means derive the usual set."""

    runtimes: tuple = ()
    """Engines known to run it. Empty means derive them from the formats."""

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.n_heads
        if self.active_params is None:
            self.active_params = self.params
        self.tags = tuple(self.tags)
        self.capabilities = tuple(self.capabilities)
        self.formats = tuple(self.formats)
        self.runtimes = tuple(self.runtimes)

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

    @property
    def active_params_b(self) -> float:
        return self.active_params / 1e9

    @property
    def attention_kind(self) -> str:
        """Multi-head, grouped-query or multi-query, spelled out.

        This is the single field that most often explains why one 7B model
        needs 8x the KV cache of another at the same context length.
        """
        if self.n_kv_heads <= 1:
            return "MQA"
        if self.n_kv_heads >= self.n_heads:
            return "MHA"
        return "GQA {:.0f}:1".format(self.gqa_ratio)

    @property
    def kv_bytes_per_token(self) -> float:
        """KV cache cost of one token at fp16, across every layer."""
        return self.kv_cache_bytes(context=1)

    @property
    def ffn_ratio(self) -> Optional[float]:
        """Feed-forward width as a multiple of the hidden size."""
        if not self.intermediate_size:
            return None
        return self.intermediate_size / max(self.hidden_size, 1)

    # --- capabilities, formats, runtimes -----------------------------------

    def support(self, extra_tags: tuple = ()) -> "caps.SupportProfile":
        """What this model is for, how it ships, and what will run it."""
        return caps.profile(self, extra_tags)

    @property
    def capability_keys(self) -> Tuple[str, ...]:
        return caps.infer_capabilities(self)

    def has_capability(self, key: str) -> bool:
        return key.lower().strip() in self.capability_keys

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

    def architecture_items(self) -> List[Tuple[str, str]]:
        """Label/value pairs describing the architecture, for display.

        Fields that are unknown for a given model are omitted rather than
        printed as a blank or a guess.
        """
        out: List[Tuple[str, str]] = [
            ("Parameters", "{:.2f}B".format(self.params_b)),
        ]
        if self.is_moe:
            out.append(("Active per token", "{:.2f}B".format(self.active_params_b)))
            if self.n_experts:
                routed = "{} experts".format(self.n_experts)
                if self.n_active_experts:
                    routed += ", {} used per token".format(self.n_active_experts)
                out.append(("Experts", routed))
        out += [
            ("Layers", str(self.n_layers)),
            ("Hidden size", str(self.hidden_size)),
            ("Attention", "{} ({} query / {} KV heads, head dim {})".format(
                self.attention_kind, self.n_heads, self.n_kv_heads, self.head_dim)),
        ]
        if self.intermediate_size:
            ratio = self.ffn_ratio
            out.append(("Feed-forward", "{} wide ({:.1f}x hidden)".format(
                self.intermediate_size, ratio)))
        if self.moe_intermediate_size and self.moe_intermediate_size != self.intermediate_size:
            out.append(("Expert width", str(self.moe_intermediate_size)))
        out.append(("Vocabulary", "{:,} tokens{}".format(
            self.vocab_size, " (tied embeddings)" if self.tie_word_embeddings else "")))
        out.append(("Max context", "{:,} tokens".format(self.max_context)))
        if self.sliding_window:
            out.append(("Sliding window", "{:,} tokens".format(self.sliding_window)))
        if self.rope_theta:
            out.append(("RoPE theta", "{:,.0f}".format(self.rope_theta)))
        out.append(("KV cache", "{:.2f} MB per 1k tokens at fp16".format(
            self.kv_bytes_per_token * 1024 / (1024 ** 2))))
        if self.torch_dtype:
            out.append(("Published as", self.torch_dtype))
        if self.architecture:
            out.append(("Architecture", self.architecture))
        if self.family:
            out.append(("Family", self.family))
        if self.license:
            out.append(("License", self.license))
        if self.hf_id:
            out.append(("Repository", self.hf_id))
        return out

    def as_dict(self, support: bool = True) -> Dict[str, object]:
        """Everything known about the model, as JSON-ready data."""
        out: Dict[str, object] = {
            "name": self.name,
            "hf_id": self.hf_id,
            "family": self.family,
            "license": self.license,
            "tags": list(self.tags),
            "params": self.params,
            "params_b": round(self.params_b, 3),
            "active_params_b": round(self.active_params_b, 3),
            "is_moe": self.is_moe,
            "n_experts": self.n_experts,
            "n_active_experts": self.n_active_experts,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attention": self.attention_kind,
            "gqa_ratio": round(self.gqa_ratio, 2),
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
            "max_context": self.max_context,
            "sliding_window": self.sliding_window,
            "rope_theta": self.rope_theta,
            "torch_dtype": self.torch_dtype,
            "architecture": self.architecture,
            "kv_bytes_per_token": round(self.kv_bytes_per_token, 1),
            "describe": self.describe(),
        }
        if support:
            out.update(self.support().as_dict())
        return out
