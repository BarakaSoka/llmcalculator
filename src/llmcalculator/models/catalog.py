"""The built-in model catalog, plus live lookup from Hugging Face.

The bundled catalog covers the models people actually run locally. For
anything else, `from_hf()` reads the real `config.json` off the Hub so the
estimate uses the model's true architecture rather than a guess.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .spec import ModelSpec


class UnsupportedArchitecture(ValueError):
    """Raised for models this estimator deliberately will not guess at."""


_CATALOG_PATH = Path(__file__).with_name("catalog.json")

# The analytic parameter count assumes a standard attention + SwiGLU decoder.
# State-space and hybrid models allocate their weights differently, so the
# formula would return a confident wrong number rather than no number.
_UNSUPPORTED_ARCH = (
    "mamba", "rwkv", "hyena", "retnet", "griffin", "recurrentgemma",
    "nemotronh", "zamba", "jamba", "falconmamba", "bamba", "plamo2",
)
_cache: Optional[Dict[str, ModelSpec]] = None


def _load() -> Dict[str, ModelSpec]:
    global _cache
    if _cache is None:
        raw = json.loads(_CATALOG_PATH.read_text())
        specs = {}
        for entry in raw:
            entry = dict(entry)
            entry["tags"] = tuple(entry.get("tags", ()))
            spec = ModelSpec(**entry)
            specs[spec.name.lower()] = spec
        _cache = specs
    return _cache


def all_models() -> List[ModelSpec]:
    """Every model in the catalog, smallest first."""
    return sorted(_load().values(), key=lambda m: m.params)


def get(name: str) -> ModelSpec:
    """Look up a model by catalog name, with fuzzy fallback.

    Accepts `llama3.1:8b`, `llama3.1-8b`, `Llama 3.1 8B`, or a bare `8b`
    style suffix when it is unambiguous.
    """
    models = _load()
    key = name.lower().strip()
    if key in models:
        return models[key]

    norm = re.sub(r"[\s_-]+", "", key).replace(":", "")
    for k, spec in models.items():
        if re.sub(r"[\s_:-]+", "", k) == norm:
            return spec

    matches = [s for k, s in models.items() if norm in re.sub(r"[\s_:-]+", "", k)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(sorted(m.name for m in matches))
        raise KeyError("{!r} is ambiguous. Did you mean: {}".format(name, names))
    raise KeyError("Unknown model {!r}. Try `llmcalculator models` to list, or pass --hf <repo-id>.".format(name))


def search(query: str) -> List[ModelSpec]:
    """Substring search across name, family, tags and capabilities.

    Capabilities are searched too, so `search("vision")` finds the multimodal
    models even though none of them carry that word in their name.
    """
    q = query.lower().strip()
    out = []
    for spec in all_models():
        haystack = " ".join([spec.name, spec.family, " ".join(spec.tags), spec.hf_id,
                             " ".join(spec.capability_keys)]).lower()
        if q in haystack:
            out.append(spec)
    return out


def by_capability(key: str) -> List[ModelSpec]:
    """Every catalog model with a given capability, e.g. `code` or `tools`."""
    want = key.lower().strip()
    return [m for m in all_models() if want in m.capability_keys]


def capability_counts() -> Dict[str, int]:
    """How many catalog models have each capability, commonest first."""
    counts: Dict[str, int] = {}
    for m in all_models():
        for cap in m.capability_keys:
            counts[cap] = counts.get(cap, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def families() -> List[str]:
    return sorted({s.family for s in all_models() if s.family})


# --- Hugging Face ---------------------------------------------------------

def from_hf(repo_id: str, token: Optional[str] = None, timeout: float = 15.0) -> ModelSpec:
    """Build a ModelSpec from a model's config.json on the Hugging Face Hub.

    Only the standard library is used, so this works without `huggingface_hub`
    installed. Gated repos need a token (env: HF_TOKEN).
    """
    import urllib.error
    import urllib.request

    url = "https://huggingface.co/{}/resolve/main/config.json".format(repo_id)
    req = urllib.request.Request(url, headers={"User-Agent": "llmcalculator"})
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer {}".format(token))

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cfg = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "{} is gated or private. Set HF_TOKEN to a token with access.".format(repo_id)
            ) from exc
        if exc.code == 404:
            raise RuntimeError("No config.json found for {!r}.".format(repo_id)) from exc
        raise RuntimeError("Hugging Face returned HTTP {} for {}".format(exc.code, repo_id)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Hugging Face: {}".format(exc.reason)) from exc

    return _spec_from_config(repo_id, cfg)


def _spec_from_config(repo_id: str, cfg: dict) -> ModelSpec:
    """Translate a transformers config dict into a ModelSpec."""
    # Multimodal configs nest the language model one level down.
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg = {**cfg, **cfg["text_config"]}

    def pick(*keys, default=None):
        for k in keys:
            if cfg.get(k) is not None:
                return cfg[k]
        return default

    arch = " ".join(cfg.get("architectures") or []) + " " + str(cfg.get("model_type", ""))
    lowered = arch.lower().replace("_", "").replace("-", "")
    for bad in _UNSUPPORTED_ARCH:
        if bad in lowered:
            raise UnsupportedArchitecture(
                "{} uses a {} architecture. llmcalculator sizes standard "
                "transformer decoders; its parameter and KV-cache maths do not "
                "apply here, so it declines to guess.".format(repo_id, bad))

    n_layers = pick("num_hidden_layers", "n_layer", "num_layers", default=32)
    hidden = pick("hidden_size", "n_embd", "d_model", default=4096)
    n_heads = pick("num_attention_heads", "n_head", default=32)
    n_kv = pick("num_key_value_heads", "num_kv_heads", default=n_heads)
    head_dim = pick("head_dim", default=hidden // max(n_heads, 1))
    vocab = pick("vocab_size", "padded_vocab_size", default=32000)
    ctx = pick("max_position_embeddings", "n_positions", default=8192)
    intermediate = pick("intermediate_size", "ffn_dim", default=hidden * 4)

    n_experts = pick("num_local_experts", "num_experts", "n_routed_experts", default=0) or 0
    n_active = pick("num_experts_per_tok", "moe_topk", default=0) or 0
    n_shared = pick("n_shared_experts", "shared_expert_intermediate_size", default=0) or 0
    if n_shared and n_shared > 64:
        # Qwen-style configs give a width here rather than a count.
        n_shared = 1
    # Experts are usually much narrower than the dense FFN. Using the dense
    # `intermediate_size` for them overstates a model by the expert count.
    moe_inter = pick("moe_intermediate_size", default=intermediate)
    dense_first = pick("first_k_dense_replace", default=0) or 0
    rope = pick("rope_theta", "rotary_emb_base", default=0) or 0
    window = pick("sliding_window", "attention_window_size", default=0) or 0
    if cfg.get("use_sliding_window") is False:
        window = 0

    # Tied embeddings share one matrix between input and output. Assuming two
    # overstates a small model with a large vocabulary badly: Qwen2.5-0.5B ties
    # a 151936 x 896 matrix, which is 27% of the whole model counted twice.
    tied = bool(cfg.get("tie_word_embeddings", False))

    params = _estimate_params(
        n_layers, hidden, n_kv * head_dim, n_heads * head_dim, intermediate, vocab,
        n_experts=n_experts, moe_intermediate=moe_inter, n_shared=n_shared,
        dense_layers=dense_first, tied_embeddings=tied)
    if n_experts and n_active:
        active = _estimate_params(
            n_layers, hidden, n_kv * head_dim, n_heads * head_dim, intermediate, vocab,
            n_experts=n_active, moe_intermediate=moe_inter, n_shared=n_shared,
            dense_layers=dense_first, tied_embeddings=tied)
    else:
        active = params

    # Config facts that do not change the size but do change what the model
    # is useful for. A vision or audio tower nested in the config is the only
    # reliable signal that a repo is multimodal.
    extra_caps = []
    if isinstance(cfg.get("vision_config"), dict) or "vision_tower" in cfg:
        extra_caps.append("vision")
    if isinstance(cfg.get("audio_config"), dict):
        extra_caps.append("audio")

    return ModelSpec(
        name=repo_id.split("/")[-1].lower(),
        params=params,
        active_params=active,
        n_layers=int(n_layers),
        hidden_size=int(hidden),
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv),
        head_dim=int(head_dim),
        vocab_size=int(vocab),
        max_context=int(ctx),
        intermediate_size=int(intermediate) if intermediate else None,
        moe_intermediate_size=int(moe_inter) if n_experts and moe_inter else None,
        n_experts=int(n_experts),
        n_active_experts=int(n_active),
        tie_word_embeddings=tied,
        rope_theta=float(rope) if rope else None,
        sliding_window=int(window) if window else None,
        torch_dtype=str(cfg.get("torch_dtype") or cfg.get("dtype") or ""),
        architecture=" ".join(cfg.get("architectures") or []) or str(cfg.get("model_type", "")),
        capabilities=tuple(extra_caps),
        family=str(cfg.get("model_type", "")),
        hf_id=repo_id,
        tags=("from-hf",),
    )


def _estimate_params(n_layers, hidden, kv_dim, q_dim, intermediate, vocab,
                     n_experts=0, moe_intermediate=None, n_shared=0,
                     dense_layers=0, tied_embeddings=False) -> float:
    """Analytic parameter count for a decoder-only transformer.

    Handles the mixture-of-experts case, where the routed experts are usually
    far narrower than the dense feed-forward width. Qwen3-30B-A3B, for example,
    has `intermediate_size` 6144 but `moe_intermediate_size` 768; sizing its 128
    experts with the former overstates the model roughly eightfold.

    Some architectures (DeepSeek) keep the first few layers dense, which
    `dense_layers` accounts for.
    """
    attn = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    dense_mlp = 3 * hidden * intermediate  # gate, up, down (SwiGLU)
    norms = 2 * hidden

    if n_experts:
        expert_width = moe_intermediate or intermediate
        moe_mlp = 3 * hidden * expert_width * (n_experts + n_shared)
        dense_layers = min(int(dense_layers), int(n_layers))
        moe_layers = max(int(n_layers) - dense_layers, 0)
        body = (dense_layers * (attn + dense_mlp + norms)
                + moe_layers * (attn + moe_mlp + norms))
    else:
        body = n_layers * (attn + dense_mlp + norms)

    embeddings = (1 if tied_embeddings else 2) * vocab * hidden
    return float(body + embeddings)
