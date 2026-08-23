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

_CATALOG_PATH = Path(__file__).with_name("catalog.json")
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
    """Substring search across name, family and tags."""
    q = query.lower().strip()
    out = []
    for spec in all_models():
        haystack = " ".join([spec.name, spec.family, " ".join(spec.tags), spec.hf_id]).lower()
        if q in haystack:
            out.append(spec)
    return out


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

    n_layers = pick("num_hidden_layers", "n_layer", "num_layers", default=32)
    hidden = pick("hidden_size", "n_embd", "d_model", default=4096)
    n_heads = pick("num_attention_heads", "n_head", default=32)
    n_kv = pick("num_key_value_heads", "num_kv_heads", default=n_heads)
    head_dim = pick("head_dim", default=hidden // max(n_heads, 1))
    vocab = pick("vocab_size", default=32000)
    ctx = pick("max_position_embeddings", "n_positions", default=8192)
    intermediate = pick("intermediate_size", "ffn_dim", default=hidden * 4)

    n_experts = pick("num_local_experts", "num_experts", "n_routed_experts", default=0) or 0
    n_active = pick("num_experts_per_tok", "moe_topk", default=0) or 0

    params = _estimate_params(n_layers, hidden, n_kv * head_dim, n_heads * head_dim,
                              intermediate, vocab, n_experts)
    if n_experts and n_active:
        active = _estimate_params(n_layers, hidden, n_kv * head_dim, n_heads * head_dim,
                                  intermediate, vocab, n_active)
    else:
        active = params

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
        family=str(cfg.get("model_type", "")),
        hf_id=repo_id,
        tags=("from-hf",),
    )


def _estimate_params(n_layers, hidden, kv_dim, q_dim, intermediate, vocab, n_experts=0) -> float:
    """Analytic parameter count for a standard decoder-only transformer."""
    attn = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    mlp = 3 * hidden * intermediate  # gate, up, down (SwiGLU)
    if n_experts:
        mlp *= n_experts
    per_layer = attn + mlp + 2 * hidden  # + two RMSNorms
    embeddings = 2 * vocab * hidden  # input + output, untied worst case
    return float(n_layers * per_layer + embeddings)
