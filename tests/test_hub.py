"""Hugging Face Hub search, config resolution, and the parameter maths.

Tests that need the network are marked and skip cleanly when it is absent, so
a failure here means a real regression rather than a flaky connection.
"""

import json
import os

import pytest

from llmcalculator.models import hub
from llmcalculator.models.catalog import UnsupportedArchitecture, _spec_from_config


def _online() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/api/models?limit=1", timeout=8)
        return True
    except Exception:
        return False


needs_net = pytest.mark.skipif(not _online(), reason="needs network access")


# --- parameter maths (offline, the part most worth pinning) ---------------

def test_moe_uses_the_expert_width_not_the_dense_width():
    """Qwen3-30B-A3B has intermediate_size 6144 but moe_intermediate_size 768.
    Sizing its 128 experts with the dense width overstates it eightfold."""
    cfg = {
        "model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
        "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "intermediate_size": 6144, "moe_intermediate_size": 768,
        "num_experts": 128, "num_experts_per_tok": 8,
        "vocab_size": 151936, "max_position_embeddings": 262144,
    }
    spec = _spec_from_config("Qwen/Qwen3-30B-A3B", cfg)
    assert 29.0 < spec.params_b < 32.0
    assert 3.0 < spec.active_params / 1e9 < 3.8
    assert spec.is_moe


def test_mixtral_experts_use_the_dense_width_when_no_moe_width_is_given():
    cfg = {
        "model_type": "mixtral", "num_hidden_layers": 32, "hidden_size": 4096,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2,
        "vocab_size": 32000,
    }
    spec = _spec_from_config("mistralai/Mixtral-8x7B", cfg)
    assert 44.0 < spec.params_b < 49.0
    assert 11.5 < spec.active_params / 1e9 < 14.0


def test_tied_embeddings_are_not_double_counted():
    """Qwen2.5-0.5B ties a 151936 x 896 matrix. Counting it twice adds 27%."""
    base = {
        "model_type": "qwen2", "num_hidden_layers": 24, "hidden_size": 896,
        "num_attention_heads": 14, "num_key_value_heads": 2, "head_dim": 64,
        "intermediate_size": 4864, "vocab_size": 151936,
    }
    tied = _spec_from_config("Qwen/Qwen2.5-0.5B", {**base, "tie_word_embeddings": True})
    untied = _spec_from_config("Qwen/Qwen2.5-0.5B", {**base, "tie_word_embeddings": False})
    assert tied.params < untied.params
    assert 0.45e9 < tied.params < 0.55e9   # published: 0.49B


def test_dense_prefix_layers_are_counted_as_dense():
    """DeepSeek keeps the first few layers dense; treating them as MoE inflates."""
    cfg = {
        "model_type": "deepseek_v3", "num_hidden_layers": 61, "hidden_size": 7168,
        "num_attention_heads": 128, "num_key_value_heads": 128, "head_dim": 56,
        "intermediate_size": 18432, "moe_intermediate_size": 2048,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "n_shared_experts": 1,
        "first_k_dense_replace": 3, "vocab_size": 129280,
    }
    spec = _spec_from_config("deepseek-ai/DeepSeek-V3", cfg)
    assert 600 < spec.params_b < 750       # published: 671B
    assert 30 < spec.active_params / 1e9 < 45   # published: 37B


@pytest.mark.parametrize("arch", [
    "NemotronHForCausalLM", "MambaForCausalLM", "JambaForCausalLM", "RwkvForCausalLM",
])
def test_state_space_architectures_are_refused_not_guessed(arch):
    cfg = {"architectures": [arch], "num_hidden_layers": 32, "hidden_size": 4096,
           "num_attention_heads": 32, "vocab_size": 32000}
    with pytest.raises(UnsupportedArchitecture):
        _spec_from_config("some/model", cfg)


def test_chatglm_padded_vocab_key_is_understood():
    cfg = {"model_type": "chatglm", "num_layers": 40, "hidden_size": 4096,
           "num_attention_heads": 32, "padded_vocab_size": 151552,
           "ffn_hidden_size": 13696}
    spec = _spec_from_config("THUDM/glm-4-9b", cfg)
    assert spec.vocab_size == 151552


# --- cache (offline) ------------------------------------------------------

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    hub._cache_write("unit_test_key", {"hello": "world"})
    assert hub._cache_read("unit_test_key") == {"hello": "world"}


def test_cache_expires(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    hub._cache_write("expiring", {"a": 1})
    assert hub._cache_read("expiring", ttl=-1) is None


def test_clear_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    hub._cache_write("one", {}); hub._cache_write("two", {})
    assert hub.clear_cache() >= 2


def test_cache_write_failure_is_survivable(tmp_path, monkeypatch):
    """A broken cache must degrade to 'slower', never to 'crashed'."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(hub.Path, "write_text",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    hub._cache_write("x", {"a": 1})  # must not raise


# --- network --------------------------------------------------------------

@needs_net
def test_search_returns_hits():
    hits = hub.search("qwen", limit=5)
    assert hits
    assert all(h.id for h in hits)


@needs_net
def test_search_filters_gguf_by_default():
    assert all(not h.is_gguf for h in hub.search("llama gguf", limit=10))


@needs_net
def test_resolve_matches_published_size():
    spec = hub.resolve("Qwen/Qwen2.5-7B-Instruct")
    assert spec is not None
    assert 7.0 < spec.params_b < 8.2          # published: 7.62B
    assert spec.n_kv_heads == 4


@needs_net
def test_resolve_unknown_repo_returns_none():
    assert hub.resolve("definitely/not-a-real-repo-xyz") is None


@needs_net
def test_search_resolved_explains_unresolvable_hits():
    results = hub.search_resolved("llama", limit=8)
    assert results
    for r in results:
        assert r.resolved or r.error   # never silently dropped
