"""Catalog integrity and lookup behaviour."""

import pytest

from llmcalculator.models import catalog
from llmcalculator.models.catalog import _spec_from_config


def test_catalog_loads():
    assert len(catalog.all_models()) >= 40


def test_every_model_is_internally_consistent():
    for m in catalog.all_models():
        assert m.params > 0, m.name
        assert m.n_layers > 0, m.name
        assert m.n_kv_heads <= m.n_heads, m.name
        assert m.hidden_size > 0, m.name
        assert m.active_params <= m.params, m.name
        assert m.max_context >= 2048, m.name
        assert m.head_dim > 0, m.name


def test_exact_lookup():
    assert catalog.get("llama3.1:8b").params_b == pytest.approx(8.0, abs=0.2)


@pytest.mark.parametrize("alias", ["llama3.1-8b", "Llama3.1:8B", "llama3.1 8b"])
def test_fuzzy_lookup(alias):
    assert catalog.get(alias).name == "llama3.1:8b"


def test_ambiguous_lookup_lists_candidates():
    with pytest.raises(KeyError) as exc:
        catalog.get("qwen2.5")
    assert "ambiguous" in str(exc.value)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        catalog.get("definitely-not-a-real-model")


def test_search_by_tag_and_family():
    assert catalog.search("code")
    assert {m.name for m in catalog.search("moe")} >= {"mixtral:8x7b", "qwen3:30b-a3b"}


def test_search_covers_hf_id():
    """Searching 'qwen' should surface the DeepSeek distills of Qwen too, since
    that is what someone looking for a Qwen-based model actually wants."""
    names = {m.name for m in catalog.search("qwen")}
    assert "qwen2.5:7b" in names
    assert "deepseek-r1:7b" in names  # hf_id is DeepSeek-R1-Distill-Qwen-7B


def test_moe_flag():
    assert catalog.get("mixtral:8x7b").is_moe
    assert not catalog.get("llama3.1:8b").is_moe


def test_spec_from_hf_config():
    """A transformers config.json should map onto a usable ModelSpec."""
    cfg = {
        "model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 14336, "vocab_size": 128256,
        "max_position_embeddings": 131072,
    }
    spec = _spec_from_config("meta-llama/Llama-3.1-8B", cfg)
    assert 7.0e9 < spec.params < 9.0e9
    assert spec.n_kv_heads == 8
    assert spec.gqa_ratio == 4.0


def test_spec_from_multimodal_config_uses_text_config():
    cfg = {"model_type": "gemma3", "text_config": {
        "num_hidden_layers": 34, "hidden_size": 2560, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 10240,
        "vocab_size": 262144}}
    spec = _spec_from_config("google/gemma-3-4b-it", cfg)
    assert spec.n_layers == 34
    assert spec.hidden_size == 2560
