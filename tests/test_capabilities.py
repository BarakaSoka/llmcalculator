"""Model capabilities, weight formats and runtime support.

These describe what a model is for rather than how big it is, so the tests
here are about correct derivation and honest gaps: nothing should be claimed
that the catalog, the config file or the repository tags do not support.
"""

import json
import subprocess
import sys

import pytest

from llmcalculator.models import capabilities as caps
from llmcalculator.models import catalog, hub
from llmcalculator.models.catalog import _spec_from_config
from llmcalculator.models.spec import ModelSpec


def run(args):
    return subprocess.run([sys.executable, "-m", "llmcalculator.cli"] + args,
                          capture_output=True, text=True, timeout=120)


# --- the registries -------------------------------------------------------

@pytest.mark.parametrize("kind", ["capability", "format", "runtime"])
def test_every_trait_is_explained(kind):
    """A trait with no description is a chip nobody can interpret."""
    for t in caps.known(kind):
        assert t.label, t.key
        assert len(t.description) > 30, t.key


def test_unknown_keys_pass_through_rather_than_raising():
    """A Hub tag we have never seen is still worth showing."""
    t = caps.capability("brand-new-thing")
    assert t.label == "brand new thing"
    assert t.description == ""


# --- catalog data ---------------------------------------------------------

def test_every_catalog_model_states_what_it_is_for():
    for m in catalog.all_models():
        assert m.capabilities, m.name
        assert m.support().capabilities, m.name


def test_capabilities_are_all_from_the_registry():
    """Curated data drifting away from the registry would leave undescribed
    chips in every interface at once."""
    known = set(caps.CAPABILITIES)
    for m in catalog.all_models():
        unknown = set(m.capabilities) - known
        assert not unknown, "{}: {}".format(m.name, unknown)


def test_capability_lookup_and_counts():
    assert {m.name for m in catalog.by_capability("vision")} == {
        "gemma3:4b", "gemma3:12b", "gemma3:27b"}
    counts = catalog.capability_counts()
    assert counts["chat"] == len(catalog.all_models())
    assert list(counts) == sorted(counts, key=lambda k: (-counts[k], k))


def test_search_finds_models_by_capability():
    """`vision` appears in no model name, only in its capabilities."""
    assert {m.name for m in catalog.search("vision")} >= {"gemma3:12b"}


# --- derivation -----------------------------------------------------------

def test_architecture_facts_are_added_even_when_uncurated():
    m = catalog.get("qwen3-coder:30b")
    keys = m.capability_keys
    assert "moe" in keys           # from active_params, not from the catalog
    assert "long-context" in keys  # from max_context, which is 256k here
    assert "edge" not in keys
    # and a 32k sibling is not silently promoted
    assert "long-context" not in catalog.get("qwen3:30b-a3b").capability_keys


def test_small_models_are_marked_for_the_edge():
    assert catalog.get("qwen2.5:0.5b").has_capability("edge")
    assert not catalog.get("qwen2.5:72b").has_capability("edge")


def test_runtimes_follow_from_formats():
    """GGUF is what llama.cpp reads; Ollama and LM Studio are llama.cpp."""
    sup = catalog.get("llama3.1:8b").support()
    assert "gguf" in sup.format_keys
    for engine in ("llama.cpp", "ollama", "lm-studio"):
        assert engine in sup.runtime_keys


def test_a_declared_format_is_not_overwritten_by_the_derived_set():
    """gpt-oss publishes MXFP4 natively, which no derivation would guess."""
    assert "mxfp4" in catalog.get("gpt-oss:20b").support().format_keys


def test_vision_models_carry_the_sizing_caveat():
    notes = " ".join(catalog.get("gemma3:12b").support().notes)
    assert "vision tower" in notes


def test_moe_note_quotes_both_parameter_counts():
    note = " ".join(catalog.get("mixtral:8x7b").support().notes)
    assert "12.9B" in note and "46.7B" in note


def test_non_commercial_licences_are_called_out():
    assert any("non-commercial" in n for n in catalog.get("command-r:35b").support().notes)
    assert not any("non-commercial" in n for n in catalog.get("mistral:7b").support().notes)


# --- config and Hub metadata ---------------------------------------------

def test_config_fields_reach_the_spec():
    cfg = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 14336,
        "vocab_size": 128256, "max_position_embeddings": 131072,
        "rope_theta": 500000.0, "torch_dtype": "bfloat16",
    }
    spec = _spec_from_config("meta-llama/Llama-3.1-8B-Instruct", cfg)
    assert spec.intermediate_size == 14336
    assert spec.rope_theta == 500000.0
    assert spec.torch_dtype == "bfloat16"
    assert spec.architecture == "LlamaForCausalLM"
    assert spec.attention_kind == "GQA 4:1"


def test_a_nested_vision_tower_marks_the_model_multimodal():
    cfg = {"model_type": "gemma3", "vision_config": {"hidden_size": 1152},
           "text_config": {"num_hidden_layers": 34, "hidden_size": 2560,
                           "num_attention_heads": 8, "num_key_value_heads": 4,
                           "head_dim": 256, "intermediate_size": 10240,
                           "vocab_size": 262144}}
    spec = _spec_from_config("google/gemma-3-4b-it", cfg)
    assert spec.has_capability("vision")


def test_sliding_window_is_ignored_when_the_config_disables_it():
    cfg = {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584,
           "num_attention_heads": 28, "num_key_value_heads": 4,
           "sliding_window": 131072, "use_sliding_window": False}
    assert _spec_from_config("Qwen/Qwen2.5-7B", cfg).sliding_window is None


def test_hub_tags_add_formats_and_a_licence():
    """A config file cannot say which formats a repo publishes; tags can."""
    spec = ModelSpec(name="x", params=7e9, n_layers=32, hidden_size=4096,
                     n_heads=32, n_kv_heads=8, hf_id="acme/x")
    hit = hub.HubModel(id="acme/x", library="transformers",
                       tags=("awq", "conversational", "license:apache-2.0",
                             "base_model:acme/x-base"),
                       pipeline="text-generation")
    hub.apply_hub_metadata(spec, hit)
    assert spec.license == "apache-2.0"
    assert "awq" in spec.formats
    assert "chat" in spec.capabilities
    assert "exllamav2" not in spec.runtimes   # nothing implied EXL2


# --- serialisation and the interfaces ------------------------------------

def test_spec_as_dict_is_json_ready_and_complete():
    d = catalog.get("qwen3-coder:30b").as_dict()
    assert json.dumps(d)
    assert d["n_experts"] == 128 and d["n_active_experts"] == 8
    assert {t["key"] for t in d["capabilities"]} >= {"code", "moe", "tools"}
    assert all(t["description"] for t in d["formats"])


def test_architecture_items_omit_what_is_not_known():
    labels = [k for k, _ in catalog.get("llama3.1:8b").architecture_items()]
    assert "Feed-forward" not in labels   # not in the bundled catalog data
    assert "Attention" in labels and "KV cache" in labels


def test_info_command_prints_descriptions():
    r = run(["info", "gemma3:12b"])
    assert r.returncode == 0, r.stderr
    assert "Capabilities" in r.stdout
    assert "Accepts images alongside text" in r.stdout
    assert "Weight formats" in r.stdout and "Runtimes" in r.stdout


def test_info_json_round_trips():
    r = run(["info", "mixtral:8x7b", "--json"])
    assert r.returncode == 0
    assert json.loads(r.stdout)["is_moe"] is True


def test_capabilities_command_lists_the_vocabulary():
    d = json.loads(run(["capabilities", "--json"]).stdout)
    assert {"capabilities", "formats", "runtimes"} == set(d)
    assert any(t["key"] == "gguf" for t in d["formats"])


def test_models_can_be_filtered_by_capability():
    r = run(["models", "-c", "vision", "--json"])
    assert r.returncode == 0
    assert {m["name"] for m in json.loads(r.stdout)} == {
        "gemma3:4b", "gemma3:12b", "gemma3:27b"}


def test_models_reports_an_unknown_capability_clearly():
    r = run(["models", "-c", "telepathy"])
    assert r.returncode == 1
    assert "Known capabilities" in r.stdout
