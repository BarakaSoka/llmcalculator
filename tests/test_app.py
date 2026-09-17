"""The local web app's JSON API."""

import json

import pytest

from llmcalculator.ui import app as webapp


def test_hardware_endpoint():
    d = webapp.api_hardware()
    assert d["ram_gb"] > 0
    assert len(d["capabilities"]) == 5
    assert json.dumps(d)


def test_models_endpoint_filters():
    everything = webapp.api_models()
    filtered = webapp.api_models("qwen")
    assert filtered["count"] < everything["count"]
    haystacks = [" ".join([m["name"], m["family"], m["hf_id"]]).lower()
                 for m in filtered["models"]]
    assert all("qwen" in h for h in haystacks)


def test_detail_endpoint_shape():
    d = webapp.api_detail("llama3.1:8b")
    for key in ("breakdown", "workloads", "quants", "contexts", "verdict"):
        assert key in d
    assert json.dumps(d)


def test_detail_rejects_unknown_model():
    with pytest.raises(KeyError):
        webapp.api_detail("not-a-real-model")


def test_compare_endpoint():
    d = webapp.api_compare(["llama3.2:1b", "llama3.1:8b"])
    assert len(d["results"]) == 2


def test_recommend_endpoint():
    d = webapp.api_recommend(limit=3)
    assert len(d["recommendations"]) <= 3


def test_page_file_is_installed():
    assert webapp._PAGE.exists()
    assert b"llmcalculator" in webapp._PAGE.read_bytes()


def test_hub_endpoint_empty_query_is_not_an_error():
    d = webapp.api_hub_search("")
    assert d["count"] == 0
    assert "error" not in d


def test_hub_endpoint_is_registered():
    assert "/api/hub" in webapp.ROUTES


def test_capabilities_endpoint_explains_every_trait():
    d = webapp.api_capabilities()
    assert "/api/capabilities" in webapp.ROUTES
    for kind in ("capabilities", "formats", "runtimes"):
        assert d[kind] and all(t["description"] for t in d[kind])
    assert d["counts"]["chat"] > 0


def test_models_endpoint_filters_by_capability():
    d = webapp.api_models(capability="vision")
    assert {m["name"] for m in d["models"]} == {"gemma3:4b", "gemma3:12b", "gemma3:27b"}
    assert all("Vision" in m["capabilities"] for m in d["models"])


def test_detail_endpoint_carries_the_model_information():
    d = webapp.api_detail("mixtral:8x7b")
    assert d["spec"]["n_experts"] == 8
    assert {t["key"] for t in d["spec"]["capabilities"]} >= {"chat", "moe"}
    assert any(i["label"] == "Attention" for i in d["architecture_items"])
    assert json.dumps(d)
