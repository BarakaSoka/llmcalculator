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
