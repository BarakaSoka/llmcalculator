"""Tests for the sizing math.

The reference numbers are drawn from published model cards and from what
llama.cpp and bitsandbytes actually allocate in practice, so a failure here
means the estimator has drifted from reality rather than from itself.
"""

import pytest

from llmcalculator import quant, workloads
from llmcalculator.estimate import Verdict, estimate, max_model_size, recommended_quant
from llmcalculator.hardware import manual
from llmcalculator.models import catalog

GB = 1024 ** 3


@pytest.fixture
def rtx4090():
    return manual(vram_gb=24, ram_gb=64, gpu_name="RTX 4090")


@pytest.fixture
def mac36():
    from llmcalculator.hardware.base import CPU, Accelerator, HardwareProfile
    return HardwareProfile(
        cpu=CPU(name="Apple M3 Pro", cores=12, threads=12, bandwidth_gbs=150),
        ram_bytes=36 * GB,
        accelerators=[Accelerator(name="Apple M3 Pro", vendor="apple",
                                  memory_bytes=27 * GB, bandwidth_gbs=150.0,
                                  fp16_tflops=7.4, unified_memory=True)],
    )


@pytest.fixture
def cpu_only():
    from llmcalculator.hardware.base import CPU, HardwareProfile
    return HardwareProfile(cpu=CPU(name="Ryzen 5600", cores=6, threads=12,
                                   bandwidth_gbs=50.0),
                           ram_bytes=16 * GB, accelerators=[])


# --- weight math ----------------------------------------------------------

def test_q4_weights_match_published_gguf_size():
    """Llama 3.1 8B at Q4_K_M ships as roughly a 4.9 GB GGUF."""
    m = catalog.get("llama3.1:8b")
    size = quant.get("Q4_K_M").weight_bytes(m.params) / 1e9
    assert 4.5 < size < 5.5


def test_fp16_is_two_bytes_per_param():
    m = catalog.get("llama3.1:8b")
    assert quant.get("fp16").weight_bytes(m.params) == pytest.approx(m.params * 2)


# --- KV cache -------------------------------------------------------------

def test_kv_cache_scales_linearly_with_context():
    m = catalog.get("llama3.1:8b")
    assert m.kv_cache_bytes(8192) == pytest.approx(m.kv_cache_bytes(4096) * 2)


def test_gqa_shrinks_kv_cache():
    """Grouped-query attention is the whole reason long context is affordable."""
    gqa = catalog.get("llama3.1:8b")      # 32 heads / 8 kv heads
    mha = catalog.get("phi3:3.8b")        # 32 heads / 32 kv heads
    assert gqa.gqa_ratio == 4.0
    assert mha.gqa_ratio == 1.0


def test_llama8b_kv_at_128k_is_about_16gb():
    m = catalog.get("llama3.1:8b")
    assert 15 < m.kv_cache_bytes(131072) / GB < 18


# --- verdicts -------------------------------------------------------------

def test_8b_q4_fits_on_24gb(rtx4090):
    e = estimate(catalog.get("llama3.1:8b"), rtx4090, quant_name="Q4_K_M")
    assert e.fits and e.verdict == Verdict.EASY


def test_70b_q4_does_not_fit_on_24gb(rtx4090):
    e = estimate(catalog.get("llama3.1:70b"), rtx4090, quant_name="Q4_K_M")
    assert not e.fits
    assert e.total_gb > 35


def test_32b_q4_fits_on_36gb_mac(mac36):
    e = estimate(catalog.get("qwen2.5:32b"), mac36, quant_name="Q4_K_M")
    assert e.fits


def test_verdict_thresholds_are_ordered():
    assert Verdict.from_ratio(50, 100) == Verdict.EASY
    assert Verdict.from_ratio(80, 100) == Verdict.OK
    assert Verdict.from_ratio(95, 100) == Verdict.TIGHT
    assert Verdict.from_ratio(120, 100) == Verdict.NO


# --- workloads ------------------------------------------------------------

def test_training_costs_far_more_than_inference(mac36):
    m = catalog.get("llama3.2:1b")
    infer = estimate(m, mac36, workloads.INFERENCE, "Q4_K_M")
    full = estimate(m, mac36, workloads.FULL_FINETUNE, context=2048)
    assert full.breakdown.total > infer.breakdown.total * 8


def test_qlora_needs_less_than_lora(mac36):
    m = catalog.get("qwen2.5:7b")
    q = estimate(m, mac36, workloads.QLORA, context=2048)
    l = estimate(m, mac36, workloads.LORA, context=2048)
    assert q.breakdown.total < l.breakdown.total


def test_full_finetune_of_7b_does_not_fit_on_consumer_hardware(rtx4090, mac36):
    m = catalog.get("qwen2.5:7b")
    for hw in (rtx4090, mac36):
        assert not estimate(m, hw, workloads.FULL_FINETUNE, context=2048).fits


def test_qlora_base_is_always_four_bit(mac36):
    """QLoRA is defined by its 4-bit base; the picker must not substitute fp16."""
    e = recommended_quant(catalog.get("qwen2.5:7b"), mac36, workloads.QLORA, context=2048)
    assert e.quant_name == "nf4"


def test_max_model_size_ordering(mac36):
    sizes = [max_model_size(mac36, w) for w in
             (workloads.INFERENCE, workloads.QLORA, workloads.LORA, workloads.FULL_FINETUNE)]
    assert sizes == sorted(sizes, reverse=True)


# --- recommendation behaviour --------------------------------------------

def test_recommended_prefers_q4_over_fp16_when_both_fit(mac36):
    """Having room for fp16 is not a reason to use it."""
    e = recommended_quant(catalog.get("llama3.2:3b"), mac36)
    assert e.quant_name == "Q4_K_M"
    assert any("maximum quality" in n for n in e.notes)


def test_unfittable_model_reports_the_default_not_the_worst_quant(mac36):
    e = recommended_quant(catalog.get("llama3.1:405b"), mac36)
    assert e.quant_name == "Q4_K_M"
    assert not e.fits
    assert any("still does not fit" in n for n in e.notes)


# --- performance ----------------------------------------------------------

def test_speed_is_bandwidth_bound(mac36):
    """Halving the weight size should roughly double generation speed."""
    m = catalog.get("llama3.1:8b")
    q8 = estimate(m, mac36, quant_name="Q8_0")
    q4 = estimate(m, mac36, quant_name="Q4_K_M")
    assert q4.tokens_per_sec > q8.tokens_per_sec * 1.4


def test_moe_runs_faster_than_its_total_size(mac36):
    """A 30B MoE with 3B active should beat a 30B dense model comfortably."""
    moe = estimate(catalog.get("qwen3:30b-a3b"), mac36, quant_name="Q4_K_M")
    dense = estimate(catalog.get("qwen2.5:32b"), mac36, quant_name="Q4_K_M")
    assert moe.tokens_per_sec > dense.tokens_per_sec * 3


def test_no_speed_reported_when_it_does_not_fit(rtx4090):
    e = estimate(catalog.get("llama3.1:405b"), rtx4090, quant_name="Q4_K_M")
    assert e.tokens_per_sec == 0


def test_cpu_only_machine_still_produces_estimates(cpu_only):
    e = estimate(catalog.get("llama3.2:3b"), cpu_only, quant_name="Q4_K_M")
    assert e.fits
    assert e.tokens_per_sec > 0


# --- breakdown integrity --------------------------------------------------

def test_breakdown_sums_to_total(mac36):
    e = estimate(catalog.get("llama3.1:8b"), mac36, quant_name="Q4_K_M")
    parts = sum(v for _, v in e.breakdown.items_gb())
    assert parts == pytest.approx(e.total_gb, rel=0.02)


def test_as_dict_is_json_serialisable(mac36):
    import json
    e = estimate(catalog.get("llama3.1:8b"), mac36)
    assert json.loads(json.dumps(e.as_dict()))["model"] == "llama3.1:8b"


def test_context_is_clamped_to_model_maximum(mac36):
    m = catalog.get("gemma2:9b")  # 8k limit
    e = estimate(m, mac36, context=131072)
    assert e.context == m.max_context


def test_unknown_gpu_does_not_fabricate_a_speed_tradeoff():
    """An unrecognised GPU has no bandwidth figure, so speed is unknown. The
    advice must not render that as '0 tok/s instead of 0'."""
    from llmcalculator.hardware import manual
    hw = manual(vram_gb=24, ram_gb=64, gpu_name="Some Unreleased GPU")
    e = recommended_quant(catalog.get("llama3.1:8b"), hw)
    assert e.tokens_per_sec == 0
    assert not any("0 tok/s" in n for n in e.notes)
    assert any("some speed" in n for n in e.notes)
    assert any("not in the database" in n for n in e.notes)


def test_known_gpu_still_reports_the_speed_tradeoff():
    from llmcalculator.hardware import manual
    hw = manual(vram_gb=24, ram_gb=64, gpu_name="RTX 4090")
    e = recommended_quant(catalog.get("llama3.1:8b"), hw)
    assert e.tokens_per_sec > 0
    assert any("tok/s instead of" in n for n in e.notes)
