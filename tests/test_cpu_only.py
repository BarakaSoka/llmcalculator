"""Machines with no GPU at all.

Plenty of people run models on a laptop with integrated graphics or an old
office desktop. Those machines must get real answers, not zeroes and not a
crash. GitHub's CI runners have no GPU either, so this path is exercised on
every push across Linux, macOS and Windows.
"""

import pytest

from llmcalculator import compare, workloads
from llmcalculator.estimate import estimate, max_model_size, recommended_quant
from llmcalculator.hardware.base import GB, CPU, HardwareProfile
from llmcalculator.models import catalog


def cpu_machine(ram_gb: float, cores: int = 4, bandwidth: float = 50.0) -> HardwareProfile:
    return HardwareProfile(
        cpu=CPU(name="Generic x86", cores=cores, threads=cores * 2,
                bandwidth_gbs=bandwidth),
        ram_bytes=int(ram_gb * GB),
        accelerators=[],
    )


@pytest.mark.parametrize("ram", [4, 8, 16, 32, 64])
def test_every_ram_size_produces_a_usable_profile(ram):
    hw = cpu_machine(ram)
    assert not hw.has_gpu
    assert hw.budget_bytes("auto") > 0
    assert hw.budget_bytes("cpu") > 0
    assert hw.summary_lines()


def test_auto_device_falls_back_to_system_ram():
    hw = cpu_machine(16)
    assert hw.budget_bytes("auto") == hw.budget_bytes("cpu")


def test_os_reserve_leaves_room_to_breathe():
    """A machine must not be told it can use every byte it has."""
    for ram in (4, 8, 16, 32, 64):
        hw = cpu_machine(ram)
        assert hw.budget_bytes("cpu") < ram * GB


def test_small_model_runs_on_a_modest_machine():
    e = recommended_quant(catalog.get("qwen2.5:1.5b"), cpu_machine(8))
    assert e.fits
    assert e.tokens_per_sec > 0


def test_speed_is_estimated_without_a_gpu():
    """CPU inference is slower, not immeasurable."""
    e = estimate(catalog.get("llama3.2:3b"), cpu_machine(16), quant_name="Q4_K_M")
    assert e.tokens_per_sec > 0


def test_cpu_is_slower_than_a_gpu_of_the_same_size():
    from llmcalculator.hardware import manual
    model = catalog.get("llama3.1:8b")
    cpu = estimate(model, cpu_machine(32), quant_name="Q4_K_M")
    gpu = estimate(model, manual(vram_gb=24, ram_gb=32, gpu_name="RTX 4090"),
                   quant_name="Q4_K_M")
    assert cpu.tokens_per_sec < gpu.tokens_per_sec


def test_recommendations_exist_even_on_a_4gb_machine():
    recs = compare.recommend(cpu_machine(4), limit=3)
    assert recs, "a 4 GB machine should still be offered something"
    for r in recs:
        assert r.estimate.fits


def test_asking_for_a_gpu_budget_without_a_gpu_explains_itself():
    """The unhelpful version of this told the user to lower quantization,
    which cannot help when the budget is zero."""
    e = recommended_quant(catalog.get("llama3.2:1b"), cpu_machine(32), device="gpu")
    assert not e.fits
    joined = " ".join(e.notes)
    assert "No GPU was detected" in joined
    assert "--device cpu" in joined
    # and it must not offer irrelevant advice
    assert "Dropping to" not in joined
    assert "smallest quantization" not in joined


def test_training_is_honestly_out_of_reach_on_small_machines():
    hw = cpu_machine(8)
    e = estimate(catalog.get("llama3.1:8b"), hw, workloads.FULL_FINETUNE, context=512)
    assert not e.fits


def test_max_model_size_shrinks_with_ram():
    sizes = [max_model_size(cpu_machine(r), workloads.INFERENCE) for r in (4, 8, 16, 64)]
    assert sizes == sorted(sizes)


def test_largest_usable_applies_a_quality_floor():
    """The headline answer should not be a model crushed to 3 bits."""
    hw = cpu_machine(16)
    floored = compare.largest_that_fits(hw, workloads.INFERENCE)
    unfloored = compare.largest_that_fits(hw, workloads.INFERENCE, min_quality=0.0)
    assert floored is not None and unfloored is not None
    assert floored.quality >= compare.USABLE_QUALITY
    assert unfloored.model.params >= floored.model.params


def test_scan_json_on_a_gpuless_machine(tmp_path):
    """The shape CI asserts on every runner."""
    from llmcalculator.cli import _scan_dict
    d = _scan_dict(cpu_machine(16))
    assert d["accelerators"] == []
    assert d["budgets_gb"]["gpu"] == 0.0
    assert d["budgets_gb"]["cpu"] > 0
    assert d["capabilities"]["inference"]["max_params_b"] > 0
