"""Hardware detection and budget arithmetic."""

import pytest

from llmcalculator.hardware import detect, manual
from llmcalculator.hardware.base import GB, CPU, Accelerator, HardwareProfile
from llmcalculator.hardware import gpu_db


def test_detect_returns_a_usable_profile():
    """Detection must never raise, whatever the machine."""
    hw = detect()
    assert hw.ram_bytes > 0
    assert hw.cpu.cores >= 1
    assert hw.budget_bytes("auto") > 0
    assert isinstance(hw.summary_lines(), list)


def test_manual_profile():
    hw = manual(vram_gb=24, ram_gb=64, gpu_name="RTX 4090")
    assert hw.primary.memory_gb == pytest.approx(24)
    assert hw.primary.bandwidth_gbs == 1008.0


def test_gpu_db_prefers_longest_match():
    """'RTX 4080 Super' must not resolve to the plain 4080."""
    assert gpu_db.lookup("NVIDIA GeForce RTX 4080 SUPER")[0] == 736.0
    assert gpu_db.lookup("NVIDIA GeForce RTX 4080")[0] == 717.0


def test_gpu_db_unknown_returns_zeros():
    assert gpu_db.lookup("Some Future GPU 9000") == (0.0, 0.0)


def test_cpu_budget_reserves_room_for_the_os():
    hw = HardwareProfile(cpu=CPU("test", 8, 16), ram_bytes=16 * GB)
    assert hw.budget_bytes("cpu") < 16 * GB
    assert hw.budget_bytes("cpu") > 10 * GB


def test_multi_gpu_total():
    accels = [Accelerator("A", "nvidia", 24 * GB), Accelerator("B", "nvidia", 24 * GB)]
    hw = HardwareProfile(cpu=CPU("x", 8, 16), ram_bytes=64 * GB, accelerators=accels)
    assert hw.budget_bytes("all-gpus") == 48 * GB
    assert hw.budget_bytes("gpu") == 24 * GB


def test_machine_with_no_gpu():
    hw = HardwareProfile(cpu=CPU("x", 8, 16), ram_bytes=16 * GB)
    assert not hw.has_gpu
    assert hw.budget_bytes("auto") == hw.budget_bytes("cpu")


def test_bf16_support_by_compute_capability():
    ampere = Accelerator("RTX 3090", "nvidia", 24 * GB, compute_capability="8.6")
    turing = Accelerator("RTX 2080 Ti", "nvidia", 11 * GB, compute_capability="7.5")
    assert ampere.supports_bf16()
    assert not turing.supports_bf16()
