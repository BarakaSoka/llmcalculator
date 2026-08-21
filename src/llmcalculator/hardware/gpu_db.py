"""Bandwidth and throughput figures for common accelerators.

Vendor tools report a GPU's name and memory but not its bandwidth, and
bandwidth is what actually decides token generation speed. This table fills
that gap. Values are manufacturer specifications: memory bandwidth in GB/s
and dense (non-sparse) fp16 tensor throughput in TFLOPS.
"""

from __future__ import annotations

from typing import Optional, Tuple

# name fragment -> (bandwidth GB/s, fp16 TFLOPS)
GPU_SPECS = {
    # NVIDIA datacenter
    "b200": (8000.0, 2250.0), "h200": (4800.0, 989.0), "h100": (3350.0, 989.0),
    "a100": (2039.0, 312.0), "l40s": (864.0, 362.0), "l40": (864.0, 181.0),
    "a40": (696.0, 149.0), "a10g": (600.0, 125.0), "a10": (600.0, 125.0),
    "v100": (900.0, 125.0), "t4": (320.0, 65.0),
    # NVIDIA RTX 50
    "rtx 5090": (1792.0, 419.0), "rtx 5080": (960.0, 225.0),
    "rtx 5070 ti": (896.0, 176.0), "rtx 5070": (672.0, 123.0),
    # NVIDIA RTX 40
    "rtx 4090": (1008.0, 165.0), "rtx 4080 super": (736.0, 104.0), "rtx 4080": (717.0, 98.0),
    "rtx 4070 ti super": (672.0, 88.0), "rtx 4070 ti": (504.0, 80.0),
    "rtx 4070 super": (504.0, 71.0), "rtx 4070": (504.0, 58.0),
    "rtx 4060 ti": (288.0, 44.0), "rtx 4060": (272.0, 30.0),
    # NVIDIA RTX 30
    "rtx 3090 ti": (1008.0, 80.0), "rtx 3090": (936.0, 71.0),
    "rtx 3080 ti": (912.0, 68.0), "rtx 3080": (760.0, 59.0),
    "rtx 3070": (448.0, 40.0), "rtx 3060 ti": (448.0, 32.0), "rtx 3060": (360.0, 25.0),
    # NVIDIA workstation / older
    "rtx a6000": (768.0, 155.0), "rtx a5000": (768.0, 111.0), "rtx 6000 ada": (960.0, 182.0),
    "rtx 2080 ti": (616.0, 27.0), "gtx 1080 ti": (484.0, 11.0),
    # Apple Silicon
    "m1": (68.0, 2.6), "m1 pro": (200.0, 5.2), "m1 max": (400.0, 10.4), "m1 ultra": (800.0, 21.0),
    "m2": (100.0, 3.6), "m2 pro": (200.0, 6.8), "m2 max": (400.0, 13.6), "m2 ultra": (800.0, 27.2),
    "m3": (100.0, 4.1), "m3 pro": (150.0, 7.4), "m3 max": (400.0, 14.2), "m3 ultra": (800.0, 28.4),
    "m4": (120.0, 4.6), "m4 pro": (273.0, 9.2), "m4 max": (546.0, 18.4),
    "m5": (153.0, 6.0), "m5 pro": (300.0, 12.0), "m5 max": (600.0, 24.0),
    # AMD
    "mi300x": (5300.0, 1307.0), "mi250x": (3277.0, 383.0),
    "rx 7900 xtx": (960.0, 123.0), "rx 7900 xt": (800.0, 103.0),
    "rx 7800 xt": (624.0, 74.0), "rx 6900 xt": (512.0, 46.0), "rx 6800": (512.0, 32.0),
    # Intel
    "arc a770": (560.0, 39.0), "arc b580": (456.0, 46.0),
}

# Longest keys first so "rtx 4080 super" wins over "rtx 4080".
_ORDERED = sorted(GPU_SPECS.items(), key=lambda kv: -len(kv[0]))


def lookup(name: str) -> Tuple[float, float]:
    """Best-effort (bandwidth GB/s, fp16 TFLOPS) for a GPU name.

    Returns (0.0, 0.0) when the name is not recognised; callers should treat
    that as "speed unknown" rather than "slow".
    """
    n = " ".join(name.lower().replace("-", " ").split())
    for key, spec in _ORDERED:
        if key in n:
            return spec
    return (0.0, 0.0)


def cpu_bandwidth(arch: str, name: str = "") -> float:
    """Rough system-RAM bandwidth in GB/s for CPU-only inference."""
    n = (name + " " + arch).lower()
    if "apple" in n or "arm" in n:
        return 100.0
    if "xeon" in n or "epyc" in n or "threadripper" in n:
        return 100.0  # multi-channel server memory
    return 50.0  # typical dual-channel DDR4/DDR5 desktop
