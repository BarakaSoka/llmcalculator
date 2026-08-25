"""llmcalculator - work out which AI models your machine can run.

    from llmcalculator import detect, check
    print(check("llama3.1:8b").label())
"""

from __future__ import annotations

__version__ = "0.2.2"

from .hardware import detect, manual, HardwareProfile, Accelerator, CPU
from .models import ModelSpec, catalog
from . import estimate as estimate_module
from .estimate import (
    Estimate, Verdict, estimate, best_quant, max_model_size, sweep,
)
from . import quant, workloads, compare


def check(model, workload="inference", hardware=None, **kw) -> Estimate:
    """Size a model on this machine. The one-liner entry point.

    >>> check("llama3.1:8b").fits
    True
    >>> check("llama3.1:70b", "qlora").label()
    "Won't fit"
    """
    hw = hardware or detect()
    spec = model if isinstance(model, ModelSpec) else catalog.get(model)
    wl = workloads.get(workload) if isinstance(workload, str) else workload
    return estimate(spec, hw, wl, **kw)


__all__ = [
    "__version__", "check", "detect", "manual", "estimate", "best_quant",
    "max_model_size", "sweep", "compare", "catalog", "quant", "workloads",
    "HardwareProfile", "Accelerator", "CPU", "ModelSpec", "Estimate", "Verdict",
]
