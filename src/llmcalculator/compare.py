"""Comparing models against each other, and picking good ones for a machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from . import quant, workloads
from .estimate import (
    Estimate, Verdict, best_quant, estimate as estimate_one, recommended_quant,
)
from .hardware.base import HardwareProfile
from .models import catalog
from .models.spec import ModelSpec
from .workloads import Workload


@dataclass
class Recommendation:
    estimate: Estimate
    score: float
    reason: str


def compare(names: Sequence[str], hardware: HardwareProfile,
            workload: Workload = workloads.INFERENCE,
            quant_name: Optional[str] = None, context: int = 8192,
            device: str = "auto") -> List[Estimate]:
    """Size several named models side by side under identical settings."""
    out = []
    for name in names:
        try:
            model = catalog.get(name)
        except KeyError:
            if "/" in name:
                model = catalog.from_hf(name)
            else:
                raise
        out.append(estimate_one(model, hardware, workload, quant_name,
                                    context=context, device=device))
    return out


def recommend(hardware: HardwareProfile, workload: Workload = workloads.INFERENCE,
              context: int = 8192, limit: int = 10, device: str = "auto",
              tag: Optional[str] = None,
              min_tokens_per_sec: float = 0.0) -> List[Recommendation]:
    """Rank catalog models by how well they suit this machine.

    The score rewards using the machine well: a model that leaves 20 GB idle
    is scored below one that fills the budget, and quality loss from heavy
    quantization is penalised. Speed acts as a floor, not a goal, because past
    conversational speed extra tokens per second stop mattering.
    """
    results: List[Recommendation] = []

    for model in catalog.all_models():
        # A tag and a capability are the same thing to someone at the command
        # line asking for "code" - match either.
        if tag and tag not in model.tags and not model.has_capability(tag):
            continue

        # Score every quantization and keep the best-scoring one, rather than
        # assuming the highest precision that fits is the right choice. For a
        # small model on a large machine it usually is not.
        candidates = []
        for e in quant_table(model, hardware, workload, context=context, device=device):
            if not e.fits:
                continue
            if min_tokens_per_sec and e.tokens_per_sec < min_tokens_per_sec:
                continue
            candidates.append((e, _score(e)))
        if not candidates:
            continue
        best, (score, reason) = max(candidates, key=lambda c: c[1][0])
        results.append(Recommendation(best, score, reason))

    results.sort(key=lambda r: -r.score)
    return results[:limit]


def _score(e: Estimate) -> tuple:
    """Score an estimate 0-100 and explain the verdict in one clause."""
    # Capability: bigger models are better, with diminishing returns.
    import math
    capability = math.log10(max(e.model.params_b, 0.1)) / math.log10(500) * 60

    # Quality retained after quantization, weighted heavily below ~0.97.
    quality = (e.quality ** 8) * 20

    # Speed: full marks above 20 tok/s, falling off sharply below 8.
    if e.tokens_per_sec >= 20:
        speed = 15.0
    elif e.tokens_per_sec > 0:
        speed = 15.0 * (e.tokens_per_sec / 20.0) ** 0.6
    else:
        speed = 7.0

    # Headroom: penalise both wasted memory and a dangerously tight fit.
    util = e.utilization
    if util > 0.92:
        headroom = 1.0
    elif util > 0.55:
        headroom = 5.0
    else:
        headroom = 5.0 * (util / 0.55)

    score = capability + quality + speed + headroom

    if e.workload.key != "inference":
        # Training runs are not scored on generation speed; headroom is what matters.
        if util > 0.85:
            reason = "fits, but only just"
        elif e.model.is_moe:
            reason = "MoE - expert routing makes training less predictable"
        else:
            reason = "trains with {:.0f} GB to spare".format(max(e.headroom_gb, 0))
    elif e.model.is_moe:
        reason = "MoE - {:.0f}B of knowledge at {:.1f}B of speed".format(
            e.model.params_b, e.model.active_params / 1e9)
    elif e.quality >= 0.99 and e.tokens_per_sec >= 20:
        reason = "fast and near-lossless"
    elif util > 0.85:
        reason = "uses nearly all your memory"
    elif e.tokens_per_sec >= 30:
        reason = "very fast, room to spare"
    elif e.tokens_per_sec < 8:
        reason = "large but slow"
    else:
        reason = "balanced fit"
    return score, reason


def quant_table(model: ModelSpec, hardware: HardwareProfile,
                workload: Workload = workloads.INFERENCE,
                context: int = 8192, device: str = "auto") -> List[Estimate]:
    """One row per quantization format, for a single model."""
    if workload.key != "inference":
        # A training workload's base precision is part of its definition.
        return [estimate_one(model, hardware, workload, workload.default_base_quant,
                             context=context, device=device)]
    return [estimate_one(model, hardware, workload, f.name, context=context, device=device)
            for f in quant.ladder()]


def workload_table(model: ModelSpec, hardware: HardwareProfile,
                   context: int = 4096, device: str = "auto") -> List[Estimate]:
    """One row per workload, each at its own natural base precision."""
    out = []
    for wl in workloads.ALL:
        ctx = context if wl.key == "inference" else min(context, 2048)
        out.append(estimate_one(model, hardware, wl, context=ctx, device=device))
    return out


# Below this, quantization damage is severe enough that a smaller model at
# higher precision is almost always the better answer. Headline "largest that
# fits" figures stay above it so they are advice, not just arithmetic.
USABLE_QUALITY = 0.95


def largest_that_fits(hardware: HardwareProfile, workload: Workload,
                      context: int = 8192, device: str = "auto",
                      min_quality: float = USABLE_QUALITY) -> Optional[Estimate]:
    """The biggest catalog model this machine can usefully run for a workload.

    Deliberately not the biggest that fits at any precision. A 24B squeezed
    into 11 GB at 3 bits technically fits, but it is worse than a 9B at 5 bits
    and quoting it as the headline answer misleads.
    """
    best: Optional[Estimate] = None
    for model in sorted(catalog.all_models(), key=lambda m: -m.params):
        e = best_quant(model, hardware, workload, min_quality=min_quality,
                       context=context, device=device)
        if e and e.fits:
            if best is None or e.model.params > best.model.params:
                best = e
    return best
