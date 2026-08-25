"""The estimator: does this model fit on this machine, for this workload?

Every number is built from the model's real architecture rather than a
rule of thumb, then compared against a memory budget that already excludes
what the OS needs. The output carries its own breakdown so a user can see
which component is the problem when something does not fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import quant, workloads
from .hardware.base import GB, HardwareProfile
from .models.spec import ModelSpec
from .workloads import Workload

GB_F = float(GB)


class Verdict:
    """How well something fits, ordered worst to best."""

    NO = "no"
    TIGHT = "tight"
    OK = "ok"
    EASY = "easy"

    ORDER = [NO, TIGHT, OK, EASY]
    LABEL = {NO: "Won't fit", TIGHT: "Tight", OK: "Fits", EASY: "Comfortable"}
    SYMBOL = {NO: "x", TIGHT: "!", OK: "+", EASY: "*"}

    @staticmethod
    def from_ratio(used: float, budget: float) -> str:
        if budget <= 0:
            return Verdict.NO
        r = used / budget
        if r > 1.0:
            return Verdict.NO
        if r > 0.90:
            return Verdict.TIGHT
        if r > 0.70:
            return Verdict.OK
        return Verdict.EASY


@dataclass
class Breakdown:
    """Where the memory goes."""

    weights: float = 0.0
    kv_cache: float = 0.0
    gradients: float = 0.0
    optimizer: float = 0.0
    activations: float = 0.0
    overhead: float = 0.0

    @property
    def total(self) -> float:
        return (self.weights + self.kv_cache + self.gradients
                + self.optimizer + self.activations + self.overhead)

    def items_gb(self) -> List:
        out = [("Weights", self.weights / GB_F)]
        if self.kv_cache:
            out.append(("KV cache", self.kv_cache / GB_F))
        if self.gradients:
            out.append(("Gradients", self.gradients / GB_F))
        if self.optimizer:
            out.append(("Optimizer", self.optimizer / GB_F))
        if self.activations:
            out.append(("Activations", self.activations / GB_F))
        out.append(("Overhead", self.overhead / GB_F))
        return [(k, v) for k, v in out if v > 0.005]


@dataclass
class Estimate:
    """A single (model, workload, quantization, context) sizing result."""

    model: ModelSpec
    workload: Workload
    quant_name: str
    context: int
    batch: int
    device: str

    breakdown: Breakdown
    budget_bytes: float
    verdict: str

    tokens_per_sec: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    disk_gb: float = 0.0
    quality: float = 1.0
    notes: List[str] = field(default_factory=list)

    @property
    def total_gb(self) -> float:
        return self.breakdown.total / GB_F

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / GB_F

    @property
    def headroom_gb(self) -> float:
        return (self.budget_bytes - self.breakdown.total) / GB_F

    @property
    def utilization(self) -> float:
        return self.breakdown.total / self.budget_bytes if self.budget_bytes else float("inf")

    @property
    def fits(self) -> bool:
        return self.verdict != Verdict.NO

    def label(self) -> str:
        return Verdict.LABEL[self.verdict]

    def as_dict(self) -> Dict:
        return {
            "model": self.model.name,
            "params_b": round(self.model.params_b, 2),
            "workload": self.workload.key,
            "quant": self.quant_name,
            "context": self.context,
            "batch": self.batch,
            "device": self.device,
            "verdict": self.verdict,
            "fits": self.fits,
            "required_gb": round(self.total_gb, 2),
            "budget_gb": round(self.budget_gb, 2),
            "headroom_gb": round(self.headroom_gb, 2),
            "utilization": round(self.utilization, 3),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "prefill_tokens_per_sec": round(self.prefill_tokens_per_sec, 0),
            "disk_gb": round(self.disk_gb, 2),
            "quality": round(self.quality, 3),
            "breakdown_gb": {k: round(v, 3) for k, v in self.breakdown.items_gb()},
            "notes": list(self.notes),
        }


# --- core ------------------------------------------------------------------

def estimate(
    model: ModelSpec,
    hardware: HardwareProfile,
    workload: Workload = workloads.INFERENCE,
    quant_name: Optional[str] = None,
    context: Optional[int] = None,
    batch: int = 1,
    device: str = "auto",
    kv_quant: str = "fp16",
    gradient_checkpointing: bool = True,
) -> Estimate:
    """Size one model for one workload on one machine."""
    fmt = quant.get(quant_name or workload.default_base_quant)
    context = context or min(model.max_context, 8192 if workload.key == "inference" else 2048)
    context = min(context, model.max_context)

    b = Breakdown()
    b.weights = fmt.weight_bytes(model.params)
    b.overhead = workload.overhead_bytes

    if workload.key == "inference":
        b.kv_cache = model.kv_cache_bytes(context, batch, quant.kv_cache_bytes_per_element(kv_quant))
        # llama.cpp-style compute buffer, scales with context and width
        b.activations = 0.15e9 + context * model.hidden_size * 4 * batch / 1e3
    else:
        trainable = model.params * workload.trainable_fraction
        b.gradients = trainable * workload.grad_bytes
        b.optimizer = trainable * workload.optimizer_bytes
        b.activations = model.activation_bytes(context, batch, gradient_checkpointing)
        # a training step still caches K/V for the forward pass
        b.kv_cache = model.kv_cache_bytes(context, batch, 2.0) * 0.5

    budget = float(hardware.budget_bytes(device))
    verdict = Verdict.from_ratio(b.total, budget)

    est = Estimate(
        model=model, workload=workload, quant_name=fmt.name, context=context,
        batch=batch, device=device, breakdown=b, budget_bytes=budget, verdict=verdict,
        disk_gb=fmt.weight_bytes(model.params) / 1e9,
        quality=fmt.quality,
    )
    _add_performance(est, hardware, fmt)
    _add_notes(est, hardware, fmt, workload)
    return est


def _add_performance(est: Estimate, hw: HardwareProfile, fmt: quant.QuantFormat) -> None:
    """Estimate generation and prefill speed.

    Token generation is memory-bandwidth bound: each token reads every active
    weight once. Prefill is compute bound and uses roughly 2 FLOPs per
    parameter per token.
    """
    accel = hw.primary
    on_cpu = est.device == "cpu" or accel is None or accel.vendor == "cpu"

    if on_cpu:
        bandwidth = hw.cpu.bandwidth_gbs or 50.0
        efficiency = 0.35
        tflops = max(hw.cpu.cores * 0.05, 0.2)
    else:
        bandwidth = accel.bandwidth_gbs
        efficiency = 0.75 if accel.unified_memory else 0.80
        tflops = accel.fp16_tflops

    active_bytes = fmt.weight_bytes(est.model.active_params) / 1e9
    if est.workload.key == "inference" and bandwidth > 0 and active_bytes > 0:
        kv_read = est.breakdown.kv_cache / 1e9 * 0.5
        est.tokens_per_sec = (bandwidth * efficiency) / (active_bytes + kv_read)

    if tflops > 0 and est.model.active_params > 0:
        flops_per_token = 2 * est.model.active_params
        est.prefill_tokens_per_sec = (tflops * 1e12 * 0.35) / flops_per_token

    if not est.fits:
        est.tokens_per_sec = 0.0
        est.prefill_tokens_per_sec = 0.0


def _add_notes(est: Estimate, hw: HardwareProfile, fmt: quant.QuantFormat,
               wl: Workload) -> None:
    """Attach the advice that actually helps, and nothing else."""
    b = est.breakdown

    if est.device in ("gpu", "all-gpus") and not hw.has_gpu:
        # Nothing about quantization or context helps here, and offering that
        # advice would send someone chasing a fix for the wrong problem.
        est.notes.append(
            "No GPU was detected, so the GPU budget is zero and nothing can fit. "
            "This machine can still run models on the CPU: drop --device, or pass "
            "--device cpu, to size against your {:.0f} GB of system RAM.".format(hw.ram_gb))
        return

    if not est.fits:
        over = (b.total - est.budget_bytes) / GB_F
        est.notes.append("Over budget by {:.1f} GB.".format(over))
        if b.kv_cache > b.weights * 0.4:
            est.notes.append(
                "KV cache is {:.1f} GB of that. Halving context to {} would save "
                "about {:.1f} GB.".format(b.kv_cache / GB_F, est.context // 2,
                                          b.kv_cache / GB_F / 2))
        if wl.key == "inference":
            cheaper = _next_smaller_format(fmt)
            if cheaper:
                saved = (fmt.bytes_per_weight - cheaper.bytes_per_weight) * est.model.params / GB_F
                est.notes.append(
                    "Dropping to {} would save about {:.1f} GB.".format(cheaper.name, saved))
        elif wl.key == "full":
            est.notes.append("LoRA or QLoRA on this model needs far less memory - try those.")
        elif wl.key == "lora":
            est.notes.append("QLoRA quantizes the frozen base to 4-bit and usually fits where LoRA does not.")

    if est.verdict == Verdict.TIGHT:
        est.notes.append("Fits, but with little headroom. Expect swapping if anything else is running.")

    if wl.key == "train":
        est.notes.append(
            "Memory is only half the problem: pre-training a {:.0f}B model needs trillions "
            "of tokens and GPU-months. Treat a local run as an experiment, not a real "
            "pre-train.".format(est.model.params_b))

    if est.fits and est.tokens_per_sec and est.tokens_per_sec < 5:
        est.notes.append(
            "At about {:.0f} tok/s this will feel slow for interactive chat, though it is "
            "fine for batch jobs.".format(est.tokens_per_sec))

    if est.model.is_moe and wl.key == "inference":
        est.notes.append(
            "Mixture-of-experts: all {:.0f}B parameters must be resident, but only {:.1f}B "
            "are read per token, so it runs much faster than its size suggests.".format(
                est.model.params_b, est.model.active_params / 1e9))

    if hw.disk_free_bytes and est.disk_gb > hw.disk_free_gb:
        est.notes.append("Not enough free disk: needs {:.0f} GB, {:.0f} GB available.".format(
            est.disk_gb, hw.disk_free_gb))

    if est.fits and not est.tokens_per_sec and wl.key == "inference":
        accel = hw.primary
        name = accel.name if accel else "this device"
        est.notes.append(
            "Speed is not estimated: memory bandwidth for {} is not in the "
            "database. Pass --gpu-name to compare against a known GPU.".format(name))

    if fmt.quality < 0.95:
        est.notes.append("{} loses noticeable quality; prefer a smaller model at "
                         "higher precision if you can.".format(fmt.name))


def _next_smaller_format(fmt: quant.QuantFormat) -> Optional[quant.QuantFormat]:
    ladder = quant.ladder()
    for i, f in enumerate(ladder):
        if f.name == fmt.name and i + 1 < len(ladder):
            return ladder[i + 1]
    return None


# --- higher-level queries --------------------------------------------------

def best_quant(model: ModelSpec, hardware: HardwareProfile,
               workload: Workload = workloads.INFERENCE,
               min_quality: float = 0.0, **kw) -> Optional[Estimate]:
    """The highest-quality quantization of this model that still fits.

    Only inference gets to search the ladder. A training workload fixes its own
    base precision by definition - QLoRA means a 4-bit frozen base, LoRA means a
    16-bit one - so substituting a different format would describe a different
    technique, not a cheaper version of the same one.
    """
    if workload.key != "inference":
        est = estimate(model, hardware, workload, workload.default_base_quant, **kw)
        return est if est.fits else None
    for fmt in quant.ladder():
        if fmt.quality < min_quality:
            continue
        est = estimate(model, hardware, workload, fmt.name, **kw)
        if est.fits:
            return est
    return None


def max_model_size(hardware: HardwareProfile, workload: Workload,
                   quant_name: Optional[str] = None, context: int = 4096,
                   device: str = "auto") -> float:
    """Largest parameter count that fits, in billions.

    Solves the memory equation for parameter count using a representative
    dense architecture, then returns the result rounded to something sane.
    """
    fmt = quant.get(quant_name or workload.default_base_quant)
    budget = float(hardware.budget_bytes(device)) - workload.overhead_bytes
    if budget <= 0:
        return 0.0

    per_param = fmt.bytes_per_weight
    if workload.key != "inference":
        per_param += workload.trainable_fraction * (workload.grad_bytes + workload.optimizer_bytes)

    # Reserve a slice of the budget for context-dependent memory.
    reserve = 0.20 if workload.key == "inference" else 0.35
    usable = budget * (1 - reserve)
    return max(0.0, usable / per_param / 1e9)


def sweep(model: ModelSpec, hardware: HardwareProfile,
          contexts: Optional[List[int]] = None, **kw) -> List[Estimate]:
    """The same model across a range of context lengths."""
    contexts = contexts or [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    out = []
    for ctx in contexts:
        if ctx > model.max_context:
            break
        out.append(estimate(model, hardware, context=ctx, **kw))
    return out


def recommended_quant(model: ModelSpec, hardware: HardwareProfile,
                      workload: Workload = workloads.INFERENCE, **kw) -> Estimate:
    """The format most people should actually use.

    Distinct from `best_quant`, which returns the highest precision that fits.
    Having room for fp16 is not a reason to run fp16: it costs three times the
    memory of Q4_K_M and runs three times slower for a quality difference most
    users cannot detect. So this starts at the sensible default and only moves
    down the ladder when that does not fit.
    """
    if workload.key != "inference":
        return estimate(model, hardware, workload, workload.default_base_quant, **kw)

    ladder = quant.ladder()
    names = [f.name for f in ladder]
    start = names.index(quant.DEFAULT_INFERENCE) if quant.DEFAULT_INFERENCE in names else 0

    smallest = None
    for fmt in ladder[start:]:
        est = estimate(model, hardware, workload, fmt.name, **kw)
        if est.fits:
            _note_higher_precision(est, model, hardware, workload, ladder, **kw)
            return est
        smallest = est

    # Nothing fits. Report the sensible default rather than the most degraded
    # option, so the headline number is the one the user would actually face.
    fallback = estimate(model, hardware, workload, ladder[start].name, **kw)
    no_gpu = fallback.device in ("gpu", "all-gpus") and not hardware.has_gpu
    if not no_gpu and smallest is not None and smallest.quant_name != fallback.quant_name:
        fallback.notes.append(
            "Even {} - the smallest quantization worth using - needs {:.1f} GB and "
            "still does not fit.".format(smallest.quant_name, smallest.total_gb))
    return fallback


def _note_higher_precision(est: Estimate, model: ModelSpec, hardware: HardwareProfile,
                           workload: Workload, ladder, **kw) -> None:
    """Mention spare capacity for a better format, without defaulting to it."""
    idx = [f.name for f in ladder].index(est.quant_name)
    for fmt in ladder[:idx]:  # highest quality first
        higher = estimate(model, hardware, workload, fmt.name, **kw)
        if higher.fits:
            # Speed is unknown for unrecognised GPUs; do not dress a pair of
            # zeroes up as a tradeoff.
            if est.tokens_per_sec and higher.tokens_per_sec:
                est.notes.append(
                    "You have room for {} ({:.1f} GB) if you want maximum quality, at "
                    "about {:.0f} tok/s instead of {:.0f}.".format(
                        fmt.name, higher.total_gb, higher.tokens_per_sec,
                        est.tokens_per_sec))
            else:
                est.notes.append(
                    "You have room for {} ({:.1f} GB) if you want maximum quality, "
                    "at the cost of some speed.".format(fmt.name, higher.total_gb))
            return
