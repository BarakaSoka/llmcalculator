"""What a model can do, how it ships, and what will actually run it.

Sizing answers "does it fit". It does not answer "is this the model I want":
a 7B coder and a 7B vision model have identical memory profiles and almost no
overlap in use. These three registries carry the rest of the picture -

  capabilities  what the model was trained to be good at
  formats       how its weights are distributed on disk
  runtimes      the engines that will load those weights

Formats and runtimes are derived rather than hand-listed, because they follow
from the architecture: a standard transformer decoder converts to GGUF, GGUF
is what llama.cpp reads, and Ollama and LM Studio are llama.cpp underneath.
Capabilities cannot be derived from a config file, so the catalog states them
and Hub lookups infer what they can from repository tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, and avoids an import cycle
    from .spec import ModelSpec


@dataclass(frozen=True)
class Trait:
    """One capability, weight format or runtime, with prose to explain it."""

    key: str
    label: str
    description: str

    def as_dict(self) -> Dict[str, str]:
        return {"key": self.key, "label": self.label, "description": self.description}


def _registry(*rows: Tuple[str, str, str]) -> Dict[str, Trait]:
    """Build a registry. Insertion order is display order."""
    return {key: Trait(key, label, desc) for key, label, desc in rows}


# --- capabilities ---------------------------------------------------------

CAPABILITIES: Dict[str, Trait] = _registry(
    ("chat", "Chat", "Instruction-tuned for conversation; follows a chat template out of the box."),
    ("base", "Base model", "Pre-trained only, with no instruction tuning. Completes text rather than answering; expect to fine-tune it."),
    ("reasoning", "Reasoning", "Trained to think step by step before answering. Spends far more output tokens per reply, so generation speed matters more than the raw benchmark suggests."),
    ("code", "Code", "Trained heavily on source code: generation, explanation and refactoring."),
    ("fim", "Fill-in-the-middle", "Supports infilling between a prefix and a suffix, which is what editor autocomplete needs."),
    ("math", "Maths", "Strong on arithmetic, symbolic manipulation and word problems."),
    ("vision", "Vision", "Accepts images alongside text. The vision tower adds memory beyond the language model sized here."),
    ("audio", "Audio", "Accepts speech or audio input alongside text."),
    ("tools", "Tool calling", "Emits structured function calls, so it can drive APIs and tools."),
    ("agentic", "Agentic", "Tuned for multi-step agent loops: planning, calling tools, and reacting to their output."),
    ("rag", "Retrieval", "Tuned for grounded answering over supplied documents, usually with citations."),
    ("multilingual", "Multilingual", "Explicitly trained on many languages rather than English with leakage."),
    ("long-context", "Long context", "Ships with a 128k token window or more. The KV cache, not the weights, is what will stop you using it."),
    ("moe", "Mixture of experts", "Routes each token through a few experts. Fast for its size, but every expert must be resident in memory."),
    ("embedding", "Embeddings", "Produces vectors for search and clustering rather than generated text."),
    ("edge", "Edge", "Small enough for phones, single-board computers and CPU-only machines."),
)

# --- weight formats -------------------------------------------------------

FORMATS: Dict[str, Trait] = _registry(
    ("safetensors", "Safetensors", "The original fp16/bf16 weights as published. Largest download, reference quality, and what every other format is converted from."),
    ("gguf", "GGUF", "The llama.cpp container format, with k-quants from 2 to 8 bits. Runs on CPU, GPU, or a split across both."),
    ("mlx", "MLX", "Apple silicon format, quantized to 4 or 8 bits. Uses unified memory, so system RAM is the budget."),
    ("awq", "AWQ", "Activation-aware 4-bit for GPU serving. Keeps the salient weight channels at higher precision."),
    ("gptq", "GPTQ", "Post-training 4-bit for GPU serving; the older sibling of AWQ and about as accurate."),
    ("exl2", "EXL2", "ExLlamaV2's variable-bitrate format. Fastest single-GPU generation when the whole model fits in VRAM."),
    ("fp8", "FP8", "Native 8-bit floats on Hopper, Ada and MI300. Half the memory of bf16 at near-identical quality."),
    ("mxfp4", "MXFP4", "Microscaling 4-bit, used as the native published format for some newer models rather than as a conversion."),
)

# --- runtimes -------------------------------------------------------------

RUNTIMES: Dict[str, Trait] = _registry(
    ("transformers", "Transformers", "The Hugging Face reference implementation. Runs anything, optimised for nothing."),
    ("llama.cpp", "llama.cpp", "C++ inference over GGUF. The best option when the model does not fit in VRAM, because it can offload layers to the CPU."),
    ("ollama", "Ollama", "A packaging layer over llama.cpp. `ollama run <model>` pulls a sensible GGUF quantization for you."),
    ("lm-studio", "LM Studio", "A desktop GUI over llama.cpp and MLX, for people who would rather not use a terminal."),
    ("vllm", "vLLM", "Paged-attention GPU server. The throughput option when many requests share one model."),
    ("sglang", "SGLang", "GPU server built around prefix caching; strong for agent loops that resend a long shared prompt."),
    ("mlx-lm", "MLX LM", "Apple's runtime for Apple silicon. Reads MLX weights and can also train LoRA adapters."),
    ("tgi", "TGI", "Hugging Face Text Generation Inference, the server behind Hub inference endpoints."),
    ("exllamav2", "ExLlamaV2", "Single-GPU runtime for EXL2 and GPTQ weights, tuned for latency over throughput."),
)

_KINDS = {"capability": CAPABILITIES, "format": FORMATS, "runtime": RUNTIMES}


def trait(kind: str, key: str) -> Trait:
    """Look up one trait. Unknown keys are passed through rather than raising,
    because a Hub tag we have never seen is still worth showing the user."""
    table = _KINDS[kind]
    known = table.get(key.lower().strip())
    if known is not None:
        return known
    return Trait(key, key.replace("-", " ").replace("_", " "), "")


def traits(kind: str, keys: Iterable[str]) -> Tuple[Trait, ...]:
    return tuple(trait(kind, k) for k in keys)


def capability(key: str) -> Trait:
    return trait("capability", key)


def weight_format(key: str) -> Trait:
    return trait("format", key)


def runtime(key: str) -> Trait:
    return trait("runtime", key)


def known(kind: str) -> List[Trait]:
    return list(_KINDS[kind].values())


def _ordered(kind: str, keys: Iterable[str]) -> Tuple[str, ...]:
    """Dedupe, then sort into registry order with unknown keys last."""
    order = list(_KINDS[kind])
    seen, out = set(), []
    for k in keys:
        k = str(k).lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return tuple(sorted(out, key=lambda k: (order.index(k) if k in order else len(order), k)))


# --- inference ------------------------------------------------------------

# Hub and catalog tags that map straight onto a capability.
_TAG_CAPABILITY = {
    "code": "code", "coding": "code", "code-generation": "code",
    "reasoning": "reasoning", "thinking": "reasoning", "chain-of-thought": "reasoning",
    "math": "math", "mathematics": "math",
    "moe": "moe", "mixture-of-experts": "moe",
    "vision": "vision", "multimodal": "vision", "image-text-to-text": "vision",
    "vision-language": "vision", "vlm": "vision",
    "audio": "audio", "speech": "audio", "audio-text-to-text": "audio",
    "tools": "tools", "tool-use": "tools", "function-calling": "tools",
    "agent": "agentic", "agentic": "agentic",
    "rag": "rag", "retrieval": "rag",
    "multilingual": "multilingual",
    "edge": "edge", "tiny": "edge", "on-device": "edge",
    "conversational": "chat", "chat": "chat", "instruct": "chat",
    "fill-in-the-middle": "fim", "fim": "fim",
    "sentence-similarity": "embedding", "feature-extraction": "embedding",
    "long-context": "long-context",
}

# Substrings in a repo id or model name that imply a capability.
_NAME_CAPABILITY = (
    ("coder", "code"), ("codestral", "code"), ("devstral", "code"),
    ("starcoder", "code"), ("codegemma", "code"), ("codellama", "code"),
    ("-r1", "reasoning"), ("magistral", "reasoning"), ("qwq", "reasoning"),
    ("math", "math"),
    ("-vl", "vision"), ("vision", "vision"), ("llava", "vision"),
    ("embed", "embedding"), ("bge", "embedding"), ("gte", "embedding"),
    ("instruct", "chat"), ("-it", "chat"), ("chat", "chat"),
)

LONG_CONTEXT_TOKENS = 131072
EDGE_PARAMS = 4e9


def infer_capabilities(spec: "ModelSpec", extra_tags: Sequence[str] = ()) -> Tuple[str, ...]:
    """Everything we can work out about what a model is for.

    Starts from whatever the catalog states, then adds what the architecture
    and the repository tags make certain. Architecture-derived entries (MoE,
    long context) are always added, because those are facts rather than
    claims - a config file cannot be wrong about its own expert count.
    """
    found = list(spec.capabilities)

    haystack = " ".join([spec.name, spec.hf_id, spec.family]).lower()
    for tag in list(spec.tags) + list(extra_tags):
        mapped = _TAG_CAPABILITY.get(str(tag).lower().strip())
        if mapped:
            found.append(mapped)
    for needle, cap in _NAME_CAPABILITY:
        if needle in haystack:
            found.append(cap)

    if spec.is_moe:
        found.append("moe")
    if spec.max_context >= LONG_CONTEXT_TOKENS:
        found.append("long-context")
    if spec.params <= EDGE_PARAMS:
        found.append("edge")
    if "code" in found and "fim" not in found and "code" in haystack:
        found.append("fim")
    if "base" in found:
        found = [c for c in found if c != "chat"]
    return _ordered("capability", found)


def infer_formats(spec: "ModelSpec", extra_tags: Sequence[str] = ()) -> Tuple[str, ...]:
    """Which weight formats this model is realistically available in.

    Safetensors is a given for anything with a `config.json`. GGUF and MLX
    follow from being a standard transformer decoder, which is the only thing
    this estimator will size in the first place - conversion tooling covers
    that whole class. Everything else has to be evidenced by a tag, because
    community GPU quantizations exist for popular models and not for the rest.
    """
    found = list(spec.formats) or ["safetensors", "gguf", "mlx"]
    if "safetensors" not in found:
        found.append("safetensors")
    for tag in list(spec.tags) + list(extra_tags):
        t = str(tag).lower().strip()
        for name in ("gguf", "mlx", "awq", "gptq", "exl2", "fp8", "mxfp4"):
            if name in t:
                found.append(name)
    return _ordered("format", found)


_FORMAT_RUNTIMES = {
    "safetensors": ("transformers", "vllm", "sglang", "tgi"),
    "gguf": ("llama.cpp", "ollama", "lm-studio"),
    "mlx": ("mlx-lm", "lm-studio"),
    "awq": ("vllm", "sglang"),
    "gptq": ("vllm", "exllamav2"),
    "exl2": ("exllamav2",),
    "fp8": ("vllm", "sglang"),
    "mxfp4": ("vllm", "transformers"),
}


def infer_runtimes(spec: "ModelSpec", formats: Sequence[str] = ()) -> Tuple[str, ...]:
    """Engines that can load this model, derived from its weight formats."""
    if spec.runtimes:
        return _ordered("runtime", spec.runtimes)
    found = ["transformers"]
    for fmt in (formats or infer_formats(spec)):
        found.extend(_FORMAT_RUNTIMES.get(fmt, ()))
    return _ordered("runtime", found)


def support_notes(spec: "ModelSpec", capabilities: Sequence[str] = (),
                  formats: Sequence[str] = ()) -> Tuple[str, ...]:
    """Caveats worth stating before someone downloads 40 GB of weights."""
    caps = set(capabilities or infer_capabilities(spec))
    out: List[str] = []

    if "vision" in caps:
        out.append("Sized here as a language model only. The vision tower and image "
                   "features add memory on top, and GGUF builds often ship the text "
                   "tower alone unless a separate mmproj file is published.")
    if "reasoning" in caps:
        out.append("Reasoning models emit long hidden thinking before the answer, so "
                   "plan for several thousand extra output tokens per reply and a "
                   "correspondingly larger KV cache.")
    if "moe" in caps:
        out.append("Mixture of experts: only {:.1f}B parameters are active per token, "
                   "but all {:.1f}B must be resident, so memory tracks the total and "
                   "speed tracks the active count.".format(
                       spec.active_params / 1e9, spec.params_b))
    if "long-context" in caps:
        out.append("The advertised {}k window is rarely the practical one - KV cache "
                   "grows linearly with context, so check the context table before "
                   "assuming you can fill it.".format(spec.max_context // 1024))
    if "embedding" in caps:
        out.append("An embedding model. Generation speed and quantization quality "
                   "figures below describe text generation and do not apply.")
    if spec.license and "nc" in spec.license.lower().split("-"):
        out.append("The {} license is non-commercial.".format(spec.license))
    return tuple(out)


# --- the bundle the UIs actually use --------------------------------------

@dataclass(frozen=True)
class SupportProfile:
    """Everything non-numeric that is worth knowing about a model."""

    capabilities: Tuple[Trait, ...] = ()
    formats: Tuple[Trait, ...] = ()
    runtimes: Tuple[Trait, ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def capability_keys(self) -> Tuple[str, ...]:
        return tuple(t.key for t in self.capabilities)

    @property
    def format_keys(self) -> Tuple[str, ...]:
        return tuple(t.key for t in self.formats)

    @property
    def runtime_keys(self) -> Tuple[str, ...]:
        return tuple(t.key for t in self.runtimes)

    def summary(self, kind: str = "capability", limit: int = 0) -> str:
        """A one-line comma-joined list, for tables and chips."""
        items = {"capability": self.capabilities, "format": self.formats,
                 "runtime": self.runtimes}[kind]
        labels = [t.label for t in items]
        if limit and len(labels) > limit:
            return ", ".join(labels[:limit]) + " +{}".format(len(labels) - limit)
        return ", ".join(labels)

    def as_dict(self) -> Dict[str, object]:
        return {
            "capabilities": [t.as_dict() for t in self.capabilities],
            "formats": [t.as_dict() for t in self.formats],
            "runtimes": [t.as_dict() for t in self.runtimes],
            "notes": list(self.notes),
        }


def profile(spec: "ModelSpec", extra_tags: Sequence[str] = ()) -> SupportProfile:
    """Resolve a model's capabilities, formats, runtimes and caveats."""
    caps = infer_capabilities(spec, extra_tags)
    fmts = infer_formats(spec, extra_tags)
    return SupportProfile(
        capabilities=traits("capability", caps),
        formats=traits("format", fmts),
        runtimes=traits("runtime", infer_runtimes(spec, fmts)),
        notes=support_notes(spec, caps, fmts),
    )
