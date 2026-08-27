# llmcalculator  
Work out which AI model architectures can actually run on your computer/machine — for inference,
fine-tuning, and training.



<img width="884" height="682" alt="Screenshot 2026-08-25 at 15 55 50" src="https://github.com/user-attachments/assets/2ffa57cd-692f-4bab-973b-d27d9473cd6e" />

---


It reads your real hardware, then sizes each model from its **actual
architecture** rather than a rule of thumb. That distinction matters: two 7B
models can differ by 8x in KV cache at long context, which is usually what
decides whether something fits.

```
$ llmcalculator scan

Your machine
  CPU        Apple M3 Pro (12 cores / 12 threads)
  RAM        36.0 GB
  GPU        Apple M3 Pro - 27.0 GB, 150 GB/s (unified)

What this machine can do
 Workload             Max size   Precision   Largest that fits
 ─────────────────────────────────────────────────────────────
 Inference            ~36B       Q4_K_M      mixtral:8x7b (IQ4_XS)
 QLoRA fine-tune      ~30B       nf4         command-r:35b (nf4)
 LoRA fine-tune       ~9B        bf16        gemma2:9b (bf16)
 Full fine-tune       ~1B        bf16        qwen2.5:1.5b (bf16)
 Train from scratch   ~1B        bf16        qwen2.5:1.5b (bf16)
```

## Install

```bash
pip install llmcalculator          # core + CLI, zero dependencies
pip install "llmcalculator[all]"   # adds prettier tables and the TUI
```

**Not comfortable with a terminal?** Download the launcher for your system from
`launchers/` and double-click it. It installs what it needs and opens the app in
your browser.

| System  | File |
|---------|------|
| macOS   | `llmcalculator-mac.command` |
| Windows | `llmcalculator-windows.bat` |
| Linux   | `llmcalculator-linux.sh` |

## Three interfaces

### 1. Command line

```bash
llmcalculator scan                          # what can this machine do?
llmcalculator check llama3.1:8b             # will this one model fit?
llmcalculator check qwen2.5:7b -a           # ...across every workload
llmcalculator compare llama3.1:8b qwen2.5:32b gpt-oss:20b
llmcalculator recommend --tag code          # good coding models for this machine
llmcalculator context llama3.1:8b           # how much does long context cost?
llmcalculator models qwen                   # search the built-in catalog
```

### Searching all of Hugging Face

The built-in catalog covers models commonly run locally. To search the whole
Hub and size every result against your machine:

```bash
llmcalculator search granite            # any query
llmcalculator search coder -n 25        # more results
llmcalculator trending                  # what the Hub is trending now
```

```
$ llmcalculator search granite

 Model                                 Size    Quant      Needs   Verdict
 ────────────────────────────────────────────────────────────────────────
 ibm-granite/granite-4.1-8b            8.4B    Q4_K_M    6.9 GB   * Comfortable
 ibm-granite/granite-4.1-30b           28.9B   Q4_K_M   19.6 GB   + Fits
 ibm-granite/granite-4.1-3b            3.4B    Q4_K_M    3.2 GB   * Comfortable
```

Results are cached for a week, so repeat searches are instant and work offline.
Manage the cache with `llmcalculator cache` and `llmcalculator cache --clear`.

Any single model works by id, catalogued or not:

```bash
llmcalculator check mistralai/Mistral-Small-24B-Instruct-2501
```

Planning a machine you do not own yet:

```bash
llmcalculator recommend --vram 24 --ram 64 --gpu-name "RTX 4090"
```

### 2. Terminal UI

```bash
llmcalculator tui
```

Browse the catalog with live verdicts.

| Key | Does |
|---|---|
| `up` / `down` | move through the list — **works while you are typing a filter** |
| `j` / `k` | same, vim-style |
| `g` / `G` | jump to the top / bottom |
| `page up` / `page down`, `home` / `end` | move faster |
| `/` | jump to the filter box |
| `enter` / `escape` | leave the filter box, keeping the filter |
| `w` / `c` | cycle workload / context |
| `r` | show only models worth running here |
| `h` | search all of Hugging Face for the current filter text |
| `a` | back to the full catalog |
| `q` | quit |

### 3. Browser app

```bash
llmcalculator app
```

Opens a local page at `127.0.0.1:8770`. The server is your own machine, and the
page itself is fully self-contained. The **Hugging Face** tab searches the Hub
live; every other tab works offline.

## What it tells you

For every model and workload it reports **where the memory goes**, which is the
part that lets you fix a bad fit rather than just learn about it:

```
$ llmcalculator check llama3.1:8b

llama3.1:8b  -  Inference
  8.0B params, 32 layers, GQA 4:1, 128k ctx

  * Comfortable   |####..............|  6.4 GB needed of 27.0 GB available

  Settings   Q4_K_M quantization, 8k token context, batch 1
  Speed      ~20 tokens/sec generation

  Memory breakdown
    Weights         4.71 GB  |########....|
    KV cache        1.00 GB  |#...........|
    Activations     0.14 GB  |............|
    Overhead        0.56 GB  |#...........|
```

## The five workloads

Memory cost per parameter differs by more than an order of magnitude across
these. It is why a machine that runs a 32B model comfortably can only fully
fine-tune a 1B one.

| Workload | Bytes/param | What it means |
|---|---|---|
| **Inference** | ~0.6 (Q4) | Running the model |
| **QLoRA** | ~1.1 | Adapters on a 4-bit frozen base |
| **LoRA** | ~2.8 | Adapters on a 16-bit frozen base |
| **Full fine-tune** | ~16 | Every weight updated, bf16 + Adam |
| **Training** | ~18 | From random initialisation |

## Python API

```python
import llmcalculator as lc

lc.check("llama3.1:8b").fits                      # True
lc.check("llama3.1:70b", "qlora").label()         # "Won't fit"

est = lc.check("qwen2.5:32b", context=32768)
print(est.total_gb, est.tokens_per_sec)
print(est.breakdown.items_gb())
print(est.as_dict())                              # JSON-ready

hw = lc.detect()
lc.max_model_size(hw, lc.workloads.QLORA)         # 29.8

# Size a machine you are thinking of buying
lc.check("llama3.1:70b", hardware=lc.manual(vram_gb=48, ram_gb=128))
```

Every command also takes `--json`, so it composes with other tooling:

```bash
llmcalculator scan --json | jq '.capabilities.qlora.max_params_b'
```

## How the numbers are worked out

**Parameter counts** for uncatalogued models are computed analytically from
`config.json`, accounting for grouped-query attention, tied embeddings, and
mixture-of-experts layouts where routed experts are far narrower than the dense
feed-forward width. Architectures the formula does not cover — Mamba, RWKV and
other hybrids — are refused rather than guessed at.

**Weights** use measured effective bytes-per-weight for each format, not the
nominal bit count. Q4_K_M is nominally 4 bits but lands near 4.8 once scales,
mins and the higher-precision attention tensors are counted.

**KV cache** is `2 × layers × kv_heads × head_dim × context × batch × bytes`,
using each model's real grouped-query-attention configuration.

**Training** adds gradients and optimizer state for the trainable fraction —
about 0.5% for LoRA, all of it for a full fine-tune — plus activations, which
assume gradient checkpointing is on.

**Speed** is bandwidth-bound for generation (each token reads every active
weight once) and compute-bound for prefill. Mixture-of-experts models count
only active parameters, which is why a 30B MoE outruns a 30B dense model by
several times.

Expect estimates within a few percent of real usage. Runtimes differ slightly
in allocator behaviour and buffer sizing.

## Does a GPU matter?

Yes for speed, no for possibility. Ollama, llama.cpp and this tool all work on
CPU-only machines — roughly 3-10x slower than a GPU of the same memory size.

**Machines with no GPU are fully supported.** `scan` detects the absence, sizes
everything against system RAM minus what the OS needs, and estimates CPU speed
from memory bandwidth. This path runs on every push: GitHub's CI runners have no
GPU, so Linux, macOS and Windows are all exercised without one.

```
$ llmcalculator scan          # on a 16 GB machine with no GPU

  CPU        AMD Ryzen 5 5600
  RAM        16.0 GB
  GPU        none detected - CPU inference only

 Workload             Rough ceiling   Largest usable model
 ─────────────────────────────────────────────────────────
 Inference            ~16B (Q4_K_M)   gpt-oss:20b (IQ4_XS)
 QLoRA fine-tune      ~12B (nf4)      phi4:14b (nf4)
 LoRA fine-tune       ~4B (bf16)      qwen3:4b-2507 (bf16)
 Full fine-tune       ~0.4B (bf16)    qwen2.5:0.5b (bf16)
```

`--device cpu` forces sizing against system RAM even on a machine that has a
GPU, which is what you want when a model is too big for VRAM.

On Apple Silicon there is no separate VRAM: CPU and GPU share one pool, and
macOS caps the GPU's share (about 75% of RAM, or RAM minus 8 GB above 36 GB).
`llmcalculator` reads that limit rather than assuming it.

## Contributing

`main` is protected: everything lands through a pull request that has passed CI
on Linux, macOS and Windows and been approved by a code owner.

The most valuable contribution is **reporting a wrong estimate** — if the tool
said a model fits and it did not, that is a real bug with real cost. There is an
issue template for it.

- [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, how to add a model or a GPU,
  and the rules around changing the estimator
- [SECURITY.md](SECURITY.md) — report privately, never in a public issue
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [Discussions](https://github.com/BarakaSoka/llmcalculator/discussions) — ask
  before building something large

## Development

```bash
git clone https://github.com/llmcalculator/llmcalculator
cd llmcalculator
pip install -e ".[dev]"
pytest              # 121 tests
```

Adding a model means one entry in `src/llmcalculator/models/catalog.json`.
Copy the architecture fields straight from the model's `config.json` on
Hugging Face — see [CONTRIBUTING.md](CONTRIBUTING.md).

Note you may not need to: `llmcalculator search` covers the whole Hub already.
The catalog exists so common models work offline and appear in `recommend`.

## License

MIT
