# llmcalculator

Work out which AI models your computer can actually run — for inference,
fine-tuning, and training — before you spend an hour downloading one.

It reads your real hardware, then sizes each model from its **actual
architecture** rather than a rule of thumb. That distinction matters: two 7B
models can differ by 8x in KV cache at long context, which is usually what
decides whether something fits.

```
$ llmcalc scan

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
llmcalc scan                          # what can this machine do?
llmcalc check llama3.1:8b             # will this one model fit?
llmcalc check qwen2.5:7b -a           # ...across every workload
llmcalc compare llama3.1:8b qwen2.5:32b gpt-oss:20b
llmcalc recommend --tag code          # good coding models for this machine
llmcalc context llama3.1:8b           # how much does long context cost?
llmcalc models qwen                   # search the catalog
```

Any model on Hugging Face works, not just the built-in catalog:

```bash
llmcalc check mistralai/Mistral-Small-24B-Instruct-2501
```

Planning a machine you do not own yet:

```bash
llmcalc recommend --vram 24 --ram 64 --gpu-name "RTX 4090"
```

### 2. Terminal UI

```bash
llmcalc tui
```

Browse the catalog with live verdicts. `w` cycles workload, `c` cycles context,
`/` searches, `r` filters to models worth running on your machine.

### 3. Browser app

```bash
llmcalc app
```

Opens a local page at `127.0.0.1:8770`. Nothing is uploaded and nothing is
fetched from the internet — the server is your own machine.

## What it tells you

For every model and workload it reports **where the memory goes**, which is the
part that lets you fix a bad fit rather than just learn about it:

```
$ llmcalc check llama3.1:8b

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
llmcalc scan --json | jq '.capabilities.qlora.max_params_b'
```

## How the numbers are worked out

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
`llmcalc scan --device cpu` sizes against system RAM instead of VRAM.

On Apple Silicon there is no separate VRAM: CPU and GPU share one pool, and
macOS caps the GPU's share (about 75% of RAM, or RAM minus 8 GB above 36 GB).
`llmcalc` reads that limit rather than assuming it.

## Development

```bash
git clone https://github.com/llmcalculator/llmcalculator
cd llmcalculator
pip install -e ".[dev]"
pytest              # 65 tests
```

Adding a model means one entry in `src/llmcalculator/models/catalog.json`.
Copy the architecture fields straight from the model's `config.json` on
Hugging Face.

## License

MIT
