# Contributing

## Setup

```bash
git clone https://github.com/BarakaSoka/llmcalculator
cd llmcalculator
pip install -e ".[dev]"
pytest
```

## Adding a model

One entry in `src/llmcalculator/models/catalog.json`. Copy the architecture
fields straight from the model's `config.json` on Hugging Face — do not
estimate them, because the KV cache math depends on `num_key_value_heads` and
`head_dim` being exact.

```json
{
  "name": "qwen3:8b",
  "hf_id": "Qwen/Qwen3-8B",
  "family": "qwen3",
  "params": 8.19e9,
  "n_layers": 36,
  "hidden_size": 4096,
  "n_heads": 32,
  "n_kv_heads": 8,
  "head_dim": 128,
  "vocab_size": 151936,
  "max_context": 32768,
  "license": "apache-2.0",
  "tags": ["general", "reasoning"]
}
```

For mixture-of-experts models add `active_params` — the count actually read per
token. It drives the speed estimate and is why a 30B MoE outruns a 30B dense
model several times over.

`tests/test_catalog.py` checks every entry for internal consistency.

## Adding a GPU

`src/llmcalculator/hardware/gpu_db.py` maps a name fragment to
`(memory bandwidth GB/s, dense fp16 TFLOPS)`. Use manufacturer specifications,
not benchmark results, and use the dense figure rather than the sparse one that
marketing material often quotes.

Keys are matched longest-first, so `rtx 4080 super` correctly beats `rtx 4080`.
An unknown GPU returns `(0.0, 0.0)`, which the estimator reports as "speed
unknown" rather than guessing.

## Changing the estimator

The sizing math lives in `estimate.py` and is the part most worth being careful
with — a wrong number here is worse than no number, because someone will act on
it. If you change a coefficient, add a test that pins it to something
observable: a published GGUF file size, a documented KV cache figure, or a
measured out-of-memory threshold.

`tests/test_estimate.py` is written this way; follow its lead.

## Reporting a wrong estimate

Genuinely useful, and the main way this improves. Include the `llmcalculator
scan --json` output, what the tool predicted, what actually happened, and which
runtime you used. There is an issue template for it.
