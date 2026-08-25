# Contributing

Thanks for considering it. This document is longer than most because the
project has one unusual property worth understanding before you change
anything: **a wrong number here is worse than no number, because people act on
it.** Someone reads "fits comfortably", downloads 20 GB over a slow connection,
and finds out otherwise. That shapes most of the rules below.

## Quick start

```bash
git clone https://github.com/BarakaSoka/llmcalculator
cd llmcalculator
pip install -e ".[dev]"
pytest
```

105 tests should pass in about 30 seconds. Tests needing network access skip
themselves cleanly when you are offline.

## The workflow

`main` is protected. Nobody pushes to it directly, including the maintainer.
Everything arrives through a pull request that has passed CI and been approved
by a code owner.

1. **Fork**, then branch: `git checkout -b fix-kv-cache-for-gemma`
2. Make the change, **with a test that fails without it**
3. `pytest` locally
4. Open a PR. The template asks a few questions — the estimator ones matter most
5. CI runs on Linux, macOS and Windows across Python 3.9–3.13
6. A code owner reviews. Expect questions about *how you verified* a number

Small PRs get reviewed quickly. A 900-line PR touching the estimator, the CLI
and the catalog at once will sit for a while, so please split it.

### Before building something large

Open a [discussion](https://github.com/BarakaSoka/llmcalculator/discussions) or
an issue first. It is genuinely disappointing to decline a week of someone's
work because the design does not fit, and that is avoidable with a short
conversation up front.

## What is most useful

**Reporting a wrong estimate.** The single most valuable contribution. If the
tool said a model fits and it did not, that is a real bug with real cost. Use
the "Wrong estimate" template and include your `llmcalculator scan --json`.

**Adding a GPU.** `src/llmcalculator/hardware/gpu_db.py` maps a name fragment
to `(memory bandwidth GB/s, dense fp16 TFLOPS)`. Use manufacturer
specifications, not benchmarks, and the **dense** figure rather than the sparse
one marketing material often quotes. Keys match longest-first, so
`rtx 4080 super` correctly beats `rtx 4080`. An unrecognised GPU returns
`(0.0, 0.0)`, which the estimator reports as "speed unknown" rather than
guessing — preserve that behaviour.

**Adding a model.** One entry in `src/llmcalculator/models/catalog.json`.

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

Copy every field from the model's real `config.json`. Do not estimate them: the
KV cache maths depends on `num_key_value_heads` and `head_dim` being exact, and
getting them wrong produces a plausible number that is quietly incorrect.

For mixture-of-experts models add `active_params` — the count actually read per
token. It drives the speed estimate, and is why a 30B MoE outruns a 30B dense
model several times over.

Verify with `llmcalculator check <name>` before opening the PR: the reported
size should match the published parameter count. If it does not, the
architecture probably departs from the standard attention + SwiGLU decoder the
formula assumes, and the model may not belong in the catalog at all.

**Note:** you often do not need to add a model. `llmcalculator search` already
covers the entire Hub. The catalog exists so common models work offline and can
appear in `recommend`.

## Changing the estimator

This is the part to be careful with. `estimate.py`, `quant.py`, `workloads.py`
and `models/spec.py` are code-owner reviewed for this reason.

If you change a coefficient, **pin it to something observable**:

- a published GGUF file size on Hugging Face
- a documented KV cache figure from a model card or paper
- a measured out-of-memory threshold on real hardware, with the runtime named

Do not pin a test to what the code currently outputs. That locks in whatever is
there, including the bug you were meant to catch. `tests/test_estimate.py` is
written against external references throughout; follow its lead.

Say in the PR what you checked against. "Llama 3.1 8B at Q4_K_M ships as a
4.92 GB GGUF, and the estimator now reports 4.71 GB of weights" is a reviewable
claim. "Improved accuracy" is not.

### Architectures the maths does not cover

State-space and hybrid models — Mamba, RWKV, Jamba, Nemotron-H — raise
`UnsupportedArchitecture` on purpose. They allocate weights differently, so the
formula would return a confident wrong number. **Declining to answer is the
correct behaviour**; please do not replace it with an approximation.

## Style

Match the surrounding code. Specifically:

- Comments explain **why**, not what. If a constant looks arbitrary, say where
  it came from
- Public functions get a docstring saying what the caller gets, and any
  assumption baked in
- No new required dependencies in the core package. `pip install llmcalculator`
  must stay standard-library-only; that is a feature, not an accident. Optional
  extras (`[tui]`, `[pretty]`) are the place for anything else
- Python 3.9 compatible. CI enforces it
- The CLI must work with `rich` absent — there is a CI job asserting this

## Tests

```bash
pytest                      # everything
pytest tests/test_estimate.py -v
pytest -k "moe"             # by name
pytest -m "not slow"
```

Network-dependent tests use a `needs_net` guard and skip offline. Keep new
network tests behind it so contributors on a plane are not blocked.

## Dependency updates

Dependabot proposes GitHub Actions updates monthly. They are handled
automatically, with one deliberate exception:

- **patch and minor** bumps are approved and queued for auto-merge. CI still
  has to pass — auto-merge waits for the required check, it does not skip it
- **major** bumps are labelled `major-update` and left for a human

The exception exists because a green CI run is weaker evidence than it looks
for a major bump. `release.yml` is never run by CI, so a change confined to
that file arrives untested; `actions/download-artifact` v4 to v8 is the
worked example, since it changed digest-mismatch handling from a warning to a
hard failure. Read the release notes before merging one of those.

If both `upload-artifact` and `download-artifact` have updates pending, merge
them together — they have to agree on the artifact format.

## Security

Do not open a public issue for a security problem. See [SECURITY.md](SECURITY.md).

Never commit a credential. `.gitignore` blocks `.pypirc`, `.env`, `*.pem` and
`*.key`, but that is a safety net rather than a substitute for checking your
diff.

## Releasing

Maintainers only:

1. Bump `version` in `pyproject.toml` and `src/llmcalculator/__init__.py`
2. Add a `CHANGELOG.md` entry
3. `git tag vX.Y.Z && git push origin vX.Y.Z`

The release workflow refuses to publish if the tag and `pyproject.toml`
disagree, and uses PyPI Trusted Publishing, so no token exists to leak.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
