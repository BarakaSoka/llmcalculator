## What this changes

<!-- One or two sentences. What is different after this PR? -->

## Why

<!-- The problem being solved. Link an issue with "Fixes #123" if there is one. -->

## Type of change

- [ ] Bug fix
- [ ] New model(s) in the catalog
- [ ] New GPU(s) in the hardware database
- [ ] Estimator change (memory or speed maths)
- [ ] New feature
- [ ] Documentation
- [ ] Build, CI, or packaging

## Checks

- [ ] `pytest` passes locally
- [ ] I added a test that fails without this change
- [ ] I did not commit any credential, token, or `.pypirc`

## If this changes the estimator

A wrong number is worse than no number, because people act on it. So:

- [ ] I pinned the new behaviour to something observable — a published GGUF
      file size, a documented KV cache figure, or a measured out-of-memory
      threshold — rather than to what the code currently does

**What did you check it against?**

<!-- e.g. "Llama 3.1 8B Q4_K_M is a 4.92 GB GGUF on Hugging Face; the
     estimator now says 4.71 GB of weights, within the expected margin." -->

## If this adds a model

- [ ] Architecture fields are copied from the model's `config.json`, not estimated
- [ ] `active_params` is set if it is a mixture-of-experts model
- [ ] `llmcalculator check <name>` gives a size matching the published one

<!--
Note: you may not need to add a model at all. `llmcalculator search` already
covers the whole Hub. The catalog exists so common models work offline and
appear in `recommend`.
-->

## Anything you are unsure about

<!-- Genuinely useful. Half-finished PRs with a clear question are welcome. -->
