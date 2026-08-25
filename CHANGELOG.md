# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-08-25

### Added

- **Model list navigation in the TUI.** Up and down now move through the list
  even while the cursor is in the filter box, which is the common case: you
  type a few characters to narrow the list, then want to walk the results
  without tabbing back to the table first. Focus stays where you left it.
- `j` / `k` for vim-style movement, `g` / `G` to jump to either end, and
  `page up` / `page down` / `home` / `end` for larger steps.
- `enter` or `escape` leaves the filter box for the list, keeping the filter.
- 12 tests covering navigation, including bounds and the empty-list case.


## [0.2.0] - 2026-08-25

### Added

- **`search`** — search the whole Hugging Face Hub, not just the catalog, with
  every result sized against your machine. Responses are cached for a week, so
  repeat searches are instant and work offline.
- **`trending`** — what the Hub is trending right now, sized for your hardware.
- **`cache`** — inspect or clear the Hub response cache.
- Hub search in the TUI (`h`) and a **Hugging Face** tab in the web app.
- Catalog grown from 44 to 64 models across 21 families, spanning 0.5B to
  1033B. Every new entry was generated from the model's real `config.json`
  rather than transcribed by hand.

### Fixed

- **Mixture-of-experts parameter counts were wildly overstated.** Routed experts
  use `moe_intermediate_size`, which is typically far narrower than the dense
  `intermediate_size`. Qwen3-30B-A3B was reported as 233B rather than 30.5B.
- **Tied embeddings were double-counted.** Models setting
  `tie_word_embeddings` share one matrix between input and output; assuming two
  overstated Qwen2.5-0.5B by 27%.
- **State-space and hybrid architectures** (Mamba, RWKV, Jamba, Nemotron-H) are
  now refused with an explanation instead of returning a confident wrong number.
- DeepSeek-style dense prefix layers (`first_k_dense_replace`) and shared
  experts are accounted for; ChatGLM's `padded_vocab_size` is understood.
- The quantization advice no longer renders an unknown speed as
  "about 0 tok/s instead of 0", and explains why the estimate is missing.


## [0.1.0] - 2026-08-23

First release.

### Added

- **Hardware detection** for Apple Silicon, NVIDIA, AMD, Intel, and CPU-only
  machines. Reads the macOS unified-memory wired limit rather than assuming it,
  and degrades to a partial picture instead of failing when a vendor tool is
  missing.
- **Five workloads** — inference, QLoRA, LoRA, full fine-tune, and pre-training
  — spanning more than an order of magnitude in bytes per parameter.
- **Architecture-accurate sizing.** KV cache is computed from each model's real
  grouped-query-attention configuration, which is usually what decides whether
  a model fits at long context.
- **19 quantization formats** with measured effective bytes-per-weight rather
  than nominal bit counts.
- **44-model catalog**, plus live `config.json` lookup for any model on Hugging
  Face via the standard library alone.
- **Speed estimates**: bandwidth-bound for generation, compute-bound for
  prefill, counting only active parameters for mixture-of-experts models.
- **Three interfaces**: a zero-dependency CLI, a Textual TUI, and a local web
  app served from the standard library that works fully offline.
- **Launchers** for macOS, Windows, and Linux, plus a PyInstaller build script
  for a standalone binary that needs no Python installed.
- 65 tests validating the sizing math against published model sizes.

### Notes

- The console script is `llmcalculator`. There is deliberately no short
  `llmcalc` alias: an unrelated PyPI package of that name already ships a
  console script called `llmcalc`, and installing both would leave whichever
  landed last owning the command.

[Unreleased]: https://github.com/BarakaSoka/llmcalculator/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.2.1
[0.2.0]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.2.0
[0.1.0]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.1.0
