# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Model information now covers what a model *is*, not only how large it is.
  Every spec carries **capabilities** (chat, reasoning, code, vision, tool
  calling, agentic, multilingual, long context, MoE, edge and more), the
  **weight formats** it ships in, and the **runtimes** that load them - each
  with a one-line description rather than a bare label.
- More architecture detail on every spec: feed-forward width, routed and
  active expert counts, expert width, tied embeddings, RoPE theta, sliding
  window, published dtype, the `architectures` entry, attention kind spelled
  out (MHA / GQA n:1 / MQA), and KV cache cost per 1k tokens.
- `llmcalculator info <model>` prints all of it for one model, with no
  hardware involved. `--brief` drops the descriptions; `--json` emits the lot.
- `llmcalculator capabilities` explains the whole vocabulary - every
  capability, weight format and runtime, with how many catalog models have
  each capability.
- `llmcalculator models --capability/-c <cap>` filters the catalog, the
  listing gained a Capabilities column, and `recommend --tag` now matches a
  capability as well as a tag.
- `ModelSpec.support()`, `.has_capability()`, `.architecture_items()` and
  `.as_dict()`; `catalog.by_capability()` and `catalog.capability_counts()`.
- The browser app gained a capability filter, a "What this model is" section
  with per-trait descriptions, an architecture table, and a
  `/api/capabilities` endpoint. The TUI detail panel gained the same
  information. Hub search results carry capabilities and formats too.

### Changed

- `catalog.search()` matches capabilities as well as names, families and tags,
  so `search("vision")` finds the multimodal models that never say so in their
  name.
- Hub search results merge repository tags into the resolved spec: which
  formats a repo actually publishes, and its licence, come from the tags,
  since a `config.json` cannot say. Formats beyond the derivable set (AWQ,
  GPTQ, EXL2, native 4-bit releases) are only claimed when something evidences
  them.
- `check` now prints a three-line capability / format / runtime summary above
  the memory breakdown.

- Dependabot's patch and minor updates are now approved and auto-merged;
  major updates are labelled `major-update` and left for a human. CI is still
  required in both cases — auto-merge waits for the check rather than skipping
  it.
- Branch protection now requires one approving review rather than one from a
  code owner. On this repository the two are near-identical for humans, since
  only accounts with write access can cast an approval that counts, and there
  is exactly one. The change is what lets the Dependabot workflow approve its
  own routine bumps without a stored credential.


## [0.2.3] - 2026-08-25

No functional change. This release exists to exercise the release workflow on
its updated GitHub Actions.

### Changed

- CI and release workflows moved to `actions/checkout@v7`,
  `actions/setup-python@v7`, `actions/upload-artifact@v7` and
  `actions/download-artifact@v8` (#1, #2, #3, #4).

  `download-artifact@v8` is a breaking release: digest mismatches now fail the
  run rather than logging a warning, and non-zipped downloads are no longer
  unzipped blindly. Both are improvements for a workflow that publishes to
  PyPI. The upload and download actions were updated together, since they
  have to agree on the artifact format.


## [0.2.2] - 2026-08-25

### Fixed

- **Asking for a GPU budget on a machine without a GPU gave useless advice.**
  It reported "1.7 GB needed of 0.0 GB available" and suggested lowering the
  quantization, which cannot help when the budget is zero. It now explains that
  no GPU was found and points at `--device cpu`.
- **The "largest that fits" figure reached for severely degraded quantization.**
  On a 16 GB machine it named a 24B model squeezed to 3 bits, which is worse
  than a 9B at 5 bits. Headline figures now apply a quality floor, and the
  column is labelled "largest usable model".
- `scan` juxtaposed a "max size" quoted at one precision with an example using
  another, which read as a contradiction. The precision is now attached to the
  number it describes.
- Parameter ceilings below 1B rendered as "~0B", which told the reader nothing.

### Added

- 16 tests covering GPU-less machines from 4 GB to 64 GB.


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

[Unreleased]: https://github.com/BarakaSoka/llmcalculator/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.2.3
[0.2.2]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.2.2
[0.2.1]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.2.1
[0.2.0]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.2.0
[0.1.0]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.1.0
