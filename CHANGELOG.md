# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/BarakaSoka/llmcalculator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BarakaSoka/llmcalculator/releases/tag/v0.1.0
