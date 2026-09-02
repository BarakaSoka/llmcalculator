"""Command line interface.

Built on argparse so `llmcalc` works on a bare Python install with nothing
pip-installed beyond this package itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__, compare, quant, workloads
from .estimate import (
    Verdict, best_quant, estimate as estimate_one, max_model_size,
    recommended_quant, sweep,
)
from .hardware import detect, manual
from .hardware.base import HardwareProfile
from .models import capabilities as caps
from .models import catalog
from .models import hub
from .ui import render


def _parse_context(value: str) -> int:
    """Accept 8192, 8k, 128K."""
    v = str(value).strip().lower()
    try:
        if v.endswith("k"):
            return int(float(v[:-1]) * 1024)
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("Context must be a number like 8192 or 8k, got {!r}".format(value))


def _hardware_from_args(args) -> HardwareProfile:
    if getattr(args, "vram", None) is not None or getattr(args, "ram", None) is not None:
        return manual(
            vram_gb=args.vram or 0.0,
            ram_gb=args.ram or 16.0,
            gpu_name=getattr(args, "gpu_name", None) or "Custom GPU",
        )
    return detect()


def _resolve_model(name: str, use_hf: bool = False):
    if use_hf or ("/" in name and name.count("/") == 1):
        try:
            return catalog.get(name)
        except KeyError:
            return catalog.from_hf(name)
    return catalog.get(name)


# --- commands --------------------------------------------------------------

def cmd_scan(args) -> int:
    """Show the machine and what it can do across every workload."""
    hw = _hardware_from_args(args)

    if args.json:
        print(json.dumps(_scan_dict(hw), indent=2))
        return 0

    render.heading("Your machine")
    for line in hw.summary_lines():
        print("  " + line)

    render.heading("What this machine can do")
    rows, colors = [], []
    for wl in workloads.ALL:
        ctx = 8192 if wl.key == "inference" else 2048
        max_b = max_model_size(hw, wl, context=ctx)
        best = compare.largest_that_fits(hw, wl, context=ctx)
        if best:
            example = "{} ({})".format(best.model.name, best.quant_name)
            colors.append(None)
        else:
            # This column applies a quality floor. Saying "nothing fits" when a
            # heavily quantized option exists would contradict the
            # recommendations printed directly below.
            degraded = compare.largest_that_fits(hw, wl, context=ctx, min_quality=0.0)
            if degraded:
                example = "{} ({}, low quality)".format(
                    degraded.model.name, degraded.quant_name)
                colors.append("yellow")
            else:
                example = "nothing in the catalog fits"
                colors.append("red")
        # Precision belongs with the number it describes: the rough ceiling is
        # quoted at one format and the concrete example may use another.
        rows.append([wl.label, "{} ({})".format(_fmt_size(max_b), wl.default_base_quant), example])
    render.table(["Workload", "Rough ceiling", "Largest usable model"], rows, colors=colors)

    render.heading("Recommended models for inference")
    recs = compare.recommend(hw, limit=5, context=args.context)
    rows = []
    for r in recs:
        e = r.estimate
        rows.append([e.model.name, e.quant_name, "{:.1f} GB".format(e.total_gb),
                     "{:.0f} tok/s".format(e.tokens_per_sec), r.reason])
    if rows:
        render.table(["Model", "Quant", "Memory", "Speed", "Why"], rows)
    else:
        print("  Nothing in the catalog fits. Try a smaller context with --context 2048.")

    for note in hw.notes:
        print()
        print(render.paint("  Note: " + note, "dim"))

    print()
    print(render.paint("  Next: llmcalculator check <model>   or   llmcalculator compare a b c", "dim"))
    return 0


def _fmt_size(params_b: float) -> str:
    """Render a parameter ceiling. Rounding 0.4B to "~0B" tells the reader
    nothing; below 1B the interesting fact is the order of magnitude."""
    if params_b < 0.1:
        return "under 0.1B"
    if params_b < 1:
        return "~{:.1f}B".format(params_b)
    return "~{:.0f}B".format(params_b)


def _scan_dict(hw: HardwareProfile) -> dict:
    out = {
        "cpu": {"name": hw.cpu.name, "cores": hw.cpu.cores, "arch": hw.cpu.arch},
        "ram_gb": round(hw.ram_gb, 1),
        "disk_free_gb": round(hw.disk_free_gb, 1),
        "platform": hw.platform,
        "accelerators": [
            {"name": a.name, "vendor": a.vendor, "memory_gb": round(a.memory_gb, 1),
             "bandwidth_gbs": a.bandwidth_gbs, "fp16_tflops": a.fp16_tflops,
             "unified_memory": a.unified_memory}
            for a in hw.accelerators
        ],
        "budgets_gb": {d: round(hw.budget_bytes(d) / (1024 ** 3), 1)
                       for d in ("gpu", "cpu", "all-gpus")},
        "capabilities": {},
        "notes": hw.notes,
    }
    for wl in workloads.ALL:
        ctx = 8192 if wl.key == "inference" else 2048
        best = compare.largest_that_fits(hw, wl, context=ctx)
        out["capabilities"][wl.key] = {
            "max_params_b": round(max_model_size(hw, wl, context=ctx), 1),
            "largest_model": best.model.name if best else None,
        }
    return out


def cmd_check(args) -> int:
    """Size one model, optionally across every workload."""
    hw = _hardware_from_args(args)
    try:
        model = _resolve_model(args.model, args.hf)
    except (KeyError, RuntimeError) as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2

    if args.all_workloads:
        ests = compare.workload_table(model, hw, context=args.context, device=args.device)
        if args.json:
            print(json.dumps([e.as_dict() for e in ests], indent=2))
            return 0
        render.heading("{} on your machine".format(model.name))
        print(render.paint("  {}".format(model.describe()), "dim"))
        print(render.paint("  Budget: {:.1f} GB at {} context".format(
            ests[0].budget_gb, render.fmt_ctx(args.context)), "dim"))
        rows, colors = [], []
        for e in ests:
            rows.append([e.workload.label, e.quant_name, "{:.1f} GB".format(e.total_gb),
                         render.bar(e.utilization, 12), render.verdict_cell(e)])
            colors.append(render.VERDICT_COLOR[e.verdict])
        render.table(["Workload", "Base", "Needs", "Usage", "Verdict"],
                     rows, colors=colors)
        _print_first_blocker(ests)
        return 0

    wl = workloads.get(args.workload)
    if args.quant:
        est = estimate_one(model, hw, wl, args.quant, context=args.context,
                               batch=args.batch, device=args.device, kv_quant=args.kv_quant)
    else:
        est = recommended_quant(model, hw, wl, context=args.context, batch=args.batch,
                                device=args.device, kv_quant=args.kv_quant)

    if args.json:
        print(json.dumps(est.as_dict(), indent=2))
        return 0 if est.fits else 1

    render.estimate_detail(est)

    if args.quant is None and wl.key == "inference":
        render.heading("All quantizations")
        rows, colors = [], []
        for e in compare.quant_table(model, hw, wl, context=args.context, device=args.device):
            rows.append([e.quant_name, "{:.1f} GB".format(e.total_gb),
                         render.bar(e.utilization, 12),
                         "{:.0f} tok/s".format(e.tokens_per_sec) if e.tokens_per_sec else "-",
                         "{:.0f}%".format(e.quality * 100), render.verdict_cell(e)])
            colors.append(render.VERDICT_COLOR[e.verdict])
        render.table(["Format", "Memory", "Usage", "Speed", "Quality", "Verdict"],
                     rows, colors=colors)
    print()
    return 0 if est.fits else 1


def _print_first_blocker(ests) -> None:
    for e in ests:
        if not e.fits and e.notes:
            print()
            print(render.paint("  {}: {}".format(e.workload.label, e.notes[0]), "yellow"))
            for extra in e.notes[1:2]:
                print(render.paint("  {}".format(extra), "dim"))
            return


def cmd_compare(args) -> int:
    """Put several models side by side."""
    hw = _hardware_from_args(args)
    wl = workloads.get(args.workload)
    try:
        ests = compare.compare(args.models, hw, wl, args.quant, args.context, args.device)
    except (KeyError, RuntimeError) as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([e.as_dict() for e in ests], indent=2))
        return 0

    render.heading("Comparison - {} at {} context".format(wl.label, render.fmt_ctx(args.context)))
    rows, colors = [], []
    for e in ests:
        rows.append([
            e.model.name,
            "{:.1f}B".format(e.model.params_b),
            e.quant_name,
            "{:.1f} GB".format(e.total_gb),
            "{:.0f}%".format(e.utilization * 100),
            "{:.0f} tok/s".format(e.tokens_per_sec) if e.tokens_per_sec else "-",
            render.verdict_cell(e),
        ])
        colors.append(render.VERDICT_COLOR[e.verdict])
    render.table(["Model", "Size", "Quant", "Memory", "Used", "Speed", "Verdict"],
                 rows, colors=colors)

    fitting = [e for e in ests if e.fits]
    if fitting:
        best = max(fitting, key=lambda e: e.model.params)
        fastest = max(fitting, key=lambda e: e.tokens_per_sec)
        print()
        print("  Most capable that fits: {}".format(render.paint(best.model.name, "green")))
        if fastest.model.name != best.model.name:
            print("  Fastest that fits:      {} ({:.0f} tok/s)".format(
                render.paint(fastest.model.name, "green"), fastest.tokens_per_sec))
    print()
    return 0


def cmd_models(args) -> int:
    """List or search the catalog."""
    models = catalog.search(args.query) if args.query else catalog.all_models()
    if args.capability:
        models = [m for m in models if m.has_capability(args.capability)]
    if args.json:
        print(json.dumps([m.as_dict() for m in models], indent=2))
        return 0
    if not models:
        if args.capability and not args.query:
            print("No catalogued model has the capability {!r}.".format(args.capability))
            print()
            print("Known capabilities: {}".format(
                ", ".join(catalog.capability_counts())))
            print("Explain them with: llmcalculator capabilities")
            return 1
        print("No catalogued model matches {!r}.".format(args.query))
        print()
        print("The catalog covers {} models commonly run locally. To search all of"
              .format(len(catalog.all_models())))
        print("Hugging Face instead:")
        print()
        print("    llmcalculator search {}".format(args.query))
        return 1
    rows = [[m.name, "{:.1f}B".format(m.params_b),
             "{:.1f}B".format(m.active_params / 1e9) if m.is_moe else "-",
             m.family, render.fmt_ctx(m.max_context),
             m.support().summary("capability", limit=4)]
            for m in models]
    render.table(["Model", "Params", "Active", "Family", "Context", "Capabilities"], rows,
                 title="{} models".format(len(models)))
    print()
    print(render.paint("  Full detail for any of these: llmcalculator info <model>", "dim"))
    return 0


def cmd_info(args) -> int:
    """Everything known about one model, with no hardware in the picture."""
    try:
        model = _resolve_model(args.model, args.hf)
    except (KeyError, RuntimeError) as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(model.as_dict(), indent=2))
        return 0

    render.model_info(model, describe_traits=not args.brief)
    print()
    print(render.paint(
        "  Next: llmcalculator check {}   to size it for this machine".format(model.name),
        "dim"))
    return 0


def cmd_capabilities(args) -> int:
    """Explain the capability, format and runtime vocabulary."""
    kinds = {"capabilities": ("capability", "Capabilities"),
             "formats": ("format", "Weight formats"),
             "runtimes": ("runtime", "Runtimes")}
    wanted = ([(args.kind, *kinds[args.kind])] if args.kind != "all"
              else [(name, *pair) for name, pair in kinds.items()])

    if args.json:
        print(json.dumps({name: [t.as_dict() for t in caps.known(kind)]
                          for name, kind, _ in wanted}, indent=2))
        return 0

    counts = catalog.capability_counts()
    for _, kind, title in wanted:
        render.heading(title)
        if kind == "capability":
            rows = [[t.label, t.key, str(counts.get(t.key, 0)), t.description]
                    for t in caps.known(kind)]
            render.table(["Name", "Key", "Models", "Description"], rows)
        else:
            rows = [[t.label, t.key, t.description] for t in caps.known(kind)]
            render.table(["Name", "Key", "Description"], rows)
    print()
    print(render.paint("  Filter the catalog with: llmcalculator models -c code", "dim"))
    return 0


def cmd_recommend(args) -> int:
    """Rank catalog models for this machine."""
    hw = _hardware_from_args(args)
    wl = workloads.get(args.workload)
    recs = compare.recommend(hw, wl, context=args.context, limit=args.limit,
                             device=args.device, tag=args.tag,
                             min_tokens_per_sec=args.min_speed)
    if args.json:
        print(json.dumps([{**r.estimate.as_dict(), "score": round(r.score, 1),
                           "reason": r.reason} for r in recs], indent=2))
        return 0
    if not recs:
        print(render.paint("Nothing fits with these settings. Try --context 2048 "
                           "or a different workload.", "yellow"))
        return 1
    render.heading("Best {} models for your machine".format(wl.label.lower()))
    rows = []
    for r in recs:
        e = r.estimate
        rows.append([e.model.name, "{:.1f}B".format(e.model.params_b), e.quant_name,
                     "{:.1f} GB".format(e.total_gb),
                     "{:.0f} tok/s".format(e.tokens_per_sec) if e.tokens_per_sec else "-",
                     r.reason])
    render.table(["Model", "Size", "Quant", "Memory", "Speed", "Why"], rows)
    print()
    return 0


def cmd_context(args) -> int:
    """Show how context length changes the memory picture."""
    hw = _hardware_from_args(args)
    try:
        model = _resolve_model(args.model, args.hf)
    except (KeyError, RuntimeError) as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2
    ests = sweep(model, hw, quant_name=args.quant, device=args.device)
    if args.json:
        print(json.dumps([e.as_dict() for e in ests], indent=2))
        return 0
    render.heading("{} - context length vs memory".format(model.name))
    rows, colors = [], []
    for e in ests:
        rows.append([render.fmt_ctx(e.context),
                     "{:.1f} GB".format(e.breakdown.weights / (1024 ** 3)),
                     "{:.1f} GB".format(e.breakdown.kv_cache / (1024 ** 3)),
                     "{:.1f} GB".format(e.total_gb),
                     render.bar(e.utilization, 12),
                     render.verdict_cell(e)])
        colors.append(render.VERDICT_COLOR[e.verdict])
    render.table(["Context", "Weights", "KV cache", "Total", "Usage", "Verdict"],
                 rows, colors=colors)
    fitting = [e for e in ests if e.fits]
    if fitting:
        print()
        print("  Longest context that fits: {}".format(
            render.paint(render.fmt_ctx(max(fitting, key=lambda e: e.context).context), "green")))
    print()
    return 0


def cmd_search(args) -> int:
    """Search the whole Hugging Face Hub and size every hit."""
    hw = _hardware_from_args(args)
    wl = workloads.get(args.workload)
    try:
        results = hub.search_resolved(
            args.query, limit=args.limit, sort=args.sort,
            include_gguf=args.gguf, use_cache=not args.no_cache)
    except RuntimeError as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2

    if not results:
        print("Nothing on the Hub matches {!r}.".format(args.query))
        return 1

    rows, colors, payload = [], [], []
    for r in results:
        if not r.resolved:
            rows.append([r.hub.id, "-", "-", "-", r.error])
            colors.append("dim")
            continue
        est = recommended_quant(r.spec, hw, wl, context=args.context, device=args.device)
        payload.append({**est.as_dict(), "hf_id": r.hub.id,
                        "downloads": r.hub.downloads, "likes": r.hub.likes})
        rows.append([
            r.hub.id,
            "{:.1f}B".format(r.spec.params_b),
            est.quant_name,
            "{:.1f} GB".format(est.total_gb),
            render.verdict_cell(est),
        ])
        colors.append(render.VERDICT_COLOR[est.verdict])

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    render.heading("Hugging Face results for {!r}".format(args.query))
    render.table(["Model", "Size", "Quant", "Needs", "Verdict"], rows, colors=colors)

    fits = [r for r in results if r.resolved]
    unresolved = len(results) - len(fits)
    if unresolved:
        print()
        print(render.paint(
            "  {} result(s) could not be sized. Gated repos need HF_TOKEN set."
            .format(unresolved), "dim"))
    print()
    print(render.paint("  Size any of these in full with: llmcalculator check <model-id>", "dim"))
    return 0


def cmd_trending(args) -> int:
    """What the Hub is trending right now, sized for this machine."""
    hw = _hardware_from_args(args)
    wl = workloads.get(args.workload)
    try:
        results = hub.trending(limit=args.limit, use_cache=not args.no_cache)
    except RuntimeError as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2

    rows, colors = [], []
    for r in results:
        if not r.resolved:
            continue
        est = recommended_quant(r.spec, hw, wl, context=args.context, device=args.device)
        rows.append([r.hub.id, "{:.1f}B".format(r.spec.params_b), est.quant_name,
                     "{:.1f} GB".format(est.total_gb), render.verdict_cell(est)])
        colors.append(render.VERDICT_COLOR[est.verdict])

    if not rows:
        print("Could not resolve any trending models right now.")
        return 1
    render.heading("Trending on Hugging Face")
    render.table(["Model", "Size", "Quant", "Needs", "Verdict"], rows, colors=colors)
    print()
    return 0


def cmd_cache(args) -> int:
    """Inspect or clear the Hub response cache."""
    d = hub.cache_dir()
    files = list(d.glob("*.json"))
    if args.clear:
        n = hub.clear_cache()
        print("Cleared {} cached response(s) from {}".format(n, d))
        return 0
    total = sum(f.stat().st_size for f in files) if files else 0
    print("Cache directory : {}".format(d))
    print("Cached responses: {}".format(len(files)))
    print("Size            : {:.1f} KB".format(total / 1024))
    print()
    print("Clear it with: llmcalculator cache --clear")
    return 0


def cmd_tui(args) -> int:
    from .ui.tui import run
    return run(_hardware_from_args(args))


def cmd_app(args) -> int:
    from .ui.app import serve
    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


# --- parser ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmcalculator",
        description="Work out which AI models your computer can run, fine-tune, or train.",
        epilog="Run `llmcalculator scan` first if you are not sure where to start.",
    )
    p.add_argument("--version", action="version", version="llmcalculator {}".format(__version__))
    sub = p.add_subparsers(dest="command")

    def common(sp, context_default=8192):
        sp.add_argument("--context", type=_parse_context, default=context_default,
                        metavar="N", help="context length, e.g. 8192 or 8k")
        sp.add_argument("--device", choices=["auto", "gpu", "cpu", "all-gpus"], default="auto",
                        help="which memory budget to size against")
        sp.add_argument("--vram", type=float, metavar="GB",
                        help="assume this much VRAM instead of detecting")
        sp.add_argument("--ram", type=float, metavar="GB",
                        help="assume this much system RAM instead of detecting")
        sp.add_argument("--gpu-name", metavar="NAME",
                        help="GPU model to assume, for speed estimates")
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        return sp

    s = common(sub.add_parser("scan", help="detect your hardware and summarise what it can do"))
    s.set_defaults(func=cmd_scan)

    s = common(sub.add_parser("check", help="check whether one model fits"))
    s.add_argument("model", help="catalog name, or a Hugging Face repo id")
    s.add_argument("--workload", "-w", default="inference",
                   help="inference, qlora, lora, full, or train")
    s.add_argument("--quant", "-q", help="force a quantization format, e.g. Q4_K_M")
    s.add_argument("--batch", type=int, default=1, help="batch size")
    s.add_argument("--kv-quant", default="fp16", help="KV cache precision: fp16, q8_0, q4_0")
    s.add_argument("--all-workloads", "-a", action="store_true",
                   help="show inference, fine-tuning and training together")
    s.add_argument("--hf", action="store_true", help="force a Hugging Face lookup")
    s.set_defaults(func=cmd_check)

    s = common(sub.add_parser("compare", help="compare several models side by side"))
    s.add_argument("models", nargs="+", help="two or more model names")
    s.add_argument("--workload", "-w", default="inference")
    s.add_argument("--quant", "-q", help="use the same format for every model")
    s.set_defaults(func=cmd_compare)

    s = common(sub.add_parser("recommend", help="suggest good models for your machine"))
    s.add_argument("--workload", "-w", default="inference")
    s.add_argument("--limit", "-n", type=int, default=10)
    s.add_argument("--tag", help="only models with this tag or capability, "
                                 "e.g. code, reasoning, tools, vision")
    s.add_argument("--min-speed", type=float, default=0.0, metavar="TOK",
                   help="drop anything slower than this")
    s.set_defaults(func=cmd_recommend)

    s = common(sub.add_parser("context", help="show memory across context lengths"))
    s.add_argument("model")
    s.add_argument("--quant", "-q")
    s.add_argument("--hf", action="store_true")
    s.set_defaults(func=cmd_context)

    s = sub.add_parser("models", help="list or search the model catalog")
    s.add_argument("query", nargs="?", help="filter by name, family, tag or capability")
    s.add_argument("--capability", "-c", metavar="CAP",
                   help="only models with this capability, e.g. code, vision, tools")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("info", help="everything known about one model")
    s.add_argument("model", help="catalog name, or a Hugging Face repo id")
    s.add_argument("--brief", action="store_true",
                   help="list capabilities and formats without the descriptions")
    s.add_argument("--hf", action="store_true", help="force a Hugging Face lookup")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("capabilities",
                       help="explain every capability, weight format and runtime")
    s.add_argument("--kind", choices=["all", "capabilities", "formats", "runtimes"],
                   default="all")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_capabilities)

    s = common(sub.add_parser("search", help="search all of Hugging Face, not just the catalog"))
    s.add_argument("query", help="what to look for, e.g. granite, coder, 7b")
    s.add_argument("--limit", "-n", type=int, default=15)
    s.add_argument("--sort", default="downloads",
                   choices=["downloads", "likes", "trendingScore", "lastModified"])
    s.add_argument("--workload", "-w", default="inference")
    s.add_argument("--gguf", action="store_true", help="include GGUF-only repositories")
    s.add_argument("--no-cache", action="store_true", help="bypass the local cache")
    s.set_defaults(func=cmd_search)

    s = common(sub.add_parser("trending", help="what the Hub is trending, sized for you"))
    s.add_argument("--limit", "-n", type=int, default=15)
    s.add_argument("--workload", "-w", default="inference")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(func=cmd_trending)

    s = sub.add_parser("cache", help="inspect or clear the Hugging Face cache")
    s.add_argument("--clear", action="store_true", help="delete every cached response")
    s.set_defaults(func=cmd_cache)

    s = common(sub.add_parser("tui", help="interactive terminal interface"))
    s.set_defaults(func=cmd_tui)

    s = sub.add_parser("app", help="open the point-and-click app in your browser")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser")
    s.set_defaults(func=cmd_app)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # Bare `llmcalc` should do something useful rather than print usage.
        args = parser.parse_args(["scan"])
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        return 130
    except (KeyError, RuntimeError, ValueError) as exc:
        print(render.paint("Error: {}".format(exc), "red"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
