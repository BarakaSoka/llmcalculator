"""Terminal rendering.

Uses `rich` when it is installed and falls back to plain ASCII when it is not,
so the CLI has no hard dependencies but still looks good for people who have
a normal Python environment.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional, Sequence

from ..estimate import Estimate, Verdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

_WRAPPING_COLUMNS = {"Why", "Reason", "Notes", "Tags", "Largest that fits"}

VERDICT_COLOR = {
    Verdict.EASY: "green",
    Verdict.OK: "cyan",
    Verdict.TIGHT: "yellow",
    Verdict.NO: "red",
}

_ANSI = {"green": "\033[32m", "cyan": "\033[36m", "yellow": "\033[33m",
         "red": "\033[31m", "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m"}


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def paint(text: str, color: str) -> str:
    if not supports_color():
        return text
    return "{}{}{}".format(_ANSI.get(color, ""), text, _ANSI["reset"])


def console():
    return Console() if _RICH else None


def rule(title: str = "") -> None:
    width = min(shutil.get_terminal_size((88, 24)).columns, 100)
    if title:
        line = "-- {} ".format(title).ljust(width, "-")
    else:
        line = "-" * width
    print(paint(line, "dim"))


def heading(text: str) -> None:
    print()
    print(paint(text, "bold"))


def verdict_cell(est: Estimate) -> str:
    return "{} {}".format(Verdict.SYMBOL[est.verdict], est.label())


def bar(fraction: float, width: int = 18) -> str:
    """A utilization bar. Overflow past 100% is shown as a distinct segment."""
    fraction = max(0.0, fraction)
    filled = int(min(fraction, 1.0) * width)
    out = "#" * filled + "." * (width - filled)
    if fraction > 1.0:
        out = out[:width - 1] + ">"
    # Pipes rather than square brackets: rich would parse "[...]" as markup.
    return "|{}|".format(out)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]],
          title: Optional[str] = None, colors: Optional[Sequence[Optional[str]]] = None) -> None:
    """Print a table, coloured per row when `colors` is given."""
    if _RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold",
                  header_style="bold", pad_edge=False)
        for i, h in enumerate(headers):
            justify = "right" if i and _numeric_column(rows, i) else "left"
            # Only prose columns wrap; a wrapped verdict or size is unreadable.
            wraps = h in _WRAPPING_COLUMNS
            t.add_column(h, justify=justify, no_wrap=not wraps,
                         overflow="fold" if wraps else "ellipsis")
        for i, row in enumerate(rows):
            style = colors[i] if colors and i < len(colors) else None
            t.add_row(*[str(c) for c in row], style=style)
        Console().print(t)
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    if title:
        print()
        print(paint(title, "bold"))
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(paint(sep, "dim"))
    for i, row in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(row))
        color = colors[i] if colors and i < len(colors) else None
        print(paint(line, color) if color else line)


def _numeric_column(rows: Sequence[Sequence[str]], idx: int) -> bool:
    for row in rows:
        if idx >= len(row):
            continue
        cell = str(row[idx]).replace(".", "").replace("%", "").replace(" GB", "")
        cell = cell.replace(" tok/s", "").replace("~", "").strip()
        if cell and not cell.replace("-", "").isdigit():
            return False
    return True


def estimate_detail(est: Estimate) -> None:
    """The full breakdown for a single estimate."""
    m, wl = est.model, est.workload
    heading("{}  -  {}".format(m.name, wl.label))
    print(paint("  {}".format(m.describe()), "dim"))
    if m.hf_id:
        print(paint("  {}".format(m.hf_id), "dim"))

    print()
    color = VERDICT_COLOR[est.verdict]
    print("  {}  {}  {:.1f} GB needed of {:.1f} GB available".format(
        paint(verdict_cell(est).ljust(14), color),
        bar(est.utilization),
        est.total_gb, est.budget_gb))

    print()
    print("  Settings   {} quantization, {} token context, batch {}".format(
        est.quant_name, _fmt_ctx(est.context), est.batch))
    if est.tokens_per_sec:
        print("  Speed      ~{:.0f} tokens/sec generation".format(est.tokens_per_sec))
        if est.prefill_tokens_per_sec:
            print("             ~{:,.0f} tokens/sec prompt processing".format(
                est.prefill_tokens_per_sec))
    if est.disk_gb:
        print("  Download   ~{:.1f} GB".format(est.disk_gb))

    print()
    print("  Memory breakdown")
    items = est.breakdown.items_gb()
    widest = max(len(k) for k, _ in items)
    for label, gb in items:
        share = gb / max(est.total_gb, 1e-9)
        print("    {}  {:>7.2f} GB  {}".format(
            label.ljust(widest), gb, paint(bar(share, 12), "dim")))

    if est.notes:
        print()
        for note in est.notes:
            print(paint("  - " + note, "yellow" if not est.fits else "dim"))


def _fmt_ctx(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return "{}k".format(n // 1024)
    return str(n)


def fmt_ctx(n: int) -> str:
    return _fmt_ctx(n)
