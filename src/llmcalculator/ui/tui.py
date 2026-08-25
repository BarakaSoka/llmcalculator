"""Interactive terminal interface.

Browse the catalog, switch workload and context, and watch the verdict change
live. Requires `textual`; `llmcalc tui` explains how to install it if missing.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .. import compare, workloads
from ..estimate import Verdict, recommended_quant
from ..hardware.base import HardwareProfile
from ..models import catalog, hub
from ..models.spec import ModelSpec

_MISSING = """The TUI needs the `textual` package.

    pip install "llmcalculator[tui]"

Everything else works without it - try `llmcalculator scan` or `llmcalculator app`."""


def run(hardware: Optional[HardwareProfile] = None) -> int:
    try:
        import textual  # noqa: F401
    except ImportError:
        print(_MISSING, file=sys.stderr)
        return 3
    from ..hardware import detect
    app = LLMCalcApp(hardware or detect())
    app.run()
    return 0


try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import (
        DataTable, Footer, Header, Input, Label, ProgressBar, Select, Static,
    )
    _HAS_TEXTUAL = True
except ImportError:  # pragma: no cover - exercised only without textual
    _HAS_TEXTUAL = False
    App = object
    ComposeResult = object


if _HAS_TEXTUAL:

    VERDICT_STYLE = {
        Verdict.EASY: "bold green",
        Verdict.OK: "bold cyan",
        Verdict.TIGHT: "bold yellow",
        Verdict.NO: "bold red",
    }

    class HardwarePanel(Static):
        """Fixed summary of the detected machine."""

        def __init__(self, hw: HardwareProfile) -> None:
            super().__init__(id="hardware")
            self.hw = hw

        def render(self) -> str:
            hw = self.hw
            lines = ["[bold]Your machine[/bold]"]
            lines.append("  CPU  {}".format(hw.cpu.name))
            lines.append("  RAM  {:.0f} GB".format(hw.ram_gb))
            for a in hw.accelerators:
                if a.vendor == "cpu":
                    continue
                extra = " unified" if a.unified_memory else ""
                lines.append("  GPU  {} - {:.0f} GB{}".format(a.name, a.memory_gb, extra))
            if not hw.has_gpu:
                lines.append("  GPU  [yellow]none - CPU only[/yellow]")
            lines.append("")
            lines.append("  Budget [bold]{:.1f} GB[/bold]".format(
                hw.budget_bytes("auto") / (1024 ** 3)))
            return "\n".join(lines)

    class DetailPanel(VerticalScroll):
        """Everything about the currently selected model."""

        def __init__(self, hw: HardwareProfile) -> None:
            super().__init__(id="detail")
            self.hw = hw
            self.model: Optional[ModelSpec] = None
            self.workload = workloads.INFERENCE
            self.context = 8192

        def compose(self) -> ComposeResult:
            yield Static("", id="detail-body")

        def update_for(self, model: ModelSpec, workload, context: int) -> None:
            self.model, self.workload, self.context = model, workload, context
            self.query_one("#detail-body", Static).update(self._body())

        def _body(self) -> str:
            if self.model is None:
                return "Select a model to see the breakdown."
            est = recommended_quant(self.model, self.hw, self.workload,
                                    context=self.context)
            style = VERDICT_STYLE[est.verdict]
            out = [
                "[bold]{}[/bold]".format(est.model.name),
                "[dim]{}[/dim]".format(est.model.describe()),
                "",
                "[{}]{}[/{}]  {:.1f} GB of {:.1f} GB".format(
                    style, est.label(), style, est.total_gb, est.budget_gb),
                _bar_markup(est.utilization),
                "",
                "[bold]Settings[/bold]",
                "  {}  |  {} context  |  {}".format(
                    est.quant_name, _fmt_ctx(est.context), est.workload.label),
            ]
            if est.tokens_per_sec:
                out.append("  ~{:.0f} tok/s generation".format(est.tokens_per_sec))
            if est.disk_gb:
                out.append("  ~{:.1f} GB download".format(est.disk_gb))

            out += ["", "[bold]Memory breakdown[/bold]"]
            for label, gb in est.breakdown.items_gb():
                pct = gb / max(est.total_gb, 1e-9)
                out.append("  {:<13} {:>6.2f} GB  [dim]{}[/dim]".format(
                    label, gb, _bar(pct, 14)))

            others = [e for e in compare.workload_table(
                self.model, self.hw, context=min(self.context, 4096))]
            out += ["", "[bold]Across workloads[/bold]"]
            for e in others:
                st = VERDICT_STYLE[e.verdict]
                out.append("  {:<20} {:>7.1f} GB  [{}]{}[/{}]".format(
                    e.workload.label, e.total_gb, st, e.label(), st))

            if est.notes:
                out += ["", "[bold]Notes[/bold]"]
                for n in est.notes:
                    out.append("  [dim]- {}[/dim]".format(n))
            return "\n".join(out)

    class LLMCalcApp(App):
        """The `llmcalc tui` application."""

        CSS = """
        Screen { layout: vertical; }
        #controls { height: 3; padding: 0 1; }
        #controls Select { width: 26; }
        #controls Input { width: 1fr; }
        #main { height: 1fr; }
        #sidebar { width: 34; border-right: solid $panel-darken-2; padding: 1; }
        #hardware { height: auto; margin-bottom: 1; }
        #models { height: 1fr; }
        #detail { width: 1fr; padding: 1 2; }
        DataTable { height: 1fr; }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("/", "focus_search", "Search"),
            Binding("w", "cycle_workload", "Workload"),
            Binding("c", "cycle_context", "Context"),
            Binding("r", "show_recommended", "Recommended"),
            Binding("a", "show_all", "All models"),
            Binding("h", "search_hub", "Hugging Face"),
        ]

        CONTEXTS = [2048, 4096, 8192, 16384, 32768, 131072]

        workload_key = reactive("inference")
        context = reactive(8192)

        def __init__(self, hw: HardwareProfile) -> None:
            super().__init__()
            self.hw = hw
            self.title = "llmcalculator"
            self.sub_title = "what your machine can run"
            self._models: List[ModelSpec] = catalog.all_models()
            self._filter = ""
            self._hub_specs: dict = {}  # Hub results, keyed by repo id

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="controls"):
                yield Select([(w.label, w.key) for w in workloads.ALL],
                             value="inference", id="workload", allow_blank=False)
                yield Select([(_fmt_ctx(c) + " context", c) for c in self.CONTEXTS],
                             value=8192, id="context", allow_blank=False)
                yield Input(placeholder="Filter models (name, family or tag)...", id="search")
            with Horizontal(id="main"):
                with Vertical(id="sidebar"):
                    yield HardwarePanel(self.hw)
                    yield DataTable(id="models", cursor_type="row", zebra_stripes=True)
                yield DetailPanel(self.hw)
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#models", DataTable)
            table.add_columns("Model", "Size", "Fit")
            self._refresh_models()
            table.focus()

        # --- data ---------------------------------------------------------

        def _current_workload(self):
            return workloads.get(self.workload_key)

        def _refresh_models(self) -> None:
            table = self.query_one("#models", DataTable)
            table.clear()
            wl = self._current_workload()
            ctx = self.context if wl.key == "inference" else min(self.context, 2048)

            models = catalog.search(self._filter) if self._filter else catalog.all_models()
            self._models = models
            for m in models:
                est = recommended_quant(m, self.hw, wl, context=ctx)
                style = VERDICT_STYLE[est.verdict]
                table.add_row(
                    m.name,
                    "{:.1f}B".format(m.params_b),
                    "[{}]{}[/{}]".format(style, Verdict.SYMBOL[est.verdict], style),
                    key=m.name,
                )
            if models:
                self._show(models[0])

        def _show(self, model: ModelSpec) -> None:
            wl = self._current_workload()
            ctx = self.context if wl.key == "inference" else min(self.context, 2048)
            self.query_one(DetailPanel).update_for(model, wl, ctx)

        # --- events -------------------------------------------------------

        def on_data_table_row_highlighted(self, event) -> None:
            if not (event.row_key and event.row_key.value):
                return
            key = str(event.row_key.value)
            spec = self._hub_specs.get(key)
            if spec is not None:
                self._show(spec)
                return
            try:
                self._show(catalog.get(key))
            except KeyError:
                pass

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "search":
                self._filter = event.value.strip()
                self._refresh_models()

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "workload":
                self.workload_key = str(event.value)
            elif event.select.id == "context":
                self.context = int(event.value)
            self._refresh_models()

        # --- actions ------------------------------------------------------

        def action_focus_search(self) -> None:
            self.query_one("#search", Input).focus()

        def action_cycle_workload(self) -> None:
            keys = [w.key for w in workloads.ALL]
            nxt = keys[(keys.index(self.workload_key) + 1) % len(keys)]
            self.query_one("#workload", Select).value = nxt

        def action_cycle_context(self) -> None:
            i = self.CONTEXTS.index(self.context) if self.context in self.CONTEXTS else 2
            self.query_one("#context", Select).value = self.CONTEXTS[(i + 1) % len(self.CONTEXTS)]

        def action_show_recommended(self) -> None:
            """Filter the list down to models actually worth running here."""
            recs = compare.recommend(self.hw, self._current_workload(),
                                     context=self.context, limit=12)
            names = {r.estimate.model.name for r in recs}
            table = self.query_one("#models", DataTable)
            table.clear()
            for r in recs:
                e = r.estimate
                style = VERDICT_STYLE[e.verdict]
                table.add_row(e.model.name, "{:.1f}B".format(e.model.params_b),
                              "[{}]{}[/{}]".format(style, Verdict.SYMBOL[e.verdict], style),
                              key=e.model.name)
            if recs:
                self._show(recs[0].estimate.model)
            self.notify("Showing {} models suited to this machine".format(len(names)))

        def action_search_hub(self) -> None:
            """Search the whole Hub for whatever is in the filter box."""
            query = self._filter.strip()
            if not query:
                self.notify("Type a search first, then press h", severity="warning")
                self.query_one("#search", Input).focus()
                return
            self.notify("Searching Hugging Face for {!r}...".format(query))
            self.run_worker(self._do_hub_search(query), exclusive=True)

        async def _do_hub_search(self, query: str) -> None:
            """Network work belongs off the UI thread; a worker keeps it responsive."""
            import asyncio

            wl = self._current_workload()
            ctx = self.context if wl.key == "inference" else min(self.context, 2048)
            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: hub.search_resolved(query, limit=25))
            except RuntimeError as exc:
                self.notify(str(exc), severity="error")
                return

            table = self.query_one("#models", DataTable)
            table.clear()
            self._hub_specs = {}
            shown = 0
            for r in results:
                if not r.resolved:
                    continue
                est = recommended_quant(r.spec, self.hw, wl, context=ctx)
                style = VERDICT_STYLE[est.verdict]
                self._hub_specs[r.hub.id] = r.spec
                table.add_row(r.hub.id, "{:.1f}B".format(r.spec.params_b),
                              "[{}]{}[/{}]".format(style, Verdict.SYMBOL[est.verdict], style),
                              key=r.hub.id)
                shown += 1
            if shown:
                first = next(iter(self._hub_specs.values()))
                self._show(first)
                self.notify("{} Hub models sized. Press a to return to the catalog.".format(shown))
            else:
                self.notify("Nothing on the Hub could be sized for {!r}".format(query),
                            severity="warning")

        def action_show_all(self) -> None:
            self._filter = ""
            self._hub_specs = {}
            self.query_one("#search", Input).value = ""
            self._refresh_models()


def _bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, fraction)
    filled = int(min(fraction, 1.0) * width)
    out = "#" * filled + "." * (width - filled)
    if fraction > 1.0:
        out = out[:width - 1] + ">"
    return out


def _bar_markup(fraction: float, width: int = 30) -> str:
    color = "green" if fraction <= 0.7 else "yellow" if fraction <= 1.0 else "red"
    return "[{}]{}[/{}]  {:.0f}%".format(color, _bar(fraction, width), color, fraction * 100)


def _fmt_ctx(n: int) -> str:
    return "{}k".format(n // 1024) if n >= 1024 and n % 1024 == 0 else str(n)
