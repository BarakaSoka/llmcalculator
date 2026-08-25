"""Terminal UI behaviour, driven headlessly through Textual's test pilot."""

import asyncio

import pytest

textual = pytest.importorskip("textual", reason="TUI needs textual installed")

from llmcalculator.hardware.base import GB, CPU, Accelerator, HardwareProfile
from llmcalculator.ui.tui import DetailPanel, LLMCalcApp


@pytest.fixture
def hw():
    return HardwareProfile(
        cpu=CPU(name="Test CPU", cores=8, threads=16, bandwidth_gbs=50),
        ram_bytes=32 * GB,
        accelerators=[Accelerator(name="RTX 4090", vendor="nvidia",
                                  memory_bytes=24 * GB, bandwidth_gbs=1008.0,
                                  fp16_tflops=165.0)],
    )


def run_app(hw, body):
    async def main():
        app = LLMCalcApp(hw)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await body(app, pilot)
    asyncio.run(asyncio.wait_for(main(), timeout=90))


def test_list_populates(hw):
    async def body(app, pilot):
        assert app.query_one("#models").row_count > 50
    run_app(hw, body)


def test_arrow_keys_move_the_cursor(hw):
    async def body(app, pilot):
        t = app.query_one("#models")
        start = t.cursor_row
        for _ in range(3):
            await pilot.press("down")
        await pilot.pause()
        assert t.cursor_row == start + 3
        await pilot.press("up")
        await pilot.pause()
        assert t.cursor_row == start + 2
    run_app(hw, body)


def test_arrows_reach_the_list_from_the_search_box(hw):
    """Typing a filter then pressing Down is the common case; it must not
    require tabbing back to the table first."""
    async def body(app, pilot):
        t = app.query_one("#models")
        app.query_one("#search").focus()
        await pilot.pause()
        start = t.cursor_row
        for _ in range(4):
            await pilot.press("down")
        await pilot.pause()
        assert t.cursor_row == start + 4
        assert app.focused.id == "search"   # focus must not be stolen
    run_app(hw, body)


def test_detail_panel_follows_the_cursor(hw):
    async def body(app, pilot):
        d = app.query_one(DetailPanel)
        before = d._body()
        for _ in range(5):
            await pilot.press("down")
        await pilot.pause()
        assert d._body() != before
    run_app(hw, body)


def test_letters_still_type_into_the_search_box(hw):
    """j and k are bound to movement, so they must not leak into typing."""
    async def body(app, pilot):
        app.query_one("#search").focus()
        await pilot.pause()
        for ch in "jk":
            await pilot.press(ch)
        await pilot.pause()
        assert app.query_one("#search").value == "jk"
    run_app(hw, body)


def test_vim_keys_move_the_list(hw):
    async def body(app, pilot):
        t = app.query_one("#models")
        await pilot.press("G")
        await pilot.pause()
        assert t.cursor_row == t.row_count - 1
        await pilot.press("g")
        await pilot.pause()
        assert t.cursor_row == 0
        await pilot.press("j")
        await pilot.pause()
        assert t.cursor_row == 1
    run_app(hw, body)


def test_home_end_and_paging_from_the_search_box(hw):
    async def body(app, pilot):
        t = app.query_one("#models")
        app.query_one("#search").focus()
        await pilot.pause()
        await pilot.press("end")
        await pilot.pause()
        assert t.cursor_row == t.row_count - 1
        await pilot.press("home")
        await pilot.pause()
        assert t.cursor_row == 0
        await pilot.press("pagedown")
        await pilot.pause()
        assert t.cursor_row == 10
    run_app(hw, body)


def test_cursor_stays_in_bounds(hw):
    async def body(app, pilot):
        t = app.query_one("#models")
        app.query_one("#search").focus()
        await pilot.pause()
        for _ in range(3):
            await pilot.press("up")
        await pilot.pause()
        assert t.cursor_row == 0
        for _ in range(300):
            await pilot.press("down")
        await pilot.pause()
        assert t.cursor_row == t.row_count - 1
    run_app(hw, body)


def test_enter_and_escape_hand_focus_to_the_list(hw):
    async def body(app, pilot):
        app.query_one("#search").focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.focused.id == "models"
        app.query_one("#search").focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused.id == "models"
    run_app(hw, body)


def test_filtering_narrows_the_list(hw):
    async def body(app, pilot):
        t = app.query_one("#models")
        full = t.row_count
        app.query_one("#search").value = "coder"
        await pilot.pause()
        assert 0 < t.row_count < full
    run_app(hw, body)


def test_workload_cycling_changes_the_estimate(hw):
    async def body(app, pilot):
        d = app.query_one(DetailPanel)
        before = d._body()
        for _ in range(3):
            await pilot.press("w")
        await pilot.pause()
        assert app.workload_key != "inference"
        assert d._body() != before
    run_app(hw, body)


def test_navigation_on_an_empty_list_does_not_crash(hw):
    async def body(app, pilot):
        app.query_one("#search").value = "zzzz-no-such-model"
        await pilot.pause()
        assert app.query_one("#models").row_count == 0
        for key in ("down", "up", "home", "end", "pagedown"):
            await pilot.press(key)
        await pilot.pause()   # must not raise
    run_app(hw, body)
