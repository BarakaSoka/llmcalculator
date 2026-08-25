"""CLI behaviour: exit codes, JSON output, argument parsing."""

import json
import subprocess
import sys

import pytest

from llmcalculator.cli import _parse_context, main


def run(args):
    return subprocess.run([sys.executable, "-m", "llmcalculator.cli"] + args,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("value,expected", [
    ("8192", 8192), ("8k", 8192), ("128k", 131072), ("2K", 2048),
])
def test_context_parsing(value, expected):
    assert _parse_context(value) == expected


def test_context_parsing_rejects_nonsense():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_context("lots")


def test_scan_json_is_valid():
    r = run(["scan", "--json"])
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["ram_gb"] > 0
    assert set(data["capabilities"]) == {"inference", "qlora", "lora", "full", "train"}


def test_check_json_and_exit_code_for_fitting_model():
    r = run(["check", "llama3.2:1b", "--json", "--vram", "24", "--ram", "64"])
    assert r.returncode == 0
    assert json.loads(r.stdout)["fits"] is True


def test_check_exit_code_is_one_when_it_does_not_fit():
    r = run(["check", "llama3.1:405b", "--json", "--vram", "8", "--ram", "16"])
    assert r.returncode == 1
    assert json.loads(r.stdout)["fits"] is False


def test_unknown_model_exits_two():
    r = run(["check", "not-a-model"])
    assert r.returncode == 2
    assert "Unknown model" in r.stderr


def test_compare_json():
    r = run(["compare", "llama3.2:1b", "llama3.1:8b", "--json",
             "--vram", "24", "--ram", "64"])
    assert r.returncode == 0
    assert len(json.loads(r.stdout)) == 2


def test_models_list_json():
    r = run(["models", "--json"])
    assert r.returncode == 0
    assert len(json.loads(r.stdout)) >= 40


def test_recommend_respects_manual_hardware():
    r = run(["recommend", "--json", "--vram", "8", "--ram", "16", "-n", "5"])
    assert r.returncode == 0
    for rec in json.loads(r.stdout):
        assert rec["required_gb"] <= 8.0


def test_bare_invocation_runs_scan():
    r = run([])
    assert r.returncode == 0
    assert "Your machine" in r.stdout


def test_help_works():
    assert run(["--help"]).returncode == 0


# --- Hub-backed commands --------------------------------------------------

def test_models_suggests_hub_search_when_catalog_misses():
    r = run(["models", "definitely-not-in-the-catalog"])
    assert r.returncode == 1
    assert "llmcalculator search" in r.stdout


def test_cache_command_reports_location():
    r = run(["cache"])
    assert r.returncode == 0
    assert "Cache directory" in r.stdout


def test_search_requires_a_query():
    r = run(["search"])
    assert r.returncode == 2   # argparse rejects the missing argument
