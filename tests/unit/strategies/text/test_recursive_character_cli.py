"""CLI: list-strategies and chunk --strategy recursive_character."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SAMPLE = REPO / "test_data" / "sample_simple_text.txt"
EXAMPLE = REPO / "config_examples" / "strategy_configs" / "recursive_character_chunker.yaml"


def _cli(*args, timeout=60):
    return subprocess.run(
        [sys.executable, "-m", "chunking_strategy.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _strategy_used(payload: dict) -> str:
    return payload.get("strategy_used") or (payload.get("metadata") or {}).get("strategy_used")


def test_cli_list_includes_recursive_character():
    result = _cli("list-strategies", "--format", "simple")
    assert result.returncode == 0, result.stderr
    names = result.stdout.split()
    assert "recursive_character" in names
    assert "recursive" in names


def test_cli_chunk_strategy_flag(tmp_path):
    src = SAMPLE if SAMPLE.exists() else tmp_path / "in.txt"
    if not SAMPLE.exists():
        src.write_text("Hello world.\n\nSecond paragraph.\n\nThird.", encoding="utf-8")
    out = tmp_path / "out.json"
    result = _cli(
        "chunk",
        str(src),
        "--strategy", "recursive_character",
        "--chunk-size", "80",
        "--output", str(out),
        "--format", "json",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert _strategy_used(data) == "recursive_character"
    assert data["chunks"]


def test_cli_chunk_from_yaml_config(tmp_path):
    src = SAMPLE if SAMPLE.exists() else tmp_path / "in.txt"
    if not SAMPLE.exists():
        src.write_text("Hello world.\n\nSecond paragraph.\n\nThird.", encoding="utf-8")
    out = tmp_path / "out.json"
    result = _cli(
        "chunk",
        str(src),
        "--config", str(EXAMPLE),
        "--output", str(out),
        "--format", "json",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert _strategy_used(data) == "recursive_character"
    assert data["chunks"]
