"""CLI for regex_custom."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "config_examples" / "strategy_configs" / "regex_custom_chunker.yaml"


def _cli(*args, timeout=60):
    return subprocess.run(
        [sys.executable, "-m", "chunking_strategy.cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_list_includes_regex_custom():
    result = _cli("list-strategies", "--format", "simple")
    assert result.returncode == 0, result.stderr
    assert "regex_custom" in result.stdout.split()


def test_cli_chunk_from_yaml(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n\ntwo\n\nthree", encoding="utf-8")
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
    used = data.get("strategy_used") or (data.get("metadata") or {}).get("strategy_used")
    assert used == "regex_custom"
    assert data["chunks"]
