"""YAML + orchestrator for regex_custom."""

from pathlib import Path

import yaml

from chunking_strategy import create_chunker
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.strategies.text.regex_custom_chunker import RegexCustomChunker

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "config_examples" / "strategy_configs" / "regex_custom_chunker.yaml"


def test_example_yaml_loads():
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert data["strategies"]["primary"] == "regex_custom"
    chunker = RegexCustomChunker(**data["regex_custom"])
    assert chunker.pattern == "\\n\\n"
    via = create_chunker("regex_custom", **data["regex_custom"])
    assert isinstance(via, RegexCustomChunker)


def test_orchestrator_yaml_primary(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n\ntwo\n\nthree", encoding="utf-8")
    orch = ChunkerOrchestrator(config_path=EXAMPLE, enable_hardware_optimization=False)
    result = orch.chunk_file(src)
    assert result.strategy_used == "regex_custom"
    assert len(result.chunks) >= 2
