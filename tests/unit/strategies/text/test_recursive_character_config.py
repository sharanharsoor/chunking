"""YAML + orchestrator wiring for recursive_character."""

from pathlib import Path

import yaml

from chunking_strategy import create_chunker
from chunking_strategy.orchestrator import ChunkerOrchestrator
from chunking_strategy.strategies.text.recursive_character_chunker import RecursiveCharacterChunker

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "config_examples" / "strategy_configs" / "recursive_character_chunker.yaml"
SAMPLE = REPO / "test_data" / "sample_simple_text.txt"


def test_example_yaml_loads_and_constructs():
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert data["strategies"]["primary"] == "recursive_character"
    params = data["recursive_character"]
    chunker = RecursiveCharacterChunker(**params)
    assert chunker.chunk_size == 1000
    via_factory = create_chunker("recursive_character", **params)
    assert isinstance(via_factory, RecursiveCharacterChunker)


def test_orchestrator_uses_yaml_primary_and_parameters(tmp_path):
    text = SAMPLE.read_text(encoding="utf-8") if SAMPLE.exists() else (
        "Hello world.\n\nSecond paragraph lives here.\n\nThird block of prose."
    )
    src = tmp_path / "in.txt"
    src.write_text(text, encoding="utf-8")
    orch = ChunkerOrchestrator(
        config_path=EXAMPLE,
        enable_hardware_optimization=False,
    )
    result = orch.chunk_file(src)
    assert result.strategy_used == "recursive_character"
    assert result.chunks
    chunker = orch._get_chunker("recursive_character")
    assert isinstance(chunker, RecursiveCharacterChunker)
    assert chunker.chunk_size == 1000
