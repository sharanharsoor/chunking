"""compare CLI prints strategy, n_chunks, avg_size, quality_score, elapsed."""

from click.testing import CliRunner

from chunking_strategy.cli import main


def test_compare_json(tmp_path):
    src = tmp_path / "hello.txt"
    src.write_text("Hello. World. Done.\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["compare", str(src), "-s", "sentence_based,fixed_size", "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    assert "sentence_based" in result.output
    assert "n_chunks" in result.output
    assert "quality_score" in result.output
