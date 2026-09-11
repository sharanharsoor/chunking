#!/usr/bin/env python3
"""Core install smoke: import, two strategies, CLI. CI runs this on windows-latest."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chunking_strategy import __version__, create_chunker

TEXT = "Dr. Muller said hello. The heading stays with the next line.\n\nSecond paragraph."


def main() -> None:
    assert __version__, "missing version"
    a = create_chunker("fixed_size", chunk_size=40)
    b = create_chunker("sentence_based", max_sentences=1)
    ra, rb = a.chunk(TEXT), b.chunk(TEXT)
    assert ra.chunks, "fixed_size returned no chunks"
    assert rb.chunks, "sentence_based returned no chunks"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    listed = subprocess.check_output(
        [sys.executable, "-m", "chunking_strategy", "list-strategies", "--tier", "lab", "--format", "simple"],
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert "fixed_size" in listed, listed
    assert "sentence_based" in listed, listed
    assert "recursive" in listed, listed
    assert "recursive_character" in listed, listed
    print("ok", __version__, "a", len(ra.chunks), "b", len(rb.chunks))


if __name__ == "__main__":
    main()
