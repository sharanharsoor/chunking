"""Golden fixtures stay in lockstep with Python."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_all_fixtures_match():
    script = REPO / "tools" / "run_fixture.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
