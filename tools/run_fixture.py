#!/usr/bin/env python3
"""Run golden fixtures. --write regenerates expected.json; CI uses --check only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_ROOT = REPO_ROOT / "fixtures"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _input_path(folder: Path) -> Path:
    for name in ("input.txt", "input.bin"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No input.txt or input.bin in {folder}")


def _load_params(folder: Path) -> dict:
    return json.loads((folder / "params.json").read_text(encoding="utf-8"))


def run_one(folder: Path, *, write: bool) -> str:
    from chunking_strategy import create_chunker
    from chunking_strategy.core.canonical import sha256_bytes

    params_doc = _load_params(folder)
    strategy = params_doc["strategy"]
    params = dict(params_doc.get("params") or {})
    source_path = _input_path(folder)
    raw = source_path.read_bytes()
    chunker = create_chunker(strategy, **params)
    if source_path.suffix == ".bin":
        result = chunker.chunk(raw, source_info={"source": source_path.name})
    else:
        result = chunker.chunk(
            raw.decode("utf-8"),
            source_info={"source": source_path.name},
        )
    source = {
        "name": source_path.name,
        "size": len(raw),
        "sha256": sha256_bytes(raw),
    }
    actual = result.to_canonical_json(fixture=True, params=params, source=source)
    expected_path = folder / "expected.json"
    if write:
        expected_path.write_text(actual, encoding="utf-8", newline="\n")
        return actual
    if not expected_path.exists():
        raise FileNotFoundError(f"Missing {expected_path}; run with --write")
    expected = expected_path.read_text(encoding="utf-8")
    if actual != expected:
        raise AssertionError(
            f"Fixture mismatch in {folder.relative_to(REPO_ROOT)}\n"
            "Regenerate with: python tools/run_fixture.py "
            f"{folder.relative_to(REPO_ROOT)} --write"
        )
    return actual


def iter_fixture_dirs(root: Path) -> list:
    dirs = []
    if not root.exists():
        return dirs
    for params in sorted(root.rglob("params.json")):
        folder = params.parent
        try:
            _input_path(folder)
        except FileNotFoundError:
            continue
        dirs.append(folder)
    return dirs


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        help="Fixture folder relative to repo root, or omit to run all",
    )
    parser.add_argument("--write", action="store_true", help="Regenerate expected.json")
    parser.add_argument("--check", action="store_true", help="Compare only (default)")
    args = parser.parse_args(argv)

    write = args.write
    if args.folder:
        folder = (REPO_ROOT / args.folder).resolve()
        if not folder.is_dir():
            # also allow path relative to fixtures/
            alt = (FIXTURES_ROOT / args.folder).resolve()
            folder = alt if alt.is_dir() else folder
        run_one(folder, write=write)
        print(("wrote " if write else "ok ") + str(folder.relative_to(REPO_ROOT)))
        return 0

    failures = []
    ran = 0
    for folder in iter_fixture_dirs(FIXTURES_ROOT):
        ran += 1
        try:
            run_one(folder, write=write)
            print(("wrote " if write else "ok ") + str(folder.relative_to(REPO_ROOT)))
        except Exception as exc:
            failures.append(f"{folder.relative_to(REPO_ROOT)}: {exc}")
            print(f"FAIL {folder.relative_to(REPO_ROOT)}: {exc}", file=sys.stderr)
    if not ran:
        print("no fixtures found", file=sys.stderr)
        return 1
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
