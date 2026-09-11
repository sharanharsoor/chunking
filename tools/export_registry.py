#!/usr/bin/env python3
"""Stub: validate schemas/strategy-tiers.yaml. Full registry merge comes later."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TIERS_PATH = REPO_ROOT / "schemas" / "strategy-tiers.yaml"
VALID = {"lab", "lab_later", "python_only"}


def load_tiers() -> dict:
    data = yaml.safe_load(TIERS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{TIERS_PATH} must be a mapping of tier -> names")
    return data


def check() -> None:
    data = load_tiers()
    seen = {}
    for tier, names in data.items():
        if tier not in VALID:
            raise ValueError(f"Unknown tier {tier!r}")
        if not isinstance(names, list):
            raise ValueError(f"Tier {tier} must be a list")
        for name in names:
            if name in seen:
                raise ValueError(f"{name} listed in both {seen[name]} and {tier}")
            seen[name] = tier
    lab = set(data.get("lab") or [])
    later = set(data.get("lab_later") or [])
    if "recursive" in lab:
        raise ValueError("recursive must not be in lab (homepage); see §7.6")
    if "token_based" in lab:
        raise ValueError("token_based must not be in lab until tiktoken is in the tab")
    if "token_based" not in later:
        raise ValueError("token_based must stay in lab_later until tiktoken is in the tab")
    required_lab = {
        "fixed_size",
        "sentence_based",
        "paragraph_based",
        "overlapping_window",
        "markdown_chunker",
        "python_code",
        "json_chunker",
        "csv_chunker",
        "xml_html_chunker",
        "javascript_code",
        "css_code",
        "go_code",
        "java_code",
        "c_cpp_code",
        "rolling_hash",
        "fixed_length_word",
        "fastcdc",
    }
    missing = required_lab - lab
    if missing:
        raise ValueError(f"lab tier missing {sorted(missing)}")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        check()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"ok {TIERS_PATH.relative_to(REPO_ROOT)}")
    if not args.check:
        print("registry JSON generation not implemented yet (Gate 0 stub)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
