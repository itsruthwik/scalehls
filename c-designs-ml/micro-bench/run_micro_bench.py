#!/usr/bin/env python3
"""Run all micro-bench C designs through the default accel C driver."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
RESULTS_DIR = THIS_DIR / "results"
DRIVER = REPO_ROOT / "tools" / "scalehls-c-to-cpp.py"


def discover_designs() -> list[Path]:
    return sorted(
        path for path in THIS_DIR.glob("*.cpp") if path.is_file() and path.parent == THIS_DIR
    )


def run_design(design: Path, print_commands: bool) -> dict:
    artifact_dir = RESULTS_DIR / f"{design.stem}_default"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    cmd = [
        sys.executable,
        str(DRIVER),
        str(design),
        "--pipeline-mode",
        "accel",
        "--gemm-only",
        "--artifact-dir",
        str(artifact_dir),
    ]
    if print_commands:
        print("+ " + " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result_path = artifact_dir / "result.json"
    payload = {
        "design": design.name,
        "artifact_root": str(artifact_dir),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result_json": str(result_path) if result_path.exists() else None,
    }
    if result_path.exists():
        payload["result"] = json.loads(result_path.read_text(encoding="utf-8"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print each invoked driver command",
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [run_design(design, args.print_commands) for design in discover_designs()]
    summary_path = RESULTS_DIR / "summary_default.json"
    summary_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")

    failures = [run for run in runs if run["returncode"] != 0]
    if failures:
        print(f"wrote {summary_path} with {len(failures)} failed run(s)", file=sys.stderr)
        return 1
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
