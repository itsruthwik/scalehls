#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
TOOL = REPO_ROOT / "tools" / "scalehls-c-to-cpp.py"
ARTIFACT_ROOT = (
    REPO_ROOT
    / "conductor"
    / "tracks"
    / "gemm_only_accel_stack_revision_20260414"
    / "artifacts"
    / "c_test_matrix"
)


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for design in sorted(THIS_DIR.glob("*.cpp")):
        artifact_dir = ARTIFACT_ROOT / design.stem
        cmd = [
            "python3",
            str(TOOL),
            str(design),
            "--pipeline-mode",
            "accel",
            "--artifact-dir",
            str(artifact_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        result_path = artifact_dir / "result.json"
        if proc.returncode != 0 or not result_path.exists():
            failures.append(f"{design.name}: pipeline failed")
            continue
        result = json.loads(result_path.read_text())
        if not result.get("frontend_succeeded"):
            failures.append(f"{design.name}: frontend failed")
        if not result.get("mapper_succeeded"):
            failures.append(f"{design.name}: mapper failed")
        if not result.get("mapped"):
            failures.append(f"{design.name}: not mapped")
        if not result.get("emit_succeeded"):
            failures.append(f"{design.name}: emit failed")

    if failures:
        print("INT8 accel test matrix failures:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("INT8 accel test matrix passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
