#!/usr/bin/env python3
"""Run all micro-bench C designs through the default accel C driver."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
RESULTS_DIR = THIS_DIR / "results"
DRIVER = REPO_ROOT / "tools" / "scalehls-c-to-cpp.py"


def discover_designs() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.cpp", "*.c"):
        paths.extend(
            [path for path in THIS_DIR.glob(pattern) if path.is_file() and path.parent == THIS_DIR]
        )
    return sorted(paths)


def run_design(design: Path, print_commands: bool) -> dict:
    artifact_dir = RESULTS_DIR / f"{design.stem}_default"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    cmd = [
        sys.executable,
        str(DRIVER),
        str(design),
        "--pipeline-mode",
        "accel",
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
        "result_json": str(result_path) if result_path.exists() else None,
        "stdout_preview": completed.stdout[:4000],
        "stderr_preview": completed.stderr[:4000],
    }
    if result_path.exists():
        import json
        payload["result"] = json.loads(result_path.read_text(encoding="utf-8"))
    return payload


def _status_text(run: dict) -> str:
    result = run.get("result")
    if run["returncode"] != 0:
        return "fail"
    if not result:
        return "no-result"
    if result.get("mapped"):
        return "mapped"
    if result.get("emit_succeeded"):
        return "ok"
    return "unmapped"


def _mapped_symbols_text(run: dict) -> str:
    result = run.get("result") or {}
    symbols = result.get("mapped_symbols") or []
    return ", ".join(symbols) if symbols else "-"


def _reason_text(run: dict) -> str:
    result = run.get("result") or {}
    return (
        result.get("skip_reason")
        or result.get("failure_summary")
        or (run.get("stderr_preview") or "").splitlines()[0] if run.get("stderr_preview") else ""
        or "-"
    )


def _candidate_count(run: dict) -> int:
    result = run.get("result") or {}
    return len(result.get("candidate_entries") or [])


def build_summary_report(runs: list[dict]) -> str:
    headers = ["Design", "Status", "Candidates", "Mapped Symbols", "Reason"]
    rows = [
        [
            run["design"],
            _status_text(run),
            str(_candidate_count(run)),
            _mapped_symbols_text(run),
            _reason_text(run),
        ]
        for run in runs
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = []
    lines.append("ScaleHLS Micro-Bench Summary")
    lines.append("===========================")
    lines.append("")
    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(fmt_row(row))
    lines.append("")
    lines.append("Design Details")
    lines.append("==============")
    lines.append("")

    for run in runs:
        result = run.get("result") or {}
        lines.append(f"{run['design']}")
        lines.append("-" * len(run["design"]))
        lines.append(f"artifact_root: {run['artifact_root']}")
        lines.append(f"returncode: {run['returncode']}")
        lines.append(f"status: {_status_text(run)}")
        lines.append(f"candidate_count: {_candidate_count(run)}")
        lines.append(f"mapped_symbols: {_mapped_symbols_text(run)}")
        if result.get("manifest_paths"):
            lines.append("manifest_paths:")
            for path in result["manifest_paths"]:
                lines.append(f"  - {path}")
        if result.get("candidate_report"):
            lines.append(f"candidate_report: {result['candidate_report']}")
        if result.get("manifest_file"):
            lines.append(f"manifest_file: {result['manifest_file']}")
        if result.get("skip_reason"):
            lines.append(f"skip_reason: {result['skip_reason']}")
        if result.get("failure_stage"):
            lines.append(f"failure_stage: {result['failure_stage']}")
        if result.get("failure_summary"):
            lines.append(f"failure_summary: {result['failure_summary']}")
        if run.get("stderr_preview"):
            lines.append("stderr_preview:")
            for line in run["stderr_preview"].strip().splitlines()[:10]:
                lines.append(f"  {line}")
        lines.append("")

    return "\n".join(lines)


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
    legacy_summary_path = RESULTS_DIR / "summary_default.json"
    legacy_summary_path.unlink(missing_ok=True)
    summary_path = RESULTS_DIR / "summary_default.txt"
    summary_path.write_text(build_summary_report(runs), encoding="utf-8")

    failures = [run for run in runs if run["returncode"] != 0]
    if failures:
        print(f"wrote {summary_path} with {len(failures)} failed run(s)", file=sys.stderr)
        return 1
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
