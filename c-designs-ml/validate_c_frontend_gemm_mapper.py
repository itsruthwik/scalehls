#!/usr/bin/env python3
"""Validate the canonical GEMM-mapped C samples through the C frontend."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from utils import CFrontendRunResult, run_c_frontend, write_result_json


TRACK_ARTIFACT_ROOT = (
    REPO_ROOT
    / "conductor"
    / "tracks"
    / "c_frontend_gemm_mapper_validation_20260411"
    / "artifacts"
    / "c_design_validation"
)

TEST_DIR = THIS_DIR / "test"
ARCHIVE_DIR = THIS_DIR / "archive"


def resolve_design_path(name: str) -> Path:
    path_name = name if name.endswith(".cpp") else f"{name}.cpp"
    for base in (TEST_DIR, THIS_DIR, ARCHIVE_DIR):
        candidate = base / path_name
        if candidate.exists():
            return candidate
    return THIS_DIR / path_name


def discover_designs(designs: list[str] | None) -> list[Path]:
    if designs:
        return [resolve_design_path(name) for name in designs]
    return [
        TEST_DIR / "gemmv_fp32.cpp",
        TEST_DIR / "gemm_fp32.cpp",
        TEST_DIR / "conv2d_fp32.cpp",
    ]


def summarize_result(result: CFrontendRunResult) -> dict:
    mapper_status = "not_run"
    if result.pipeline_mode in {"gemm", "accel"}:
        if not result.mapper_succeeded:
            mapper_status = "failed"
        elif result.mapped:
            mapper_status = "mapped"
        else:
            mapper_status = "unmapped"
    return {
        "design": result.input_path.stem,
        "source": str(result.input_path),
        "frontend_succeeded": result.frontend_succeeded,
        "mapper_succeeded": result.mapper_succeeded,
        "emit_succeeded": result.emit_succeeded,
        "mapper_status": mapper_status,
        "mapped": result.mapped,
        "mapped_symbols": result.mapped_symbols or [],
        "skip_reason": result.skip_reason,
        "failure_stage": result.failure_stage,
        "failure_summary": result.failure_summary,
        "candidate_entries": result.candidate_entries or [],
        "manifest_paths": result.manifest_paths or [],
        "artifact_root": str(result.artifact_root),
    }


def render_markdown_report(summary: dict) -> str:
    lines = [
        "# C Frontend Accelerator Mapper Validation",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Designs evaluated: {summary['counts']['total']}",
        f"- Frontend lowering succeeded: {summary['counts']['frontend_ok']}/{summary['counts']['total']}",
        f"- Accelerator mapper rewrote designs: {summary['counts']['mapped']}/{summary['counts']['total']}",
        f"- Accelerator mapper reported unmapped designs: {summary['counts']['unmapped']}/{summary['counts']['total']}",
        f"- Accelerator mapper failures: {summary['counts']['mapper_failed']}/{summary['counts']['total']}",
        f"- HLS C++ emission succeeded: {summary['counts']['emit_ok']}/{summary['counts']['total']}",
        "",
        "| Design | Frontend | Mapper | Emit | Mapped Symbols | Reason / Failure |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        mapper = result["mapper_status"]
        reason = result["skip_reason"] or result["failure_summary"] or ""
        mapped_symbols = ", ".join(result["mapped_symbols"]) if result["mapped_symbols"] else ""
        lines.append(
            f"| {result['design']} | {'ok' if result['frontend_succeeded'] else 'fail'} "
            f"| {mapper} | {'ok' if result['emit_succeeded'] else 'fail'} "
            f"| {mapped_symbols} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Validation uses the reusable C frontend harness in `tools/scalehls-c-to-cpp.py` with `--pipeline-mode accel`.",
            "- The active design set is limited to canonical GEMM-mapped C samples under `c-designs-ml/test/`.",
            "- Archived older or exploratory C designs are kept under `c-designs-ml/archive/` when needed.",
            "",
            "## Per-Design Artifacts",
            "",
        ]
    )
    for result in summary["results"]:
        lines.append(f"### {result['design']}")
        lines.append("")
        lines.append(f"- Artifact root: `{result['artifact_root']}`")
        if result["failure_stage"]:
            lines.append(f"- Failure stage: `{result['failure_stage']}`")
        if result["manifest_paths"]:
            lines.append(f"- Manifest files: `{', '.join(result['manifest_paths'])}`")
        if result["candidate_entries"]:
            lines.append(f"- Candidate log entries: `{json.dumps(result['candidate_entries'], sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_summary(results: list[dict]) -> dict:
    counts = {
        "total": len(results),
        "frontend_ok": sum(1 for result in results if result["frontend_succeeded"]),
        "mapped": sum(1 for result in results if result["mapper_status"] == "mapped"),
        "unmapped": sum(1 for result in results if result["mapper_status"] == "unmapped"),
        "mapper_failed": sum(1 for result in results if result["mapper_status"] == "failed"),
        "emit_ok": sum(1 for result in results if result["emit_succeeded"]),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=TRACK_ARTIFACT_ROOT)
    parser.add_argument("--designs", nargs="+", help="Subset of design stems to evaluate")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args(argv)

    design_paths = discover_designs(args.designs)
    missing = [str(path) for path in design_paths if not path.exists()]
    if missing:
        print(f"missing design inputs: {', '.join(missing)}", file=sys.stderr)
        return 2

    results: list[dict] = []
    for design_path in design_paths:
        artifact_root = args.output_root.resolve() / "designs" / design_path.stem
        run_result = run_c_frontend(
            design_path,
            top_func=design_path.stem,
            artifact_dir=artifact_root,
            repo_root=args.repo_root,
            pipeline_mode="accel",
            axi_interface=False,
            print_commands=args.print_commands,
        )
        write_result_json(run_result)
        summary = summarize_result(run_result)
        results.append(summary)
        print(
            f"{summary['design']}: frontend={'ok' if summary['frontend_succeeded'] else 'fail'} "
            f"mapper={summary['mapper_status']} emit={'ok' if summary['emit_succeeded'] else 'fail'}"
        )

    aggregate = build_summary(results)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.md").write_text(
        render_markdown_report(aggregate),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
