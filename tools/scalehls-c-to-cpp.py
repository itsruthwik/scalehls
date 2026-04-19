#!/usr/bin/env python3
"""Run a C/C++ frontend input through the ScaleHLS pipeline and emit HLS C++."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import run_c_frontend, write_result_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="C/C++ source file to lower")
    parser.add_argument("--top-func", help="Entry function name; defaults to the input stem")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for emitted HLS C++; stdout when omitted",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Keep lowered MLIR, mapped MLIR, logs, and reports here",
    )
    parser.add_argument("--repo-root", type=Path, help="Repository root override")
    parser.add_argument(
        "--pipeline-mode",
        choices=("default", "accel"),
        default="default",
        help="ScaleHLS pipeline variant to run",
    )
    parser.add_argument(
        "--axi-interface",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass axi-interface=<bool> into the default C pipeline",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print subprocess commands to stderr",
    )
    parser.add_argument(
        "--ip-size",
        help="GEMM IP tile size as MxNxK for accel pipeline mode",
    )
    parser.add_argument(
        "--serial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one tiled GEMM helper across all output tiles in accel pipeline mode",
    )
    parser.add_argument(
        "--allow-large-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow affine semantic GEMM normalization to materialize large operand temporaries; pass --no-allow-large-normalization to enable conservative size-aware skipping",
    )
    parser.add_argument(
        "--max-normalized-operand-elements",
        type=int,
        help="Maximum operand temporary elements allowed before conservative size-aware skipping rejects normalization",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_c_frontend(
        args.input,
        top_func=args.top_func,
        artifact_dir=args.artifact_dir,
        repo_root=args.repo_root,
        pipeline_mode=args.pipeline_mode,
        axi_interface=args.axi_interface,
        print_commands=args.print_commands,
        ip_size=args.ip_size,
        serial=args.serial,
        allow_large_normalization=args.allow_large_normalization,
        max_normalized_operand_elements=args.max_normalized_operand_elements,
    )
    write_result_json(result)

    if not result.emit_succeeded:
        print(result.failure_summary or "pipeline failed", file=sys.stderr)
        return 1

    emitted_cpp = Path(result.emitted_cpp) if result.emitted_cpp is not None else None
    if args.output is not None and emitted_cpp is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(emitted_cpp.read_text(encoding="utf-8"), encoding="utf-8")
    elif emitted_cpp is not None:
        sys.stdout.write(emitted_cpp.read_text(encoding="utf-8"))

    if args.artifact_dir:
        print(f"artifacts: {Path(args.artifact_dir).resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
