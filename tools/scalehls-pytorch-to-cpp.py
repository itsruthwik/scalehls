#!/usr/bin/env python3
"""Run a PyTorch frontend script through the ScaleHLS pipeline and emit HLS C++."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import (
    find_repo_root,
    normalize_file,
    resolve_tool,
    run_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PyTorch script that prints Torch-MLIR")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for emitted HLS C++; stdout when omitted",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Keep raw MLIR, normalized MLIR, logs, and optional mapper reports here",
    )
    parser.add_argument("--repo-root", type=Path, help="Repository root override")
    parser.add_argument(
        "--venv-python",
        type=Path,
        help="Python interpreter with torch/torch-mlir installed",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("default", "accel"),
        default="default",
        help="ScaleHLS pipeline variant to run",
    )
    parser.add_argument("--top-func", default="forward")
    parser.add_argument("--loop-tile-size", type=int, default=8)
    parser.add_argument("--loop-unroll-factor", type=int, default=4)
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
        "--axi-interface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass axi-interface=<bool> into the ScaleHLS pipeline",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print subprocess commands to stderr",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = find_repo_root(args.repo_root)

    input_script = args.input.resolve()
    if not input_script.exists():
        print(f"missing input: {input_script}", file=sys.stderr)
        return 2

    venv_python = args.venv_python
    if venv_python is None:
        default_venv = repo_root / ".venv" / "pytorch_frontend_20260408_py311" / "bin" / "python"
        venv_python = default_venv if default_venv.exists() else Path(sys.executable)
    if not venv_python.exists():
        print(f"missing venv python: {venv_python}", file=sys.stderr)
        return 2

    scalehls_opt = resolve_tool("scalehls-opt", repo_root)
    scalehls_translate = resolve_tool("scalehls-translate", repo_root)

    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir
        else repo_root / "tmp" / input_script.stem
    )
    raw_mlir_path = artifact_dir / "mlir_raw" / f"{input_script.stem}.mlir"
    mlir_path = artifact_dir / "mlir" / f"{input_script.stem}.mlir"
    cpp_path = artifact_dir / "cpp" / f"{input_script.stem}.cpp"
    torch_stderr = artifact_dir / "logs" / f"{input_script.stem}_torch_mlir.stderr"
    normalize_stderr = artifact_dir / "logs" / f"{input_script.stem}_normalize.stderr"
    emit_stderr = artifact_dir / "logs" / f"{input_script.stem}_emit.stderr"
    candidate_report = artifact_dir / "gemm_reports" / "report.txt"
    manifest_file = artifact_dir / "gemm_reports" / "manifest.json"
    shutil.rmtree(artifact_dir / "gemm_reports", ignore_errors=True)

    run_command(
        [str(venv_python), "-u", str(input_script)],
        cwd=repo_root,
        stdout_path=raw_mlir_path,
        stderr_path=torch_stderr,
        print_commands=args.print_commands,
    )

    normalize_stderr.parent.mkdir(parents=True, exist_ok=True)
    applied = normalize_file(raw_mlir_path, mlir_path)
    with normalize_stderr.open("w", encoding="utf-8") as handle:
        if not applied:
            handle.write("no rewrites applied\n")
        else:
            for rule in applied:
                handle.write(f"{rule.source} -> {rule.target}\n")

    pipeline = (
        f"top-func={args.top_func} "
        f"loop-tile-size={args.loop_tile_size} "
        f"loop-unroll-factor={args.loop_unroll_factor} "
        f"axi-interface={'true' if args.axi_interface else 'false'}"
    )
    pipeline_name = "scaleflow-pytorch-pipeline"
    if args.pipeline_mode == "accel":
        pipeline_name = "pytorch-accel-pipeline"
        pipeline += (
            f" manifest-file={manifest_file} "
            f"candidate-report={candidate_report}"
        )
        if args.ip_size:
            pipeline += f" ip-size={args.ip_size}"
        if args.serial:
            pipeline += " serial=true"

    shell_cmd = (
        f"set -o pipefail; "
        f"{scalehls_opt} {mlir_path} "
        f"-{pipeline_name}='{pipeline}' "
        f"| {scalehls_translate} -scalehls-emit-hlscpp"
    )
    run_command(
        ["bash", "-lc", shell_cmd],
        cwd=repo_root,
        stdout_path=cpp_path,
        stderr_path=emit_stderr,
        print_commands=args.print_commands,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(cpp_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        sys.stdout.write(cpp_path.read_text(encoding="utf-8"))

    if args.artifact_dir:
        print(f"artifacts: {artifact_dir}", file=sys.stderr)
        if args.pipeline_mode == "accel":
            print(f"candidate_report: {candidate_report}", file=sys.stderr)
            print(f"manifest_file: {manifest_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
