#!/usr/bin/env python3
"""Shared helpers for repo-local ScaleHLS frontend scripts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RewriteRule:
    source: str
    target: str
    reason: str


REWRITE_RULES: tuple[RewriteRule, ...] = (
    RewriteRule(
        source="arith.maximumf",
        target="arith.maxf",
        reason="LLVM 16 uses arith.maxf for floating-point maximum",
    ),
    RewriteRule(
        source="arith.minimumf",
        target="arith.minf",
        reason="LLVM 16 uses arith.minf for floating-point minimum",
    ),
)


class CommandFailure(RuntimeError):
    """Structured command failure with artifact paths for debugging."""

    def __init__(
        self,
        cmd: Sequence[str],
        returncode: int,
        stdout_path: Path | None,
        stderr_path: Path | None,
    ) -> None:
        self.cmd = [str(part) for part in cmd]
        self.returncode = returncode
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        joined = format_command(cmd)
        details = [f"command failed ({returncode}): {joined}"]
        if stdout_path is not None:
            details.append(f"stdout: {stdout_path}")
        if stderr_path is not None:
            details.append(f"stderr: {stderr_path}")
        super().__init__("\n".join(details))


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def find_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()

    markers = (
        ("tools", "CMakeLists.txt"),
        ("CMakeLists.txt",),
    )
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if all((candidate / Path(*marker)).exists() for marker in markers):
            return candidate
    raise RuntimeError("could not resolve repository root; pass --repo-root")


def resolve_tool(tool_name: str, repo_root: Path) -> Path:
    built = repo_root / "build" / "bin" / tool_name
    if built.exists():
        return built
    if tool_name == "cgeist":
        polygeist = repo_root / "polygeist" / "build" / "bin" / tool_name
        if polygeist.exists():
            return polygeist
    located = shutil.which(tool_name)
    if located:
        return Path(located)
    raise RuntimeError(f"missing required tool: {tool_name}")


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    print_commands: bool = False,
) -> subprocess.CompletedProcess[str]:
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    if print_commands:
        print("+", format_command(cmd), file=sys.stderr)

    stdout_handle = (
        stdout_path.open("w", encoding="utf-8")
        if stdout_path is not None
        else subprocess.PIPE
    )
    stderr_handle = (
        stderr_path.open("w", encoding="utf-8")
        if stderr_path is not None
        else subprocess.PIPE
    )
    try:
        completed = subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    finally:
        if stdout_path is not None:
            stdout_handle.close()
        if stderr_path is not None:
            stderr_handle.close()

    if completed.returncode != 0:
        raise CommandFailure(cmd, completed.returncode, stdout_path, stderr_path)
    return completed


def summarize_log(log_path: Path | None) -> str | None:
    if log_path is None or not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("PLEASE submit a bug report"):
            continue
        if stripped.startswith("Stack dump:"):
            continue
        return stripped
    return None


def normalize_mlir_text(text: str) -> tuple[str, list[RewriteRule]]:
    normalized = text
    applied: list[RewriteRule] = []
    for rule in REWRITE_RULES:
        if rule.source in normalized:
            normalized = normalized.replace(rule.source, rule.target)
            applied.append(rule)
    return normalized, applied


def normalize_file(input_path: Path, output_path: Path) -> list[RewriteRule]:
    normalized, applied = normalize_mlir_text(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized, encoding="utf-8")
    return applied


OUTLINED_SYMBOL_RE = re.compile(r"scalehls\.gemm_outlined = @([A-Za-z_0-9$.]+)")
SKIP_REASON_RE = re.compile(r'scalehls\.gemm_skip_reason = "([^"]+)"')
ACCEL_OP_RE = re.compile(r'"accel\.(gemmv|gemm|conv)"')


@dataclass(frozen=True)
class CArtifactLayout:
    output_root: Path
    stem: str
    frontend_mlir: Path
    optimized_mlir: Path
    mapped_mlir: Path
    emitted_cpp: Path
    candidate_log: Path
    manifest_dir: Path
    cgeist_stderr: Path
    opt_stderr: Path
    mapper_stderr: Path
    emit_stderr: Path
    result_json: Path


@dataclass
class CFrontendRunResult:
    input_path: Path
    top_func: str
    pipeline_mode: str
    artifact_root: Path
    frontend_succeeded: bool = False
    mapper_succeeded: bool = False
    emit_succeeded: bool = False
    mapped: bool = False
    mapped_symbols: list[str] | None = None
    skip_reason: str | None = None
    candidate_entries: list[dict] | None = None
    manifest_paths: list[str] | None = None
    failure_stage: str | None = None
    failure_summary: str | None = None
    frontend_mlir: str | None = None
    mapped_mlir: str | None = None
    optimized_mlir: str | None = None
    emitted_cpp: str | None = None
    candidate_log: str | None = None
    manifest_dir: str | None = None

    def to_dict(self) -> dict:
        return {
            "input_path": str(self.input_path),
            "top_func": self.top_func,
            "pipeline_mode": self.pipeline_mode,
            "artifact_root": str(self.artifact_root),
            "frontend_succeeded": self.frontend_succeeded,
            "mapper_succeeded": self.mapper_succeeded,
            "emit_succeeded": self.emit_succeeded,
            "mapped": self.mapped,
            "mapped_symbols": self.mapped_symbols or [],
            "skip_reason": self.skip_reason,
            "candidate_entries": self.candidate_entries or [],
            "manifest_paths": self.manifest_paths or [],
            "failure_stage": self.failure_stage,
            "failure_summary": self.failure_summary,
            "frontend_mlir": self.frontend_mlir,
            "mapped_mlir": self.mapped_mlir,
            "optimized_mlir": self.optimized_mlir,
            "emitted_cpp": self.emitted_cpp,
            "candidate_log": self.candidate_log,
            "manifest_dir": self.manifest_dir,
        }


def build_artifact_layout(output_root: Path, stem: str) -> CArtifactLayout:
    return CArtifactLayout(
        output_root=output_root,
        stem=stem,
        frontend_mlir=output_root / "mlir" / f"{stem}.mlir",
        optimized_mlir=output_root / "mlir_opt" / f"{stem}.mlir",
        mapped_mlir=output_root / "mlir_mapped" / f"{stem}.mlir",
        emitted_cpp=output_root / "cpp" / f"{stem}.cpp",
        candidate_log=output_root / "gemm_reports" / "candidate_logs" / f"{stem}.json",
        manifest_dir=output_root / "gemm_reports" / "manifests" / stem,
        cgeist_stderr=output_root / "logs" / f"{stem}_cgeist.stderr",
        opt_stderr=output_root / "logs" / f"{stem}_opt.stderr",
        mapper_stderr=output_root / "logs" / f"{stem}_gemm_mapper.stderr",
        emit_stderr=output_root / "logs" / f"{stem}_emit.stderr",
        result_json=output_root / "result.json",
    )


def parse_mapped_symbols(mlir_text: str) -> list[str]:
    return sorted(set(OUTLINED_SYMBOL_RE.findall(mlir_text)))


def parse_skip_reason(mlir_text: str) -> str | None:
    match = SKIP_REASON_RE.search(mlir_text)
    return match.group(1) if match else None


def has_accel_ops(mlir_text: str) -> bool:
    return ACCEL_OP_RE.search(mlir_text) is not None


def load_candidate_entries(candidate_log: Path) -> list[dict]:
    if not candidate_log.exists():
        return []
    data = json.loads(candidate_log.read_text(encoding="utf-8"))
    return data.get("candidates", [])


def first_candidate_for(entries: list[dict], top_func: str) -> dict | None:
    for entry in entries:
        if entry.get("function") == top_func:
            return entry
    return entries[0] if entries else None


def _record_failure(
    result: CFrontendRunResult,
    *,
    stage: str,
    fallback_message: str,
    stderr_path: Path | None,
) -> CFrontendRunResult:
    result.failure_stage = stage
    result.failure_summary = summarize_log(stderr_path) or fallback_message
    return result


def run_c_frontend(
    input_path: Path,
    *,
    top_func: str | None = None,
    artifact_dir: Path | None = None,
    repo_root: Path | None = None,
    pipeline_mode: str = "default",
    axi_interface: bool = False,
    print_commands: bool = False,
) -> CFrontendRunResult:
    resolved_repo_root = find_repo_root(repo_root)
    resolved_input = input_path.resolve()
    selected_top_func = top_func or resolved_input.stem
    output_root = (
        artifact_dir.resolve()
        if artifact_dir is not None
        else resolved_repo_root / "tmp" / resolved_input.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)
    layout = build_artifact_layout(output_root, resolved_input.stem)
    result = CFrontendRunResult(
        input_path=resolved_input,
        top_func=selected_top_func,
        pipeline_mode=pipeline_mode,
        artifact_root=output_root,
        frontend_mlir=str(layout.frontend_mlir),
        mapped_mlir=str(layout.mapped_mlir),
        optimized_mlir=str(layout.optimized_mlir),
        emitted_cpp=str(layout.emitted_cpp),
        candidate_log=str(layout.candidate_log),
        manifest_dir=str(layout.manifest_dir),
        mapped_symbols=[],
        candidate_entries=[],
        manifest_paths=[],
    )

    cgeist = resolve_tool("cgeist", resolved_repo_root)
    translate = resolve_tool("scalehls-translate", resolved_repo_root)

    try:
        run_command(
            [
                str(cgeist),
                str(resolved_input),
                f"-function={selected_top_func}",
                "-S",
                "-memref-fullrank",
                "-raise-scf-to-affine",
            ],
            cwd=resolved_repo_root,
            stdout_path=layout.frontend_mlir,
            stderr_path=layout.cgeist_stderr,
            print_commands=print_commands,
        )
        result.frontend_succeeded = True
    except CommandFailure as exc:
        return _record_failure(
            result,
            stage="frontend",
            fallback_message=str(exc),
            stderr_path=layout.cgeist_stderr,
        )

    if pipeline_mode == "default":
        scalehls_opt = resolve_tool("scalehls-opt", resolved_repo_root)
        try:
            run_command(
                [
                    str(scalehls_opt),
                    str(layout.frontend_mlir),
                    (
                        "-scaleflow-cpp-pipeline="
                        f"top-func={selected_top_func} "
                        f"axi-interface={'true' if axi_interface else 'false'}"
                    ),
                ],
                cwd=resolved_repo_root,
                stdout_path=layout.optimized_mlir,
                stderr_path=layout.opt_stderr,
                print_commands=print_commands,
            )
        except CommandFailure as exc:
            return _record_failure(
                result,
                stage="opt",
                fallback_message=str(exc),
                stderr_path=layout.opt_stderr,
            )

        try:
            run_command(
                [str(translate), "-scalehls-emit-hlscpp", str(layout.optimized_mlir)],
                cwd=resolved_repo_root,
                stdout_path=layout.emitted_cpp,
                stderr_path=layout.emit_stderr,
                print_commands=print_commands,
            )
            result.emit_succeeded = True
        except CommandFailure as exc:
            return _record_failure(
                result,
                stage="emit",
                fallback_message=str(exc),
                stderr_path=layout.emit_stderr,
            )
        return result

    if pipeline_mode == "accel":
        mapper = resolve_tool("scalehls-opt", resolved_repo_root)
        try:
            run_command(
                [
                    str(mapper),
                    (
                        "-c-accel-pipeline="
                        f"top-func={selected_top_func} "
                        f"manifest-dir={layout.manifest_dir} "
                        f"candidate-log={layout.candidate_log}"
                    ),
                    str(layout.frontend_mlir),
                ],
                cwd=resolved_repo_root,
                stdout_path=layout.mapped_mlir,
                stderr_path=layout.mapper_stderr,
                print_commands=print_commands,
            )
            result.mapper_succeeded = True
        except CommandFailure as exc:
            return _record_failure(
                result,
                stage="mapper",
                fallback_message=str(exc),
                stderr_path=layout.mapper_stderr,
            )

        mapped_text = layout.mapped_mlir.read_text(encoding="utf-8")
        result.mapped_symbols = parse_mapped_symbols(mapped_text)
        result.mapped = bool(result.mapped_symbols)
        result.candidate_entries = load_candidate_entries(layout.candidate_log)
        candidate = first_candidate_for(result.candidate_entries, selected_top_func)
        if candidate is not None and candidate.get("skip_reason"):
            result.skip_reason = candidate["skip_reason"]
        result.manifest_paths = sorted(
            str(path) for path in layout.manifest_dir.glob("*.json")
        )

        try:
            run_command(
                [str(translate), "-scalehls-emit-hlscpp", str(layout.mapped_mlir)],
                cwd=resolved_repo_root,
                stdout_path=layout.emitted_cpp,
                stderr_path=layout.emit_stderr,
                print_commands=print_commands,
            )
            result.emit_succeeded = True
        except CommandFailure as exc:
            return _record_failure(
                result,
                stage="emit",
                fallback_message=str(exc),
                stderr_path=layout.emit_stderr,
            )

        return result

    raise ValueError(f"unsupported pipeline mode: {pipeline_mode}")


def write_result_json(result: CFrontendRunResult) -> None:
    output_path = result.artifact_root / "result.json"
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
